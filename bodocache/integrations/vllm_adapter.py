from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bodocache.agent.node_agent import NodeAgent, NodeExecStats
from bodocache.planner.api import PlannerConfig, plan_window
from bodocache.telemetry.trace import PrefetchEvent, TraceRecorder

from .base import KVRequest, PlannerInputs

ReadyCallback = Callable[[dict[str, Any]], None]
HintProvider = Callable[[int, Sequence[KVRequest]], Sequence[KVRequest]]


@dataclass
class PrefetchResult:
    plan_df: pd.DataFrame
    evict_df: pd.DataFrame
    admission_df: pd.DataFrame
    exec_stats: NodeExecStats
    metrics: dict[str, Any] | None = None


@dataclass
class ContextParallelSpec:
    """Optional context-parallel sharding descriptor.

    When provided, the adapter shards requests by page id modulo `world_size`.
    This matches a common context-parallel scheme where each rank owns a
    subset of positions. We conservatively split non-contiguous pages into
    single-page requests to preserve correctness under modulo sharding.

    Note: Engines that already pre-shard requests per rank should leave this
    unset; this is a convenience for simple integrations and tests.
    """

    world_size: int
    rank: int


class VLLMBCacheAdapter:
    """Thin adapter to drive BCache planning/execution from vLLM.

    Usage pattern inside vLLM (conceptually):
      - Each decode step prepares KV page interval demands across active sequences.
      - Call `prefetch()` with those demands and timing info.
      - Optionally provide a `dest_resolver` that maps a (layer,page range) to a
        destination GPU buffer/pointer in vLLM's KV cache.
      - Receive metrics; the adapter issues on_ready callbacks per completed copy.
    """

    def __init__(
        self,
        agent: NodeAgent,
        *,
        node: str,
        model_id: str,
        model_version: str,
        pmin: float = 1.0,
        umin: float = 0.0,
        min_io_bytes: int = 512 * 1024,
        alpha: float = 1.0,
        beta: float = 0.0,
        window_ms: int = 20,
        max_ops_per_tier: int = 64,
        enforce_tier_caps: bool = True,
        on_evict: Callable[[pd.DataFrame], None] | None = None,
        on_admit: Callable[[pd.DataFrame], None] | None = None,
        capture_metrics: bool = True,
        trace: TraceRecorder | None = None,
        hint_provider: HintProvider | None = None,
    ) -> None:
        self.agent = agent
        self.node = node
        self.model_id = model_id
        self.model_version = model_version
        # Planner knobs
        self.pmin = pmin
        self.umin = umin
        self.min_io = min_io_bytes
        self.alpha = alpha
        self.beta = beta
        self.window_ms = window_ms
        self.max_ops = max_ops_per_tier
        self.enforce_caps = enforce_tier_caps
        self.on_evict = on_evict
        self.on_admit = on_admit
        self.capture_metrics = capture_metrics
        self.trace = trace
        self.hint_provider = hint_provider

    @staticmethod
    def _request_key(req: KVRequest) -> tuple[str, int, int, int, str, int, int]:
        return (
            req.prefix_id,
            int(req.layer),
            int(req.page_start),
            int(req.page_end),
            req.tenant,
            int(req.tier_src),
            int(req.tier_dst),
        )

    def prefetch(
        self,
        requests: Sequence[KVRequest],
        *,
        now_ms: int,
        ctx_shard: ContextParallelSpec | None = None,
        bandwidth_caps: dict[int, int] | None = None,
        free_bytes: dict[int, int] | None = None,
        layer_lat_ms: dict[int, float] | None = None,
        tenant_caps: list[tuple[str, int, int]] | None = None,
        on_ready: ReadyCallback | None = None,
        dest_resolver: Callable[[dict[str, Any]], Any] | None = None,
    ) -> PrefetchResult:
        live_requests = list(requests)
        hint_requests: Sequence[KVRequest] = ()
        if self.hint_provider is not None:
            with contextlib.suppress(Exception):
                hint_requests = tuple(self.hint_provider(now_ms, live_requests))

        combined: dict[tuple[str, int, int, int, str, int, int], KVRequest] = {}
        source_by_key: dict[tuple[str, int, int, int, str, int, int], str] = {}
        for req in live_requests:
            key = self._request_key(req)
            combined[key] = req
            source_by_key[key] = "live"
        for hint in hint_requests:
            key = self._request_key(hint)
            if key in combined:
                continue
            combined[key] = hint
            source_by_key[key] = "hint"

        merged_requests: list[KVRequest] = list(combined.values())

        # Optional context-parallel sharding by page-id modulo
        if ctx_shard is not None and ctx_shard.world_size > 1:
            ws = int(ctx_shard.world_size)
            rk = int(ctx_shard.rank) % ws

            sharded: list[KVRequest] = []
            sharded_sources: dict[tuple[str, int, int, int, str, int, int], str] = {}
            for r in merged_requests:
                # Split into single-page requests owned by this rank.
                start = int(r.page_start)
                end = int(r.page_end)
                if end < start:
                    continue
                origin_source = source_by_key.get(self._request_key(r), "live")
                for pid in range(start, end + 1):
                    if (pid % ws) != rk:
                        continue
                    new_req = KVRequest(
                        req_id=f"{r.req_id}-sh{pid}",
                        node=r.node,
                        model_id=r.model_id,
                        model_version=r.model_version,
                        prefix_id=r.prefix_id,
                        layer=int(r.layer),
                        page_start=pid,
                        page_end=pid,
                        page_bytes=int(r.page_bytes),
                        tenant=r.tenant,
                        est_fill_ms=float(r.est_fill_ms),
                        tier_src=int(r.tier_src),
                        tier_dst=int(r.tier_dst),
                        deadline_ms=int(r.deadline_ms),
                    )
                    sharded.append(new_req)
                    sharded_sources[self._request_key(new_req)] = origin_source
            merged_requests = sharded
            source_by_key = sharded_sources

        # Prepare planner inputs
        pi = PlannerInputs(
            requests=list(merged_requests),
            window_ms=self.window_ms,
            now_ms=now_ms,
            bandwidth_caps=bandwidth_caps,
            free_bytes=free_bytes,
            tenant_caps=tenant_caps,
            layer_lat_ms=layer_lat_ms,
        )
        normalized_requests = [
            req.model_copy(
                update={
                    "node": self.node,
                    "model_id": self.model_id,
                    "model_version": self.model_version,
                }
            )
            for req in pi.requests
        ]
        merged_requests = normalized_requests
        pi = pi.model_copy(update={"requests": normalized_requests})
        window = pi.to_planner_window()
        plan_result = plan_window(
            window,
            PlannerConfig(
                pmin=self.pmin,
                umin=self.umin,
                min_io_bytes=self.min_io,
                alpha=self.alpha,
                beta=self.beta,
                window_ms=self.window_ms,
                max_ops_per_tier=self.max_ops,
                enable_admission=True,
                enable_eviction=True,
                enforce_tier_caps=self.enforce_caps,
            ),
        )
        plan_df, evict_df, admission_df = plan_result.as_dataframes()

        if not plan_df.empty:
            route_hints: list[str | None] = []
            for row in plan_df.itertuples(index=False):
                row_layer = int(getattr(row, "layer", -1))
                row_src = int(getattr(row, "tier_src", -1))
                row_dst = int(getattr(row, "tier_dst", -1))
                row_start = int(getattr(row, "start_pid", -1))
                row_end = int(getattr(row, "end_pid", -1))
                candidates: list[KVRequest] = [
                    req
                    for req in merged_requests
                    if int(req.layer) == row_layer
                    and int(req.tier_src) == row_src
                    and int(req.tier_dst) == row_dst
                    and int(req.page_end) >= row_start
                    and int(req.page_start) <= row_end
                ]
                chosen: KVRequest | None = None
                for req in candidates:
                    if source_by_key.get(self._request_key(req), "live") == "live":
                        chosen = req
                        break
                if chosen is None and candidates:
                    chosen = candidates[0]
                route_hints.append(f"prefix:{chosen.prefix_id}" if chosen is not None else None)
            plan_df = plan_df.assign(route_hint=route_hints)

        # Optional eviction/admission side-effects
        if self.on_evict is not None and not evict_df.empty:
            with contextlib.suppress(Exception):
                self.on_evict(evict_df)
        if self.on_admit is not None and not admission_df.empty:
            with contextlib.suppress(Exception):
                self.on_admit(admission_df)

        # Execute via NodeAgent (simulated or native engine)
        deadlines: dict[tuple[int, int, int], float] = {}
        if self.capture_metrics and not plan_df.empty:
            base = float(now_ms)
            for r in plan_df.itertuples(index=False):
                deadline_key = (
                    int(getattr(r, "layer", -1)),
                    int(getattr(r, "start_pid", -1)),
                    int(getattr(r, "end_pid", -1)),
                )
                deadlines[deadline_key] = float(getattr(r, "deadline_ms", base)) - base

        ready_count = 0
        on_time_count = 0

        def _wrap_on_ready(info: dict[str, Any]):
            nonlocal ready_count, on_time_count
            ready_count += 1
            if self.capture_metrics and deadlines:
                info_key = (
                    int(info.get("layer", -1)),
                    int(info.get("start_pid", -1)),
                    int(info.get("end_pid", -1)),
                )
                finish = (time.time() * 1000.0) - float(now_ms)
                deadline = deadlines.get(info_key, float("inf"))
                if finish <= deadline:
                    on_time_count += 1
                if self.trace is not None:
                    with contextlib.suppress(Exception):
                        ev = PrefetchEvent(
                            window_ms=self.window_ms,
                            now_ms=int(now_ms),
                            node=self.node,
                            model_id=self.model_id,
                            model_version=self.model_version,
                            layer=int(info.get("layer", -1)),
                            start_pid=int(info.get("start_pid", -1)),
                            end_pid=int(info.get("end_pid", -1)),
                            bytes=int(info.get("bytes", 0)),
                            deadline_rel_ms=float(deadline),
                            finish_rel_ms=float(finish),
                            on_time=int(1 if finish <= deadline else 0),
                        )
                        self.trace.record(ev)
            if on_ready is not None:
                on_ready(info)

        stats = self.agent.execute(
            plan_df,
            model_id=self.model_id,
            model_version=self.model_version,
            on_ready=_wrap_on_ready if (self.capture_metrics or on_ready is not None) else None,
            dest_resolver=dest_resolver,
        )

        metrics = None
        if self.capture_metrics and ready_count > 0:
            metrics = {
                "ready_count": int(ready_count),
                "on_time_ratio": float(on_time_count / max(1, ready_count)),
            }

        return PrefetchResult(
            plan_df=plan_df,
            evict_df=evict_df,
            admission_df=admission_df,
            exec_stats=stats,
            metrics=metrics,
        )
