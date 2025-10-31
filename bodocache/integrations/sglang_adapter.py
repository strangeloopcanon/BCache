from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bodocache.agent.node_agent import NodeAgent
from bodocache.planner.api import PlannerConfig, plan_window
from bodocache.telemetry.trace import PrefetchEvent, TraceRecorder

from .base import KVRequest, PlannerInputs

ReadyCallback = Callable[[dict[str, Any]], None]


@dataclass
class PrefetchResult:
    plan_df: pd.DataFrame
    evict_df: pd.DataFrame
    admission_df: pd.DataFrame
    exec_stats: dict[str, Any]
    metrics: dict[str, Any] | None = None


class SGLangBCacheAdapter:
    """BCache adapter for SGLang.

    Mirrors the vLLM adapter interface for symmetry.
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

    def prefetch(
        self,
        requests: Sequence[KVRequest],
        *,
        now_ms: int,
        bandwidth_caps: dict[int, int] | None = None,
        free_bytes: dict[int, int] | None = None,
        layer_lat_ms: dict[int, float] | None = None,
        tenant_caps: list[tuple[str, int, int]] | None = None,
        on_ready: ReadyCallback | None = None,
        dest_resolver: Callable[[dict[str, Any]], Any] | None = None,
    ) -> PrefetchResult:
        # Prepare planner inputs
        pi = PlannerInputs(
            requests=list(requests),
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

        # Optional eviction/admission
        if self.on_evict is not None and not evict_df.empty:
            with contextlib.suppress(Exception):
                self.on_evict(evict_df)
        if self.on_admit is not None and not admission_df.empty:
            with contextlib.suppress(Exception):
                self.on_admit(admission_df)

        deadlines: dict[tuple, float] = {}
        if self.capture_metrics and not plan_df.empty:
            base = float(now_ms)
            for r in plan_df.itertuples(index=False):
                key = (
                    int(getattr(r, "layer", -1)),
                    int(getattr(r, "start_pid", -1)),
                    int(getattr(r, "end_pid", -1)),
                )
                deadlines[key] = float(getattr(r, "deadline_ms", base)) - base

        ready_count = 0
        on_time_count = 0

        def _wrap_on_ready(info: dict[str, Any]):
            nonlocal ready_count, on_time_count
            ready_count += 1
            if self.capture_metrics and deadlines:
                key = (
                    int(info.get("layer", -1)),
                    int(info.get("start_pid", -1)),
                    int(info.get("end_pid", -1)),
                )
                finish = (time.time() * 1000.0) - float(now_ms)
                deadline = deadlines.get(key, float("inf"))
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
