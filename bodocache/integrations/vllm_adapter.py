from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
import time

import pandas as pd

from bodocache.planner.scheduler import run_window
from bodocache.agent.node_agent import NodeAgent
from .base import KVRequest, PlannerInputs, build_dataframes
from bodocache.telemetry.trace import TraceRecorder, PrefetchEvent


ReadyCallback = Callable[[Dict[str, Any]], None]


@dataclass
class PrefetchResult:
    plan_df: pd.DataFrame
    evict_df: pd.DataFrame
    admission_df: pd.DataFrame
    exec_stats: Dict[str, Any]
    metrics: Dict[str, Any] | None = None


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
        on_evict: Optional[Callable[[pd.DataFrame], None]] = None,
        on_admit: Optional[Callable[[pd.DataFrame], None]] = None,
        capture_metrics: bool = True,
        trace: Optional[TraceRecorder] = None,
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
        bandwidth_caps: Optional[dict[int, int]] = None,
        free_bytes: Optional[dict[int, int]] = None,
        layer_lat_ms: Optional[dict[int, float]] = None,
        tenant_caps: Optional[List[tuple[str, int, int]]] = None,
        on_ready: Optional[ReadyCallback] = None,
        dest_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
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
        req_df, heat_df, tier_caps_df, tenant_caps_df, layer_lat_df = build_dataframes(pi)

        # Inject node/model identifiers for all rows
        if not req_df.empty:
            req_df["node"] = self.node
            req_df["model_id"] = self.model_id
            req_df["model_version"] = self.model_version

        # Plan
        plan_df, evict_df, admission_df = run_window(
            req_df,
            heat_df,
            tier_caps_df,
            tenant_caps_df,
            layer_lat_df,
            now_ms=now_ms,
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
        )

        # Optional eviction/admission side-effects
        if self.on_evict is not None and not evict_df.empty:
            try:
                self.on_evict(evict_df)
            except Exception:
                pass
        if self.on_admit is not None and not admission_df.empty:
            try:
                self.on_admit(admission_df)
            except Exception:
                pass

        # Execute via NodeAgent (simulated or native engine)
        deadlines: Dict[tuple, float] = {}
        if self.capture_metrics and not plan_df.empty:
            base = float(now_ms)
            for r in plan_df.itertuples(index=False):
                key = (int(getattr(r, 'layer', -1)), int(getattr(r, 'start_pid', -1)), int(getattr(r, 'end_pid', -1)))
                deadlines[key] = float(getattr(r, 'deadline_ms', base)) - base

        ready_count = 0
        on_time_count = 0

        def _wrap_on_ready(info: Dict[str, Any]):
            nonlocal ready_count, on_time_count
            ready_count += 1
            if self.capture_metrics and deadlines:
                key = (int(info.get('layer', -1)), int(info.get('start_pid', -1)), int(info.get('end_pid', -1)))
                finish = (time.time() * 1000.0) - float(now_ms)
                deadline = deadlines.get(key, float('inf'))
                if finish <= deadline:
                    on_time_count += 1
                if self.trace is not None:
                    try:
                        ev = PrefetchEvent(
                            window_ms=self.window_ms,
                            now_ms=int(now_ms),
                            node=self.node,
                            model_id=self.model_id,
                            model_version=self.model_version,
                            layer=int(info.get('layer', -1)),
                            start_pid=int(info.get('start_pid', -1)),
                            end_pid=int(info.get('end_pid', -1)),
                            bytes=int(info.get('bytes', 0)),
                            deadline_rel_ms=float(deadline),
                            finish_rel_ms=float(finish),
                            on_time=int(1 if finish <= deadline else 0),
                        )
                        self.trace.record(ev)
                    except Exception:
                        pass
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

        return PrefetchResult(plan_df=plan_df, evict_df=evict_df, admission_df=admission_df, exec_stats=stats, metrics=metrics)
