from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .base import KVRequest
from .vllm_adapter import VLLMBCacheAdapter


BuildRequestsFn = Callable[[Any], Sequence[KVRequest]]
DestResolverFn = Callable[[Dict[str, Any]], Any]
ReadyCallback = Callable[[Dict[str, Any]], None]


class VLLMHook:
    """Drive BCache prefetch from vLLM by calling `prefetch()` per decode window.

    This glue does not import vLLM. You pass callables that know how to
    translate vLLM's internal state into KVRequest sequences and how to map
    a (layer, page-range) into a device destination pointer.
    """

    def __init__(
        self,
        adapter: VLLMBCacheAdapter,
        *,
        build_requests: BuildRequestsFn,
        dest_resolver: Optional[DestResolverFn] = None,
        on_ready: Optional[ReadyCallback] = None,
        bandwidth_caps: Optional[dict[int, int]] = None,
        free_bytes: Optional[dict[int, int]] = None,
        layer_lat_ms: Optional[dict[int, float]] = None,
    ) -> None:
        self.adapter = adapter
        self.build_requests = build_requests
        self.dest_resolver = dest_resolver
        self.on_ready = on_ready
        self.bandwidth_caps = bandwidth_caps
        self.free_bytes = free_bytes
        self.layer_lat_ms = layer_lat_ms

    def prefetch_step(self, vllm_state: Any, now_ms: int) -> None:
        reqs = self.build_requests(vllm_state)
        self.adapter.prefetch(
            reqs,
            now_ms=now_ms,
            bandwidth_caps=self.bandwidth_caps,
            free_bytes=self.free_bytes,
            layer_lat_ms=self.layer_lat_ms,
            on_ready=self.on_ready,
            dest_resolver=self.dest_resolver,
        )

