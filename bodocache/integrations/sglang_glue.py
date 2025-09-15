from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from .base import KVRequest
from .sglang_adapter import SGLangBCacheAdapter


BuildRequestsFn = Callable[[Any], Sequence[KVRequest]]
DestResolverFn = Callable[[Dict[str, Any]], Any]
ReadyCallback = Callable[[Dict[str, Any]], None]


class SGLangHook:
    """Drive BCache prefetch from SGLang by calling `prefetch()` per decode window.

    This glue relies on callables that understand SGLang's scheduler/kv state and
    produce KVRequest sequences and destination pointers.
    """

    def __init__(
        self,
        adapter: SGLangBCacheAdapter,
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

    def prefetch_step(self, sglang_state: Any, now_ms: int) -> None:
        reqs = self.build_requests(sglang_state)
        self.adapter.prefetch(
            reqs,
            now_ms=now_ms,
            bandwidth_caps=self.bandwidth_caps,
            free_bytes=self.free_bytes,
            layer_lat_ms=self.layer_lat_ms,
            on_ready=self.on_ready,
            dest_resolver=self.dest_resolver,
        )

