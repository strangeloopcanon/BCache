from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .sglang_adapter import SGLangBCacheAdapter, PrefetchResult
from .vllm_blocks import VLLMCacheConfig, build_requests_from_blocks
from .config import KVOverrides, apply_kv_overrides


CollectBlocksFn = Callable[[Any], Dict[int, Sequence[int]]]
DestResolverFn = Callable[[Dict[str, Any]], Any]
GetConfigFn = Callable[[Any], VLLMCacheConfig]


def _safe_get(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = obj
    for name in path:
        if cur is None:
            return default
        cur = getattr(cur, name, None)
    return default if cur is None else cur


def _maybe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _maybe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _derive_config(engine: Any) -> VLLMCacheConfig:
    # SGLang-like detection with similar fields
    block_size = _maybe_int(_safe_get(engine, ("cache_config", "block_size"), 16), 16)
    kv_dtype = _maybe_str(
        _safe_get(engine, ("cache_config", "cache_dtype"), None)
        or _safe_get(engine, ("cache_config", "kv_dtype"), "float16"),
        "float16",
    )
    mc = getattr(engine, "model_config", None) or _safe_get(engine, ("engine", "model_config"), None)
    num_layers = _maybe_int(getattr(mc, "num_hidden_layers", None), 0)
    num_kv_heads = 0
    if mc is not None:
        if hasattr(mc, "get_num_kv_heads"):
            try:
                num_kv_heads = int(mc.get_num_kv_heads())
            except Exception:
                pass
        if num_kv_heads <= 0:
            num_kv_heads = _maybe_int(getattr(mc, "num_key_value_heads", None), 0)
        if num_kv_heads <= 0:
            num_kv_heads = _maybe_int(getattr(mc, "num_attention_heads", None), 0)
    head_size = _maybe_int(getattr(mc, "head_size", None), 0)
    if head_size <= 0:
        hidden = _maybe_int(getattr(mc, "hidden_size", None), 0)
        attn_heads = _maybe_int(getattr(mc, "num_attention_heads", None), 1) or 1
        if hidden > 0 and attn_heads > 0:
            head_size = int(hidden // attn_heads)
    if num_layers <= 0:
        num_layers = 1
    if num_kv_heads <= 0:
        num_kv_heads = 8
    if head_size <= 0:
        head_size = 64
    return VLLMCacheConfig(
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_dtype=kv_dtype,
    )


class SGLangIntegration:
    def __init__(
        self,
        engine: Any,
        adapter: SGLangBCacheAdapter,
        *,
        tenant: str = "default",
        get_config: Optional[GetConfigFn] = None,
        collect_blocks: Optional[CollectBlocksFn] = None,
        dest_resolver: Optional[DestResolverFn] = None,
        kv_overrides: Optional[KVOverrides | Dict[str, Any]] = None,
        deadline_offset_ms: Optional[int] = None,
    ) -> None:
        self.engine = engine
        self.adapter = adapter
        self.tenant = tenant
        self._get_config = get_config
        self._collect_blocks = collect_blocks
        self._dest_resolver = dest_resolver
        self._kv_overrides = kv_overrides
        self._deadline_offset_ms = deadline_offset_ms

    def _config(self) -> VLLMCacheConfig:
        cfg = self._get_config(self.engine) if self._get_config is not None else _derive_config(self.engine)
        return apply_kv_overrides(cfg, self._kv_overrides)

    def prefetch_step(
        self,
        state: Any,
        *,
        prefix_id: str,
        now_ms: int,
        layer_lat_ms: Optional[Dict[int, float]] = None,
        bandwidth_caps: Optional[Dict[int, int]] = None,
        free_bytes: Optional[Dict[int, int]] = None,
    ) -> Optional[PrefetchResult]:
        cfg = self._config()
        if self._collect_blocks is None:
            return None
        layer_to_blocks = self._collect_blocks(state)
        kv_reqs = build_requests_from_blocks(
            cfg,
            node=getattr(self.engine, "node", "n0"),
            model_id=_maybe_str(getattr(self.engine, "model_id", "m"), "m"),
            model_version=_maybe_str(getattr(self.engine, "model_version", "v"), "v"),
            tenant=self.tenant,
            prefix_id=prefix_id,
            layer_to_blocks=layer_to_blocks,
            now_ms=now_ms,
            deadline_offset_ms=int(self._deadline_offset_ms if self._deadline_offset_ms is not None else self.adapter.window_ms),
        )
        return self.adapter.prefetch(
            kv_reqs,
            now_ms=now_ms,
            bandwidth_caps=bandwidth_caps,
            free_bytes=free_bytes,
            layer_lat_ms=layer_lat_ms,
            dest_resolver=self._dest_resolver,
        )
