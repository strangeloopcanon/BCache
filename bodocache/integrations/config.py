from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from .vllm_blocks import VLLMCacheConfig


@dataclass
class KVOverrides:
    block_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_size: Optional[int] = None
    kv_dtype: Optional[str] = None


def apply_kv_overrides(cfg: VLLMCacheConfig, overrides: KVOverrides | Dict[str, Any] | None) -> VLLMCacheConfig:
    if overrides is None:
        return cfg
    if isinstance(overrides, dict):
        o = KVOverrides(**{k: overrides.get(k) for k in KVOverrides.__annotations__.keys()})
    else:
        o = overrides
    return VLLMCacheConfig(
        block_size=int(o.block_size if o.block_size is not None else cfg.block_size),
        num_layers=int(o.num_layers if o.num_layers is not None else cfg.num_layers),
        num_kv_heads=int(o.num_kv_heads if o.num_kv_heads is not None else cfg.num_kv_heads),
        head_size=int(o.head_size if o.head_size is not None else cfg.head_size),
        kv_dtype=str(o.kv_dtype if o.kv_dtype is not None else cfg.kv_dtype),
    )


def load_kv_overrides(path: str) -> KVOverrides:
    if yaml is None:
        raise ImportError("pyyaml is required to load overrides from YAML")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    kv = data.get("kv", {})
    return KVOverrides(
        block_size=kv.get("block_size"),
        num_layers=kv.get("num_layers"),
        num_kv_heads=kv.get("num_kv_heads"),
        head_size=kv.get("head_size"),
        kv_dtype=kv.get("kv_dtype"),
    )

