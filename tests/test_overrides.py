from __future__ import annotations

from bodocache.integrations.vllm_blocks import VLLMCacheConfig
from bodocache.integrations.config import KVOverrides, apply_kv_overrides


def test_kv_overrides_apply():
    cfg = VLLMCacheConfig(block_size=16, num_layers=2, num_kv_heads=8, head_size=64, kv_dtype="float16")
    ov = KVOverrides(block_size=32, num_layers=24, kv_dtype="bfloat16")
    out = apply_kv_overrides(cfg, ov)
    assert out.block_size == 32
    assert out.num_layers == 24
    assert out.num_kv_heads == 8
    assert out.head_size == 64
    assert out.kv_dtype == "bfloat16"

