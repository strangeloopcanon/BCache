from __future__ import annotations

from bodocache.integrations.vllm_blocks import VLLMCacheConfig, build_requests_from_blocks, coalesce_blocks


def test_coalesce_blocks():
    assert coalesce_blocks([]) == []
    assert coalesce_blocks([1, 2, 3]) == [(1, 3)]
    assert coalesce_blocks([1, 3, 2, 5, 6]) == [(1, 3), (5, 6)]


def test_build_requests_from_blocks():
    cfg = VLLMCacheConfig(block_size=16, num_layers=2, num_kv_heads=8, head_size=64, kv_dtype="float16")
    now_ms = 1000
    reqs = build_requests_from_blocks(
        cfg,
        node="n0",
        model_id="m",
        model_version="v",
        tenant="t",
        prefix_id="p",
        layer_to_blocks={0: [0, 1, 2, 4], 1: [3]},
        now_ms=now_ms,
        deadline_offset_ms=20,
    )
    # Expect ranges: layer 0 -> (0,2) and (4,4); layer 1 -> (3,3)
    assert len(reqs) == 3
    pbytes = cfg.bytes_per_block()
    for r in reqs:
        assert r.page_bytes == pbytes
        assert r.deadline_ms == now_ms + 20

