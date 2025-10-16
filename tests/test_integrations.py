from __future__ import annotations

import secrets
import time

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent
from bodocache.integrations.base import KVRequest
from bodocache.integrations.vllm_adapter import VLLMBCacheAdapter, ContextParallelSpec
from bodocache.agent.copy_engine import SimCopyEngine


def test_vllm_adapter_prefetch_sim(tmp_path):
    be = SegmentedFileBackend(str(tmp_path))
    # seed pages
    for pid in range(8):
        be.write_page("m", "v", 0, pid, 4096, secrets.token_bytes(4096))
    agent = NodeAgent(be, page_bytes=4096)
    adapter = VLLMBCacheAdapter(agent, node="n0", model_id="m", model_version="v", min_io_bytes=0)
    now_ms = int(time.time() * 1000)
    reqs = [
        KVRequest(
            req_id="r0",
            node="n0",
            model_id="m",
            model_version="v",
            prefix_id="p0",
            layer=0,
            page_start=0,
            page_end=3,
            page_bytes=4096,
            tenant="t",
            est_fill_ms=1.0,
            tier_src=0,
            tier_dst=2,
            deadline_ms=now_ms + 1000,
        )
    ]
    ready = []
    res = adapter.prefetch(reqs, now_ms=now_ms, on_ready=lambda info: ready.append(info))
    assert not res.plan_df.empty
    assert res.exec_stats["bytes"] >= 4 * 4096
    assert ready, "should receive on_ready callbacks in simulation"
    assert res.metrics is None or 0.0 <= res.metrics.get("on_time_ratio", 1.0) <= 1.0


def test_node_agent_with_engine_path(tmp_path):
    be = SegmentedFileBackend(str(tmp_path))
    # seed
    for pid in range(2):
        be.write_page("m", "v", 0, pid, 4096, secrets.token_bytes(4096))
    eng = SimCopyEngine()
    agent = NodeAgent(be, page_bytes=4096, copy_engine=eng)
    adapter = VLLMBCacheAdapter(agent, node="n0", model_id="m", model_version="v", min_io_bytes=0)
    now_ms = int(time.time() * 1000)
    reqs = [KVRequest("r", "n0", "m", "v", "p", 0, 0, 1, 4096, "t", 1.0, 0, 2, now_ms + 1000)]
    ready = []
    # dest_resolver returns a dummy device pointer integer; SimCopyEngine doesn't deref it.
    res = adapter.prefetch(reqs, now_ms=now_ms, dest_resolver=lambda info: 0, on_ready=lambda info: ready.append(info))
    assert res.exec_stats["ops"] >= 1
    assert ready, "on_ready should be called via engine path"


def test_vllm_adapter_context_parallel_shard(tmp_path):
    be = SegmentedFileBackend(str(tmp_path))
    # Seed 4 pages for layer 0
    page_bytes = 4096
    for pid in range(4):
        be.write_page("m", "v", 0, pid, page_bytes, secrets.token_bytes(page_bytes))
    agent = NodeAgent(be, page_bytes=page_bytes)
    adapter = VLLMBCacheAdapter(agent, node="n0", model_id="m", model_version="v", min_io_bytes=0)
    now_ms = int(time.time() * 1000)
    # Global request for pages [0..3]
    req = KVRequest(
        req_id="r",
        node="n0",
        model_id="m",
        model_version="v",
        prefix_id="p",
        layer=0,
        page_start=0,
        page_end=3,
        page_bytes=page_bytes,
        tenant="t",
        est_fill_ms=1.0,
        tier_src=0,
        tier_dst=2,
        deadline_ms=now_ms + 1000,
    )
    # Rank 0 of 2 should own pages {0,2}
    res0 = adapter.prefetch([req], now_ms=now_ms, ctx_shard=ContextParallelSpec(world_size=2, rank=0))
    assert res0.exec_stats["bytes"] >= 2 * page_bytes
    # Rank 1 of 2 should own pages {1,3}
    res1 = adapter.prefetch([req], now_ms=now_ms, ctx_shard=ContextParallelSpec(world_size=2, rank=1))
    assert res1.exec_stats["bytes"] >= 2 * page_bytes
