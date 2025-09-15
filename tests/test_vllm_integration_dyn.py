from __future__ import annotations

import time
from typing import Dict, List

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent
from bodocache.integrations.vllm_adapter import VLLMBCacheAdapter
from bodocache.integrations.vllm_integration import VLLMIntegration


class DummyCacheConfig:
    def __init__(self):
        self.block_size = 16
        self.cache_dtype = "float16"


class DummyModelConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.hidden_size = 512
        self.num_attention_heads = 8

    def get_num_kv_heads(self):
        return 8


class DummyEngine:
    def __init__(self):
        self.cache_config = DummyCacheConfig()
        self.model_config = DummyModelConfig()
        self.node = "n0"
        self.model_id = "m"
        self.model_version = "v"


def test_vllm_integration_prefetch(tmp_path):
    # Seed some data
    be = SegmentedFileBackend(str(tmp_path))
    page_bytes = 2 * 8 * 16 * 64 * 2  # 2 * kv_heads * block_size * head_size * dtype_bytes
    for pid in range(4):
        be.write_page("m", "v", 0, pid, page_bytes, b"x" * page_bytes)
    agent = NodeAgent(be, page_bytes=page_bytes)
    adapter = VLLMBCacheAdapter(agent, node="n0", model_id="m", model_version="v", min_io_bytes=0, window_ms=20)

    eng = DummyEngine()

    def collector(state) -> Dict[int, List[int]]:
        return {0: [0, 1, 2, 3]}

    integ = VLLMIntegration(eng, adapter, collect_blocks=collector)
    now_ms = int(time.time() * 1000)
    res = integ.prefetch_step(state=None, prefix_id="sess", now_ms=now_ms)
    assert res is not None
    assert res.exec_stats["bytes"] >= 4 * page_bytes

