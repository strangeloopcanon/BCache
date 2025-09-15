from __future__ import annotations

from bodocache.integrations.vllm_collectors import make_vllm_collector, make_vllm_dest_resolver


class DummyBM:
    def get_required_blocks(self, state):
        return {0: [0, 1, 2], 1: [3]}


class DummyKV:
    class Tensor:
        def __init__(self, addr: int):
            self._addr = addr

        def data_ptr(self):
            return self._addr

    def get_tensor_slice(self, layer, start, end):
        return self.Tensor(1234 + layer + start + end)


class DummyEngine:
    def __init__(self):
        self.block_manager = DummyBM()


def test_collectors_vllm():
    engine = DummyEngine()
    collector = make_vllm_collector(engine)
    out = collector(state=None)
    assert 0 in out and 1 in out
    kv = DummyKV()
    resolver = make_vllm_dest_resolver(kv)
    ptr = resolver({"layer": 0, "start_pid": 0, "end_pid": 2})
    assert isinstance(ptr, object)  # capsule-like

