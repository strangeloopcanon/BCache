from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from bodocache.adapters import bwrt_adapter


def test_ptr_from_obj_numpy():
    arr = np.arange(4, dtype=np.float32)
    ptr = bwrt_adapter._ptr_from_obj(arr)
    assert isinstance(ptr, int)
    assert ptr == arr.__array_interface__["data"][0]


def test_bwrt_adapter_ctypes_path(monkeypatch):
    adapter_module = importlib.reload(bwrt_adapter)

    class _StubRuntime:
        def __init__(self, device_index: int = 0) -> None:
            self.device_index = device_index
            self.submissions = []
            self.waits = []
            self.last_weight = None

        def submit_wave(self, spec, a_ptr, b_ptr, c_ptr):
            self.submissions.append((spec, a_ptr, b_ptr, c_ptr))
            return SimpleNamespace(handle=1)

        def wait(self, evt, timeout: int | None = None):
            self.waits.append((evt, timeout))

        def sample(self):
            return {"ok": True}

        def set_weights(self, ptr):
            self.last_weight = ptr

    monkeypatch.setattr(adapter_module, "_HAVE_PYBIND", False, raising=False)
    monkeypatch.setattr(adapter_module, "_HAVE_CTYPES", True, raising=False)
    monkeypatch.setattr(adapter_module, "_BwRt", _StubRuntime, raising=False)
    monkeypatch.setattr(adapter_module, "_ws_from_dict", lambda spec: spec, raising=False)
    monkeypatch.setattr(adapter_module, "_ws_from_proto", lambda spec: spec, raising=False)

    adapter = adapter_module.BwRuntimeAdapter(device_index=3)
    spec = {
        "pack_order": [1, 2],
        "tile_order": [(0, 0), (0, 1)],
        "bm": 128,
        "bn": 128,
        "bk": 64,
        "cluster_shape": (2, 1),
        "tmem_layout": {"columns": 8, "phases": 4, "double_buffer": True, "stage_n": 2},
        "io_extents": [("0", 0, 1)],
        "swap_window": (0, 4),
    }
    metrics = adapter.submit_and_wait(spec, 1, 2, 3)
    assert metrics == {"ok": True}
    runtime = adapter._runtime  # type: ignore[attr-defined]
    assert runtime.device_index == 3
    assert runtime.submissions[0][0]["bm"] == 128
    adapter.set_weights(99)
    assert runtime.last_weight == 99


def test_bwrt_adapter_raises_when_runtime_missing(monkeypatch):
    adapter_module = importlib.reload(bwrt_adapter)
    monkeypatch.setattr(adapter_module, "_HAVE_PYBIND", False, raising=False)
    monkeypatch.setattr(adapter_module, "_HAVE_CTYPES", False, raising=False)
    with pytest.raises(RuntimeError):
        adapter_module.BwRuntimeAdapter()
