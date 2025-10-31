from __future__ import annotations

import contextlib
from typing import Any

from .ptr import from_torch_tensor


def make_sglang_collector(engine: Any):
    bm = getattr(engine, "block_manager", None) or getattr(engine, "cache_engine", None)

    def collector(state: Any) -> dict[int, list[int]]:
        if bm is not None:
            for name in ("next_required_blocks", "get_required_blocks", "collect_required_blocks"):
                fn = getattr(bm, name, None)
                if callable(fn):
                    out = fn(state)
                    if out is not None:
                        return {int(k): list(map(int, v)) for k, v in out.items()}
        for name in ("next_required_blocks", "get_required_blocks", "collect_required_blocks"):
            fn = getattr(engine, name, None)
            if callable(fn):
                out = fn(state)
                if out is not None:
                    return {int(k): list(map(int, v)) for k, v in out.items()}
        m = getattr(state, "layer_to_blocks", None)
        if isinstance(m, dict):
            return {int(k): list(map(int, v)) for k, v in m.items()}
        raise RuntimeError("could not collect required blocks from engine/state")

    return collector


def make_sglang_dest_resolver(kv_manager: Any):
    def _as_ptr(obj: Any) -> Any:
        if hasattr(obj, "data_ptr") and callable(obj.data_ptr):
            return from_torch_tensor(obj)
        with contextlib.suppress(Exception):
            return int(obj)
        return None

    def resolver(info: dict[str, Any]):
        layer = int(info["layer"])
        start = int(info["start_pid"])
        end = int(info["end_pid"])
        for name in (
            "get_tensor_slice",
            "get_block_tensor",
            "get_range_tensor",
            "get_dest_tensor",
        ):
            fn = getattr(kv_manager, name, None)
            if callable(fn):
                t = fn(layer, start, end)
                ptr = _as_ptr(t)
                if ptr is not None:
                    return ptr
        with contextlib.suppress(Exception):
            t = kv_manager[(layer, start, end)]
            ptr = _as_ptr(t)
            if ptr is not None:
                return ptr
        return None

    return resolver
