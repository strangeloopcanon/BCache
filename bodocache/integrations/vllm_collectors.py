from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .ptr import from_torch_tensor


def make_vllm_collector(engine: Any):
    """Return a collector(state) -> Dict[layer, List[int]] for vLLM-like engines.

    Tries several common APIs; override if your engine differs.
    """

    bm = getattr(engine, "block_manager", None) or getattr(engine, "cache_engine", None)

    def collector(state: Any) -> Dict[int, List[int]]:
        # Preferred engine APIs
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
        # Fallback: accept a mapping on state directly
        m = getattr(state, "layer_to_blocks", None)
        if isinstance(m, dict):
            return {int(k): list(map(int, v)) for k, v in m.items()}
        raise RuntimeError("could not collect required blocks from engine/state")

    return collector


def make_vllm_dest_resolver(kv_manager: Any):
    """Return dest_resolver(info) -> device pointer for vLLM KV destinations.

    Tries common KV accessors and wraps torch tensors via from_torch_tensor.
    """

    def _as_ptr(obj: Any) -> Any:
        # torch tensor
        if hasattr(obj, "data_ptr") and callable(getattr(obj, "data_ptr")):
            return from_torch_tensor(obj)
        # raw int pointer
        try:
            return int(obj)
        except Exception:
            return None

    def resolver(info: Dict[str, Any]):
        layer = int(info["layer"])
        start = int(info["start_pid"])
        end = int(info["end_pid"])
        # Try a set of common method names
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
        # Try mapping access
        try:
            t = kv_manager[(layer, start, end)]
            ptr = _as_ptr(t)
            if ptr is not None:
                return ptr
        except Exception:
            pass
        return None

    return resolver

