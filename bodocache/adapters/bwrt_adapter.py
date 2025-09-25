from __future__ import annotations

from typing import Any, Iterable, Sequence

try:  # pragma: no cover - optional fast path
    from bwrt import _bwrt as bw  # type: ignore
    _HAVE_PYBIND = True
except Exception:  # pragma: no cover - optional fast path
    bw = None  # type: ignore
    _HAVE_PYBIND = False

if not _HAVE_PYBIND:
    try:  # pragma: no cover - optional fallback
        from bwrt.runtime import BwRuntime as _BwRt  # type: ignore
        from bwrt.wavespec_adapter import (  # type: ignore
            wavespec_from_dict as _ws_from_dict,
            wavespec_from_proto as _ws_from_proto,
        )
        _HAVE_CTYPES = True
    except Exception:  # pragma: no cover - runtime not installed
        _BwRt = None  # type: ignore
        _ws_from_dict = _ws_from_proto = None  # type: ignore
        _HAVE_CTYPES = False
else:
    _BwRt = None  # type: ignore
    _ws_from_dict = _ws_from_proto = None  # type: ignore
    _HAVE_CTYPES = False


def _ptr_from_obj(obj: Any) -> int:
    if isinstance(obj, int):
        return obj
    ai = getattr(obj, "__array_interface__", None) or getattr(obj, "array_interface", None)
    if ai and "data" in ai:
        return int(ai["data"][0])
    cai = getattr(obj, "__cuda_array_interface__", None) or getattr(obj, "cuda_array_interface", None)
    if cai and "data" in cai:
        data = cai["data"][0]
        if isinstance(data, (tuple, list)):
            return int(data[0])
        return int(data)
    buf = getattr(obj, "data", None)
    if buf is not None and hasattr(buf, "ptr"):
        return int(buf.ptr)
    raise TypeError(
        "Provide int pointer, NumPy/CuPy array, or object with array_interface/cuda_array_interface"
    )


def _normalise_tile_order(to_seq: Any) -> List[Tuple[int, int]] | None:
    if to_seq is None:
        return None
    if isinstance(to_seq, (list, tuple)):
        out: List[Tuple[int, int]] = []
        for item in to_seq:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((int(item[0]), int(item[1])))
            else:
                try:
                    # Treat as flattened index; decode later if runtime provides dims.
                    idx = int(item)
                    out.append((idx, 0))
                except Exception:  # pragma: no cover - defensive
                    continue
        return out
    return None


def _flatten_tile_order(tile_pairs: List[Tuple[int, int]] | None) -> List[int] | None:
    if not tile_pairs:
        return None
    return [int(r) << 16 | int(c) for r, c in tile_pairs]


class BwRuntimeAdapter:
    """Thin wrapper that normalises bw-runtime's pybind / ctypes APIs."""

    def __init__(self, device_index: int = 0) -> None:
        if _HAVE_PYBIND:
            self._fast = True
            self._runtime = bw.Runtime(device_index)  # type: ignore[attr-defined]
        elif _HAVE_CTYPES:
            self._fast = False
            self._runtime = _BwRt(device_index=device_index)  # type: ignore[operator]
        else:
            raise RuntimeError(
                "bwrt runtime is not installed; install bwrt wheel with either pybind or ctypes backend"
            )

    def _to_spec(self, spec_any: Any):
        if _HAVE_PYBIND:
            if isinstance(spec_any, dict):
                bm = int(spec_any.get("bm", 0))
                bn = int(spec_any.get("bn", 0))
                bk = int(spec_any.get("bk", 0))
                cluster = spec_any.get("cluster_shape", (1, 1))
                cx, cy = int(cluster[0]), int(cluster[1])
                swap_window = spec_any.get("swap_window", (0, 1))
                sb, se = int(swap_window[0]), int(swap_window[1])
                tile_pairs = _normalise_tile_order(spec_any.get("tile_order"))
                pack_order = [int(x) for x in spec_any.get("pack_order", [])]
                tmem = spec_any.get("tmem_layout", {})
                io_extents = spec_any.get("io_extents", [])
            else:
                bm = int(spec_any.bm)
                bn = int(spec_any.bn)
                bk = int(spec_any.bk)
                cluster = getattr(spec_any, "cluster_shape", (getattr(spec_any, "cluster_x", 1), getattr(spec_any, "cluster_y", 1)))
                cx, cy = int(cluster[0]), int(cluster[1])
                swap_window = getattr(spec_any, "swap_window", (getattr(spec_any, "swap_begin", 0), getattr(spec_any, "swap_end", 1)))
                sb, se = int(swap_window[0]), int(swap_window[1])
                tile_pairs = _normalise_tile_order(getattr(spec_any, "tile_order", None))
                if not tile_pairs:
                    tile_pairs = _normalise_tile_order(getattr(spec_any, "tile_order_flat", None))
                pack_order = [int(x) for x in getattr(spec_any, "pack_order", [])]
                tmem = getattr(spec_any, "tmem_layout", {})
                io_extents = getattr(spec_any, "io_extents", [])
            spec = bw.WaveSpec()  # type: ignore[attr-defined]
            spec.bm, spec.bn, spec.bk = bm, bn, bk
            if hasattr(spec, "cluster_shape"):
                spec.cluster_shape = (cx, cy)
            if hasattr(spec, "cluster_x"):
                spec.cluster_x, spec.cluster_y = cx, cy
            spec.swap_begin, spec.swap_end = sb, se
            if hasattr(spec, "swap_window"):
                spec.swap_window = (sb, se)
            if hasattr(spec, "pack_order"):
                spec.pack_order = pack_order
            tile_flat = _flatten_tile_order(tile_pairs)
            if hasattr(spec, "tile_order") and tile_pairs is not None:
                spec.tile_order = [(int(r), int(c)) for r, c in tile_pairs]  # type: ignore[attr-defined]
            elif hasattr(spec, "tile_order_flat") and tile_flat is not None:
                spec.tile_order_flat = tile_flat  # type: ignore[attr-defined]
            if hasattr(spec, "tmem_columns") and isinstance(tmem, dict):
                spec.tmem_columns = int(tmem.get("columns", 0))
                spec.tmem_phases = int(tmem.get("phases", 0))
                spec.tmem_double_buffer = bool(tmem.get("double_buffer", False))
            if hasattr(spec, "tmem_layout") and isinstance(tmem, dict):
                spec.tmem_layout = tmem  # type: ignore[attr-defined]
            if hasattr(spec, "io_extents"):
                spec.io_extents = list(io_extents)
            return spec
        if isinstance(spec_any, dict):
            if not _HAVE_CTYPES:
                raise RuntimeError("bwrt ctypes adapter unavailable; install bwrt runtime")
            cluster = spec_any.get("cluster_shape", (spec_any.get("cluster_x", 1), spec_any.get("cluster_y", 1)))
            swap_window = spec_any.get("swap_window", (spec_any.get("swap_begin", 0), spec_any.get("swap_end", 1)))
            enriched = dict(spec_any)
            enriched.setdefault("cluster_x", int(cluster[0]))
            enriched.setdefault("cluster_y", int(cluster[1]))
            enriched.setdefault("swap_begin", int(swap_window[0]))
            enriched.setdefault("swap_end", int(swap_window[1]))
            tile_flat = _flatten_tile_order(_normalise_tile_order(spec_any.get("tile_order")))
            if tile_flat and "tile_order_flat" not in enriched:
                enriched["tile_order_flat"] = tile_flat
            return _ws_from_dict(enriched)  # type: ignore[misc]
        if not _HAVE_CTYPES:
            raise RuntimeError("bwrt ctypes adapter unavailable; install bwrt runtime")
        return _ws_from_proto(spec_any)  # type: ignore[misc]

    def submit_and_wait(self, wavespec_any: Any, A: Any, B: Any, C: Any):
        spec = self._to_spec(wavespec_any)
        if self._fast:
            evt = self._runtime.submit_wave(spec, A, B, C)
            self._runtime.wait(evt, 0)
            return self._runtime.sample()
        a_ptr = _ptr_from_obj(A)
        b_ptr = _ptr_from_obj(B)
        c_ptr = _ptr_from_obj(C)
        evt = self._runtime.submit_wave(spec, a_ptr, b_ptr, c_ptr)
        self._runtime.wait(evt)
        return self._runtime.sample()

    def set_weights(self, weights: Any) -> None:
        if self._fast:
            self._runtime.set_weights(weights)
            return
        w_ptr = _ptr_from_obj(weights)
        self._runtime.set_weights(w_ptr)


__all__: Iterable[str] = ("BwRuntimeAdapter", "_ptr_from_obj")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import numpy as np

    adapter = BwRuntimeAdapter()
    spec = {
        "pack_order": [0, 1],
        "tile_order": [(0, 0)],
        "bm": 2,
        "bn": 2,
        "bk": 2,
        "cluster_shape": (1, 1),
        "tmem_layout": {"columns": 1, "phases": 1, "double_buffer": False, "stage_n": 1},
        "io_extents": [("0", 0, 0)],
        "swap_window": (0, 3),
    }
    A = np.array([[1, 2], [3, 4]], dtype=np.float32, order="C")
    B = np.array([[5, 6], [7, 8]], dtype=np.float32, order="C")
    C = np.zeros((2, 2), dtype=np.float32, order="C")
    metrics = adapter.submit_and_wait(spec, A, B, C)
    print("C:", C)
    if metrics is not None:
        for field in ("wgmma_active", "tma_occ", "l2_hit"):
            print(field, getattr(metrics, field, None))
