from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol


@dataclass
class CopyOp:
    """Generic copy operation descriptor.

    This is intentionally vendor-neutral. Native backends may reinterpret
    fields as device pointers/stream ids. The simulation backend uses
    Python bytes for src.
    """

    # In a native engine, `src` would be a pointer into pinned host memory.
    # In the sim backend we pass Python `bytes`.
    src: Any
    # Destination descriptor (engine/adapter-specific). For native engines,
    # this would be a device pointer plus shape/stride metadata managed by the
    # integration adapter.
    dst: Any
    bytes: int
    stream_id: int = 0
    gpu_id: int = 0
    deadline_ms: int = 0


class AbstractCopyEngine(Protocol):
    def submit(self, ops: List[CopyOp], callback: Callable[[CopyOp], None]) -> None:
        """Submit a batch of copy ops and invoke callback as they complete."""


class SimCopyEngine:
    """A minimal in-process, CPU-only engine.

    This is a placeholder to preserve functionality without requiring a
    compiled backend. It immediately invokes the callback for each op.
    """

    def submit(self, ops: List[CopyOp], callback: Callable[[CopyOp], None]) -> None:  # type: ignore[override]
        # Micro-sleep to mimic async behavior without blocking too long.
        for op in ops:
            # 0.05ms per op for a tiny hint of asynchrony
            time.sleep(0.00005)
            callback(op)

    def acquire_host_buffer(self, nbytes: int):  # type: ignore[override]
        # Return a writable bytearray as a stand-in for pinned memory.
        return memoryview(bytearray(nbytes))


def load_native_copy_engine() -> Optional[AbstractCopyEngine]:
    """Try loading a native (pybind11) copy engine if available.

    The expected module name can be standardized later. For now, we probe a
    couple of common names and fall back to None if not present.
    """
    candidates = (
        "bodocache_agent_copy_engine",
        "bodocache.copy_engine_native",
        "copy_engine_native",
    )
    for name in candidates:
        try:  # pragma: no cover - import optional native module
            mod = __import__(name, fromlist=["CopyEngine"])  # type: ignore
            engine = getattr(mod, "CopyEngine", None)
            if engine is not None:
                return engine()  # type: ignore[return-value]
        except Exception:
            continue
    return None


def get_copy_engine(prefer_native: bool = True) -> AbstractCopyEngine:
    """Return a usable copy engine instance.

    - If prefer_native is True and a native engine is importable, use it.
    - Otherwise use the SimCopyEngine.
    """
    if prefer_native:
        native = load_native_copy_engine()
        if native is not None:
            return native
    return SimCopyEngine()
