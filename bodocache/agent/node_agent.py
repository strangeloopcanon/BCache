from __future__ import annotations

import time
from typing import Callable, Dict, Any, Optional

import numpy as np
import pandas as pd

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from .copy_engine import AbstractCopyEngine, CopyOp, get_copy_engine


class NodeAgent:
    """Python Node Agent that executes a plan using a segmented file backend.

    This models STORAGE->CPU transfers as coalesced reads from segment files.
    """

    def __init__(
        self,
        backend: SegmentedFileBackend,
        page_bytes: int = 256 * 1024,
        copy_engine: Optional[AbstractCopyEngine] = None,
    ):
        self.backend = backend
        self.page_bytes = page_bytes
        # Optional device copy engine. Falls back to a simulated engine if explicitly requested.
        self.copy_engine = copy_engine

    def execute(
        self,
        plan_df: pd.DataFrame,
        model_id: str,
        model_version: str,
        on_ready: Optional[Callable[[Dict[str, Any]], None]] = None,
        dest_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
        prefer_native_engine: bool = True,
    ) -> Dict[str, Any]:
        if plan_df.empty:
            return {"ops": 0, "bytes": 0, "duration_ms": 0.0}
        t0 = time.time()
        total_bytes = 0
        for r in plan_df.itertuples(index=False):
            layer = int(r.layer)
            start_pid = int(r.start_pid)
            end_pid = int(r.end_pid)
            page_bytes = int(getattr(r, "page_bytes", self.page_bytes))
            # Compute total bytes for this coalesced read
            nbytes = (end_pid - start_pid + 1) * page_bytes if end_pid >= start_pid else 0
            total_bytes += nbytes

            # If a device copy engine is available and a destination is provided, enqueue a copy.
            # Otherwise, treat the read as "ready" immediately.
            if self.copy_engine is None and dest_resolver is not None and prefer_native_engine:
                # Try to lazily load a native engine if requested and not provided.
                self.copy_engine = get_copy_engine(prefer_native=prefer_native_engine)

            dst = dest_resolver({
                "node": getattr(r, "node", ""),
                "layer": layer,
                "start_pid": start_pid,
                "end_pid": end_pid,
                "bytes": nbytes,
            }) if dest_resolver is not None else None

            if self.copy_engine is not None and dst is not None:
                # Use pinned buffer path if supported by the engine
                src_buf = None
                acquire = getattr(self.copy_engine, "acquire_host_buffer", None)
                if callable(acquire) and nbytes > 0:
                    try:
                        src_buf = acquire(nbytes)  # expected to be a writable memoryview
                    except Exception:
                        src_buf = None

                if src_buf is not None:
                    # Read directly into pinned buffer and submit device copy
                    self.backend.read_range_into(
                        model_id,
                        model_version,
                        layer,
                        start_pid,
                        end_pid,
                        page_bytes,
                        src_buf,
                    )
                    op = CopyOp(
                        src=src_buf,
                        dst=dst,
                        bytes=nbytes,
                        stream_id=int(getattr(r, "overlap", 1)) - 1 if hasattr(r, "overlap") else 0,
                        gpu_id=int(getattr(r, "gpu_id", 0)) if hasattr(r, "gpu_id") else 0,
                        deadline_ms=int(getattr(r, "deadline_ms", 0)) if hasattr(r, "deadline_ms") else 0,
                    )

                    def _done(_op: CopyOp, _r=r) -> None:
                        if on_ready is not None:
                            on_ready(
                                {
                                    "node": getattr(_r, "node", ""),
                                    "layer": int(_r.layer),
                                    "start_pid": int(_r.start_pid),
                                    "end_pid": int(_r.end_pid),
                                    "bytes": nbytes,
                                }
                            )

                    # Submit as a single-op batch to keep context simple.
                    self.copy_engine.submit([op], _done)
                    continue

            # Fallback: CPU read and mark ready
            data = self.backend.read_range(
                model_id,
                model_version,
                layer,
                start_pid,
                end_pid,
                page_bytes,
            )
            if on_ready is not None and nbytes > 0:
                on_ready(
                    {
                        "node": getattr(r, "node", ""),
                        "layer": layer,
                        "start_pid": start_pid,
                        "end_pid": end_pid,
                        "bytes": len(data),
                    }
                )
        dt = (time.time() - t0) * 1000.0
        return {"ops": int(len(plan_df)), "bytes": int(total_bytes), "duration_ms": float(dt)}

    def prefetch_wave(
        self,
        wave: Dict[str, Any],
        *,
        model_id: str,
        model_version: str,
        on_ready: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the I/O extents specified by a WaveSpec.

        This is a minimal, simulator-friendly path for correctness. It reads the
        (layer,start_pid,end_pid) ranges and invokes on_ready when done.
        """
        io_extents = wave.get("io_extents", [])
        total_bytes = 0
        t0 = time.time()
        for layer, start_pid, end_pid in io_extents:
            page_bytes = int(wave.get("page_bytes", getattr(self, "page_bytes", 256 * 1024)))
            if end_pid < start_pid:
                continue
            nbytes = (int(end_pid) - int(start_pid) + 1) * page_bytes
            _ = self.backend.read_range(model_id, model_version, int(layer), int(start_pid), int(end_pid), page_bytes)
            total_bytes += nbytes
            if on_ready is not None:
                on_ready({
                    "layer": int(layer),
                    "start_pid": int(start_pid),
                    "end_pid": int(end_pid),
                    "bytes": int(nbytes),
                })
        dt = (time.time() - t0) * 1000.0
        return {"ops": int(len(io_extents)), "bytes": int(total_bytes), "duration_ms": float(dt)}
