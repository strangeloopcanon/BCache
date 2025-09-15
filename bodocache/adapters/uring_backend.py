from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class SegmentedUringBackend:
    """Segmented backend using io_uring (native module) for high-throughput reads.

    Falls back to raising ImportError if the native module is not available.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            import bodocache_agent_io_uring as uring  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("bodocache_agent_io_uring module not found. Build with -DUSE_URING=ON")
        self._uring = uring

    def _seg_path(self, model_id: str, model_version: str, layer: int) -> Path:
        p = self.root / model_id / model_version / f"layer_{layer}.seg"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_range_into(
        self,
        model_id: str,
        model_version: str,
        layer: int,
        start_pid: int,
        end_pid: int,
        page_bytes: int,
        out_buf,
    ) -> int:
        if end_pid < start_pid:
            return 0
        p = self._seg_path(model_id, model_version, layer)
        size = (end_pid - start_pid + 1) * page_bytes
        offset = start_pid * page_bytes
        return int(self._uring.read_range_into(str(p), int(offset), int(size), out_buf))

