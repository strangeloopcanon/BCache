from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


class SegmentedFileBackend:
    """
    Segment-per-layer file backend.
    Each (model_id, model_version, layer) maps to one file with fixed-size pages.
    Page offset = page_id * page_bytes.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _seg_path(self, model_id: str, model_version: str, layer: int) -> Path:
        return self.root / model_id / model_version / f"layer_{layer}.seg"

    def ensure_segment(self, model_id: str, model_version: str, layer: int):
        p = self._seg_path(model_id, model_version, layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()

    def write_page(self, model_id: str, model_version: str, layer: int, page_id: int, page_bytes: int, data: bytes):
        assert len(data) == page_bytes, "data length must equal page_bytes"
        self.ensure_segment(model_id, model_version, layer)
        p = self._seg_path(model_id, model_version, layer)
        with p.open('r+b') as f:
            off = page_id * page_bytes
            f.seek(off)
            f.write(data)

    def read_range(self, model_id: str, model_version: str, layer: int, start_pid: int, end_pid: int, page_bytes: int) -> bytes:
        """Read a consecutive page range [start_pid, end_pid] inclusive as one coalesced IO.

        Raises IOError if the segment does not contain the full requested range.
        """
        if end_pid < start_pid:
            return b""
        self.ensure_segment(model_id, model_version, layer)
        p = self._seg_path(model_id, model_version, layer)
        size = (end_pid - start_pid + 1) * page_bytes
        with p.open('rb') as f:
            off = start_pid * page_bytes
            f.seek(0, os.SEEK_END)
            seg_size = f.tell()
            if off + size > seg_size:
                raise IOError(f"segment too small for read: need {off+size} bytes, have {seg_size} (layer={layer} start={start_pid} end={end_pid})")
            f.seek(off)
            buf = f.read(size)
            if len(buf) != size:
                raise IOError(f"short read: expected {size} bytes, got {len(buf)}")
            return buf

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
        """Read range directly into a writable buffer supporting the buffer protocol.

        Returns the number of bytes written.
        """
        if end_pid < start_pid:
            return 0
        self.ensure_segment(model_id, model_version, layer)
        p = self._seg_path(model_id, model_version, layer)
        size = (end_pid - start_pid + 1) * page_bytes
        mv = memoryview(out_buf)
        if mv.readonly:
            raise ValueError("out_buf must be writable")
        if mv.nbytes < size:
            raise ValueError(f"out_buf too small: need {size}, have {mv.nbytes}")
        with p.open('rb') as f:
            off = start_pid * page_bytes
            f.seek(0, os.SEEK_END)
            seg_size = f.tell()
            if off + size > seg_size:
                raise IOError(
                    f"segment too small for read: need {off+size} bytes, have {seg_size} "
                    f"(layer={layer} start={start_pid} end={end_pid})"
                )
            f.seek(off)
            # Use readinto for zero-copy into provided buffer
            view = mv.cast('B')[:size]
            n = f.readinto(view)
            if n != size:
                raise IOError(f"short read: expected {size} bytes, got {n}")
            return n
