from __future__ import annotations

import time
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend


class NodeAgent:
    """Python Node Agent that executes a plan using a segmented file backend.

    This models STORAGE->CPU transfers as coalesced reads from segment files.
    """

    def __init__(self, backend: SegmentedFileBackend, page_bytes: int = 256 * 1024):
        self.backend = backend
        self.page_bytes = page_bytes

    def execute(self, plan_df: pd.DataFrame, model_id: str, model_version: str, on_ready: Callable[[Dict[str, Any]], None] | None = None) -> Dict[str, Any]:
        if plan_df.empty:
            return {"ops": 0, "bytes": 0, "duration_ms": 0.0}
        t0 = time.time()
        total_bytes = 0
        for r in plan_df.itertuples(index=False):
            data = self.backend.read_range(model_id, model_version, int(r.layer), int(r.start_pid), int(r.end_pid), int(getattr(r, 'page_bytes', self.page_bytes)))
            total_bytes += len(data)
            if on_ready is not None:
                on_ready({
                    "node": getattr(r, 'node', ''),
                    "layer": int(r.layer),
                    "start_pid": int(r.start_pid),
                    "end_pid": int(r.end_pid),
                    "bytes": len(data),
                })
        dt = (time.time() - t0) * 1000.0
        return {"ops": int(len(plan_df)), "bytes": int(total_bytes), "duration_ms": float(dt)}

