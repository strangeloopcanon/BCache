from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd


class TelemetryLogger:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _write(self, df: pd.DataFrame, name: str):
        if df is None or len(df) == 0:
            return
        path = os.path.join(self.base_dir, f"{name}.csv")
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False)

    def log_window(
        self,
        req: pd.DataFrame,
        heat: pd.DataFrame,
        tiers: pd.DataFrame,
        lats: pd.DataFrame,
        plan: pd.DataFrame,
        exec_df: Optional[pd.DataFrame] = None,
        evict_df: Optional[pd.DataFrame] = None,
        admission_df: Optional[pd.DataFrame] = None,
    ):
        ts = int(time.time() * 1000)
        req = req.copy()
        req["ts"] = ts
        heat = heat.copy()
        heat["ts"] = ts
        tiers = tiers.copy()
        tiers["ts"] = ts
        lats = lats.copy()
        lats["ts"] = ts
        plan = plan.copy()
        plan["ts"] = ts
        if exec_df is not None:
            exec_df = exec_df.copy()
            exec_df["ts"] = ts
        if evict_df is not None:
            evict_df = evict_df.copy()
            evict_df["ts"] = ts
        if admission_df is not None:
            admission_df = admission_df.copy()
            admission_df["ts"] = ts
        self._write(req, "requests")
        self._write(heat, "heat")
        self._write(tiers, "tiers")
        self._write(lats, "latencies")
        self._write(plan, "plans")
        if exec_df is not None:
            self._write(exec_df, "exec")
        if evict_df is not None and len(evict_df) > 0:
            self._write(evict_df, "evict")
        if admission_df is not None and len(admission_df) > 0:
            self._write(admission_df, "admission")
