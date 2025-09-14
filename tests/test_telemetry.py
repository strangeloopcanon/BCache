from __future__ import annotations

import os
import pandas as pd

from bodocache.telemetry.logger import TelemetryLogger


def test_telemetry_logger_append(tmp_path):
    tl = TelemetryLogger(base_dir=str(tmp_path))
    req = pd.DataFrame([[0]], columns=["req_id"])  # minimal
    heat = pd.DataFrame([[0,0,1,1.0]], columns=["layer","page_id","decay_hits","tenant_weight"])
    tiers = pd.DataFrame([[0,1,0,1]], columns=["tier","free_bytes","inflight_io","bandwidth_caps"])
    lats = pd.DataFrame([[0,1.0]], columns=["layer","lat_ms"])
    plan = pd.DataFrame([["n0",0,1,0,0,0,1024,0,1,1.0,0,1,256*1024,256*1024]],
                        columns=["node","tier_src","tier_dst","pcluster","layer","run_id","bytes","deadline_ms","fanout","overlap","priority","start_pid","end_pid","page_bytes"])

    tl.log_window(req, heat, tiers, lats, plan)
    tl.log_window(req, heat, tiers, lats, plan)
    # Files created and have > 1 row (two appends)
    for name in ["requests","heat","tiers","latencies","plans"]:
        path = os.path.join(str(tmp_path), f"{name}.csv")
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) >= 2
