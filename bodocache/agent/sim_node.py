from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_plan(plan_df: pd.DataFrame, tier_caps_df: pd.DataFrame, window_ms: int = 20) -> pd.DataFrame:
    """Simulate execution of a plan per (node,tier_dst) using simple bandwidth models.

    - Assumes each (node,tier) has bandwidth_caps bytes per window_ms.
    - Computes finish_time for each op as cumulative_bytes/bandwidth_caps * window_ms.
    - Computes timeliness vs relative deadlines (relative to min deadline per group).
    Returns a DataFrame with per-op metrics.
    """
    if plan_df.empty:
        return plan_df.assign(finish_ms=np.float64(0.0),
                              deadline_rel_ms=np.float64(0.0),
                              on_time=np.int64(1))
    df = plan_df.copy()
    caps = tier_caps_df[["tier", "bandwidth_caps"]].rename(columns={"tier": "tier_dst"})
    df = df.merge(caps, on="tier_dst", how="left")
    df = df.sort_values(by=["node", "tier_dst", "deadline_ms"]).reset_index(drop=True)
    grp = ["node", "tier_dst"]
    df["cum_bytes"] = df.groupby(grp)["bytes"].cumsum()
    denom = df["bandwidth_caps"].astype(np.float64).where(lambda x: x > 0, 1.0)
    df["finish_ms"] = (df["cum_bytes"].astype(np.float64) / denom) * float(window_ms)
    base_deadline = df.groupby(grp)["deadline_ms"].transform("min")
    df["deadline_rel_ms"] = (df["deadline_ms"] - base_deadline).astype(np.float64)
    df["on_time"] = (df["finish_ms"] <= df["deadline_rel_ms"]).astype(np.int64)
    return df


def summarize_metrics(exec_df: pd.DataFrame) -> dict:
    if exec_df.empty:
        return {
            "prefetch_timeliness": 1.0,
            "avg_finish_ms": 0.0,
            "avg_io_bytes": 0.0,
            "ops": 0,
    }
    return {
        "prefetch_timeliness": float(exec_df["on_time"].mean()),
        "avg_finish_ms": float(exec_df["finish_ms"].mean()),
        "avg_io_bytes": float(exec_df["bytes"].mean()),
        "ops": int(len(exec_df)),
    }


def simulate_plan_streams(
    plan_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
    window_ms: int = 20,
    streams_per_tier: int = 4,
    use_overlap: bool = True,
    layer_lat_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Multistream simulation with overlap/priority hints.

    - Sort by (node,tier_dst, priority desc, deadline asc).
    - Each (node,tier) has N equal-bandwidth streams: bw_stream = bandwidth_caps / N.
    - If use_overlap, an op gets speedup = min(overlap, N); effective bytes = bytes / speedup.
    - Assign each op to earliest-available stream and compute finish time.
    - Compare finish time to relative deadlines to estimate on-time completion.
    """
    if plan_df.empty:
        return plan_df.assign(finish_ms=np.float64(0.0),
                              deadline_rel_ms=np.float64(0.0),
                              on_time=np.int64(1))
    df = plan_df.copy()
    caps = tier_caps_df[["tier", "bandwidth_caps"]].rename(columns={"tier": "tier_dst"})
    df = df.merge(caps, on="tier_dst", how="left")
    df = df.sort_values(by=["node", "tier_dst", "priority"], ascending=[True, True, False]).reset_index(drop=True)
    # Build per-layer cumulative compute deadline if provided
    cum_deadline = None
    if layer_lat_df is not None and len(layer_lat_df) > 0:
        lat = layer_lat_df.sort_values('layer')
        lat['cum_ms'] = lat['lat_ms'].astype(float).cumsum()
        cum_deadline = dict(zip(lat['layer'].tolist(), lat['cum_ms'].tolist()))

    results = []
    for (node, tier), grp in df.groupby(["node", "tier_dst"], sort=False):
        bw_total = float(grp["bandwidth_caps"].iloc[0])
        bw_per = bw_total / max(1, streams_per_tier)
        # Stream availability times (ms)
        stream_time = [0.0 for _ in range(max(1, streams_per_tier))]
        for row in grp.itertuples(index=False):
            # Effective bytes with overlap
            ov = int(getattr(row, "overlap", 1)) if use_overlap else 1
            eff = max(1, min(ov, streams_per_tier))
            bytes_eff = float(row.bytes) / float(eff)
            dur_ms = (bytes_eff / max(1.0, bw_per)) * float(window_ms)
            # Assign to earliest-available stream
            sidx = min(range(len(stream_time)), key=lambda i: stream_time[i])
            start = stream_time[sidx]
            finish = start + dur_ms
            stream_time[sidx] = finish
            # Required time: from base window start to compute arrival for layer
            req_deadline = float(row.deadline_ms)
            if cum_deadline is not None:
                req_deadline = float(cum_deadline.get(int(getattr(row, 'layer', 0)), 0.0))
            results.append((node, row.tier_dst, getattr(row, 'pcluster', -1), getattr(row, 'layer', -1), float(row.priority), req_deadline, finish, float(row.bytes)))

    out = pd.DataFrame.from_records(
        results,
        columns=["node", "tier_dst", "pcluster", "layer", "priority", "deadline_ms", "finish_ms", "bytes"],
    )
    # On-time if finished before required compute arrival for that layer
    out['on_time'] = (out['finish_ms'] <= out['deadline_ms']).astype(np.int64)
    out['deadline_rel_ms'] = out['deadline_ms']
    return out
