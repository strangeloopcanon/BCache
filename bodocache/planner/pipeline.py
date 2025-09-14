from __future__ import annotations

import numpy as np
import pandas as pd


def score_and_filter(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    now_ms: int,
    pmin: float,
    umin: float,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    """Merge heat, compute popularity/urgency, and filter candidates.

    Returns a copy of requests with added columns: decay_hits, tenant_weight, pop, urgency.
    """
    df = requests_df.copy()
    heat = heat_df.copy()
    df["page_id"] = df["page_start"]
    heat_cols = ["layer", "page_id", "decay_hits", "tenant_weight"]
    # Ensure optional columns exist
    if "tenant_weight" not in heat.columns:
        heat["tenant_weight"] = 1.0
    heat = heat[heat_cols]
    df = df.merge(heat, on=["layer", "page_id"], how="left")
    df["decay_hits"] = df["decay_hits"].fillna(0).astype(np.int64)
    df["tenant_weight"] = df["tenant_weight"].astype(float).where(df["tenant_weight"].notna(), 1.0)
    df["pop"] = float(alpha) * df["decay_hits"] + float(beta) * df["tenant_weight"]
    denom = np.maximum(df["est_fill_ms"].astype(float), 1.0)
    df["urgency"] = (df["deadline_ms"] - now_ms) / denom
    return df[(df["pop"] > float(pmin)) | (df["urgency"] > float(umin))].copy()


def apply_tenant_caps(cand: pd.DataFrame, tenant_caps_df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-(node,tier_dst,tenant) bandwidth caps as cumulative bytes gate.

    Expects columns: tenant, tier_dst, page_start, page_end, page_bytes.
    """
    out = cand.copy()
    out["length"] = (out["page_end"] - out["page_start"] + 1).astype(np.int64)
    out["bytes_row"] = out["length"].astype(np.int64) * out["page_bytes"].astype(np.int64)
    tcap = tenant_caps_df.rename(columns={"tier": "tier_dst", "bandwidth_caps": "tenant_cap"})
    out = out.merge(tcap[["tenant", "tier_dst", "tenant_cap"]], on=["tenant", "tier_dst"], how="left")
    out["tenant_cap"] = out["tenant_cap"].astype(float).where(out["tenant_cap"].notna(), 9_223_372_036_854_775_807)
    out = out.sort_values(by=["node", "tier_dst", "tenant", "deadline_ms"]).reset_index(drop=True)
    grp_t = ["node", "tier_dst", "tenant"]
    out["cum_bytes_tenant"] = out.groupby(grp_t)["bytes_row"].cumsum()
    return out[out["cum_bytes_tenant"] <= out["tenant_cap"]]


def coalesce_intervals(
    cand: pd.DataFrame,
    min_io_bytes: int,
) -> pd.DataFrame:
    """Coalesce contiguous/overlapping intervals into runs and filter by min_io_bytes.

    Returns columns: node,tier_src,tier_dst,pcluster,layer,run_id,bytes,deadline_ms,fanout,urgency_min,start_pid,end_pid,page_bytes.
    """
    sort_cols = ["node", "tier_src", "tier_dst", "pcluster", "layer", "page_start", "page_end"]
    c = cand.sort_values(by=sort_cols)
    grp = ["node", "tier_src", "tier_dst", "pcluster", "layer"]
    prev_end = c.groupby(grp)["page_end"].shift(1).fillna(-1)
    new_run = (c["page_start"] > (prev_end + 1)).astype(np.int64)
    c = c.copy()
    c["new_run"] = new_run
    c["run_id"] = c.groupby(grp)["new_run"].cumsum()
    run_grp = grp + ["run_id"]
    # Compute previous cumulative max of end within each run, shifting inside the group
    g = c.groupby(run_grp)
    cummax_end = g["page_end"].cummax()
    prev_cummax_end = (
        c.assign(_cummax_end=cummax_end)
         .groupby(run_grp)["_cummax_end"].shift(1)
         .fillna(-1)
    )
    eff_start = np.maximum(c["page_start"].astype(np.int64), (prev_cummax_end + 1).astype(np.int64))
    pages = np.maximum(0, c["page_end"].astype(np.int64) - eff_start + 1)
    c["_pages"] = pages
    runs = (
        c.groupby(run_grp)
        .agg(
            pages=("_pages", "sum"),
            page_bytes=("page_bytes", "max"),
            deadline_ms=("deadline_ms", "min"),
            fanout=("page_start", "count"),
            urgency_min=("urgency", "min"),
            start_pid=("page_start", "min"),
            end_pid=("page_end", "max"),
        )
        .reset_index()
    )
    runs["bytes"] = runs["pages"].astype(np.int64) * runs["page_bytes"].astype(np.int64)
    plan = runs[runs["bytes"] >= int(min_io_bytes)][[
        "node", "tier_src", "tier_dst", "pcluster", "layer", "run_id", "bytes", "deadline_ms", "fanout", "urgency_min", "start_pid", "end_pid", "page_bytes"
    ]]
    return plan.reset_index(drop=True)


def apply_caps(
    plan_runs: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
    layer_lat_df: pd.DataFrame,
    window_ms: int,
    max_ops_per_tier: int,
    enforce_tier_caps: bool,
) -> pd.DataFrame:
    """Apply per-tier caps, limit ops per tier, compute overlap and priority.
    Returns final plan DataFrame.
    """
    plan = plan_runs.merge(
        tier_caps_df[["tier", "bandwidth_caps", "free_bytes"]].rename(columns={"tier": "tier_dst"}),
        on="tier_dst", how="left",
    )
    lat = layer_lat_df[["layer", "lat_ms"]]
    plan = plan.merge(lat, on="layer", how="left")
    plan["lat_ms"] = plan["lat_ms"].astype(float).where(plan["lat_ms"].notna(), 1.0)
    plan = plan.sort_values(by=["node", "tier_src", "tier_dst", "deadline_ms"]).reset_index(drop=True)
    grp2 = ["node", "tier_src", "tier_dst"]
    plan["cum_bytes"] = plan.groupby(grp2)["bytes"].cumsum()
    eff_cap = np.minimum(
        plan["bandwidth_caps"].fillna(9_223_372_036_854_775_807),
        plan["free_bytes"].fillna(9_223_372_036_854_775_807),
    )
    if enforce_tier_caps:
        plan = plan[plan["cum_bytes"] <= eff_cap]
    # Limit ops per (node,tier_dst)
    MAX_OPS = np.int64(max_ops_per_tier)
    plan["one"] = np.int64(1)
    plan["op_rank"] = plan.groupby(["node", "tier_dst"])['one'].cumsum()
    # Estimate copy time per op
    bc = plan["bandwidth_caps"].astype(float)
    denom_cap = bc.where(bc > 0, 1.0)
    plan["est_copy_ms"] = (plan["bytes"].astype(float) / np.maximum(denom_cap, 1.0)) * float(window_ms)
    plan = plan[plan["op_rank"] <= MAX_OPS]
    # Overlap hint
    gt1 = (plan["est_copy_ms"] > plan["lat_ms"]).astype(np.int64)
    gt2 = (plan["est_copy_ms"] > (2.0 * plan["lat_ms"]).astype(float)).astype(np.int64)
    plan["overlap"] = np.minimum(np.int64(3), np.int64(1) + gt1 + gt2)
    plan["priority"] = plan["urgency_min"]
    plan = plan[[
        "node", "tier_src", "tier_dst", "pcluster", "layer", "run_id",
        "bytes", "deadline_ms", "fanout", "overlap", "priority",
        "start_pid", "end_pid", "page_bytes",
    ]]
    return plan.reset_index(drop=True)
