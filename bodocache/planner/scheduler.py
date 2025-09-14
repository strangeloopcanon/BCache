from __future__ import annotations

from typing import Tuple
import os

import numpy as np
import pandas as pd

# Optional Bodo dependency: provide a no-op fallback for tests/runtime without Bodo
try:  # pragma: no cover - trivial import/fallback
    import bodo  # type: ignore
except Exception:  # Bodo not available; define a minimal shim with a no-op jit decorator
    class _NoBodo:  # pragma: no cover - simple decorator shim
        def jit(self, func=None, **kwargs):
            if func is None:
                def wrapper(f):
                    return f
                return wrapper
            return func

    bodo = _NoBodo()  # type: ignore

from .pipeline import score_and_filter, apply_tenant_caps, coalesce_intervals, apply_caps

@bodo.jit
def run_window_core(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
    tenant_caps_df: pd.DataFrame,
    layer_lat_df: pd.DataFrame,
    now_ms: int,
    pmin: float,
    umin: float,
    min_io_bytes: int,
    alpha: float,
    beta: float,
    window_ms: int,
    max_ops_per_tier: int,
    enforce_tier_caps: bool | np.bool_,
) -> pd.DataFrame:
    """
    Bodo-compiled core planner. Uses interval union + coalescing without Python loops.

    requests_df columns:
      [req_id,node,model_id,model_version,prefix_id,layer,page_start,page_end,
       tier_src,tier_dst,deadline_ms,page_bytes,tenant,est_fill_ms]
    heat_df columns:
      [layer,page_id,decay_hits,tenant_weight]

    enforce_tier_caps: when true, applies per-(node,tier_dst) cumulative bytes gating
    against min(bandwidth_caps, free_bytes). When false, computes caps but does not
    drop ops based on caps.
    """
    df = requests_df
    heat = heat_df

    # Expect est_fill_ms and tenant_weight to be present; avoid dtype changes inside JIT

    # Join heat via (layer, page_start) -> decay_hits
    df["page_id"] = df["page_start"]
    heat_cols = ["layer", "page_id", "decay_hits", "tenant_weight"]
    heat = heat[heat_cols]
    df = df.merge(heat, on=["layer", "page_id"], how="left")
    df["decay_hits"] = df["decay_hits"].fillna(0).astype(np.int64)
    df["tenant_weight"] = df["tenant_weight"].astype(np.float64).where(df["tenant_weight"].notna(), 1.0)

    # Scores
    df["pop"] = alpha * df["decay_hits"] + beta * df["tenant_weight"]
    denom = np.maximum(df["est_fill_ms"].astype(np.float64), 1.0)
    df["urgency"] = (df["deadline_ms"] - now_ms) / denom

    cand = df[(df["pop"] > pmin) | (df["urgency"] > umin)].copy()

    # Apply per-tenant credits (token-bucket) per (node,tier_dst,tenant)
    # Approximate bytes demand per request interval: (page_end-page_start+1)*page_bytes
    cand["length"] = (cand["page_end"] - cand["page_start"] + 1).astype(np.int64)
    cand["bytes_row"] = cand["length"].astype(np.int64) * cand["page_bytes"].astype(np.int64)
    tcap = tenant_caps_df.rename(columns={"tier": "tier_dst", "bandwidth_caps": "tenant_cap"})
    cand = cand.merge(tcap[["tenant", "tier_dst", "tenant_cap"]], on=["tenant", "tier_dst"], how="left")
    cand["tenant_cap"] = cand["tenant_cap"].astype(np.float64).where(cand["tenant_cap"].notna(), 9_223_372_036_854_775_807)
    cand = cand.sort_values(by=["node", "tier_dst", "tenant", "deadline_ms"]).reset_index(drop=True)
    grp_t = ["node", "tier_dst", "tenant"]
    cand["cum_bytes_tenant"] = cand.groupby(grp_t)["bytes_row"].cumsum()
    cand = cand[cand["cum_bytes_tenant"] <= cand["tenant_cap"]]

    # Prefix clustering for fan-out: use precomputed numeric cluster id
    # (prepared outside JIT in run_window)

    # Sort to detect contiguous/overlapping unions per group including cluster
    sort_cols = ["node", "tier_src", "tier_dst", "pcluster", "layer", "page_start", "page_end"]
    cand = cand.sort_values(by=sort_cols)

    grp = ["node", "tier_src", "tier_dst", "pcluster", "layer"]
    prev_end = cand.groupby(grp)["page_end"].shift(1).fillna(-1)
    new_run = (cand["page_start"] > (prev_end + 1)).astype(np.int64)

    cand["new_run"] = new_run
    cand["run_id"] = cand.groupby(grp)["new_run"].cumsum()

    # Union length via cumulative max of end within each run
    run_grp = grp + ["run_id"]
    # Compute previous cumulative max per run_id within group, shifting inside the group
    g = cand.groupby(run_grp)
    cummax_end = g["page_end"].cummax()
    prev_cummax_end = (
        cand.assign(_cummax_end=cummax_end)
            .groupby(run_grp)["_cummax_end"].shift(1)
            .fillna(-1)
    )
    eff_start = np.maximum(cand["page_start"].astype(np.int64), (prev_cummax_end + 1).astype(np.int64))
    pages = np.maximum(0, cand["page_end"].astype(np.int64) - eff_start + 1)
    cand["_pages"] = pages

    runs = (
        cand.groupby(run_grp)
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
    plan = runs[runs["bytes"] >= int(min_io_bytes)][["node", "tier_src", "tier_dst", "pcluster", "layer", "run_id", "bytes", "deadline_ms", "fanout", "urgency_min", "start_pid", "end_pid", "page_bytes"]]

    # Apply per-tier bandwidth caps per node (greedy by earliest deadline)
    caps = tier_caps_df[["tier", "bandwidth_caps", "free_bytes"]].rename(columns={"tier": "tier_dst"})
    plan = plan.merge(caps, on="tier_dst", how="left")
    # Join per-layer latency (ms)
    lat = layer_lat_df[["layer", "lat_ms"]]
    plan = plan.merge(lat, on="layer", how="left")
    plan["lat_ms"] = plan["lat_ms"].astype(np.float64).where(plan["lat_ms"].notna(), 1.0)
    plan = plan.sort_values(by=["node", "tier_src", "tier_dst", "deadline_ms"]).reset_index(drop=True)
    grp2 = ["node", "tier_src", "tier_dst"]
    plan["cum_bytes"] = plan.groupby(grp2)["bytes"].cumsum()
    # Compute effective cap and optionally enforce it
    eff_cap = np.minimum(
        plan["bandwidth_caps"].fillna(9_223_372_036_854_775_807),
        plan["free_bytes"].fillna(9_223_372_036_854_775_807),
    )
    if enforce_tier_caps:
        keep = plan["cum_bytes"] <= eff_cap
        plan = plan[keep]
    # Limit number of ops per (node,tier_dst) to avoid stream overload
    MAX_OPS = np.int64(max_ops_per_tier)
    plan["one"] = np.int64(1)
    plan["op_rank"] = plan.groupby(["node", "tier_dst"])['one'].cumsum()
    # Estimate copy time (ms) for each op from cap per window
    # bandwidth_caps is bytes per window => ms = (bytes / cap) * window_ms
    bc = plan["bandwidth_caps"].astype(np.float64)
    denom_cap = bc.where(bc > 0, 1.0)
    est_copy_ms = (plan["bytes"].astype(np.float64) / np.maximum(denom_cap, 1.0)) * float(window_ms)
    plan["est_copy_ms"] = est_copy_ms
    plan = plan[plan["op_rank"] <= MAX_OPS]
    # Overlap depth hint: deeper when predicted copy time exceeds per-layer latency budget
    # overlap = 1 + I(copy>lat) + I(copy>2*lat) capped at 3
    gt1 = (plan["est_copy_ms"] > plan["lat_ms" ]).astype(np.int64)
    gt2 = (plan["est_copy_ms"] > (2.0 * plan["lat_ms" ])).astype(np.int64)
    plan["overlap"] = np.minimum(np.int64(3), np.int64(1) + gt1 + gt2)
    plan["priority"] = plan["urgency_min"]
    plan = plan[[
        "node", "tier_src", "tier_dst", "pcluster", "layer", "run_id",
        "bytes", "deadline_ms", "fanout", "overlap", "priority",
        "start_pid", "end_pid", "page_bytes",
    ]]
    return plan.reset_index(drop=True)


def run_window_core_py(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
    tenant_caps_df: pd.DataFrame,
    layer_lat_df: pd.DataFrame,
    now_ms: int,
    pmin: float,
    umin: float,
    min_io_bytes: int,
    alpha: float,
    beta: float,
    window_ms: int,
    max_ops_per_tier: int,
    enforce_tier_caps: bool,
) -> pd.DataFrame:
    # Pure-Python fallback with structured stages mirroring the JIT path
    cand0 = score_and_filter(
        requests_df, heat_df, now_ms, pmin, umin, alpha, beta
    )
    cand1 = apply_tenant_caps(cand0, tenant_caps_df)
    runs = coalesce_intervals(cand1, min_io_bytes=min_io_bytes)
    plan = apply_caps(
        runs,
        tier_caps_df=tier_caps_df,
        layer_lat_df=layer_lat_df,
        window_ms=window_ms,
        max_ops_per_tier=max_ops_per_tier,
        enforce_tier_caps=bool(enforce_tier_caps),
    )
    return plan


def run_window(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
    tenant_caps_df: pd.DataFrame,
    layer_lat_df: pd.DataFrame,
    now_ms: int,
    pmin: float = 1.0,
    umin: float = 0.0,
    min_io_bytes: int = 512 * 1024,
    alpha: float = 1.0,
    beta: float = 0.0,
    window_ms: int = 20,
    max_ops_per_tier: int = 64,
    enable_admission: bool | np.bool_ = True,
    enable_eviction: bool | np.bool_ = True,
    enforce_tier_caps: bool | np.bool_ = True,
):
    # Ensure numeric prefix clusters for JIT-friendly fan-out grouping
    if "pcluster" not in requests_df.columns:
        codes, _ = pd.factorize(requests_df["prefix_id"], sort=False)
        requests_df = requests_df.copy()
        requests_df["pcluster"] = codes.astype(np.int64)

    FORCE_PY = str(os.environ.get("BODOCACHE_PURE_PY", "")).lower() in ("1", "true", "yes")
    if FORCE_PY:
        plan_df = run_window_core_py(
            requests_df,
            heat_df,
            tier_caps_df,
            tenant_caps_df,
            layer_lat_df,
            now_ms,
            pmin,
            umin,
            min_io_bytes,
            alpha,
            beta,
            window_ms,
            max_ops_per_tier,
            bool(enforce_tier_caps),
        )
    else:
        try:
            plan_df = run_window_core(
                requests_df,
                heat_df,
                tier_caps_df,
                tenant_caps_df,
                layer_lat_df,
                now_ms,
                pmin,
                umin,
                min_io_bytes,
                alpha,
                beta,
                window_ms,
                max_ops_per_tier,
                bool(enforce_tier_caps),
            )
        except Exception:
            # Fallback to pure-Python core if JIT compilation/execution fails
            plan_df = run_window_core_py(
                requests_df,
                heat_df,
                tier_caps_df,
                tenant_caps_df,
                layer_lat_df,
                now_ms,
                pmin,
                umin,
                min_io_bytes,
                alpha,
                beta,
                window_ms,
                max_ops_per_tier,
                bool(enforce_tier_caps),
            )
    # Prepare heat_df for JIT eviction (ensure size_bytes present)
    heat2 = heat_df.copy()
    if "size_bytes" not in heat2.columns:
        heat2["size_bytes"] = np.int64(256 * 1024)
    if bool(enable_eviction):
        if FORCE_PY:
            evict = eviction_core_py(plan_df, heat2, tier_caps_df)
        else:
            try:
                evict = eviction_core(plan_df, heat2, tier_caps_df)
            except Exception:
                evict = eviction_core_py(plan_df, heat2, tier_caps_df)
    else:
        evict = heat2[["layer", "page_id"]].head(0)
    if bool(enable_admission):
        if FORCE_PY:
            admission = admission_core_py(requests_df, heat_df, reuse_threshold=10.0)
        else:
            try:
                admission = admission_core(requests_df, heat_df, reuse_threshold=10.0)
            except Exception:
                admission = admission_core_py(requests_df, heat_df, reuse_threshold=10.0)
    else:
        admission = heat_df[["layer", "page_id"]].head(0)
    return plan_df, evict, admission


@bodo.jit
def admission_core(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    reuse_threshold: float,
) -> pd.DataFrame:
    df = requests_df
    heat = heat_df
    df["page_id"] = df["page_start"]
    heat_cols = ["layer", "page_id", "decay_hits"]
    heat = heat[heat_cols]
    df = df.merge(heat, on=["layer", "page_id"], how="left")
    df["decay_hits"] = df["decay_hits"].fillna(0).astype(np.int64)
    mask = df["decay_hits"].astype(np.float64) >= float(reuse_threshold)
    admit = df[["layer", "page_id"]].copy()
    admit["tier_dst"] = np.int64(0)
    admit = admit[mask][["layer", "page_id", "tier_dst"]]
    # drop_duplicates replacement: groupby and take first
    admit = admit.groupby(["layer", "page_id"]).agg(tier_dst=("tier_dst", "first")).reset_index()
    return admit


def admission_core_py(
    requests_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    reuse_threshold: float,
) -> pd.DataFrame:
    df = requests_df.copy()
    heat = heat_df.copy()
    df["page_id"] = df["page_start"]
    heat_cols = ["layer", "page_id", "decay_hits"]
    heat = heat[heat_cols]
    df = df.merge(heat, on=["layer", "page_id"], how="left")
    df["decay_hits"] = df["decay_hits"].fillna(0).astype(np.int64)
    mask = df["decay_hits"].astype(float) >= float(reuse_threshold)
    admit = df[["layer", "page_id"]].copy()
    admit["tier_dst"] = np.int64(0)
    admit = admit[mask][["layer", "page_id", "tier_dst"]]
    admit = admit.groupby(["layer", "page_id"]).agg(tier_dst=("tier_dst", "first")).reset_index()
    return admit


@bodo.jit
def eviction_core(
    plan_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
) -> pd.DataFrame:
    # Evict cold pages if planned bytes exceed free_bytes for a tier
    if len(plan_df) == 0:
        return heat_df[["layer", "page_id"]].head(0)
    used = plan_df.groupby(["tier_dst"]).agg(bytes=("bytes", "sum")).reset_index()
    caps = tier_caps_df[["tier", "free_bytes"]].rename(columns={"tier": "tier_dst"})
    need = used.merge(caps, on="tier_dst", how="left")
    need["deficit"] = (need["bytes"] - need["free_bytes"]).astype(np.int64)
    need = need[need["deficit"] > 0][["tier_dst", "deficit"]]
    if len(need) == 0:
        return heat_df[["layer", "page_id"]].head(0)

    # Choose coldest pages cluster-wide (no per-tier mapping in heat; approximate)
    df = heat_df.copy()
    df = df.sort_values("decay_hits", ascending=True)[["layer", "page_id", "size_bytes"]]
    df["cum"] = df["size_bytes"].cumsum()
    target = need["deficit"].sum()
    ev = df[df["cum"] <= target][["layer", "page_id"]]
    return ev.reset_index(drop=True)


def eviction_core_py(
    plan_df: pd.DataFrame,
    heat_df: pd.DataFrame,
    tier_caps_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(plan_df) == 0:
        return heat_df[["layer", "page_id"]].head(0)
    used = plan_df.groupby(["tier_dst"]).agg(bytes=("bytes", "sum")).reset_index()
    caps = tier_caps_df[["tier", "free_bytes"]].rename(columns={"tier": "tier_dst"})
    need = used.merge(caps, on="tier_dst", how="left")
    need["deficit"] = (need["bytes"] - need["free_bytes"]).astype(np.int64)
    need = need[need["deficit"] > 0][["tier_dst", "deficit"]]
    if len(need) == 0:
        return heat_df[["layer", "page_id"]].head(0)
    df = heat_df.copy()
    df = df.sort_values("decay_hits", ascending=True)[["layer", "page_id", "size_bytes"]]
    df["cum"] = df["size_bytes"].cumsum()
    target = need["deficit"].sum()
    ev = df[df["cum"] <= target][["layer", "page_id"]]
    return ev.reset_index(drop=True)
