from __future__ import annotations

import time

import pandas as pd
from bodocache.planner.api import (
    HeatEntry,
    LayerLatency,
    PlannerConfig,
    PlannerRequest,
    PlannerWindow,
    TenantCapacity,
    TierCapacity,
    plan_window,
)
from bodocache.planner.models import DEFAULT_PAGE_BYTES


def _window_from_frames(
    req: pd.DataFrame,
    heat: pd.DataFrame,
    tiers: pd.DataFrame,
    tenant_caps: pd.DataFrame,
    lats: pd.DataFrame,
    now_ms: int,
) -> PlannerWindow:
    requests_payload = [
        PlannerRequest(
            req_id=str(row["req_id"]),
            node=str(row["node"]),
            model_id=str(row["model_id"]),
            model_version=str(row["model_version"]),
            prefix_id=str(row["prefix_id"]),
            layer=int(row["layer"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            tier_src=int(row["tier_src"]),
            tier_dst=int(row["tier_dst"]),
            deadline_ms=int(row["deadline_ms"]),
            page_bytes=int(row["page_bytes"]),
            tenant=str(row["tenant"]),
            est_fill_ms=float(row["est_fill_ms"]),
        )
        for row in req.to_dict(orient="records")
    ]
    heat_entries = [
        HeatEntry(
            layer=int(row["layer"]),
            page_id=int(row["page_id"]),
            decay_hits=int(row["decay_hits"]),
            tenant_weight=float(row["tenant_weight"]),
            size_bytes=int(row.get("size_bytes", DEFAULT_PAGE_BYTES)),
        )
        for row in heat.to_dict(orient="records")
    ]
    tier_entries = [
        TierCapacity(
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
            free_bytes=int(row["free_bytes"]),
        )
        for row in tiers.to_dict(orient="records")
    ]
    tenant_entries = [
        TenantCapacity(
            tenant=str(row["tenant"]),
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
        )
        for row in tenant_caps.to_dict(orient="records")
    ]
    lat_entries = [
        LayerLatency(layer=int(row["layer"]), lat_ms=float(row["lat_ms"]))
        for row in lats.to_dict(orient="records")
    ]
    return PlannerWindow(
        requests=requests_payload,
        now_ms=int(now_ms),
        heat=heat_entries,
        tier_caps=tier_entries,
        tenant_caps=tenant_entries,
        layer_latencies=lat_entries,
    )


def test_coalescing_runs_min_io():
    now_ms = int(time.time() * 1000)
    # Two contiguous pages on same node/tier should coalesce; one small op filtered out
    req = pd.DataFrame(
        [
            [0, "n0", "m", "v", "p1", 0, 0, 1, 0, 1, now_ms + 1000, 300 * 1024, "t", 1],
            [1, "n0", "m", "v", "p2", 0, 2, 2, 0, 1, now_ms + 1000, 128 * 1024, "t", 1],
        ],
        columns=[
            "req_id",
            "node",
            "model_id",
            "model_version",
            "prefix_id",
            "layer",
            "page_start",
            "page_end",
            "tier_src",
            "tier_dst",
            "deadline_ms",
            "page_bytes",
            "tenant",
            "est_fill_ms",
        ],
    )
    heat = pd.DataFrame(
        [[0, 0, 10, 1.0]], columns=["layer", "page_id", "decay_hits", "tenant_weight"]
    )
    tiers = pd.DataFrame(
        [[0, 1, 0, 1], [1, 1, 0, 1]],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )
    t_caps = pd.DataFrame(
        [["t", 0, 10**12], ["t", 1, 10**12]], columns=["tenant", "tier", "bandwidth_caps"]
    )  # big caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer", "lat_ms"])  # minimal lat profile
    window = _window_from_frames(req, heat, tiers, t_caps, lats, now_ms)
    result = plan_window(
        window,
        PlannerConfig(
            pmin=0.0,
            umin=-1.0,
            enforce_tier_caps=False,
        ),
    )
    plan_df, _, _ = result.as_dataframes()
    # Expect first request pages (2x300KB) to coalesce into >=512KB op.
    # The second (128KB) should be filtered out.
    assert len(plan_df) >= 1
    assert plan_df["bytes"].min() >= 512 * 1024


def test_prefix_cluster_groups():
    now_ms = int(time.time() * 1000)
    # Two identical intervals for two different prefixes should produce two ops
    # due to pcluster grouping.
    req = pd.DataFrame(
        [
            [0, "n0", "m", "v", "p1", 0, 0, 1, 0, 1, now_ms + 1000, 300 * 1024, "t", 1],
            [1, "n0", "m", "v", "p2", 0, 0, 1, 0, 1, now_ms + 1000, 300 * 1024, "t", 1],
        ],
        columns=[
            "req_id",
            "node",
            "model_id",
            "model_version",
            "prefix_id",
            "layer",
            "page_start",
            "page_end",
            "tier_src",
            "tier_dst",
            "deadline_ms",
            "page_bytes",
            "tenant",
            "est_fill_ms",
        ],
    )
    heat = pd.DataFrame(
        [[0, 0, 10, 1.0]], columns=["layer", "page_id", "decay_hits", "tenant_weight"]
    )
    tiers = pd.DataFrame(
        [[0, 1 << 30, 0, 1 << 30], [1, 1 << 30, 0, 1 << 30]],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )
    t_caps = pd.DataFrame(
        [["t", 0, 1 << 40], ["t", 1, 1 << 40]], columns=["tenant", "tier", "bandwidth_caps"]
    )  # big caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer", "lat_ms"])  # minimal lat profile
    window = _window_from_frames(req, heat, tiers, t_caps, lats, now_ms)
    result = plan_window(window, PlannerConfig(pmin=0.0, umin=-1.0))
    plan_df, _, _ = result.as_dataframes()
    assert "pcluster" in plan_df.columns
    # Expect two ops, one per prefix cluster
    assert len(plan_df) == 2


def test_max_ops_cap():
    now_ms = int(time.time() * 1000)
    # Many small coalesced ops across same node/tier; cap should limit per (node,tier)
    rows = []
    for i in range(200):
        rows.append(
            [i, "n0", "m", "v", f"p{i}", 0, i, i, 0, 1, now_ms + 1000 + (i % 5), 256 * 1024, "t", 1]
        )
    req = pd.DataFrame(
        rows,
        columns=[
            "req_id",
            "node",
            "model_id",
            "model_version",
            "prefix_id",
            "layer",
            "page_start",
            "page_end",
            "tier_src",
            "tier_dst",
            "deadline_ms",
            "page_bytes",
            "tenant",
            "est_fill_ms",
        ],
    )
    heat = pd.DataFrame(
        [[0, i, 1, 1.0] for i in range(200)],
        columns=["layer", "page_id", "decay_hits", "tenant_weight"],
    )
    tiers = pd.DataFrame(
        [[0, 1 << 30, 0, 1 << 30], [1, 1 << 30, 0, 1 << 30]],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )
    t_caps = pd.DataFrame(
        [["t", 0, 1 << 40], ["t", 1, 1 << 40]], columns=["tenant", "tier", "bandwidth_caps"]
    )  # big caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer", "lat_ms"])  # minimal lat profile
    window = _window_from_frames(req, heat, tiers, t_caps, lats, now_ms)
    result = plan_window(
        window,
        PlannerConfig(
            pmin=0.0,
            umin=-1.0,
            min_io_bytes=0,
            max_ops_per_tier=8,
        ),
    )
    plan_df, _, _ = result.as_dataframes()
    # Ensure per (node,tier_dst) op count <= cap
    assert plan_df.groupby(["node", "tier_dst"]).size().max() <= 8


def test_admission_eviction_gating():
    now_ms = int(time.time() * 1000)
    req = pd.DataFrame(
        [[0, "n0", "m", "v", "p1", 0, 0, 0, 0, 1, now_ms + 1000, 256 * 1024, "t", 1]],
        columns=[
            "req_id",
            "node",
            "model_id",
            "model_version",
            "prefix_id",
            "layer",
            "page_start",
            "page_end",
            "tier_src",
            "tier_dst",
            "deadline_ms",
            "page_bytes",
            "tenant",
            "est_fill_ms",
        ],
    )
    heat = pd.DataFrame(
        [[0, 0, 10, 1.0]], columns=["layer", "page_id", "decay_hits", "tenant_weight"]
    )
    tiers = pd.DataFrame(
        [[0, 1 << 30, 0, 1 << 30], [1, 1 << 30, 0, 1 << 30]],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )
    t_caps = pd.DataFrame(
        [["t", 0, 1 << 40], ["t", 1, 1 << 40]], columns=["tenant", "tier", "bandwidth_caps"]
    )  # big caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer", "lat_ms"])  # minimal lat profile
    window = _window_from_frames(req, heat, tiers, t_caps, lats, now_ms)
    result = plan_window(
        window,
        PlannerConfig(
            pmin=0.0,
            umin=-1.0,
            enable_admission=False,
            enable_eviction=False,
        ),
    )
    plan_df, evict_df, admission_df = result.as_dataframes()
    assert evict_df.empty
    assert admission_df.empty
