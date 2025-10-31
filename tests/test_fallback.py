from __future__ import annotations

import time

import bodocache.planner.scheduler as sched
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


def _make_small_inputs():
    now_ms = int(time.time() * 1000)
    req = pd.DataFrame(
        [
            [0, "n0", "m", "v", "p1", 0, 0, 1, 0, 1, now_ms + 1000, 256 * 1024, "t", 1],
            [1, "n0", "m", "v", "p2", 0, 2, 2, 0, 1, now_ms + 1005, 256 * 1024, "t", 1],
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
        [[0, 0, 10, 1.0], [0, 2, 5, 1.0]],
        columns=["layer", "page_id", "decay_hits", "tenant_weight"],
    )
    tiers = pd.DataFrame(
        [[0, 1 << 30, 0, 1 << 30], [1, 1 << 30, 0, 1 << 30]],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )
    t_caps = pd.DataFrame(
        [["t", 0, 1 << 40], ["t", 1, 1 << 40]], columns=["tenant", "tier", "bandwidth_caps"]
    )  # large caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer", "lat_ms"])  # minimal lat profile
    return req, heat, tiers, t_caps, lats, now_ms


def test_run_window_fallback_matches_py_core(monkeypatch):
    req, heat, tiers, t_caps, lats, now_ms = _make_small_inputs()

    # Compute reference via pure-Python core
    ref = sched.run_window_core_py(
        req.assign(pcluster=0),
        heat,
        tiers,
        t_caps,
        lats,
        now_ms,
        pmin=0.0,
        umin=-1.0,
        min_io_bytes=0,
        alpha=1.0,
        beta=0.0,
        window_ms=20,
        max_ops_per_tier=64,
        enforce_tier_caps=True,
    )

    # Force fallback by making JIT core raise
    def boom(*a, **k):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(sched, "run_window_core", boom)

    req_with_cluster = req.assign(pcluster=0)
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
            pcluster=int(row["pcluster"]),
        )
        for row in req_with_cluster.to_dict(orient="records")
    ]
    heat_entries = [
        HeatEntry(
            layer=int(row["layer"]),
            page_id=int(row["page_id"]),
            decay_hits=int(row["decay_hits"]),
            tenant_weight=float(row["tenant_weight"]),
            size_bytes=int(DEFAULT_PAGE_BYTES),
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
        for row in t_caps.to_dict(orient="records")
    ]
    lat_entries = [
        LayerLatency(layer=int(row["layer"]), lat_ms=float(row["lat_ms"]))
        for row in lats.to_dict(orient="records")
    ]
    window = PlannerWindow(
        requests=requests_payload,
        now_ms=int(now_ms),
        heat=heat_entries,
        tier_caps=tier_entries,
        tenant_caps=tenant_entries,
        layer_latencies=lat_entries,
    )
    plan_result = plan_window(
        window,
        PlannerConfig(
            pmin=0.0,
            umin=-1.0,
            min_io_bytes=0,
            alpha=1.0,
            beta=0.0,
            window_ms=20,
            max_ops_per_tier=64,
            enforce_tier_caps=True,
        ),
    )
    plan_df, _, _ = plan_result.as_dataframes()
    # Same shape/columns and content equality for this deterministic input
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), plan_df.reset_index(drop=True))
