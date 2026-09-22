"""Tests for the planner review changes:
- configurable admission reuse threshold
- heat signals on PlanOp.pop / EvictionEntry.decay_hits
- run_window no longer mutates caller inputs
"""
from __future__ import annotations

import time

from bodocache.planner import scheduler
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


def _window(decay_hits: int = 5) -> PlannerWindow:
    now_ms = int(time.time() * 1000)
    req = PlannerRequest(
        req_id="r0",
        node="n0",
        model_id="m",
        model_version="v",
        prefix_id="p1",
        layer=0,
        page_start=0,
        page_end=1,
        tier_src=0,
        tier_dst=1,
        deadline_ms=now_ms + 1000,
        page_bytes=300 * 1024,
        tenant="t",
        est_fill_ms=1.0,
    )
    heat = HeatEntry(
        layer=0,
        page_id=0,
        decay_hits=decay_hits,
        tenant_weight=1.0,
        size_bytes=DEFAULT_PAGE_BYTES,
    )
    tiers = [
        TierCapacity(tier=0, bandwidth_caps=1 << 40, free_bytes=1 << 60),
        TierCapacity(tier=1, bandwidth_caps=1 << 40, free_bytes=1 << 60),
        TierCapacity(tier=2, bandwidth_caps=1 << 40, free_bytes=1 << 60),
    ]
    tcaps = [TenantCapacity(tenant="t", tier=t, bandwidth_caps=1 << 60) for t in (0, 1, 2)]
    lats = [LayerLatency(layer=0, lat_ms=5.0)]
    return PlannerWindow(
        requests=[req],
        now_ms=now_ms,
        heat=[heat],
        tier_caps=tiers,
        tenant_caps=tcaps,
        layer_latencies=lats,
    )


def test_admission_reuse_threshold_is_configurable():
    window = _window(decay_hits=5)
    # Default threshold (10.0): page with 5 hits is not admitted.
    result = plan_window(window, PlannerConfig(pmin=0.0, umin=-1.0))
    _, _, admission_df = result.as_dataframes()
    assert admission_df.empty
    # Lowered threshold admits it (persist to storage tier 0).
    result = plan_window(
        window, PlannerConfig(pmin=0.0, umin=-1.0, admission_reuse_threshold=5.0)
    )
    _, _, admission_df = result.as_dataframes()
    assert len(admission_df) == 1
    assert admission_df["tier_dst"].iloc[0] == 0


def test_plan_op_carries_popularity_signal():
    window = _window(decay_hits=10)
    result = plan_window(window, PlannerConfig(pmin=0.0, umin=-1.0))
    assert len(result.plan) >= 1
    op = result.plan[0]
    # alpha=1.0, beta=0.0 -> pop == decay_hits == 10
    assert op.pop == 10.0


def test_eviction_entry_carries_decay_hits():
    window = _window(decay_hits=10)
    # Starve tier 1 of free bytes so the planned op forces eviction.
    window.tier_caps = [
        TierCapacity(tier=0, bandwidth_caps=1 << 40, free_bytes=1 << 60),
        TierCapacity(tier=1, bandwidth_caps=1 << 40, free_bytes=1),
        TierCapacity(tier=2, bandwidth_caps=1 << 40, free_bytes=1 << 60),
    ]
    result = plan_window(
        window, PlannerConfig(pmin=0.0, umin=-1.0, enforce_tier_caps=False)
    )
    assert len(result.evictions) >= 1
    for entry in result.evictions:
        assert entry.decay_hits == 10


def test_run_window_does_not_mutate_inputs():
    window = _window(decay_hits=10)
    req_df, heat_df, tier_caps_df, tenant_caps_df, layer_lat_df = window.to_dataframes()
    before = {
        "req": (list(req_df.columns), len(req_df)),
        "heat": (list(heat_df.columns), len(heat_df)),
    }
    scheduler.run_window(
        req_df,
        heat_df,
        tier_caps_df,
        tenant_caps_df,
        layer_lat_df,
        now_ms=window.now_ms,
        pmin=0.0,
        umin=-1.0,
    )
    assert (list(req_df.columns), len(req_df)) == before["req"]
    assert (list(heat_df.columns), len(heat_df)) == before["heat"]


def test_heat_signal_columns_present_in_dataframes():
    window = _window(decay_hits=10)
    result = plan_window(window, PlannerConfig(pmin=0.0, umin=-1.0))
    plan_df, _, _ = result.as_dataframes()
    assert "pop" in plan_df.columns
    assert plan_df["pop"].notna().all()


def test_empty_plan_still_exposes_pop_column():
    # No request passes the filters -> empty plan, but the schema stays stable.
    window = _window(decay_hits=0)
    result = plan_window(
        window, PlannerConfig(pmin=100.0, umin=10**9, min_io_bytes=10**12)
    )
    plan_df, _, _ = result.as_dataframes()
    assert plan_df.empty


def test_plan_from_payload_without_optional_columns():
    # prefix_tokens / pcluster arrive as NaN when absent from the JSON payload;
    # the service must not blow up on them.
    from bodocache.planner.service_http import plan_from_payload

    now_ms = int(time.time() * 1000)
    payload = {
        "requests": [
            {
                "req_id": "r0",
                "node": "n0",
                "model_id": "m",
                "model_version": "v",
                "prefix_id": "p1",
                "layer": 0,
                "page_start": 0,
                "page_end": 1,
                "tier_src": 0,
                "tier_dst": 1,
                "deadline_ms": now_ms + 1000,
                "page_bytes": 300 * 1024,
                "tenant": "t",
                "est_fill_ms": 1.0,
            }
        ],
        "heat": [
            {
                "layer": 0,
                "page_id": 0,
                "decay_hits": 7,
                "tenant_weight": 1.0,
                "size_bytes": DEFAULT_PAGE_BYTES,
            }
        ],
        "tier_caps": [
            {"tier": 0, "bandwidth_caps": 1 << 40, "free_bytes": 1 << 60},
            {"tier": 1, "bandwidth_caps": 1 << 40, "free_bytes": 1 << 60},
        ],
        "tenant_caps": [
            {"tenant": "t", "tier": 0, "bandwidth_caps": 1 << 60},
            {"tenant": "t", "tier": 1, "bandwidth_caps": 1 << 60},
        ],
        "layer_lat": [{"layer": 0, "lat_ms": 5.0}],
        "now_ms": now_ms,
        "knobs": {"pmin": 0.0, "umin": -1.0, "admission_reuse_threshold": 5.0},
    }
    plan_df, _, admission_df = plan_from_payload(payload)
    assert len(plan_df) == 1
    assert plan_df["pop"].iloc[0] == 7.0
    assert len(admission_df) == 1
