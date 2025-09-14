from __future__ import annotations

import time
import pandas as pd

import bodocache.planner.scheduler as sched


def _make_small_inputs():
    now_ms = int(time.time() * 1000)
    req = pd.DataFrame([
        [0, "n0", "m", "v", "p1", 0, 0, 1, 0, 1, now_ms + 1000, 256 * 1024, "t", 1],
        [1, "n0", "m", "v", "p2", 0, 2, 2, 0, 1, now_ms + 1005, 256 * 1024, "t", 1],
    ], columns=[
        "req_id","node","model_id","model_version","prefix_id","layer","page_start","page_end","tier_src","tier_dst","deadline_ms","page_bytes","tenant","est_fill_ms"
    ])
    heat = pd.DataFrame([[0, 0, 10, 1.0],[0,2,5,1.0]], columns=["layer","page_id","decay_hits","tenant_weight"])
    tiers = pd.DataFrame([[0, 1<<30, 0, 1<<30], [1, 1<<30, 0, 1<<30]], columns=["tier","free_bytes","inflight_io","bandwidth_caps"])
    t_caps = pd.DataFrame([["t", 0, 1<<40], ["t", 1, 1<<40]], columns=["tenant","tier","bandwidth_caps"])  # large caps
    lats = pd.DataFrame([[0, 5.0]], columns=["layer","lat_ms"])  # minimal lat profile
    return req, heat, tiers, t_caps, lats, now_ms


def test_run_window_fallback_matches_py_core(monkeypatch):
    req, heat, tiers, t_caps, lats, now_ms = _make_small_inputs()

    # Compute reference via pure-Python core
    ref = sched.run_window_core_py(
        req.assign(pcluster=0), heat, tiers, t_caps, lats,
        now_ms, pmin=0.0, umin=-1.0, min_io_bytes=0, alpha=1.0, beta=0.0, window_ms=20, max_ops_per_tier=64,
        enforce_tier_caps=True,
    )

    # Force fallback by making JIT core raise
    def boom(*a, **k):
        raise RuntimeError("force fallback")
    monkeypatch.setattr(sched, 'run_window_core', boom)

    plan_df, _, _ = sched.run_window(
        req.assign(pcluster=0), heat, tiers, t_caps, lats, now_ms,
        pmin=0.0, umin=-1.0, min_io_bytes=0, alpha=1.0, beta=0.0,
        window_ms=20, max_ops_per_tier=64, enforce_tier_caps=True,
    )
    # Same shape/columns and content equality for this deterministic input
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), plan_df.reset_index(drop=True))
