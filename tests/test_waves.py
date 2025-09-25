from __future__ import annotations

import time

import pandas as pd

from bodocache.planner.scheduler import run_window
import pytest

from bodocache.planner.waves import (
    TileConfig,
    build_wave_specs,
    validate_wave_spec,
)
from bodocache.sim.utils import (
    synthetic_requests,
    synthetic_heat,
    synthetic_tier_caps,
    synthetic_tenant_caps,
    synthetic_layer_lat,
)


def test_build_wave_specs_basic():
    now_ms = int(time.time() * 1000)
    req = synthetic_requests(n_req=32, n_layers=4)
    heat = synthetic_heat(req)
    tiers = synthetic_tier_caps()
    lats = synthetic_layer_lat(n_layers=4)
    tenant_caps = synthetic_tenant_caps(req['tenant'], 1 << 60)

    plan_df, _, _ = run_window(
        req,
        heat,
        tiers,
        tenant_caps,
        lats,
        now_ms,
        pmin=0.0,
        umin=-1.0,
        min_io_bytes=0,
        window_ms=20,
        max_ops_per_tier=16,
    )

    waves = build_wave_specs(plan_df, req, window_ms=20, dtype="float16")
    assert isinstance(waves, list)
    assert len(waves) >= 1
    w = waves[0]
    # Required keys exist
    for k in [
        "pack_order",
        "tile_order",
        "bm",
        "bn",
        "bk",
        "cluster_shape",
        "tmem_layout",
        "io_extents",
        "swap_window",
    ]:
        assert k in w
    # K granularity (in bytes) satisfies 32B
    bk = int(w["bk"])
    assert (bk * 2) % 32 == 0  # fp16/bf16
    # extents non-empty and valid
    assert len(w["io_extents"]) >= 1
    assert int(w["tmem_layout"]["stage_n"]) >= 1
    for layer, a, b in w["io_extents"]:
        assert isinstance(layer, str)
        assert int(b) >= int(a)
    sb, se = w["swap_window"]
    assert int(sb) < int(se)


def test_validate_wave_spec_whitelist_and_granularity():
    spec = {
        "pack_order": [1, 2, 3],
        "tile_order": [(0, 0), (0, 1)],
        "bm": 128,
        "bn": 128,
        "bk": 64,
        "cluster_shape": (2, 1),
        "tmem_layout": {"columns": 8, "phases": 4, "double_buffer": True, "stage_n": 2},
        "io_extents": [("0", 0, 3)],
        "swap_window": (0, 8),
    }
    validate_wave_spec(spec)

    invalid_k = spec | {"bk": 33}
    with pytest.raises(ValueError):
        validate_wave_spec(invalid_k)

    whitelist = [TileConfig(64, 64, 32, 2, (1, 1))]
    with pytest.raises(ValueError):
        validate_wave_spec(spec, whitelist=whitelist)

    good = spec | {
        "bm": 64,
        "bn": 64,
        "bk": 32,
        "cluster_shape": (1, 1),
        "tmem_layout": {"columns": 8, "phases": 4, "double_buffer": True, "stage_n": 2},
    }
    validate_wave_spec(good, whitelist=whitelist)


def test_validate_wave_spec_swap_window():
    spec = {
        "pack_order": [],
        "tile_order": [(0, 0)],
        "bm": 128,
        "bn": 128,
        "bk": 64,
        "cluster_shape": (2, 1),
        "tmem_layout": {"columns": 8, "phases": 4, "double_buffer": True, "stage_n": 2},
        "io_extents": [("0", 0, 0)],
        "swap_window": (4, 4),
    }
    with pytest.raises(ValueError):
        validate_wave_spec(spec)


def test_build_wave_specs_rejects_non_granular_shape():
    plan_df = pd.DataFrame([
        {
            "node": "n0",
            "tier_dst": 1,
            "layer": 0,
            "start_pid": 0,
            "end_pid": 1,
            "page_bytes": 256 * 1024,
        }
    ])
    with pytest.raises(ValueError):
        build_wave_specs(plan_df, pd.DataFrame(), window_ms=20, shapes=[(64, 64, 33)])
