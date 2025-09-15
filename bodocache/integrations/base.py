from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class KVRequest:
    """A single KV page interval request originating from an engine.

    All fields map directly to columns expected by the planner scheduler.
    """

    req_id: str
    node: str
    model_id: str
    model_version: str
    prefix_id: str
    layer: int
    page_start: int
    page_end: int
    page_bytes: int = 256 * 1024
    tenant: str = "default"
    est_fill_ms: float = 1.0
    tier_src: int = 0  # storage
    tier_dst: int = 2  # gpu
    deadline_ms: int = 0


@dataclass
class PlannerInputs:
    """Planner side inputs for a single planning window."""

    requests: List[KVRequest]
    window_ms: int = 20
    now_ms: int = 0
    # Per-tier capacities: bytes per window and free bytes; indices by tier id
    bandwidth_caps: Optional[dict[int, int]] = None
    free_bytes: Optional[dict[int, int]] = None
    # Per-tenant bandwidth caps (bytes per window) per tier
    tenant_caps: Optional[List[tuple[str, int, int]]] = None  # (tenant, tier, cap)
    # Per-layer latencies (ms)
    layer_lat_ms: Optional[dict[int, float]] = None


def _default_bandwidth_caps(window_ms: int) -> dict[int, int]:
    # Approximate bytes-per-window caps (20ms) for tiers: STORAGE=0, CPU=1, GPU=2
    # GPU: ~25 GB/s -> ~500 MB per 20ms window
    # CPU: ~10 GB/s -> ~200 MB per 20ms window
    # STORAGE: ~3 GB/s -> ~60 MB per 20ms window (coalesced reads)
    return {0: 60 * 1024 * 1024, 1: 200 * 1024 * 1024, 2: 500 * 1024 * 1024}


def build_dataframes(pi: PlannerInputs):
    """Construct the DataFrames required by scheduler.run_window from inputs."""
    if not pi.requests:
        # Empty frames with correct columns
        cols_req = [
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
        ]
        empty = pd.DataFrame(columns=cols_req)
        heat = pd.DataFrame(columns=["layer", "page_id", "decay_hits", "tenant_weight", "size_bytes"])
        caps = pd.DataFrame(columns=["tier", "bandwidth_caps", "free_bytes"])
        tcaps = pd.DataFrame(columns=["tenant", "tier", "bandwidth_caps"])
        ll = pd.DataFrame(columns=["layer", "lat_ms"])
        return empty, heat, caps, tcaps, ll

    # requests_df
    rows = [
        (
            r.req_id,
            r.node,
            r.model_id,
            r.model_version,
            r.prefix_id,
            int(r.layer),
            int(r.page_start),
            int(r.page_end),
            int(r.tier_src),
            int(r.tier_dst),
            int(r.deadline_ms),
            int(r.page_bytes),
            r.tenant,
            float(r.est_fill_ms),
        )
        for r in pi.requests
    ]
    requests_df = pd.DataFrame(
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

    # heat_df: default decay_hits=1, tenant_weight=1.0, size_bytes=page_bytes
    heat_df = requests_df[["layer", "page_start", "page_bytes"]].copy()
    heat_df = heat_df.rename(columns={"page_start": "page_id", "page_bytes": "size_bytes"})
    heat_df["decay_hits"] = np.int64(1)
    heat_df["tenant_weight"] = np.float64(1.0)
    heat_df = heat_df.groupby(["layer", "page_id"], as_index=False).agg(
        decay_hits=("decay_hits", "sum"), tenant_weight=("tenant_weight", "first"), size_bytes=("size_bytes", "max")
    )

    # tier_caps_df
    bw_caps = pi.bandwidth_caps or _default_bandwidth_caps(pi.window_ms)
    free = pi.free_bytes or {0: 1 << 60, 1: 1 << 60, 2: 1 << 60}
    caps_rows = [
        (tier, int(bw_caps.get(tier, 0)), int(free.get(tier, 1 << 60))) for tier in sorted(set([0, 1, 2]).union(bw_caps.keys()).union(free.keys()))
    ]
    tier_caps_df = pd.DataFrame(caps_rows, columns=["tier", "bandwidth_caps", "free_bytes"])

    # tenant_caps_df: if not provided, set very large caps
    if pi.tenant_caps:
        trows = [(t, int(tier), int(cap)) for (t, tier, cap) in pi.tenant_caps]
    else:
        tenants = requests_df["tenant"].unique().tolist()
        trows = [(t, int(tier), int(1 << 60)) for t in tenants for tier in [0, 1, 2]]
    tenant_caps_df = pd.DataFrame(trows, columns=["tenant", "tier", "bandwidth_caps"])

    # layer_lat_df
    if pi.layer_lat_ms:
        lrows = [(int(k), float(v)) for k, v in sorted(pi.layer_lat_ms.items())]
    else:
        layers = requests_df["layer"].unique().tolist()
        lrows = [(int(ly), 1.0) for ly in layers]
    layer_lat_df = pd.DataFrame(lrows, columns=["layer", "lat_ms"])

    return requests_df, heat_df, tier_caps_df, tenant_caps_df, layer_lat_df

