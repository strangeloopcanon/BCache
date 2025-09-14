from __future__ import annotations

import random
import time
from typing import List

import numpy as np
import pandas as pd


def synthetic_requests(n_req: int = 200, n_layers: int = 8) -> pd.DataFrame:
    rows: List[tuple] = []
    now_ms = int(time.time() * 1000)
    for rid in range(n_req):
        node = f"node-{random.randint(0,3)}"
        model_id = "m70b"
        model_version = "v1"
        base = random.randint(0, 9)
        delta = random.randint(0, 3)
        prefix_id = f"pfx-{base}-{delta}"
        layer = random.randint(0, n_layers - 1)
        length = random.choice([1, 2, 4, 8, 16])
        start = random.randint(0, 1024 - length)
        end = start + length - 1
        tier_src = 0
        tier_dst = 1
        deadline_ms = now_ms + random.randint(5, 60) * 10
        page_bytes = random.choice([128, 256, 512]) * 1024
        tenant = random.choice(["A", "B", "C"])
        est_fill_ms = random.choice([1, 2, 5, 10, 20])
        tok_base = [base] * 64
        tok = tok_base + [delta] * 16
        rows.append(
            (
                rid,
                node,
                model_id,
                model_version,
                prefix_id,
                tok,
                layer,
                start,
                end,
                tier_src,
                tier_dst,
                deadline_ms,
                page_bytes,
                tenant,
                est_fill_ms,
            )
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "req_id",
            "node",
            "model_id",
            "model_version",
            "prefix_id",
            "prefix_tokens",
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


def synthetic_heat(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df[["layer", "page_start"]]
        .rename(columns={"page_start": "page_id"})
        .groupby(["layer", "page_id"])  # type: ignore
        .size()
        .reset_index(name="decay_hits")
    )
    counts["tenant_weight"] = 1.0
    return counts


def synthetic_tier_caps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, 64 * 1024 * 1024, 0, 64 * 1024 * 1024],
            [1, 16 * 1024 * 1024, 0, 16 * 1024 * 1024],
        ],
        columns=["tier", "free_bytes", "inflight_io", "bandwidth_caps"],
    )


def synthetic_tenant_caps(tenants: pd.Series, credits_bytes: int) -> pd.DataFrame:
    uniq = tenants.unique().tolist()
    rows = []
    for t in uniq:
        rows.append([t, 0, credits_bytes])
        rows.append([t, 1, credits_bytes])
    return pd.DataFrame(rows, columns=["tenant", "tier", "bandwidth_caps"])


def synthetic_layer_lat(n_layers: int = 8) -> pd.DataFrame:
    rows = []
    for l in range(n_layers):
        rows.append([l, 5.0 + 0.5 * l])
    return pd.DataFrame(rows, columns=["layer", "lat_ms"])

