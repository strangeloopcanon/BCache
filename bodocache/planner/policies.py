from __future__ import annotations

from typing import Tuple

import pandas as pd


def selective_write_through(cand_df: pd.DataFrame, heat_df: pd.DataFrame, reuse_threshold: float = 10.0) -> pd.DataFrame:
    """
    Decide which pages to persist (admission to STORAGE) using a simple reuse threshold
    on decay_hits. Returns DataFrame [page_key, tier_dst].
    """
    heat = heat_df[["page_key", "decay_hits"]].drop_duplicates()
    df = cand_df.merge(heat, on="page_key", how="left").fillna({"decay_hits": 0})
    admit = df[df["decay_hits"] >= reuse_threshold][["page_key"]].drop_duplicates()
    if admit.empty:
        return pd.DataFrame(columns=["page_key", "tier_dst"])  # none
    admit["tier_dst"] = 0  # STORAGE
    return admit


def eviction_candidates(heat_df: pd.DataFrame, tier_state_df: pd.DataFrame, target_free_bytes: int = 0) -> pd.DataFrame:
    """
    Placeholder: evict coldest pages until target_free_bytes is achieved.
    Expects heat_df[page_key, decay_hits, size_bytes]. If size unknown, assume 256KB.
    Returns DataFrame [page_key].
    """
    if target_free_bytes <= 0:
        return pd.DataFrame(columns=["page_key"])  # no eviction needed
    df = heat_df.copy()
    if "size_bytes" not in df.columns:
        df["size_bytes"] = 256 * 1024
    df = df.sort_values("decay_hits", ascending=True)
    df["cum"] = df["size_bytes"].cumsum()
    return df[df["cum"] <= target_free_bytes][["page_key"]]

