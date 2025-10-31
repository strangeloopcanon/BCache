from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    STORAGE = 0
    CPU = 1
    GPU = 2


@dataclass(frozen=True)
class PageKey:
    model_id: str
    model_version: str
    dtype: str
    n_kv_heads: int
    d_head: int
    layer: int
    page_id: int

    def as_tuple(self) -> tuple:
        return (
            self.model_id,
            self.model_version,
            self.dtype,
            self.n_kv_heads,
            self.d_head,
            self.layer,
            self.page_id,
        )

    def encode(self) -> str:
        # Compact string key for hashing and storage adapters
        return (
            f"{self.model_id}:{self.model_version}:{self.dtype}:"
            f"{self.n_kv_heads}:{self.d_head}:{self.layer}:{self.page_id}"
        )


@dataclass(frozen=True)
class Request:
    req_id: int
    node: str
    model_id: str
    model_version: str
    prefix_id: str
    # Inclusive page_id range per layer to fetch
    layer: int
    page_start: int
    page_end: int
    tier_src: Tier
    tier_dst: Tier
    deadline_ms: int
    page_bytes: int
    tenant: str = "default"


@dataclass
class TierState:
    tier: Tier
    free_bytes: int
    inflight_io: int
    bandwidth_caps: int  # bytes per window


@dataclass
class CopyOp:
    page_key: PageKey
    offset: int
    bytes: int
    gpu_id: int | None
    stream_id: int | None
    deadline_ms: int
    src: Tier
    dst: Tier


@dataclass
class Plan:
    ops: list[CopyOp]
    evict_keys: list[PageKey]
    # Optional: admission decisions (target tier)
    admission: list[tuple[PageKey, Tier]]


DEFAULT_PAGE_BYTES = 256 * 1024
