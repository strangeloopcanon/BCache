from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .base import KVRequest


_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float8_e5m2": 1,
    "float8_e4m3fn": 1,
}


@dataclass
class VLLMCacheConfig:
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    kv_dtype: str = "float16"

    def bytes_per_block(self) -> int:
        # 2 for K and V tensors
        bpe = _DTYPE_BYTES.get(self.kv_dtype, 2)
        return 2 * self.num_kv_heads * self.block_size * self.head_size * bpe


def coalesce_blocks(block_ids: Sequence[int]) -> List[Tuple[int, int]]:
    if not block_ids:
        return []
    s = sorted(set(int(b) for b in block_ids))
    ranges = []
    a = s[0]
    prev = a
    for b in s[1:]:
        if b == prev + 1:
            prev = b
            continue
        ranges.append((a, prev))
        a = prev = b
    ranges.append((a, prev))
    return ranges


def build_requests_from_blocks(
    cfg: VLLMCacheConfig,
    *,
    node: str,
    model_id: str,
    model_version: str,
    tenant: str,
    prefix_id: str,
    layer_to_blocks: Dict[int, Sequence[int]],
    now_ms: int,
    deadline_offset_ms: int = 20,
) -> List[KVRequest]:
    page_bytes = cfg.bytes_per_block()
    reqs: List[KVRequest] = []
    for layer, blocks in layer_to_blocks.items():
        for start, end in coalesce_blocks(blocks):
            reqs.append(
                KVRequest(
                    req_id=f"{prefix_id}:{layer}:{start}-{end}",
                    node=node,
                    model_id=model_id,
                    model_version=model_version,
                    prefix_id=prefix_id,
                    layer=int(layer),
                    page_start=int(start),
                    page_end=int(end),
                    page_bytes=int(page_bytes),
                    tenant=tenant,
                    est_fill_ms=1.0,
                    tier_src=0,
                    tier_dst=2,
                    deadline_ms=int(now_ms + deadline_offset_ms),
                )
            )
    return reqs

