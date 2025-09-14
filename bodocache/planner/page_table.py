from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import PageKey, Tier


@dataclass
class Location:
    tier: Tier
    node: Optional[str] = None
    path: Optional[str] = None  # for file backend
    gpu_id: Optional[int] = None


class PageTable:
    """
    Minimal in-memory page table mapping PageKey -> Location and metadata.
    Provides helpers to get contiguous runs by (layer, page_id).
    """

    def __init__(self):
        self._loc: Dict[str, Location] = {}

    @staticmethod
    def encode_key(k: PageKey) -> str:
        return k.encode()

    def set(self, key: PageKey, location: Location):
        self._loc[self.encode_key(key)] = location

    def get(self, key: PageKey) -> Optional[Location]:
        return self._loc.get(self.encode_key(key))

    def exists(self, key: PageKey) -> bool:
        return self.encode_key(key) in self._loc

    def bulk_get(self, keys: Iterable[PageKey]) -> List[Optional[Location]]:
        return [self.get(k) for k in keys]

    def iter_layer_pages(
        self, model_id: str, model_version: str, layer: int
    ) -> Iterable[Tuple[PageKey, Location]]:
        for encoded, loc in self._loc.items():
            parts = encoded.split(":")
            if len(parts) != 7:
                continue
            mid, mver, dtype, nheads, dhead, lay, pid = parts
            if mid == model_id and mver == model_version and int(lay) == layer:
                yield (
                    PageKey(mid, mver, dtype, int(nheads), int(dhead), int(lay), int(pid)),
                    loc,
                )

    @staticmethod
    def contiguous_runs(page_ids: List[int]) -> List[Tuple[int, int]]:
        if not page_ids:
            return []
        page_ids = sorted(page_ids)
        runs = []
        start = page_ids[0]
        prev = start
        for p in page_ids[1:]:
            if p == prev + 1:
                prev = p
                continue
            runs.append((start, prev))
            start = p
            prev = p
        runs.append((start, prev))
        return runs

