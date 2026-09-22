from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass


def _stable_hash(x: str, i: int, seed: int) -> int:
    """Deterministic 31-bit hash for sketch indexing.

    NOTE: this previously used Python's salted builtin ``hash()``, which made
    identical workloads produce different heat estimates in every process.
    blake2b keeps estimates reproducible across runs and machines.
    """
    digest = hashlib.blake2b(f"{seed}|{i}|{x}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


@dataclass
class CountMinSketch:
    width: int
    depth: int
    seed: int = 1337

    def __post_init__(self):
        self.table = [[0] * self.width for _ in range(self.depth)]

    def _hash(self, x: str, i: int) -> int:
        return _stable_hash(x, i, self.seed) % self.width

    def add(self, x: str, c: int = 1):
        for i in range(self.depth):
            j = self._hash(x, i)
            self.table[i][j] += c

    def query(self, x: str) -> int:
        return min(self.table[i][self._hash(x, i)] for i in range(self.depth))


class SpaceSaving:
    def __init__(self, k: int):
        self.k = k
        self.counters: dict[str, tuple[int, int]] = {}  # key -> (count, err)

    def add(self, x: str, c: int = 1):
        if x in self.counters:
            cnt, err = self.counters[x]
            self.counters[x] = (cnt + c, err)
            return
        if len(self.counters) < self.k:
            self.counters[x] = (c, 0)
            return
        # Replace min
        min_k = min(self.counters, key=lambda k: self.counters[k][0])
        min_cnt, _ = self.counters[min_k]
        del self.counters[min_k]
        self.counters[x] = (min_cnt + c, min_cnt)

    def topk(self) -> Iterable[tuple[str, int, int]]:
        for k, (cnt, err) in self.counters.items():
            yield (k, cnt, err)


class HeatSketch:
    """
    Combined Count-Min + SpaceSaving with exponential decay to approximate hotness.
    Values decay by exp(-lambda * dt) where dt is seconds since last decay.
    """

    def __init__(
        self,
        width: int = 4096,
        depth: int = 4,
        k: int = 4096,
        decay_lambda: float = 0.01,
        seed: int = 1337,
    ):
        self.cms = CountMinSketch(width, depth, seed=seed)
        self.ss = SpaceSaving(k)
        self.decay_lambda = decay_lambda
        self._last_decay_ts = time.time()

    def _decay_factor(self) -> float:
        now = time.time()
        dt = max(0.0, now - self._last_decay_ts)
        self._last_decay_ts = now
        return math.exp(-self.decay_lambda * dt)

    def add(self, key: str, c: int = 1):
        self.cms.add(key, c)
        self.ss.add(key, c)

    def decay(self):
        f = self._decay_factor()
        # Decay SS counters in-place; CMS typically not decayed, we emulate by scaling top-k only
        for k, (cnt, err) in list(self.ss.counters.items()):
            new_cnt = int(cnt * f)
            new_err = int(err * f)
            self.ss.counters[k] = (new_cnt, new_err)

    def estimate(self, key: str) -> int:
        # Use CMS for guaranteed upper bound; intersect with SS if present
        est = self.cms.query(key)
        if key in self.ss.counters:
            est = min(est, self.ss.counters[key][0])
        return est

    def export_heat(self) -> dict[str, int]:
        return {k: cnt for k, cnt, _ in self.ss.topk()}
