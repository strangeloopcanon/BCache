from __future__ import annotations

from typing import Iterable, List

# Optional blake3 with fallback for environments without the library
try:  # pragma: no cover - trivial import/fallback
    from blake3 import blake3 as _blake3_ctor  # type: ignore

    def _blake_digest(b: bytes) -> bytes:
        return _blake3_ctor(b).digest()

except Exception:  # pragma: no cover - fallback path
    import hashlib

    def _blake_digest(b: bytes) -> bytes:
        return hashlib.blake2b(b, digest_size=32).digest()


def prefix_id(token_ids: Iterable[int], P: int = 128) -> str:
    """Compute prefix identity hash for the first P tokens."""
    buf = bytearray()
    for t in list(token_ids)[:P]:
        buf += int(t).to_bytes(4, "little", signed=False)
    return _blake_digest(bytes(buf)).hex()


def minhash_bucket(ngrams: List[int], bands: int = 16, rows: int = 4) -> int:
    """Very simple minhash bucket placeholder for near-duplicate prefix grouping."""
    # Not a true minhash; replace with proper LSH in production.
    h = _blake_digest(bytes(ngrams))
    return int.from_bytes(h[:4], "little")
