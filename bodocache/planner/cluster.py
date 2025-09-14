from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

# Optional blake3 dependency with safe fallback
try:  # pragma: no cover - trivial import/fallback
    from blake3 import blake3 as _blake3_ctor  # type: ignore

    def _blake_digest(b: bytes) -> bytes:
        return _blake3_ctor(b).digest()

except Exception:  # pragma: no cover - fallback path
    import hashlib

    def _blake_digest(b: bytes) -> bytes:
        # Use blake2b as a deterministic fallback; digest size 32 bytes
        return hashlib.blake2b(b, digest_size=32).digest()


def hash_bucket(s: str, buckets: int = 64) -> int:
    d = _blake_digest(s.encode('utf-8'))
    return int.from_bytes(d[:4], 'little') % max(1, buckets)


def assign_pclusters(df: pd.DataFrame, buckets: int = 64) -> pd.DataFrame:
    """Assign a numeric prefix cluster id (pcluster) to requests based on prefix_id.

    This is a placeholder for near-duplicate (minhash/LSH) clustering. It produces stable
    numeric clusters using a fast hash so the Bodo JIT hot path can group by `pcluster`.
    """
    if 'prefix_id' not in df.columns:
        raise KeyError('prefix_id column required')
    out = df.copy()
    out['pcluster'] = out['prefix_id'].apply(lambda s: hash_bucket(str(s), buckets)).astype(np.int64)
    return out


def _k_shingles(s: str, k: int = 5):
    s = str(s)
    if len(s) <= k:
        return [s]
    return [s[i:i+k] for i in range(len(s) - k + 1)]


def _hash_with_seed(x: str, seed: int) -> int:
    # Mix seed in first 4 bytes for simple families
    h = _blake_digest((str(seed) + '|' + x).encode('utf-8'))
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


def assign_pclusters_minhash(
    df: pd.DataFrame,
    num_hashes: int = 32,
    bands: int = 8,
    k: int = 5,
) -> pd.DataFrame:
    """Assign numeric clusters via simple minhash + banding.

    Prefer token-level minhash over `prefix_tokens` (list of ints). If not present,
    fall back to minhash over string `prefix_id` shingles.

    Returns a copy of df with an added int64 column 'pcluster'.
    """
    if num_hashes % bands != 0:
        raise ValueError('num_hashes must be divisible by bands')
    use_tokens = 'prefix_tokens' in df.columns
    rows: List[int] = []
    r = num_hashes // bands
    seeds = list(range(num_hashes))

    if use_tokens:
        # Token-level: build k-grams from integer token ids
        token_seqs: List[List[int]] = df['prefix_tokens'].tolist()
        for toks in token_seqs:
            toks = list(map(int, toks)) if toks is not None else []
            if len(toks) < k:
                grams = [tuple(toks)] if toks else []
            else:
                grams = [tuple(toks[i:i+k]) for i in range(len(toks) - k + 1)]
            sig: List[int] = []
            for seed in seeds:
                # hash each gram with seed, take min
                vals = []
                for g in grams:
                    # Mix seed and ints into bytes deterministically
                    b = seed.to_bytes(4, 'little') + b''.join(int(x).to_bytes(4, 'little', signed=False) for x in g)
                    h = _blake_digest(b)[:4]
                    vals.append(int.from_bytes(h, 'little') & 0x7FFFFFFF)
                sig.append(min(vals) if vals else 0)
            band_keys: List[int] = []
            for bidx in range(bands):
                start = bidx * r
                end = start + r
                chunk = tuple(sig[start:end])
                band_h = _blake_digest(str(chunk).encode('utf-8'))[:4]
                band_keys.append(int.from_bytes(band_h, 'little'))
            combo = _blake_digest(b''.join(int(x).to_bytes(4, 'little') for x in band_keys))[:4]
            rows.append(int.from_bytes(combo, 'little'))
    else:
        if 'prefix_id' not in df.columns:
            raise KeyError('prefix_tokens or prefix_id column required')
        for s in df['prefix_id'].astype(str).tolist():
            shingles = _k_shingles(s, k=k)
            sig: List[int] = []
            for seed in seeds:
                vals = [_hash_with_seed(sh, seed) for sh in shingles]
                sig.append(min(vals) if vals else 0)
            band_keys: List[int] = []
            for bidx in range(bands):
                start = bidx * r
                end = start + r
                chunk = tuple(sig[start:end])
                band_h = _blake_digest(str(chunk).encode('utf-8'))[:4]
                band_keys.append(int.from_bytes(band_h, 'little'))
            combo = _blake_digest(b''.join(int(x).to_bytes(4, 'little') for x in band_keys))[:4]
            rows.append(int.from_bytes(combo, 'little'))

    out = df.copy()
    codes, _ = pd.factorize(pd.Series(rows), sort=False)
    out['pcluster'] = pd.Series(codes).astype(np.int64)
    return out
