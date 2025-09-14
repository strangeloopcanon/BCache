from __future__ import annotations

import pandas as pd

from bodocache.planner.cluster import assign_pclusters, assign_pclusters_minhash, hash_bucket


def test_hash_bucket_determinism_and_range():
    # Using a fixed string and bucket count should be deterministic
    v1 = hash_bucket("prefix-123", buckets=64)
    v2 = hash_bucket("prefix-123", buckets=64)
    assert v1 == v2
    assert 0 <= v1 < 64


def test_assign_pclusters_string_prefix():
    df = pd.DataFrame({
        'prefix_id': ['a', 'b', 'a', 'c']
    })
    out = assign_pclusters(df, buckets=8)
    assert 'pcluster' in out.columns
    # Same prefix_id should map to same pcluster
    a_codes = out[out['prefix_id']=='a']['pcluster'].unique()
    assert len(a_codes) == 1


def test_assign_pclusters_minhash_tokens_vs_string():
    # Token path present: cluster by token n-grams
    df_tok = pd.DataFrame({
        'prefix_tokens': [[1,1,1,2,2,2], [1,1,1,2,2,2], [3,3,3,3,3,3]]
    })
    out_tok = assign_pclusters_minhash(df_tok, num_hashes=16, bands=4, k=3)
    assert 'pcluster' in out_tok.columns
    # First two rows should likely be the same cluster
    assert out_tok['pcluster'].iloc[0] == out_tok['pcluster'].iloc[1]

    # String fallback path when tokens absent
    df_str = pd.DataFrame({'prefix_id': ['aaaab', 'aaaab', 'zzzzz']})
    out_str = assign_pclusters_minhash(df_str, num_hashes=16, bands=4, k=3)
    assert 'pcluster' in out_str.columns
    assert out_str['pcluster'].iloc[0] == out_str['pcluster'].iloc[1]

