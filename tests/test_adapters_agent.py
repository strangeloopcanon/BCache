from __future__ import annotations

import secrets
import pandas as pd

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent


def test_segmented_file_backend_rw(tmp_path):
    be = SegmentedFileBackend(str(tmp_path))
    model_id = 'm'
    model_version = 'v'
    layer = 0
    page_bytes = 4096
    # Write two pages
    be.write_page(model_id, model_version, layer, 0, page_bytes, secrets.token_bytes(page_bytes))
    be.write_page(model_id, model_version, layer, 1, page_bytes, secrets.token_bytes(page_bytes))
    # Read coalesced range
    out = be.read_range(model_id, model_version, layer, 0, 1, page_bytes)
    assert len(out) == 2 * page_bytes

    # Read into a preallocated buffer
    buf = bytearray(2 * page_bytes)
    n = be.read_range_into(model_id, model_version, layer, 0, 1, page_bytes, buf)
    assert n == 2 * page_bytes


def test_node_agent_exec(tmp_path):
    be = SegmentedFileBackend(str(tmp_path))
    agent = NodeAgent(be, page_bytes=4096)
    # Seed pages for a small plan (two ops on layer 0)
    for pid in range(4):
        be.write_page('m', 'v', 0, pid, 4096, secrets.token_bytes(4096))
    plan_df = pd.DataFrame([
        ["n0", 0, 1, 0, 0, 0, 1, 4096],
        ["n0", 0, 1, 0, 0, 2, 3, 4096],
    ], columns=["node","tier_src","tier_dst","pcluster","layer","start_pid","end_pid","page_bytes"])
    stats = agent.execute(plan_df, model_id='m', model_version='v')
    assert stats['ops'] == 2
    assert stats['bytes'] == 4 * 4096  # two ranges of two pages each
