from __future__ import annotations

import os
import time
import secrets

import argparse

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent
from bodocache.integrations.base import KVRequest
from bodocache.integrations.vllm_adapter import VLLMBCacheAdapter
from bodocache.agent.copy_engine import get_copy_engine


def maybe_torch_device_ptr(nbytes: int):
    try:
        import torch
        t = torch.empty((nbytes,), dtype=torch.uint8, device='cuda')
        from bodocache.integrations.ptr import from_torch_tensor
        return from_torch_tensor(t)
    except Exception as e:
        return None


def run_once(page_bytes: int, pages: int, streams: int):
    root = os.environ.get("BCACHE_TMP", ".bcache_tmp")
    be = SegmentedFileBackend(root)
    model_id, model_version, layer = "m", "v", 0
    for pid in range(pages):
        be.write_page(model_id, model_version, layer, pid, page_bytes, secrets.token_bytes(page_bytes))

    eng = get_copy_engine(prefer_native=True)
    agent = NodeAgent(be, page_bytes=page_bytes, copy_engine=eng)
    adapter = VLLMBCacheAdapter(agent, node="n0", model_id=model_id, model_version=model_version, min_io_bytes=0, max_ops_per_tier=streams)
    now_ms = int(time.time() * 1000)

    req = KVRequest(
        req_id="r0",
        node="n0",
        model_id=model_id,
        model_version=model_version,
        prefix_id="p",
        layer=layer,
        page_start=0,
        page_end=pages - 1,
        page_bytes=page_bytes,
        tenant="t",
        est_fill_ms=1.0,
        tier_src=0,
        tier_dst=2,
        deadline_ms=now_ms + 1000,
    )

    dst = maybe_torch_device_ptr(pages * page_bytes)
    ready = []
    t0 = time.time()
    res = adapter.prefetch([req], now_ms=now_ms, dest_resolver=(lambda info: dst) if dst is not None else None, on_ready=lambda info: ready.append(info))
    dt = (time.time() - t0) * 1000.0
    return res.exec_stats.get("bytes", 0), dt, len(ready), dst is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-bytes", type=int, default=1 << 20)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--streams", type=int, default=4)
    args = ap.parse_args()

    bytes_copied, dt_ms, ready_events, used_device = run_once(args.page_bytes, args.pages, args.streams)
    mode = "native" if used_device else "simulation"
    print(f"mode={mode} bytes={bytes_copied} dt_ms={dt_ms:.3f} ready_events={ready_events}")


if __name__ == "__main__":
    main()
