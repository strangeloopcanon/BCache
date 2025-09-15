from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import yaml

from bodocache.agent.copy_engine import get_copy_engine
from bodocache.agent.node_agent import NodeAgent
from bodocache.integrations.config import KVOverrides
from bodocache.integrations.loader import load_callable
from bodocache.integrations.ptr import from_torch_tensor  # noqa: F401 (hint for users)
from bodocache.telemetry.trace import TraceRecorder


def load_backend(root: str, use_uring: bool):
    if use_uring:
        try:
            from bodocache.adapters.uring_backend import SegmentedUringBackend

            return SegmentedUringBackend(root)
        except Exception as e:
            print(f"io_uring backend not available ({e}); falling back to file backend.")
    from bodocache.adapters.segmented_file_backend import SegmentedFileBackend

    return SegmentedFileBackend(root)


def build_adapter(engine_name: str, agent: NodeAgent, node: str, model_id: str, model_version: str, window_ms: int):
    if engine_name == "vllm":
        from bodocache.integrations.vllm_adapter import VLLMBCacheAdapter

        return VLLMBCacheAdapter(agent, node=node, model_id=model_id, model_version=model_version, window_ms=window_ms, trace=TraceRecorder())
    elif engine_name == "sglang":
        from bodocache.integrations.sglang_adapter import SGLangBCacheAdapter

        return SGLangBCacheAdapter(agent, node=node, model_id=model_id, model_version=model_version, window_ms=window_ms, trace=TraceRecorder())
    else:
        raise ValueError(f"unknown engine: {engine_name}")


def build_integration(engine_name: str, engine: Any, adapter, section: Dict[str, Any]):
    kv = section.get("kv", {})
    deadline_offset_ms = section.get("deadline_offset_ms")
    ov = KVOverrides(
        block_size=kv.get("block_size"),
        num_layers=kv.get("num_layers"),
        num_kv_heads=kv.get("num_kv_heads"),
        head_size=kv.get("head_size"),
        kv_dtype=kv.get("kv_dtype"),
    )
    collect_spec = section.get("collect_blocks")
    dest_spec = section.get("dest_resolver")
    collect_blocks = load_callable(collect_spec) if collect_spec else None
    dest_resolver = load_callable(dest_spec) if dest_spec else None

    if engine_name == "vllm":
        from bodocache.integrations.vllm_integration import VLLMIntegration

        return VLLMIntegration(engine, adapter, collect_blocks=collect_blocks, dest_resolver=dest_resolver, kv_overrides=ov, deadline_offset_ms=deadline_offset_ms)
    else:
        from bodocache.integrations.sglang_integration import SGLangIntegration

        return SGLangIntegration(engine, adapter, collect_blocks=collect_blocks, dest_resolver=dest_resolver, kv_overrides=ov, deadline_offset_ms=deadline_offset_ms)


def main():
    ap = argparse.ArgumentParser(description="BCache integration stub runner")
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--config", type=str, required=True, help="YAML config with overrides and callables")
    ap.add_argument("--segments-root", type=str, default=".bcache_segments")
    ap.add_argument("--node", type=str, default="n0")
    ap.add_argument("--model-id", type=str, default="m")
    ap.add_argument("--model-version", type=str, default="v")
    ap.add_argument("--window-ms", type=int, default=20)
    ap.add_argument("--prefer-native", action="store_true")
    ap.add_argument("--io-uring", action="store_true")
    ap.add_argument("--prefix-id", type=str, default="session")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    section = cfg.get(args.engine, {})
    if not section:
        print(f"Config missing section for engine '{args.engine}'.")
        sys.exit(1)

    backend = load_backend(args.segments_root, args.io_uring)
    eng = get_copy_engine(prefer_native=args.prefer_native) if args.prefer_native else None
    agent = NodeAgent(backend, copy_engine=eng)
    adapter = build_adapter(args.engine, agent, args.node, args.model_id, args.model_version, args.window_ms)
    integration = build_integration(args.engine, engine={}, adapter=adapter, section=section)

    # Run a single prefetch step using provided callables. Users can adapt this to their decode loop.
    now_ms = int(time.time() * 1000)
    try:
        res = integration.prefetch_step(state=None, prefix_id=args.prefix_id, now_ms=now_ms)
    except Exception as e:
        print(f"Prefetch failed: {e}")
        sys.exit(2)
    if res is None:
        print("No collect_blocks callable provided; nothing to prefetch.")
        return
    metrics = res.metrics or {}
    print(f"planned_ops={len(res.plan_df)} bytes={res.exec_stats.get('bytes', 0)} on_time_ratio={metrics.get('on_time_ratio')}\n")


if __name__ == "__main__":
    main()

