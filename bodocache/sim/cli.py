from __future__ import annotations

import argparse
import time
import secrets

import pandas as pd

from bodocache.planner.scheduler import run_window
from bodocache.planner.cluster import assign_pclusters_minhash
from bodocache.agent.sim_node import simulate_plan_streams, summarize_metrics
from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent
from bodocache.config import load_config_typed
from bodocache.telemetry.logger import TelemetryLogger
from bodocache.sim.utils import (
    synthetic_requests,
    synthetic_heat,
    synthetic_tier_caps,
    synthetic_tenant_caps,
    synthetic_layer_lat,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-ms', type=int, help='Planner window duration in ms')
    ap.add_argument('--min-io', type=int, help='Minimum IO size in bytes for coalesced ops')
    ap.add_argument('--max-ops', type=int, help='Max ops per (node,tier) per window')
    ap.add_argument('--pmin', type=float, help='Popularity threshold')
    ap.add_argument('--umin', type=float, help='Urgency threshold')
    ap.add_argument('--alpha', type=float, help='Popularity weight alpha')
    ap.add_argument('--beta', type=float, help='Popularity weight beta')
    ap.add_argument('--enable-prefix-fanout', dest='enable_prefix_fanout', action='store_true')
    ap.add_argument('--disable-prefix-fanout', dest='enable_prefix_fanout', action='store_false')
    ap.add_argument('--enable-tenant-credits', dest='enable_tenant_credits', action='store_true')
    ap.add_argument('--disable-tenant-credits', dest='enable_tenant_credits', action='store_false')
    ap.add_argument('--enable-admission', dest='enable_admission', action='store_true')
    ap.add_argument('--disable-admission', dest='enable_admission', action='store_false')
    ap.add_argument('--enable-eviction', dest='enable_eviction', action='store_true')
    ap.add_argument('--disable-eviction', dest='enable_eviction', action='store_false')
    ap.add_argument('--enable-overlap', dest='enable_overlap', action='store_true')
    ap.add_argument('--disable-overlap', dest='enable_overlap', action='store_false')
    ap.add_argument('--enforce-tier-caps', dest='enforce_tier_caps', action='store_true')
    ap.add_argument('--no-enforce-tier-caps', dest='enforce_tier_caps', action='store_false')
    ap.set_defaults(enable_prefix_fanout=None, enable_tenant_credits=None,
                    enable_admission=None, enable_eviction=None,
                    enable_overlap=None, enforce_tier_caps=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config_typed()
    # Apply CLI overrides
    if args.window_ms is not None:
        cfg.window_ms = int(args.window_ms)
    if args.min_io is not None:
        cfg.min_io_bytes = int(args.min_io)
    if args.max_ops is not None:
        cfg.max_ops_per_tier = int(args.max_ops)
    if args.pmin is not None:
        cfg.thresholds.pmin = float(args.pmin)
    if args.umin is not None:
        cfg.thresholds.umin = float(args.umin)
    if args.alpha is not None:
        cfg.popularity.alpha = float(args.alpha)
    if args.beta is not None:
        cfg.popularity.beta = float(args.beta)
    if args.enable_prefix_fanout is not None:
        cfg.ab_flags.enable_prefix_fanout = bool(args.enable_prefix_fanout)
    if args.enable_tenant_credits is not None:
        cfg.ab_flags.enable_tenant_credits = bool(args.enable_tenant_credits)
    if args.enable_admission is not None:
        cfg.ab_flags.enable_admission = bool(args.enable_admission)
    if args.enable_eviction is not None:
        cfg.ab_flags.enable_eviction = bool(args.enable_eviction)
    if args.enable_overlap is not None:
        cfg.ab_flags.enable_overlap = bool(args.enable_overlap)
    if args.enforce_tier_caps is not None:
        cfg.ab_flags.enforce_tier_caps = bool(args.enforce_tier_caps)

    req = synthetic_requests()
    if cfg.ab_flags.enable_prefix_fanout:
        req = assign_pclusters_minhash(req, num_hashes=32, bands=8, k=4)
    else:
        req = req.copy(); req['pcluster'] = req['req_id'].astype(int)
    heat = synthetic_heat(req)
    tiers = synthetic_tier_caps()
    lats = synthetic_layer_lat()
    now_ms = int(time.time() * 1000)
    if cfg.ab_flags.enable_tenant_credits:
        tenant_caps = synthetic_tenant_caps(req['tenant'], cfg.tenant_credits_bytes)
    else:
        tenant_caps = synthetic_tenant_caps(req['tenant'], 1<<62)

    plan_df, evict_df, admission_df = run_window(
        req, heat, tiers, tenant_caps, lats, now_ms,
        pmin=cfg.thresholds.pmin,
        umin=cfg.thresholds.umin,
        min_io_bytes=int(cfg.min_io_bytes),
        alpha=cfg.popularity.alpha,
        beta=cfg.popularity.beta,
        window_ms=int(cfg.window_ms),
        max_ops_per_tier=int(cfg.max_ops_per_tier),
        enable_admission=bool(cfg.ab_flags.enable_admission),
        enable_eviction=bool(cfg.ab_flags.enable_eviction),
        enforce_tier_caps=bool(cfg.ab_flags.enforce_tier_caps),
    )

    if plan_df.empty:
        print("No plan ops produced.")
        return
    avg_io = int(plan_df["bytes"].mean()) if not plan_df.empty else 0
    total_ops = len(plan_df)
    total_bytes = int(plan_df["bytes"].sum())
    print("Plan summary:")
    print(f"  ops={total_ops} avg_io={avg_io/1024:.1f}KB total={total_bytes/1024/1024:.2f}MB")
    if "fanout" in plan_df.columns:
        print(f"  mean_fanout={float(plan_df['fanout'].mean()):.2f} max_fanout={int(plan_df['fanout'].max()) if total_ops else 0}")
    show_cols = [c for c in ["node","tier_src","tier_dst","pcluster","layer","run_id","bytes","deadline_ms","fanout","overlap","priority"] if c in plan_df.columns]
    print(plan_df.head(10)[show_cols].to_string(index=False))

    exec_df = simulate_plan_streams(
        plan_df, tiers, window_ms=int(cfg.window_ms),
        streams_per_tier=4, use_overlap=cfg.ab_flags.enable_overlap, layer_lat_df=lats)
    m = summarize_metrics(exec_df)
    print(f"  prefetch_timeliness={m['prefetch_timeliness']:.2f} avg_finish_ms={m['avg_finish_ms']:.1f} ops={m['ops']} (multistream)")

    seg_root = 'segments'
    be = SegmentedFileBackend(seg_root)
    for r in plan_df.itertuples(index=False):
        for pid in range(int(getattr(r, 'start_pid', 0)), int(getattr(r, 'end_pid', -1)) + 1):
            be.write_page('m70b', 'v1', int(r.layer), pid, int(getattr(r, 'page_bytes', 256*1024)), secrets.token_bytes(int(getattr(r, 'page_bytes', 256*1024))))
    agent = NodeAgent(be)
    stats = agent.execute(plan_df, model_id='m70b', model_version='v1')
    print(f"  node_agent_exec: ops={stats['ops']} bytes={stats['bytes']/1024/1024:.2f}MB duration_ms={stats['duration_ms']:.1f}")
    TelemetryLogger().log_window(req, heat, tiers, lats, plan_df, exec_df, evict_df=evict_df, admission_df=admission_df)


if __name__ == "__main__":
    main()

