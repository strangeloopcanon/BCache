from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend
from bodocache.agent.node_agent import NodeAgent
from bodocache.agent.sim_node import simulate_plan_streams, summarize_metrics
from bodocache.config import load_config_typed
from bodocache.planner.api import (
    HeatEntry,
    LayerLatency,
    PlannerConfig,
    PlannerRequest,
    PlannerWindow,
    TenantCapacity,
    TierCapacity,
    plan_window,
)
from bodocache.planner.cluster import assign_pclusters_minhash
from bodocache.planner.waves import build_wave_specs
from bodocache.sim.utils import (
    synthetic_heat,
    synthetic_layer_lat,
    synthetic_requests,
    synthetic_tenant_caps,
    synthetic_tier_caps,
)
from bodocache.telemetry.logger import TelemetryLogger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-ms", type=int, help="Planner window duration in ms")
    ap.add_argument("--min-io", type=int, help="Minimum IO size in bytes for coalesced ops")
    ap.add_argument("--max-ops", type=int, help="Max ops per (node,tier) per window")
    ap.add_argument("--pmin", type=float, help="Popularity threshold")
    ap.add_argument("--umin", type=float, help="Urgency threshold")
    ap.add_argument("--alpha", type=float, help="Popularity weight alpha")
    ap.add_argument("--beta", type=float, help="Popularity weight beta")
    ap.add_argument("--enable-prefix-fanout", dest="enable_prefix_fanout", action="store_true")
    ap.add_argument("--disable-prefix-fanout", dest="enable_prefix_fanout", action="store_false")
    ap.add_argument("--enable-tenant-credits", dest="enable_tenant_credits", action="store_true")
    ap.add_argument("--disable-tenant-credits", dest="enable_tenant_credits", action="store_false")
    ap.add_argument("--enable-admission", dest="enable_admission", action="store_true")
    ap.add_argument("--disable-admission", dest="enable_admission", action="store_false")
    ap.add_argument("--enable-eviction", dest="enable_eviction", action="store_true")
    ap.add_argument("--disable-eviction", dest="enable_eviction", action="store_false")
    ap.add_argument("--enable-overlap", dest="enable_overlap", action="store_true")
    ap.add_argument("--disable-overlap", dest="enable_overlap", action="store_false")
    ap.add_argument("--enforce-tier-caps", dest="enforce_tier_caps", action="store_true")
    ap.add_argument("--no-enforce-tier-caps", dest="enforce_tier_caps", action="store_false")
    ap.add_argument("--metrics-json", type=str, help="Write run summary JSON to this path")
    ap.set_defaults(
        enable_prefix_fanout=None,
        enable_tenant_credits=None,
        enable_admission=None,
        enable_eviction=None,
        enable_overlap=None,
        enforce_tier_caps=None,
    )
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
        req = req.copy()
        req["pcluster"] = req["req_id"].astype(int)
    heat = synthetic_heat(req)
    tiers = synthetic_tier_caps()
    lats = synthetic_layer_lat()
    now_ms = int(time.time() * 1000)
    if cfg.ab_flags.enable_tenant_credits:
        tenant_caps = synthetic_tenant_caps(req["tenant"], cfg.tenant_credits_bytes)
    else:
        tenant_caps = synthetic_tenant_caps(req["tenant"], 1 << 62)

    requests_payload = [
        PlannerRequest(
            req_id=str(row["req_id"]),
            node=str(row["node"]),
            model_id=str(row["model_id"]),
            model_version=str(row["model_version"]),
            prefix_id=str(row["prefix_id"]),
            layer=int(row["layer"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            tier_src=int(row["tier_src"]),
            tier_dst=int(row["tier_dst"]),
            deadline_ms=int(row["deadline_ms"]),
            page_bytes=int(row["page_bytes"]),
            tenant=str(row["tenant"]),
            est_fill_ms=float(row["est_fill_ms"]),
            prefix_tokens=list(row.get("prefix_tokens", []) or []),
            pcluster=int(row.get("pcluster", -1)),
        )
        for row in req.to_dict(orient="records")
    ]
    heat_entries = [
        HeatEntry(
            layer=int(row["layer"]),
            page_id=int(row["page_id"]),
            decay_hits=int(row["decay_hits"]),
            tenant_weight=float(row["tenant_weight"]),
            size_bytes=int(row["size_bytes"]),
        )
        for row in heat.to_dict(orient="records")
    ]
    tier_cap_entries = [
        TierCapacity(
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
            free_bytes=int(row["free_bytes"]),
        )
        for row in tiers.to_dict(orient="records")
    ]
    tenant_cap_entries = [
        TenantCapacity(
            tenant=str(row["tenant"]),
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
        )
        for row in tenant_caps.to_dict(orient="records")
    ]
    layer_lat_entries = [
        LayerLatency(layer=int(row["layer"]), lat_ms=float(row["lat_ms"]))
        for row in lats.to_dict(orient="records")
    ]
    window = PlannerWindow(
        requests=requests_payload,
        now_ms=now_ms,
        heat=heat_entries,
        tier_caps=tier_cap_entries,
        tenant_caps=tenant_cap_entries,
        layer_latencies=layer_lat_entries,
    )
    planner_cfg = PlannerConfig(
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
    plan_result = plan_window(window, planner_cfg)
    plan_df, evict_df, admission_df = plan_result.as_dataframes()

    if plan_df.empty:
        print("No plan ops produced.")
        return
    avg_io = int(plan_df["bytes"].mean()) if not plan_df.empty else 0
    total_ops = len(plan_df)
    total_bytes = int(plan_df["bytes"].sum())
    print("Plan summary:")
    print(f"  ops={total_ops} avg_io={avg_io/1024:.1f}KB total={total_bytes/1024/1024:.2f}MB")
    if "fanout" in plan_df.columns:
        mean_fanout = float(plan_df["fanout"].mean())
        max_fanout = int(plan_df["fanout"].max()) if total_ops else 0
        print(f"  mean_fanout={mean_fanout:.2f} max_fanout={max_fanout}")
    show_cols = [
        c
        for c in [
            "node",
            "tier_src",
            "tier_dst",
            "pcluster",
            "layer",
            "run_id",
            "bytes",
            "deadline_ms",
            "fanout",
            "overlap",
            "priority",
        ]
        if c in plan_df.columns
    ]
    print(plan_df.head(10)[show_cols].to_string(index=False))

    exec_df = simulate_plan_streams(
        plan_df,
        tiers,
        window_ms=int(cfg.window_ms),
        streams_per_tier=4,
        use_overlap=cfg.ab_flags.enable_overlap,
        layer_lat_df=lats,
    )
    m = summarize_metrics(exec_df)
    prefetch_timeliness = m["prefetch_timeliness"]
    avg_finish = m["avg_finish_ms"]
    print(
        "  prefetch_timeliness="
        f"{prefetch_timeliness:.2f} avg_finish_ms={avg_finish:.1f} "
        f"ops={m['ops']} (multistream)"
    )

    # Build and show WaveSpec for this window (prototype contract)
    waves = build_wave_specs(plan_df, req, window_ms=int(cfg.window_ms), dtype="float16")
    if waves:
        w = waves[0]
        bk_bytes = 2 * int(w["bk"])  # fp16/bf16
        print("WaveSpec:")
        cluster = tuple(int(x) for x in w.get("cluster_shape", (1, 1)))
        stage = w.get("tmem_layout", {}).get("stage_n", "na")
        print(
            "  shape="
            f"{w['bm']}x{w['bn']}x{w['bk']} "
            f"(bk_bytes={bk_bytes}, cluster={cluster}, stage={stage})"
        )
        swap = w.get("swap_window", (None, None))
        print(
            "  tiles="
            f"{len(w.get('tile_order', []))}"
            f" extents={len(w['io_extents'])}"
            f" swap={swap}"
        )

    seg_root = "segments"
    be = SegmentedFileBackend(seg_root)
    for r in plan_df.itertuples(index=False):
        for pid in range(int(getattr(r, "start_pid", 0)), int(getattr(r, "end_pid", -1)) + 1):
            be.write_page(
                "m70b",
                "v1",
                int(r.layer),
                pid,
                int(getattr(r, "page_bytes", 256 * 1024)),
                secrets.token_bytes(int(getattr(r, "page_bytes", 256 * 1024))),
            )
    agent = NodeAgent(be)
    stats = agent.execute(plan_df, model_id="m70b", model_version="v1")
    total_mb = stats.bytes / 1024 / 1024
    print(
        "  node_agent_exec: "
        f"ops={stats.ops} bytes={total_mb:.2f}MB duration_ms={stats.duration_ms:.1f}"
    )
    TelemetryLogger().log_window(
        req, heat, tiers, lats, plan_df, exec_df, evict_df=evict_df, admission_df=admission_df
    )

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp_ms": int(time.time() * 1000),
            "plan": {
                "ops": total_ops,
                "total_bytes": total_bytes,
                "avg_io_bytes": avg_io,
            },
            "execution": stats.as_dict(),
            "streams": m,
        }
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
