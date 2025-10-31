from __future__ import annotations

import argparse
import time

import pandas as pd
import yaml
from bodocache.agent.sim_node import simulate_plan_streams, summarize_metrics
from bodocache.config import load_config
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


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def sweep_params(
    req: pd.DataFrame,
    heat: pd.DataFrame,
    tiers: pd.DataFrame,
    lats: pd.DataFrame,
    min_io_list=(256 * 1024, 512 * 1024, 1024 * 1024),
    credits_list=(16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024),
    pmin_list=(0.0, 0.5, 1.0),
    umin_list=(-1.0, 0.0, 0.5),
):
    results = []
    now_ms = int(time.time() * 1000)
    # Assign clusters
    req = assign_pclusters_minhash(req, num_hashes=32, bands=8, k=4)
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
    layer_lat_entries = [
        LayerLatency(layer=int(row["layer"]), lat_ms=float(row["lat_ms"]))
        for row in lats.to_dict(orient="records")
    ]
    for mio in min_io_list:
        for credits in credits_list:
            for pmin in pmin_list:
                for umin in umin_list:
                    t_caps = (
                        req[["tenant"]]
                        .drop_duplicates()
                        .assign(tier=1, bandwidth_caps=credits)
                        .reset_index(drop=True)
                    )
                    tenant_cap_entries = [
                        TenantCapacity(
                            tenant=str(row["tenant"]),
                            tier=int(row["tier"]),
                            bandwidth_caps=int(row["bandwidth_caps"]),
                        )
                        for row in t_caps.to_dict(orient="records")
                    ]
                    window = PlannerWindow(
                        requests=requests_payload,
                        now_ms=now_ms,
                        heat=heat_entries,
                        tier_caps=tier_cap_entries,
                        tenant_caps=tenant_cap_entries,
                        layer_latencies=layer_lat_entries,
                    )
                    plan_result = plan_window(
                        window,
                        PlannerConfig(
                            pmin=float(pmin),
                            umin=float(umin),
                            min_io_bytes=int(mio),
                            alpha=1.0,
                            beta=0.0,
                            window_ms=20,
                            max_ops_per_tier=64,
                            enable_admission=False,
                            enable_eviction=False,
                        ),
                    )
                    plan, _, _ = plan_result.as_dataframes()
                    exec_df = simulate_plan_streams(
                        plan,
                        tiers,
                        window_ms=20,
                        streams_per_tier=4,
                        use_overlap=True,
                        layer_lat_df=lats,
                    )
                    m = summarize_metrics(exec_df)
                    results.append(
                        {
                            "min_io": int(mio),
                            "credits": int(credits),
                            "pmin": float(pmin),
                            "umin": float(umin),
                            "prefetch_timeliness": m["prefetch_timeliness"],
                            "avg_finish_ms": m["avg_finish_ms"],
                            "avg_io_bytes": m["avg_io_bytes"],
                            "ops": m["ops"],
                        }
                    )
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=False, help="CSV with requests")
    ap.add_argument("--heat", required=False, help="CSV with heat")
    ap.add_argument("--tiers", required=False, help="CSV with tier caps")
    ap.add_argument("--lats", required=False, help="CSV with per-layer latencies")
    ap.add_argument(
        "--write-staged", default="configs/staged.yaml", help="Write best config to this YAML path"
    )
    args = ap.parse_args()

    # Placeholder: generate synthetic if not provided
    from bodocache.sim.utils import (
        synthetic_heat,
        synthetic_layer_lat,
        synthetic_requests,
        synthetic_tier_caps,
    )

    req = load_csv(args.req) if args.req else synthetic_requests()
    heat = load_csv(args.heat) if args.heat else synthetic_heat(req)
    tiers = load_csv(args.tiers) if args.tiers else synthetic_tier_caps()
    lats = load_csv(args.lats) if args.lats else synthetic_layer_lat()

    res = sweep_params(req, heat, tiers, lats)
    best = res.sort_values(["prefetch_timeliness", "avg_io_bytes"], ascending=[False, False]).head(
        5
    )
    print("Top configs:")
    print(best.to_string(index=False))

    print("\nSuggested min_io_bytes / credits:")
    print(best[["min_io", "credits"]].head(1).to_string(index=False))

    # Write staged.yaml with top choice
    top = best.iloc[0]
    cfg = load_config()  # start from defaults/runtime; only override tuned knobs
    cfg["min_io_bytes"] = int(top["min_io"])
    cfg["tenant_credits_bytes"] = int(top["credits"])
    # thresholds tuned too
    cfg.setdefault("thresholds", {})
    cfg["thresholds"]["pmin"] = float(top["pmin"])
    cfg["thresholds"]["umin"] = float(top["umin"])
    with open(args.write_staged, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nWrote staged config to {args.write_staged}")


if __name__ == "__main__":
    main()
