from __future__ import annotations

import argparse
import time
import pandas as pd
import numpy as np

from bodocache.planner.scheduler import run_window
from bodocache.planner.cluster import assign_pclusters_minhash
from bodocache.agent.sim_node import simulate_plan_streams, summarize_metrics
from bodocache.config import load_config
import yaml


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def sweep_params(req: pd.DataFrame, heat: pd.DataFrame, tiers: pd.DataFrame, lats: pd.DataFrame,
                 min_io_list=(256*1024, 512*1024, 1024*1024),
                 credits_list=(16*1024*1024, 32*1024*1024, 64*1024*1024),
                 pmin_list=(0.0, 0.5, 1.0),
                 umin_list=(-1.0, 0.0, 0.5)):
    results = []
    now_ms = int(time.time()*1000)
    # Assign clusters
    req = assign_pclusters_minhash(req, num_hashes=32, bands=8, k=4)
    for mio in min_io_list:
        for credits in credits_list:
            for pmin in pmin_list:
                for umin in umin_list:
                    t_caps = (req[['tenant']].drop_duplicates().assign(tier=1, bandwidth_caps=credits)
                              .reset_index(drop=True))
                    plan, _, _ = run_window(
                        req, heat, tiers, t_caps, lats, now_ms,
                        pmin=float(pmin), umin=float(umin), min_io_bytes=int(mio),
                        alpha=1.0, beta=0.0, window_ms=20, max_ops_per_tier=64,
                        enable_admission=False, enable_eviction=False,
                    )
                    exec_df = simulate_plan_streams(plan, tiers, window_ms=20, streams_per_tier=4, use_overlap=True, layer_lat_df=lats)
                    m = summarize_metrics(exec_df)
                    results.append({
                        'min_io': int(mio), 'credits': int(credits), 'pmin': float(pmin), 'umin': float(umin),
                        'prefetch_timeliness': m['prefetch_timeliness'],
                        'avg_finish_ms': m['avg_finish_ms'],
                        'avg_io_bytes': m['avg_io_bytes'],
                        'ops': m['ops'],
                    })
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--req', required=False, help='CSV with requests')
    ap.add_argument('--heat', required=False, help='CSV with heat')
    ap.add_argument('--tiers', required=False, help='CSV with tier caps')
    ap.add_argument('--lats', required=False, help='CSV with per-layer latencies')
    ap.add_argument('--write-staged', default='configs/staged.yaml', help='Write best config to this YAML path')
    args = ap.parse_args()

    # Placeholder: generate synthetic if not provided
    from bodocache.sim.utils import (
        synthetic_requests,
        synthetic_heat,
        synthetic_tier_caps,
        synthetic_layer_lat,
    )
    req = load_csv(args.req) if args.req else synthetic_requests()
    heat = load_csv(args.heat) if args.heat else synthetic_heat(req)
    tiers = load_csv(args.tiers) if args.tiers else synthetic_tier_caps()
    lats = load_csv(args.lats) if args.lats else synthetic_layer_lat()

    res = sweep_params(req, heat, tiers, lats)
    best = res.sort_values(['prefetch_timeliness','avg_io_bytes'], ascending=[False, False]).head(5)
    print('Top configs:')
    print(best.to_string(index=False))

    print('\nSuggested min_io_bytes / credits:')
    print(best[['min_io','credits']].head(1).to_string(index=False))

    # Write staged.yaml with top choice
    top = best.iloc[0]
    cfg = load_config()  # start from defaults/runtime; only override tuned knobs
    cfg['min_io_bytes'] = int(top['min_io'])
    cfg['tenant_credits_bytes'] = int(top['credits'])
    # thresholds tuned too
    cfg.setdefault('thresholds', {})
    cfg['thresholds']['pmin'] = float(top['pmin'])
    cfg['thresholds']['umin'] = float(top['umin'])
    with open(args.write_staged, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nWrote staged config to {args.write_staged}")

if __name__ == '__main__':
    main()
