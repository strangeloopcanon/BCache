from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import time

from scripts.microbench_copy import run_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=16)
    ap.add_argument("--streams", type=str, default="2,4,8")
    ap.add_argument("--page-bytes", type=str, default="524288,1048576,2097152")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    streams_list = [int(x) for x in args.streams.split(",")]
    page_bytes_list = [int(x) for x in args.page_bytes.split(",")]

    results = []
    for streams, pb in itertools.product(streams_list, page_bytes_list):
        timings = []
        bytes_vals = []
        ready_counts = []
        used_device = None
        for _ in range(args.repeats):
            b, dt, ready, used = run_once(pb, args.pages, streams)
            timings.append(dt)
            bytes_vals.append(b)
            ready_counts.append(ready)
            used_device = used
        avg_ms = statistics.mean(timings)
        avg_bytes = statistics.mean(bytes_vals)
        results.append({
            "streams": streams,
            "page_bytes": pb,
            "pages": args.pages,
            "avg_ms": avg_ms,
            "avg_bytes": avg_bytes,
            "ready_events": int(statistics.mean(ready_counts)),
            "mode": "native" if used_device else "simulation",
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

