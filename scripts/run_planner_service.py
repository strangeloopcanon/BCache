from __future__ import annotations

import argparse

from bodocache.planner.service_http import serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"Planner HTTP service listening on {args.host}:{args.port}")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()

