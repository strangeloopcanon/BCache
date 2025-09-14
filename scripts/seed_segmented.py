from __future__ import annotations

import os
import argparse
import secrets

from bodocache.adapters.segmented_file_backend import SegmentedFileBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='segments', help='segment root directory')
    ap.add_argument('--model_id', default='m70b')
    ap.add_argument('--model_version', default='v1')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--pages', type=int, default=2048)
    ap.add_argument('--page_bytes', type=int, default=256*1024)
    args = ap.parse_args()

    be = SegmentedFileBackend(args.root)
    for l in range(args.layers):
        for p in range(args.pages):
            data = secrets.token_bytes(args.page_bytes)
            be.write_page(args.model_id, args.model_version, l, p, args.page_bytes, data)
    print('Seeded segments under', args.root)

if __name__ == '__main__':
    main()

