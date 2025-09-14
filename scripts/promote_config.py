from __future__ import annotations

import argparse
import shutil
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', default='configs/staged.yaml')
    ap.add_argument('--runtime', default='configs/runtime.yaml')
    args = ap.parse_args()
    if not os.path.exists(args.staged):
        print('No staged config found:', args.staged)
        return
    shutil.copy2(args.staged, args.runtime)
    print('Promoted', args.staged, '->', args.runtime)


if __name__ == '__main__':
    main()

