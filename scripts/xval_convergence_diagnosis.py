#!/usr/bin/env python3
"""Cross-validate TensorFVM convergence-diagnosis v2 against OpenFOAM-10 rules.

Thin CLI over :func:`tensorfvm.convergence_xval.run_xval`; the package module is
the single implementation so tests and this script cannot drift apart.

    PYTHONPATH=src python scripts/xval_convergence_diagnosis.py \
        --out artifacts/convergence-diagnosis-v2 --max-iter 2000 --figures
"""

from __future__ import annotations

import argparse
import json
import sys

from tensorfvm.convergence_xval import run_xval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/convergence-diagnosis-v2")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    payload = run_xval(out_dir=args.out, max_iter=args.max_iter,
                       figures=args.figures)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["benchmark_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())