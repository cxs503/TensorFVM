#!/usr/bin/env python3
"""Entry point for CASE-FVM-CHANNEL-KMM180 (turbulent channel vs KMM1987 DNS).

Thin CLI over :func:`tensorfvm.benchmark_channel_kmm1987.run_benchmark`; the
artifact set (benchmark.json / metrics.csv / stations.csv / profiles.csv /
manifest.json / REPORT.md / figures) lands in ``--out``.

    PYTHONPATH=src python scripts/run_case_channel_kmm180.py --out docs/case-fvm-channel-kmm180

Exit code 0 only when the graded closure meets the predeclared criterion.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from tensorfvm.benchmark_channel_kmm1987 import CASE, main as benchmark_main  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180',
                        help='artifact directory (default: docs/case-fvm-channel-kmm180)')
    parser.add_argument('--quick', action='store_true',
                        help='graded closure plus controls at n=128 only')
    parser.add_argument('--re-tau', type=float, default=180.0)
    args = parser.parse_args(argv)
    print(f'[{CASE}] writing artifacts to {args.out}', file=sys.stderr)
    return benchmark_main(['--out', str(args.out), '--re-tau', str(args.re_tau)]
                          + (['--quick'] if args.quick else []))


if __name__ == '__main__':
    raise SystemExit(main())