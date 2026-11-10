#!/usr/bin/env python
"""Run the P254W1-N01 two-phase verification chain and write its evidence.

Usage
-----
    python scripts/run_two_phase_chain.py                 # production settings
    python scripts/run_two_phase_chain.py --quick         # fast, for CI smoke
    python scripts/run_two_phase_chain.py --out /tmp/mychain --no-figures

The script writes, into ``--out`` (default ``docs/two_phase_chain``):

* ``provenance.sqlite`` -- the SQL ledger with runs, metrics (value + gate +
  verdict), artifacts and figure hashes;
* ``report.json`` -- the report (numbers fetched from the ledger by SQL) with the
  figures embedded as base64;
* ``REPORT.md`` -- the human-readable version of the same report;
* ``figures/*.png`` -- the four figures.

The verdict table is printed; the exit status is 0 even when a criterion fails
honestly (an honest FAIL is a result, not a crash) -- inspect the table or the
report for the verdicts.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfvm.two_phase.chain import ChainConfig, run_chain     # noqa: E402
from tensorfvm.two_phase.provenance import ProvenanceDB, sha256_file  # noqa: E402
from tensorfvm.two_phase.report import build_report, write_report  # noqa: E402

DOC_PATH = os.path.join("docs", "reports", "P254W1-N01_two_phase_chain.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("docs", "two_phase_chain"))
    parser.add_argument("--quick", action="store_true",
                        help="reduced resolution (smoke test)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-algebraic", action="store_true",
                        help="skip the algebraic-scheme cross-check")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads (1 avoids oversubscription)")
    args = parser.parse_args(argv)

    try:
        import torch
        torch.set_num_threads(max(1, args.threads))
    except Exception:                                  # pragma: no cover
        pass

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "provenance.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)

    cfg = ChainConfig(figure_dir=os.path.join(out_dir, "figures"),
                      make_figures=not args.no_figures,
                      vof_compare_algebraic=not args.no_algebraic)
    if args.quick:
        cfg = ChainConfig(drift_cells=50, vof_nx=60, vof_ny=20, vof_t_star=2.0,
                          vof_max_steps=5000, cavity_cells=100,
                          stefan_cells=200, stefan_t_end=300.0,
                          figure_dir=os.path.join(out_dir, "figures"),
                          make_figures=not args.no_figures,
                          vof_compare_algebraic=not args.no_algebraic)

    print(f"[chain] output directory: {out_dir}")
    run_chain(db_path, cfg)

    db = ProvenanceDB(db_path)
    try:
        for name in sorted(os.listdir(os.path.join(out_dir, "figures"))) \
                if os.path.isdir(os.path.join(out_dir, "figures")) else []:
            db.add_artifact(None, os.path.join(out_dir, "figures", name),
                            "figure")
    finally:
        db.close()

    report = build_report(
        db_path,
        title="TensorFVM two-phase verification chain (drift-flux, PLIC-VOF, "
              "Rayleigh-Plesset, Stefan)",
        mission_id=10862,
        scenario_key="tensorfvm/two_phase_chain",
        tags=["tensorfvm", "two-phase", "vof", "drift-flux", "cavitation",
              "stefan", "verification", "P254W1-N01"],
        figure_dir=os.path.join(out_dir, "figures"))
    json_path, md_path = write_report(report, out_dir)

    db = ProvenanceDB(db_path)
    try:
        for path, kind in ((json_path, "report"), (md_path, "report")):
            db.add_artifact(None, path, kind)
    finally:
        db.close()

    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(md_path, encoding="utf-8") as src, \
            open(DOC_PATH, "w", encoding="utf-8") as dst:
        dst.write(src.read())

    print("\n[verdicts]")
    for cid, value in sorted(report["verdicts"].items()):
        status = "PASS" if value["passed"] else "FAIL"
        print(f"  {cid}: {status}  ({value['description']})")
        for row in value["metrics"]:
            print(f"      {row['segment']:<11} {row['name']:<36} "
                  f"{row['value']!s:<12} {row['comparator']} "
                  f"{row['limit_value']!s:<10} -> "
                  f"{'PASS' if row['passed'] else 'FAIL'}")
    failed = [cid for cid, value in report["verdicts"].items()
              if not value["passed"]]
    print(f"\n[report] {json_path}")
    print(f"[report] {md_path}")
    print(f"[report] {DOC_PATH}")
    print(f"[sha256] report.json {sha256_file(json_path)}")
    print(f"[verdict] {'ALL PASS' if not failed else 'honest FAIL: ' + ', '.join(sorted(failed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())