"""Single-resolution CLI for the ``CASE-FVM-CAVITY-RE1000`` case entry.

Thin command-line wrapper around :mod:`tensorfvm.case_cavity_re1000`; the
campaign (grid sequence + continuation control + report bundle) lives in
``docs/CASE-FVM-CAVITY-RE1000/run_case.py``.

    PYTHONPATH=src python scripts/case_cavity_re1000.py --n 64 --out-dir /tmp/cav
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.case_cavity_re1000 import (  # noqa: E402
    CASE_SLUG, CRITERIA, run_resolution, write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64, help="grid cells per direction (even)")
    ap.add_argument("--out-dir", default="artifacts/CASE-FVM-CAVITY-RE1000")
    ap.add_argument("--reynolds", type=float, default=1000.0)
    ap.add_argument("--tolerance", type=float, default=1e-07)
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--cfl", type=float, default=0.15)
    args = ap.parse_args()

    rec, _ = run_resolution(args.n, tolerance=args.tolerance,
                            max_steps=args.max_steps, cfl=args.cfl,
                            reynolds=args.reynolds)
    out = pathlib.Path(args.out_dir)
    path = out / f"lid_cavity_Re{int(args.reynolds)}_n{args.n}.json"
    write_json(path, rec)
    print(f"wrote {path}")
    print(json.dumps({"case": CASE_SLUG, "n": args.n, "verdict": rec["verdict"],
                      "criteria": {k: v["value"] for k, v in rec["criteria"].items()
                                   if not k.startswith("_")}}, indent=1))
    failed = [c["id"] for c in CRITERIA if not rec["criteria"][c["id"]]["pass"]]
    if failed:
        print("failed criteria:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()