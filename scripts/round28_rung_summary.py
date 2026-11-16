#!/usr/bin/env python3
"""Round 28 helper: print the gradeable numbers of one ladder rung.

Round 27's three ladder jobs died before the interpreter started
(``cd: ... Permission denied``), yet the queue surface reported them
``succeeded`` / ``rc = 0``, so the round-27 table showed the rungs as "running"
while nothing had been measured.  This helper exists so every rung prints its
gradeable numbers *into the job log* as well as into its artefact, i.e. a rung
can still be read if the artefact fetch is lost.

It reads ``round25_<closure>_Re<re>_<tag>.{json,npz}`` and recomputes
``x_r/H`` from the stored mean wall shear with the certified operator
(:func:`tensorfvm.verification.wall_shear_bubbles.separation_reattachment`);
nothing is copied from the payload's derived field.

Usage
-----
    PYTHONPATH=src python scripts/round28_rung_summary.py 144x72
    PYTHONPATH=src python scripts/round28_rung_summary.py 144x72 --out-dir docs/case-periodic-hills
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_round11 import _REATTACHMENT_WINDOW
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def summarise(tag: str, out_dir: Path) -> dict:
    stem = f"round25_laminar_Re2800_{tag}"
    js = out_dir / f"{stem}.json"
    npz_path = out_dir / f"{stem}.npz"
    if not js.exists():
        return {"tag": tag, "status": "missing_json", "path": str(js)}
    payload = json.loads(js.read_text())
    out = {
        "tag": tag,
        "status": "ok",
        "wall_clock_s": payload.get("wall_clock_s"),
        "samples": payload.get("samples"),
        "stationarity": payload.get("stationarity"),
        "gate_verdict": (payload.get("gate") or {}).get("verdict"),
    }
    if not npz_path.exists():
        out["status"] = "missing_npz"
        out["npz"] = None
        return out
    with np.load(npz_path) as z:
        tau = np.asarray(z["mean_tau_b"], dtype=float)
        x = np.asarray(z["x_over_h"], dtype=float)
        u = np.asarray(z["mean_u"], dtype=float) if "mean_u" in z else None
    out["field_max_abs_u"] = float(np.max(np.abs(u))) if u is not None else None
    out["field_max_abs_tau"] = float(np.max(np.abs(tau)))
    out["zero_field"] = out["field_max_abs_u"] is not None and out["field_max_abs_u"] < 1e-9
    if out["zero_field"] or not np.all(np.isfinite(tau)):
        out["status"] = "unusable_field"
        return out
    rea = separation_reattachment(tau, x, window=_REATTACHMENT_WINDOW)
    first = separation_reattachment(tau, x, window=_REATTACHMENT_WINDOW,
                                    primary="first") if rea["bubbles"] else rea
    out.update({
        "x_s_over_H": rea["x_s_over_H"],
        "x_r_over_H": rea["x_r_over_H"],
        "x_s_first_over_H": first["x_s_over_H"],
        "x_r_first_over_H": first["x_r_over_H"],
        "bubbles": len(rea["bubbles"]),
        "operator_status": rea["status"],
    })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--out-dir", default=str(CASEDIR))
    args = ap.parse_args(argv)
    print("RUNG_SUMMARY " + json.dumps(summarise(args.tag, Path(args.out_dir)),
                                       sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())