#!/usr/bin/env python3
"""Round 31: Re=700 grid-convergence study for CASE-FVM-PERIODIC-HILLS.

Why this round exists
---------------------
Rounds 20/21 probed the *lower-Re sister cases* of the anchor at a single
mesh (96x48).  The Re=700 run is the one regime where a two-dimensional
solution is physically admissible (the flow is laminar, so there is no
spanwise instability for a 2-D mean to miss), yet it still came back
``x_r/H = 8.08`` against the DNS reference ``5.24`` -- a **+54 %** error that
is *larger* than the Re=2800 error.  If 2-D is valid at Re=700, that gap can
only be (a) a mesh artifact, (b) a statistical-window artifact, or (c) a
genuine solver/model-form effect.  No round had ever refined the mesh at
Re=700, so the gap was never decomposed.

This script closes that gap.  It re-scores the stored mean bottom-wall shear
of every Re=700 laminar artefact with the *certified* round-17 operator
(``wall_shear_bubbles.separation_reattachment``), never trusting the payload's
derived ``x_r_over_H``, and reports a grid-convergence ladder against the
Re=700 reference.  It is zero-compute: it only reads ``.npz``/``.json``.

Usage
-----
    PYTHONPATH=src python3 scripts/round31_re700_grid_convergence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorfvm.verification.periodic_hill_lower_re import (  # noqa: E402
    reattachment_reference,
)
from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
# (tag, nx, ny) in refinement order; the 96x48 rung is the round-20 artefact.
RUNGS = (("96x48", 96, 48), ("144x72", 144, 72), ("192x96", 192, 96))
WINDOW = (1.9, 7.1)  # same certified window as round 17/20


def _stem(tag: str) -> str:
    return f"round20_laminar_Re700_{tag}"


def _score(tag: str) -> dict:
    """Score one stored artefact with the certified operator (or report absent)."""
    npz_path = OUTDIR / f"{_stem(tag)}.npz"
    json_path = OUTDIR / f"{_stem(tag)}.json"
    row = {"tag": tag, "present": npz_path.is_file()}
    if not row["present"]:
        row.update(x_r_over_H=None, x_s_over_H=None, status="ABSENT",
                   n_bubbles=None, error=None, below_10=None, below_3=None)
        return row
    with np.load(npz_path) as npz:
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
        x = np.asarray(npz["x_over_h"], dtype=float)
    scored = separation_reattachment(tau, x, window=WINDOW)
    x_r = scored["x_r_over_H"]
    ref = reattachment_reference(700.0)
    ref_val = ref["x_r_over_H"] if ref else None
    err = (abs(x_r - ref_val) / abs(ref_val)
           if (np.isfinite(x_r) and ref_val) else None)
    row.update(
        x_r_over_H=None if not np.isfinite(x_r) else float(x_r),
        x_s_over_H=None if not np.isfinite(scored["x_s_over_H"])
        else float(scored["x_s_over_H"]),
        status=scored["status"],
        n_bubbles=int(scored["n_bubbles"]),
        all_crossings=[float(c["x_over_H"]) for c in scored["all_crossings"]],
        reference_x_r_over_H=ref_val,
        error=None if err is None else float(err),
        below_10=None if err is None else bool(err <= 0.10),
        below_3=None if err is None else bool(err <= 0.03),
    )
    # cross-check against the payload's own derived number (never preferred)
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        row["payload_x_r_over_H"] = (payload.get("time_averaged") or {}) \
            .get("x_r_over_H")
        row["stationarity_ok"] = (payload.get("stationarity", {})
                                  .get("x_r_over_H", {}).get("ok"))
    return row


def main() -> int:
    rows = [_score(tag) for tag, _, _ in RUNGS]
    present = [r for r in rows if r["present"] and r["x_r_over_H"] is not None]
    out = {
        "round": 31,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": ("Re=700 (laminar, 2-D-admissible) grid-convergence ladder: "
                    "decompose the +54 % 96x48 error into mesh vs model-form"),
        "reynolds": 700.0,
        "reference": reattachment_reference(700.0),
        "rows": rows,
        "n_measured": len(present),
    }
    (OUTDIR / "round31_re700_grid_convergence.json").write_text(
        json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({k: r.get(k) for r in rows
                      for k in ("tag", "present", "x_r_over_H", "error",
                                "below_10", "status")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
