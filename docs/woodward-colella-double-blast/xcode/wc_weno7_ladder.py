"""Produce the WENO7 reference ladder for the Woodward-Colella two-blast case.

Why this file exists
-------------------
``docs/XVAL-FVM-M18737-ROUND2/CERTIFIED-REF-CLOSURE.md`` closed the certified-<1 %
question with an *honest FAIL*: the graded MUSCL/HLLC ``nx=3200`` solution sits
L1 = 1.776 % from the independent WENO5@12800 reference. ``REF-ATTRIBUTION.md``
then had to bound that residual with the *same-family* difference
``WENO5@12800 vs WENO5@6400 = 0.548 %`` — which measures how fast the reference
family is still moving, not how far its limit sits from the graded solution.

``xcode/wc_weno7.py`` (7th-order reconstruction, coefficients derived not typed)
was written for exactly this and its entropy-wave order was measured at 6.0, but
it had **never been run on the blast datum**. This script fills the ladder:

    ladder_profile_nx{N}_weno7.npz    cell averages (x, rho, p, u)
    weno7_metrics_nx{N}.json          steps / seconds / peak / positivity / mass

so that ``docs/XVAL-FVM-M18737-ROUND2/ref_order_r6.py`` can grade the reference
family by *reconstruction order* at fixed resolution instead of only by
refinement. No parameter here is chosen after seeing a result: the rungs and the
frozen predictions live in ``docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-r6.md``.

    python docs/woodward-colella-double-blast/xcode/wc_weno7_ladder.py --rungs 1600,3200,6400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wc_weno7 as w7  # noqa: E402

DEFAULT_DATA = HERE.parent / "ladder" / "data"


def run_rung(nx: int, data_dir: Path, overwrite: bool = False) -> dict:
    """Solve one rung, store profile + metrics, return the metrics dict."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    prof = data_dir / f"ladder_profile_nx{nx}_weno7.npz"
    metr = data_dir / f"weno7_metrics_nx{nx}_weno7.json"
    if prof.exists() and metr.exists() and not overwrite:
        return {"nx": nx, "skipped": True, "metrics": json.loads(metr.read_text())}

    t0 = time.time()
    r = w7.weno7_solve(nx)
    out = {
        "family": "weno7", "nx": int(r["nx"]), "datum": r["datum"],
        "method": r["method"], "cfl": r["cfl"], "t_end": r["t_end"],
        "steps": int(r["steps"]), "seconds": float(r["seconds"]),
        "positivity_fallbacks": int(r["positivity_fallbacks"]),
        "positivity_limiter": bool(r.get("positivity_limiter", True)),
        "pp_limited_stages": int(r.get("pp_limited_stages", 0)),
        "theta_min": float(r.get("theta_min", 1.0)),
        "pp_failures": int(r.get("pp_failures", 0)),
        "first_order_inadmissible": int(r.get("first_order_inadmissible", 0)),
        "admissible_final": bool(r.get("admissible_final", True)),
        "step_status": str(r.get("step_status", "complete")),
        "mass": float(r["mass"]), "mass_drift": float(r["mass_drift"]),
        "rho_max": float(r["rho_max"]), "x_at_rho_max": float(r["x_at_rho_max"]),
        "p_max": float(r["p_max"]),
        "wall_seconds": round(time.time() - t0, 2),
        "finite": bool(np.all(np.isfinite(r["rho"]))),
    }
    if not out["finite"] or out["step_status"] != "complete" or not out["admissible_final"]:
        # A diverged rung must never become a reference: record it as such and store no
        # profile, so that downstream loaders report ``not measured`` instead of NaN.
        out["status"] = ("diverged" if not out["finite"] else "truncated")
        metr.write_text(json.dumps(out, indent=1) + "\n")
        if prof.exists():
            prof.unlink()
        return {"nx": nx, "skipped": False, "diverged": True, "metrics": out}
    out["status"] = "ok"
    np.savez_compressed(
        prof,
        x=np.asarray(r["x"], float), rho=np.asarray(r["rho"], float),
        u=np.asarray(r["u"], float), p=np.asarray(r["p"], float),
    )
    metr.write_text(json.dumps(out, indent=1) + "\n")
    return {"nx": nx, "skipped": False, "metrics": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", default="1600,3200,6400")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    rungs = [int(s) for s in args.rungs.split(",") if s.strip()]
    for nx in rungs:
        res = run_rung(nx, Path(args.data_dir), overwrite=args.overwrite)
        m = res["metrics"]
        print(json.dumps({"nx": nx, "skipped": res["skipped"],
                          "diverged": res.get("diverged", False),
                          "seconds": m.get("seconds"), "steps": m.get("steps"),
                          "peak": m.get("rho_max"),
                          "fallbacks": m.get("positivity_fallbacks"),
                          "finite": m.get("finite")}, sort_keys=True))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())