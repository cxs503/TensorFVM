"""Generate the missing WENO5@6400 CFL 0.8 reference rung for R15 closure.

Runs the independent WENO5/SSP-RK3/Rusanov solver (zero-shared-discretisation
with the graded MUSCL/HLLC family) at nx=6400, CFL 0.8, and writes the
artifact pair (npz + metrics json) into the ladder data dir, exactly as
``ladder/wc_ladder_weno5_cfl.py`` would.  This is the single missing rung that
licenses the E4 ``eps_ref`` error bar (all reference rungs on one instrument).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "woodward-colella-double-blast"
_XCODE = ROOT / "xcode"
for _p in (str(HERE), str(_XCODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wc_crosscode import weno5_solve  # noqa: E402

NX = 6400
CFL = 0.8
T_END = 0.038
OUTDIR = ROOT / "ladder" / "data"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = f"nx{NX}_weno5_cfl{CFL:g}"
    prof = OUTDIR / f"ladder_profile_{tag}.npz"
    metr = OUTDIR / f"ladder_metrics_{tag}.json"

    t0 = time.time()
    r = weno5_solve(NX, t_end=T_END, cfl=CFL)
    x = np.asarray(r["x"], float)
    rho = np.asarray(r["rho"], float)
    order = np.argsort(x)
    out = {
        "nx": int(NX), "dx": 1.0 / NX, "method": r["method"], "datum": r["datum"],
        "cfl": float(CFL), "t_end": float(T_END),
        "steps": int(r["steps"]), "seconds_solver": float(r["seconds"]),
        "seconds_total": round(time.time() - t0, 2),
        "positivity_fallbacks": int(r["positivity_fallbacks"]),
        "mass": float(r["mass"]), "mass_drift": float(r["mass_drift"]),
        "rho_max": float(rho.max()), "x_at_rho_max": float(x[int(np.argmax(rho))]),
        "p_max": float(r["p_max"]),
        "finite": bool(np.all(np.isfinite(rho))),
    }
    if not out["finite"]:
        out["status"] = "diverged"
        metr.write_text(json.dumps(out, indent=1) + "\n")
        print("DIVERGED", flush=True)
        return
    out["status"] = "ok"
    np.savez(prof, x=x[order], rho=rho[order], u=np.asarray(r["u"], float)[order],
             p=np.asarray(r["p"], float)[order])
    metr.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"tag": tag, "steps": out["steps"],
                      "seconds": out["seconds_solver"], "rho_max": out["rho_max"],
                      "mass_drift": out["mass_drift"], "status": "ok"}), flush=True)


if __name__ == "__main__":
    main()
