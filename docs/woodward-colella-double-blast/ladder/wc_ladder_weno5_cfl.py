"""CFL-tagged WENO5 ladder rungs (instrument-cost reduction for the nx=25600 rung).

Why this file exists
-------------------
``docs/XVAL-FVM-M18737-ROUND2/R5-REFERENCE-EXTENSION-CLOSURE.md`` has been stuck in
state **待回填** since round 5: the extended reference rung ``WENO5@25600`` was submitted
as job ``cj3710`` and never came back, so Q1-Q5 (and round 4's P1/P2) are still
``not_measured``.  The reason is cost: at the frozen instrument (CFL 0.4, SSP-RK3) the
25600 rung needs ``steps x cells`` ~ 2.6e9 cell-updates, i.e. ~2.3 h single core on the
reference host -- longer than the per-job wall clock, so a single submit cannot deliver it.

This script produces the same rung with the **time step relaxed to CFL 0.8**, which halves
the step count (step count is the binding cost: ``steps ~ 1/CFL``).  Nothing else about the
instrument changes: same 5th-order reconstruction, same SSP-RK3, same Rusanov flux, same
``t_end``.  Relaxing CFL is only admissible if the resulting *solution* is indistinguishable
from the frozen-CFL solution, so the admissibility test is frozen **before** the extended
rung is computed, in ``docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-r7.md`` (E1: at the
control rung nx=6400 the CFL-0.8 profile must sit within ``eps_ref/10`` in window L1 and
within ``eps_peak`` at the collision peak of the CFL-0.4 profile).

Outputs (identical layout to ``wc_ladder_weno5.py``, CFL-tagged so downstream loaders
never silently mix instruments):

    ladder_profile_nx{N}_weno5_cfl{cfl}.npz    x, rho, u, p (cell averages)
    ladder_metrics_nx{N}_weno5_cfl{cfl}.json   steps / seconds / CFL / peak / fallbacks / mass

    PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder_weno5_cfl.py \
        --sizes 6400 --cfl 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_XCODE = HERE.parent / "xcode"
for _p in (str(HERE), str(_XCODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wc_crosscode import weno5_solve  # noqa: E402

WC_T_END = 0.038  # frozen in xcode/wc_crosscode.py (Woodward & Colella 1984)


def cfl_tag(cfl: float) -> str:
    return f"cfl{float(cfl):g}"


def run_rung(nx: int, cfl: float, outdir: Path, t_end: float = WC_T_END) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"nx{nx}_weno5_{cfl_tag(cfl)}"
    prof = outdir / f"ladder_profile_{tag}.npz"
    metr = outdir / f"ladder_metrics_{tag}.json"

    t0 = time.time()
    r = weno5_solve(nx, t_end=t_end, cfl=cfl)
    x = np.asarray(r["x"], float)
    rho = np.asarray(r["rho"], float)
    order = np.argsort(x)
    out = {
        "nx": int(nx), "dx": 1.0 / nx, "method": r["method"], "datum": r["datum"],
        "cfl": float(cfl), "t_end": float(t_end),
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
        if prof.exists():
            prof.unlink()
        return {"tag": tag, "diverged": True, "metrics": out}

    out["status"] = "ok"
    np.savez_compressed(prof, x=x[order], rho=rho[order],
                        u=np.asarray(r["u"], float)[order],
                        p=np.asarray(r["p"], float)[order])
    metr.write_text(json.dumps(out, indent=1) + "\n")
    return {"tag": tag, "diverged": False, "metrics": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="6400")
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--t-end", type=float, default=WC_T_END)
    ap.add_argument("--outdir", type=Path, default=HERE / "data")
    args = ap.parse_args(argv)

    for s in [s for s in args.sizes.split(",") if s.strip()]:
        res = run_rung(int(s), args.cfl, args.outdir, t_end=args.t_end)
        m = res["metrics"]
        print(json.dumps({"tag": res["tag"], "diverged": res["diverged"],
                          "status": m.get("status"), "cfl": m.get("cfl"),
                          "steps": m.get("steps"), "seconds": m.get("seconds_total"),
                          "peak": m.get("rho_max"),
                          "fallbacks": m.get("positivity_fallbacks")},
                         sort_keys=True))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())