"""Sod shock tube: TensorFVM HLLC solution vs the published exact solution.

Why this case anchors the OSS inventory wave
--------------------------------------------
The inventory (``docs/oss-inventory-fvm/``) proposes eight capability gaps for
TensorFVM.  Before any of them is scheduled, the *baseline* the gaps are
measured against has to be trustworthy.  The compressible path is that baseline
for two of the P0 gaps (implicit dual time, low-Mach preconditioning), so this
script re-verifies it against a reference that does **not** come from the
repository:

* an exact Riemann solver implemented here in NumPy (independent of
  ``tensorfvm.compressible.riemann.ExactRiemann``), and
* the published star-state values of Sod (1978), as tabulated in Toro (2009).

The script measures four plateau quantities (``p*``, ``u*``, ``rho*_L``,
``rho*_R``) plus the four wave positions (rarefaction head/tail, contact,
shock) and reports ``ref vs got`` with relative errors and a <3 % verdict.

Usage::

    python scripts/oss_inventory_fvm/sod_literature_benchmark.py \
        --out docs/oss-inventory-fvm/artifacts

Artifacts written: ``sod_literature_benchmark.json`` and two PNG figures.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GAMMA = 1.4
T_END = 0.2

# Sod (1978) / Toro (2009) reference values for this exact initial state.
PUBLISHED = {
    "p_star": 0.30313,
    "u_star": 0.92745,
    "rho_star_left": 0.42632,
    "rho_star_right": 0.26557,
    "shock_speed": 1.75216,
    "contact_speed": 0.92745,
    "rarefaction_head_speed": -1.18322,
    "rarefaction_tail_speed": -0.07042,
}
PUBLISHED_SOURCE = (
    "Sod, G.A. (1978) 'A survey of several finite difference methods for systems "
    "of nonlinear hyperbolic conservation laws', J. Comput. Phys. 27, 1-31; "
    "star-state values as tabulated in Toro, E.F. (2009) 'Riemann Solvers and "
    "Numerical Methods for Fluid Dynamics', 3rd ed., Ch. 4 (standard Sod test)."
)

LEFT = {"rho": 1.0, "u": 0.0, "p": 1.0}
RIGHT = {"rho": 0.125, "u": 0.0, "p": 0.1}


# ---------------------------------------------------------------------------
# independent exact Riemann solver (do not import the repository's solver)
# ---------------------------------------------------------------------------
def _f(p, rho_k, p_k, c_k):
    """Pressure function of the left/right wave (Toro eq. 4.6 / 4.7)."""
    if p > p_k:                                     # shock
        a = 2.0 / ((GAMMA + 1.0) * rho_k)
        b = (GAMMA - 1.0) / (GAMMA + 1.0) * p_k
        return (p - p_k) * math.sqrt(a / (p + b))
    return 2.0 * c_k / (GAMMA - 1.0) * ((p / p_k) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)


def exact_star_state() -> dict:
    """Solve f_L(p*) + f_R(p*) = 0 by bisection (independent implementation)."""
    c_l = math.sqrt(GAMMA * LEFT["p"] / LEFT["rho"])
    c_r = math.sqrt(GAMMA * RIGHT["p"] / RIGHT["rho"])
    lo, hi = 1e-9, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = _f(mid, LEFT["rho"], LEFT["p"], c_l) + _f(mid, RIGHT["rho"], RIGHT["p"], c_r)
        if f > 0.0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    fl = _f(p, LEFT["rho"], LEFT["p"], c_l)
    fr = _f(p, RIGHT["rho"], RIGHT["p"], c_r)
    u = 0.5 * (LEFT["u"] + RIGHT["u"]) + 0.5 * (fr - fl)

    def star_density(rho_k, p_k, c_k):
        if p > p_k:                                  # Rankine-Hugoniot
            return rho_k * ((GAMMA + 1.0) * p + (GAMMA - 1.0) * p_k) / (
                (GAMMA - 1.0) * p + (GAMMA + 1.0) * p_k)
        return rho_k * (p / p_k) ** (1.0 / GAMMA)    # isentropic rarefaction

    rho_l = star_density(LEFT["rho"], LEFT["p"], c_l)
    rho_r = star_density(RIGHT["rho"], RIGHT["p"], c_r)
    c_star_l = math.sqrt(GAMMA * p / rho_l)
    c_star_r = math.sqrt(GAMMA * p / rho_r)
    # wave speeds (Toro eq. 4.47 shock, 4.56 rarefaction)
    if p > LEFT["p"]:
        s_head_l = LEFT["u"] - c_l * math.sqrt(
            1.0 + (GAMMA + 1.0) / (2.0 * GAMMA) * (p / LEFT["p"] - 1.0))
        tail_l = s_head_l
    else:
        s_head_l = LEFT["u"] - c_l
        tail_l = u - c_star_l
    if p > RIGHT["p"]:
        s_shock_r = RIGHT["u"] + c_r * math.sqrt(
            1.0 + (GAMMA + 1.0) / (2.0 * GAMMA) * (p / RIGHT["p"] - 1.0))
        head_r = s_shock_r
    else:
        s_shock_r = RIGHT["u"] + c_r
        head_r = u + c_star_r
    return {
        "p_star": p, "u_star": u,
        "rho_star_left": rho_l, "rho_star_right": rho_r,
        "c_star_left": c_star_l, "c_star_right": c_star_r,
        "shock_speed": s_shock_r,
        "contact_speed": u,
        "rarefaction_head_speed": s_head_l,
        "rarefaction_tail_speed": tail_l,
    }


def exact_sample(xi, star: dict):
    """Sample the exact solution at similarity coordinate xi = (x-x0)/t."""
    g = GAMMA
    c_l = math.sqrt(g * LEFT["p"] / LEFT["rho"])
    c_r = math.sqrt(g * RIGHT["p"] / RIGHT["rho"])
    w_l = LEFT["u"] - c_l
    out = {"rho": np.empty_like(xi), "u": np.empty_like(xi), "p": np.empty_like(xi)}
    for i, s in enumerate(xi):
        if s <= star["u_star"]:                       # left of the contact
            if star["p_star"] > LEFT["p"]:            # left shock
                if s >= star["shock_speed"]:
                    rho, u, p = LEFT["rho"], LEFT["u"], LEFT["p"]
                else:
                    rho, u, p = star["rho_star_left"], star["u_star"], star["p_star"]
            elif s <= w_l:                            # undisturbed left state
                rho, u, p = LEFT["rho"], LEFT["u"], LEFT["p"]
            elif s >= star["u_star"] - star["c_star_left"]:   # star left
                rho, u, p = star["rho_star_left"], star["u_star"], star["p_star"]
            else:                                     # inside the fan
                u = 2.0 / (g + 1.0) * (c_l + (g - 1.0) / 2.0 * LEFT["u"] + s)
                c = 2.0 / (g + 1.0) * (c_l + (g - 1.0) / 2.0 * (LEFT["u"] - s))
                rho = LEFT["rho"] * (c / c_l) ** (2.0 / (g - 1.0))
                p = LEFT["p"] * (c / c_l) ** (2.0 * g / (g - 1.0))
        else:                                         # right of the contact
            if star["p_star"] > RIGHT["p"]:           # right shock
                if s <= star["shock_speed"]:
                    rho, u, p = star["rho_star_right"], star["u_star"], star["p_star"]
                else:
                    rho, u, p = RIGHT["rho"], RIGHT["u"], RIGHT["p"]
            elif s >= RIGHT["u"] + c_r:
                rho, u, p = RIGHT["rho"], RIGHT["u"], RIGHT["p"]
            elif s <= star["u_star"] + star["c_star_right"]:
                rho, u, p = star["rho_star_right"], star["u_star"], star["p_star"]
            else:
                u = 2.0 / (g + 1.0) * (-c_r + (g - 1.0) / 2.0 * RIGHT["u"] + s)
                c = 2.0 / (g + 1.0) * (c_r - (g - 1.0) / 2.0 * (RIGHT["u"] - s))
                rho = RIGHT["rho"] * (c / c_r) ** (2.0 / (g - 1.0))
                p = RIGHT["p"] * (c / c_r) ** (2.0 * g / (g - 1.0))
        out["rho"][i], out["u"][i], out["p"][i] = rho, u, p
    return out


# ---------------------------------------------------------------------------
# TensorFVM run
# ---------------------------------------------------------------------------
def run_tensorfvm(nx: int, order: int = 2):
    from tensorfvm.compressible.cases import sod

    solver, meta = sod(nx, order=order)
    t0 = time.time()
    solver.run(meta["t_end"])
    elapsed = time.time() - t0
    prim = solver.primitive()
    x = meta["xc"].numpy()
    return {
        "x": x,
        "rho": prim["rho"][:, 0].numpy(),
        "u": prim["u"][:, 0].numpy(),
        "p": prim["p"][:, 0].numpy(),
        "steps": int(solver.nsteps),
        "seconds": elapsed,
        "nx": nx,
        "order": order,
    }


def plateau_means(run: dict, star: dict, t: float, x0: float = 0.5) -> dict:
    """Average the numerical solution over the interior of each constant region."""
    x = run["x"]
    pos = {k: x0 + star[k] * t for k in
           ("rarefaction_tail_speed", "contact_speed", "shock_speed")}
    left_window = (pos["rarefaction_tail_speed"] + 0.03, pos["contact_speed"] - 0.03)
    right_window = (pos["contact_speed"] + 0.03, pos["shock_speed"] - 0.03)
    left = (x > left_window[0]) & (x < left_window[1])
    right = (x > right_window[0]) & (x < right_window[1])
    return {
        "p_star": float(np.concatenate([run["p"][left], run["p"][right]]).mean()),
        "u_star": float(np.concatenate([run["u"][left], run["u"][right]]).mean()),
        "rho_star_left": float(run["rho"][left].mean()),
        "rho_star_right": float(run["rho"][right].mean()),
        "cells_left": int(left.sum()),
        "cells_right": int(right.sum()),
        "contact_x": pos["contact_speed"],
        "shock_x": pos["shock_speed"],
        "tail_x": pos["rarefaction_tail_speed"],
    }


def measure_wave_positions(run: dict, star: dict, t: float, x0: float = 0.5) -> dict:
    """Locate the waves in the numerical profile by half-jump crossings."""
    x, rho, p = run["x"], run["rho"], run["p"]
    dx = float(x[1] - x[0])
    # shock: first crossing of the half jump between rho*_R and rho_R (right state)
    target = 0.5 * (star["rho_star_right"] + RIGHT["rho"])
    idx = np.where((rho[:-1] >= target) & (rho[1:] < target))[0]
    shock_x = float(x[idx[-1]]) if idx.size else float("nan")
    # contact: half jump between rho*_L and rho*_R, searched right of the tail
    target = 0.5 * (star["rho_star_left"] + star["rho_star_right"])
    tail_x = x0 + star["rarefaction_tail_speed"] * t
    idx = np.where((x[:-1] > tail_x) & (rho[:-1] >= target) & (rho[1:] < target))[0]
    contact_x = float(x[idx[0]]) if idx.size else float("nan")
    # rarefaction fan edges: u leaves 0 at the head and reaches u* at the tail;
    # the fan is continuous, so use 2 %/98 % thresholds and accept the smearing
    head_idx = np.where(run["u"] >= 0.02 * star["u_star"])[0]
    head_meas = float(x[head_idx[0]]) if head_idx.size else float("nan")
    tail_idx = np.where(run["u"] >= 0.98 * star["u_star"])[0]
    tail_meas = float(x[tail_idx[0]]) if tail_idx.size else float("nan")
    return {"shock_x": shock_x, "contact_x": contact_x,
            "head_x": head_meas, "tail_x": tail_meas, "dx": dx}


def make_figures(run: dict, star: dict, outdir: Path, t: float = T_END) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xi = (run["x"] - 0.5) / t
    exact = exact_sample(xi, star)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, key, label in zip(axes, ("rho", "u", "p"),
                              (r"$\rho$", r"$u$", r"$p$")):
        ax.plot(run["x"], run[key], "o", ms=2.0, label=f"TensorFVM HLLC (nx={run['nx']})")
        ax.plot(run["x"], exact[key], "-", lw=1.2, label="exact Riemann (independent)")
        ax.set_xlabel("x"); ax.set_ylabel(label)
        ax.set_title(f"{label} at t={t}")
        ax.legend(fontsize=7)
    fig.suptitle("Sod shock tube: TensorFVM vs published exact solution", fontsize=10)
    fig.tight_layout()
    f1 = outdir / "sod_profiles_vs_exact.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    rows = published_comparison_table(run, star, t)
    labels = [r["quantity"] for r in rows]
    errs = [r["rel_err_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(range(len(labels)), errs, color="#3b6ea5")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.0, label="3 % acceptance line")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("relative error vs published [%]")
    ax.set_title("Sod reference quantities: solver vs Sod (1978)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = outdir / "sod_published_errors.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    return [f1, f2]


def published_comparison_table(run: dict, star: dict, t: float = T_END) -> list[dict]:
    got = plateau_means(run, star, t)
    waves = measure_wave_positions(run, star, t)
    rows = []
    for key in ("p_star", "u_star", "rho_star_left", "rho_star_right"):
        ref = PUBLISHED[key]
        val = got[key]
        rows.append({"quantity": key, "ref": ref, "got": val,
                     "rel_err_pct": abs(val - ref) / abs(ref) * 100.0,
                     "kind": "plateau", "criterion_pct": 3.0})
    for key, meas, speed_key in (
            ("rarefaction_head_x", waves["head_x"], "rarefaction_head_speed"),
            ("rarefaction_tail_x", waves["tail_x"], "rarefaction_tail_speed"),
            ("contact_x", waves["contact_x"], "contact_speed"),
            ("shock_x", waves["shock_x"], "shock_speed")):
        ref = 0.5 + PUBLISHED[speed_key] * t
        rows.append({"quantity": key, "ref": ref, "got": meas,
                     "rel_err_pct": abs(meas - ref) / 0.5 * 100.0,
                     "kind": "position (rel. to unit domain)", "criterion_pct": 1.0,
                     "criterion_cells": 4.0 if "rarefaction" in key else 2.0,
                     "err_cells": abs(meas - ref) / waves["dx"]})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                                             / "docs" / "oss-inventory-fvm" / "artifacts"))
    parser.add_argument("--nx", type=int, default=400)
    args = parser.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    star = exact_star_state()
    self_check = {k: {"computed": star[k], "published": PUBLISHED[k],
                      "rel_err_pct": abs(star[k] - PUBLISHED[k]) / abs(PUBLISHED[k]) * 100.0}
                  for k in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                            "shock_speed", "contact_speed")}

    # grid family: the coarsest two sizes are cheap and pin the convergence order
    runs = [run_tensorfvm(nx, order=2) for nx in (100, 200, args.nx)]
    finest = runs[-1]
    rows = published_comparison_table(finest, star)
    convergence = []
    for run in runs:
        xi = (run["x"] - 0.5) / T_END
        ex = exact_sample(xi, star)
        convergence.append({
            "nx": run["nx"],
            "l1_rho_vs_independent_exact": float(np.abs(run["rho"] - ex["rho"]).mean()),
            "l1_u_vs_independent_exact": float(np.abs(run["u"] - ex["u"]).mean()),
            "l1_p_vs_independent_exact": float(np.abs(run["p"] - ex["p"]).mean()),
            "steps": run["steps"],
            "seconds": round(run["seconds"], 2),
        })
    orders = [
        float(math.log(convergence[i]["l1_rho_vs_independent_exact"]
                       / convergence[i + 1]["l1_rho_vs_independent_exact"])
              / math.log(convergence[i + 1]["nx"] / convergence[i]["nx"]))
        for i in range(len(convergence) - 1)
    ]

    max_err = max(r["rel_err_pct"] for r in rows if r["kind"] == "plateau")
    pos_rows = [r for r in rows if r["kind"] != "plateau"]
    max_pos_cells = max((r["err_cells"] for r in pos_rows), default=float("nan"))
    verdict = {
        "plateau_max_rel_err_pct": max_err,
        "plateau_all_within_3pct": bool(max_err < 3.0),
        "max_position_error_cells": max_pos_cells,
        "positions_within_tolerance": bool(
            all(r["err_cells"] <= r["criterion_cells"] for r in pos_rows)),
        "observed_order_rho": orders,
        "observed_order_ge_0_7": bool(min(orders) >= 0.7),
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[2]).strip()
    except Exception:
        commit = "unknown"

    artifact = {
        "case": "sod_shock_tube",
        "feature_slug": "FEAT-OSS-INVENTORY-FVM",
        "mission": 14887,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorfvm_commit": commit,
        "setup": {
            "geometry": "1-D, x in [0,1], diaphragm at x0=0.5",
            "initial_state": {"left": LEFT, "right": RIGHT, "gamma": GAMMA},
            "t_end": T_END, "cfl": 0.4, "flux": "HLLC", "reconstruction": "MUSCL-MC",
            "order": 2, "nx_family": [100, 200, args.nx],
        },
        "reference": {
            "type": "published exact solution (independent of the repository)",
            "published_values": PUBLISHED,
            "source": PUBLISHED_SOURCE,
            "independent_solver_self_check": self_check,
            "self_check_note": "the NumPy exact solver in this script is validated "
                               "against the published star state before it is used "
                               "to grade TensorFVM; errors are reported, not hidden",
        },
        "rows": rows,
        "convergence": convergence,
        "verdict": verdict,
        "repro_cmd": "python scripts/oss_inventory_fvm/sod_literature_benchmark.py "
                     "--out docs/oss-inventory-fvm/artifacts",
        "caveats": [
            "plateau means are averaged over the interior of each constant region, "
            "so they measure the scheme's diffusion bias, not pointwise accuracy",
            "positions are located by half-jump crossings; one-cell uncertainty "
            "is inherent and reported as err_cells",
            "this verifies the compressible baseline that two proposed gaps "
            "(implicit dual time, low-Mach preconditioning) will be measured "
            "against; it does not validate any of the proposed gaps themselves",
        ],
    }
    figures = make_figures(finest, star, outdir)
    artifact["figures"] = [p.name for p in figures]
    path = outdir / "sod_literature_benchmark.json"
    path.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "verdict": verdict, "artifact": str(path)},
                     indent=1))
    return 0 if (verdict["plateau_all_within_3pct"]
                 and verdict["positions_within_tolerance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())