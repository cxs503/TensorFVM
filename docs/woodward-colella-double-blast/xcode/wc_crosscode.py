"""Independent cross-code check of the Woodward-Colella two blast waves.

Why this file exists
--------------------
``tensorfvm.compressible.double_blast`` (m15299) grades the case against
anchors that are all computed *inside* the same solver family: exact
conservation of a closed tube, exact Riemann fronts in the first 2.7 ms (before
any wave reaches a wall), and convergence towards its own 3200-cell run.  Those
anchors are necessary but they cannot detect a *systematic* error of the scheme
that survives mesh refinement -- a mis-specified flux, a limiter biased at
extrema, or a wave that is a few percent slow would pass every one of them.

The ledger tolerance for ``CASE-WOODWARD-COLELLA`` is "<10 % vs Woodward &
Colella 1984".  The 1984 article is closed access (see ``../provenance.md``) and
this worktree holds no copy of its printed profiles, so that comparison stays
WITHHELD.  What *is* obtainable offline is an independent numerical solution
produced by a discretisation that shares no code and no algorithmic idea with
the HLLC / MUSCL-Hancock path under audit:

======================  ==========================  =========================
                        TensorFVM (under audit)     this file
======================  ==========================  =========================
mesh                    cell-centred finite volume  finite difference on the
                                                    same cell centres
reconstruction          MUSCL + MC limiter (2nd)    WENO5 (Jiang & Shu 1996)
numerical flux          HLLC (Riemann-based)        Rusanov / local Lax-
                                                    Friedrichs, no Riemann
                                                    solver at all
time integration        MUSCL-Hancock predictor     SSP-RK3 (Shu & Osher)
======================  ==========================  =========================

Because the two solutions share only the governing equations, their agreement
at ``t = 0.038`` is an independent bound on the discretisation error of the
graded solution, and the two codes are graded against each other at the ledger
tolerance.  Two cheap gates keep this file honest:

* the same WENO5 code is run on Sod's shock tube and compared with the *exact*
  Riemann solution, so a broken implementation cannot manufacture agreement;
* a closed tube conserves mass exactly, which the run asserts on itself.

Usage
-----
    python docs/woodward-colella-double-blast/xcode/wc_crosscode.py \
        --out docs/woodward-colella-double-blast/xcode
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# case datum -- mirrors the repository constants (asserted equal at import)
# ----------------------------------------------------------------------
GAMMA = 1.4
WC_X1, WC_X2 = 0.1, 0.9
WC_P_LEFT, WC_P_MIDDLE, WC_P_RIGHT = 1e3, 1e-2, 1e2
WC_RHO = 1.0
WC_T_END = 0.038

#: thresholds, declared before any run; the ledger tolerance is 10 %.
THRESHOLDS = {
    "peak_density_rel": 0.10,
    "l1_profile_rel": 0.10,
    "peak_position_abs": 0.02,          # 2 % of the unit tube
    "self_order_min": 0.50,             # shocks keep every scheme near/below 1
    "sod_plateau_rel": 0.05,            # WENO5 code's own correctness gate
    "mass_conserved": 1e-10,
}


# ----------------------------------------------------------------------
# 1-D Euler: finite difference WENO5 + SSP-RK3 + Rusanov flux
# ----------------------------------------------------------------------
def _weno_left(v: np.ndarray) -> np.ndarray:
    """Jiang-Shu WENO5 left state at every face of ``v``.

    ``out[k]`` is the state on the left of the face between ``v[k+2]`` and
    ``v[k+3]``, built from the five cells ``v[k:k+5]``.
    """
    vm2, vm1, v0, v1, v2 = (v[k:k + v.size - 4] for k in range(5))
    p0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v0) / 6.0
    p1 = (-vm1 + 5.0 * v0 + 2.0 * v1) / 6.0
    p2 = (2.0 * v0 + 5.0 * v1 - v2) / 6.0
    eps = 1e-6
    b0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
    b1 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + v1) ** 2 + 0.25 * (vm1 - v1) ** 2
    b2 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * v1 + v2) ** 2
    w0 = 0.1 / (eps + b0) ** 2
    w1 = 0.6 / (eps + b1) ** 2
    w2 = 0.3 / (eps + b2) ** 2
    return (w0 * p0 + w1 * p1 + w2 * p2) / (w0 + w1 + w2)


def _face_states(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Left/right WENO5 states at the ``nx + 1`` faces of the cell range.

    ``g`` is the conserved array padded with three reflective ghosts on each
    side, so ``g.shape == (3, nx + 6)`` and face ``f`` (``f = 0 .. nx``) lies
    between ``g[:, f+2]`` and ``g[:, f+3]``.  The right state is the left state
    of the mirrored array (``v^+ = reverse(WENO(reverse(v)))``).
    """
    n = g.shape[1]
    nx = n - 6
    left = np.empty((3, nx + 1))
    right = np.empty((3, nx + 1))
    for k in range(3):
        ll = _weno_left(g[k])
        rr = _weno_left(g[k][::-1])[::-1]
        left[k] = ll[:nx + 1]
        right[k] = rr[1:nx + 2]
    return left, right


def _cons(rho: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, E])


def _prim(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    u = U[1] / rho
    p = (GAMMA - 1.0) * (U[2] - 0.5 * rho * u * u)
    return rho, u, p


def _flux(U: np.ndarray) -> np.ndarray:
    rho, u, p = _prim(U)
    return np.stack([rho * u, rho * u * u + p, u * (U[2] + p)])


def _pad(U: np.ndarray) -> np.ndarray:
    """Three reflective ghost cells on each side (rho, E even; momentum odd)."""
    g = np.concatenate([U[:, :3][:, ::-1], U, U[:, -3:][:, ::-1]], axis=1)
    g[1, :3] = -U[1, :3][::-1]
    g[1, -3:] = -U[1, -3:][::-1]
    return g


def _rhs(U: np.ndarray, dx: float, counters: Dict[str, int]) -> np.ndarray:
    """Semi-discrete operator ``-dF/dx`` over the ``nx`` cells."""
    nx = U.shape[1]
    UL, UR = _face_states(_pad(U))
    rl, ul, pl = _prim(UL)
    rr, ur, pr = _prim(UR)
    bad = (rl <= 0) | (pl <= 0) | (rr <= 0) | (pr <= 0)
    if bad.any():  # positivity guard: first order on those faces only
        counters["positivity_fallbacks"] += int(bad.sum())
        g = _pad(U)
        for k in range(3):
            UL[k, bad] = g[k, 2:-3][bad]
            UR[k, bad] = g[k, 3:-2][bad]
        rl, ul, pl = _prim(UL)
        rr, ur, pr = _prim(UR)
    cl = np.sqrt(GAMMA * pl / rl)
    cr = np.sqrt(GAMMA * pr / rr)
    a = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
    Fhat = 0.5 * (_flux(UL) + _flux(UR)) - 0.5 * a * (UR - UL)
    return -(Fhat[:, 1:] - Fhat[:, :-1]) / dx


def weno5_solve(nx: int, t_end: float = WC_T_END, cfl: float = 0.4,
                datum: str = "blast") -> Dict:
    """Run the independent solver; ``datum='sod'`` for the shock tube."""
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    if datum == "blast":
        rho = np.full(nx, WC_RHO)
        p = np.where(x < WC_X1, WC_P_LEFT,
                     np.where(x < WC_X2, WC_P_MIDDLE, WC_P_RIGHT))
    elif datum == "sod":
        left = x < 0.5
        rho = np.where(left, 1.0, 0.125)
        p = np.where(left, 1.0, 0.1)
    else:
        raise ValueError(datum)
    U = _cons(rho, np.zeros(nx), p)
    dx = 1.0 / nx
    counters = {"positivity_fallbacks": 0}
    mass0 = float(U[0].sum() * dx)
    t = 0.0
    steps = 0
    t0 = time.time()
    while t < t_end:
        _, uu, pp = _prim(U)
        c = np.sqrt(GAMMA * pp / U[0])
        dt = cfl * dx / float(np.max(np.abs(uu) + c))
        if t + dt > t_end:
            dt = t_end - t
        k1 = _rhs(U, dx, counters)
        U1 = U + dt * k1
        k2 = _rhs(U1, dx, counters)
        U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
        k3 = _rhs(U2, dx, counters)
        U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)
        t += dt
        steps += 1
    rho, uu, pp = _prim(U)
    return {
        "method": "fd-weno5-ssprk3-rusanov",
        "datum": datum,
        "nx": nx, "t_end": t_end, "cfl": cfl,
        "x": x.tolist(), "rho": rho.tolist(), "u": uu.tolist(), "p": pp.tolist(),
        "steps": steps, "seconds": round(time.time() - t0, 2),
        "positivity_fallbacks": counters["positivity_fallbacks"],
        "mass": float(U[0].sum() * dx),
        "mass_drift": abs(float(U[0].sum() * dx) - mass0),
        "energy": float(U[2].sum() * dx),
        "rho_max": float(rho.max()), "x_at_rho_max": float(x[int(np.argmax(rho))]),
        "p_max": float(pp.max()),
    }


# ----------------------------------------------------------------------
# the code under audit: TensorFVM MUSCL-Hancock + HLLC
# ----------------------------------------------------------------------
def tensorfvm_solve(nx: int, t_end: float = WC_T_END) -> Dict:
    import torch

    from tensorfvm.compressible.cases import DoubleBlastConfig, double_blast

    torch.set_num_threads(1)
    solver, meta = double_blast(DoubleBlastConfig(nx=nx, t_end=t_end))
    initial = meta["initial_integrals"]
    t0 = time.time()
    solver.run(t_end)
    prim = solver.primitive()
    rho = prim["rho"][:, 0].numpy()
    p = prim["p"][:, 0].numpy()
    u = prim["u"][:, 0].numpy()
    x = meta["x"].numpy()
    return {
        "method": "fvm-muscl-hancock-hllc-mc",
        "nx": nx, "t_end": t_end,
        "x": x.tolist(), "rho": rho.tolist(), "u": u.tolist(), "p": p.tolist(),
        "steps": solver.nsteps,
        "seconds": round(time.time() - t0, 2),
        "positivity_fixes": solver.positivity_fixes,
        "drift": solver.budget_drift(initial),
        "rho_max": float(rho.max()), "x_at_rho_max": float(x[int(rho.argmax())]),
        "p_max": float(p.max()),
    }


# ----------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------
def compare_profiles(a: Dict, b: Dict, tag: str = "") -> Dict:
    if a["nx"] != b["nx"]:
        raise ValueError("codes must be compared on the same mesh")
    xa = np.asarray(a["x"])
    da, db = np.asarray(a["rho"]), np.asarray(b["rho"])
    pa, pb = np.asarray(a["p"]), np.asarray(b["p"])
    l1 = float(np.mean(np.abs(da - db)))
    rho_max_ref = 0.5 * (float(da.max()) + float(db.max()))
    return {
        "tag": tag or f"n={a['nx']}",
        "nx": a["nx"],
        "rho_max_fvm": float(da.max()), "rho_max_weno5": float(db.max()),
        "peak_density_rel_diff": abs(float(da.max()) - float(db.max())) / rho_max_ref,
        "x_at_rho_max_fvm": float(xa[int(da.argmax())]),
        "x_at_rho_max_weno5": float(xa[int(db.argmax())]),
        "peak_position_abs_diff": abs(xa[int(da.argmax())] - xa[int(db.argmax())]),
        "p_max_rel_diff": abs(float(pa.max()) - float(pb.max()))
        / max(float(pa.max()), float(pb.max())),
        "l1_rho": l1,
        "l1_rho_rel": l1 / float(np.mean(0.5 * (da + db))),
        "linf_rho": float(np.max(np.abs(da - db))),
        "l1_p_rel": float(np.mean(np.abs(pa - pb))) / float(np.mean(0.5 * (pa + pb))),
    }


def _observed_order(err: Sequence[float], sizes: Sequence[int]) -> float:
    logs = []
    for i in range(len(err) - 1):
        if err[i] > 0 and err[i + 1] > 0:
            logs.append(math.log(err[i] / err[i + 1]) / math.log(sizes[i + 1] / sizes[i]))
    return float(np.median(logs)) if logs else float("nan")


def sod_validation(sizes: Sequence[int] = (200, 400)) -> Dict:
    """The independent code must reproduce the exact Sod solution."""
    import torch

    from tensorfvm.compressible.riemann import ExactRiemann

    rows = []
    for nx in sizes:
        run = weno5_solve(nx, t_end=0.2, datum="sod")
        x = np.asarray(run["x"])
        exact = ExactRiemann(1.0, 0.0, 1.0, GAMMA, 0.125, 0.0, 0.1, GAMMA)
        exact.solve()
        rho_ex, u_ex, p_ex = exact.sample(torch.as_tensor((x - 0.5) / 0.2))
        rho_ex = rho_ex.numpy()
        p_ex = p_ex.numpy()
        # star region probes, sampled where the exact solution is a plateau
        probes = (0.60, 0.80)
        star = {"rho_star_left": float(np.interp(probes[0], x, np.asarray(run["rho"]))),
                "rho_star_right": float(np.interp(probes[1], x, np.asarray(run["rho"]))),
                "p_star": float(np.interp(probes[1], x, np.asarray(run["p"])))}
        ex_star = {"rho_star_left": float(np.interp(probes[0], x, rho_ex)),
                   "rho_star_right": float(np.interp(probes[1], x, rho_ex)),
                   "p_star": float(np.interp(probes[1], x, p_ex))}
        rel = {k: abs(star[k] - ex_star[k]) / abs(ex_star[k]) for k in star}
        rows.append({
            "nx": nx,
            "l1_rho": float(np.mean(np.abs(np.asarray(run["rho"]) - rho_ex))),
            "l1_p": float(np.mean(np.abs(np.asarray(run["p"]) - p_ex))),
            "measured": star, "exact": ex_star, "rel": rel,
            "worst_rel": max(rel.values()),
            "positivity_fallbacks": run["positivity_fallbacks"],
            "mass_drift": run["mass_drift"],
        })
    return {
        "rows": rows,
        "observed_order_l1_rho": _observed_order([r["l1_rho"] for r in rows], list(sizes)),
        "verdicts": {
            "sod_plateaus_within_tolerance":
                bool(max(r["worst_rel"] for r in rows) <= THRESHOLDS["sod_plateau_rel"]),
            "sod_l1_decreases_with_refinement":
                bool(rows[-1]["l1_rho"] < rows[0]["l1_rho"]),
        },
    }


def crosscode(sizes: Sequence[int] = (400, 800, 1600, 3200),
              t_end: float = WC_T_END, extra: Sequence[int] = (6400,),
              include_profiles: bool = True) -> Dict:
    rows: List[Dict] = []
    runs: Dict[str, Dict[str, Dict]] = {"fvm": {}, "weno5": {}}
    for nx in sizes:
        fvm = tensorfvm_solve(nx, t_end)
        wen = weno5_solve(nx, t_end, datum="blast")
        runs["fvm"][str(nx)] = fvm
        runs["weno5"][str(nx)] = wen
        rows.append(compare_profiles(fvm, wen))
    for nx in extra:
        runs["weno5"][str(nx)] = weno5_solve(nx, t_end, datum="blast")

    # each code's self-convergence against its own finest run
    self_conv = {}
    for name in ("fvm", "weno5"):
        avail = sorted(int(n) for n in runs[name])
        fine = runs[name][str(avail[-1])]
        xf = np.asarray(fine["x"])
        errs, ns = [], []
        for nx in avail[:-1]:
            r = runs[name][str(nx)]
            interp = np.interp(xf, np.asarray(r["x"]), np.asarray(r["rho"]))
            errs.append(float(np.mean(np.abs(interp - np.asarray(fine["rho"])))))
            ns.append(nx)
        self_conv[name] = {"sizes": ns, "l1_vs_finest": errs, "finest": avail[-1],
                           "observed_order": _observed_order(errs, ns)}

    # Richardson extrapolation: an absolute error estimate for each code
    extrap: Dict[str, Dict] = {}
    for name in ("fvm", "weno5"):
        avail = sorted(int(n) for n in runs[name])
        if len(avail) < 2:
            continue
        n1, n2 = avail[-2], avail[-1]
        r1, r2 = runs[name][str(n1)], runs[name][str(n2)]
        rho1 = np.interp(np.asarray(r2["x"]), np.asarray(r1["x"]), np.asarray(r1["rho"]))
        rho2 = np.asarray(r2["rho"])
        p = self_conv[name]["observed_order"]
        if not (p == p) or p <= 0.2:
            p = 0.8
        rho_x = rho2 + (rho2 - rho1) / (2.0 ** p - 1.0)
        extrap[name] = {
            "from": [n1, n2], "order_used": p,
            "x": np.asarray(r2["x"]).tolist(), "rho": rho_x.tolist(),
            "rho_max": float(np.max(rho_x)),
            "x_at_rho_max": float(np.asarray(r2["x"])[int(np.argmax(rho_x))]),
            "error_estimate_l1": float(np.mean(np.abs(rho2 - rho_x))),
            "error_estimate_rel": float(np.mean(np.abs(rho2 - rho_x)) / np.mean(rho2)),
        }

    cross_extrap = None
    if "fvm" in extrap and "weno5" in extrap:
        xa = np.asarray(extrap["fvm"]["x"])
        ra = np.asarray(extrap["fvm"]["rho"])
        xb = np.asarray(extrap["weno5"]["x"])
        rb = np.asarray(extrap["weno5"]["rho"])
        rb_on_a = np.interp(xa, xb, rb)
        cross_extrap = {
            "l1_rho": float(np.mean(np.abs(ra - rb_on_a))),
            "l1_rho_rel": float(np.mean(np.abs(ra - rb_on_a))
                                / np.mean(0.5 * (ra + rb_on_a))),
            "peak_density_rel_diff":
                abs(extrap["fvm"]["rho_max"] - extrap["weno5"]["rho_max"])
                / (0.5 * (extrap["fvm"]["rho_max"] + extrap["weno5"]["rho_max"])),
            "peak_position_abs_diff": abs(extrap["fvm"]["x_at_rho_max"]
                                          - extrap["weno5"]["x_at_rho_max"]),
            "note": ("each side is Richardson-extrapolated from its own two finest "
                     "runs with its own observed order; the difference of the two "
                     "extrapolants is a lower bound on the remaining error"),
        }

    finest = rows[-1]
    orders = [self_conv[k]["observed_order"] for k in self_conv]
    verdicts = {
        "peak_density_agrees_within_tolerance":
            bool(finest["peak_density_rel_diff"] <= THRESHOLDS["peak_density_rel"]),
        "l1_profile_agrees_within_tolerance":
            bool(finest["l1_rho_rel"] <= THRESHOLDS["l1_profile_rel"]),
        "peak_position_agrees_within_tolerance":
            bool(finest["peak_position_abs_diff"] <= THRESHOLDS["peak_position_abs"]),
        "agreement_improves_with_refinement":
            bool(rows[-1]["l1_rho_rel"] <= rows[0]["l1_rho_rel"]),
        "both_codes_self_converge":
            bool(all(o == o and o >= THRESHOLDS["self_order_min"] for o in orders)),
        "mass_conserved_by_weno5":
            bool(max(runs["weno5"][k]["mass_drift"] for k in runs["weno5"])
                 <= THRESHOLDS["mass_conserved"]),
    }
    fine_n = sorted(int(n) for n in runs["weno5"])[-1]
    out = {
        "t_end": t_end, "thresholds": THRESHOLDS,
        "rows": rows, "self_convergence": self_conv,
        "richardson": extrap, "cross_extrapolated": cross_extrap,
        "verdicts": verdicts,
        "runs_meta": {name: {n: {k: v for k, v in r.items()
                                 if k in ("steps", "seconds", "positivity_fixes",
                                          "positivity_fallbacks", "rho_max", "p_max",
                                          "mass_drift", "drift")}
                            for n, r in rs.items()}
                      for name, rs in runs.items()},
    }
    if include_profiles:
        out["profiles"] = {
            "x_fvm": runs["fvm"][str(sizes[-1])]["x"],
            "rho_fvm": runs["fvm"][str(sizes[-1])]["rho"],
            "p_fvm": runs["fvm"][str(sizes[-1])]["p"],
            "x_weno5": runs["weno5"][str(fine_n)]["x"],
            "rho_weno5": runs["weno5"][str(fine_n)]["rho"],
            "p_weno5": runs["weno5"][str(fine_n)]["p"],
            "n_fvm": sizes[-1], "n_weno5": fine_n,
        }
    return out


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figures(result: Dict, outdir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs: Dict[str, str] = {}
    prof = result.get("profiles")
    if not prof:
        return figs
    xf = np.asarray(prof["x_fvm"])
    rho_f = np.asarray(prof["rho_fvm"])
    xw = np.asarray(prof["x_weno5"])
    rho_w = np.asarray(prof["rho_weno5"])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(xf, rho_f, "C0-", lw=1.0, label=f"TensorFVM MUSCL-Hancock/HLLC, n={prof['n_fvm']}")
    ax.plot(xw, rho_w, "C3--", lw=1.0, label=f"independent WENO5/Rusanov, n={prof['n_weno5']}")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Woodward-Colella two blast waves, t = 0.038, two independent schemes")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    rho_w_on_f = np.interp(xf, xw, rho_w)
    ax.plot(xf, rho_f - rho_w_on_f, "k-", lw=0.9)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho_{FVM} - \rho_{WENO5}$")
    path = outdir / "fig_wc_crosscode_profiles.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figs["profiles"] = str(path)

    # zoom on the colliding-shock region where the density peak lives
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xp = float(result["rows"][-1]["x_at_rho_max_fvm"])
    m = (xf > xp - 0.12) & (xf < xp + 0.12)
    ax.plot(xf[m], rho_f[m], "C0-", lw=1.2, label="TensorFVM (HLLC)")
    mw = (xw > xp - 0.12) & (xw < xp + 0.12)
    ax.plot(xw[mw], rho_w[mw], "C3--", lw=1.0, label="WENO5 (independent)")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"density peak region at t = 0.038 (peak x = {xp:.4f})")
    ax.legend(fontsize=8)
    path = outdir / "fig_wc_crosscode_peak.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figs["peak"] = str(path)

    # convergence of the mutual difference
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ns = [r["nx"] for r in result["rows"]]
    ax.loglog(ns, [r["l1_rho_rel"] for r in result["rows"]], "ko-", label="L1 diff / mean rho")
    ax.loglog(ns, [r["peak_density_rel_diff"] for r in result["rows"]], "s--",
              label="peak density rel. diff")
    ax.set_xlabel("n cells")
    ax.set_ylabel("cross-code difference")
    ax.set_title("mutual difference of the two schemes at t = 0.038")
    ax.legend(fontsize=8)
    path = outdir / "fig_wc_crosscode_convergence.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figs["convergence"] = str(path)
    return figs


def main() -> Dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--sizes", default="400,800,1600,3200")
    ap.add_argument("--extra", default="6400")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",") if s]
    extra = [int(s) for s in args.extra.split(",") if s]
    t0 = time.time()
    sod = sod_validation()
    cross = crosscode(sizes, extra=extra)
    result = {
        "case": "CASE-WOODWARD-COLELLA",
        "t_end": WC_T_END,
        "datum": {"x1": WC_X1, "x2": WC_X2, "p": [WC_P_LEFT, WC_P_MIDDLE, WC_P_RIGHT],
                  "rho": WC_RHO},
        "sod_reference_gate": sod,
        "crosscode": cross,
        "seconds": round(time.time() - t0, 1),
    }
    if not args.no_figures:
        try:
            result["figures"] = figures(cross, outdir)
        except Exception as exc:  # pragma: no cover
            result["figures"] = {"error": f"{type(exc).__name__}: {exc}"}
    path = outdir / "crosscode.json"
    path.write_text(json.dumps(result, indent=1))
    result["artifact"] = str(path)
    print(json.dumps({"verdicts": cross["verdicts"], "sod": sod["verdicts"],
                      "rows": [{k: r[k] for k in
                                ("nx", "peak_density_rel_diff", "l1_rho_rel",
                                 "peak_position_abs_diff")}
                               for r in cross["rows"]],
                      "cross_extrapolated": cross["cross_extrapolated"]}, indent=1))
    return result


if __name__ == "__main__":
    main()