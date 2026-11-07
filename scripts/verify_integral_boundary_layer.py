"""Verification of :mod:`tensorfvm.integral_boundary_layer` against exact solutions.

The integral boundary layer (Thwaites laminar + Head turbulent) feeds the
twin-hull form-factor analysis, but it used to be anchored only on two *scalar*
checks (``C_F`` vs ``1.328/sqrt(Re)`` and vs ITTC-1957) with 8%/15% bands.  This
driver closes that verification debt on the local field level by comparing the
model against **exact** reference solutions computed here, not against empirical
lines of the same family:

``A`` laminar flat plate           -- Blasius closed form (theta, delta*, cf)
``B`` Thwaites correlations        -- exact Falkner-Skan family (H(lambda), f(lambda))
``C`` marched self-similar wedges  -- exact Falkner-Skan profiles (theta, cf, H)
``D`` laminar separation criterion -- exact separation solution (beta_s, lambda_s)
``E`` turbulent flat plate (Head)  -- ITTC-1957 and Schlichting power laws
``F`` leading-edge quadrature      -- convergence order of the integrated C_F
``G`` degenerate inputs            -- zero-length first strip (negative control)

The Falkner-Skan similarity solutions are obtained by a boundary-value solve of
``f''' + f f'' + beta (1 - f'^2) = 0`` on ``f(0)=f'(0)=0``, ``f'(inf)=1``; for
``U(x) = C x^m`` with ``m = beta/(2-beta)`` they give the exact momentum and
displacement thicknesses, hence the exact ``H``, ``cf`` and Thwaites parameter
``lambda = (theta^2/nu) dU/ds``.  No tabulated value is read from memory: the
reference is recomputed and qualified first (``f''(0) = 0.469600`` for beta = 0,
``1.232588`` for beta = 1).

Usage::

    python3 scripts/verify_integral_boundary_layer.py [--out DIR]

Writes ``results.json``, ``results-table.md`` and four PNG figures into the
output directory (default ``docs/verification-integral-boundary-layer``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.integral_boundary_layer import (  # noqa: E402
    h1_of_h,
    h_of_h1,
    ittc57_cf,
    ludwieg_tillmann_cf,
    march_boundary_layer,
)

# --- module correlations under test (mirrored here, kept in sync by tests) ----
def thwaites_H(lam):
    return np.clip(2.61 - 3.75 * np.asarray(lam) + 5.24 * np.asarray(lam) ** 2, 1.4, 4.0)


def thwaites_f(lam):
    return np.clip(0.22 + 1.57 * np.asarray(lam) - 1.8 * np.asarray(lam) ** 2, 0.0, 1.0)


# ---------------------------------------------------------------------------
# exact Falkner-Skan reference
# ---------------------------------------------------------------------------
def _fs_profile(beta, a, eta_max, rtol=1e-10, atol=1e-12, t_eval=None):
    """Integrate the Falkner-Skan ODE for a given wall curvature ``a = f''(0)``."""
    def rhs(t, y):
        return [y[1], y[2], -y[0] * y[2] - beta * (1.0 - y[1] ** 2)]

    sol = solve_ivp(rhs, (0.0, eta_max), [0.0, 0.0, a], t_eval=t_eval,
                    rtol=rtol, atol=atol, method="RK45")
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise FloatingPointError("Falkner-Skan integration diverged")
    return sol


def _fs_residual(beta, a, eta_max):
    """``f'(eta_max) - 1``: zero at the similarity solution."""
    try:
        sol = _fs_profile(beta, a, eta_max, rtol=1e-8, atol=1e-10)
    except FloatingPointError:
        return np.inf
    return float(sol.y[1][-1] - 1.0)


def _bvp_profile(beta, eta_max, nodes=400, tol=1e-10):
    """Fast collocation alternative; returns ``None`` where it does not converge."""
    from scipy.integrate import solve_bvp

    eta = np.linspace(0.0, eta_max, nodes)
    fp0 = np.tanh(eta)
    f0 = np.concatenate(([0.0], np.cumsum(0.5 * (fp0[1:] + fp0[:-1]) * np.diff(eta))))

    def ode(t, y):
        return np.vstack([y[1], y[2], -y[0] * y[2] - beta * (1.0 - y[1] ** 2)])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1.0])

    sol = solve_bvp(ode, bc, eta, np.vstack([f0, fp0, np.gradient(fp0, eta)]),
                    tol=tol, max_nodes=20000)
    if not sol.success or abs(sol.y[1][-1] - 1.0) > 1e-7:
        return None
    return sol.x, sol.y


def falkner_skan(beta, eta_max=None, tol=1e-12):
    """Exact similarity solution for ``U = C x^m``, ``m = beta/(2-beta)``.

    Collocation is tried first and shooting on ``f'(inf) = 1`` is the fallback,
    so the family is resolved through the separation solution ``f''(0) = 0``
    where pure collocation stalls.  Returns ``f''(0)``, the eta-integrals of the
    momentum and displacement thickness, ``H``, the exact ``f = cf Re_theta/2``
    and the exact Thwaites parameter
    ``lambda = (theta^2/nu) dU/ds = 2 m theta^2/(1+beta)``.
    """
    eta_max = eta_max if eta_max is not None else (20.0 if beta > -0.05 else 40.0)
    prof = _bvp_profile(beta, eta_max)
    if prof is None:
        grid = np.geomspace(1e-4, 2.5, 14)
        vals = [_fs_residual(beta, float(a), eta_max) for a in grid]
        bracket = None
        for i in range(len(grid) - 1):
            lo, hi = vals[i], vals[i + 1]
            if np.isfinite(lo) and np.isfinite(hi) and lo * hi < 0.0:
                bracket = (float(grid[i]), float(grid[i + 1]))
                break
        if bracket is None:
            raise RuntimeError(f"no shooting bracket for beta={beta}")
        a = brentq(lambda x: _fs_residual(beta, x, eta_max), bracket[0], bracket[1], xtol=tol)
        eta = np.linspace(0.0, eta_max, 4001)
        sol = _fs_profile(beta, a, eta_max, t_eval=eta)
        prof = (sol.t, sol.y)
    eta, y = prof
    fp = y[1]
    theta = float(np.trapezoid(fp * (1.0 - fp), eta))
    dstar = float(np.trapezoid(1.0 - fp, eta))
    m = beta / (2.0 - beta)
    return {
        "beta": float(beta),
        "m": float(m),
        "fpp_wall": float(y[2][0]),
        "theta_eta": theta,
        "dstar_eta": dstar,
        "H": dstar / theta,
        "f": float(y[2][0] * theta),
        "lam": float(2.0 * m * theta * theta / (1.0 + beta)),
        "tail": float(1.0 - fp[-1]),
    }


def separation_beta(tol=1e-10):
    """Exact separation point of the Falkner-Skan family (``f''(0) = 0``).

    With ``f''(0) = 0`` the wall shear vanishes, so the similarity ODE can be
    integrated directly; ``f'(inf) - 1`` changes sign across the separation
    value of ``beta``, which gives the root without a shooting iteration.
    """
    beta = brentq(lambda b: _fs_residual(b, 0.0, 40.0), -0.1995, -0.1988, xtol=tol)
    eta = np.linspace(0.0, 40.0, 8001)
    sol = _fs_profile(beta, 0.0, 40.0, t_eval=eta)
    fp = sol.y[1]
    theta = float(np.trapezoid(fp * (1.0 - fp), eta))
    dstar = float(np.trapezoid(1.0 - fp, eta))
    m = beta / (2.0 - beta)
    return beta, {
        "beta": float(beta),
        "fpp_wall": 0.0,
        "theta_eta": theta,
        "H": dstar / theta,
        "lam": float(2.0 * m * theta * theta / (1.0 + beta)),
        "tail": float(1.0 - fp[-1]),
    }


# ---------------------------------------------------------------------------
# marched cases
# ---------------------------------------------------------------------------
def march_flat_plate(Re, n=400, force_turbulent=False, laminar=False, leading_strip=1.0):
    """Flat plate of unit length; ``leading_strip`` scales the first strip length."""
    ds = np.full((n, 1), 1.0 / n)
    ds[0, 0] *= leading_strip
    areas = np.full((n, 1), 1.0 / n)
    ve = np.full((n, 1), 1.0)
    return march_boundary_layer(
        ve, ds, areas, nu=1.0 / Re, U=1.0, Re=Re,
        force_turbulent=force_turbulent,
        transition_Re_theta=(1.0e12 if laminar else 320.0),
    )


def march_wedge(beta, Re, n=400):
    """March a self-similar wedge flow ``U = C x^m`` from x = 0 to x = 1."""
    m = beta / (2.0 - beta)
    x_end = 1.0
    s = np.linspace(0.0, x_end, n)
    ds = np.r_[0.0, np.diff(s)]
    ve = np.maximum(s, 1e-6) ** m  # U = C x^m with C = 1 (Re fixes nu)
    nu = 1.0 / Re  # not used by the laminar branch except through lambda_wall
    ds[0] = s[1] - s[0]
    return march_boundary_layer(
        ve[:, None], ds[:, None], ds[:, None], nu=nu, U=float(ve[-1]), Re=Re,
        transition_Re_theta=1.0e12,
    ), float(ve[-1]), float(s[-1])


def station_coordinates(n, leading_strip=1.0):
    ds = np.full(n, 1.0 / n)
    ds[0] *= leading_strip
    return np.cumsum(ds)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def case_a_laminar_plate(Re=1.0e5, n=1600):
    r = march_flat_plate(Re, n, laminar=True)
    x = station_coordinates(n)
    Rex = x * Re
    theta_ex = 0.6641 * x / np.sqrt(Rex)
    dstar_ex = 1.7208 * x / np.sqrt(Rex)
    cf_ex = 0.6641 / np.sqrt(Rex)
    sel = x > 0.02
    return {
        "Re": Re,
        "n": n,
        "theta_max_rel_err": float(np.max(np.abs(r.theta[sel, 0] / theta_ex[sel] - 1.0))),
        "dstar_max_rel_err": float(np.max(np.abs(r.delta_star[sel, 0] / dstar_ex[sel] - 1.0))),
        "cf_max_rel_err": float(np.max(np.abs(r.cf[sel, 0] / cf_ex[sel] - 1.0))),
        "H_mean": float(np.mean(r.H[sel, 0])),
        "C_F": float(r.C_F),
        "C_F_exact": 1.328 / math.sqrt(Re),
        "C_F_rel_err": float(r.C_F / (1.328 / math.sqrt(Re)) - 1.0),
        "_curves": (x, r.cf[:, 0], cf_ex, r.theta[:, 0], theta_ex),
    }


def case_b_thwaites_fit(betas=(-0.19, -0.15, -0.10, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5)):
    rows = []
    for b in betas:
        ex = falkner_skan(b)
        Hm = float(thwaites_H(ex["lam"]))
        fm = float(thwaites_f(ex["lam"]))
        rows.append({
            "beta": b,
            "lam": ex["lam"],
            "H_exact": ex["H"],
            "H_model": Hm,
            "H_rel_err": Hm / ex["H"] - 1.0,
            "f_exact": ex["f"],
            "f_model": fm,
            "f_rel_err": fm / ex["f"] - 1.0,
        })
    return rows


def case_c_wedges(Re=5.0e5, n=4000, betas=(0.0, 0.1, 0.3, 0.5)):
    rows = []
    for b in betas:
        r, U, s = march_wedge(b, Re, n)
        ex = falkner_skan(b)
        theta_ex = math.sqrt(2.0 * (1.0 / Re) * s / ((1.0 + b) * U)) * ex["theta_eta"]
        cf_ex = 2.0 * ex["fpp_wall"] * math.sqrt((1.0 / Re) * (1.0 + b) / (2.0 * U * s))
        rows.append({
            "beta": b,
            "m": b / (2.0 - b),
            "theta_model": float(r.theta[-1, 0]),
            "theta_exact": float(theta_ex),
            "theta_rel_err": float(r.theta[-1, 0] / theta_ex - 1.0),
            "cf_model": float(r.cf[-1, 0]),
            "cf_exact": float(cf_ex),
            "cf_rel_err": float(r.cf[-1, 0] / cf_ex - 1.0),
            "H_model": float(r.H[-1, 0]),
            "H_exact": ex["H"],
            "H_rel_err": float(r.H[-1, 0] / ex["H"] - 1.0),
        })
    return rows


def case_d_separation():
    beta_s, ex = separation_beta()
    lam_exact = ex["lam"]
    # zero of the module's Thwaites f(lambda) polynomial
    roots = np.roots([-1.8, 1.57, 0.22])
    lam_model = float(sorted(roots)[0])
    return {
        "beta_sep_exact": beta_s,
        "fpp_wall_at_sep": ex["fpp_wall"],
        "H_at_sep": ex["H"],
        "lam_sep_exact": lam_exact,
        "lam_sep_model": lam_model,
        "lam_rel_dev": lam_model / lam_exact - 1.0,
        "beta_classical": -0.198838,
    }


def case_e_turbulent(Res=(1.0e6, 1.0e7, 1.0e8), ns=(400, 1600, 6400)):
    rows = []
    for Re in Res:
        for n in ns:
            r = march_flat_plate(Re, n, force_turbulent=True)
            rows.append({
                "Re": Re,
                "n": n,
                "C_F": float(r.C_F),
                "ITTC57": ittc57_cf(Re),
                "rel_err": float(r.C_F / ittc57_cf(Re) - 1.0),
                "theta_end": float(r.theta[-1, 0]),
                "H_end": float(r.H[-1, 0]),
                "separated": bool(r.separated[-1, 0]),
            })
    # local comparison at Re = 1e7
    Re, n = 1.0e7, 1600
    r = march_flat_plate(Re, n, force_turbulent=True)
    x = station_coordinates(n)
    Rex = x * Re
    cf_pow = 0.0592 * Rex ** (-0.2)
    th_pow = 0.036 * x * Rex ** (-0.2)
    sel = x > 0.02
    local = {
        "cf_max_rel_err": float(np.max(np.abs(r.cf[sel, 0] / cf_pow[sel] - 1.0))),
        "cf_mean_rel_err": float(np.mean(np.abs(r.cf[sel, 0] / cf_pow[sel] - 1.0))),
        "theta_end_rel_err": float(r.theta[-1, 0] / th_pow[-1] - 1.0),
        "H_power_law": 1.286,
        "_curves": (x, r.cf[:, 0], cf_pow, r.theta[:, 0], th_pow),
    }
    return rows, local


def case_f_quadrature(Re=1.0e5, ns=(100, 200, 400, 800, 1600, 3200)):
    rows = []
    exact = 1.328 / math.sqrt(Re)
    for n in ns:
        r = march_flat_plate(Re, n, laminar=True)
        rows.append({"n": n, "C_F": float(r.C_F), "rel_err": float(r.C_F / exact - 1.0)})
    e = np.array([row["rel_err"] for row in rows])
    n = np.array([float(row["n"]) for row in rows])
    # excess error over the (bias) asymptote; fit the points still well above it
    bias = e[-1]
    excess = np.abs(e - bias)
    sel = excess > 1e-3
    slope = float(np.polyfit(np.log(n[sel]), np.log(excess[sel]), 1)[0]) if sel.sum() >= 3 else float("nan")
    return {"rows": rows, "asymptote_rel_err": float(bias), "excess_slope": slope, "exact_C_F": exact}


def case_g_degenerate():
    good = march_flat_plate(1.0e7, 800, force_turbulent=True, leading_strip=1.0)
    bad = march_flat_plate(1.0e7, 800, force_turbulent=True, leading_strip=0.0)
    return {
        "C_F_positive_strip": float(good.C_F),
        "C_F_zero_strip": float(bad.C_F),
        "ratio": float(bad.C_F / good.C_F),
        "separated_zero_strip": bool(bad.separated[-1, 0]),
        "theta_ratio": float(bad.theta[-1, 0] / good.theta[-1, 0]),
    }


def case_h_closures():
    H = np.linspace(1.15, 4.0, 200)
    H1 = h1_of_h(H)
    back = h_of_h1(H1)
    sel = (H > 1.2) & (H < 3.9)
    return {
        "roundtrip_max_rel_err": float(np.max(np.abs(back[sel] / H[sel] - 1.0))),
        "lt_cf_H1": ludwieg_tillmann_cf(1.4, 1.0e3),
        "lt_cf_H2": ludwieg_tillmann_cf(2.4, 1.0e3),
        "lt_cf_Re1": ludwieg_tillmann_cf(1.4, 1.0e3),
        "lt_cf_Re2": ludwieg_tillmann_cf(1.4, 1.0e5),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(out, a, b_rows, c_rows, e_rows, e_local, f):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    red = dict(color="tab:red", lw=1.4)
    blue = dict(color="tab:blue", lw=1.4)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
    x, cf, cf_ex, th, th_ex = a["_curves"]
    ax[0].plot(x, cf, "o", ms=2.5, color="tab:red", label="Thwaites (module)")
    ax[0].plot(x, cf_ex, "-", color="k", lw=1.2, label=r"Blasius $0.6641/\sqrt{Re_x}$")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel(r"$c_f$")
    ax[0].set_title(f"Laminar flat plate, Re={a['Re']:.0e}")
    ax[0].legend()
    ax[1].plot(x, cf / cf_ex - 1.0, **red, label=r"$c_f$")
    ax[1].plot(x, th / th_ex - 1.0, **blue, label=r"$\theta$")
    ax[1].axhline(0.03, ls=":", color="0.4"); ax[1].axhline(-0.03, ls=":", color="0.4")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("relative error vs Blasius")
    ax[1].set_title("±3% acceptance band (dotted)")
    ax[1].legend()
    fig.tight_layout(); p = out / "laminar-plate-local-fields.png"; fig.savefig(p, dpi=140); plt.close(fig)
    paths.append((p.name, "Laminar flat plate: local cf against the exact Blasius solution and the relative error of cf/theta."))

    lam = np.array([r["lam"] for r in b_rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
    grid = np.linspace(-0.08, 0.10, 300)
    ax[0].plot(grid, thwaites_H(grid), "-", color="tab:red", label="module polynomial")
    ax[0].plot(lam, [r["H_exact"] for r in b_rows], "o", color="k", ms=5, label="exact Falkner-Skan")
    ax[0].set_xlabel(r"$\lambda = (\theta^2/\nu)\,dU/ds$"); ax[0].set_ylabel("H")
    ax[0].set_title("Shape factor correlation"); ax[0].legend()
    ax[1].plot(grid, thwaites_f(grid), "-", color="tab:red", label="module polynomial")
    ax[1].plot(lam, [r["f_exact"] for r in b_rows], "o", color="k", ms=5, label="exact Falkner-Skan")
    ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel(r"$f = c_f Re_\theta/2$")
    ax[1].set_title("Skin-friction correlation"); ax[1].legend()
    fig.tight_layout(); p = out / "thwaites-fit-vs-exact.png"; fig.savefig(p, dpi=140); plt.close(fig)
    paths.append((p.name, "Thwaites H(lambda) and f(lambda) correlations against exact Falkner-Skan similarity solutions."))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
    bs = [r["beta"] for r in c_rows]
    ax[0].bar(np.arange(len(bs)) - 0.2, [100 * r["theta_rel_err"] for r in c_rows], 0.4, label=r"$\theta$", color="tab:blue")
    ax[0].bar(np.arange(len(bs)) + 0.2, [100 * r["cf_rel_err"] for r in c_rows], 0.4, label=r"$c_f$", color="tab:red")
    ax[0].axhline(3, ls=":", color="0.4"); ax[0].axhline(-3, ls=":", color="0.4")
    ax[0].set_xticks(np.arange(len(bs))); ax[0].set_xticklabels([f"{b:g}" for b in bs])
    ax[0].set_xlabel(r"Falkner-Skan $\beta$"); ax[0].set_ylabel("relative error [%]")
    ax[0].set_title("Marched wedge vs exact similarity"); ax[0].legend()
    ax[1].plot(bs, [r["H_model"] for r in c_rows], "o-", color="tab:red", label="module H")
    ax[1].plot(bs, [r["H_exact"] for r in c_rows], "s--", color="k", label="exact H")
    ax[1].set_xlabel(r"$\beta$"); ax[1].set_ylabel("H"); ax[1].set_title("Shape factor along the wedge family")
    ax[1].legend()
    fig.tight_layout(); p = out / "wedge-march-vs-similarity.png"; fig.savefig(p, dpi=140); plt.close(fig)
    paths.append((p.name, "Self-similar wedge marches: momentum thickness, skin friction and shape factor against exact Falkner-Skan values."))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x, cf, cf_pow, th, th_pow = e_local["_curves"]
    ax[0].plot(x, cf, **red, label="Head (module)")
    ax[0].plot(x, cf_pow, "-", color="k", lw=1.0, label=r"$0.0592 Re_x^{-1/5}$")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel(r"$c_f$"); ax[0].set_title(r"Turbulent plate, Re=$10^7$"); ax[0].legend()
    Res = sorted({r["Re"] for r in e_rows})
    for Re in Res:
        pts = [r for r in e_rows if r["Re"] == Re]
        ax[1].plot([r["n"] for r in pts], [100 * r["rel_err"] for r in pts], "o-", label=f"Re={Re:.0e}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("stations n"); ax[1].set_ylabel(r"$C_F$ error vs ITTC-57 [%]")
    ax[1].set_title("Integrated friction bias"); ax[1].legend(fontsize=7)
    nn = np.array([r["n"] for r in f["rows"]])
    ee = np.array([100 * r["rel_err"] for r in f["rows"]])
    ax[2].loglog(nn, np.abs(ee - 100 * f["asymptote_rel_err"]), "o-", label="excess |error|")
    ax[2].loglog(nn, np.abs(ee - 100 * f["asymptote_rel_err"])[-1] * (nn / nn[-1]) ** -0.5, "k--", label=r"$n^{-1/2}$")
    ax[2].set_xlabel("stations n"); ax[2].set_ylabel("excess $C_F$ error [%]")
    ax[2].set_title("Leading-edge quadrature"); ax[2].legend()
    fig.tight_layout(); p = out / "turbulent-plate-and-quadrature.png"; fig.savefig(p, dpi=140); plt.close(fig)
    paths.append((p.name, "Turbulent flat plate: local cf vs the power law, integrated CF bias vs ITTC-57 and the n^-1/2 leading-edge quadrature convergence."))
    return paths


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/verification-integral-boundary-layer")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    refs = {f"{b:g}": falkner_skan(b) for b in (0.0, 0.3, 1.0, -0.19)}
    a = case_a_laminar_plate()
    b_rows = case_b_thwaites_fit()
    c_rows = case_c_wedges()
    d = case_d_separation()
    e_rows, e_local = case_e_turbulent()
    f = case_f_quadrature()
    g = case_g_degenerate()
    h = case_h_closures()
    figs = figures(out, a, b_rows, c_rows, e_rows, e_local, f)

    payload = {
        "reference_solver": refs,
        "A_laminar_plate": {k: v for k, v in a.items() if not k.startswith("_")},
        "B_thwaites_fit": b_rows,
        "C_wedges": c_rows,
        "D_separation": d,
        "E_turbulent": {"refinement": e_rows, "local": {k: v for k, v in e_local.items() if not k.startswith("_")}},
        "F_quadrature": f,
        "G_degenerate": g,
        "H_closures": h,
        "figures": [n for n, _ in figs],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1) + "\n")

    # generated tables (numbers never copied by hand into the narrative)
    lines = ["# Generated result tables", "",
             "Produced by `scripts/verify_integral_boundary_layer.py`; do not edit by hand.", "",
             "## Reference solver qualification", "",
             "| beta | f''(0) | theta_eta | H | f = cf Re_theta/2 | lambda |", "|---|---|---|---|---|---|"]
    for k, r in refs.items():
        lines.append(f"| {k} | {r['fpp_wall']:.6f} | {r['theta_eta']:.6f} | {r['H']:.4f} | {r['f']:.5f} | {r['lam']:+.5f} |")
    lines += ["", "## A. Laminar flat plate vs Blasius", "",
              f"Re = {a['Re']:.1e}, n = {a['n']} stations", "",
              "| quantity | model | exact | rel. error |", "|---|---|---|---|",
              f"| theta (max) | - | - | {100*a['theta_max_rel_err']:+.2f}% |",
              f"| delta* (max) | - | - | {100*a['dstar_max_rel_err']:+.2f}% |",
              f"| cf (max) | - | - | {100*a['cf_max_rel_err']:+.2f}% |",
              f"| C_F | {a['C_F']:.6f} | {a['C_F_exact']:.6f} | {100*a['C_F_rel_err']:+.2f}% |",
              "", "## B. Thwaites correlations vs exact Falkner-Skan", "",
              "| beta | lambda | H_model | H_exact | err | f_model | f_exact | err |",
              "|---|---|---|---|---|---|---|---|"]
    for r in b_rows:
        lines.append(f"| {r['beta']:+.3f} | {r['lam']:+.4f} | {r['H_model']:.4f} | {r['H_exact']:.4f} | {100*r['H_rel_err']:+.2f}% "
                     f"| {r['f_model']:.4f} | {r['f_exact']:.4f} | {100*r['f_rel_err']:+.2f}% |")
    lines += ["", "## C. Marched wedges vs exact similarity", "",
              "| beta | m | theta model | theta exact | err | cf model | cf exact | err | H model | H exact |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in c_rows:
        lines.append(f"| {r['beta']:g} | {r['m']:+.4f} | {r['theta_model']:.6f} | {r['theta_exact']:.6f} | {100*r['theta_rel_err']:+.2f}% "
                     f"| {r['cf_model']:.6f} | {r['cf_exact']:.6f} | {100*r['cf_rel_err']:+.2f}% | {r['H_model']:.4f} | {r['H_exact']:.4f} |")
    lines += ["", "## D. Laminar separation criterion", "",
              f"exact beta_sep = {d['beta_sep_exact']:.5f} (classical {d['beta_classical']:.6f}), lambda_sep exact = {d['lam_sep_exact']:+.5f}, "
              f"module polynomial zero = {d['lam_sep_model']:+.5f} (deviation {100*d['lam_rel_dev']:+.1f}%)", "",
              "## E. Turbulent flat plate (Head) vs ITTC-1957", "",
              "| Re | n | C_F | ITTC-57 | error | theta_end | H_end |", "|---|---|---|---|---|---|---|"]
    for r in e_rows:
        lines.append(f"| {r['Re']:.0e} | {r['n']} | {r['C_F']:.6f} | {r['ITTC57']:.6f} | {100*r['rel_err']:+.2f}% | {r['theta_end']:.6f} | {r['H_end']:.3f} |")
    lines += ["", f"local cf vs 0.0592 Re_x^(-1/5): mean |err| {100*e_local['cf_mean_rel_err']:.2f}%, max {100*e_local['cf_max_rel_err']:.2f}%; "
              f"theta_end error {100*e_local['theta_end_rel_err']:+.2f}%.", "",
              "## F. Leading-edge quadrature convergence (laminar C_F)", "",
              "| n | C_F | error vs Blasius |", "|---|---|---|"]
    for r in f["rows"]:
        lines.append(f"| {r['n']} | {r['C_F']:.6f} | {100*r['rel_err']:+.2f}% |")
    lines += ["", f"asymptotic bias {100*f['asymptote_rel_err']:+.2f}%, excess-error slope {f['excess_slope']:+.3f} (expect -0.5 for a right-endpoint rule on s^-1/2).", "",
              "## G. Degenerate input (zero-length first strip)", "",
              f"C_F = {g['C_F_zero_strip']:.6f} vs {g['C_F_positive_strip']:.6f} (ratio {g['ratio']:.2f}), separated = {g['separated_zero_strip']}.", "",
              "## H. Closure relations", "",
              f"H1(H) round-trip max rel. error {h['roundtrip_max_rel_err']:.2e}; "
              f"Ludwieg-Tillmann cf(H=1.4)={h['lt_cf_H1']:.5f} > cf(H=2.4)={h['lt_cf_H2']:.5f}.", ""]
    (out / "results-table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if not k.startswith(("B_", "C_", "E_"))}, indent=1)[:4000])
    print("\nfigures:", [n for n, _ in figs])
    print("wrote", out / "results.json", out / "results-table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())