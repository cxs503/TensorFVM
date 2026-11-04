"""Literature audit of the Spalding all-y-plus wall-law constitutive stack.

The shipping tests for ``tensorfvm.wall_functions`` are self-consistency tests
(manufactured shear, Jacobian-versus-differences, torch/NumPy parity). None of
them asks whether the *law itself* agrees with externally published wall laws
and friction correlations. This script closes that verification debt with a
50-digit reference inverse, two literature anchors (Reichardt 1951 composite
profile, Dean 1978 channel friction correlation) and a validity-window sweep.

Run: PYTHONPATH=src python3 scripts/audit_wall_law_literature.py
Writes JSON + PNG + Markdown into src/tensorfvm/verification/artifacts/wall_law/
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.wall_functions import KAPPA, E, spalding_y_plus, u_plus_at_y_plus  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "src/tensorfvm/verification/artifacts/wall_law"
K, B_INTERCEPT = KAPPA, float(np.log(E) / KAPPA)


def reference_inverse(y_plus, digits=50):
    """Independent high-precision inverse of y+ = u+ + remainder(u+)/E."""
    import mpmath as mp
    mp.mp.dps = digits
    kap, e = mp.mpf(str(K)), mp.mpf(str(E))
    out = []
    for yp in np.atleast_1d(y_plus):
        target = mp.mpf(str(float(yp)))
        guess = mp.mpf(str(float(u_plus_at_y_plus(yp))))
        g = lambda u: u + (mp.e**(kap * u) - 1 - kap * u - (kap * u) ** 2 / 2 - (kap * u) ** 3 / 6) / e - target
        out.append(mp.findroot(g, guess, tol=mp.mpf(10) ** (-digits + 5), maxsteps=200))
    return np.array([float(v) for v in out])


def reichardt(y_plus, kappa=KAPPA):
    """Reichardt (1951) composite law: viscous sublayer through log region."""
    yp = np.asarray(y_plus, float)
    return (np.log1p(kappa * yp) / kappa
            + 7.8 * (1 - np.exp(-yp / 11) - (yp / 11) * np.exp(-yp / 3)))


MASTER = np.concatenate([[0.], np.geomspace(1e-4, 1e6, 300000)])  # geometric: resolves y+->0 and Re_tau->1e6
CUMULATIVE = {}


def bulk_velocity(Re_tau, kappa=KAPPA, E_const=E):
    """U_b/u_tau from the module's own u+(y+) profile (trapezoid in y+)."""
    key = (float(kappa), float(E_const))
    if key not in CUMULATIVE:
        up = u_plus_at_y_plus(MASTER, kappa, E_const)
        CUMULATIVE[key] = np.concatenate([[0.], np.cumsum(.5 * (up[1:] + up[:-1]) * np.diff(MASTER))])
    return np.interp(Re_tau, MASTER, CUMULATIVE[key]) / Re_tau


def pvk_darcy(Re_D):
    """Prandtl-von Karman smooth-pipe law: 1/sqrt(f) = 2 log10(Re*sqrt(f)) - 0.8."""
    Re_D = np.atleast_1d(np.asarray(Re_D, float))
    f = 0.02 * np.ones_like(Re_D)
    for _ in range(80):
        root = np.sqrt(f)
        f = 1. / (2 * np.log10(Re_D * root) - 0.8) ** 2
    return f


def dean(Re_b):
    """Dean (1978) channel friction correlation Cf = 0.073 Re_b^-0.25."""
    return 0.073 * np.asarray(Re_b, float) ** -0.25


def cf_from_law(Re_b, kappa=KAPPA, E_const=E):
    """Channel Cf implied by the wall law: Cf = 2 (u_tau/U_b)^2, Re_b = 2 Re_tau U_b/u_tau."""
    Re_b = np.atleast_1d(np.asarray(Re_b, float))
    out = np.empty_like(Re_b)
    for i, target in enumerate(Re_b):
        lo, hi = 30., 1e6
        for _ in range(200):
            mid = .5 * (lo + hi)
            if 2 * mid * bulk_velocity(mid, kappa=kappa, E_const=E_const) < target:
                lo = mid
            else:
                hi = mid
        Re_tau = .5 * (lo + hi)
        ratio = bulk_velocity(Re_tau, kappa=kappa, E_const=E_const)
        out[i] = 2. / ratio ** 2
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"constants": {"kappa": K, "E": E, "B": B_INTERCEPT}}

    # (1) Inverse-law accuracy against the 50-digit reference.
    yp_grid = np.logspace(-6, 6, 25)
    ref = reference_inverse(yp_grid)
    got = u_plus_at_y_plus(yp_grid)
    rel = np.abs(got - ref) / ref
    result["inverse_accuracy"] = {
        "y_plus_range": [float(yp_grid[0]), float(yp_grid[-1])],
        "samples": 25,
        "max_relative_error": float(rel.max()),
        "at_y_plus": float(yp_grid[int(rel.argmax())]),
        "reference": "mpmath, 50 significant digits, bracketed Newton on the law",
    }

    # (2) Region characterisation: where does the law leave the two asymptotes?
    fine = np.logspace(-3, 5, 4001)
    up = u_plus_at_y_plus(fine)
    viscous_dev = np.abs(up - fine) / up
    loglaw = np.log(fine) / K + B_INTERCEPT
    log_dev = np.abs(up - loglaw) / up
    result["regions"] = {
        "viscous_1pct_y_plus": float(fine[np.argmax(viscous_dev > 1e-2)]),
        "log_1pct_y_plus": float(fine[np.argmax(log_dev < 1e-2)]),
        "log_half_pct_y_plus": float(fine[np.argmax(log_dev < 5e-3)]),
        "kappa_times_y_plus_at_log_entry": float(K * fine[np.argmax(log_dev < 1e-2)]),
        "note": "asymptote deviation is measured relative to u+ itself",
    }

    # (3) Reichardt (1951) composite profile as an independent literature anchor.
    yp_lit = np.logspace(0, 3, 601)
    up_lit = u_plus_at_y_plus(yp_lit)
    du = up_lit - reichardt(yp_lit)
    band = (yp_lit >= 30) & (yp_lit <= 1000)
    state = dict(speed=10.0, distance=1e-3, nu=1e-6, rho=1000.)
    # Physical use case: U=10 m/s sampled at y=1 mm in water; invert the module
    # law and the Reichardt law at the same y+ to compare implied u_tau.
    yp_state = np.array([0.2, 1., 5., 30., 100., 300.])
    ut_ref = np.empty_like(yp_state)
    lo, hi = 1e-4, 10.0
    for i, ypv in enumerate(yp_state):
        a, b = lo, hi
        for _ in range(200):
            m = .5 * (a + b)
            if m * reichardt(state["distance"] * m / state["nu"]) < state["speed"]:
                a = m
            else:
                b = m
        ut_ref[i] = .5 * (a + b)
    ut_mod = np.empty_like(yp_state)
    for i, ypv in enumerate(yp_state):
        a, b = lo, hi
        for _ in range(200):
            m = .5 * (a + b)
            if m * float(u_plus_at_y_plus(state["distance"] * m / state["nu"])) < state["speed"]:
                a = m
            else:
                b = m
        ut_mod[i] = .5 * (a + b)
    result["reichardt_1951"] = {
        "formula": "u+ = ln(1+kappa*y+)/kappa + 7.8*(1 - exp(-y+/11) - (y+/11)exp(-y+/3))",
        "max_abs_delta_u_plus_1_1000": float(np.abs(du[yp_lit <= 1000]).max()),
        "max_abs_delta_u_plus_30_1000": float(np.abs(du[band]).max()),
        "rms_delta_u_plus_30_1000": float(np.sqrt(np.mean(du[band] ** 2))),
        "state": state,
        "implied_u_tau": {"y_plus_at_state": float(np.mean(state["distance"] * ut_mod / state["nu"])),
                          "module": float(ut_mod[0]), "reichardt": float(ut_ref[0]),
                          "relative_difference": float(abs(ut_mod[0] - ut_ref[0]) / ut_ref[0])},
    }

    # (4) Channel friction factor against Dean (1978).
    Re_b = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    cf = cf_from_law(Re_b)
    cf_dean = dean(Re_b)
    err = cf / cf_dean - 1
    n_fit = -np.polyfit(np.log(Re_b), np.log(cf), 1)[0]
    # Matched-definition smooth-pipe check: hydraulic diameter D_h = 4h, so
    # f_Darcy = 4 Cf (both from tau_w/(rho U_b^2)) and Re_Dh = 4h U_b/nu = 2 Re_b.
    f_pvk = pvk_darcy(2 * Re_b)
    cf_pvk = f_pvk / 4
    err_pvk = cf / cf_pvk - 1
    result["channel_friction"] = {
        "reference": "Dean (1978) Cf = 0.073 Re_b^-0.25",
        "Re_b": Re_b.tolist(), "Cf_law": cf.tolist(), "Cf_dean": cf_dean.tolist(),
        "relative_error": err.tolist(),
        "max_abs_relative_error": float(np.abs(err).max()),
        "fitted_exponent": float(n_fit),
        "pipe_prandtl_von_karman": {
            "law": "1/sqrt(f_D) = 2 log10(Re_D sqrt(f_D)) - 0.8, compared at Re_D = 2 Re_b, f_D = 4 Cf",
            "Re_Dh": (2 * Re_b).tolist(), "Cf_pipe_equivalent": cf_pvk.tolist(),
            "relative_error": err_pvk.tolist(),
            "max_abs_relative_error": float(np.abs(err_pvk).max()),
        },
        "definition": "Cf = 2 tau_w/(rho U_b^2), Re_b = 2h U_b/nu",
    }

    # (5) Constant sensitivity: how much Cf moves when kappa/E move.
    sens = {}
    for label, (kk, ee) in {"kappa=0.40,E=9.0": (0.40, 9.0), "kappa=0.41,E=9.8 (shipped)": (0.41, 9.8),
                            "kappa=0.4187,E=9.793": (0.4187, 9.793)}.items():
        sens[label] = dict(Cf_Re1e5=float(cf_from_law(np.array([1e5]), kk, ee)[0]),
                           B=float(np.log(ee) / kk))
    result["constant_sensitivity"] = sens

    # (6) Validity window (the two defects repaired in this pass).
    wide = np.array([1e6, 1e10, 1e20, 1e30, 1e60])
    resid = np.array([float(spalding_y_plus(float(u_plus_at_y_plus(v)))) / v - 1 for v in wide])
    result["validity_window"] = {
        "y_plus": wide.tolist(),
        "u_plus": [float(u_plus_at_y_plus(v)) for v in wide],
        "log_law_estimate": [float((np.log(v) + np.log(E)) / K) for v in wide],
        "law_residual": resid.tolist(),
        "regression": "pre-fix bisection on [0, y+] returned 403.90 instead of 174.05 at y+=1e30",
    }
    (OUT / "wall-law-literature.json").write_text(json.dumps(result, indent=2) + "\n")

    # Figure: law-of-the-wall comparison + channel friction.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.logspace(-1, 4, 800)
    ax[0].semilogx(grid, u_plus_at_y_plus(grid), "k-", lw=2, label="Spalding (shipped module)")
    ax[0].semilogx(grid, reichardt(grid), "r--", lw=1.5, label="Reichardt 1951")
    ax[0].semilogx(grid, grid, ":", color="0.5", label="$u^+=y^+$")
    ax[0].semilogx(grid, np.log(grid) / K + B_INTERCEPT, "-.", color="0.4",
                   label=r"$\ln(Ey^+)/\kappa$, $B$=%.2f" % B_INTERCEPT)
    ax[0].set(xlim=(1e-1, 1e4), ylim=(0, 30), xlabel="$y^+$", ylabel="$u^+$",
              title="Law of the wall vs literature (max |Δu+| = %.3f)" % result["reichardt_1951"]["max_abs_delta_u_plus_1_1000"])
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].loglog(Re_b, cf_dean, "s-", color="tab:blue", label="Dean 1978")
    ax[1].loglog(Re_b, cf, "o--", color="tab:red", label="Spalding law, this module")
    ax[1].loglog(Re_b, cf_pvk, "^-", color="tab:green", lw=1.2,
                 label="smooth pipe PvK, $f_D/4$ at $Re_{D_h}=2Re_b$")
    ax[1].set(xlabel="$Re_b=2hU_b/\\nu$", ylabel="$C_f$",
              title="Channel friction vs pipe PvK (max |err| = %.1f%%)" % (100 * np.abs(err_pvk).max()))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "wall-law-literature.png", dpi=140)

    print(json.dumps({k: v for k, v in result.items()
                      if k in ("constants", "inverse_accuracy", "regions", "channel_friction")}, indent=2))
    print(json.dumps(result["reichardt_1951"], indent=2))
    print(json.dumps(result["constant_sensitivity"], indent=2))
    print(json.dumps(result["validity_window"], indent=2))


if __name__ == "__main__":
    main()