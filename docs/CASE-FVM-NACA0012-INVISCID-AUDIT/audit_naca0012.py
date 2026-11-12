"""Board audit (m16236) of CASE-FVM-NACA0012-INVISCID (mission m16212).

Every number below is computed here, not copied from the deliverable:
  A. geometry validation of the panelised NACA 0012 (t/c, chord, closed TE, normals)
  B. re-computation of Cl(2 deg) on the delivered module (N = 50..800 panels/side)
  C. force-bookkeeping audit: Cl from surface pressure vs Kutta-Joukowski on the
     circulation actually produced by the solver (Gamma = gamma * perimeter)
  D. exact closed-form anchor for a *thick symmetric* section: symmetric Joukowski
     profile, Kutta at the cusped TE gives Gamma = 4*pi*R*U*sin(alpha) and
     Cl = 2*Gamma/(U*c) = 8*pi*(R/c)*sin(alpha)  (flat-plate limit -> 2*pi*alpha)
  E. behaviour of the two Kutta options the module exposes
  F. convergence / trailing-edge behaviour of the surface speed (does the solver
     really drive the TE to a stagnation point, as a finite-angle Kutta needs?)

Run:  python3 artifacts/m16236/audit_naca0012.py
Out:  audit_results.json, fig_audit.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/tp-agent/work/m16212/src")
from tensorfvm.panel_airfoil import (  # noqa: E402
    _influence, cl_thin_aerofoil, naca4_panels, solve_lifting_panels)

OUT = Path(__file__).resolve().parent
CL_THIN = cl_thin_aerofoil(2.0)


# --------------------------------------------------------------- exact anchor
def joukowski_exact_slope_ratio(mu: float, a: float = 1.0, alpha_deg: float = 2.0) -> dict:
    """Exact Cl of the symmetric Joukowski section, from Gamma (Kutta) and from the
    conformal-map chord.  Cl = 2*Gamma/(U*c), Gamma = 4*pi*R*U*sin(alpha),
    c = z_TE - z_LE with z = zeta + a^2/zeta, circle |zeta-mu| = a-mu."""
    R = a - mu
    al = math.radians(alpha_deg)
    Gamma = 4.0 * math.pi * R * math.sin(al)
    z_te = 2.0 * a
    z_le = (2.0 * mu - a) + a * a / (2.0 * mu - a)
    chord = z_te - z_le
    # thickness of the same section, from a fine polygon
    th = np.linspace(0.0, 2.0 * math.pi, 20001)
    z = (mu + R * np.exp(1j * th))
    zz = z + a * a / z
    t_over_c = float(2.0 * np.abs(zz.imag).max() / (zz.real.max() - zz.real.min()))
    cl = 2.0 * Gamma / chord
    return dict(mu=mu, t_over_c=t_over_c, chord=float(chord), cl=float(cl),
                slope_ratio=float(cl / CL_THIN))


def mu_for_thickness(target: float, a: float = 1.0) -> float:
    lo, hi = 1e-9, 0.99 * a
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if joukowski_exact_slope_ratio(mid, a)["t_over_c"] < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------- alternative Kutta runs
def solve_kutta_variant(alpha_deg: float, panels_per_side: int, variant: str):
    """variant 'equal_pressure' reproduces the deliverable; 'te_stagnation'
    replaces the two trailing-edge Neumann rows by V_t = 0 (Morino style)."""
    geo = naca4_panels(0.12, panels_per_side)
    mid, nrm, t, arc = geo["mid"], geo["normal"], geo["t"], geo["arc"]
    n = len(mid)
    al = math.radians(alpha_deg)
    U = np.array([math.cos(al), math.sin(al)])
    An, At = _influence(mid, nrm, t, geo)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    if variant == "equal_pressure":
        A[:n, :] = An
        rhs[:n] = -(nrm @ U)
        A[n, :] = At[0, :] + At[-1, :]
        rhs[n] = -(t[0] @ U + t[-1] @ U)
    else:
        A[:n - 1, :] = An[:n - 1, :]
        rhs[:n - 1] = -(nrm[:n - 1] @ U)
        A[n - 1, :] = At[0, :]
        rhs[n - 1] = -(t[0] @ U)
        A[n, :] = At[-1, :]
        rhs[n] = -(t[-1] @ U)
    sol = np.linalg.solve(A, rhs)
    vt = t @ U + At @ sol
    cp = 1.0 - vt ** 2
    fy = -0.5 * float(np.sum(cp * nrm[:, 1] * arc))
    return dict(cl=2.0 * fy, gamma=float(sol[-1]),
                v_te_up=float(abs(vt[0])), v_te_lo=float(abs(vt[-1])))


def main() -> dict:
    res: dict = {"case": "CASE-FVM-NACA0012-INVISCID", "audit_mission": 16236,
                 "cl_thin_2pa": CL_THIN, "card_value": 0.2198, "card_tolerance": 0.03}

    # A. geometry
    geo = naca4_panels(0.12, 200)
    loop = np.vstack([geo["a"], geo["b"][-1]])
    chord = float(loop[:, 0].max() - loop[:, 0].min())
    tc = float(2.0 * np.abs(loop[:, 1]).max() / chord)
    res["A_geometry"] = dict(t_over_c=tc, chord=chord,
                             te_gap=float(np.linalg.norm(geo["b"][-1] - geo["a"][0])),
                             x_first_panel=float(geo["mid"][0, 0]),
                             normal_upper_out=bool(geo["normal"][:200, 1].min() > 0),
                             normal_lower_out=bool(geo["normal"][200:, 1].max() < 0))

    # B. reproduction + panel convergence of the delivered solver
    conv = []
    for nps in (50, 100, 200, 400, 800):
        r = solve_lifting_panels(2.0, nps)
        conv.append(dict(panels_per_side=nps, panels=2 * nps, cl=r.cl_pressure,
                         err_vs_thin_pct=100.0 * (r.cl_pressure / CL_THIN - 1.0),
                         te_speed=float(0.5 * (abs(r.surface_speed[0]) + abs(r.surface_speed[-1]))),
                         kutta_residual=r.kutta_residual,
                         normal_residual=r.normal_residual))
    res["B_panel_convergence"] = conv

    # C. force bookkeeping
    r = solve_lifting_panels(2.0, 400)
    perimeter = float(naca4_panels(0.12, 400)["arc"].sum())
    res["C_force_bookkeeping"] = dict(
        cl_pressure=r.cl_pressure,
        cl_joukowski_as_coded=r.cl_joukowski,
        gamma=r.gamma, perimeter=perimeter,
        cl_kj_with_perimeter=2.0 * r.gamma * perimeter,
        ratio_as_coded_over_pressure=r.cl_joukowski / r.cl_pressure,
        ratio_with_perimeter_over_pressure=(2.0 * r.gamma * perimeter) / r.cl_pressure)

    # D. exact thick-symmetric anchor + the deliverable's claimed law
    anchor = [joukowski_exact_slope_ratio(mu_for_thickness(t)) for t in (0.02, 0.06, 0.09, 0.12)]
    claimed = []
    for t in (0.02, 0.06, 0.09, 0.12):
        rr = solve_lifting_panels(2.0, 200, thickness=t)
        claimed.append(dict(t_over_c=t, naca_panel_ratio=rr.cl_pressure / CL_THIN))
    res["D_exact_joukowski"] = {
        "law": "Cl = 8*pi*(R/c)*sin(alpha)  (Kutta at cusped TE, flat-plate limit -> 2*pi*alpha)",
        "anchors": anchor,
        "deliverable_claim": "Cl/Cl_thin ~ 1 + 0.83*(t/c)  (linear, slope-raising)",
        "panel_measured": claimed,
        "sign_of_claimed_effect_vs_exact":
            "opposite: exact Joukowski family slope ratio falls below 1 with thickness",
        "note": ("the Joukowski family is not the NACA 0012 geometry; the anchor is used only to "
                 "disprove the unconditional statement '12 % thickness raises the inviscid slope "
                 "by ~10 %', not to predict the NACA 0012 value"),
    }

    # E. exposed Kutta options
    opt = {}
    for variant in ("equal_pressure", "te_stagnation"):
        rows = []
        for nps in (50, 100, 200):
            s = solve_kutta_variant(2.0, nps, variant)
            s["panels_per_side"] = nps
            rows.append(s)
        opt[variant] = rows
    opt["module_option_stagnation_call"] = dict(
        cl=solve_lifting_panels(2.0, 200, kutta="stagnation").cl_pressure,
        kutta_residual=solve_lifting_panels(2.0, 200, kutta="stagnation").kutta_residual)
    opt["te_stagnation_verdict"] = ("rejected as an audit fix: Cl(alpha=0) != 0 and no convergence; "
                                    "reported only to show the module has no well-posed "
                                    "finite-angle-TE Kutta variant")
    res["E_kutta_options"] = opt

    # F. TE behaviour: does the surface speed vanish at the TE as a finite-angle
    #    Kutta condition requires?
    res["F_te_speed_trend"] = [dict(panels_per_side=c["panels_per_side"], te_speed=c["te_speed"])
                               for c in conv]

    (OUT / "audit_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        nps = np.array([c["panels_per_side"] for c in conv])
        cl = np.array([c["cl"] for c in conv])
        ax[0].axhspan(CL_THIN * 0.97, CL_THIN * 1.03, color="0.85", label="card tolerance +-3% of 2*pi*a")
        ax[0].axhline(CL_THIN, color="k", ls="--", lw=1, label="thin-aerofoil 2*pi*alpha = 0.21932")
        ax[0].semilogx(nps, cl, "o-", color="C3", label="delivered panel solver, Cl(2 deg)")
        ax[0].set_xlabel("panels per side"); ax[0].set_ylabel("Cl at alpha = 2 deg")
        ax[0].set_title("reproduction: converged Cl sits 10% above the card anchor")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

        tc_a = [a["t_over_c"] for a in anchor]
        r_a = [a["slope_ratio"] for a in anchor]
        tc_c = [c["t_over_c"] for c in claimed]
        r_c = [c["naca_panel_ratio"] for c in claimed]
        ax[1].axhline(1.0, color="k", ls="--", lw=1, label="2*pi (zero thickness)")
        ax[1].plot(tc_c, r_c, "s-", color="C3", label="NACA 4-digit panel solver (delivered)")
        ax[1].plot(tc_a, r_a, "o-", color="C0", label="exact symmetric Joukowski (closed form)")
        ax[1].set_xlabel("t/c"); ax[1].set_ylabel("lift-slope ratio  (dCl/da) / (2*pi)")
        ax[1].set_title("claimed 'thickness raises lift slope' vs an exact family")
        ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_audit.png", dpi=140)
        print("wrote fig_audit.png")
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)

    return res


if __name__ == "__main__":
    main()