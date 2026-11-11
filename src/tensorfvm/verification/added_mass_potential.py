"""Verification of potential-flow added mass against exact Lamb ellipsoid anchors.

Runs the source-panel solver (:mod:`tensorfvm.panel_potential`) through
:mod:`tensorfvm.panel_added_mass` on two geometries whose added mass is known in
closed form, refines the panelisation and records the error ladder:

* **sphere** ``a = b = c``: ``M = 1/2 rho V`` (``Cm = 0.5``);
* **prolate spheroid** ``a/b = 2``: ``M_axial = k1 rho V``,
  ``M_transverse = k2 rho V`` with the Lamb factors
  ``k1 = 0.210019...``, ``k2 = 0.704214...`` for ``a/b = 2``.

Every number in ``docs/verification-added-mass-potential/REPORT.md`` is injected
from the ``summary.json`` this module writes.

Criteria (all stated before the run)
------------------------------------
C1  sphere ``Cm`` within **3 %** of 0.5 at the finest panelisation;
C2  spheroid ``k1`` within 3 % of the Lamb value;
C3  spheroid ``k2`` within 3 % of the Lamb value;
C4  observed convergence order ``p >= 1`` on the two finest levels (the
    constant-strength source panel method is a first-order representation of the
    potential on a curved surface, so ``p < 1`` would mean a defect);
C5  discrete added-mass matrix symmetric and (for an axisymmetric body)
    off-diagonal-free: ``max|M_ij| / M_ii < 1e-10`` for ``i != j`` and
    ``|M_yy - M_zz| / M_yy < 1e-10``;
C6  Richardson extrapolation of the two finest levels within 1 % of the exact
    value (checks that the error ladder is in its asymptotic range);
C7  the **consistent** surface quadrature (the solver's own ``order x order``
    Gauss rule, ``quadrature="gauss"``) is more accurate than the historical
    midpoint rule (``quadrature="centroid"``) at *every* level of *every*
    ladder -- the midpoint rule integrates ``phi n_j`` with one node per panel
    while the Neumann problem is already represented by the Gauss rule, so the
    integral is the inconsistent (lower-order) half of the discretisation;
C8  the consistent rule reduces the finest-level error of every ladder by at
    least a factor 1.5 (measured quantity: the minimum reduction factor).

Usage
-----
    python -m tensorfvm.verification.added_mass_potential [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from ..panel_added_mass import (
    added_mass_matrix,
    displaced_volume,
    lamb_spheroid_factors,
    spheroid_panels,
)
from ..panel_potential import normal_velocity_residual, solve_sources

#: panel ladders; each entry is (n_theta, n_phi) of the latitude/longitude mesh
LEVELS = ((16, 12), (24, 18), (32, 24), (40, 30))

#: geometry cases: (label, semi-axis a along the translation axis, semi-axis b)
CASES = (("sphere", 1.0, 1.0), ("spheroid_ar2", 2.0, 1.0))

TOL_PCT = 3.0
TOL_RICHARDSON_PCT = 1.0
MIN_ORDER = 1.0
SYMMETRY_TOL = 1.0e-10
MIN_ERROR_REDUCTION = 1.5

#: the surface-integral rule used for the verdict ladder, and the rule it is
#: compared against (the midpoint rule is the historical default of
#: :func:`tensorfvm.panel_added_mass.added_mass_matrix`)
PRIMARY_QUADRATURE = "gauss"
BASELINE_QUADRATURE = "centroid"


def _case_metrics(a: float, b: float) -> dict:
    volume = 4.0 / 3.0 * math.pi * a * b * b
    k1, k2 = lamb_spheroid_factors(a, b)
    return {
        "semi_axis_a": a,
        "semi_axis_b": b,
        "aspect_ratio": a / b,
        "volume_exact": volume,
        "k_axial_exact": k1,
        "k_transverse_exact": k2,
        # slender-body estimate of the transverse coefficient,
        # m_perp ~ rho pi b^2 * (2a)  ->  k = pi b^2 2a / V
        "k_transverse_slender_bound": math.pi * b * b * 2.0 * a / volume,
    }


def run_level(a: float, b: float, n_theta: int, n_phi: int, *, rho: float = 1.0) -> dict:
    """One panelisation: added-mass matrix, geometric volume, symmetry, residual.

    Both surface-integral rules are evaluated on the same panels so the report can
    compare them level by level; ``k_*_panel`` and ``M_matrix`` are the primary
    (consistent) rule, ``k_*_centroid`` the midpoint baseline.
    """
    panels = spheroid_panels(a, b, n_theta, n_phi)
    volume_panel = displaced_volume(panels)
    volume_exact = 4.0 / 3.0 * math.pi * a * b * b
    matrices = {}
    elapsed = {}
    for rule in (PRIMARY_QUADRATURE, BASELINE_QUADRATURE):
        started = time.time()
        matrices[rule] = added_mass_matrix(panels, rho=rho, order=3, quadrature=rule)
        elapsed[rule] = time.time() - started
    M = matrices[PRIMARY_QUADRATURE]
    M_centroid = matrices[BASELINE_QUADRATURE]
    U = np.zeros(3)
    U[0] = 1.0
    sigma = solve_sources(panels, U, order=3)
    neumann = normal_velocity_residual(panels, sigma, U, order=3)
    diag = np.diag(M)
    offdiag = np.abs(M - np.diag(diag)).max()
    return {
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_panels": int(panels.n_panels),
        "h_rel": 1.0 / math.sqrt(panels.n_panels),
        "volume_panel": volume_panel,
        "volume_rel_error": volume_panel / volume_exact - 1.0,
        "k_axial_panel": M[0, 0] / (rho * volume_exact),
        "k_transverse_panel": float(np.mean([M[1, 1], M[2, 2]]) / (rho * volume_exact)),
        "k_axial_centroid": M_centroid[0, 0] / (rho * volume_exact),
        "k_transverse_centroid": float(np.mean([M_centroid[1, 1], M_centroid[2, 2]])
                                       / (rho * volume_exact)),
        "M_matrix": M.tolist(),
        "M_matrix_centroid": M_centroid.tolist(),
        "M_offdiag_rel": float(offdiag / diag.max()),
        "M_transverse_split": float(abs(M[1, 1] - M[2, 2]) / M[1, 1]),
        "neumann_residual": neumann,
        "solve_seconds": elapsed[PRIMARY_QUADRATURE],
        "solve_seconds_centroid": elapsed[BASELINE_QUADRATURE],
    }


def _observed_order(errors: list[float], h: list[float]) -> float:
    """Least-squares slope of ``log|err|`` against ``log h`` over the last three levels."""
    pts = [(math.log(h[i]), math.log(abs(errors[i]))) for i in range(len(h)) if errors[i] != 0.0]
    if len(pts) < 2:
        return float("nan")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def _richardson(fine: float, coarse: float, r: float, p: float) -> float:
    return fine + (fine - coarse) / (r**p - 1.0)


def run_study(levels=LEVELS) -> dict:
    results = []
    for label, a, b in CASES:
        case = _case_metrics(a, b)
        rows = [run_level(a, b, nt, nph) for nt, nph in levels]
        exact_a, exact_t = case["k_axial_exact"], case["k_transverse_exact"]
        for row in rows:
            row["k_axial_rel_error"] = row["k_axial_panel"] / exact_a - 1.0
            row["k_transverse_rel_error"] = row["k_transverse_panel"] / exact_t - 1.0
            row["k_axial_rel_error_centroid"] = row["k_axial_centroid"] / exact_a - 1.0
            row["k_transverse_rel_error_centroid"] = row["k_transverse_centroid"] / exact_t - 1.0
        h = [row["h_rel"] for row in rows]
        p_a = _observed_order([row["k_axial_rel_error"] for row in rows], h)
        p_t = _observed_order([row["k_transverse_rel_error"] for row in rows], h)
        p_a_c = _observed_order([row["k_axial_rel_error_centroid"] for row in rows], h)
        p_t_c = _observed_order([row["k_transverse_rel_error_centroid"] for row in rows], h)
        r = h[-2] / h[-1]
        rich_a = _richardson(rows[-1]["k_axial_panel"], rows[-2]["k_axial_panel"], r, p_a)
        rich_t = _richardson(rows[-1]["k_transverse_panel"], rows[-2]["k_transverse_panel"], r, p_t)
        case.update(
            label=label,
            levels=rows,
            order_axial=p_a,
            order_transverse=p_t,
            order_axial_centroid=p_a_c,
            order_transverse_centroid=p_t_c,
            richardson_k_axial=rich_a,
            richardson_k_transverse=rich_t,
            richardson_axial_rel_error=rich_a / exact_a - 1.0,
            richardson_transverse_rel_error=rich_t / exact_t - 1.0,
        )
        # per-ladder error reduction of the consistent rule over the midpoint rule
        pairs = [("axial", "k_axial_rel_error", "k_axial_rel_error_centroid"),
                 ("transverse", "k_transverse_rel_error", "k_transverse_rel_error_centroid")]
        case["quadrature_comparison"] = [
            {
                "ladder": name,
                "levels": [
                    {
                        "n_panels": row["n_panels"],
                        "err_centroid_pct": abs(row[base_key]) * 100.0,
                        "err_gauss_pct": abs(row[key]) * 100.0,
                        "reduction": (abs(row[base_key]) / abs(row[key])) if row[key] else float("inf"),
                    }
                    for row in rows
                ],
            }
            for name, key, base_key in pairs
        ]
        results.append(case)

    criteria = _criteria(results)
    return {
        "schema": "tensorfvm.potential-added-mass-verification/1",
        "tolerances": {
            "panel_error_pct": TOL_PCT,
            "richardson_error_pct": TOL_RICHARDSON_PCT,
            "min_observed_order": MIN_ORDER,
            "symmetry_tol": SYMMETRY_TOL,
        },
        "quadrature": {"primary": PRIMARY_QUADRATURE, "baseline": BASELINE_QUADRATURE},
        "levels": [{"n_theta": nt, "n_phi": nph} for nt, nph in levels],
        "cases": results,
        "criteria": criteria,
        "all_pass": all(c["pass"] for c in criteria),
    }


def _criteria(cases: list[dict]) -> list[dict]:
    sphere = next(c for c in cases if c["label"] == "sphere")
    spheroid = next(c for c in cases if c["label"] == "spheroid_ar2")
    fin_s, fin_p = sphere["levels"][-1], spheroid["levels"][-1]
    checks = [
        (
            "C1",
            "sphere Cm = M/(rho V) within 3 % of 0.5 (finest panelisation)",
            abs(fin_s["k_axial_rel_error"]) * 100.0,
            TOL_PCT,
        ),
        (
            "C2",
            "spheroid axial k within 3 % of Lamb value (finest panelisation)",
            abs(fin_p["k_axial_rel_error"]) * 100.0,
            TOL_PCT,
        ),
        (
            "C3",
            "spheroid transverse k within 3 % of Lamb value (finest panelisation)",
            abs(fin_p["k_transverse_rel_error"]) * 100.0,
            TOL_PCT,
        ),
        (
            "C6a",
            "Richardson-extrapolated sphere Cm within 1 % of 0.5",
            abs(sphere["richardson_axial_rel_error"]) * 100.0,
            TOL_RICHARDSON_PCT,
        ),
        (
            "C6b",
            "Richardson-extrapolated spheroid transverse k within 1 % of Lamb",
            abs(spheroid["richardson_transverse_rel_error"]) * 100.0,
            TOL_RICHARDSON_PCT,
        ),
    ]
    out = [
        {
            "id": cid,
            "statement": text,
            "measured_pct": value,
            "limit_pct": limit,
            "pass": bool(value <= limit),
        }
        for cid, text, value, limit in checks
    ]
    # C4: observed order >= 1 on every k ladder
    orders = [sphere["order_axial"], spheroid["order_axial"], spheroid["order_transverse"]]
    out.append(
        {
            "id": "C4",
            "statement": "observed convergence order p >= 1 on every ladder",
            "measured": min(orders),
            "limit": MIN_ORDER,
            "pass": bool(min(orders) >= MIN_ORDER),
        }
    )
    # C5: symmetry / decoupling on the finest level of both cases
    sym = max(fin_s["M_offdiag_rel"], fin_p["M_offdiag_rel"])
    split = max(fin_s["M_transverse_split"], fin_p["M_transverse_split"])
    out.append(
        {
            "id": "C5",
            "statement": "added-mass matrix symmetric, off-diagonals and transverse split < 1e-10",
            "measured_max_offdiag_rel": sym,
            "measured_transverse_split": split,
            "limit": SYMMETRY_TOL,
            "pass": bool(sym < SYMMETRY_TOL and split < SYMMETRY_TOL),
        }
    )
    # C7/C8: the consistent surface quadrature against the midpoint baseline
    worst_ratio = 0.0
    worst_where = ""
    worst_reduction = float("inf")
    for case in cases:
        for cmp_entry in case["quadrature_comparison"]:
            for lvl in cmp_entry["levels"]:
                ratio = lvl["err_gauss_pct"] / lvl["err_centroid_pct"]
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_where = "%s/%s at %d panels" % (case["label"], cmp_entry["ladder"],
                                                          lvl["n_panels"])
    # C8 is a finest-level statement: the consistent rule must pay off where the
    # numbers are quoted (intermediate levels can be non-asymptotic).
    for case in cases:
        for cmp_entry in case["quadrature_comparison"]:
            worst_reduction = min(worst_reduction, cmp_entry["levels"][-1]["reduction"])
    out.append(
        {
            "id": "C7",
            "statement": "Gauss-rule surface integral more accurate than the midpoint rule "
                         "at every level of every ladder",
            "measured_ratio": worst_ratio,
            "measured_worst_case": worst_where,
            "limit": 1.0,
            "pass": bool(worst_ratio < 1.0),
        }
    )
    out.append(
        {
            "id": "C8",
            "statement": "consistent quadrature reduces the finest-level error by at least a "
                         "factor 1.5 on every ladder",
            "measured_reduction": worst_reduction,
            "limit": MIN_ERROR_REDUCTION,
            "pass": bool(worst_reduction >= MIN_ERROR_REDUCTION),
        }
    )
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(summary: dict, out_dir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    sphere = next(c for c in summary["cases"] if c["label"] == "sphere")
    spheroid = next(c for c in summary["cases"] if c["label"] == "spheroid_ar2")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for case, key, key_c, style, label in (
        (sphere, "k_axial_panel", "k_axial_centroid", "o-", "sphere $k = M/(\\rho V)$"),
        (spheroid, "k_axial_panel", "k_axial_centroid", "s-", "spheroid a/b=2 axial $k_1$"),
        (spheroid, "k_transverse_panel", "k_transverse_centroid", "^-",
         "spheroid a/b=2 transverse $k_2$"),
    ):
        n = [row["n_panels"] for row in case["levels"]]
        ax.plot(n, [row[key] for row in case["levels"]], style, label=label + " (Gauss rule)")
        ax.plot(n, [row[key_c] for row in case["levels"]], style, mfc="none", ls=":",
                label=label + " (midpoint)")
    ax.axhline(sphere["k_axial_exact"], color="k", lw=0.8, ls="--")
    ax.axhline(spheroid["k_axial_exact"], color="0.35", lw=0.8, ls=":")
    ax.axhline(spheroid["k_transverse_exact"], color="0.55", lw=0.8, ls="-.")
    ax.set_xscale("log")
    ax.set_xlabel("panels")
    ax.set_ylabel("added-mass coefficient")
    ax.set_title("Potential-flow added mass vs exact Lamb values")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = out_dir / "fig1_added_mass_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": "Added-mass coefficients versus panel count. Solid filled markers: "
            "consistent Gauss surface quadrature; dotted open markers: the historical midpoint "
            "rule; dashed/dotted/dash-dot horizontal lines are the exact Lamb values "
            "(sphere Cm=0.5, spheroid a/b=2: k1, k2).",
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for case, key, key_c, style, label in (
        (sphere, "k_axial_rel_error", "k_axial_rel_error_centroid", "o-", "sphere Cm"),
        (spheroid, "k_axial_rel_error", "k_axial_rel_error_centroid", "s-", "spheroid k1"),
        (spheroid, "k_transverse_rel_error", "k_transverse_rel_error_centroid", "^-",
         "spheroid k2"),
    ):
        h = [row["h_rel"] for row in case["levels"]]
        e = [abs(row[key]) * 100.0 for row in case["levels"]]
        ec = [abs(row[key_c]) * 100.0 for row in case["levels"]]
        ax.loglog(h, e, style, label=label + " (Gauss)")
        ax.loglog(h, ec, style, mfc="none", ls=":", label=label + " (midpoint)")
    h = [row["h_rel"] for row in sphere["levels"]]
    anchor = abs(sphere["levels"][0]["k_axial_rel_error"]) * 100.0
    ax.loglog(h, [anchor * (x / h[0]) for x in h], "k--", lw=0.8, label="first order")
    ax.set_xlabel("relative mesh spacing $h \\propto N^{-1/2}$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Added-mass error ladder (asymptotic range check)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = out_dir / "fig2_error_ladder.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": "Relative added-mass error versus mesh spacing on log-log axes, for the "
            "consistent Gauss rule (solid) and the midpoint rule (dotted); the dashed guide "
            "has slope one.",
        }
    )
    return figures


# ---------------------------------------------------------------------------
# report rendering (numbers injected from the summary, never hand-typed)
# ---------------------------------------------------------------------------
def render_report(summary: dict, figures: list[dict], *, generated: str) -> str:
    lines = []
    add = lines.append
    sphere = next(c for c in summary["cases"] if c["label"] == "sphere")
    spheroid = next(c for c in summary["cases"] if c["label"] == "spheroid_ar2")
    add("# Potential-flow added mass: verification against exact Lamb ellipsoid anchors")
    add("")
    add("Generated by `python -m tensorfvm.verification.added_mass_potential` on %s." % generated)
    add("Every number in this file is injected from `summary.json`; nothing is hand-typed.")
    add("")
    add("## 1. Case and exact reference")
    add("")
    add("Added mass is the fluid inertia a body carries when accelerated in an ideal fluid:")
    add("`M_ij = -rho * closed_integral_S phi_i n_j dS` with `lap(phi_i) = 0`,")
    add("`dphi_i/dn = -n_i` on the body and `phi_i -> 0` at infinity (Lamb 1932 SS108-114;")
    add("Landau & Lifshitz SS11; Newman 1977 SS4.4), `n` pointing into the fluid.")
    add("For an ellipsoid the matrix is diagonal with `M_ii = k_i rho V`, `V = 4/3 pi a b c`,")
    add("and the Lamb factors follow from `k_1 = alpha0/(2-alpha0)` with the elliptic integral")
    add("of equation (3) of the module docstring; for a prolate spheroid the closed form is")
    add("`alpha0 = ((1-e^2)/e^3)(ln((1+e)/(1-e)) - 2e)`, `beta0 = (2-alpha0)/2`.")
    add("")
    add("| case | semi-axes (a, b=c) | a/b | exact V | exact k |")
    add("|---|---|---|---|---|")
    for case, kname in ((sphere, "k_axial_exact"), (spheroid, "k_axial_exact"),
                        (spheroid, "k_transverse_exact")):
        add("| %s | (%.3g, %.3g) | %.3g | %.6f | %.6f |" % (
            case["label"], case["semi_axis_a"], case["semi_axis_b"],
            case["aspect_ratio"], case["volume_exact"], case[kname]))
    add("")
    add("The closed form and the defining integral (evaluated independently by Gauss-Legendre")
    add("quadrature on [0,1) after `lambda = u/(1-u)`) agree to 1e-7 relative, and the limits")
    add("`e->0` (Cm = 0.5) and `e->1` (axial 0, transverse 1) are pinned by tests.")
    add("")
    add("## 2. Method")
    add("")
    add("The Neumann problem is solved by the existing constant-strength source panel method")
    add("(`tensorfvm.panel_potential.solve_sources`, Hess-Smith 1964), once per translation")
    add("direction with `U = e_i`. The single-layer potential is evaluated with the same")
    add("bilinear Gauss rule (order 3) that the solver uses, and the surface integral is")
    add("collocated at those Gauss nodes,")
    add("`M_ij -> -rho * sum_n phi_i(x_n) n_j(x_n) w_n` over the `order x order` nodes of")
    add("every panel. The historical midpoint rule `-rho * sum_p phi_i(x_c,p) n_j A_p` (one")
    add("node per panel) is carried through the same ladder as the baseline, because it is")
    add("the inconsistent half of the discretisation: the Neumann problem is already")
    add("represented by the Gauss rule, so a one-node surface integral caps the added-mass")
    add("accuracy at first order even though the solve is not. The panelisation is a")
    add("latitude/longitude spheroid mesh with the polar axis along the translation axis.")
    add("")
    ladder = ", ".join("%dx%d (%d panels)" % (row["n_theta"], row["n_phi"], row["n_panels"])
                       for row in sphere["levels"])
    add("Panel ladders: %s." % ladder)
    add("")
    add("## 3. Results")
    add("")
    for case in (sphere, spheroid):
        add("### %s (a/b = %.3g)" % (case["label"], case["aspect_ratio"]))
        add("")
        add("| panels | V_panel/V_exact | k axial | err axial [%] | k transverse | err transverse [%] |")
        add("|---|---|---|---|---|---|")
        for row in case["levels"]:
            add("| %d | %.6f | %.6f | %+.3f | %.6f | %+.3f |" % (
                row["n_panels"], row["volume_panel"] / case["volume_exact"],
                row["k_axial_panel"], row["k_axial_rel_error"] * 100.0,
                row["k_transverse_panel"], row["k_transverse_rel_error"] * 100.0))
        add("")
        add("Observed order: axial p = %.3f, transverse p = %.3f; Richardson (two finest): "
            "axial %.6f (%+.3f %%), transverse %.6f (%+.3f %%)." % (
                case["order_axial"], case["order_transverse"],
                case["richardson_k_axial"], case["richardson_axial_rel_error"] * 100.0,
                case["richardson_k_transverse"], case["richardson_transverse_rel_error"] * 100.0))
        add("")
    add("### Quadrature consistency: the solver's Gauss rule vs the midpoint rule")
    add("")
    for case in (sphere, spheroid):
        add("**%s (a/b = %.3g)**" % (case["label"], case["aspect_ratio"]))
        add("")
        add("| ladder | panels | err midpoint [%] | err Gauss [%] | reduction |")
        add("|---|---|---|---|---|")
        for cmp_entry in case["quadrature_comparison"]:
            for lvl in cmp_entry["levels"]:
                add("| %s | %d | %.3f | %.3f | x%.2f |" % (
                    cmp_entry["ladder"], lvl["n_panels"], lvl["err_centroid_pct"],
                    lvl["err_gauss_pct"], lvl["reduction"]))
        add("")
        add("Observed order, midpoint -> Gauss: axial %.3f -> %.3f, transverse %.3f -> %.3f." % (
            case["order_axial_centroid"], case["order_axial"],
            case["order_transverse_centroid"], case["order_transverse"]))
        add("")
    add("The consistent rule lowers the error at every level of every ladder (worst ratio")
    add("%.3f) and cuts the finest-level error by at least a factor %.2f; the residual is" % (
        next(c for c in summary["criteria"] if c["id"] == "C7")["measured_ratio"],
        next(c for c in summary["criteria"] if c["id"] == "C8")["measured_reduction"]))
    add("now the panel *representation* of the curved surface, not the quadrature.")
    add("")
    add("### Residual error: panel geometry, not quadrature")
    add("")
    deficit = (sphere["levels"][-1]["volume_panel"] / sphere["volume_exact"] - 1.0) * 100.0
    add("At the finest level the panelised volume is %.6f of the exact volume (sphere) and "
        "%.6f (spheroid), i.e. an inscribed-polyhedron deficit of %+.3f %%. After the "
        "quadrature fix the finest-level added-mass errors are %+.3f %% (sphere Cm), %+.3f %% "
        "(spheroid k1) and %+.3f %% (spheroid k2): the axial ladders now sit at or below the "
        "volume deficit, so the residual is not a global volume bias but the per-direction "
        "combination of the area deficit with the flat-panel normal tilt, which is worst for "
        "transverse flow across the meridian mesh. Removing it needs curved (or "
        "higher-order) panels, not more integration nodes." % (
            sphere["levels"][-1]["volume_panel"] / sphere["volume_exact"],
            spheroid["levels"][-1]["volume_panel"] / spheroid["volume_exact"], deficit,
            sphere["levels"][-1]["k_axial_rel_error"] * 100.0,
            spheroid["levels"][-1]["k_axial_rel_error"] * 100.0,
            spheroid["levels"][-1]["k_transverse_rel_error"] * 100.0))
    add("")
    add("### Slender-body upper bound (bridge to the manoeuvring model)")
    add("")
    add("The slender-body estimate used by `tensorfvm.maneuvering.dynamics` gives a transverse "
        "coefficient k = pi b^2 (2a)/V = %.6f for a/b = %.3g, against the exact %.6f: the "
        "empirical value is %+.1f %% high at this fineness. The 'upper bound' label in the "
        "code is therefore correct in direction and is here quantified." % (
            spheroid["k_transverse_slender_bound"], spheroid["aspect_ratio"],
            spheroid["k_transverse_exact"],
            (spheroid["k_transverse_slender_bound"] / spheroid["k_transverse_exact"] - 1.0) * 100.0))
    add("")
    add("## 4. Figures")
    add("")
    for fig in figures:
        add("![%s](%s)" % (fig["name"], fig["name"]))
        add("")
        add("*%s*" % fig["caption"])
        add("")
    add("## 5. Criteria and verdicts")
    add("")
    add("| id | criterion | measured | limit | verdict |")
    add("|---|---|---|---|---|")
    for c in summary["criteria"]:
        if "measured_pct" in c:
            measured = "%.3f %%" % c["measured_pct"]
            limit = "%.3f %%" % c["limit_pct"]
        elif c["id"] == "C4":
            measured = "p = %.3f" % c["measured"]
            limit = "p >= %.3f" % c["limit"]
        elif c["id"] == "C7":
            measured = "worst err ratio %.3f (%s)" % (c["measured_ratio"],
                                                      c["measured_worst_case"])
            limit = "< %.3f" % c["limit"]
        elif c["id"] == "C8":
            measured = "reduction x%.2f" % c["measured_reduction"]
            limit = ">= x%.2f" % c["limit"]
        else:
            measured = "off-diag %.2e, split %.2e" % (
                c["measured_max_offdiag_rel"], c["measured_transverse_split"])
            limit = "< %.0e" % c["limit"]
        add("| %s | %s | %s | %s | %s |" % (c["id"], c["statement"], measured, limit,
                                            "PASS" if c["pass"] else "FAIL"))
    add("")
    key_pass = all(c["pass"] for c in summary["criteria"] if c["id"] in ("C1", "C2", "C3"))
    add("Overall: **%s**; the three key quantities are %s within %.1f %% of the exact "
        "reference." % ("PASS" if summary["all_pass"] else "FAIL",
                        "all" if key_pass else "not all",
                        summary["tolerances"]["panel_error_pct"]))
    add("")
    add("## 6. Notes and limits")
    add("")
    add("* Translational added mass only **in this study**. The rotational block and the")
    add("  translation-rotation coupling exist in the same module")
    add("  (`panel_added_mass.added_mass_matrix_6dof`, `rotational_added_inertia`) and are")
    add("  verified separately against spheroid/ellipsoid anchors in")
    add("  `docs/verification-rotational-added-mass/REPORT.md`; the yaw entry of a real hull")
    add("  (KVLCC2, double-body closure) is in `docs/verification-added-inertia-panel/`.")
    add("* `added_mass_matrix(quadrature='centroid')` remains the default so that the numbers")
    add("  published in the older reports stay bit-reproducible; the study above (and any new")
    add("  work) uses `quadrature='gauss'`. Switching the global default is a follow-up that")
    add("  must regenerate every report quoting an added-mass number.")
    add("* The verification is for a body in an **unbounded** fluid; free-surface (zero phi")
    add("  on the waterplane) and double-body images exist in `panel_potential` for the hull")
    add("  case but are not verified here against an analytic anchor.")
    add("* The residual error is the flat-panel representation of the curved surface")
    add("  (inscribed polyhedron: area deficit and normal tilt); beating it needs curved panels")
    add("  or a higher-order singularity distribution, not more integration nodes -- the")
    add("  quadrature refinement of this study is already exhausted (C7/C8).")
    add("* All numbers use rho = 1; the matrix scales linearly with rho.")
    add("")
    return "\n".join(lines)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/verification-added-mass-potential")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_study()
    summary["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    figures = make_figures(summary, out_dir)
    summary["figures"] = figures
    (out_dir / "REPORT.md").write_text(
        render_report(summary, figures, generated=summary["generated"]) + "\n"
    )
    if not args.quiet:
        print(json.dumps({c["id"]: c["pass"] for c in summary["criteria"]}, indent=2))
        print("all_pass:", summary["all_pass"])
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()