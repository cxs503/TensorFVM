#!/usr/bin/env python3
"""Source-panel added mass on the Lamb ellipsoid family: anchors, scan, convergence.

What this benchmark answers
---------------------------
The constant-strength source panel solver of
:mod:`tensorfvm.panel_potential` drives the potential-flow added-mass route of
:mod:`tensorfvm.panel_added_mass` (``M_ij = -rho * integral_S phi_i n_j dS`` with
``dphi_i/dn = -n_i`` on the body and ``phi_i -> 0`` at infinity).  This benchmark
puts that solver against the one family whose answer is known exactly -- the
triaxial ellipsoid of Lamb (1932) sec. 108-114 -- and asks:

1. does the analytic anchor reproduce the known limits (sphere ``Cm = 0.5``,
   slender prolate ``k_axial -> 0`` / ``k_transverse -> 1``, flat oblate disc
   ``M_33 -> 8/3 rho a^3``)?
2. for eight bodies (sphere, prolate, oblate, triaxial) and a ladder of
   panelisations, what is the relative deviation of every diagonal coefficient
   from the anchor, and is it inside the ``+-5 %`` band?
3. what is the observed order of convergence and the Grid Convergence Index
   (Roache 1994) of the finest pair of meshes?
4. which case is worst, and is the residual error attributable to surface
   faceting (mesh quality) or to the self-panel (singular) quadrature?

Nothing is random or data-dependent: geometry, meshes, quadrature and the linear
solves are deterministic, so a re-run reproduces the numbers apart from the
wall-clock timings.

Usage
-----
    python benchmarks/added_mass_lamb/run_benchmark.py            # full ladder
    python benchmarks/added_mass_lamb/run_benchmark.py --quick    # smoke ladder

Artefacts (written into ``--out``, default: the directory of this script):

    anchors.json          Lamb anchor + limit verification            (criterion 1)
    ellipsoid_scan.csv    one row per (case, level, axis)             (criteria 2, 5)
    convergence.json      order, Richardson value, GCI, worst case    (criteria 3, 4)
    README.md             the report, tables injected programmatically
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:                      # run from a bare checkout
    sys.path.insert(0, str(_SRC))
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tensorfvm.panel_added_mass import (  # noqa: E402
    added_mass_matrix,
    displaced_volume,
    lamb_ellipsoid_factors,
    lamb_integral_factors,
    sphere_added_mass,
)
from tensorfvm.verification.added_mass_triaxial import ellipsoid_panels  # noqa: E402

RHO = 1.0

# (label, a, b, c, family) -- eight bodies, seven distinct aspect-ratio classes,
# spanning the sphere, both spheroid limits and two genuinely triaxial shapes.
CASES = (
    ("sphere_1_1_1", 1.0, 1.0, 1.0, "sphere"),
    ("prolate_2_1_1", 2.0, 1.0, 1.0, "prolate"),
    ("prolate_3_1_1", 3.0, 1.0, 1.0, "prolate"),
    ("prolate_5_1_1", 5.0, 1.0, 1.0, "prolate-slender"),
    ("oblate_1_1_0.5", 1.0, 1.0, 0.5, "oblate"),
    ("oblate_1_1_0.25", 1.0, 1.0, 0.25, "oblate-flat"),
    ("triaxial_2_1_0.5", 2.0, 1.0, 0.5, "triaxial"),
    ("triaxial_1.5_1_0.75", 1.5, 1.0, 0.75, "triaxial"),
)

# (n_theta, n_phi) = (longitudes, latitudes); five levels (task needs >= 3)
LEVELS = ((16, 12), (24, 18), (32, 24), (48, 36), (64, 48))
QUICK_LEVELS = ((12, 8), (16, 12), (24, 18))

TOL_PCT = 5.0            # criterion 2: +-5 % on the diagonal coefficients
GCI_TOL_PCT = 5.0        # criterion 3: GCI < 5 %
SPHERE_TOL_PCT = 0.1     # criterion 1: sphere Cm deviation < 0.1 %
FS = 1.25                # Roache safety factor for a two-grid GCI

# quadrature / self-influence study: same bodies and meshes, both surface rules
QUAD_STUDY_CASES = ("sphere_1_1_1", "triaxial_2_1_0.5", "oblate_1_1_0.25")
QUAD_STUDY_LEVELS = ((32, 24), (48, 36))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                   # pragma: no cover
        return "unknown"


def axis_extents(panels) -> list[float]:
    """``max |x|, max |y|, max |z|`` over the panel corners (geometry check)."""
    corners = panels.corners
    if corners is None:                                 # pragma: no cover
        return [float("nan")] * 3
    pts = corners.reshape(-1, 3)
    return [float(np.abs(pts[:, k]).max()) for k in range(3)]


def panel_quality(panels) -> dict:
    """Area spread and worst edge-length aspect ratio of the panel mesh.

    The latitude/longitude panelisation is anisotropic: rows crowd towards the
    two poles, so ``area_max/area_min`` and the worst quad aspect ratio grow with
    the level and with the body's flattening.  These are the mesh-quality numbers
    the worst-case attribution is tested against.
    """
    areas = np.asarray(panels.areas, dtype=float)
    corners = panels.corners
    if corners is None:                                 # pragma: no cover
        return {"area_max_over_min": float("nan"),
                "edge_over_sqrt_area_max": float("nan")}
    edges = np.stack([np.linalg.norm(corners[:, (k + 1) % 4] - corners[:, k], axis=1)
                      for k in range(4)], axis=1)
    # The latitude rows at the two poles are degenerate quads (two corners
    # coincide, the quad is a triangle), so an edge-length *ratio* is unbounded
    # there.  ``max_edge / sqrt(area)`` is defined for both triangles and quads
    # and equals 1 for a square, so it is used as the shape metric.
    live = areas > 1e-15
    shape = edges[live].max(axis=1) / np.sqrt(areas[live])
    return {
        "area_max_over_min": float(areas.max() / areas[live].min()),
        "edge_over_sqrt_area_max": float(shape.max()),
    }


def fit_order(errors, h) -> float:
    """Least-squares slope of ``log|error|`` against ``log h``."""
    pts = [(math.log(hh), math.log(abs(ee))) for ee, hh in zip(errors, h) if ee != 0.0]
    if len(pts) < 2:
        return float("nan")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def richardson(fine: float, coarse: float, ratio: float, order: float) -> float:
    return fine + (fine - coarse) / (ratio**order - 1.0)


# ---------------------------------------------------------------------------
# 1. anchors (AN-LAMB-ELLIPSOID)
# ---------------------------------------------------------------------------
def anchor_case(a: float, b: float, c: float) -> dict:
    """Exact Lamb factors for one body, from both in-repo routes."""
    factors = lamb_ellipsoid_factors(a, b, c)
    integral = lamb_integral_factors(a, b, c)
    volume = 4.0 / 3.0 * math.pi * a * b * c
    k = [f / (2.0 - f) for f in factors]
    return {
        "semi_axes": [a, b, c],
        "volume": volume,
        "alpha0_beta0_gamma0": list(factors),
        "alpha0_beta0_gamma0_integral_route": list(integral),
        "route_max_abs_diff": float(max(abs(x - y) for x, y in zip(factors, integral))),
        "sum_rule_abs_dev": float(abs(sum(factors) - 2.0)),
        "k": k,
        "M_diag": [kk * RHO * volume for kk in k],
    }


def limit_ladders() -> list[dict]:
    """The three known limits of the Lamb family, as parameter ladders.

    * sphere           -- ``k = 1/2`` on every axis;
    * slender prolate  -- ``k_axial -> 0``, ``k_transverse -> 1`` as ``b/a -> 0``;
    * flat oblate disc -- ``gamma0 -> 2``, so ``k_3`` diverges at fixed displaced
      volume while the physical added mass culminates in the classical disc
      value ``M_33 -> 8/3 rho a^3`` (Lamb sec. 108, circular disc).
    """
    rows: list[dict] = []

    sphere = anchor_case(1.0, 1.0, 1.0)
    sphere_dev = [abs(kk / 0.5 - 1.0) * 100.0 for kk in sphere["k"]]
    rows.append({
        "limit": "sphere",
        "parameter": "a = b = c = 1",
        "quantity": "k_1, k_2, k_3 (= Cm)",
        "value": sphere["k"],
        "reference": 0.5,
        "worst_deviation_pct": max(sphere_dev),
        "tolerance_pct": SPHERE_TOL_PCT,
        "verdict": "PASS" if max(sphere_dev) <= SPHERE_TOL_PCT else "FAIL",
    })
    m_ref = sphere_added_mass(1.0, RHO)
    m_dev = abs(sphere["M_diag"][0] / m_ref - 1.0) * 100.0
    rows.append({
        "limit": "sphere",
        "parameter": "a = b = c = 1",
        "quantity": "M_11 vs 0.5 rho (4/3) pi a^3",
        "value": [sphere["M_diag"][0]],
        "reference": m_ref,
        "worst_deviation_pct": m_dev,
        "tolerance_pct": SPHERE_TOL_PCT,
        "verdict": "PASS" if m_dev <= SPHERE_TOL_PCT else "FAIL",
    })

    prolate_ladder = []
    for minor in (0.5, 0.2, 0.1, 0.05, 0.01):
        entry = anchor_case(1.0, minor, minor)
        prolate_ladder.append({
            "b_over_a": minor,
            "eccentricity": math.sqrt(1.0 - minor**2),
            "k_axial": entry["k"][0],
            "k_transverse": entry["k"][1],
            "transverse_gap_from_1": 1.0 - entry["k"][1],
        })
    decreasing_axial = all(x["k_axial"] > y["k_axial"]
                           for x, y in zip(prolate_ladder, prolate_ladder[1:]))
    rising_transverse = all(x["k_transverse"] < y["k_transverse"]
                            for x, y in zip(prolate_ladder, prolate_ladder[1:]))
    rows.append({
        "limit": "slender prolate",
        "parameter": "a = 1, b = c in {0.5, 0.2, 0.1, 0.05, 0.01}",
        "quantity": "k_axial -> 0, k_transverse -> 1",
        "value": prolate_ladder,
        "reference": [0.0, 1.0],
        "final_gap": [prolate_ladder[-1]["k_axial"],
                      prolate_ladder[-1]["transverse_gap_from_1"]],
        "tolerance_pct": None,
        "verdict": "PASS" if decreasing_axial and rising_transverse
        and prolate_ladder[-1]["k_axial"] < 0.01
        and prolate_ladder[-1]["transverse_gap_from_1"] < 0.01 else "FAIL",
    })

    disc = 8.0 / 3.0 * RHO * 1.0**3
    oblate_ladder = []
    for flat in (0.5, 0.25, 0.1, 0.01, 0.001):
        entry = anchor_case(1.0, 1.0, flat)
        factors = entry["alpha0_beta0_gamma0"]
        oblate_ladder.append({
            "c_over_a": flat,
            "gamma0": factors[2],
            "gamma0_gap_from_2": 2.0 - factors[2],
            "k_normal": entry["k"][2],
            "k_in_plane": entry["k"][0],
            "M_normal": entry["k"][2] * RHO * entry["volume"],
            "disc_reference_M": disc,
            "M_normal_over_disc": entry["k"][2] * RHO * entry["volume"] / disc,
        })
    rows.append({
        "limit": "flat oblate disc",
        "parameter": "a = b = 1, c in {0.5, 0.25, 0.1, 0.01, 0.001}",
        "quantity": "M_33/(8/3 rho a^3) -> 1, k_in_plane -> 0, gamma0 -> 2",
        "value": oblate_ladder,
        "reference": 1.0,
        "final_gap": [abs(oblate_ladder[-1]["M_normal_over_disc"] - 1.0),
                      oblate_ladder[-1]["k_in_plane"]],
        "tolerance_pct": None,
        "verdict": "PASS" if abs(oblate_ladder[-1]["M_normal_over_disc"] - 1.0) < 0.01
        and oblate_ladder[-1]["k_in_plane"] < 0.01 else "FAIL",
    })
    return rows


def build_anchors() -> dict:
    cases = []
    for label, a, b, c, family in CASES:
        entry = anchor_case(a, b, c)
        entry.update({
            "label": label,
            "family": family,
            "aspect_ratios": {"a/b": a / b, "a/c": a / c, "b/c": b / c},
        })
        cases.append(entry)
    limits = limit_ladders()
    return {
        "anchor_id": "AN-LAMB-ELLIPSOID",
        "generated_utc": utc_now(),
        "git_commit": git_commit(),
        "reference": {
            "primary": "Lamb, H. (1932) Hydrodynamics, 6th ed., sec. 108-114 "
                       "(ellipsoid added-mass coefficients alpha0/beta0/gamma0)",
            "cross_checks": [
                "Landau & Lifshitz (1987) Fluid Mechanics sec. 11 (definition of added mass)",
                "Newman (1977) Marine Hydrodynamics sec. 4.4 (k_i = alpha0/(2-alpha0))",
            ],
            "catalogue_lookup": "ship API :8002 /anchors/AN-LAMB-ELLIPSOID returns 404 "
                                "(checked 2026-11-12), so the anchor numbers are regenerated "
                                "from the two in-repo routes instead of transcribed",
        },
        "implementation": {
            "closed_form_route": "tensorfvm.panel_added_mass.lamb_ellipsoid_factors "
                                 "(sphere / prolate eq.4 / oblate eq.5)",
            "integral_route": "tensorfvm.panel_added_mass.lamb_integral_factors "
                              "(defining integral 3, log-mapped Gauss-Legendre)",
            "added_mass_definition": "M_ij = -rho * integral_S phi_i n_j dS, "
                                     "dphi_i/dn = -n_i, phi_i -> 0 at infinity",
        },
        "limit_checks": limits,
        "cases": cases,
        "route_agreement_max_abs": float(max(c["route_max_abs_diff"] for c in cases)),
        "sum_rule_max_abs_dev": float(max(c["sum_rule_abs_dev"] for c in cases)),
        "criterion_1_verdict": "PASS" if all(r["verdict"] == "PASS" for r in limits) else "FAIL",
    }


# ---------------------------------------------------------------------------
# 2. panel scan
# ---------------------------------------------------------------------------
def run_level(label: str, family: str, a: float, b: float, c: float,
              n_theta: int, n_phi: int, quadrature: str = "gauss") -> dict:
    panels = ellipsoid_panels(a, b, c, n_theta, n_phi)
    volume_exact = 4.0 / 3.0 * math.pi * a * b * c
    started = time.time()
    matrix = added_mass_matrix(panels, rho=RHO, order=3, quadrature=quadrature)
    elapsed = time.time() - started
    volume_panel = displaced_volume(panels)
    extents = axis_extents(panels)
    diag = np.diag(matrix)
    offdiag = float(np.abs(matrix - np.diag(diag)).max())
    return {
        "label": label,
        "family": family,
        "semi_axes": [a, b, c],
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_panels": int(panels.n_panels),
        "h_rel": 1.0 / math.sqrt(panels.n_panels),
        "quadrature": quadrature,
        "volume_panel": volume_panel,
        "volume_rel_error": volume_panel / volume_exact - 1.0,
        "axis_extent_rel_error": [ext / ax - 1.0 for ext, ax in zip(extents, (a, b, c))],
        "k_panel": [float(x) / (RHO * volume_exact) for x in diag],
        "M_diag": [float(x) for x in diag],
        "offdiag_rel": offdiag / float(np.abs(diag).max()),
        "solve_seconds": elapsed,
        **panel_quality(panels),
    }


def run_scan(levels) -> tuple[list[dict], list[dict]]:
    """Run every (case, level); flatten to one row per (case, level, axis)."""
    per_level: list[dict] = []
    csv_rows: list[dict] = []
    for label, a, b, c, family in CASES:
        k_exact = anchor_case(a, b, c)["k"]
        for (n_theta, n_phi) in levels:
            row = run_level(label, family, a, b, c, n_theta, n_phi)
            row["k_exact"] = k_exact
            row["rel_error"] = [kp / ke - 1.0 for kp, ke in zip(row["k_panel"], k_exact)]
            per_level.append(row)
            for axis in range(3):
                dev_pct = row["rel_error"][axis] * 100.0
                csv_rows.append({
                    "case": label,
                    "family": family,
                    "a": a, "b": b, "c": c,
                    "aspect_a_over_b": f"{a / b:.6g}",
                    "aspect_a_over_c": f"{a / c:.6g}",
                    "level": f"{n_theta}x{n_phi}",
                    "n_theta": n_theta,
                    "n_phi": n_phi,
                    "n_panels": row["n_panels"],
                    "h_rel": f"{row['h_rel']:.6e}",
                    "quadrature": row["quadrature"],
                    "axis": axis + 1,
                    "volume_rel_error": f"{row['volume_rel_error']:.6e}",
                    "axis_extent_rel_error": f"{row['axis_extent_rel_error'][axis]:.6e}",
                    "k_panel": f"{row['k_panel'][axis]:.10f}",
                    "k_exact": f"{k_exact[axis]:.10f}",
                    "dev_pct": f"{dev_pct:.4f}",
                    "pass_5pct": "PASS" if abs(dev_pct) <= TOL_PCT else "FAIL",
                    "offdiag_rel": f"{row['offdiag_rel']:.3e}",
                    "area_max_over_min": f"{row['area_max_over_min']:.3e}",
                    "edge_over_sqrt_area_max": f"{row['edge_over_sqrt_area_max']:.3f}",
                    "solve_seconds": f"{row['solve_seconds']:.2f}",
                })
    return per_level, csv_rows


# ---------------------------------------------------------------------------
# 3. convergence: order, Richardson, GCI, worst case
# ---------------------------------------------------------------------------
def run_convergence(per_level: list[dict]) -> dict:
    bundles: dict[str, dict] = {}
    for row in per_level:
        bundles.setdefault(row["label"], {"levels": [], "meta": row})
        bundles[row["label"]]["levels"].append(row)

    out_cases = []
    for label, bundle in bundles.items():
        rows = sorted(bundle["levels"], key=lambda r: r["n_panels"])
        h = [r["h_rel"] for r in rows]
        k_exact = rows[0]["k_exact"]
        components = []
        for axis in range(3):
            k = [r["k_panel"][axis] for r in rows]
            errors = [kk / k_exact[axis] - 1.0 for kk in k]
            n_coarse = max(1, len(rows) - 2)
            p_coarse = fit_order(errors[:n_coarse], h[:n_coarse])
            p_all = fit_order(errors, h)
            if len(rows) >= 3 and k[-1] != k[-2]:
                p_apparent = (math.log(abs((k[-2] - k[-3]) / (k[-1] - k[-2])))
                              / math.log(h[-3] / h[-2]))
            else:                                       # pragma: no cover
                p_apparent = float("nan")
            ratio = h[-2] / h[-1]
            p_used = p_coarse if math.isfinite(p_coarse) and p_coarse > 0.2 else p_all
            k_rich = richardson(k[-1], k[-2], ratio, p_used)
            dev_rich = k_rich / k_exact[axis] - 1.0
            gci = FS * abs((k[-1] - k_rich) / k[-1]) / (ratio**p_used - 1.0)
            components.append({
                "axis": axis + 1,
                "k_exact": k_exact[axis],
                "k_panel_by_level": k,
                "rel_error_by_level": errors,
                "p_coarse_fit": p_coarse,
                "p_all_fit": p_all,
                "p_apparent_last3": p_apparent,
                "p_used_for_richardson": p_used,
                "refinement_ratio_finest": ratio,
                "k_richardson": k_rich,
                "richardson_dev_pct": dev_rich * 100.0,
                "gci_pct": gci * 100.0,
                "finest_dev_pct": errors[-1] * 100.0,
                "finest_pass_5pct": abs(errors[-1] * 100.0) <= TOL_PCT,
                "gci_pass_5pct": gci * 100.0 < GCI_TOL_PCT,
            })
        fine = rows[-1]
        out_cases.append({
            "label": label,
            "family": bundle["meta"]["family"],
            "semi_axes": bundle["meta"]["semi_axes"],
            "levels": [f"{r['n_theta']}x{r['n_phi']}" for r in rows],
            "n_panels": [r["n_panels"] for r in rows],
            "h_rel": h,
            "volume_rel_error_by_level": [r["volume_rel_error"] for r in rows],
            "area_max_over_min_finest": fine["area_max_over_min"],
            "edge_over_sqrt_area_max_finest": fine["edge_over_sqrt_area_max"],
            "components": components,
        })

    worst = None
    for case in out_cases:
        for comp in case["components"]:
            if worst is None or abs(comp["finest_dev_pct"]) > worst["abs_dev_pct"]:
                worst = {
                    "case": case["label"],
                    "family": case["family"],
                    "semi_axes": case["semi_axes"],
                    "axis": comp["axis"],
                    "abs_dev_pct": abs(comp["finest_dev_pct"]),
                    "dev_pct": comp["finest_dev_pct"],
                    "k_panel": comp["k_panel_by_level"][-1],
                    "k_exact": comp["k_exact"],
                    "k_richardson": comp["k_richardson"],
                    "richardson_dev_pct": comp["richardson_dev_pct"],
                    "gci_pct": comp["gci_pct"],
                    "p_coarse_fit": comp["p_coarse_fit"],
                    "area_max_over_min_finest": case["area_max_over_min_finest"],
                    "edge_over_sqrt_area_max_finest": case["edge_over_sqrt_area_max_finest"],
                    "rel_error_by_level": comp["rel_error_by_level"],
                }
    return {"cases": out_cases, "worst": worst}


def run_quadrature_study() -> list[dict]:
    """Same body and mesh, two surface-integration rules.

    ``centroid`` evaluates the single-layer potential once per panel -- at the
    centroid -- and integrates ``phi n`` with the flat area; ``gauss`` evaluates
    the potential at the solver's own 3x3 Gauss nodes and integrates ``phi n``
    with the matching weights.  The self-panel integrand ``1/|x-r'|`` is singular
    at the centroid yet integrable over the flat panel, so the centroid rule
    carries a per-panel bias of order ``sqrt(area)`` that does not cancel between
    the Neumann solve and the surface integral.  The measured gap between the two
    rules bounds that term.
    """
    out = []
    lookup = {c[0]: c for c in CASES}
    for label in QUAD_STUDY_CASES:
        _, a, b, c, family = lookup[label]
        k_exact = anchor_case(a, b, c)["k"]
        for (n_theta, n_phi) in QUAD_STUDY_LEVELS:
            entry = {"case": label, "level": f"{n_theta}x{n_phi}", "k_exact": k_exact}
            for mode in ("centroid", "gauss"):
                row = run_level(label, family, a, b, c, n_theta, n_phi, quadrature=mode)
                entry[f"k_{mode}"] = row["k_panel"]
                entry[f"rel_error_{mode}_pct"] = [kp / ke * 100.0 - 100.0
                                                  for kp, ke in zip(row["k_panel"], k_exact)]
            entry["gauss_minus_centroid_pct"] = [
                g - ce for g, ce in zip(entry["rel_error_gauss_pct"],
                                        entry["rel_error_centroid_pct"])
            ]
            out.append(entry)
    return out


def criteria_table(anchors: dict, csv_rows: list[dict], conv: dict) -> list[dict]:
    comps = [c for case in conv["cases"] for c in case["components"]]
    n_components = len(comps)
    n_cases = len(conv["cases"])
    n_levels = min(len(case["levels"]) for case in conv["cases"])
    failing = [(case["label"], comp["axis"], comp["finest_dev_pct"])
               for case in conv["cases"] for comp in case["components"]
               if not comp["finest_pass_5pct"]]
    rich_ok = [(case["label"], comp["axis"], comp["richardson_dev_pct"])
               for case in conv["cases"] for comp in case["components"]
               if abs(comp["richardson_dev_pct"]) <= TOL_PCT]
    gci_ok = [(case["label"], comp["axis"], comp["gci_pct"])
              for case in conv["cases"] for comp in case["components"]
              if comp["gci_pass_5pct"]]
    sphere_dev = min(r["worst_deviation_pct"] for r in anchors["limit_checks"]
                     if r["limit"] == "sphere")
    return [
        {
            "id": 1,
            "criterion": "anchor limits: sphere Cm = 0.5 within 0.1 %; slender prolate "
                         "axial 0 / transverse 1; flat oblate disc M_33 -> 8/3 rho a^3",
            "measured": {"sphere_worst_dev_pct": sphere_dev,
                         "limit_verdicts": {f"{r['limit']} ({r['quantity']})": r["verdict"]
                                            for r in anchors["limit_checks"]}},
            "verdict": anchors["criterion_1_verdict"],
        },
        {
            "id": 2,
            "criterion": f">= 6 axis ratios x >= 3 mesh levels; every diagonal coefficient"
                         f" within +-{TOL_PCT} % of the anchor at the finest mesh",
            "measured": {
                "cases": n_cases,
                "levels_per_case": n_levels,
                "components": n_components,
                "components_passing": n_components - len(failing),
                "failing": [{"case": f[0], "axis": f[1], "dev_pct": f[2]} for f in failing],
                "rows_in_csv": len(csv_rows),
            },
            "verdict": "PASS" if not failing else "FAIL",
        },
        {
            "id": 3,
            "criterion": f"Richardson extrapolation: observed order reported and GCI < "
                         f"{GCI_TOL_PCT} %",
            "measured": {
                "observed_order_p_all": {case["label"]: [c["p_all_fit"] for c in case["components"]]
                                         for case in conv["cases"]},
                "observed_order_p_coarse": {case["label"]: [c["p_coarse_fit"] for c in case["components"]]
                                            for case in conv["cases"]},
                "gci_pct": {case["label"]: [c["gci_pct"] for c in case["components"]]
                            for case in conv["cases"]},
                "components_gci_below_tol": len(gci_ok),
                "components_richardson_within_5pct": len(rich_ok),
                "components": n_components,
                "note": "p is fitted on the coarsest levels only and then used to extrapolate "
                        "the two finest, so the extrapolation cannot tune itself",
            },
            "verdict": "PASS" if len(gci_ok) == n_components else "FAIL",
        },
        {
            "id": 4,
            "criterion": "worst case named and attributed (surface faceting vs self-panel quadrature)",
            "measured": conv["worst"],
            "verdict": "PASS" if conv["worst"] is not None else "FAIL",
        },
        {
            "id": 5,
            "criterion": "every case reported (no cherry-picking): rows = cases x levels x 3 axes",
            "measured": {"expected_rows": n_cases * n_levels * 3, "rows_in_csv": len(csv_rows)},
            "verdict": "PASS" if len(csv_rows) == n_cases * n_levels * 3 else "FAIL",
        },
    ]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(_HERE), help="artefact directory")
    parser.add_argument("--quick", action="store_true",
                        help="three coarse levels only (smoke run, ~20 s)")
    parser.add_argument("--levels", default=None,
                        help="comma separated n_theta x n_phi pairs, e.g. '16x12,32x24'")
    parser.add_argument("--no-quad-study", action="store_true")
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.levels:
        levels = tuple(tuple(int(v) for v in part.split("x")) for part in args.levels.split(","))
    else:
        levels = QUICK_LEVELS if args.quick else LEVELS

    started = time.time()
    anchors = build_anchors()
    (out / "anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    print(f"[anchors] criterion 1: {anchors['criterion_1_verdict']}")

    per_level, csv_rows = run_scan(levels)
    write_csv(out / "ellipsoid_scan.csv", csv_rows)
    print(f"[scan] {len(csv_rows)} rows over levels "
          f"{', '.join(f'{a}x{b}' for a, b in levels)}")

    conv = run_convergence(per_level)
    quad = [] if args.no_quad_study else run_quadrature_study()
    criteria = criteria_table(anchors, csv_rows, conv)
    conv["quadrature_study"] = quad
    conv["criteria"] = criteria
    conv["levels"] = [f"{a}x{b}" for a, b in levels]
    conv["tol_pct"] = TOL_PCT
    conv["gci_tol_pct"] = GCI_TOL_PCT
    conv["quadrature"] = "gauss"
    conv["wall_seconds"] = time.time() - started
    (out / "convergence.json").write_text(json.dumps(conv, indent=2) + "\n")

    worst = conv["worst"]
    print(f"[convergence] worst {worst['case']} axis {worst['axis']}: "
          f"{worst['dev_pct']:+.3f} % -> {worst['k_richardson']:.6f} "
          f"({worst['richardson_dev_pct']:+.3f} %), GCI {worst['gci_pct']:.3f} %")
    for crit in criteria:
        print(f"[criterion {crit['id']}] {crit['verdict']}")

    if not args.no_readme:
        import report
        meta = {"levels": ", ".join(f"{a}x{b}" for a, b in levels),
                "n_cases": len(CASES),
                "n_axes": 3}
        (out / "README.md").write_text(
            report.render_readme(anchors, csv_rows, conv, quad, criteria, meta))
        print(f"[readme] {out / 'README.md'}")

    print(f"[done] {time.time() - started:.1f} s -> {out}")
    return 0 if all(c["verdict"] == "PASS" for c in criteria) else 1


if __name__ == "__main__":
    raise SystemExit(main())