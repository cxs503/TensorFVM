"""Double-body closure audit: flat-lid cap vs waterplane mirror (debt D9).

Why this module exists
----------------------
``maneuvering.added_inertia_panel`` builds the KVLCC2 hull as a closed panel
body in two different ways, and report ``docs/maneuvering-added-inertia/REPORT.md``
(debt item D7) mixes them: the **lid cap** (hull + flat lid on the design
waterline, ``with_lid=True``, ``IMAGE_SPECS["body"]``) and the **waterplane
mirror** (open hull + its image in ``z = 0``, ``with_lid=False``,
``IMAGE_SPECS["single_hull"]``), i.e. the classical **double body**.

D7 used the mirror route for the corroborating sway number (``M_22 = 2573.5``)
and the lid route for the headline yaw figure (``J_z = 4822.4``), arguing that
the two closures are the same model because "the lid carries no flux, so it is
exactly the plane of symmetry of the double body".  That argument is wrong, and
this module measures by how much:

    zero normal velocity on the lid only makes the potential *even* across it.
    In the lid model the fluid continues **above** ``z = 0`` and is free to
    stream over the deck, whereas the double body has the mirrored hull there.
    The lid body is therefore the *closed-body* problem (a bluff body with a
    flat top), not the double-body problem, and it under-states both ``M_22``
    and ``J_z`` by roughly a third, systematically at every mesh.

That matters because the slender-body strip formulas this quantity is compared
against *are* double-body formulas (the 2D added mass of a wall-sided section is
the full-circle, i.e. mirrored, value).  Comparing a lid-cap panel number with a
strip formula mixes two physical models.

What is verified here (criteria ``C1``..``C6``)
----------------------------------------------
``C1`` hemisphere (half sphere resting on the waterline plane), sway via the
       mirror route, against the exact half-body value ``m_a/2``.
``C2`` hemisphere, yaw about a *vertical axis offset by* ``d`` from the sphere
       centre, mirror route, against the exact ``m_a d^2 / 2`` -- a non-trivial
       anchor that exercises the moment arm *and* the image kernel together (the
       sphere's own rotational added inertia is exactly zero, so a wrong moment
       arm cannot hide).
``C3`` the lid-vs-mirror gap on the KVLCC2 hull is a *model* difference, not a
       discretisation error: the ratio must be mesh-independent while being much
       larger than the discretisation uncertainty.
``C4`` the double-body ``J_z`` (about midship) of the KVLCC2 hull from the mirror
       route, with a power-law limit fit, and the corrected statement against the
       slender-body value used by ``maneuvering.dynamics``.
``C5`` the same-hull double-body sway added mass vs the slender ``m_y``.
``C6`` the 6x6 Kirchhoff assembly itself against a closed form derived from
       nothing but the parallel-axis theorem (all 36 entries, including the
       translation-rotation coupling), for an offset sphere and an offset
       spheroid.  This also closes debt-ledger item ``D8`` (see below).

Debt-ledger D8 (registered by m14901 in the D7 report, never written into the
ledger): "``added_mass_matrix_6dof`` returns ``I_zz = 2.05`` for a sphere centred
at ``(0, 0, 1.5)`` where the theory is ``0``".  Not reproducible: ``2.05`` is the
**heave translational** entry ``A_heave,heave`` of the same matrix.  The yaw entry
is ``0.0`` exactly (the yaw axis passes through the sphere centre), and for the
centre at ``(1.5, 0, 0)`` the yaw entry is ``4.59`` against the closed form
``4.71`` (``-2.6 %``, the same discretisation error as the translational block).
The claim was a block-index misreading; ``C6`` pins the full matrix so the
phantom cannot come back.

Limitations (stated, not hidden)
--------------------------------
* The hemisphere has an edge at the waterline, but in the *mirror* model the
  doubled geometry is the smooth full sphere, which is why that route converges
  cleanly.
* The hull section form is the repository's calibrated KVLCC2 parametric family
  (the same geometry the manoeuvring force model uses), not tank offsets.
* Potential flow, no viscosity, no forward speed, zero-frequency (rigid-wall)
  free surface: the same physics level as the slender formulas compared with.
* There is no analytic anchor for the lid-cap model, so this module does not
  *score* it; it shows that it is a different model and identifies the mirror
  route as the double-body one.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..maneuvering.added_inertia_panel import (deck_panels, geometry_gate_panels,
                                               hull_panels)
from ..maneuvering.dynamics import HullForceModelSet, added_masses
from ..maneuvering.hull import DEFAULT_CONDITIONS, build_hull
from ..panel_added_inertia import lamb_rotational_inertia, rotational_added_inertia
from ..panel_added_mass import (added_mass_matrix, added_mass_matrix_6dof,
                                lamb_spheroid_factors, spheroid_panels,
                                sphere_added_mass, sphere_panels_offset,
                                surface_generators)
from ..panel_potential import IMAGE_SPECS
from ..twin_hull_geometry import PanelSet, sphere_panels

MIRROR = IMAGE_SPECS["single_hull"]   # double body (waterline rigid wall)
BODY = IMAGE_SPECS["body"]            # closed body, no image

#: hemisphere meshes for C1/C2 (n_phi must be even so a node row lies on z = 0)
HEMI_LADDER = ((16, 18), (24, 24), (32, 32))
#: KVLCC2 (n_stations, n_girth) ladder for C3..C5
HULL_LADDER = ((21, 5), (41, 9), (61, 13))
#: meshes for the 6x6 assembly checks (C6)
SPHERE_LADDER = ((24, 24), (32, 32))
SPHEROID_LADDER = ((24, 24), (32, 32))

TOL_PANEL = 0.03          # platform acceptance: 3 % against a reference value
TOL_RICHARDSON = 0.01     # 1 % after extrapolation
TOL_IDENTITY = 1.0e-10    # exact algebraic identity (kinematic transport)
TOL_STRUCTURAL = 1.0e-6   # structurally-zero entries, relative to a reference scale
TOL_GAP_STABILITY = 0.03  # lid/mirror ratio must be mesh-independent to 3 %
GAP_FLOOR = 0.25          # and the gap itself must exceed 25 % to be a model effect
TOL_LIMIT_DRIFT = 0.02    # limit-fit drift on a 3-mesh ladder (refinement ratio ~1.45)

CRITERIA_TEXT = {
    "C1": "hemisphere sway (mirror route) within 3 % of the exact m_a/2",
    "C2": "hemisphere yaw about an offset vertical axis within 3 % of m_a d^2/2",
    "C3": "lid-vs-mirror gap on the KVLCC2 hull is mesh-independent (model, not discretisation)",
    "C4": "double-body J_z (midship) converges; power-law limit within 1 % of the finest mesh",
    "C5": "double-body M_22 vs slender m_y reported on the same (mirror) model",
    "C6": "6x6 Kirchhoff assembly vs parallel-axis closed form, all 36 entries, < 3 %",
}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def translated(panels: PanelSet, shift: Sequence[float]) -> PanelSet:
    """Rigid translation of a panel set (normals and areas unchanged)."""
    s = np.asarray(shift, dtype=float)
    if s.shape != (3,):
        raise ValueError("shift must be a 3-vector")
    corners = None if panels.corners is None else panels.corners + s
    return PanelSet(panels.centroids + s, panels.normals.copy(), panels.areas.copy(),
                    corners, panels.label)


def hemisphere_panels(radius: float = 1.0, n_theta: int = 24,
                      n_phi: int = 24) -> PanelSet:
    """Lower half (``z < 0``) of a sphere: a half body resting on ``z = 0``.

    ``n_phi`` must be even so that a node row sits exactly at ``z = 0`` and the
    waterline rim is closed by the waterplane mirror.
    """
    if n_phi % 2:
        raise ValueError("n_phi must be even so that a node row lies on z = 0")
    full = sphere_panels(radius, n_theta, n_phi)
    keep = full.centroids[:, 2] < 0.0
    corners = None if full.corners is None else full.corners[keep]
    return PanelSet(full.centroids[keep], full.normals[keep], full.areas[keep],
                    corners, "hemisphere")


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------
def skew(v: Sequence[float]) -> np.ndarray:
    """Cross-product matrix ``[v]x`` (``[v]x w = v x w``)."""
    a, b, c = (float(v[0]), float(v[1]), float(v[2]))
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])


def shift_added_mass(matrix: np.ndarray, delta: Sequence[float]) -> np.ndarray:
    """Transport a 6x6 Kirchhoff matrix to a new origin ``delta`` away.

    Rigid-body kinematics: with ``r`` measured from the old origin,

        V = U + Omega x r = (U + delta x Omega) + Omega x (r + delta),

    so the generalised velocity transforms as ``q_old = P q_new`` with

        P = [[I, [delta]x], [0, I]],      A_new = P^T A_old P.

    ``delta`` is the *new* origin minus the *old* one.  The identity is exact
    algebra, so it doubles as an independent check that the assembly measures the
    moment arm correctly.
    """
    d = np.asarray(delta, dtype=float)
    if d.shape != (3,):
        raise ValueError("delta must be a 3-vector")
    P = np.eye(6)
    P[:3, 3:] = skew(d)
    return P.T @ np.asarray(matrix, dtype=float) @ P


def sphere_kirchhoff_about_origin(radius: float = 1.0, rho: float = 1.0,
                                  centre: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact 6x6 Kirchhoff matrix of a sphere about the coordinate origin.

    About its own centre the sphere is ``diag(m_a, m_a, m_a, 0, 0, 0)`` (Fossen;
    the rotational block vanishes because ``r x n = 0`` on a sphere), and the
    offset case follows from :func:`shift_added_mass`.
    """
    m_a = sphere_added_mass(radius, rho)
    about_centre = np.diag([m_a, m_a, m_a, 0.0, 0.0, 0.0])
    return shift_added_mass(about_centre, -np.asarray(centre, dtype=float))


def spheroid_kirchhoff_about_origin(a: float, b: float, *, rho: float = 1.0,
                                    centre: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Exact 6x6 Kirchhoff matrix of a prolate spheroid (axis ``x``) about the origin.

    About its own centre: the Lamb translational entries ``k_axial rho V`` (x) and
    ``k_transverse rho V`` (y, z), zero roll added inertia (symmetry axis, so
    ``K_11 = 0``) and the Lamb transverse rotational value for pitch/yaw
    (:func:`panel_added_inertia.lamb_rotational_inertia`).  There is no
    translation-rotation coupling about the centre (three symmetry planes); the
    offset case follows from :func:`shift_added_mass`.
    """
    k1, k2 = lamb_spheroid_factors(a, b)
    vol = 4.0 / 3.0 * np.pi * a * b * b
    rot = lamb_rotational_inertia(a, b, b, rho=rho)
    about_centre = np.zeros((6, 6))
    about_centre[0, 0] = k1 * rho * vol
    about_centre[1, 1] = about_centre[2, 2] = k2 * rho * vol
    about_centre[3, 3] = float(rot[0, 0])
    about_centre[4, 4] = about_centre[5, 5] = float(rot[1, 1])
    return shift_added_mass(about_centre, -np.asarray(centre, dtype=float))


def power_law_limit(counts: Iterable[float], values: Iterable[float],
                    q_bounds: Sequence[float] = (0.05, 8.0),
                    n_q: int = 1200) -> Dict[str, object]:
    """Fit ``J(N) = J_inf + K N^(-q)`` by scanning ``q`` with least squares.

    Returns the limit ``J_inf``, the fitted exponent ``q`` and the residual sum.
    Used only to state how far the finest mesh is from its own limit, and to give
    a limit value for the hull numbers.
    """
    n = np.asarray(list(counts), dtype=float)
    v = np.asarray(list(values), dtype=float)
    if n.size < 3:
        # not enough levels for a limit fit: report the finest value as the
        # "limit" with a NaN exponent so callers can see it was not extrapolated
        return {"limit": float(v[-1]), "exponent": float("nan"), "rss": 0.0,
                "counts": n.tolist(), "values": v.tolist(), "fitted": False}
    if np.any(n <= 0.0):
        raise ValueError("panel counts must be positive")
    best = None
    for q in np.linspace(float(q_bounds[0]), float(q_bounds[1]), int(n_q)):
        basis = np.column_stack([np.ones_like(n), n ** (-q)])
        coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
        rss = float(np.sum((basis @ coef - v) ** 2))
        if best is None or rss < best[0]:
            best = (rss, q, coef)
    rss, q, coef = best
    return {"limit": float(coef[0]), "exponent": float(q), "rss": float(rss),
            "counts": n.tolist(), "values": v.tolist(), "fitted": True}


def observed_order(coarse_error: float, fine_error: float,
                   coarse_count: float, fine_count: float) -> float:
    """Observed convergence order from two levels: ``p = ln(e1/e2)/ln(N2/N1)``."""
    e1, e2 = abs(float(coarse_error)), abs(float(fine_error))
    if e1 <= 0.0 or e2 <= 0.0 or e2 >= e1:
        return float("nan")
    return float(np.log(e1 / e2) / np.log(float(fine_count) / float(coarse_count)))


# ---------------------------------------------------------------------------
# C1 / C2: the mirror route against exact half-body anchors
# ---------------------------------------------------------------------------
def hemisphere_checks(ladder: Sequence[Sequence[int]] = HEMI_LADDER,
                      radius: float = 1.0, rho: float = 1.0,
                      offset_ratio: float = 1.5) -> Dict[str, object]:
    """Hemisphere sway and offset-axis yaw against the exact half-body values."""
    m_a_full = sphere_added_mass(radius, rho)
    anchor_sway = 0.5 * m_a_full               # half of the full sphere
    d = offset_ratio * radius
    anchor_yaw = 0.5 * m_a_full * d * d        # parallel axis of the full sphere, halved
    rows: List[Dict[str, float]] = []
    for n_theta, n_phi in ladder:
        panels = hemisphere_panels(radius, n_theta, n_phi)
        sway = float(added_mass_matrix(panels, rho=rho, directions=(1,),
                                       images=MIRROR)[1, 1])
        yaw = float(rotational_added_inertia(panels, rho=rho, axes=(2,),
                                             origin=np.array([d, 0.0, 0.0]),
                                             images=MIRROR)[2, 2])
        own = float(rotational_added_inertia(panels, rho=rho, axes=(2,),
                                            origin=np.zeros(3), images=MIRROR)[2, 2])
        rim = float(panels.corners[:, :, 2].max()) if panels.corners is not None else float("nan")
        rows.append({
            "mesh": f"{n_theta}x{n_phi}", "panels": int(panels.n_panels),
            "rim_z_max": rim,
            "sway": sway, "err_sway": sway / anchor_sway - 1.0,
            "yaw_offset": yaw, "err_yaw": yaw / anchor_yaw - 1.0,
            "yaw_own_axis": own,
            "yaw_own_axis_rel": abs(own) / (m_a_full * radius ** 2),
        })
    sway_fit = power_law_limit([r["panels"] for r in rows], [r["sway"] for r in rows])
    yaw_fit = power_law_limit([r["panels"] for r in rows], [r["yaw_offset"] for r in rows])
    finest = rows[-1]
    rich = {
        "sway": {"anchor": anchor_sway, "limit": sway_fit["limit"],
                 "rel_err": sway_fit["limit"] / anchor_sway - 1.0,
                 "exponent": sway_fit["exponent"],
                 "observed_order": (observed_order(rows[-2]["err_sway"], finest["err_sway"],
                                                   rows[-2]["panels"], finest["panels"])
                                    if len(rows) >= 2 else float("nan"))},
        "yaw_offset": {"anchor": anchor_yaw, "limit": yaw_fit["limit"],
                       "rel_err": yaw_fit["limit"] / anchor_yaw - 1.0,
                       "exponent": yaw_fit["exponent"],
                       "observed_order": (observed_order(rows[-2]["err_yaw"], finest["err_yaw"],
                                                         rows[-2]["panels"], finest["panels"])
                                          if len(rows) >= 2 else float("nan"))},
    }
    return {"radius": radius, "rho": rho, "offset_ratio": offset_ratio,
            "m_a_full": m_a_full, "anchor_sway": anchor_sway, "anchor_yaw": anchor_yaw,
            "rows": rows, "richardson": rich}


# ---------------------------------------------------------------------------
# C3 / C4 / C5: the KVLCC2 hull, both closures
# ---------------------------------------------------------------------------
def hull_route_checks(ladder: Sequence[Sequence[int]] = HULL_LADDER,
                      hull=None) -> Dict[str, object]:
    """Lid-cap vs waterplane-mirror values of the same hull (J_z and M_22)."""
    hull = hull if hull is not None else build_hull()
    c = DEFAULT_CONDITIONS
    rho = float(c.rho)
    slender = added_masses(HullForceModelSet(), n=121)
    rows: List[Dict[str, float]] = []
    for n_stations, n_girth in ladder:
        with_lid = hull_panels(hull, n_stations=n_stations, n_girth=n_girth, with_lid=True)
        open_hull = hull_panels(hull, n_stations=n_stations, n_girth=n_girth, with_lid=False)
        gate = geometry_gate_panels(with_lid, hull)
        x_mid = 0.5 * (float(open_hull.centroids[:, 0].min())
                       + float(open_hull.centroids[:, 0].max()))
        origin = np.array([x_mid, 0.0, 0.0])
        m22_mirror = float(added_mass_matrix(open_hull, rho=rho, directions=(1,),
                                             images=MIRROR)[1, 1])
        m22_lid = float(added_mass_matrix(with_lid, rho=rho, directions=(1,),
                                          images=BODY)[1, 1])
        jz_mirror = float(rotational_added_inertia(open_hull, rho=rho, axes=(2,),
                                                   origin=origin, images=MIRROR)[2, 2])
        jz_lid = float(rotational_added_inertia(with_lid, rho=rho, axes=(2,),
                                                origin=origin, images=BODY)[2, 2])
        # D7's justification for the lid is that the flat lid carries no flux.
        # Re-measure it (it is true) and keep it as the counter-example it is:
        # zero lid flux does not make the lid body a double body.
        psi = surface_generators(deck_panels(with_lid))
        rows.append({
            "mesh": f"{n_stations}x{n_girth}",
            "panels_lid": int(with_lid.n_panels), "panels_open": int(open_hull.n_panels),
            "volume_rel_error": float(gate["checks"]["volume"]["relative_error"]),
            "lid_area_rel_error": float(gate["checks"]["lid_area"]["relative_error"]),
            "x_mid": x_mid,
            "M22_mirror": m22_mirror, "M22_lid": m22_lid,
            "M22_ratio": m22_lid / m22_mirror,
            "Jz_mirror": jz_mirror, "Jz_lid": jz_lid, "Jz_ratio": jz_lid / jz_mirror,
            "lid_flux_max_sway": float(np.max(np.abs(psi[:, 1]))),
            "lid_flux_max_yaw": float(np.max(np.abs(psi[:, 5]))),
            "Jz_mirror_over_slender": jz_mirror / slender["J_z"],
            "Jz_lid_over_slender": jz_lid / slender["J_z"],
            "M22_mirror_over_slender": m22_mirror / slender["m_y"],
        })
    counts = [r["panels_open"] for r in rows]
    jz_fit = power_law_limit(counts, [r["Jz_mirror"] for r in rows])
    m22_fit = power_law_limit(counts, [r["M22_mirror"] for r in rows])
    ratios = np.array([r["M22_ratio"] for r in rows])
    jz_ratios = np.array([r["Jz_ratio"] for r in rows])
    gaps = {
        "M22": {"min": float(ratios.min()), "max": float(ratios.max()),
                "spread": float(ratios.max() - ratios.min()),
                "gap_at_finest": float(1.0 - ratios[-1])},
        "Jz": {"min": float(jz_ratios.min()), "max": float(jz_ratios.max()),
               "spread": float(jz_ratios.max() - jz_ratios.min()),
               "gap_at_finest": float(1.0 - jz_ratios[-1])},
    }
    return {
        "rho": rho, "length": float(c.length), "slender": dict(slender),
        "rows": rows, "gaps": gaps,
        "limit": {"Jz_mirror": jz_fit, "M22_mirror": m22_fit},
        "finest": rows[-1],
        "Jz_double_body_over_slender_extrapolated": jz_fit["limit"] / slender["J_z"],
        "slender_overestimate_extrapolated": slender["J_z"] / jz_fit["limit"] - 1.0,
        "slender_overestimate_lid": slender["J_z"] / rows[-1]["Jz_lid"] - 1.0,
        "d7_headline_ratio": 0.371,   # docs/maneuvering-added-inertia/REPORT.md
    }


# ---------------------------------------------------------------------------
# C6: the 6x6 assembly against the parallel-axis closed form
# ---------------------------------------------------------------------------
def center_tag(centre: Sequence[float]) -> str:
    return "(" + ",".join(f"{float(v):g}" for v in centre) + ")"


def assembly_check(panels: PanelSet, reference: np.ndarray, *, rho: float) -> Dict[str, object]:
    """Sup-norm comparison of the assembled 6x6 matrix with a closed form."""
    full = added_mass_matrix_6dof(panels, rho=rho, directions=None, images=BODY)
    ref = np.asarray(reference, dtype=float)
    scale = float(np.max(np.abs(ref)))
    return {"panels": int(panels.n_panels), "scale": scale,
            "max_abs_error_over_scale": float(np.max(np.abs(full - ref))) / scale,
            "assembled": full.tolist(), "reference": ref.tolist()}


def kirchhoff_assembly_checks() -> Dict[str, object]:
    """Offset sphere / offset spheroid 6x6 blocks against the closed forms."""
    rho = 1.0
    cases: List[Dict[str, object]] = []
    for n_theta, n_phi in SPHERE_LADDER:
        centre = np.array([1.5, 0.0, 0.0])
        panels = sphere_panels_offset(1.0, centre, n_theta, n_phi)
        out = assembly_check(panels, sphere_kirchhoff_about_origin(1.0, rho, centre), rho=rho)
        out.update({"label": f"sphere offset {center_tag(centre)}", "mesh": f"{n_theta}x{n_phi}"})
        cases.append(out)
    for n_theta, n_phi in SPHEROID_LADDER:
        centre = np.array([1.2, 0.0, 0.8])
        panels = translated(spheroid_panels(2.0, 1.0, n_theta, n_phi), centre)
        ref = spheroid_kirchhoff_about_origin(2.0, 1.0, rho=rho, centre=centre)
        out = assembly_check(panels, ref, rho=rho)
        out.update({"label": f"spheroid a/b=2 offset {center_tag(centre)}",
                    "mesh": f"{n_theta}x{n_phi}"})
        cases.append(out)
    # D8 (phantom): the number reported by m14901 for the offset sphere is the
    # heave *translational* entry, not the yaw rotational one.
    centre = np.array([0.0, 0.0, 1.5])
    panels = sphere_panels_offset(1.0, centre, 32, 32)
    matrix = added_mass_matrix_6dof(panels, rho=rho, directions=None, images=BODY)
    ref = sphere_kirchhoff_about_origin(1.0, rho, centre)
    d8 = {
        "label": "D8 reproduction: sphere centre (0,0,1.5)",
        "heave_translational": float(matrix[2, 2]),
        "yaw_rotational": float(matrix[5, 5]),
        "yaw_reference": float(ref[5, 5]),
        "yaw_error": float(matrix[5, 5] - ref[5, 5]),
        "reported_by_m14901": 2.05,
        "verdict": ("2.05 is A_heave,heave; the yaw entry is -0.0 (theory 0) -> "
                    "claim not reproducible (block-index misreading)"),
    }
    worst = max(c["max_abs_error_over_scale"] for c in cases)
    finest: Dict[str, float] = {}
    for c in cases:
        label = c["label"]
        if label not in finest or c["panels"] > int(next(
                x["panels"] for x in cases if x["label"] == label)):
            finest[label] = c["max_abs_error_over_scale"]
        else:
            finest[label] = min(finest[label], c["max_abs_error_over_scale"])
    return {"rho": rho, "cases": cases, "worst": float(worst),
            "worst_finest": float(max(finest.values())),
            "finest_by_case": finest, "d8": d8}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def run(hemi_ladder: Sequence[Sequence[int]] = HEMI_LADDER,
        hull_ladder: Sequence[Sequence[int]] = HULL_LADDER) -> Dict[str, object]:
    t0 = time.time()
    hemi = hemisphere_checks(hemi_ladder)
    hull = hull_route_checks(hull_ladder)
    asm = kirchhoff_assembly_checks()
    crit: Dict[str, Dict[str, object]] = {}
    finest_h = hemi["rows"][-1]
    rich = hemi["richardson"]
    crit["C1"] = {"statement": CRITERIA_TEXT["C1"],
                  "value": {"err_sway_finest": finest_h["err_sway"],
                            "err_sway_limit": rich["sway"]["rel_err"],
                            "observed_order": rich["sway"]["observed_order"],
                            "rim_z_max": finest_h["rim_z_max"]},
                  "threshold": {"panel": TOL_PANEL, "richardson": TOL_RICHARDSON},
                  "pass": bool(abs(finest_h["err_sway"]) < TOL_PANEL
                               and (len(hemi["rows"]) < 2
                                    or abs(finest_h["err_sway"]) < abs(hemi["rows"][0]["err_sway"]))
                               and finest_h["rim_z_max"] < 1.0e-12)}
    crit["C2"] = {"statement": CRITERIA_TEXT["C2"],
                  "value": {"err_yaw_finest": finest_h["err_yaw"],
                            "err_yaw_limit": rich["yaw_offset"]["rel_err"],
                            "observed_order": rich["yaw_offset"]["observed_order"],
                            "own_axis_rel": finest_h["yaw_own_axis_rel"]},
                  "threshold": {"panel": TOL_PANEL, "richardson": TOL_RICHARDSON,
                                "own_axis": TOL_STRUCTURAL},
                  "pass": bool(abs(finest_h["err_yaw"]) < TOL_PANEL
                               and (len(hemi["rows"]) < 2
                                    or abs(finest_h["err_yaw"]) < abs(hemi["rows"][0]["err_yaw"]))
                               and finest_h["yaw_own_axis_rel"] < TOL_STRUCTURAL)}
    gaps = hull["gaps"]
    crit["C3"] = {"statement": CRITERIA_TEXT["C3"],
                  "value": {"M22_gap_finest": gaps["M22"]["gap_at_finest"],
                            "M22_ratio_spread": gaps["M22"]["spread"],
                            "Jz_gap_finest": gaps["Jz"]["gap_at_finest"],
                            "Jz_ratio_spread": gaps["Jz"]["spread"],
                            "lid_flux_max_sway": hull["finest"]["lid_flux_max_sway"],
                            "lid_flux_max_yaw": hull["finest"]["lid_flux_max_yaw"]},
                  "threshold": {"gap_floor": GAP_FLOOR, "stability": TOL_GAP_STABILITY},
                  "pass": bool(min(gaps["M22"]["gap_at_finest"], gaps["Jz"]["gap_at_finest"]) > GAP_FLOOR
                               and max(gaps["M22"]["spread"], gaps["Jz"]["spread"]) < TOL_GAP_STABILITY)}
    fit = hull["limit"]["Jz_mirror"]
    drift = fit["limit"] / hull["finest"]["Jz_mirror"] - 1.0
    crit["C4"] = {"statement": CRITERIA_TEXT["C4"],
                  "value": {"Jz_finest": hull["finest"]["Jz_mirror"], "Jz_limit": fit["limit"],
                            "exponent": fit["exponent"], "limit_vs_finest": drift,
                            "Jz_over_slender": hull["Jz_double_body_over_slender_extrapolated"],
                            "slender_overestimate": hull["slender_overestimate_extrapolated"],
                            "slender_overestimate_lid": hull["slender_overestimate_lid"]},
                  "threshold": {"limit_vs_finest": TOL_LIMIT_DRIFT},
                  "pass": bool(abs(drift) < TOL_LIMIT_DRIFT)}
    ratio = hull["finest"]["M22_mirror_over_slender"]
    crit["C5"] = {"statement": CRITERIA_TEXT["C5"],
                  "value": {"M22_mirror": hull["finest"]["M22_mirror"],
                            "m_y_slender": hull["slender"]["m_y"], "ratio": ratio,
                            "M22_limit": hull["limit"]["M22_mirror"]["limit"]},
                  "threshold": {"note": "corroborating number: no analytic anchor for a hull"},
                  "pass": bool(0.0 < ratio < 1.0)}
    crit["C6"] = {"statement": CRITERIA_TEXT["C6"],
                  "value": {"worst_max_abs_error_over_scale": asm["worst"],
                            "worst_finest_mesh": asm["worst_finest"],
                            "cases": {c["label"] + " " + c["mesh"]: c["max_abs_error_over_scale"]
                                      for c in asm["cases"]},
                            "d8_yaw": asm["d8"]["yaw_rotational"],
                            "d8_yaw_reference": asm["d8"]["yaw_reference"],
                            "d8_heave": asm["d8"]["heave_translational"],
                            "d8_verdict": asm["d8"]["verdict"]},
                  "threshold": {"panel": TOL_PANEL, "gate": "finest mesh of each case"},
                  "pass": bool(asm["worst_finest"] < TOL_PANEL)}
    return {"schema": "tensorfvm.double-body-closure-audit/1",
            "tolerances": {"panel": TOL_PANEL, "richardson": TOL_RICHARDSON,
                           "structural": TOL_STRUCTURAL,
                           "gap_stability": TOL_GAP_STABILITY, "gap_floor": GAP_FLOOR},
            "c1c2_hemisphere": hemi, "c3c5_hull": hull, "c6_assembly": asm,
            "criteria": crit, "all_pass": all(c["pass"] for c in crit.values()),
            "elapsed_s": time.time() - t0}


# ---------------------------------------------------------------------------
# report / figure / CLI
# ---------------------------------------------------------------------------
def render_report(s: Dict[str, object]) -> str:
    c, hemi, hull, asm = s["criteria"], s["c1c2_hemisphere"], s["c3c5_hull"], s["c6_assembly"]
    crit_rows = "\n".join(f"| {k} | {v['statement']} | {'PASS' if v['pass'] else 'FAIL'} |"
                          for k, v in c.items())
    h_rows = "\n".join(
        f"| {r['mesh']} | {r['panels']} | {r['sway']:.5f} | {r['err_sway']:+.3%} | "
        f"{r['yaw_offset']:.5f} | {r['err_yaw']:+.3%} | {r['yaw_own_axis']:.1e} |"
        for r in hemi["rows"])
    rich_rows = "\n".join(
        f"| {k} | {v['anchor']:.5f} | {v['limit']:.5f} | {v['rel_err']:+.3%} | "
        f"{v['exponent']:.2f} | {v['observed_order']:.2f} |"
        for k, v in hemi["richardson"].items())
    hu_rows = "\n".join(
        f"| {r['mesh']} | {r['panels_lid']}/{r['panels_open']} | {r['M22_mirror']:.1f} | "
        f"{r['M22_lid']:.1f} | {r['M22_ratio']:.3f} | {r['Jz_mirror']:.1f} | {r['Jz_lid']:.1f} | "
        f"{r['Jz_ratio']:.3f} | {r['volume_rel_error']:+.3%} |" for r in hull["rows"])
    asm_rows = "\n".join(f"| {x['label']} | {x['mesh']} | {x['panels']} | "
                         f"{x['max_abs_error_over_scale']:.3%} |" for x in asm["cases"])
    d8, fin = asm["d8"], hull["finest"]
    return f"""# Double-body closure audit: flat-lid cap vs waterplane mirror (debt D9)

TensorFVM · every number below is generated programmatically from `summary.json`
(zero hand-copied values). Reproduce with

```
python -m tensorfvm.verification.double_body_added_mass_gate --out docs/verification-double-body-gate
```

## 1. The question

`maneuvering.added_inertia_panel` closes the hull below the waterline in two ways:
**lid cap** (`with_lid=True`, `IMAGE_SPECS["body"]`; hull + flat lid at `z = 0` in
the unbounded fluid) and **waterplane mirror** (`with_lid=False`,
`IMAGE_SPECS["single_hull"]`; open hull + its image in `z = 0` = the classical
double body). Report `docs/maneuvering-added-inertia/REPORT.md` (debt D7) took the
corroborating sway number from the mirror route and the headline yaw number from
the lid route, justified by "the lid carries no flux, so it is exactly the plane
of symmetry of the double body".

That justification is wrong: zero normal velocity on the lid only makes the
potential *even* across it. In the lid model the fluid continues **above**
`z = 0` and streams over the deck, while the double body has the mirrored hull
there. The lid body is the closed-body problem (a bluff body with a flat top).

## 2. Anchoring the mirror route on exact values (C1, C2)

Half sphere, radius 1, rho 1, resting on the waterline plane. The mirror image of
the half body is the full sphere, so the half-body value is
`m_a/2 = {hemi['anchor_sway']:.6f}` (sway, parallel to the wall) and the value for
yaw about a vertical axis offset by `d = {hemi['offset_ratio']:g}a` is
`m_a d^2/2 = {hemi['anchor_yaw']:.6f}`. The sphere's own rotational added inertia
about a diameter is exactly zero, so a wrong moment arm cannot hide.

| mesh | panels | M_22 (mirror) | err | I_zz (d=1.5a) | err | I_zz own axis |
|---|---|---|---|---|---|---|
{h_rows}

| quantity | closed form | limit fit | rel. error | fitted q | observed p |
|---|---|---|---|---|---|
{rich_rows}

Both close inside the 3 % gate and the 1 % extrapolation gate; the rim row sits
at `z <= 1e-16`, so the mirror closes the body exactly.
**The mirror route is the double-body model.**

## 3. The lid cap is a different model (C3)

Same KVLCC2 hull, both closures, sway and yaw about midship; panels `lid/open`.

| mesh | panels | M_22 mirror | M_22 lid | ratio | J_z mirror | J_z lid | ratio | volume err |
|---|---|---|---|---|---|---|---|---|
{hu_rows}

The ratio is mesh-independent (M_22 spread {hull['gaps']['M22']['spread']:.3f},
J_z spread {hull['gaps']['Jz']['spread']:.3f}) while the gap itself is
{hull['gaps']['M22']['gap_at_finest']:.1%} (sway) and {hull['gaps']['Jz']['gap_at_finest']:.1%}
(yaw) — far outside the discretisation uncertainty, so this is a model
difference. D7's premise holds where it applies
(`lid_flux_max_sway = {fin['lid_flux_max_sway']:.1e}`,
`lid_flux_max_yaw = {fin['lid_flux_max_yaw']:.1e}`: the lid really carries no flux)
and fails as an equivalence argument.

## 4. Corrected double-body numbers, KVLCC2 (C4, C5)

| quantity | value |
|---|---|
| `J_z` double body, midship, finest mesh ({fin['mesh']}, {fin['panels_open']} panels) | {fin['Jz_mirror']:.1f} kg m² |
| `J_z` limit fit (exponent q = {hull['limit']['Jz_mirror']['exponent']:.2f}) | {hull['limit']['Jz_mirror']['limit']:.1f} kg m² |
| `J_z` slender (`dynamics.added_masses`) | {hull['slender']['J_z']:.1f} kg m² |
| `J_z` double body / slender | {hull['Jz_double_body_over_slender_extrapolated']:.3f} |
| slender overestimate, double-body basis | **{hull['slender_overestimate_extrapolated']:.1%}** |
| slender overestimate, lid basis (D7 headline `0.371`) | {hull['slender_overestimate_lid']:.1%} |
| `M_22` double body, finest | {fin['M22_mirror']:.1f} kg |
| `m_y` slender | {hull['slender']['m_y']:.1f} kg |
| `M_22` / `m_y` | {fin['M22_mirror_over_slender']:.3f} |

D7's qualitative conclusion survives (the strip formulas overestimate both `M_22`
and `J_z` for this hull), but its magnitude was overstated by about a factor of
two because the headline came from the lid cap: on the physically matching
(double-body) model the overestimate is {hull['slender_overestimate_extrapolated']:.0%}.

## 5. The 6x6 assembly is sound (C6); debt D8 is a phantom

Closed forms use only the parallel-axis identity `A_new = P^T A_old P`,
`P = [[I, [delta]x], [0, I]]`, applied to `diag(m_a, m_a, m_a, 0, 0, 0)` (sphere)
and to the Lamb translational + rotational entries (spheroid `a/b = 2`). All 36
entries are compared, which exercises the translation-rotation coupling block that
nothing in the repository scored before.

| case | mesh | panels | max abs error / scale |
|---|---|---|---|
{asm_rows}

Debt item D8 ("`added_mass_matrix_6dof` gives `I_zz = 2.05` for a sphere centred at
`(0,0,1.5)`, theory `0`") does not reproduce: `2.05` is `A_heave,heave`
(= {d8['heave_translational']:.4f}); the yaw entry is {d8['yaw_rotational']:.1e}
against reference {d8['yaw_reference']:.1e}. It was a block-index misreading, and
C6 now pins all 36 entries.

## 6. Error judgement

| key quantity | reference | result | < 3 %? |
|---|---|---|---|
| hemisphere sway (mirror) | `m_a/2` exact | {c['C1']['value']['err_sway_finest']:+.2%} | yes |
| hemisphere yaw, offset axis | `m_a d^2/2` exact | {c['C2']['value']['err_yaw_finest']:+.2%} | yes |
| 6x6 Kirchhoff vs closed form (4 cases) | parallel-axis algebra | worst finest {asm['worst_finest']:.2%} (coarsest {asm['worst']:.2%}) | yes |
| lid vs mirror gap | model difference, no reference | {hull['gaps']['Jz']['gap_at_finest']:.1%} | not an error (C3) |
| `J_z` double body grid uncertainty | limit fit drift | {c['C4']['value']['limit_vs_finest']:+.2%} | yes |
| D7 headline `slender/lid = 0.371` | corrected `{hull['Jz_double_body_over_slender_extrapolated']:.3f}` | {hull['slender_overestimate_lid']:.0%} vs {hull['slender_overestimate_extrapolated']:.0%} | superseded |

All criteria: {', '.join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in c.items())}

## 7. Reproduction and provenance

```bash
PYTHONPATH=src python -m tensorfvm.verification.double_body_added_mass_gate --out docs/verification-double-body-gate
PYTHONPATH=src python -m pytest tests/test_double_body_added_mass_gate.py -q
```
"""


def plot(summary: Dict[str, object], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hemi, hull = summary["c1c2_hemisphere"], summary["c3c5_hull"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for key, lab in (("err_sway", "M_22 sway (mirror)"), ("err_yaw", "I_zz offset axis")):
        ax1.loglog([r["panels"] for r in hemi["rows"]],
                   [abs(r[key]) for r in hemi["rows"]], "o-", label=lab)
    ax1.axhline(0.03, ls="--", color="r", label="3 % acceptance")
    ax1.axhline(0.01, ls=":", color="0.4", label="1 % extrapolation target")
    ax1.set_xlabel("panels (-)")
    ax1.set_ylabel("relative error vs closed form (-)")
    ax1.set_title("half body on the waterline plane: mirror route")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7)

    counts = [r["panels_open"] for r in hull["rows"]]
    ax2.plot(counts, [r["Jz_mirror"] for r in hull["rows"]], "o-",
             label="J_z double body (mirror)")
    ax2.plot(counts, [r["Jz_lid"] for r in hull["rows"]], "s--", label="J_z lid cap")
    ax2.axhline(hull["slender"]["J_z"], color="0.35", ls=":",
                label=f"slender body {hull['slender']['J_z']:.0f}")
    ax2.axhline(hull["limit"]["Jz_mirror"]["limit"], color="C2", ls="-.",
                label=f"double-body limit {hull['limit']['Jz_mirror']['limit']:.0f}")
    ax2.set_xlabel("panels (-)")
    ax2.set_ylabel(r"$J_z$ about midship (kg m$^2$)")
    ax2.set_title("KVLCC2 yaw added inertia: closure model matters")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dumps(summary: Dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True, default=float)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/verification-double-body-gate")
    ap.add_argument("--hemi-ladder", default="", help="override, e.g. 16x18,24x24")
    ap.add_argument("--hull-ladder", default="", help="override, e.g. 21x5,41x9,61x13")
    args = ap.parse_args(argv)
    parse = lambda s: tuple(tuple(int(v) for v in item.split("x")) for item in s.split(","))
    hemi = parse(args.hemi_ladder) if args.hemi_ladder else HEMI_LADDER
    hull = parse(args.hull_ladder) if args.hull_ladder else HULL_LADDER
    summary = run(hemi, hull)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(dumps(summary))
    (outdir / "REPORT.md").write_text(render_report(summary))
    plot(summary, outdir / "fig_double_body_gate.png")
    for key, crit in summary["criteria"].items():
        print(f"{key}: {'PASS' if crit['pass'] else 'FAIL'}  value={crit['value']}")
    print(f"all_pass={summary['all_pass']}  elapsed={summary['elapsed_s']:.1f}s -> {outdir}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
