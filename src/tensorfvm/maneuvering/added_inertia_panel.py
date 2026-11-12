"""Panel-method yaw added inertia of the KVLCC2 hull (verification-debt item D7).

Why this module exists
----------------------
:func:`tensorfvm.maneuvering.dynamics.added_masses` supplies ``J_z`` from
slender-body strip theory (``J_z = int rho pi T_e(x)^2 x^2 dx``).  That formula
is a documented *upper bound* -- it ignores the three-dimensional end effects of
a real hull -- but until now there was no independent value for the same ship,
so the manoeuvring equations could not state how much the placeholder mattered
(``docs/VERIFICATION_DEBT_LEDGER.md`` item D7; the rotational panel machinery
itself was verified against spheroid anchors by D1/m14707 and m14738).

This module builds the **same hull** -- the repository's calibrated KVLCC2
parametric form used by the manoeuvring model, ``Kvlcc2Hull.half_breadth(x, z)``
-- as a closed quadrilateral panel body, and evaluates

    J_z = -rho * integral_S phi (e_z x r) . n dS ,      r = x - x0 ,
    laplace(phi) = 0 ,  dphi/dn = (e_z x r) . n on S ,  phi -> 0 at infinity,

with :func:`tensorfvm.panel_added_inertia.rotational_added_inertia` (source
panel method, ``e_z x r`` rigid-rotation field, ``x0`` = midship).

The double-body closure is exact for yaw
---------------------------------------
The body is closed with a flat lid on the design waterline ``z = 0``.  For
rotation about the vertical axis the surface velocity ``e_z x r`` is
*horizontal*, so its normal component on the horizontal lid vanishes
identically: the lid carries no flux and is exactly the plane of symmetry of the
double-body (rigid-wall waterplane) model.  The unbounded-fluid solution of the
closed hull+lid body is therefore the double-body yaw flow -- the same physical
model the slender-body strip integral assumes.  Nothing else is imposed: no
free-surface elevation, no forward speed, no lifting effects.

Geometry gate (platform rule: verify the geometry before the hydrodynamics)
--------------------------------------------------------------------------
The closed body must enclose ``Cb L B T`` and its lid must have plan area
``Cwp L B``; both are checked by :func:`geometry_gate_panels` and asserted by the
tests before any inertia number is quoted.

Limitations (stated, not hidden)
--------------------------------
* The lid is a mathematical closure, not a real deck: fine for yaw (zero flux),
  *not* a valid model for sway/roll added mass, which is why only the ``I_zz``
  entry is produced here.
* The section form is the calibrated KVLCC2 parametric family (superellipse
  exponent from the sectional-area fullness), i.e. the same geometry the
  manoeuvring force model uses -- not digitized tank offsets.
* The panel value is an unbounded-fluid potential-flow result: viscosity and
  free-surface elevation (both absent from strip theory too) are out of scope.

Correction (debt D9, ``verification.double_body_added_mass_gate``)
-----------------------------------------------------------------
The lid closure is **not** the double body, so the numbers here are not directly
comparable with the slender-body strip formulas or with
:func:`sway_added_mass_panel` (which uses the waterplane-mirror route).  Zero flux
through the lid only makes the potential even across it; the fluid still exists
above ``z = 0`` in this model and streams over the deck, whereas the double body
has the mirrored hull there.  Measured on the calibrated KVLCC2 hull the lid cap
under-states ``J_z`` by ~32 % and ``M_22`` by ~36 % at every mesh, and the gap is
mesh-independent (model, not discretisation).  For the double-body (like-for-like
with strip theory) yaw value use the mirror route,
``rotational_added_inertia(panels_without_lid, images=IMAGE_SPECS["single_hull"],
origin=<midship>)``; the audited value is ``J_z = 7048 kg m^2`` against the
slender ``13411 kg m^2``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..panel_added_inertia import rotational_added_inertia
from ..twin_hull_geometry import PanelSet, _panelise_nodes
from .hull import DEFAULT_CONDITIONS, Kvlcc2Conditions, build_hull

__all__ = [
    "hull_panels",
    "deck_panels",
    "enclosed_volume",
    "lid_area",
    "geometry_gate_panels",
    "yaw_added_inertia_panel",
    "slender_yaw_added_inertia",
    "compare_yaw_added_inertia",
]


def _cluster(n: int) -> np.ndarray:
    """Cosine clustering of ``n+1`` stations on ``[0, 1]`` (both ends refined)."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * s))


def _draft_profile(hull, x: np.ndarray) -> np.ndarray:
    """Per-station draft ``T(x)`` (constant unless the form provides ``draft_at``)."""
    fn = getattr(hull, "draft_at", None)
    if fn is None:
        return np.full(np.asarray(x).shape, float(hull.T))
    return np.asarray(fn(x), dtype=float)


def _section_grid(hull, xi: np.ndarray, n_girth: int, with_lid: bool = True):
    """Closed girth grid ``(n_stations+1, 2*n_girth+3, 3)`` of the hull (+ lid).

    Column order: port waterline -> keel -> starboard waterline -> lid apex
    (``y = 0, z = 0``) -> port waterline (repeated as the last column).  The
    repeated first column closes the lid strip without leaving a gap and without
    degenerate wrap-around panels.  With ``with_lid=False`` the two lid columns
    are dropped (the body is then open on the design waterline and must be paired
    with the waterplane mirror image, see :func:`sway_added_mass_panel`).
    """
    x = xi * float(hull.L)
    draft = _draft_profile(hull, x)
    g = _cluster(n_girth)                 # 0 at the waterline -> 1 at the keel
    n_st = x.size
    n_dep = g.size
    # ``half_breadth`` broadcasts ``z`` against ``x``, so it is evaluated station
    # by station with an independent 1-D depth grid (this also keeps the
    # variable-draft form possible).
    y_half = np.empty((n_st, n_dep))
    for i in range(n_st):
        y_half[i] = np.asarray(
            hull.half_breadth(np.array([x[i]]), draft[i] * g), dtype=float).ravel()
    y_half[:, 0] = np.asarray(
        [hull.half_breadth(np.array([x[i]]), np.zeros(1)).ravel()[0]
         for i in range(n_st)], dtype=float)
    y_half[:, -1] = 0.0                   # exact keel point on the centreplane
    m = n_dep - 1                         # keel node index

    def col(y, z):
        return np.stack([x, y, -z], axis=1)

    cols = []
    for j in range(m + 1):                # port side: waterline -> keel
        cols.append(col(-y_half[:, j], draft * g[j]))
    for j in range(m - 1, -1, -1):        # starboard side: back up to the waterline
        cols.append(col(y_half[:, j], draft * g[j]))
    if with_lid:
        cols.append(col(np.zeros(n_st), np.zeros(n_st)))    # lid apex, centreplane
        cols.append(cols[0])                                # close the lid strip
    return np.stack(cols, axis=1), m


def _orient_outward(panels: PanelSet, hull, xi: np.ndarray, per_station: int,
                    with_lid: bool = True) -> None:
    """Flip normals that point into the body (reference: mid-draught keel line)."""
    x_st = xi * float(hull.L)
    x_mid = 0.5 * (x_st[:-1] + x_st[1:])
    draft_mid = 0.5 * (_draft_profile(hull, x_st[:-1])
                       + _draft_profile(hull, x_st[1:]))
    per_station = panels.n_panels // x_mid.size
    ref = np.stack([np.repeat(x_mid, per_station),
                    np.zeros(panels.n_panels),
                    -np.repeat(0.5 * draft_mid, per_station)], axis=1)
    outward = np.einsum("ij,ij->i", panels.normals, panels.centroids - ref)
    flip = outward < 0.0
    panels.normals[flip] *= -1.0
    if panels.corners is not None:
        panels.corners[flip] = panels.corners[flip][:, ::-1]


def hull_panels(hull=None, *, n_stations: int = 41, n_girth: int = 9,
                with_lid: bool = True) -> PanelSet:
    """Quadrilateral panel body of the manoeuvring hull form.

    ``n_stations`` = longitudinal stations (cosine clustered at bow/stern),
    ``n_girth`` = girth intervals from the waterline to the keel.  With
    ``with_lid`` (default) the body is closed by the design-waterline lid and has
    ``n_stations * (2 * n_girth + 2)`` panels; without it the hull is open on the
    waterline (``n_stations * 2 * n_girth`` panels) and must be used with the
    waterplane mirror image.
    """
    hull = hull if hull is not None else build_hull()
    if n_stations < 2 or n_girth < 1:
        raise ValueError("need at least 2 stations and 1 girth interval")
    xi = _cluster(n_stations)
    grid, _ = _section_grid(hull, xi, n_girth, with_lid=with_lid)
    label = "hull+lid" if with_lid else "hull+mirror"
    body = _panelise_nodes(grid, label=label)
    _orient_outward(body, hull, xi, per_station=2 * n_girth + 2, with_lid=with_lid)
    return body


def deck_panels(panels: PanelSet) -> PanelSet:
    """The lid panels of a :func:`hull_panels` body.

    The lid is the only part of the surface whose four corners all lie on the
    design waterline, so its centroid ``z`` is exactly zero.  (Testing ``n_z``
    instead would also catch the flat-topped section panels near the waterline,
    which is why the geometric test is used.)
    """
    sel = np.abs(panels.centroids[:, 2]) < 1.0e-12 * max(
        float(np.ptp(panels.centroids[:, 0])), 1.0)
    return PanelSet(panels.centroids[sel], panels.normals[sel], panels.areas[sel],
                    None if panels.corners is None else panels.corners[sel],
                    label="kvlcc2-lid")


def enclosed_volume(panels: PanelSet) -> float:
    """Volume enclosed by a closed panel body (divergence theorem)."""
    return float(np.sum(np.einsum("ij,ij->i", panels.centroids, panels.normals)
                        * panels.areas)) / 3.0


def lid_area(panels: PanelSet) -> float:
    """Plan area of the lid panels [m^2] (should equal ``Cwp L B``)."""
    return float(np.sum(deck_panels(panels).areas))


def geometry_gate_panels(panels: PanelSet, hull=None) -> Dict[str, object]:
    """Panel-body geometry gate: enclosed volume vs ``Cb L B T``, lid vs ``Cwp L B``.

    Tolerances: enclosed volume within 2 % of ``Cb L B T``, lid plan area within
    1 % of ``Cwp L B``.  The volume tolerance has to absorb the *inscribed*
    polyhedral error of the nearly box-shaped midbody sections on the default
    41x9 mesh (measured -1.2 %; the ladder in
    ``docs/maneuvering-added-inertia/REPORT.md`` closes to -0.56 % at 61x13), so
    it is a mesh-quality gate, not a statement about the hull form -- the form
    itself matches the fitted integrals to 1e-4.
    """
    hull = hull if hull is not None else build_hull()
    c = DEFAULT_CONDITIONS
    v_target = c.block_coefficient * hull.L * hull.B * hull.T
    a_target = c.waterplane_coefficient * hull.L * hull.B
    volume = enclosed_volume(panels)
    area = lid_area(panels)
    checks = {
        "volume": {"panel": volume, "target": v_target,
                   "relative_error": (volume - v_target) / v_target},
        "lid_area": {"panel": area, "target": a_target,
                     "relative_error": (area - a_target) / a_target},
    }
    passed = (abs(checks["volume"]["relative_error"]) <= 0.02
              and abs(checks["lid_area"]["relative_error"]) <= 0.01)
    return {"passed": bool(passed), "checks": checks,
            "n_panels": int(panels.n_panels), "surface_area": panels.total_area}


def sway_added_mass_panel(hull=None, *, n_stations: int = 41, n_girth: int = 9,
                          rho: Optional[float] = None, order: int = 3,
                          hull_obj=None) -> float:
    """Double-body sway added mass ``M_22`` of the same hull [kg].

    The waterline is the rigid-wall plane of the double-body model, so the wet
    surface is paired with its waterplane mirror image
    (``IMAGE_SPECS["single_hull"]``) and no lid is used -- this is the
    two-dimensional strip model's three-dimensional counterpart.  It is a
    *corroborating* number for the yaw finding, not a verification target.
    """
    from ..panel_added_mass import added_mass_matrix
    from ..panel_potential import IMAGE_SPECS

    hull = hull_obj if hull_obj is not None else (
        hull if hull is not None else build_hull())
    rho = DEFAULT_CONDITIONS.rho if rho is None else float(rho)
    panels = hull_panels(hull, n_stations=n_stations, n_girth=n_girth,
                         with_lid=False)
    m = added_mass_matrix(panels, rho=rho, directions=(1,), order=order,
                          images=IMAGE_SPECS["single_hull"])
    return float(m[1, 1])


def yaw_added_inertia_panel(panels: PanelSet, *, rho: Optional[float] = None,
                            origin=None, order: int = 3) -> float:
    """``I_zz`` about the vertical axis through ``origin`` (default: midship).

    The moment arm of a rotation about a vertical axis does not depend on ``z``,
    so any point on that axis gives the same value; the default is midship,
    matching the ``x`` measured from ``L/2`` in the slender-body integral.
    """
    rho = DEFAULT_CONDITIONS.rho if rho is None else float(rho)
    if origin is None:
        x_mid = 0.5 * (float(panels.centroids[:, 0].min())
                       + float(panels.centroids[:, 0].max()))
        origin = np.array([x_mid, 0.0, 0.0])
    out = rotational_added_inertia(panels, rho=rho, axes=(2,),
                                   origin=np.asarray(origin, dtype=float),
                                   order=order)
    return float(out[2, 2])


def slender_yaw_added_inertia(model_hull=None, n: int = 121) -> Dict[str, float]:
    """Slender-body ``m_y``/``J_z`` of the same hull (the current placeholder).

    Delegates to :func:`tensorfvm.maneuvering.dynamics.added_masses` so the two
    routes can never drift apart; the import is local to keep this module free of
    a circular dependency at import time.
    """
    from .dynamics import HullForceModelSet, added_masses

    model_hull = model_hull if model_hull is not None else HullForceModelSet()
    return added_masses(model_hull, n=n)


def compare_yaw_added_inertia(*, n_stations: int = 41, n_girth: int = 9,
                              hull=None) -> Dict[str, object]:
    """Panel vs slender-body yaw added inertia of the same hull."""
    hull = hull if hull is not None else build_hull()
    c = DEFAULT_CONDITIONS
    panels = hull_panels(hull, n_stations=n_stations, n_girth=n_girth)
    gate = geometry_gate_panels(panels, hull)
    j_panel = yaw_added_inertia_panel(panels)
    slender = slender_yaw_added_inertia()
    j_slender = float(slender["J_z"])
    return {
        "n_stations": int(n_stations),
        "n_girth": int(n_girth),
        "n_panels": int(panels.n_panels),
        "geometry": gate,
        "J_z_panel": j_panel,
        "J_z_slender": j_slender,
        "panel_over_slender": j_panel / j_slender,
        "panel_over_slender_minus_one": j_panel / j_slender - 1.0,
        "J_z_panel_over_mass_L2": j_panel / (c.mass * c.length ** 2),
        "J_z_slender_over_mass_L2": j_slender / (c.mass * c.length ** 2),
        "m_y_slender": float(slender["m_y"]),
    }