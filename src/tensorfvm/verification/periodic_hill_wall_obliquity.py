"""Is the published ``x_r/H`` a *mesh-obliquity* artefact, not truncation order?

Why this module exists (round 25 of ``CASE-PERIODIC-HILLS``)
------------------------------------------------------------
Round 24 audited the **stencil order** of the wall-normal derivative that feeds
every published ``tau_b(x)`` and left one limitation explicit:

    "all three orders read the profile along an *eta line*, and near the wall
    the eta line is not perpendicular to the wall (``|beta/alpha|`` up to
    ~0.79).  The order-to-order band therefore removes only the
    profile-curvature error; it is a lower bound on the post-processing
    uncertainty, never an upper bound."

This module closes that hole.  On the terrain-following mesh of
:class:`tensorfvm.periodic_hill.PeriodicHillMesh` the vector from a wall face
centre ``W`` to the centre of the wall-adjacent cell ``C`` decomposes exactly as

    d0 = C - W = alpha * n + beta * t,          alpha = d0 . n,  beta = d0 . t

with ``n`` the inward face normal and ``t`` the downstream wall tangent (both
unit, ``n . t = 0``; the decomposition is verified to round-off by
``tests/test_periodic_hill_wall_obliquity.py``).  For any smooth scalar field
``phi`` the two-point one-sided derivative the solver uses is therefore

    (phi(C) - phi(W)) / alpha  =  (grad phi . n)  +  (beta/alpha) (grad phi . t)

i.e. the published wall shear carries an **obliquity bias** that is *first order
in* ``beta/alpha`` -- a quantity that reaches 0.85 on this mesh, so the concern
is not cosmetic.  Only the ``(beta/alpha)(grad phi . t)`` term can move a zero
crossing (a constant factor cannot, round 24), and on this geometry
``alpha ~ 0.018-0.032 H`` is small compared with the hill's tangential scale,
which is what makes the term small in practice.  How small is measured here, not
argued.

Three estimators of ``d u_t / d n`` at the wall
-----------------------------------------------
``raw``       ``(u_t(C) - u_t(W)) / alpha`` -- the solver's operator
              (:meth:`PeriodicHillSolver.wall_shear`), reproduced bit-for-bit by
              :func:`tensorfvm.verification.periodic_hill_wall_shear_order.wall_gradient`.
``deferred``  ``raw - (beta/alpha) d u_t / d s`` -- the same two-point stencil
              with the tangential part removed using the wall-parallel
              derivative of ``u_t`` at the wall-adjacent row
              (:func:`wall_gradient_deferred`).  For a field that is *affine*
              in ``(x, y)`` this is exact to round-off, at any obliquity.
``lsq``       a 2-D least-squares gradient at the wall built from the wall point
              (no-slip, ``u_t = 0``) and the three wall-adjacent cell centres
              around it (:func:`wall_gradient_lsq`).  It never mentions the eta
              line, so it has **no obliquity bias at all**; what it keeps is an
              ``O(alpha)`` truncation of the wall-normal profile, which is the
              same order the raw estimator has.

So ``deferred - raw`` isolates obliquity at first order, and ``lsq - raw`` is an
independent operator difference that mixes obliquity and the two operators'
different truncation behaviour.  Both are measured on all stored artefacts; the
analytic anchors are machine-precision, not "close enough".

What this module does **not** do: it does not touch the solver, does not re-run
any case and does not change any published number.  It adds one error axis to
the uncertainty budget of a published metric.
"""

from __future__ import annotations

import numpy as np

from ..periodic_hill import PeriodicHillMesh

__all__ = [
    "ESTIMATORS",
    "wall_geometry",
    "wall_tangential_velocity",
    "wall_tangential_derivative",
    "wall_gradient_deferred",
    "wall_gradient_lsq",
    "obliquity_ratio",
    "obliquity_sensitivity",
]

ESTIMATORS = {
    "raw": "two-point eta-line one-sided derivative (PeriodicHillSolver.wall_shear)",
    "deferred": "raw with the tangential (beta/alpha) * du_t/ds term removed",
    "lsq": "2-D least-squares gradient at the wall (obliquity free)",
}


def _wall_level(mesh: PeriodicHillMesh, wall: str) -> int:
    if wall == "bottom":
        return 0
    if wall == "top":
        return mesh.config.ny
    raise ValueError(f"wall must be 'bottom' or 'top', got {wall!r}")


def _row_indices(mesh: PeriodicHillMesh, wall: str, rows: int = 1):
    """Cell-row indices counted from the wall (``rows`` rows, wall first)."""
    level = _wall_level(mesh, wall)
    ny = mesh.config.ny
    if wall == "bottom":
        idx = [j for j in range(rows)]
    else:
        idx = [ny - 1 - j for j in range(rows)]
    if not 1 <= rows <= ny:  # pragma: no cover - guarded by callers
        raise ValueError(f"rows must be in [1, {ny}], got {rows}")
    return level, idx


def wall_geometry(mesh: PeriodicHillMesh, wall: str = "bottom") -> dict:
    """Exact wall-local frame and the ``d0 = alpha n + beta t`` decomposition.

    Returns a dict with, per wall face centre (``nx`` entries):

    ``normal`` ``tangent``
        inward unit normal and downstream unit tangent, shape ``(nx, 2)``.
    ``alpha`` ``beta``
        normal and tangential components of ``d0 = C - W``.
    ``obliquity``
        ``|beta / alpha|`` -- ``0`` on a flat wall, up to ~0.85 on the hill.
    ``d0``
        the wall-to-cell vector, minimum image in ``x``, shape ``(nx, 2)``.
    ``spacing``
        arc length from wall node ``i`` to ``i + 1`` (periodic).
    """
    level = _wall_level(mesh, wall)
    cell_row = _row_indices(mesh, wall, 1)[1][0]
    length = mesh.config.length_m
    normal = np.stack([mesh.unx_n[level], mesh.uny_n[level]], axis=-1)
    if wall == "top":
        # the top deck's stored normal points up (out of the fluid); the
        # decomposition this module is built on uses the *inward* normal so that
        # alpha > 0 on both walls and du_t/dn keeps the solver's sign
        normal = -normal
    tangent = mesh.tau_bottom if wall == "bottom" else mesh.tau_top
    xw, yw = mesh.x_n[level], mesh.y_n[level]
    dx = mesh.xc[cell_row] - xw
    dx = dx - length * np.round(dx / length)          # minimum image across the seam
    d0 = np.stack([dx, mesh.yc[cell_row] - yw], axis=-1)
    alpha = d0[:, 0] * normal[:, 0] + d0[:, 1] * normal[:, 1]
    beta = d0[:, 0] * tangent[:, 0] + d0[:, 1] * tangent[:, 1]
    if np.any(alpha <= 0.0):
        raise ValueError("non-positive projected wall distance")
    dxs = np.roll(xw, -1) - xw
    dxs = dxs - length * np.round(dxs / length)
    dys = np.roll(yw, -1) - yw
    spacing = np.hypot(dxs, dys)
    return {
        "normal": normal,
        "tangent": tangent,
        "alpha": alpha,
        "beta": beta,
        "obliquity": np.abs(beta / alpha),
        "d0": d0,
        "spacing": spacing,
        "x_over_h": xw / mesh.config.height_m,
    }


def wall_tangential_velocity(u, v, mesh: PeriodicHillMesh, wall: str = "bottom",
                             rows: int = 1, tangent=None) -> np.ndarray:
    """``u . t`` for the first ``rows`` cell rows (wall-adjacent row first).

    ``tangent`` overrides the per-column tangent; passing one fixed tangent
    (``nx x 2``) is what makes the oblique-identity calibration of the tests
    machine-precision (the projected field is then genuinely affine).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = mesh.config.ny, mesh.config.nx
    if u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError(f"u and v must have shape {(ny, nx)}, got {u.shape}/{v.shape}")
    _, idx = _row_indices(mesh, wall, rows)
    t = wall_geometry(mesh, wall)["tangent"] if tangent is None else np.asarray(tangent)
    return np.stack([u[j] * t[:, 0] + v[j] * t[:, 1] for j in idx])


def _stencil(mesh: PeriodicHillMesh, wall: str):
    """Wall-adjacent row index and the periodic ``i-1, i, i+1`` column triple."""
    level, idx = _row_indices(mesh, wall, 1)
    nx = mesh.config.nx
    plus = (np.arange(nx) + 1) % nx
    minus = (np.arange(nx) - 1) % nx
    cols = np.stack([minus, np.arange(nx), plus], axis=1)
    return level, idx[0], cols, plus, minus


def _fixed_tangent_projection(u, v, mesh: PeriodicHillMesh, wall: str,
                              cols=None) -> np.ndarray:
    """``u . t_i`` on a stencil, every point projected on the *node's* tangent.

    Projecting all stencil points on the same ``t_i`` (instead of each column's
    own tangent) is what makes the projected field affine whenever the velocity
    is affine -- and it is the projection the ``d0 = alpha n + beta t``
    decomposition is written in.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = mesh.config.ny, mesh.config.nx
    if u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError(f"u and v must have shape {(ny, nx)}, got {u.shape}/{v.shape}")
    level, j, cols_default, plus, minus = _stencil(mesh, wall)
    cols = cols_default if cols is None else cols
    t = wall_geometry(mesh, wall)["tangent"]
    psi = u[j][cols] * t[:, None, 0] + v[j][cols] * t[:, None, 1]
    return {"psi": psi, "j": j, "level": level, "cols": cols,
            "plus": plus, "minus": minus, "tangent": t}


def wall_tangential_derivative(u, v, mesh: PeriodicHillMesh,
                               wall: str = "bottom") -> np.ndarray:
    """``d u_t / d t`` at the wall-adjacent row, along the downstream tangent.

    A directional derivative: central differences of the projected velocity over
    the displacement between the two neighbouring wall-adjacent cell centres,
    divided by its *projection on the wall tangent* ``t``.  For a velocity field
    that is affine in ``(x, y)`` this is exact to round-off (tested), which is
    what makes the deferred correction of :func:`wall_gradient_deferred` an
    exact obliquity removal at first order.

    The wall value of ``u_t`` itself is pinned by no-slip (zero), so this is the
    only route by which the tangential variation of the wall shear reaches the
    two-point normal derivative.
    """
    packed = _fixed_tangent_projection(u, v, mesh, wall)
    psi, plus, minus, t = packed["psi"], packed["plus"], packed["minus"], packed["tangent"]
    length = mesh.config.length_m
    xw, yw = mesh.x_n[packed["level"]], mesh.y_n[packed["level"]]
    xc, yc = mesh.xc[packed["j"]], mesh.yc[packed["j"]]
    dx = xc[plus] - xc[minus]
    dx = dx - length * np.round(dx / length)          # minimum image across the seam
    dy = yc[plus] - yc[minus]
    span = dx * t[:, 0] + dy * t[:, 1]                 # displacement . tangent
    if np.any(span <= 0.0):
        raise ValueError("degenerate wall-parallel stencil (non-positive tangent span)")
    del xw, yw
    return (psi[:, 2] - psi[:, 0]) / span


def wall_gradient_deferred(u, v, mesh: PeriodicHillMesh, wall: str = "bottom",
                           ut_wall=None) -> np.ndarray:
    """``d u_t / d n`` at the wall with the obliquity term removed.

    ``raw - (beta/alpha) * d u_t / d s``.  For an affine velocity field this is
    exact to round-off; for a curved profile it keeps the raw estimator's
    ``O(alpha)`` truncation but carries no first-order obliquity bias.
    """
    geom = wall_geometry(mesh, wall)
    ut = wall_tangential_velocity(u, v, mesh, wall=wall, rows=1)[0]
    wall_value = np.zeros_like(ut) if ut_wall is None else np.asarray(ut_wall, dtype=float)
    raw = (ut - wall_value) / geom["alpha"]
    dut_ds = wall_tangential_derivative(u, v, mesh, wall=wall)
    return raw - (geom["beta"] / geom["alpha"]) * dut_ds


def wall_gradient_lsq(u, v, mesh: PeriodicHillMesh, wall: str = "bottom",
                      ut_wall=None) -> np.ndarray:
    """``d u_t / d n`` at the wall from a 2-D least-squares gradient.

    For each wall face centre the three wall-adjacent cell centres
    ``i-1, i, i+1`` give three (over-determined) equations for the two unknown
    Cartesian components of ``grad (u . t)``; the solve never uses the eta
    line, so the result carries no obliquity bias.  ``n . grad`` is returned.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = mesh.config.ny, mesh.config.nx
    if u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError(f"u and v must have shape {(ny, nx)}, got {u.shape}/{v.shape}")
    level, idx = _row_indices(mesh, wall, 1)
    j = idx[0]
    length = mesh.config.length_m
    geom = wall_geometry(mesh, wall)
    t = geom["tangent"]
    xw, yw = mesh.x_n[level], mesh.y_n[level]
    plus = (np.arange(nx) + 1) % nx
    minus = (np.arange(nx) - 1) % nx
    cols = np.stack([minus, np.arange(nx), plus], axis=1)      # (nx, 3)
    dx = mesh.xc[j][cols] - xw[:, None]
    dx = dx - length * np.round(dx / length)
    dy = mesh.yc[j][cols] - yw[:, None]
    # project every stencil point on the *node's own* tangent: that projection
    # is an affine function of (x, y) whenever the velocity is affine, which is
    # what makes the estimator exact in the tests
    psi = u[j][cols] * t[:, None, 0] + v[j][cols] * t[:, None, 1]
    wall_value = np.zeros(nx) if ut_wall is None else np.asarray(ut_wall, dtype=float)
    rhs = psi - wall_value[:, None]
    sxx = (dx * dx).sum(axis=1)
    sxy = (dx * dy).sum(axis=1)
    syy = (dy * dy).sum(axis=1)
    rx = (dx * rhs).sum(axis=1)
    ry = (dy * rhs).sum(axis=1)
    det = sxx * syy - sxy * sxy
    if np.any(det <= 0.0):
        raise ValueError("degenerate wall least-squares stencil")
    gx = (syy * rx - sxy * ry) / det
    gy = (sxx * ry - sxy * rx) / det
    return gx * geom["normal"][:, 0] + gy * geom["normal"][:, 1]


def obliquity_ratio(mesh: PeriodicHillMesh, wall: str = "bottom") -> dict:
    """Summary of ``|beta/alpha|`` on one wall (how oblique is this mesh?)."""
    geom = wall_geometry(mesh, wall)
    ob = geom["obliquity"]
    return {
        "max": float(ob.max()),
        "mean": float(ob.mean()),
        "p95": float(np.percentile(ob, 95)),
        "at": int(np.argmax(ob)),
        "x_over_h_at_max": float(geom["x_over_h"][int(np.argmax(ob))]),
        "alpha_min": float(geom["alpha"].min()),
        "alpha_max": float(geom["alpha"].max()),
    }


def obliquity_sensitivity(u, v, mesh: PeriodicHillMesh, *, primary: str = "first",
                          wall: str = "bottom", x_over_h=None,
                          estimators=("raw", "deferred", "lsq")) -> dict:
    """``x_s/H`` and ``x_r/H`` for each wall-derivative estimator.

    ``raw`` is the published operator, so ``x_r(raw)`` is the published number;
    the spread over the requested estimators is the *obliquity/operator* axis of
    the post-processing uncertainty, to be compared with round 24's
    *truncation-order* axis -- they are independent error sources.
    """
    from .periodic_hill_field_metrics import bubble_object_report

    geom = wall_geometry(mesh, wall)
    xoh = geom["x_over_h"] if x_over_h is None else np.asarray(x_over_h, dtype=float)
    funcs = {
        "raw": lambda: wall_tangential_velocity(u, v, mesh, wall=wall, rows=1)[0]
        / geom["alpha"],
        "deferred": lambda: wall_gradient_deferred(u, v, mesh, wall=wall),
        "lsq": lambda: wall_gradient_lsq(u, v, mesh, wall=wall),
    }
    unknown = [name for name in estimators if name not in funcs]
    if unknown:
        raise ValueError(f"unknown estimator(s) {unknown}; choose from {list(funcs)}")

    def _finite(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    report = {}
    for name in estimators:
        tau = funcs[name]()
        bubbles = bubble_object_report(tau, xoh, primary=primary)
        crest = bubbles["crest"]
        report[name] = {
            "x_s": _finite(crest["separation"]),
            "x_r": _finite(crest["reattachment"]),
            "n_bubbles": int(bubbles["n_bubbles"]),
            "flag": bubbles["flag"],
            "max_abs_tau": float(np.abs(tau).max()),
            "rms_tau": float(np.sqrt(np.mean(tau ** 2))),
        }

    xs = [r["x_r"] for r in report.values() if r["x_r"] is not None]
    base = report[estimators[0]]["x_r"] if estimators else None
    if len(xs) < 2 or base in (None, 0):
        spread = spread_rel = None
    else:
        spread = float(max(xs) - min(xs))
        spread_rel = float(spread / abs(base))
    return {"estimators": report, "x_r_spread": spread,
            "x_r_spread_rel": spread_rel, "reference_estimator": estimators[0]}