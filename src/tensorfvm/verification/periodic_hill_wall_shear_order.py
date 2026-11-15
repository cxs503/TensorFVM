"""How much of a published ``x_r/H`` is post-processing, not physics?

Why this module exists (round 24 of ``CASE-PERIODIC-HILLS``)
-----------------------------------------------------------
Every ``x_r/H`` this repository has published for the periodic hill is read off
``tau_b(x)`` -- the streamwise wall-shear profile.  Round 17 certified the
*crossing* operator that turns that profile into a separation/reattachment pair
(:mod:`tensorfvm.verification.wall_shear_bubbles`), round 21 cross-checked the
profile against the raw ``mean_u`` field, round 22 made the *bubble object*
explicit.  What nobody has audited is the step in front of all three: the
**wall-normal derivative** that produces ``tau_b`` in the first place.

:meth:`tensorfvm.periodic_hill.PeriodicHillSolver.wall_shear` uses

    tau_b(x) = (u_0 . t) / dn_0(x)

with ``u_0`` the wall-adjacent cell-centred velocity, ``t`` the wall tangent and
``dn_0`` the projected distance from the wall face to that cell centre.  For a
profile that is linear in the wall-normal coordinate this is exact; for any
curved profile it carries an ``O(dn)`` error, and ``dn`` near the wall is
``1/ny`` of the local channel height (~0.03 H at ``ny = 48``), i.e. the size of
the recirculation features being measured is *not* much bigger than the
truncation scale.

``tau_b``'s zero crossings are invariant under a *constant* rescaling, so this
module's first result is a warning-as-a-test: multiplying ``tau_b`` by any
positive constant leaves ``x_s`` and ``x_r`` bit-identical
(:func:`crossing_scale_invariance`).  Any *x-dependent* bias, however, moves the
crossings -- and ``dn_0(x)`` does depend on ``x`` on a terrain-following mesh.
So the question is quantitative, not rhetorical:

    how far does the published ``x_r/H`` move when the wall-normal derivative
    is computed with a quadratic (order 2) or cubic (order 3) one-sided stencil
    instead of the solver's two-point (order 1) one?

Methods
-------
``wall_gradient(u, v, mesh, order, wall)`` returns ``d u_t / d n`` at the wall
face centres.  All orders share the same geometry (the wall-normal distances
``s_j`` are the cumulated projected face distances of
:class:`tensorfvm.periodic_hill.PeriodicHillMesh`, and the tangential velocity
is the same projection the solver uses), and all impose the exact no-slip value
``u_t(0) = 0`` at the wall.  They differ only in how many cell rows they read:

* ``order=1``: ``a1 = u_t(s_0) / s_0``                     (the solver's operator)
* ``order=2``: ``a1 s + a2 s^2`` through rows 0-1
* ``order=3``: ``a1 s + a2 s^2 + a3 s^3`` through rows 0-2

Their accuracy is *proved* on an analytic case, not asserted: on a flat wall
(``hill_amplitude = 0``) the plane-Poiseuille solution
(:func:`tensorfvm.periodic_hill.analytic_flat_channel`) is exactly quadratic in
``s``, so order 2 and 3 reproduce the exact wall gradient to round-off while
order 1 converges at first order in ``dn``
(``tests/test_periodic_hill_wall_shear_order.py``).  On the hill mesh the
obliquity of the eta-lines is *common* to all orders (they all read the profile
along an eta-line), so the order-to-order difference isolates the
profile-curvature error -- which is exactly the part the stencil order controls.
It is therefore an honest *lower bound* on the full post-processing
uncertainty, never an upper bound.

What this module does **not** do: it does not re-integrate anything, does not
touch the solver, and does not re-run any case.  It reads stored mean fields.
"""

from __future__ import annotations

import numpy as np

from ..periodic_hill import PeriodicHillMesh

__all__ = [
    "wall_normal_distances",
    "wall_tangential_velocity",
    "wall_gradient",
    "wall_shear_profile",
    "crossing_scale_invariance",
    "order_sensitivity",
    "ORDER_MEANINGS",
]

ORDER_MEANINGS = {
    1: "two-point one-sided: u_t(s0)/s0 (PeriodicHillSolver.wall_shear)",
    2: "quadratic one-sided through rows 0-1, exact no-slip at the wall",
    3: "cubic one-sided through rows 0-2, exact no-slip at the wall",
}


def _wall_arrays(mesh: PeriodicHillMesh, wall: str):
    """Wall tangent, face abscissa and the row ordering seen from that wall."""
    if wall == "bottom":
        return mesh.tau_bottom, mesh.x_n[0], 1
    if wall == "top":
        return mesh.tau_top, mesh.x_n[-1], -1
    raise ValueError(f"wall must be 'bottom' or 'top', got {wall!r}")


def wall_normal_distances(mesh: PeriodicHillMesh, wall: str = "bottom",
                          rows: int | None = None) -> np.ndarray:
    """Cumulated wall-normal distance of the first ``rows`` cell rows.

    Returns an array ``(rows, nx)``; ``s[0]`` is the distance from the wall face
    to the first cell centre, ``s[j]`` that to row ``j`` (counted from the wall).
    Both walls are raised from the same projected face distances
    ``mesh.dn_n``, from level ``0`` at the bottom and ``ny`` at the top.
    """
    _wall_arrays(mesh, wall)  # validates ``wall``
    ny = mesh.config.ny
    n = ny if rows is None else int(rows)
    if not 1 <= n <= ny:
        raise ValueError(f"rows must be in [1, {ny}], got {rows}")
    levels = range(n) if wall == "bottom" else range(ny, ny - n, -1)
    s = np.zeros((n, mesh.config.nx))
    acc = np.zeros(mesh.config.nx)
    for j, level in enumerate(levels):
        acc = acc + mesh.dn_n[level]
        s[j] = acc
    return s


def wall_tangential_velocity(u, v, mesh: PeriodicHillMesh,
                             wall: str = "bottom",
                             rows: int | None = None) -> np.ndarray:
    """``u . t`` (rows from the wall, wall face centres) as the solver projects it."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = mesh.config.ny, mesh.config.nx
    if u.shape != (ny, nx) or v.shape != (ny, nx):
        raise ValueError(f"u and v must have shape {(ny, nx)}, got {u.shape}/{v.shape}")
    tangent, _, step = _wall_arrays(mesh, wall)
    n = ny if rows is None else int(rows)
    idx = [0 + j for j in range(n)] if step > 0 else [ny - 1 - j for j in range(n)]
    return np.stack([u[j] * tangent[:, 0] + v[j] * tangent[:, 1] for j in idx])


def _fit_wall_gradient(ut: np.ndarray, s: np.ndarray, order: int) -> np.ndarray:
    """One-sided polynomial fit with the wall value pinned at zero.

    ``ut`` and ``s`` both have shape ``(k, nx)`` with ``k >= order``; returns the
    derivative at ``s = 0``, shape ``(nx,)``.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if ut.shape[0] < order:
        raise ValueError(f"order {order} needs at least {order} cell rows, "
                         f"got {ut.shape[0]}")
    if order == 1:
        return ut[0] / s[0]
    # power basis s^1 .. s^order, no constant term (no-slip is exact)
    powers = np.arange(1, order + 1, dtype=float)
    # one independent order x order system per column: point m reads
    #   sum_k a_{k+1} s_m ** (k+1) = u_t(s_m)
    # so A[x, m, k] = s_m(x) ** powers_k
    A = s[:order].T[:, :, None] ** powers[None, None, :]
    b = ut[:order].T  # (nx, order)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):  # pragma: no cover
        raise ValueError("non-finite wall stencil input")
    try:
        coeff = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:  # pragma: no cover - geometry guard
        raise ValueError(f"degenerate wall stencil (order {order})") from exc
    return coeff[:, 0]


def wall_gradient(u, v, mesh: PeriodicHillMesh, order: int = 1,
                  wall: str = "bottom") -> np.ndarray:
    """``d u_t / d n`` at the wall (kinematic strain rate, ``1/s``).

    ``order`` selects the one-sided stencil (1, 2 or 3 -- see the module
    docstring); the geometry and the tangential projection are identical for all
    three, so differences between them are pure truncation-order effects.
    ``order=1`` reproduces
    :meth:`tensorfvm.periodic_hill.PeriodicHillSolver.wall_shear` bit-for-bit.
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2 or 3, got {order}")
    rows = max(order, 1)
    s = wall_normal_distances(mesh, wall=wall, rows=rows)
    ut = wall_tangential_velocity(u, v, mesh, wall=wall, rows=rows)
    return _fit_wall_gradient(ut, s, order)


def wall_shear_profile(u, v, mesh: PeriodicHillMesh, order: int = 1,
                       wall: str = "bottom"):
    """``(x_over_h, d u_t / d n)`` at the wall face centres."""
    grad = wall_gradient(u, v, mesh, order=order, wall=wall)
    _, x_n, _ = _wall_arrays(mesh, wall)
    return x_n / mesh.config.height_m, grad


def crossing_scale_invariance(tau, x_over_h, *, primary: str = "first"):
    """``x_s``/``x_r`` before and after multiplying ``tau`` by a positive constant.

    The certified crossing operator is applied to ``tau`` and to ``1489 * tau``.
    Bit-identical output is the *reason* the constant factor hidden in
    ``tau = d u_t / d n`` (no viscosity, no density) cannot move a published
    ``x_r/H`` -- and the reason only the ``x``-dependent part of the truncation
    error is worth auditing.
    """
    from .periodic_hill_field_metrics import bubble_object_report

    tau = np.asarray(tau, dtype=float)
    base = bubble_object_report(tau, x_over_h, primary=primary)
    scaled = bubble_object_report(1489.0 * tau, x_over_h, primary=primary)
    keys = ("separation", "reattachment")
    same = all(base["crest"][k] == scaled["crest"][k] for k in keys)
    return {"tau_scale": 1489.0, "identical": bool(same),
            "unscaled": {k: base["crest"][k] for k in keys},
            "scaled": {k: scaled["crest"][k] for k in keys},
            "n_bubbles": base["n_bubbles"], "flag": base["flag"]}


def order_sensitivity(u, v, mesh: PeriodicHillMesh, *, orders=(1, 2, 3),
                      x_over_h=None, primary: str = "first",
                      wall: str = "bottom") -> dict:
    """``x_r/H`` and its order-to-order spread for one stored mean field.

    ``x_over_h`` defaults to the wall-face abscissae of ``mesh`` (they depend on
    nothing but ``nx``).  ``primary="first"`` selects the **crest bubble**, which
    is the object the Breuer et al. 2009 reference ``x_r/H = 5.41`` refers to
    (round 22); pass ``"largest_area"`` for the historical ``口径``.

    Returns ``{"orders": {order: {x_s, x_r, n_bubbles, flag}}, "x_r", "spread",
    "spread_rel", "reference_order"}`` with ``spread = max - min`` over the
    requested orders and ``spread_rel`` that difference over the order-1 value.
    """
    from .periodic_hill_field_metrics import bubble_object_report

    if x_over_h is None:
        _, x_n, _ = _wall_arrays(mesh, wall)
        x_over_h = x_n / mesh.config.height_m
    x_over_h = np.asarray(x_over_h, dtype=float)

    def _finite(value):
        # the certified crossing operator returns nan when the profile has no
        # bubble; report that as None ("not measured"), never as a number
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    report = {}
    for order in orders:
        tau = wall_gradient(u, v, mesh, order=order, wall=wall)
        bubbles = bubble_object_report(tau, x_over_h, primary=primary)
        crest = bubbles["crest"]
        report[int(order)] = {
            "x_s": _finite(crest["separation"]),
            "x_r": _finite(crest["reattachment"]),
            "n_bubbles": int(bubbles["n_bubbles"]),
            "flag": bubbles["flag"],
            "selected_is_crest": bool(bubbles["selected_is_crest"]),
            "x_r_largest_area": _finite(bubbles["x_r_over_H"]["largest_area"]),
        }

    xs = [r["x_r"] for r in report.values() if r["x_r"] is not None]
    base = report[int(orders[0])]["x_r"]
    if len(xs) < 2 or base is None:
        spread = spread_rel = None
    else:
        spread = float(max(xs) - min(xs))
        spread_rel = float(spread / abs(base)) if base else float("inf")
    return {"orders": report, "x_r": [report[int(o)]["x_r"] for o in orders],
            "n_measured": len(xs),
            "spread": spread, "spread_rel": spread_rel,
            "reference_order": int(orders[0]),
            "primary": primary, "wall": wall}