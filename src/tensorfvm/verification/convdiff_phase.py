"""Phase-accuracy sweep for the 1-D convection-diffusion operator.

Case ``CASE-FVM-CONVDIFF-PHASE`` (TensorFVM case pool)
-----------------------------------------------------
Linear convection-diffusion equation on a periodic interval::

    dphi/dt + c dphi/ds = D d2phi/ds2 ,      phi(s, 0) = sin(k s),  k = 2 pi / L

with the classical travelling-wave solution (separation of variables for the
advection-diffusion operator; the closed form is certified *in code* by the PDE
residual probe :func:`pde_residual`, never transcribed):

    phi(s, t) = exp(-D k^2 t) * sin(k (s - c t)).

The reference wave travels at exactly ``c`` (phase never changes) and decays at
the exactly known rate ``exp(-D k^2 t)``.  A scheme that is not non-dispersive
shifts the phase, so the phase of the computed wave against the analytic sine is
a one-number dispersion probe.

What is exercised (no new transport code)
-----------------------------------------
The sweep marches the *shipped* ALE scalar transport kernel
:func:`tensorfvm.moving_mesh.ale_scalar_step` (explicit first-order upwind
convection + point-implicit diffusion, first-order in time) on a *closed* ring
mesh built by the shipped :class:`StructuredCurvilinearMesh` with
``periodic=(True, False, False)``.  On a static mesh the ALE sweep term is zero,
so the kernel is the ordinary cell-centred convection-diffusion update; the ring
turns the periodic direction into a genuinely closed geometry (a straight block
cannot be periodic -- see the friction note in the report), and the transverse
wall faces carry no flux (zero-gradient closure adds the same conductance to
diagonal and numerator, so it cancels exactly).

Measurement layer added here
----------------------------
* :func:`ring_coefficients` reads the *measured* dimensionless stencil
  coefficients ``C = u dt A/V`` and ``a = D dt A/(dperp V)`` off the mesh metrics
  and reports their uniformity, so nothing about the scheme's constants is
  assumed;
* :func:`discrete_symbol` is the closed-form amplification factor of that
  stencil (von Neumann analysis)::

      G(theta) = [1 - C (1 - e^{-i theta}) + 2 a cos(theta)] / (1 + 2 a)

  with per-step phase error ``arg G + C theta`` and the classical leading-order
  upwind result ``C theta^3 (1-C)(1-2C)/6`` (phase-exact at ``C = 1/2`` and
  ``C = 1``);
* the modified-equation amplitude prediction ``D_num = c ds (1-C)/2`` used as the
  reference for the amplitude decay of the diffusive-free scheme.
"""
from __future__ import annotations

import math

import numpy as np

from ..moving_mesh import StructuredCurvilinearMesh, ale_scalar_step

__all__ = [
    "sine_exact",
    "pde_residual",
    "build_periodic_ring",
    "ring_coefficients",
    "discrete_symbol",
    "upwind_phase_error_leading_order",
    "upwind_numerical_diffusivity",
    "phase_metrics",
    "march_sine",
    "courant_sweep",
    "grid_sweep",
    "diffusivity_sweep",
    "observed_order",
]


# ---------------------------------------------------------------------------
# reference: analytic sine solution, certified by a PDE residual
# ---------------------------------------------------------------------------
def sine_exact(s, t, *, c: float, D: float, k: float):
    """Exact solution ``exp(-D k^2 t) sin(k (s - c t))`` of the model problem."""
    s = np.asarray(s, dtype=float)
    return np.exp(-D * k * k * t) * np.sin(k * (s - c * t))


def pde_residual(*, c: float = 1.0, D: float = 1.0e-2, L: float = 1.0,
                 t: float = 0.37) -> float:
    """Max ``|phi_t + c phi_x - D phi_xx|`` of :func:`sine_exact` (must be ~0).

    Central differences in time and space on a fine grid: a wrong reference
    formula would show a macroscopic residual instead of hiding inside the
    truncation error.  This is the "zero transcription" guarantee for the target
    values -- the reference is a verified closed form, not a copied number.
    """
    n = 2048
    s = np.arange(n) * (L / n)
    h = L / n
    k = 2.0 * math.pi / L
    f = lambda tt, ss: sine_exact(ss, tt, c=c, D=D, k=k)
    ht = 1.0e-6
    phi_t = (f(t + ht, s) - f(t - ht, s)) / (2.0 * ht)
    phi_s = (f(t, s + h) - f(t, s - h)) / (2.0 * h)
    phi_ss = (f(t, s + h) - 2.0 * f(t, s) + f(t, s - h)) / h ** 2
    return float(np.abs(phi_t + c * phi_s - D * phi_ss).max())


# ---------------------------------------------------------------------------
# closed-form theory of the shipped stencil
# ---------------------------------------------------------------------------
def discrete_symbol(C: float, theta: float, a: float = 0.0, w: float = 0.0) -> complex:
    """Amplification factor of the shipped one-step stencil (von Neumann).

    ``w`` is the measured patch-closure term ``dt sum_wall diff / V``: the
    zero-gradient patch closure puts the same conductance in the point-implicit
    diagonal and in the numerator, so a uniform field is untouched while every
    *change* of a varying field is diluted by ``1 + w + 2a`` (it appears on both
    sides, hence the ``1 + w`` in the numerator).  ``w`` is zero only on a mesh
    whose patch area over cell volume is negligible; it is measured here, never
    dropped.
    """
    return ((1.0 + w - C * (1.0 - np.exp(-1j * theta)) + 2.0 * a * np.cos(theta))
            / (1.0 + w + 2.0 * a))


def upwind_phase_error_leading_order(C: float, theta: float) -> float:
    """``C theta^3 (1-C)(1-2C)/6`` -- leading-order per-step phase error of upwind."""
    return C * theta ** 3 * (1.0 - C) * (1.0 - 2.0 * C) / 6.0


def upwind_numerical_diffusivity(c: float, ds: float, C: float) -> float:
    """Modified-equation diffusivity of first-order upwind, ``c ds (1-C)/2``."""
    return 0.5 * c * ds * (1.0 - C)


# ---------------------------------------------------------------------------
# geometry: closed ring so that the periodic direction is genuinely closed
# ---------------------------------------------------------------------------
def build_periodic_ring(n_theta: int, *, radius: float = 1.0,
                        thickness: float = 0.05, depth: float = 0.05) -> StructuredCurvilinearMesh:
    """Annular-sector ring: ``n_theta`` cells around, 1 radial cell, 1 z layer.

    Node plane ``n_theta`` coincides with plane ``0`` (closed geometry), which is
    what ``periodic=(True, False, False)`` requires; the ``i`` direction is the
    periodic one, the radial and spanwise directions are walls.
    """
    th = np.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    rad = np.array([radius - 0.5 * thickness, radius + 0.5 * thickness])
    zz = np.array([0.0, depth])
    nodes = np.empty((n_theta + 1, 2, 2, 3))
    nodes[..., 0] = np.cos(th)[:, None, None] * rad[None, :, None]
    nodes[..., 1] = np.sin(th)[:, None, None] * rad[None, :, None]
    nodes[..., 2] = zz[None, None, :]
    patches = {(1, "low"): "wall", (1, "high"): "wall",
               (2, "low"): "wall", (2, "high"): "wall"}
    return StructuredCurvilinearMesh(nodes, periodic=(True, False, False), patches=patches)


def ring_coefficients(n_theta: int, dt: float, *, speed: float, diffusivity: float,
                      radius: float = 1.0, thickness: float = 0.05,
                      depth: float = 0.05) -> dict:
    """Measured stencil coefficients and effective length scale of the ring.

    Returns ``C``, ``a`` (per-step dimensionless convection/diffusion of the
    three-point stencil), the between-centre spacing ``ds = dperp`` of the
    periodic faces, and the largest relative spread over the ring -- a spread
    above a few percent would invalidate the "uniform 1-D stencil" reading.
    """
    mesh = build_periodic_ring(n_theta, radius=radius, thickness=thickness, depth=depth)
    g = mesh.metrics()
    interior = mesh.interior
    if int(interior.sum()) != n_theta:
        raise RuntimeError("periodic direction does not pair every cell")
    area = g.area_mag[interior]
    dperp = g.dperp[interior]
    vol = g.volume
    C = dt * speed * area / vol
    a = dt * diffusivity * area / (dperp * vol)
    spread = lambda v: float((v.max() - v.min()) / v.mean()) if abs(v.mean()) > 1e-300 else 0.0
    # the zero-gradient patch closure enters the point-implicit diagonal with the
    # same conductance it puts in the numerator, so it preserves a uniform field
    # but relaxes every *varying* field towards its own old value; that spurious
    # term is measured here (w = dt * sum_wall diff / V) instead of assumed O(0)
    diff = diffusivity * g.area_mag / g.dperp
    diag_wall = np.bincount(mesh.owner[~interior], weights=diff[~interior],
                            minlength=mesh.n_cells)
    return {
        "mesh": mesh, "geom": g, "C": float(C.mean()), "a": float(a.mean()),
        "area": float(area.mean()), "volume": float(vol.mean()),
        "ds": float(dperp.mean()), "volume_over_area": float((vol / area).mean()),
        "spread_C": spread(C), "spread_a": spread(a), "spread_volume": spread(vol),
        "spread_dperp": spread(dperp),
        "w": float((dt * diag_wall / vol).mean()), "spread_w": spread(dt * diag_wall / vol),
    }


# ---------------------------------------------------------------------------
# measurement layer
# ---------------------------------------------------------------------------
def phase_metrics(phi, *, steps: int, C: float, a: float, theta: float) -> dict:
    """Least-squares sine fit of the computed ring field, in index space.

    On the ring cell ``i`` carries the mode ``sin(theta i)`` with
    ``theta = 2 pi / N``, so the fit projects on ``{sin theta i, cos theta i}``:
    ``phi ~ R sin(theta (i - i_adv))``.  The exact advance after ``steps`` steps
    is ``C steps`` cells (the analytic wave moves ``c dt = C ds`` per step) and
    the exact amplitude factor is ``exp(-a theta^2 steps)``.

    Everything is measured in *index* units and only then converted to radians,
    which is where a mixed-unit bug would otherwise hide; the residual advance is
    wrapped to one period, so ``steps`` may carry the wave past a full turn.
    A positive ``lag_*`` means the numerical wave is **behind** the analytic one
    (the classical statement for first-order upwind at ``C < 1/2``).
    """
    phi = np.asarray(phi, dtype=float)
    n = phi.size
    i = np.arange(n)
    s, cs = np.sin(theta * i), np.cos(theta * i)
    A = 2.0 * float(np.sum(phi * s)) / n
    B = 2.0 * float(np.sum(phi * cs)) / n
    R = float(math.hypot(A, B))
    i_adv = math.atan2(-B, A) / theta                 # index units, wrapped fit
    i_exact = C * steps
    d = i_adv - i_exact                               # index units
    d -= n * round(d / n)                             # wrap to one period
    lag_rad = -d * theta                              # positive = behind
    amp_exact = math.exp(-a * theta * theta * steps)
    ref = np.sin(theta * (i - i_exact))
    return {
        "amplitude": R,
        "amplitude_exact": amp_exact,
        "amplitude_ratio": R / amp_exact,
        "advance_measured": i_adv,
        "advance_exact": i_exact,
        "lag_rad": lag_rad,
        "lag_deg": lag_rad * 180.0 / math.pi,
        "lag_per_step": lag_rad / steps,
        "lag_wavelength": lag_rad / (2.0 * math.pi),
        "lag_relative": lag_rad / (C * theta * steps) if C * steps else float("nan"),
        "l2_relative": float(np.linalg.norm(phi - ref) / np.linalg.norm(np.sin(theta * i))),
    }


def march_sine(*, nx: int = 64, courant: float = 0.25,
               diffusion_number: float = 0.0, steps: int = 200,
               radius: float = 1.0, thickness: float = 0.05, depth: float = 0.05,
               keep_history: bool = False) -> dict:
    """March the analytic sine on the ring with the shipped kernel.

    The case is posed in the dimensionless stencil coordinates ``(C, a, N)``:
    the ring geometry fixes ``ds`` and ``A/V``, from which the physical kernel
    inputs follow as ``speed = C (V/A)/dt`` and
    ``diffusivity = a ds (V/A)/dt``.  The *effective* continuum coefficients of
    that stencil are ``c = C ds/dt`` and ``D_eff = a ds^2/dt``; both are returned
    together with the geometry factor ``ds/(V/A)`` that separates them from the
    kernel inputs.
    """
    theta = 2.0 * math.pi / nx
    probe = ring_coefficients(nx, 1.0, speed=1.0, diffusivity=0.0,
                              radius=radius, thickness=thickness, depth=depth)
    volume_over_area = probe["volume_over_area"]
    ds = probe["ds"]
    dt = 1.0
    speed = courant * volume_over_area / dt
    diffusivity = diffusion_number * ds * volume_over_area / dt
    coef = ring_coefficients(nx, dt, speed=speed, diffusivity=diffusivity,
                             radius=radius, thickness=thickness, depth=depth)
    mesh, geom = coef["mesh"], coef["geom"]
    face_velocity = np.zeros((mesh.n_faces, 3))
    # tangential unit vector at each face: the ring is centred on the z axis, so
    # the azimuthal direction at azimuth phi is (-sin phi, cos phi, 0); face
    # centres give phi directly.
    xf = geom.face_center
    r = np.hypot(xf[:, 0], xf[:, 1])
    face_velocity[:, 0] = -speed * xf[:, 1] / r
    face_velocity[:, 1] = speed * xf[:, 0] / r
    # (spanwise and radial faces need no special case: their normals are
    #  perpendicular to the tangential velocity, so their flux is zero)
    phi = np.sin(theta * np.arange(nx))
    sweep = np.zeros(mesh.n_faces)
    history = []
    for _ in range(steps):
        phi = ale_scalar_step(geom, geom, field_old=phi, face_velocity=face_velocity,
                              sweep=sweep, dt=dt, diffusivity=diffusivity)
        if keep_history:
            history.append(phi.copy())
    m = phase_metrics(phi, steps=steps, C=coef["C"], a=coef["a"], theta=theta)
    c_eff = coef["C"] * ds / dt
    D_eff = coef["a"] * ds * ds / dt
    G = discrete_symbol(coef["C"], theta, coef["a"], coef["w"])
    m.update({
        "nx": nx, "theta": theta, "steps": steps, "dt": dt,
        "courant_measured": coef["C"], "diffusion_number_measured": coef["a"],
        "courant_target": courant, "diffusion_number_target": diffusion_number,
        "spread_C": coef["spread_C"], "spread_a": coef["spread_a"],
        "spread_volume": coef["spread_volume"],
        "patch_relaxation_w": coef["w"], "spread_w": coef["spread_w"],
        "speed": speed, "diffusivity": diffusivity,
        "ds": ds, "geometry_factor_ds_over_VoverA": ds / volume_over_area,
        "c_eff": c_eff, "D_eff": D_eff,
        "cell_peclet": c_eff * ds / D_eff if D_eff > 0.0 else float("inf"),
        "symbol_phase_per_step": float(np.angle(G)),
        "symbol_phase_error_per_step": float(np.angle(G) + coef["C"] * theta),
        "theory_phase_error_per_step": upwind_phase_error_leading_order(coef["C"], theta),
        "symbol_abs_per_step": float(abs(G)),
        "D_numerical_upwind": upwind_numerical_diffusivity(c_eff, ds, coef["C"]),
        "phi": phi, "phi_initial": np.sin(theta * np.arange(nx)),
        "phi_exact": np.sin(theta * (np.arange(nx) - coef["C"] * steps)),
        "history": history,
    })
    return m


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def courant_sweep(*, nx: int = 64, courants=(0.0625, 0.125, 0.25, 0.375, 0.5,
                                            0.625, 0.75, 0.875, 1.0),
                  diffusion_number: float = 0.0, steps: int = 200) -> list[dict]:
    """Phase / amplitude error versus Courant number at fixed grid and step count."""
    out = []
    for C in courants:
        m = march_sine(nx=nx, courant=C, diffusion_number=diffusion_number, steps=steps)
        out.append({k: m[k] for k in (
            "nx", "theta", "steps", "courant_measured", "diffusion_number_measured",
            "lag_deg", "lag_per_step", "lag_wavelength",
            "lag_relative", "amplitude_ratio",
            "symbol_phase_error_per_step", "theory_phase_error_per_step",
            "symbol_abs_per_step")})
        out[-1]["courant"] = C
    return out


def grid_sweep(*, sizes=(16, 32, 64, 128), courant: float = 0.25,
               diffusion_number: float = 0.0, steps: int = 200) -> list[dict]:
    """Phase-error convergence under refinement at fixed Courant number."""
    out = []
    for nx in sizes:
        m = march_sine(nx=nx, courant=courant, diffusion_number=diffusion_number,
                       steps=steps)
        out.append({
            "nx": nx, "ds_over_L": 1.0 / nx, "theta": m["theta"], "steps": steps,
            "courant_measured": m["courant_measured"],
            "lag_rad": m["lag_rad"],
            "lag_per_step": m["lag_per_step"],
            "lag_relative": m["lag_relative"],
            "amplitude_ratio": m["amplitude_ratio"],
            "symbol_phase_error_per_step": m["symbol_phase_error_per_step"],
            "l2_relative": m["l2_relative"],
        })
    return out


def diffusivity_sweep(*, nx: int = 64, courant: float = 0.25,
                      diffusion_numbers=(0.0, 1.0e-3, 1.0e-2, 1.0e-1),
                      steps: int = 200) -> list[dict]:
    """Phase and decay versus diffusion number (cell Peclet swept)."""
    out = []
    for a in diffusion_numbers:
        m = march_sine(nx=nx, courant=courant, diffusion_number=a, steps=steps)
        out.append({
            "diffusion_number": m["diffusion_number_measured"], "a_target": a,
            "courant_measured": m["courant_measured"], "cell_peclet": m["cell_peclet"],
            "lag_deg": m["lag_deg"],
            "lag_relative": m["lag_relative"],
            "lag_per_step": m["lag_per_step"],
            "symbol_phase_error_per_step": m["symbol_phase_error_per_step"],
            "amplitude_ratio": m["amplitude_ratio"],
            "symbol_abs_pow": m["symbol_abs_per_step"] ** steps,
            "amplitude_measured": m["amplitude"],
            "amplitude_exact": m["amplitude_exact"],
            "patch_relaxation_w": m["patch_relaxation_w"],
            "l2_relative": m["l2_relative"],
        })
    return out


def observed_order(errors, h) -> float:
    """Least-squares slope of ``log(error)`` versus ``log(h)`` (positive = order)."""
    e = np.asarray(errors, dtype=float)
    x = np.asarray(h, dtype=float)
    mask = (e > 0.0) & (x > 0.0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(e[mask]), 1)[0])