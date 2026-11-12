"""One-dimensional equilibrium half-channel solve for the validation matrix.

Geometry and normalisation: half channel ``y in [0, 1]`` (``y = 0`` is the wall,
``y = 1`` the symmetry plane), unit density, unit friction velocity, kinematic
viscosity ``nu = 1/Re_tau`` and the constant pressure gradient ``dp*/dx = -1``.
With those choices the momentum equation integrates exactly to

    (nu + nu_t) dU/dy = 1 - y                                                 (1)

so the mean velocity is recovered from the closure's shear without iterating
momentum.  What *is* iterated is the pair of transport equations

    d/dy[Gamma_k dk/dy] + P - sink_k k     = 0
    d/dy[Gamma_v dv/dy] + P_v - sink_v v   = 0                                (2)

with ``v`` = omega (k-omega, SST) or epsilon (k-epsilon), ``Gamma`` the model's
diffusion coefficient and the sources split into a non-negative production and a
linear sink by :mod:`.models`.  Diffusion uses harmonic face conductances and the
equations are marched with a pseudo-time step and under-relaxation.  The reported
residual is the per-cell relative defect of the *steady* equations, so a
converged result solves (1)-(2) and not the pseudo-time system.

Two wall treatments, matching the models:

* ``low_re`` (SST-2003, Wilcox k-omega): the domain starts at the wall, ``k = 0``
  and the published omega wall value are imposed on the wall face.
* ``wall_function`` (standard k-epsilon): the domain starts at ``y+ = 30``, i.e.
  at the first grid point of a wall-function mesh.  The sublayer is not resolved:
  the first-node values ``k = u_tau^2/sqrt(C_mu)`` and ``epsilon = u_tau^3/(kappa
  y)`` are imposed on that face, the wall-adjacent velocity comes from Spalding's
  all-y+ law, and the unresolved part of the mass flux ``int_0^{y+} u+ dy`` is
  added analytically.  A wall-function channel cannot be refined towards the wall
  -- ``y+ = 30`` is 17 percent of the half height at ``Re_tau = 180`` -- which is
  exactly why the matrix grades k-epsilon only where the log layer is thicker.

Centre plane: zero scalar flux (natural) and zero total shear (implicit in (1)).

This is a diagnostic for a parallel shear flow.  It does not exercise
convection, geometry or pressure-velocity coupling, and its agreement with a wall
law is not evidence that the 3-D solvers are accurate.
"""
import numpy as np
from scipy.linalg import solve_banded

from ..wall_functions import u_plus_at_y_plus
from .models import (VARIABLE, WALL_LAW_KAPPA, WALL_TREATMENT, constants_for,
                     equilibrium_closure, implied_kappa, wall_values)
from .roughness import rough_wall_value


def _stretched(n, stretch):
    xi = np.linspace(0., 1., n + 1)
    if stretch <= 0:
        return xi
    return np.expm1(stretch * xi) / np.expm1(stretch)


def _faces(n, stretch, offset):
    """Wall-normal faces on ``[offset, 1]`` with exponential clustering at ``offset``."""
    return offset + (1 - offset) * _stretched(n, stretch)


def _first_center(n, stretch, offset):
    faces = _faces(n, stretch, offset)
    return .5 * (faces[0] + faces[1])


def stretch_for_first_center(n, target, offset=0.):
    """Exponential stretch that puts the first cell centre at ``target``."""
    uniform = offset + (1 - offset) * .5 / n
    if not offset < target < uniform * (1 + 1e-12):
        raise ValueError('unreachable first cell centre %.6g; it must lie in (%.6g, %.6g]'
                         % (target, offset, uniform))
    low, high = 1e-6, 200.
    for _ in range(200):
        mid = .5 * (low + high)
        if _first_center(n, mid, offset) > target:
            low = mid
        else:
            high = mid
        if high - low <= 1e-14 * max(1., high):
            break
    return .5 * (low + high)


def _conductance(faces, y, coefficient, offset):
    """Harmonic face conductance; zero conductance at the centre plane."""
    g = np.zeros(len(faces))
    g[0] = coefficient[0] / (y[0] - offset)
    g[1:-1] = 1 / ((faces[1:-1] - y[:-1]) / coefficient[:-1] + (y[1:] - faces[1:-1]) / coefficient[1:])
    return g


def _gradient(value, boundary_value, y, offset):
    """Second-order wall-normal gradient on the cell centres."""
    padded = np.r_[boundary_value, value, value[-1]]
    coordinates = np.r_[offset, y, 2 - y[-1]]
    return np.gradient(padded, coordinates, edge_order=2)[1:-1]


def _scalar_step(faces, y, old, coefficient, production, sink, boundary_value, dt, offset):
    """One implicit pseudo-time step of ``d/dy[G dv/dy] + P - sink v = 0``."""
    volume = np.diff(faces)
    g = _conductance(faces, y, coefficient, offset)
    negative = np.minimum(production, 0.)
    destruction = sink + np.divide(-negative, old, out=np.zeros_like(old), where=old > 0)
    band = np.zeros((3, len(y)))
    band[0, 1:] = -g[1:-1]
    band[2, :-1] = -g[1:-1]
    band[1] = g[:-1] + g[1:] + volume * (destruction + 1 / dt)
    rhs = volume * (np.maximum(production, 0.) + old / dt)
    rhs[0] += g[0] * boundary_value
    candidate = solve_banded((1, 1), band, rhs)
    if not np.isfinite(candidate).all() or np.any(candidate < 0):
        raise RuntimeError('non-physical transport candidate; no clipping applied')
    return candidate


def _scalar_residual(faces, y, value, coefficient, production, sink, boundary_value, offset):
    """Per-cell relative defect of the converged steady equation."""
    volume = np.diff(faces)
    g = _conductance(faces, y, coefficient, offset)
    flux = np.zeros(len(faces))
    flux[0] = g[0] * (value[0] - boundary_value)
    flux[1:-1] = g[1:-1] * np.diff(value)
    source = volume * (production - sink * value)
    defect = -np.diff(flux) - source
    scale = np.abs(flux[:-1]) + np.abs(flux[1:]) + volume * (np.abs(production) + np.abs(sink * value))
    return dict(relative_local_max=float(np.max(np.abs(defect) / np.maximum(scale, 1e-30))),
                absolute_local_max=float(np.max(np.abs(defect))),
                global_balance=float(np.sum(defect)))


def _sublayer_mass_flux(y_inner, re_tau, samples=401):
    """``int_0^{y_inner} u+ d(y/delta)`` from Spalding's all-y+ law (Simpson).

    The integrand is ``u+`` itself because the bulk velocity is non-dimensionalised
    by ``u_tau`` and the wall-normal coordinate by the half height; the wall
    coordinate only enters through ``y+ = y*Re_tau``.
    """
    if y_inner <= 0:
        return 0.
    y = np.linspace(0., y_inner, samples)
    speed = u_plus_at_y_plus(np.maximum(y * re_tau, 1e-12))
    weights = np.ones(samples)
    weights[1:-1:2] = 4.
    weights[2:-1:2] = 2.
    return float(np.sum(weights * speed) * (y[1] - y[0]) / 3)


def solve_channel(model, n=256, *, re_tau=180., stretch=3., first_cell_y_plus=None, constants=None,
                  dt=.02, relaxation=.5, max_iterations=40000, tolerance=1e-10, history=None,
                  k_s=None):
    """Converge the 1-D equilibrium channel for ``model``.

    For the wall-function model ``first_cell_y_plus`` sets the wall-normal location
    of the inner boundary and defaults to 30; for the low-Reynolds models it
    replaces ``stretch`` by the mesh that puts the first cell centre at that ``y+``.
    """
    if model not in VARIABLE:
        raise ValueError('unknown model %r' % model)
    if not isinstance(n, (int, np.integer)) or n < 8:
        raise ValueError('need at least 8 wall-normal cells')
    if not np.isfinite([re_tau, dt, relaxation, tolerance]).all() or re_tau <= 0 or dt <= 0:
        raise ValueError('invalid controls')
    if not 0 < relaxation <= 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('invalid controls')
    c = constants_for(model, constants)
    wall_function = WALL_TREATMENT[model] == 'wall_function'
    nu = 1. / re_tau
    offset = 0.
    if wall_function:
        # The inner boundary *is* the first wall-function node; there is nothing
        # below it to resolve, so the stretch only clusters cells above it.
        first_cell_y_plus = 30. if first_cell_y_plus is None else float(first_cell_y_plus)
        offset = first_cell_y_plus / re_tau
        if not 0 < offset < 1:
            raise ValueError('wall-function offset must stay inside the channel')
    elif first_cell_y_plus is not None:
        stretch = stretch_for_first_center(n, first_cell_y_plus / re_tau)
    faces = _faces(n, stretch, offset)
    y = .5 * (faces[:-1] + faces[1:])
    volume = np.diff(faces)
    k_wall, variable_wall = wall_values(model, y[0] if not wall_function else offset, nu, c)
    roughness = None
    if k_s is not None:
        # Equivalent sand-grain roughness, in units of the half height.  The
        # wall value of the transported variable is capped below (see
        # .roughness); the momentum equation (1) then integrates the reduced
        # shear on its own, so no velocity boundary condition changes.
        k_s = float(k_s)
        if not np.isfinite(k_s) or k_s <= 0 or k_s >= 1:
            raise ValueError('k_s must be positive, finite and below the half height')
        variable_wall, roughness = rough_wall_value(
            model, k_s_plus=k_s * re_tau, smooth_value=variable_wall,
            u_tau_over_nu=re_tau)
    if not np.isfinite([stretch]).all() or stretch < 0:
        raise ValueError('stretch must be finite and non-negative')
    beta_star = c.get('beta_star', .09)
    if wall_function:
        k = np.full(n, k_wall)
        variable = np.full(n, variable_wall)
    else:
        k = np.full(n, .1)
        variable = 6 * nu / (c.get('beta1', c.get('beta', .075)) * y * y) + 1 / (np.sqrt(beta_star) * WALL_LAW_KAPPA * y)
    stable = 0
    iterations = 0
    state = dict()
    for iteration in range(max_iterations):
        iterations = iteration + 1
        dk = _gradient(k, k_wall, y, offset)
        dvariable = _gradient(variable, variable_wall, y, offset)
        state = equilibrium_closure(model, y, k, variable, dk, dvariable, nu, c)
        k_new = _scalar_step(faces, y, k, state['k_diffusion'], state['k_production'], state['k_sink'],
                             k_wall if wall_function else 0., dt, offset)
        variable_new = _scalar_step(faces, y, variable, state['variable_diffusion'], state['variable_production'],
                                    state['variable_sink'], variable_wall, dt, offset)
        change = max(float(np.max(np.abs(variable_new - variable) / np.maximum(variable_new, 1e-300))),
                     float(np.max(np.abs(k_new - k) / np.maximum(k_new, 1e-300))))
        k = (1 - relaxation) * k + relaxation * k_new
        variable = (1 - relaxation) * variable + relaxation * variable_new
        dk = _gradient(k, k_wall, y, offset)
        dvariable = _gradient(variable, variable_wall, y, offset)
        state = equilibrium_closure(model, y, k, variable, dk, dvariable, nu, c)
        residuals = dict(
            k=_scalar_residual(faces, y, k, state['k_diffusion'], state['k_production'], state['k_sink'],
                               k_wall if wall_function else 0., offset),
            variable=_scalar_residual(faces, y, variable, state['variable_diffusion'], state['variable_production'],
                                      state['variable_sink'], variable_wall, offset))
        error = max(row['relative_local_max'] for row in residuals.values())
        if history is not None:
            history.append(dict(iteration=iterations, relative_local_residual=error, unrelaxed_change=change))
        stable = stable + 1 if max(error, change) < tolerance else 0
        if stable >= 2:
            break
    shear = state['shear']
    inner_velocity = float(u_plus_at_y_plus(offset * re_tau)) if wall_function else 0.
    face_velocity = np.r_[inner_velocity, inner_velocity + np.cumsum(volume * shear)]
    velocity = .5 * (face_velocity[:-1] + face_velocity[1:])
    bulk = float(volume @ velocity) + _sublayer_mass_flux(offset, re_tau)
    result = dict(model=model, variable=VARIABLE[model], wall_treatment=WALL_TREATMENT[model],
                  converged=bool(stable >= 2), iterations=iterations,
                  relative_local_residual=error, unrelaxed_change=change,
                  n=int(n), stretch=float(stretch), re_tau=float(re_tau), nu=float(nu),
                  faces=faces, y=y, volume=volume, y_plus=y * re_tau,
                  k=k, variable_values=variable, nut=state['nut'], shear=shear,
                  velocity=velocity, face_velocity=face_velocity, u_plus=velocity.copy(),
                  bulk_velocity=bulk, bulk_u_plus=bulk, Cf=2 / bulk ** 2,
                  wall_k=k_wall, wall_variable=variable_wall, wall_offset=offset,
                  first_cell_y_plus=float(y[0] * re_tau),
                  inner_boundary_y_plus=float(offset * re_tau),
                  residuals=residuals, constants=c, implied_kappa=implied_kappa(model, c),
                  k_s=None if k_s is None else float(k_s),
                  k_s_plus=None if k_s is None else float(k_s) * float(re_tau),
                  roughness=roughness, closure=dict(state))
    return result
