"""Spalart--Allmaras (1994) closure for the KMM180 anchor.

Why this module exists
----------------------
``src/tensorfvm/benchmark_channel_kmm1987.py`` grades the anchor
``CASE-FVM-CHANNEL-KMM180`` (RMS relative deviation of ``u+(y+)`` for ``y+ > 30``
below 5 %, both grading windows) for the three closures listed in
``rans_validation.models.MODELS``: SST, Wilcox k-omega and standard k-epsilon.
The repository's *other* one-equation closure, Spalart--Allmaras, is implemented
for the 3-D solvers (``des_iddes_model``, ``body_fitted``, ``rans_ship_resistance``)
but has never entered this matrix, so the anchor's headline verdict -- "the
physical log layer fails with every closure in the repository" -- is only
supported for two of the three closure *families*, and SA is the one model whose
constants (``kappa = 0.41``) are calibrated directly on the log law.

This module supplies that missing row. It is a self-contained 1-D equilibrium
channel solve with the SA transport equation, discretised with **exactly** the
finite-volume operators the other three models use (they are imported from
:mod:`tensorfvm.rans_validation.channel`, so a difference between SA and SST
cannot come from a different mesh, gradient or face-conductance rule).

Geometry and normalisation are the same as ``rans_validation.channel``: half
channel ``y in [0, 1]``, unit density, unit friction velocity, ``nu = 1/Re_tau``,
constant pressure gradient, so the momentum equation integrates to

    (nu + nu_t) dU/dy = 1 - y                                                (1)

and the transported variable is ``nuTilde`` (``nu_t = nuTilde * f_v1``).

Model (standard SA, trip term omitted -- a fully developed channel has no
transition onset, so ``f_t1 = f_t2 = 0`` is *not* assumed: the ``f_t2`` term is
retained because at ``Re_tau = 180`` the wall-normal station ``y+ ~ 30`` has
``chi = nuTilde/nu ~ 30``, where ``f_t2 = C_t3 exp(-C_t4 chi^2)`` is still
``O(1e-3)``; dropping it is a modelling choice that this module does not take
silently):

    C_b1 S~ nuTilde (1 - f_t2) - (C_w1 f_w - C_b1/kappa^2 f_t2)(nuTilde/d)^2
        + (1/sigma)[d/dy((nu + nuTilde) dnuTilde/dy) + C_b2 (dnuTilde/dy)^2] = 0

with ``S~ = S + nuTilde f_v2/(kappa^2 d^2)``, ``S = |dU/dy|`` from (1),
``d`` the wall distance, and the standard auxiliary functions (see
:func:`auxiliary`).

Uncertainty statement
---------------------
The graded numbers produced here are *model* numbers, not solver verification:
they answer "does this closure reproduce the DNS log layer at Re_tau=180", which
is the anchor's question. The transport solve itself is verified by
:func:`self_check` against the model's own analytic log-layer solution
(``nu_t+ = kappa y+``) and by grid refinement in the driver.
"""
from __future__ import annotations

import numpy as np

from ..rans_validation.channel import (_conductance, _faces, _gradient,  # noqa: F401  (same operators as the matrix)
                                       _scalar_residual, _scalar_step,
                                       stretch_for_first_center)

#: Standard Spalart--Allmaras (1994) constants.  ``c_w1`` is *derived*, never
#: transcribed, so the log-layer identity ``kappa = 0.41`` cannot drift from the
#: constant it is encoded in.
SA_CONSTANTS = dict(sigma=2. / 3., c_b1=0.1355, c_b2=0.622, kappa=0.41,
                    c_w2=0.3, c_w3=2.0, c_v1=7.1, c_v2=5.0, c_t3=1.2, c_t4=0.5)


def sa_coefficients(constants=None):
    """Return the SA constant set with the derived coefficient ``c_w1``."""
    c = dict(SA_CONSTANTS if constants is None else constants)
    c['c_w1'] = c['c_b1'] / c['kappa'] ** 2 + (1 + c['c_b2']) / c['sigma']
    return c


def auxiliary(nu_tilde, shear, d, nu, c):
    """SA auxiliary functions and the eddy viscosity of a given state."""
    chi = nu_tilde / nu
    f_v1 = chi ** 3 / (chi ** 3 + c['c_v1'] ** 3)
    f_v2 = 1 - chi / (1 + chi * f_v1)
    s_tilde = shear + nu_tilde * f_v2 / (c['kappa'] ** 2 * d ** 2)
    r = np.minimum(nu_tilde / np.maximum(s_tilde, 1e-300) / (c['kappa'] ** 2 * d ** 2), 10.)
    g = r + c['c_w2'] * (r ** 6 - r)
    f_w = g * ((1 + c['c_w3'] ** 6) / (g ** 6 + c['c_w3'] ** 6)) ** (1. / 6.)
    f_t2 = c['c_t3'] * np.exp(-c['c_t4'] * chi ** 2)
    return dict(chi=chi, f_v1=f_v1, f_v2=f_v2, s_tilde=s_tilde, r=r, g=g,
                f_w=f_w, f_t2=f_t2, nut=nu_tilde * f_v1)


def _state(nu_tilde, y, nu, c):
    """Closure state of a full profile ``nu_tilde`` (momentum (1) included)."""
    nutilde = np.asarray(nu_tilde, dtype=float)
    if np.any(nutilde < 0) or not np.isfinite(nutilde).all():
        raise ValueError('nuTilde must be non-negative and finite')
    d = np.maximum(y, 1e-300)
    # (1) with a provisional eddy viscosity, then re-evaluate S~ with the shear
    aux = auxiliary(nutilde, np.zeros_like(y), d, nu, c)
    shear = (1 - y) / (nu + aux['nut'])
    aux = auxiliary(nutilde, shear, d, nu, c)
    shear = (1 - y) / (nu + aux['nut'])
    # linear (in nuTilde) source split, as required by the shared finite-volume
    # step: P - sink*nuTilde, both evaluated at the current iterate.
    production = c['c_b1'] * aux['s_tilde'] * nu_tilde * (1 - aux['f_t2'])
    sink = (c['c_w1'] * aux['f_w'] - c['c_b1'] / c['kappa'] ** 2 * aux['f_t2']) * nu_tilde / d ** 2
    return dict(production=production, sink=sink, shear=shear, aux=aux)


def solve_sa_channel(n=256, *, re_tau=180., stretch=3., first_cell_y_plus=None,
                     constants=None, dt=0.02, relaxation=0.5, max_iterations=40000,
                     tolerance=1e-10):
    """Converge the 1-D equilibrium SA channel.

    The transported variable is ``nuTilde`` on the cell centres, the wall value is
    zero (smooth wall, ``nuTilde_wall = 0``) and the centre plane is insulated,
    which is the ``low_re`` wall treatment of the other two-equation models.
    """
    if not isinstance(n, (int, np.integer)) or n < 8:
        raise ValueError('need at least 8 wall-normal cells')
    if not np.isfinite([re_tau, dt, relaxation, tolerance]).all() or re_tau <= 0 or dt <= 0:
        raise ValueError('invalid controls')
    c = sa_coefficients(constants)
    nu = 1. / re_tau / 1.0
    stretch = float(stretch)
    if first_cell_y_plus is not None:
        stretch = stretch_for_first_center(n, float(first_cell_y_plus) / re_tau)
    faces = _faces(n, stretch, 0.)
    y = .5 * (faces[:-1] + faces[1:])
    volume = np.diff(faces)
    # initial guess: the SA log-layer state nu_t+ = kappa y+ (u_tau = 1, nu = 1/Re_tau)
    nu_tilde = np.maximum(c['kappa'] * y, 1e-12)
    stability, iterations, state = 0, 0, None
    error = change = np.inf
    for iteration in range(max_iterations):
        iterations = iteration + 1
        state = _state(nu_tilde, y, nu, c)
        dnu = _gradient(nu_tilde, 0., y, 0.)
        # C_b2 |grad nuTilde|^2 enters as a production-like (non-negative) term
        production = state['production'] + c['c_b2'] / c['sigma'] * dnu ** 2
        coefficient = (nu + nu_tilde) / c['sigma']
        candidate = _scalar_step(faces, y, nu_tilde, coefficient, production,
                                 state['sink'], 0., dt, 0.)
        change = float(np.max(np.abs(candidate - nu_tilde) / np.maximum(candidate, 1e-300)))
        nu_tilde = (1 - relaxation) * nu_tilde + relaxation * candidate
        state = _state(nu_tilde, y, nu, c)
        dnu = _gradient(nu_tilde, 0., y, 0.)
        production = state['production'] + c['c_b2'] / c['sigma'] * dnu ** 2
        coefficient = (nu + nu_tilde) / c['sigma']
        residual = _scalar_residual(faces, y, nu_tilde, coefficient, production,
                                    state['sink'], 0., 0.)
        error = residual['relative_local_max']
        stability = stability + 1 if max(error, change) < tolerance else 0
        if stability >= 2:
            break
    shear = state['shear']
    face_velocity = np.r_[0., np.cumsum(volume * shear)]
    velocity = .5 * (face_velocity[:-1] + face_velocity[1:])
    bulk = float(volume @ velocity)
    return dict(model='sa', wall_treatment='low_re', n=int(n), re_tau=float(re_tau),
                nu=float(nu), stretch=stretch, faces=faces, y=y, volume=volume,
                y_plus=y * re_tau, nutilde=nu_tilde, nut=state['aux']['nut'],
                shear=shear, velocity=velocity, u_plus=velocity.copy(),
                bulk_u_plus=bulk, Cf=2 / bulk ** 2, converged=bool(stability >= 2),
                iterations=iterations, relative_local_residual=float(error),
                unrelaxed_change=float(change), residual=residual,
                first_cell_y_plus=float(y[0] * re_tau), constants=c,
                aux=state['aux'])


def log_layer_slope(result, *, low=30., high=None):
    """Local log-law slope ``dU+/dln y+`` and the implied Karman constant.

    For the von Karman law ``u+ = ln(y+)/kappa + B`` the slope is ``1/kappa``.  A
    closure reproduces the law when the measured slope matches
    ``1/constants['kappa']``; this is a *model-internal* consistency check that
    needs no reference data (the same diagnostic the two-equation matrix uses
    through ``models.implied_kappa``).
    """
    y_plus = np.asarray(result['y_plus'])
    u_plus = np.asarray(result['u_plus'])
    re_tau = float(result['re_tau'])
    high = 0.2 * re_tau if high is None else float(high)
    mask = (y_plus >= low) & (y_plus <= high)
    if mask.sum() < 2:
        raise ValueError('fewer than two samples in the log window')
    slope = float(np.polyfit(np.log(y_plus[mask]), u_plus[mask], 1)[0])
    return dict(window=(float(low), float(high)), samples=int(mask.sum()),
                slope=slope, implied_kappa=1. / slope,
                model_kappa=float(sa_coefficients(result.get('constants'))['kappa']))


def self_check(result, constants=None):
    """Two reference-free gates on a converged SA solve.

    ``log_layer``: the implied Karman constant from the computed profile.  Two
    estimates are returned because at ``Re_tau = 180`` the farthest-wall end of
    the log law is not ``y+ -> inf`` but the total-stress factor ``(1 - y/h)``
    is already ``O(1)``: ``slope`` fits ``u+`` against ``ln y+`` over
    ``y+ in [30, 120]`` (a straight line only in the limit), ``mixing_length``
    inverts the model's own equilibrium eddy viscosity ``nu_t+ = kappa y+ (1-y/h)``.
    Both must land near ``kappa = 0.41``; neither is compared with a reference
    table, so this is a self-consistency gate, not an accuracy claim.
    """
    c = sa_coefficients(constants)
    re_tau = float(result['re_tau'])
    high = min(0.667 * re_tau, 130.)
    slope = log_layer_slope(result, low=30., high=high)
    y_plus = np.asarray(result['y_plus'])
    nut_plus = np.asarray(result['nut']) * re_tau
    mask = (y_plus >= 30.) & (y_plus <= high)
    mixing = nut_plus[mask] / (c['kappa'] * y_plus[mask] * (1 - y_plus[mask] / re_tau))
    return dict(log_layer=slope,
                mixing_length=dict(window=(30., float(high)), samples=int(mask.sum()),
                                   ratio_to_mixing_length=float(np.mean(mixing)),
                                   spread=float(np.max(mixing) - np.min(mixing))),
                model_kappa=float(c['kappa']),
                kappa_relative_error=abs(slope['implied_kappa'] - c['kappa']) / c['kappa'],
                mixing_offset_from_baseline=float(np.mean(mixing)) - 1.)
