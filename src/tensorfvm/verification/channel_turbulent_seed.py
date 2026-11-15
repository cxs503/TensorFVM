"""Physical turbulence seed for the closure-free channel route (KMM180).

Why this module exists
----------------------
``docs/audit-m20573-closurefree-pass`` graded the closure-free spectral channel
run of ``CASE-FVM-CHANNEL-KMM180`` as ``VACUOUS_PASS``: its initial condition
was the blended mean profile plus *white* noise, the noise was damped by
wall-normal viscosity before anything rolled up, and the reported ``u+`` (1.6 %
RMS) was the initial condition, not a solution.  The audit left the obvious
repair open: start from a state that actually transitions, and average long
enough (``T+ >~ 1e3``) that the statistics mean something.

This module supplies the missing piece — a *physically motivated* perturbation
that drives the lift-up mechanism instead of being damped:

* a pair (or several pairs) of **streamwise vortices** ``(v, w)`` that are
  divergence-free in the wall-normal/spanwise plane by construction, with a
  wall-normal shape that vanishes at both walls;
* a weak **streamwise modulation of the streak field** (the sinuous mode that
  breaks the streaks down at low Reynolds number);
* white noise on top, of small amplitude, only to break the discrete
  symmetries.

Nothing here is a model term: the rolls are an *initial condition*, the same
role a trip wire plays in an experiment.  Whether they survive is exactly what
the run has to show, so the decision rule is pre-registered in
:func:`turbulence_gates` (identical constants to
``kmm180_closurefree_audit.CRITERIA``) and evaluated before the anchor metric.

Units follow the solver: ``u_tau = 1``, half-width ``h = 1``, so amplitudes are
in wall units and lengths in code units (``y+ = Y * Re_tau``).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .kmm180_closurefree_audit import CRITERIA

__all__ = [
    'CRITERIA',
    'roll_shape',
    'add_streamwise_rolls',
    'seed_turbulent_state',
    'bulk_velocity_plus',
    'flow_throughs',
    'turbulence_gates',
]


# --------------------------------------------------------------------------- #
# Vortex-pair seed
# --------------------------------------------------------------------------- #
def roll_shape(wall_distance: np.ndarray, half_width: float) -> np.ndarray:
    """Wall-normal shape ``f(Y) = sin(pi Y / 2h)`` and its derivative.

    ``f`` vanishes at both walls, so the induced wall-normal velocity does too;
    it peaks at the centreline, which is where the lift-up of the mean shear is
    strongest for a domain as short as ``Lx+ ~ 1130``.
    """
    k = math.pi / (2.0 * half_width)
    return np.sin(k * wall_distance), k * np.cos(k * wall_distance)


def add_streamwise_rolls(solver, *, amplitude: float = 1.5,
                         spanwise_mode: int = 2, streamwise_mode: int = 1,
                         sinusity: float = 0.3, phase: float = 0.0) -> Dict[str, float]:
    """Superimpose counter-rotating streamwise vortices on ``solver``.

    ``v = A f(Y) cos(kz z + phase)`` and ``w = -(A f'(Y)/kz) sin(kz z + phase)``
    satisfy ``dv/dY + dw/dz = 0`` analytically, so the seed injects no
    dilatation of its own; :meth:`SpectralChannel.project` removes the residual
    discretisation error.  The mean shear then tilts these rolls into
    low-speed/high-speed streaks (lift-up), and the ``sinusity`` term gives the
    streaks the streamwise wobble they need to break down.

    Parameters are in solver units: ``amplitude`` is the peak wall-normal
    velocity in ``u_tau`` (``= 1`` in this non-dimensionalisation).
    """
    if amplitude < 0.0:
        raise ValueError('amplitude must be non-negative')
    if spanwise_mode < 1:
        raise ValueError('spanwise_mode must be a positive integer')
    if not 0.0 <= sinusity <= 1.0:
        raise ValueError('sinusity must lie in [0, 1]')

    cfg = solver.config
    h = float(solver.h)
    hw = float(solver.config.half_width)
    y, yf = np.asarray(solver.y), np.asarray(solver.yf)
    f_c, df_c = roll_shape(y + hw, hw)      # cell centres: u, w
    f_f, _ = roll_shape(yf + hw, hw)        # faces: v (zero at both walls)

    kz = 2.0 * math.pi * spanwise_mode / float(cfg.length_z)
    x = np.arange(cfg.nx) * (float(cfg.length_x) / cfg.nx)
    z = np.arange(cfg.nz) * (float(cfg.length_z) / cfg.nz)
    theta = np.add.outer(np.zeros(cfg.nx), kz * z) + phase        # (nx, nz)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    wobble = 1.0 + sinusity * np.cos(
        2.0 * math.pi * streamwise_mode * x / float(cfg.length_x))[:, None]

    amp_v = amplitude * f_f[:, None, None] * cos_t[None, :, :] * wobble[None, :, :]
    amp_w = -(amplitude * df_c[:, None, None] / kz) * sin_t[None, :, :] * wobble[None, :, :]
    # Sinuous streak modulation: u' ~ A sin(kx x) f(Y) cos(kz z), unit order
    # smaller than the rolls and only there to seed the instability.
    sin_x = np.sin(2.0 * math.pi * streamwise_mode * x / float(cfg.length_x))
    amp_u = (sinusity * amplitude * f_c[:, None, None]
             * sin_x[None, :, None] * cos_t[None, :, :])

    solver.v = solver.v + amp_v
    solver.v[0] = 0.0
    solver.v[-1] = 0.0
    solver.w = solver.w + amp_w
    solver.u = solver.u + amp_u
    return dict(amplitude=float(amplitude),
                spanwise_mode=int(spanwise_mode),
                streamwise_mode=int(streamwise_mode),
                sinusity=float(sinusity),
                kz=float(kz),
                v_rms=float(np.sqrt(np.mean(amp_v ** 2))),
                w_rms=float(np.sqrt(np.mean(amp_w ** 2))),
                u_rms=float(np.sqrt(np.mean(amp_u ** 2))),
                wall_normal_flux_max=float(np.max(np.abs(amp_v[0]))),
                divergence_residual=float(np.max(np.abs(solver.divergence()))))


def seed_turbulent_state(solver, *, amplitude: float = 1.5,
                         spanwise_mode: int = 2, streamwise_mode: int = 1,
                         sinusity: float = 0.3, noise: float = 0.1,
                         seed: int = 1, mean_profile: str = 'blended') -> Dict:
    """Mean profile + vortex pairs + white noise, then project to divergence-free.

    The mean profile is the *initial condition* only (it relaxes within a few
    ``h/u_tau``); the gate in :func:`turbulence_gates` is what decides whether
    the resulting state is a turbulence solution rather than a decaying
    transient.
    """
    solver.initialize(noise=noise, seed=seed, mean_profile=mean_profile)
    info = add_streamwise_rolls(solver, amplitude=amplitude,
                                spanwise_mode=spanwise_mode,
                                streamwise_mode=streamwise_mode,
                                sinusity=sinusity)
    info['projection_residual'] = float(solver.project(dt=min(1e-3, solver.time_step)))
    info['divergence_after_projection'] = float(np.max(np.abs(solver.divergence())))
    info['noise'] = float(noise)
    info['seed'] = int(seed)
    return info


# --------------------------------------------------------------------------- #
# Pre-registered gates and the anchor metric's bookkeeping
# --------------------------------------------------------------------------- #
def bulk_velocity_plus(stats) -> float:
    """Bulk velocity in wall units: ``(1/h) * int_0^h u+ dY+``."""
    y_plus, u_plus = np.asarray(stats.y_plus), np.asarray(stats.u_plus)
    if y_plus.size < 2:
        return float('nan')
    return float(np.trapezoid(u_plus, y_plus) / stats.re_tau)


def flow_throughs(stats, length_x: float, u_tau: float = 1.0) -> float:
    """Number of streamwise flow-through times covered by the average."""
    bulk = bulk_velocity_plus(stats) * u_tau
    if not np.isfinite(bulk) or bulk <= 0.0:
        return float('nan')
    return float(stats.time_span * bulk / float(length_x))


def turbulence_gates(stats, *, length_x: float, criteria: Optional[Dict] = None,
                     u_tau: float = 1.0) -> Dict:
    """Evaluate C1/C2/C3 *before* the anchor metric is allowed to count.

    C1  share of the applied shear carried by the Reynolds stress,
        ``-<u'v'>+ / (1 - Y/h)`` in the graded window: the run must be
        turbulent, not laminar;
    C2  closure-free momentum balance residual
        ``|du+/dy+ - <u'v'>+ - (1 - Y/h)|``: the state must be a solution;
    C3  at least one streamwise flow-through of averaging.

    The constants come from ``kmm180_closurefree_audit.CRITERIA`` so the audit
    and this gate cannot drift apart.
    """
    crit = {**CRITERIA, **(criteria or {})}
    y_plus = np.asarray(stats.y_plus, dtype=float)
    uv = np.asarray(stats.uv_plus, dtype=float)
    du = np.asarray(stats.du_plus_dy_plus, dtype=float)
    target = 1.0 - y_plus / float(stats.re_tau)
    mask = ((y_plus >= crit['low_y_plus'])
            & (y_plus <= crit['log_layer_fraction'] * float(stats.re_tau)))
    if not mask.any():
        raise ValueError('graded window is empty on this grid')

    share = -uv[mask] / target[mask]
    residual = np.abs(du[mask] - uv[mask] - target[mask])
    uv_share = float(np.max(share))
    balance = float(np.max(residual))
    flow = flow_throughs(stats, length_x=length_x, u_tau=u_tau)
    time_plus = float(stats.time_span * u_tau * float(stats.re_tau))

    checks = dict(
        uv_share_min=float(crit['uv_share_min']),
        balance_residual_max=float(crit['balance_residual_max']),
        flow_throughs_min=float(crit['flow_throughs_min']),
    )
    verdict = dict(
        C1=bool(uv_share >= checks['uv_share_min']),
        C2=bool(balance <= checks['balance_residual_max']),
        C3=bool(flow >= checks['flow_throughs_min']),
    )
    return dict(uv_share=uv_share, balance_residual=balance,
                flow_throughs=flow, time_plus=time_plus,
                window=[float(y_plus[mask][0]), float(y_plus[mask][-1])],
                window_points=int(mask.sum()), checks=checks, verdict=verdict,
                admissible=bool(all(verdict.values())))