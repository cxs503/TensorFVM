"""Audit of the closure-free (``channel_spectral_dns``) PASS for KMM180.

Background
----------
``docs/case-fvm-channel-kmm180-dns`` reports, for the anchor
``CASE-FVM-CHANNEL-KMM180``, a closure-free ``u+(y+)`` at 1.61 % (legacy window)
and 2.59 % (physical log layer) RMS relative deviation — both PASS under the
anchor's 5 % criterion — and presents them as evidence that the *discretisation*
reproduces the logarithmic law without a turbulence model.

The published artifacts of that run also contain, in the same file, the two
numbers that decide whether such a claim can stand:

* ``uv_plus`` -- the resolved Reynolds shear -- is ``O(1e-6)`` at every station,
  i.e. **zero** to any meaningful precision;
* ``total_shear_plus - target_shear_plus`` -- the mean-momentum-balance residual
  of a forced channel, which is an exact identity independent of any closure --
  reaches ``0.708``, i.e. ~71 % of the wall stress is unaccounted for.

A state in which neither the viscous shear nor the Reynolds shear carries the
imposed ``1 - Y/h`` is not a solution of the forced Navier--Stokes equations;
it is (numerically) the *initial condition*, slowly relaxing on the viscous
time scale ``h^2/nu = Re_tau h/u_tau = 180 h/u_tau`` while the averaging window
lasted 1.5 ``h/u_tau``.  The graded profile differs from the run's own initial
condition by at most 0.38 wall units, i.e. by less than the 2.9 % it is graded
on.  A PASS obtained in this state is therefore **vacuous**: it grades the
initial condition, not the solver.

What this module adds
---------------------
1. ``closurefree_audit`` — a *pre-registered* admission test for a closure-free
   ``u+`` claim (criteria below), evaluated on shipped artifacts, returning a
   machine-readable verdict;
2. ``laminar_negative_control`` — the same diagnostics evaluated on the
   *discrete laminar* steady state of the same solver, where the momentum
   balance is exact: it proves that the 0.708 residual is a property of the
   turbulent run and not of the diagnostic (measured residual ``~4e-14``,
   wall shear ``0.9995`` vs ``1``);
3. ``dt_plus`` / ``flow_through_times`` — the wall-unit bookkeeping factors.
   The published report overstated the time step in wall units by a factor
   ``Re_tau`` (``39.97`` vs ``0.222``); the flow-through count it published
   (``3.61``) is correct and is reproduced here, because ``u_tau = 1`` by
   construction.  What the corrected ``dt+`` buys is the averaging window *in
   wall units*: ``T+ = T Re_tau = 271``, against the ``~1e3`` that channel DNS
   statistics are usually averaged over.

Pre-registered admission criteria (frozen before the audit was run)
-------------------------------------------------------------------
A closure-free run may be graded against the anchor's ``u+`` criterion only if
**all three** hold in the graded window ``y+ in [30, 0.2 Re_tau]``:

===============  ==========================================================  =====
id               quantity                                                    limit
===============  ==========================================================  =====
``C1``           resolved Reynolds-shear share ``-<u'v'>+/(1-Y/h)``            >= 0.50
``C2``           momentum-balance residual ``|du+/dy+ - <u'v'>+ - (1-Y/h)|``  <= 0.05
``C3``           averaging window in streamwise flow-through times           >= 1.00
===============  ==========================================================  =====

Justification, not convenience: ``C1`` -- in a wall-bounded flow at
``Re_tau = 180`` the turbulent shear carries ~75 % of ``tau_w`` at ``y+ = 30``
and > 95 % beyond ``y+ = 100`` (DNS reference in
``docs/rans-turbulence-validation-matrix/dns/0180/``), so a run with a share
below 0.5 is not in the turbulent regime the anchor is about.  ``C2`` -- the
residual is *exactly* zero for the discrete laminar fixed point of the same
solver (``4e-14``, see ``laminar_negative_control``), which is twelve orders of
magnitude below the threshold; 0.05 ``tau_w`` is a 5 % error on the quantity
that sets the whole profile.  ``C3`` -- one flow-through time is the minimum
window on which ``(x, z)``-periodic and time averaging can decorrelate the
largest structures; below it the running mean is dominated by its initial
transient.

Honesty protocol
----------------
The audit does not overwrite the shipped PASS.  It records the PASS, records
that it fails its own preconditions, and leaves the anchor's graded number as
*not established by this route*.  Whether the RANS route's FAIL (6.3 % in the
log layer, ``docs/verification-channel-kmm180``) stands is unaffected: this
module only withdraws the closure-free PASS, it does not promote the RANS
number.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .channel_spectral_dns import (
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
    total_shear_identity,
)

__all__ = [
    'CRITERIA',
    'AuditVerdict',
    'blended_initial_profile',
    'closurefree_audit',
    'dt_plus',
    'flow_through_times',
    'laminar_negative_control',
    'load_closurefree_artifacts',
]

# --------------------------------------------------------------------------- #
# Pre-registered criteria
# --------------------------------------------------------------------------- #
CRITERIA = dict(
    uv_share_min=0.50,
    balance_residual_max=0.05,
    flow_throughs_min=1.00,
    log_layer_fraction=0.20,   # graded window upper bound = 0.2 Re_tau (same as kmm180_windows)
    low_y_plus=30.0,
)

ARTIFACT_DIR = (Path(__file__).resolve().parents[3] / 'docs'
                / 'case-fvm-channel-kmm180-dns')


# --------------------------------------------------------------------------- #
# Wall-unit bookkeeping (defect D1/D2 of the shipped report)
# --------------------------------------------------------------------------- #
def dt_plus(dt: float, *, re_tau: float, u_tau: float = 1.0,
            half_width: float = 1.0) -> float:
    """Time step in wall units: ``dt+ = dt u_tau^2 / nu = dt u_tau Re_tau / h``.

    The shipped report used ``dt * re_tau ** 2 / h ** 2``, which is this value
    multiplied by ``Re_tau``.
    """
    return dt * u_tau * re_tau / half_width


def legacy_dt_plus(dt: float, *, re_tau: float, half_width: float = 1.0) -> float:
    """The shipped (wrong) formula, kept so the audit can quantify the error."""
    return dt * re_tau ** 2 / half_width ** 2


def flow_through_times(time_span: float, *, length_x: float, bulk_plus: float,
                       u_tau: float = 1.0) -> float:
    """Streamwise flow-throughs: ``T U_b / L_x`` with ``U_b`` in **code** units.

    ``bulk_plus`` is returned by the solver in *wall* units (``u_tau = 1`` by
    construction, so code units of velocity are wall units only when
    ``u_tau = 1``); the correct code-unit bulk velocity is ``bulk_plus*u_tau``.
    The shipped report wrote ``T * bulk_plus / L_x``; with ``u_tau = 1`` by
    construction (the body force fixes it) the two agree, so that number is
    reproduced rather than corrected here.
    """
    return time_span * bulk_plus * u_tau / length_x


# --------------------------------------------------------------------------- #
# Initial condition (must mirror ``SpectralChannel.initialize``)
# --------------------------------------------------------------------------- #
def blended_initial_profile(y_plus: np.ndarray) -> np.ndarray:
    """The ``blended`` mean profile used by ``initialize`` (initial condition)."""
    yp = np.maximum(np.asarray(y_plus, dtype=float), 1e-6)
    log_law = np.log(yp) / 0.41 + 5.2
    weight = 0.5 * (1.0 + np.tanh((yp - 11.0) / 3.0))
    return yp * (1.0 - weight) + log_law * weight


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@dataclass
class ClosureFreeArtifacts:
    """Shipped closure-free artifacts, parsed as arrays."""

    y_plus: np.ndarray
    u_plus: np.ndarray
    u_plus_dns: np.ndarray
    du_plus_dy_plus: np.ndarray
    uv_plus: np.ndarray
    uu_plus: np.ndarray
    vv_plus: np.ndarray
    ww_plus: np.ndarray
    total_shear_plus: np.ndarray
    target_shear_plus: np.ndarray
    metrics: Dict = field(default_factory=dict)


def load_closurefree_artifacts(directory: Optional[Path] = None) -> ClosureFreeArtifacts:
    """Load ``profiles.csv`` + ``metrics.json`` from the shipped route directory."""
    directory = Path(directory) if directory is not None else ARTIFACT_DIR
    path = directory / 'profiles.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    columns: Dict[str, List[float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))

    def column(name: str) -> np.ndarray:
        if name not in columns:
            raise KeyError(f'column {name!r} missing from {path}')
        return np.asarray(columns[name], dtype=float)

    metrics_path = directory / 'metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return ClosureFreeArtifacts(
        y_plus=column('y_plus'),
        u_plus=column('u_plus'),
        u_plus_dns=column('u_plus_dns'),
        du_plus_dy_plus=column('du_plus_dy_plus'),
        uv_plus=column('uv_plus'),
        uu_plus=column('u_rms_plus') ** 2,
        vv_plus=column('v_rms_plus') ** 2,
        ww_plus=column('w_rms_plus') ** 2,
        total_shear_plus=column('total_shear_plus'),
        target_shear_plus=column('target_shear_plus'),
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@dataclass
class AuditVerdict:
    """Outcome of the admission test, plus the numbers that produced it."""

    admitted: bool
    verdict: str
    indicators: Dict[str, float]
    failed: List[str]
    notes: List[str]

    def as_dict(self) -> Dict:
        return dict(admitted=self.admitted, verdict=self.verdict,
                    indicators=self.indicators, failed=self.failed,
                    notes=self.notes)


def closurefree_audit(artifacts: Optional[ClosureFreeArtifacts] = None,
                      *, criteria: Optional[Dict[str, float]] = None) -> AuditVerdict:
    """Evaluate the pre-registered admission criteria on shipped artifacts.

    Returns an :class:`AuditVerdict` whose ``verdict`` is one of

    * ``VACUOUS_PASS`` -- the graded window PASSes the 5 % criterion but the run
      fails its own preconditions, so the PASS is not evidence;
    * ``ADMISSIBLE`` -- the preconditions hold and the PASS may be used;
    * ``NOT_APPLICABLE`` -- a precondition fails *and* the graded number FAILs,
      in which case the FAIL is the honest outcome for this route anyway.
    """
    criteria = {**CRITERIA, **(criteria or {})}
    art = load_closurefree_artifacts() if artifacts is None else artifacts

    re_tau = float(np.max(art.y_plus)) / 1.0  # not used for the mask below
    # Graded window, identical to the anchor harness: y+ in [30, 0.2 Re_tau].
    re_tau_nominal = 180.0
    if art.metrics.get('re_tau'):
        re_tau_nominal = float(art.metrics['re_tau'])
    upper = criteria['log_layer_fraction'] * re_tau_nominal
    mask = (art.y_plus >= criteria['low_y_plus']) & (art.y_plus <= upper)
    if mask.sum() < 2:
        raise ValueError('graded window holds fewer than two stations')

    target = 1.0 - art.y_plus / re_tau_nominal
    shear = -art.uv_plus
    share = shear / np.maximum(target, 1e-12)
    residual = np.abs(art.du_plus_dy_plus - art.uv_plus - target)
    ic = blended_initial_profile(art.y_plus)
    ic_drift = np.abs(art.u_plus - ic)

    rms = float(np.sqrt(np.mean(((art.u_plus[mask] - art.u_plus_dns[mask])
                                 / art.u_plus_dns[mask]) ** 2)))
    time_span = float(art.metrics.get('averaging_time_span', 0.0))
    res = art.metrics.get('resolution', {})
    dt = float(res.get('time_step', 0.0))
    length_x = float(res.get('length_x_plus', 0.0)) / re_tau_nominal  # back to code units
    bulk_plus = float(art.metrics.get('global_balance', {}).get('bulk_u_plus', 0.0))
    flow_throughs = (flow_through_times(time_span, length_x=length_x,
                                        bulk_plus=bulk_plus)
                     if length_x and bulk_plus and time_span else float('nan'))

    indicators = dict(
        graded_rms_relative=rms,
        graded_criterion=float(art.metrics.get('tolerance', 0.05)),
        graded_verdict=('PASS' if rms < float(art.metrics.get('tolerance', 0.05))
                        else 'FAIL'),
        uv_share_max=float(np.max(share[mask])),
        uv_plus_max_abs=float(np.max(np.abs(art.uv_plus))),
        balance_residual_max=float(np.max(residual[mask])),
        balance_residual_fraction=float(np.max(residual[mask])
                                        / np.mean(target[mask])),
        initial_condition_drift_max=float(np.max(ic_drift[mask])),
        initial_condition_drift_relative=float(np.max(ic_drift[mask]
                                                      / art.u_plus[mask])),
        uu_plus_max=float(np.max(art.uu_plus)),
        vv_plus_max=float(np.max(art.vv_plus)),
        averaging_time_h_over_u_tau=time_span,
        flow_through_times=float(flow_throughs),
        averaging_time_plus=float(time_span * re_tau_nominal),
        dt_plus_correct=(dt_plus(dt, re_tau=re_tau_nominal) if dt else float('nan')),
        dt_plus_shipped=float(res.get('dt_plus', float('nan'))),
    )

    failed = []
    if not indicators['uv_share_max'] >= criteria['uv_share_min']:
        failed.append('C1 uv_share')
    if not indicators['balance_residual_max'] <= criteria['balance_residual_max']:
        failed.append('C2 balance_residual')
    if not (flow_throughs == flow_throughs
            and flow_throughs >= criteria['flow_throughs_min']):
        failed.append('C3 flow_through_times')

    notes = [
        'C1: resolved Reynolds-shear share is %.2e (limit %.2f); v_rms^2 max is '
        '%.2e u_tau^2, so the perturbation field left after the transient is '
        'numerical noise.' % (indicators['uv_share_max'],
                              criteria['uv_share_min'],
                              indicators['vv_plus_max']),
        'C2: momentum-balance residual is %.3f u_tau^2 (limit %.2f); it is '
        '%.1f%% of the mean imposed shear in the window.'
        % (indicators['balance_residual_max'],
           criteria['balance_residual_max'],
           100 * indicators['balance_residual_fraction']),
        'C3: the averaging window spans %.2f flow-through times (limit %.1f), '
        'i.e. %.2f h/u_tau = %.0f wall-time units, against a viscous relaxation '
        'time h^2/nu = %.0f h/u_tau; the criterion passes and is not what '
        'condemns the run.'
        % (flow_throughs, criteria['flow_throughs_min'], time_span,
           time_span * re_tau_nominal, re_tau_nominal),
        'The graded profile sits %.2f u+ (%.1f%%) away from the run\'s own '
        'initial condition inside the window, i.e. it never evolved by more '
        'than the deviation it is graded on.'
        % (indicators['initial_condition_drift_max'],
           100 * indicators['initial_condition_drift_relative']),
        'Bookkeeping: dt+ is %.3f, not the %.1f printed in the shipped report '
        '(that value used Re_tau**2 instead of Re_tau); the flow-through count '
        '(%.2f) is unchanged.'
        % (indicators['dt_plus_correct'], indicators['dt_plus_shipped'],
           flow_throughs),
    ]

    if failed and indicators['graded_verdict'] == 'PASS':
        verdict = 'VACUOUS_PASS'
    elif not failed:
        verdict = 'ADMISSIBLE'
    else:
        verdict = 'NOT_APPLICABLE'
    return AuditVerdict(admitted=not failed, verdict=verdict,
                        indicators=indicators, failed=failed, notes=notes)


# --------------------------------------------------------------------------- #
# Negative control: the discrete laminar fixed point of the same solver
# --------------------------------------------------------------------------- #
def laminar_negative_control(*, re_tau: float = 180.0, ny: int = 64,
                             nx: int = 8, nz: int = 8,
                             steps: int = 20) -> Dict[str, float]:
    """Run the same diagnostics on the discrete laminar steady state.

    The laminar channel also has ``<u'v'> = 0``; the difference is that its
    viscous shear carries the *whole* imposed stress, so the balance residual is
    zero.  This isolates the audit from the accusation that ``uv = 0`` alone is
    what makes the turbulent run fail: it is the *unbalanced* ``uv = 0``.
    """
    config = ChannelConfig(re_tau=re_tau, nx=nx, ny=ny, nz=nz)
    solver = SpectralChannel(config)
    profile = solver.laminar_profile()
    solver.u = np.repeat(profile[:, None, None], config.nx, axis=1)
    solver.u = np.repeat(solver.u, config.nz, axis=2).copy()
    solver.v[:] = 0.0
    solver.w[:] = 0.0
    solver.project(dt=min(1e-3, solver.time_step))
    start = solver.mean_velocity_profile().copy()
    for _ in range(int(steps)):
        solver.step()
    drift = float(np.max(np.abs(solver.mean_velocity_profile() - start)))
    acc = StatisticsAccumulator(solver)
    for _ in range(8):
        solver.step()
        acc.add()
    stats = acc.folded()
    identity = total_shear_identity(stats)
    residual = np.abs(identity['total'] - identity['target'])
    return dict(
        fixed_point_drift=drift,
        balance_residual_max=float(np.max(residual)),
        uv_plus_max_abs=float(np.max(np.abs(stats.uv_plus))),
        wall_shear=float(solver.wall_shear()),
        wall_shear_error=float(solver.wall_shear() - 1.0),
        dt_plus=dt_plus(solver.time_step, re_tau=re_tau),
        legacy_dt_plus=legacy_dt_plus(solver.time_step, re_tau=re_tau),
    )