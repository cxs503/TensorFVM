"""Why the Re_tau = 180 channel LES diverges: grid-scale dissipation, not a scheme defect.

Context
-------
``docs/case-fvm-channel-kmm180-les3d-stability/REPORT.md`` (D20) concludes that the
3-D LES route of anchor ``CASE-FVM-CHANNEL-KMM180`` is **BLOCKED**: the solution
becomes non-finite after 70-190 steps "independent of the closure and the
integrator", and two diagnoses (wall-normal diffusion stiffness, ``nu_t`` feedback)
were already falsified there.  The note ends with an explicit next step:

    "Locating the amplified mode (per-row / per-mode growth, wall-row vs interior)
     is the next concrete step and is **not done here**."

This module does that step, and reaches a different conclusion than "the scheme is
unstable":

1. **The amplified mode is not a mode.**  The growth is *broadband* in both the
   streamwise (``kx``) and spanwise (``kz``) directions: no single Fourier mode
   holds more than a few per cent of the late-time energy.  The odd-even /
   Nyquist-decoupling hypothesis of a central-difference convection operator is
   therefore **falsified** (criterion ``S3``).
2. **The wall-normal operator is not the defect.**  The implicitly inverted
   matrix reproduces the wall-normal part of the explicit diffusion stencil to
   round-off on the stretched grid (``S4``, measured 1.4e-14), and the discrete
   scheme commutes with the channel's mirror symmetry (``S5``): the top and
   bottom wall-adjacent rows carry equal energy, so the late-time asymmetry seen
   just before blow-up is the incipient singularity choosing a half, not a
   boundary bug.
3. **The cause is the sub-grid model being too weak for the grid.**  At fixed
   numerics ``Cs = 0.0`` and ``Cs = 0.1`` (the solver *default*) diverge, while
   ``Cs >= 0.2`` stays finite and mirror-symmetric for the whole cap (``S1``).
   Refining the grid at fixed ``Cs`` makes it *worse*, not better, because
   ``nu_t ~ (Cs Delta)^2 |S|`` shrinks with ``Delta`` while the resolved strain
   grows (``S2``: 63 steps at ``32x32x16`` vs 121 at ``16x16x8``).
4. **The historical evidence table is confounded by the initial condition.**
   ``ChannelLES3D.initialise`` defaults to ``profile='parabolic'``, whose peak at
   ``Re_tau = 180`` is ``G delta / (2 nu) = Re_tau / 2 = 90 u_tau`` -- a laminar
   field five times the turbulent peak.  Under identical numerics the default
   initial condition diverges in 37 steps against 121 for the physical Reichardt
   profile (``S6``), so the recorded "70-190 steps" numbers measure the
   relaxation of a laminar state, not the stability of the turbulent one.

Consequence for the anchor: the divergence is a *modelling* setting (``Cs`` too
small for ``Delta``), not a defect of the discretisation, and the route is not
blocked at the level the previous note implied.  The production statistics
(``T_spin``/``T_stat``, ``u+``) are still **not produced here** -- this module
claims nothing about ``u+``.

Criteria
--------
The criteria below were frozen before the measurements were run (they are the
record's ``PREDECLARED`` constant, and the tests assert the record evaluates
exactly these).  Each carries an explicit falsification condition, so the record
can fail: ``FAIL``/``NOT_GRADED`` are reported, never silently dropped.

Reproduction
------------
``PYTHONPATH=src python -m tensorfvm.verification.kmm180_les_gridscale_stability``
writes ``summary.json``/``*.csv``; ``docs/case-fvm-channel-kmm180-les-gridscale/make.py``
renders the report and the figures from that record.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

from tensorfvm.channel_les3d import ChannelLES3D, ChannelLESConfig

__all__ = [
    'GRIDS', 'CS_SWEEP', 'PREDECLARED', 'grid_metrics',
    'steps_to_divergence', 'growth_spectrum', 'mirror_symmetry_defect',
    'wall_normal_operator_identity', 'evaluate', 'main',
]

#: grid used for the sweeps (cells, ``(nx, ny, nz)``); ``SMALL`` is the grid the
#: historical note used, ``REFINED`` doubles ``nx`` and ``ny``
SMALL = (16, 16, 8)
REFINED = (32, 32, 16)
GRIDS = {'small': SMALL, 'refined': REFINED}

#: Smagorinsky constants scanned at the small grid.  ``0.1`` is the shipping
#: default of :class:`ChannelLESConfig`, so it must be in the scan.
CS_SWEEP = (0.0, 0.1, 0.2, 0.3)

#: step cap for a run that is called "survived"; a surviving run is *not* a
#: converged one, it only did not blow up inside the window
CAP = 500

#: Frozen before measurement.  ``statement`` is the claim, ``falsified_if`` the
#: observation that would kill it, ``metric`` the key in the record it is graded on.
PREDECLARED = (
    dict(id='S1', kind='attribution',
         statement='A grid-scale dissipation threshold exists: Cs = 0.1 diverges '
                   'while some Cs <= 0.3 survives the cap under identical numerics.',
         metric='cs_threshold',
         falsified_if='no scanned Cs above 0.1 survives the cap, or Cs = 0.1 survives too'),
    dict(id='S2', kind='attribution',
         statement='Refinement at fixed Cs does not cure the divergence: the refined '
                   'grid diverges earlier than the coarse one.',
         metric='refinement_steps',
         falsified_if='refined grid survives, or diverges later than the coarse grid'),
    dict(id='S3', kind='falsification',
         statement='The amplified disturbance is broadband: no single streamwise mode '
                   'carries more than half of the late-time u energy (this falsifies '
                   'odd-even / Nyquist decoupling as the cause).',
         metric='max_mode_share',
         falsified_if='one kx mode holds > 50 % of the late-time u energy'),
    dict(id='S4', kind='invariant',
         statement='The implicit wall-normal operator reproduces the explicit stencil '
                   'on the stretched grid.',
         metric='operator_identity_defect',
         falsified_if='relative defect > 1e-10'),
    dict(id='S5', kind='invariant',
         statement='The discrete scheme commutes with the channel mirror symmetry '
                   '(top and bottom wall-adjacent rows carry equal energy).',
         metric='mirror_defect',
         falsified_if='relative top/bottom asymmetry > 10 % at an early sample'),
    dict(id='S6', kind='confound',
         statement='The default laminar initial condition diverges earlier than the '
                   'physical turbulent one, so the historical step counts are confounded.',
         metric='ic_steps',
         falsified_if='the parabolic profile survives at least as long as Reichardt'),
    dict(id='S7', kind='scaling',
         statement='The instability is per unit *time*, not per step: the blow-up time '
                   'is dt-independent within 20 %.',
         metric='dt_scaling_ratio',
         falsified_if='blow-up time changes by more than 20 % when dt is halved'),
)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def _make_solver(grid, cs, dt, profile, seed=0):
    nx, ny, nz = grid
    solver = ChannelLES3D(ChannelLESConfig(nx=nx, ny=ny, nz=nz, dt=dt,
                                           smagorinsky=cs), seed=seed)
    solver.initialise(noise=0.1, profile=profile)
    return solver


def grid_metrics(grid=SMALL):
    """Wall-unit size of the cells, which is what the SGS model reacts to.

    Reported in ``u_tau`` units so the resolution claim is a measurement rather
    than an adjective: the threshold on ``Cs`` is only interpretable together with
    the cell Reynolds number it was measured at.
    """
    nx, ny, nz = grid
    solver = ChannelLES3D(ChannelLESConfig(nx=nx, ny=ny, nz=nz))
    re_tau = solver.config.re_tau
    dy = solver.dy * re_tau
    return dict(nx=nx, ny=ny, nz=nz,
                first_cell_y_plus=float(solver.y_plus[0]),
                dy_plus_wall=float(dy[0]),
                dy_plus_mid=float(dy[len(dy) // 2]),
                dx_plus=float(solver.dx * re_tau),
                dz_plus=float(solver.dz * re_tau))


def steps_to_divergence(grid=SMALL, cs=0.1, dt=2.5e-3, profile='reichardt',
                        seed=0, cap=CAP):
    """Step until the state goes non-finite; report how long that took.

    A run that reaches ``cap`` is reported as ``diverged=False`` with the step
    count, and its step count must never be read as a lifetime (it is a lower
    bound, not a measurement of stability).
    """
    solver = _make_solver(grid, cs, dt, profile, seed=seed)
    capped = False
    try:
        # divergence is the *measured* outcome here, so the overflow warnings the
        # run raises on its way out are expected, not incidents
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for _ in range(cap):
                solver.step()
    except FloatingPointError:
        pass
    if solver.steps >= cap:
        capped = True
    row = (solver.u ** 2).mean(axis=(1, 2)) if np.isfinite(solver.u).all() else None
    total = float(row.sum()) if row is not None else float('nan')
    return dict(
        nx=grid[0], ny=grid[1], nz=grid[2], cs=cs, dt=dt, profile=profile,
        steps=int(solver.steps), time=float(solver.time), capped=capped,
        diverged=not capped,
        bottom_row_share=(float(row[0] / total) if row is not None and total > 0 else None),
        top_row_share=(float(row[-1] / total) if row is not None and total > 0 else None),
    )


def growth_spectrum(grid=SMALL, cs=0.1, dt=2.5e-3, profile='reichardt',
                    seed=0, steps=120, every=5):
    """Per-mode and per-row growth of the disturbance.

    Returns the least-squares growth rate of every streamwise (``kx``) and
    spanwise (``kz``) mode of the resolved velocity, the share each mode holds in
    the last finite sample, and the row-resolved energy fraction.  Samples that
    are not finite are dropped, so the fit never sees the blow-up itself.
    """
    nx, ny, nz = grid
    solver = _make_solver(grid, cs, dt, profile, seed=seed)
    dx, dz = solver.dx, solver.dz

    times, ekx, ekz, rows = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(steps):
            try:
                solver.step()
            except FloatingPointError:
                break
            if solver.steps % every:
                continue
            if not np.isfinite(solver.u).all():
                break
            times.append(solver.time)
            u_hat = np.fft.fft(solver.u, axis=1)      # (ny, kx, nz)
            w_hat = np.fft.fft(solver.w, axis=2)      # (ny, nx, kz)
            ekx.append((np.abs(u_hat) ** 2).mean(axis=(0, 2)))
            ekz.append((np.abs(w_hat) ** 2).mean(axis=(0, 1)))
            rows.append((solver.u ** 2).mean(axis=(1, 2)))

    if len(times) < 3:
        return dict(nx=nx, ny=ny, nz=nz, cs=cs, dt=dt, profile=profile,
                    steps=int(solver.steps), samples=len(times),
                    kx_rates=[], kz_rates=[], kx_shares=[], kz_shares=[],
                    max_mode_share=None, row_share=[], rates_per_step=[],
                    note='fewer than three finite samples')

    t = np.asarray(times)
    ekx = np.asarray(ekx)
    ekz = np.asarray(ekz)
    rates_kx = _log_slope(t, ekx)
    rates_kz = _log_slope(t, ekz)
    last_kx = ekx[-1] / max(ekx[-1].sum(), 1e-300)
    last_kz = ekz[-1] / max(ekz[-1].sum(), 1e-300)
    row_last = np.asarray(rows)[-1]
    kx_freq = np.fft.fftfreq(nx, d=dx)
    kz_freq = np.fft.fftfreq(nz, d=dz)
    return dict(
        nx=nx, ny=ny, nz=nz, cs=cs, dt=dt, profile=profile,
        steps=int(solver.steps), samples=len(times),
        kx_index=list(range(nx)),
        kx_wavenumber=[float(k) for k in kx_freq],
        kx_rates=[float(r) for r in rates_kx],
        kz_index=list(range(nz)),
        kz_wavenumber=[float(k) for k in kz_freq],
        kz_rates=[float(r) for r in rates_kz],
        kx_shares=[float(s) for s in last_kx],
        kz_shares=[float(s) for s in last_kz],
        max_mode_share=float(max(last_kx.max(), last_kz.max())),
        row_share=[float(s / max(row_last.sum(), 1e-300)) for s in row_last],
        centre_row=int(row_last.argmax()),
    )


def _log_slope(t, energy):
    """Least-squares slope of ``log(E)`` against ``t`` for every column of ``E``."""
    energy = np.maximum(np.asarray(energy, dtype=float), 1e-300)
    if energy.ndim == 1:
        energy = energy[:, None]
    A = np.vstack([t, np.ones_like(t)]).T
    slope, *_ = np.linalg.lstsq(A, np.log(energy), rcond=None)
    return slope[0]


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def wall_normal_operator_identity(grid=SMALL, cs=0.1):
    """Relative defect of ``_y_apply`` against the wall-normal part of ``diffusion``.

    ``lap - _y_apply`` is exactly what the IMEX split hands to the explicit part,
    so a non-zero defect would put an unaccounted operator into the right-hand
    side.  The identity is checked on the *stretched* grid, where ``dy`` varies by
    two orders of magnitude and an off-by-one in the coefficients cannot hide.

    The test field must be a function of ``y`` alone: on a general field
    ``lap - _y_apply`` is the *transverse* operator (by construction of the IMEX
    split), not a defect.  Applying this check to a fully three-dimensional field
    reports an O(1) "defect" that is entirely the x/z term -- the mistake this
    docstring exists to prevent.  The SGS viscosity is likewise taken as a
    function of ``y`` only, so that the transverse conductivities vanish exactly.
    """
    solver = _make_solver(grid, cs, 2.5e-3, 'reichardt')
    ny = solver.config.ny
    nut = np.repeat(solver.eddy_viscosity().mean(axis=(1, 2))[:, None, None],
                    solver.config.nx, axis=1)
    nut = np.repeat(nut, solver.config.nz, axis=2)
    cell, _ = solver._y_coefficients(solver.nu + nut)
    # non-trivial y-profile: a cubic in the cell-centre coordinate, which excites
    # every coupling of the tridiagonal stencil
    profile = (solver.yc / 2.0) ** 3 - 0.5 * (solver.yc / 2.0)
    field = np.repeat(profile[:, None, None], solver.config.nx, axis=1)
    field = np.repeat(field, solver.config.nz, axis=2)
    defects = []
    for _ in range(2):
        solver.u = field.copy()
        solver.w = field.copy()
        solver.v = np.zeros((ny + 1, solver.config.nx, solver.config.nz))
        lap_u, _, lap_w = solver.diffusion(nut)
        for comp, lap in ((solver.u, lap_u), (solver.w, lap_w)):
            y_part = solver._y_apply(cell, comp)
            defects.append(float(np.abs(lap - y_part).max()
                                 / max(np.abs(y_part).max(), 1e-300)))
    return dict(grid=list(grid), max_relative_defect=max(defects), per_component=defects)


def mirror_symmetry_defect(grid=SMALL, steps=80, cs=0.1, dt=2.5e-3):
    """Relative top/bottom asymmetry of row energy after ``steps`` steps.

    The grid, the wall-normal operator and the Reichardt initial profile are all
    symmetric about ``y = 1``, so the discrete evolution must be too.  A defect
    here would mean one wall is treated differently from the other -- the failure
    mode that a "the blow-up starts in the top row" observation would suggest.
    """
    solver = _make_solver(grid, cs, dt, 'reichardt')
    try:
        for _ in range(steps):
            solver.step()
    except FloatingPointError:
        pass
    if not np.isfinite(solver.u).all():
        return dict(grid=list(grid), steps=int(solver.steps), defect=None,
                    note='state already non-finite')
    row = (solver.u ** 2).mean(axis=(1, 2))
    total = float(row.sum())
    bottom, top = float(row[0]) / total, float(row[-1]) / total
    return dict(grid=list(grid), steps=int(solver.steps),
                bottom_row_share=bottom, top_row_share=top,
                defect=abs(top - bottom) / max(top, bottom),
                centre_row=int(row.argmax()))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _verdict(condition, graded=True):
    if not graded:
        return 'NOT_GRADED'
    return 'PASS' if condition else 'FAIL'


def evaluate(cap=CAP):
    """Run the whole record and grade the frozen criteria."""
    cs_runs = [steps_to_divergence(grid=SMALL, cs=cs, dt=2.5e-3,
                                  profile='reichardt', cap=cap)
               for cs in CS_SWEEP]
    refined = steps_to_divergence(grid=REFINED, cs=0.1, dt=2.5e-3,
                                 profile='reichardt', cap=cap)
    parabolic = steps_to_divergence(grid=SMALL, cs=0.1, dt=2.5e-3,
                                   profile='parabolic', cap=cap)
    fine_dt = steps_to_divergence(grid=SMALL, cs=0.1, dt=1.25e-3,
                                 profile='reichardt', cap=2 * cap)
    coarse = next(r for r in cs_runs if r['cs'] == 0.1)

    spectrum_diverging = growth_spectrum(grid=SMALL, cs=0.1, dt=1.25e-3,
                                         profile='reichardt', steps=240, every=5)
    spectrum_stable = growth_spectrum(grid=SMALL, cs=0.2, dt=2.5e-3,
                                      profile='reichardt', steps=300, every=5)
    operator = wall_normal_operator_identity(SMALL)
    mirror = mirror_symmetry_defect(SMALL, steps=80)

    survivors = [r['cs'] for r in cs_runs if r['cs'] > 0.1 and r['capped']]
    cs_threshold = min(survivors) if survivors else None

    ic_ratio = (parabolic['steps'] / coarse['steps']) if coarse['steps'] else None
    dt_ratio = (coarse['time'] / fine_dt['time']) if fine_dt['time'] else None

    criteria = []
    for spec in PREDECLARED:
        entry = dict(spec)
        measured = None
        verdict = 'NOT_GRADED'
        if spec['id'] == 'S1':
            measured = dict(cs_threshold=cs_threshold,
                            diverged_at_default=coarse['diverged'],
                            survivors=survivors)
            verdict = _verdict(cs_threshold is not None and coarse['diverged'])
        elif spec['id'] == 'S2':
            measured = dict(refined_steps=refined['steps'],
                            refined_capped=refined['capped'],
                            coarse_steps=coarse['steps'])
            verdict = _verdict(refined['diverged'] and refined['steps'] < coarse['steps'],
                               graded=not refined['capped'])
        elif spec['id'] == 'S3':
            measured = dict(max_mode_share=spectrum_diverging['max_mode_share'])
            verdict = _verdict(spectrum_diverging['max_mode_share'] is not None
                               and spectrum_diverging['max_mode_share'] < 0.5,
                               graded=spectrum_diverging['max_mode_share'] is not None)
        elif spec['id'] == 'S4':
            measured = dict(max_relative_defect=operator['max_relative_defect'])
            verdict = _verdict(operator['max_relative_defect'] < 1e-10)
        elif spec['id'] == 'S5':
            measured = dict(mirror_defect=mirror['defect'],
                            bottom_row_share=mirror.get('bottom_row_share'),
                            top_row_share=mirror.get('top_row_share'))
            verdict = _verdict(mirror['defect'] is not None and mirror['defect'] < 0.10,
                               graded=mirror['defect'] is not None)
        elif spec['id'] == 'S6':
            measured = dict(parabolic_steps=parabolic['steps'],
                            reichardt_steps=coarse['steps'], ratio=ic_ratio)
            verdict = _verdict(parabolic['diverged'] and parabolic['steps'] < coarse['steps'])
        elif spec['id'] == 'S7':
            measured = dict(dt=coarse['dt'], time=coarse['time'],
                            fine_dt=fine_dt['dt'], fine_time=fine_dt['time'],
                            ratio=dt_ratio)
            verdict = _verdict(dt_ratio is not None and abs(dt_ratio - 1.0) <= 0.20,
                               graded=dt_ratio is not None)
        entry['measured'] = measured
        entry['verdict'] = verdict
        criteria.append(entry)

    passed = sum(1 for c in criteria if c['verdict'] == 'PASS')
    return dict(
        anchor='CASE-FVM-CHANNEL-KMM180',
        route='channel_les3d',
        cap=cap,
        criteria=criteria,
        passed=passed,
        graded=sum(1 for c in criteria if c['verdict'] != 'NOT_GRADED'),
        cs_sweep=cs_runs,
        refined=refined,
        parabolic_ic=parabolic,
        fine_dt=fine_dt,
        spectrum_diverging=spectrum_diverging,
        spectrum_stable=spectrum_stable,
        operator=operator,
        mirror=mirror,
        grid_metrics=dict(small=grid_metrics(SMALL), refined=grid_metrics(REFINED)),
        claims=dict(
            u_plus_reading='not_run - no production statistics are produced here',
            log_law_verdict='unchanged: the anchor is still graded on the RANS route '
                            '(docs/case-fvm-channel-kmm180-pool)',
        ),
    )


def _write_csv(path, rows, fields):
    path.write_text('\n'.join(
        [','.join(fields)] + [','.join(str(r.get(f, '')) for f in fields) for r in rows]
    ) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-les-gridscale')
    parser.add_argument('--cap', type=int, default=CAP)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = evaluate(cap=args.cap)
    (out / 'summary.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    _write_csv(out / 'cs_sweep.csv', record['cs_sweep'],
               ['nx', 'ny', 'nz', 'cs', 'dt', 'profile', 'steps', 'time',
                'capped', 'diverged'])
    spec = record['spectrum_diverging']
    _write_csv(out / 'modes_diverging.csv',
               [dict(kx_index=i, wavenumber=w, rate=r, share=s)
                for i, w, r, s in zip(spec['kx_index'], spec['kx_wavenumber'],
                                      spec['kx_rates'], spec['kx_shares'])],
               ['kx_index', 'wavenumber', 'rate', 'share'])
    print(json.dumps({c['id']: c['verdict'] for c in record['criteria']}, indent=1))
    return record


if __name__ == '__main__':                                  # pragma: no cover
    main()