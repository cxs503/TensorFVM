#!/usr/bin/env python
"""CASE-FVM-CHANNEL-KMM180 — transient probe of the closure-free route.

Why this probe exists
---------------------
``docs/case-fvm-channel-kmm180-dns`` grades the closure-free
(``channel_spectral_dns``) run at PASS (1.61 % legacy / 2.59 % log layer).
That run averaged ``T = 1.5 h/u_tau`` and its published ``profiles.csv``
carries ``uv+ = O(1e-6)`` at every station: the resolved Reynolds shear is
zero, so the mean momentum balance ``du+/dy+ - <u'v'>+ = 1 - Y/h`` is violated
by ~0.71 ``u_tau^2``.  A PASS obtained in a state that is not a solution of the
forced-channel momentum balance is not evidence about the discretisation.

This script turns that *post hoc* observation into a *prospective* measurement:
it advances the same solver from the same kind of initial condition and records,
as a time series, the quantities that decide whether a turbulent equilibrium is
being approached at all:

* the resolved Reynolds shear ``<u'v'>+`` at fixed ``y+`` stations,
* the share of the momentum flux carried by it (``-<u'v'>+ / (1-Y/h)``),
* the mean-momentum-balance residual,
* the running-mean ``u+`` at the graded stations ``y+ = 30, 50, 100``,
* the wall-shear estimate (should be 1 by construction).

Nothing is graded here: the probe answers "does this route produce Reynolds
stress on this grid and this time scale", which is a precondition for the
anchor's graded number to mean anything.  Rows are flushed after every sample so
a truncated job still leaves usable evidence.

Run::

    PYTHONPATH=src python scripts/probe_case_fvm_channel_kmm180_dns_transient.py \
        --out docs/case-fvm-channel-kmm180-dns-transient --steps 6000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from tensorfvm.verification.channel_spectral_dns import (  # noqa: E402
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
    total_shear_identity,
)

STATIONS = (30.0, 50.0, 100.0)


def wall_units_per_time(config: ChannelConfig) -> float:
    """Wall-time units per code-time unit: ``t+ = t u_tau Re_tau / h``.

    ``u_tau = sqrt(G h)`` by construction (unit density), so with the solver's
    non-dimensionalisation ``u_tau = 1`` and ``h = 1`` this is ``Re_tau``.  The
    factor is written out because the closure-free report's ``dt_plus`` used
    ``Re_tau**2`` and overstated every wall-unit time by a factor ``Re_tau``.
    """
    return config.u_tau * config.re_tau / config.half_width


def flow_through_time(config: ChannelConfig, bulk_plus: float) -> float:
    """One streamwise flow-through in code time units: ``L_x / U_b``.

    ``bulk_plus`` is measured in wall units, so the code-unit bulk velocity is
    ``bulk_plus * u_tau``.  The closure-free report divided by ``L_x`` **without**
    the ``u_tau`` factor, which inflated its flow-through count by ``1/u_tau``.
    """
    return config.length_x / (bulk_plus * config.u_tau)


def profile_at(stats_y_plus: np.ndarray, values: np.ndarray, station: float) -> float:
    """Linear interpolation of a folded profile at one ``y+`` station."""
    if station < stats_y_plus[0] or station > stats_y_plus[-1]:
        return float('nan')
    return float(np.interp(station, stats_y_plus, values))


def sample(solver: SpectralChannel, acc: StatisticsAccumulator,
           config: ChannelConfig) -> dict:
    stats = acc.folded()
    identity = total_shear_identity(stats)
    target = identity['target']
    total = identity['total']
    residual = np.abs(total - target)
    log_mask = stats.y_plus >= 30.0
    bulk = float(np.sum(stats.u_plus * np.gradient(stats.y_plus))
                 / (stats.y_plus[-1] - stats.y_plus[0]))
    row = dict(
        time=float(solver.time),
        time_plus=float(solver.time * wall_units_per_time(config)),
        flow_throughs=float(solver.time / flow_through_time(config, bulk)),
        steps=int(solver.steps),
        samples=int(stats.samples),
        bulk_plus=bulk,
        wall_shear=float(solver.wall_shear()),
        uv_share_max=float(np.max(-stats.uv_plus[log_mask] / target[log_mask]))
        if log_mask.any() else float('nan'),
        uv_plus_max_abs=float(np.max(np.abs(stats.uv_plus))),
        u_rms_max=float(np.sqrt(np.max(stats.uu_plus))),
        v_rms_max=float(np.sqrt(np.max(stats.vv_plus))),
        identity_residual_max=float(residual.max()),
        identity_residual_log=float(residual[log_mask].max())
        if log_mask.any() else float('nan'),
    )
    for station in STATIONS:
        key = str(int(station))
        row[f'u_plus_at_{key}'] = profile_at(stats.y_plus, stats.u_plus, station)
        row[f'uv_plus_at_{key}'] = profile_at(stats.y_plus, stats.uv_plus, station)
        row[f'du_dy_plus_at_{key}'] = profile_at(stats.y_plus,
                                                 stats.du_plus_dy_plus, station)
        row[f'target_at_{key}'] = profile_at(stats.y_plus, target, station)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-dns-transient')
    parser.add_argument('--re-tau', type=float, default=180.0)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--length-x', type=float, default=2.0 * np.pi)
    parser.add_argument('--length-z', type=float, default=np.pi)
    parser.add_argument('--stretching', type=float, default=1.5)
    parser.add_argument('--fft-workers', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--sample-every', type=int, default=100)
    args = parser.parse_args(argv)

    config = ChannelConfig(re_tau=args.re_tau, nx=args.nx, ny=args.ny, nz=args.nz,
                           length_x=args.length_x, length_z=args.length_z,
                           stretching=args.stretching, fft_workers=args.fft_workers)
    solver = SpectralChannel(config)
    solver.initialize(noise=args.noise, seed=args.seed)
    print('initialised', config, 'dt', solver.time_step, flush=True)

    acc = StatisticsAccumulator(solver)
    rows = []
    started = time.time()
    for step in range(1, args.steps + 1):
        solver.step()
        acc.add()
        if step % args.sample_every == 0 or step == args.steps:
            row = sample(solver, acc, config)
            row['wall_seconds'] = time.time() - started
            rows.append(row)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'transient.json').write_text(json.dumps(dict(
                config=dict(re_tau=config.re_tau, nx=config.nx, ny=config.ny,
                            nz=config.nz, stretching=config.stretching,
                            length_x=config.length_x, length_z=config.length_z),
                noise=args.noise, seed=args.seed, steps=args.steps,
                sample_every=args.sample_every,
                wall_units_per_time=wall_units_per_time(config),
                rows=rows), indent=1))
    print('done in %.1f s' % (time.time() - started), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())