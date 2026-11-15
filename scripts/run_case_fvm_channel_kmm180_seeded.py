#!/usr/bin/env python
"""Long-time seeded closure-free channel run for ``CASE-FVM-CHANNEL-KMM180``.

The audit ``docs/audit-m20573-closurefree-pass`` closed the first closure-free
attempt as ``VACUOUS_PASS`` and listed the missing work explicitly: a rerun from
a state that actually transitions, averaged over ``T+ >~ 1e3``.  This driver
does that run and applies the *pre-registered* admission gates of
:func:`tensorfvm.verification.channel_turbulent_seed.turbulence_gates` **before**
the anchor's 5 % metric is read, so a decaying transient cannot be published as
a PASS again.

Usage::

    PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_seeded.py \
        --label seeded-long --steps-transient 2500 --steps-average 6000 \
        --out docs/case-fvm-channel-kmm180-seeded

``--sweep pilot --shard-idx i`` selects entry ``i`` of :data:`PILOT_SWEEP`,
which is how the cluster shards parameterise the transition search.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
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
from tensorfvm.verification.channel_turbulent_seed import (  # noqa: E402
    bulk_velocity_plus,
    flow_throughs,
    seed_turbulent_state,
    turbulence_gates,
)
from tensorfvm.verification.kmm180_windows import (  # noqa: E402
    LOG_LAYER_FRACTION,
    LOW_Y_PLUS,
    window_deviation,
)
from tensorfvm.verification.kmm1987 import (  # noqa: E402
    TOLERANCE_LOG_REGION,
    load_kmm1987_reference,
    log_region_deviation,
)

TOLERANCE = TOLERANCE_LOG_REGION

#: Transition search: the lift-up seed amplitude, the spanwise pair count and
#: the streak wobble are the three knobs that decide whether a random-free seed
#: survives at ``Re_tau = 180``; noise only breaks discrete symmetries.
PILOT_SWEEP = (
    dict(label='a050-nz2-mod30-noise10', amplitude=0.5, spanwise_mode=2,
         sinusity=0.3, noise=0.1, seed=11),
    dict(label='a150-nz2-mod30-noise10', amplitude=1.5, spanwise_mode=2,
         sinusity=0.3, noise=0.1, seed=11),
    dict(label='a300-nz2-mod30-noise10', amplitude=3.0, spanwise_mode=2,
         sinusity=0.3, noise=0.1, seed=11),
    dict(label='a150-nz3-mod30-noise10', amplitude=1.5, spanwise_mode=3,
         sinusity=0.3, noise=0.1, seed=11),
    dict(label='a150-nz2-mod00-noise40', amplitude=1.5, spanwise_mode=2,
         sinusity=0.0, noise=0.4, seed=11),
    dict(label='a150-nz2-mod50-noise40', amplitude=1.5, spanwise_mode=2,
         sinusity=0.5, noise=0.4, seed=23),
)


#: Second sweep: the first one (``PILOT_SWEEP``) showed vortex pairs decaying
#: into streamwise-invariant streaks (u_rms^+ ~ 3 with <u'v'>^+ ~ 1e-3), i.e. the
#: seed never breaks down.  This sweep raises the three parameters that control
#: breakdown — perturbation amplitude, white-noise level that destroys the
#: streamwise invariance, and streak wobble — and lengthens the streamwise box
#: so that the streak instability has room.  Entries may also carry configuration
#: overrides (``length_x``, ``length_z``, ``nz``).
STRONG_SWEEP = (
    dict(label='s-amp3-nz2-mod50-noise100-lx4pi', amplitude=3.0, spanwise_mode=2,
         sinusity=0.5, noise=1.0, seed=5, length_x=4.0 * math.pi),
    dict(label='s-amp6-nz3-mod50-noise200-lx4pi', amplitude=6.0, spanwise_mode=3,
         sinusity=0.5, noise=2.0, seed=6, length_x=4.0 * math.pi),
    dict(label='s-amp2-nz4-mod80-noise150', amplitude=2.0, spanwise_mode=4,
         sinusity=0.8, noise=1.5, seed=7),
    dict(label='s-amp3-nz2-mod50-noise100-mx2', amplitude=3.0, spanwise_mode=2,
         sinusity=0.5, noise=1.0, seed=8, streamwise_mode=2),
    dict(label='s-amp4-nz2-mod90-noise100', amplitude=4.0, spanwise_mode=2,
         sinusity=0.9, noise=1.0, seed=10),
    dict(label='s-amp3-nz3-mod50-noise100-lx4pi', amplitude=3.0, spanwise_mode=3,
         sinusity=0.5, noise=1.0, seed=11, length_x=4.0 * math.pi),
)

STATIONS = (30.0, 50.0, 100.0)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def build_config(args) -> ChannelConfig:
    return ChannelConfig(re_tau=args.re_tau, nx=args.nx, ny=args.ny, nz=args.nz,
                         length_x=args.length_x, length_z=args.length_z,
                         stretching=args.stretching,
                         fft_workers=args.fft_workers)


def monitor(solver: SpectralChannel, acc: StatisticsAccumulator) -> dict:
    """Cheap transition diagnostics from the running time average."""
    stats = acc.folded()
    identity = total_shear_identity(stats)
    log_mask = stats.y_plus >= LOW_Y_PLUS
    target = identity['target']
    return dict(
        time=float(solver.time), steps=int(solver.steps),
        time_plus=float(solver.time * solver.re_tau),
        uv_plus_min=float(np.min(stats.uv_plus)),
        uv_share_max=float(np.max(-stats.uv_plus[log_mask] / target[log_mask]))
        if log_mask.any() else float('nan'),
        identity_residual_max=float(np.max(np.abs(identity['residual']))),
        u_rms_max=float(np.sqrt(np.max(stats.uu_plus))),
        v_rms_max=float(np.sqrt(np.max(stats.vv_plus))),
        bulk_plus=bulk_velocity_plus(stats),
        wall_shear=float(solver.wall_shear()),
        flow_throughs=flow_throughs(stats, length_x=solver.config.length_x),
        u_plus_at_100=float(np.interp(100.0, stats.y_plus, stats.u_plus)),
    )


def run_labeled(config: ChannelConfig, *, label: str, amplitude: float,
                spanwise_mode: int, sinusity: float, noise: float, seed: int,
                streamwise_mode: int = 1,
                steps_transient: int, steps_average: int, sample_every: int,
                progress_every: int, out: Path) -> dict:
    solver = SpectralChannel(config)
    seed_info = seed_turbulent_state(solver, amplitude=amplitude,
                                     spanwise_mode=spanwise_mode,
                                     streamwise_mode=streamwise_mode,
                                     sinusity=sinusity, noise=noise, seed=seed)
    dt = solver.time_step
    print(json.dumps(dict(label=label, cells=config.nx * config.ny * config.nz,
                          dt=dt, dt_plus=dt * config.re_tau,
                          seed=seed_info)), flush=True)

    history = []
    started = time.time()
    acc_t = StatisticsAccumulator(solver)
    for step in range(1, steps_transient + 1):
        solver.step()
        acc_t.add()
        if step % progress_every == 0 or step == steps_transient:
            row = monitor(solver, acc_t)
            row['wall_seconds'] = time.time() - started
            row['phase'] = 'transient'
            history.append(row)
            (out / 'transition.csv').parent.mkdir(parents=True, exist_ok=True)
            write_history(out / 'transition.csv', history)
            print('T ' + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                     for k, v in row.items()}), flush=True)
        acc_t = StatisticsAccumulator(solver) if step % 500 == 0 else acc_t

    acc = StatisticsAccumulator(solver)
    started_avg = time.time()
    for step in range(1, steps_average + 1):
        solver.step()
        acc.add()
        if step % sample_every == 0 or step == steps_average:
            row = monitor(solver, acc)
            row['wall_seconds'] = time.time() - started_avg
            row['phase'] = 'average'
            history.append(row)
    wall = time.time() - started

    stats = acc.folded()
    gates = turbulence_gates(stats, length_x=config.length_x)
    dns = load_kmm1987_reference()
    legacy = log_region_deviation(stats.y_plus, stats.u_plus, dns, low=LOW_Y_PLUS)
    log_layer = window_deviation(stats.y_plus, stats.u_plus, dns,
                                 high=LOG_LAYER_FRACTION * dns['re_tau'])
    identity = total_shear_identity(stats)
    bulk_plus = bulk_velocity_plus(stats)
    u_bulk_ref = float(dns.get('U_bulk_plus', float('nan')))
    cf = float(2.0 / max(solver.bulk_velocity(), 1e-12) ** 2)

    summary = dict(
        label=label, re_tau=float(config.re_tau),
        resolution=dict(nx=config.nx, ny=config.ny, nz=config.nz,
                        cells=int(config.nx * config.ny * config.nz),
                        length_x=float(config.length_x),
                        length_z=float(config.length_z),
                        stretching=float(config.stretching)),
        resolution_report=solver.resolution_report(),
        seed=seed_info,
        steps_transient=int(steps_transient), steps_average=int(steps_average),
        wall_seconds=float(wall),
        gates=gates,
        anchor=dict(
            legacy=dict(definition='y+ in [30, nearest reference station]',
                        rms_relative=float(legacy['rms_relative']),
                        max_relative=float(legacy.get('max_relative', float('nan'))),
                        points=int(legacy.get('cells', 0))),
            log_layer=dict(
                definition=f'y+ in [30, {LOG_LAYER_FRACTION}*Re_tau]',
                rms_relative=float(log_layer['rms_relative']),
                max_relative=float(log_layer.get('max_relative', float('nan'))),
                mean_offset=float(log_layer.get('mean_offset', float('nan'))),
                points=int(log_layer.get('cells', 0))),
            tolerance=float(TOLERANCE),
        ),
        globals=dict(bulk_plus=bulk_plus, bulk_plus_reference=u_bulk_ref,
                     Cf=cf, Cf_reference=float(dns.get('Cf', float('nan'))),
                     wall_shear=float(solver.wall_shear()),
                     uv_plus_min=float(np.min(stats.uv_plus)),
                     identity_residual_max=float(np.max(np.abs(identity['residual']))),
                     identity_residual_log=float(np.max(
                         np.abs(identity['residual'])[stats.y_plus >= LOW_Y_PLUS])),
                     u_plus_at=dict((str(int(s)), float(np.interp(
                         s, stats.y_plus, stats.u_plus))) for s in STATIONS)),
        reference=dict(y_plus=dns['y_plus'].tolist(), u_plus=dns['u_plus'].tolist(),
                       re_tau=float(dns['re_tau']),
                       source='Lee & Moser 2015 DNS (in-tree proxy for KMM1987)'),
        samples=int(stats.samples), time_span=float(stats.time_span),
        dt=float(dt), dt_plus=float(dt * config.re_tau),
        environment=dict(python=sys.version.split()[0],
                         numpy=np.__version__, platform=platform.platform()),
    )

    verdict = ('PASS' if summary['anchor']['log_layer']['rms_relative'] < TOLERANCE
               else 'FAIL')
    summary['verdict'] = dict(
        anchor=verdict,
        admissible=gates['admissible'],
        published=('PASS' if verdict == 'PASS' and gates['admissible'] else
                   'VACUOUS_PASS' if verdict == 'PASS' else 'FAIL'))

    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    write_profiles(out / 'profiles.csv', stats, identity)
    write_history(out / 'transition.csv', history)
    return summary


def write_profiles(path: Path, stats, identity) -> None:
    rows = ['y_plus,u_plus,du_plus_dy_plus,u_rms_plus,v_rms_plus,w_rms_plus,'
            'uv_plus,total_shear_plus,target_plus,residual']
    for i in range(stats.y_plus.size):
        rows.append(','.join(f'{v:.6f}' for v in (
            stats.y_plus[i], stats.u_plus[i], stats.du_plus_dy_plus[i],
            math.sqrt(max(stats.uu_plus[i], 0.0)),
            math.sqrt(max(stats.vv_plus[i], 0.0)),
            math.sqrt(max(stats.ww_plus[i], 0.0)),
            stats.uv_plus[i], identity['total'][i], identity['target'][i],
            identity['residual'][i])))
    path.write_text('\n'.join(rows) + '\n')


def write_history(path: Path, history) -> None:
    if not history:
        path.write_text('')
        return
    keys = list(history[0].keys())
    rows = [','.join(keys)]
    for row in history:
        rows.append(','.join(
            (f'{row[k]:.6g}' if isinstance(row[k], float) else str(row[k]))
            for k in keys))
    path.write_text('\n'.join(rows) + '\n')


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-seeded')
    parser.add_argument('--label', default='seeded')
    parser.add_argument('--sweep', choices=('none', 'pilot', 'strong'),
                        default='none')
    parser.add_argument('--shard-idx', type=int, default=0)
    parser.add_argument('--re-tau', type=float, default=180.0)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--length-x', type=float, default=2.0 * math.pi)
    parser.add_argument('--length-z', type=float, default=math.pi)
    parser.add_argument('--stretching', type=float, default=1.5)
    parser.add_argument('--fft-workers', type=int, default=1)
    parser.add_argument('--amplitude', type=float, default=1.5)
    parser.add_argument('--spanwise-mode', type=int, default=2)
    parser.add_argument('--sinusity', type=float, default=0.3)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--steps-transient', type=int, default=2500)
    parser.add_argument('--steps-average', type=int, default=6000)
    parser.add_argument('--sample-every', type=int, default=250)
    parser.add_argument('--progress-every', type=int, default=250)
    args = parser.parse_args(argv)

    config = build_config(args)
    params = dict(amplitude=args.amplitude, spanwise_mode=args.spanwise_mode,
                  sinusity=args.sinusity, noise=args.noise, seed=args.seed,
                  label=args.label, streamwise_mode=1)
    if args.sweep != 'none':
        table = PILOT_SWEEP if args.sweep == 'pilot' else STRONG_SWEEP
        entry = dict(table[args.shard_idx % len(table)])
        overrides = {k: entry.pop(k) for k in ('length_x', 'length_z', 'nz')
                     if k in entry}
        if overrides:
            config = build_config(argparse.Namespace(**{**vars(args), **overrides}))
        params.update(entry)
    out = Path(args.out)
    summary = run_labeled(config, steps_transient=args.steps_transient,
                          steps_average=args.steps_average,
                          sample_every=args.sample_every,
                          progress_every=args.progress_every, out=out, **params)
    print(json.dumps(dict(label=summary['label'],
                          verdict=summary['verdict'],
                          gates={k: summary['gates'][k] for k in
                                 ('uv_share', 'balance_residual', 'flow_throughs',
                                  'time_plus', 'admissible')},
                          anchor=summary['anchor'],
                          globals=summary['globals']), indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())