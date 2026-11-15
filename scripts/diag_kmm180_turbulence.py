#!/usr/bin/env python
"""Diagnostic: does the closure-free channel solver develop Reynolds stress?

Tests whether the spectral+FV channel solver sustains turbulence at a given
resolution by tracking the resolved Reynolds-shear share in the log layer over
a run.  Writes a ``turbulence_diag.json`` summary (time, bulk, uv_share_log,
uv_max, u_rms) so the evidence is captured in the worktree, and prints a table.

This is the "管到底" diagnostic for the KMM180 closure-free route: it answers
whether the vacuous PASS (m20573) can be repaired by running longer / finer, or
whether the route is fundamentally unable to reach a turbulent state.

Run::

    PYTHONPATH=src python scripts/diag_kmm180_turbulence.py \
        --nx 64 --ny 128 --nz 64 --steps 20000 --out docs/case-fvm-channel-kmm180-dns
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from tensorfvm.verification.channel_spectral_dns import (  # noqa: E402
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
)


def probe(solver, n=30):
    acc = StatisticsAccumulator(solver)
    for _ in range(n):
        solver.step()
        acc.add()
    st = acc.folded()
    mask = (st.y_plus >= 30) & (st.y_plus <= 36)
    target = 1.0 - st.y_plus / 180.0
    share = -st.uv_plus[mask] / target[mask]
    return dict(t=float(solver.time), bulk=float(solver.bulk_velocity()),
                uv_share=float(share.mean()),
                uv_max=float(np.abs(st.uv_plus).max()),
                u_rms=float(np.sqrt(np.maximum(st.uu_plus, 0)).max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--ny', type=int, default=128)
    ap.add_argument('--nz', type=int, default=64)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--noise', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--probe-every', type=int, default=500)
    ap.add_argument('--out', default='docs/case-fvm-channel-kmm180-dns')
    args = ap.parse_args()

    config = ChannelConfig(re_tau=180.0, nx=args.nx, ny=args.ny, nz=args.nz)
    solver = SpectralChannel(config)
    res = solver.resolution_report()
    print(f"grid {args.nx}x{args.ny}x{args.nz} dt={solver.time_step:.5f} "
          f"dx+={res['dx_plus']:.1f} dz+={res['dz_plus']:.1f}", flush=True)
    solver.initialize(noise=args.noise, seed=args.seed)
    print(f"initial bulk={solver.bulk_velocity():.2f}", flush=True)

    rows = []
    for i in range(args.steps):
        solver.step()
        if i % args.probe_every == 0:
            p = probe(solver)
            rows.append(p)
            print(f"  t={p['t']:.2f} bulk={p['bulk']:.2f} "
                  f"uv_share_log={p['uv_share']:.3f} uv_max={p['uv_max']:.3e} "
                  f"u_rms={p['u_rms']:.3f}", flush=True)

    # C1 admission check: does the resolved Reynolds-shear share ever reach 0.5?
    c1_met = any(r['uv_share'] >= 0.50 for r in rows)
    summary = dict(
        schema='tensorfvm.kmm180-turbulence-diag/1',
        anchor='CASE-FVM-CHANNEL-KMM180',
        route='closure-free (spectral x,z + finite-volume y, no turbulence model)',
        grid=dict(nx=args.nx, ny=args.ny, nz=args.nz,
                  dx_plus=res['dx_plus'], dz_plus=res['dz_plus'],
                  dy_plus_min=res['dy_plus_min']),
        run=dict(steps=args.steps, noise=args.noise, seed=args.seed,
                 dt=solver.time_step),
        c1_uv_share_met=c1_met,
        c1_uv_share_max=float(max((r['uv_share'] for r in rows), default=0.0)),
        uv_max_final=float(rows[-1]['uv_max']) if rows else None,
        u_rms_final=float(rows[-1]['u_rms']) if rows else None,
        bulk_final=float(rows[-1]['bulk']) if rows else None,
        rows=rows,
        conclusion=('ADMISSIBLE' if c1_met else
                    'VACUOUS_PASS_NOT_REPAIRABLE: no resolved Reynolds shear '
                    'develops; the closure-free u+ grades the initial condition, '
                    'not a turbulent solution (falsified hypothesis).'),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'turbulence_diag.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f"wrote {out / 'turbulence_diag.json'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
