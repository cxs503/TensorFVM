#!/usr/bin/env python
"""Shardable leg runner for node P447W2-N01.

    PYTHONPATH=src python docs/P447W2-N01-vof-three-case/leg_runner.py --leg dam --index 1 --count 3

``--index i --count n`` selects every ``n``-th configuration of a leg, so the
same script serves a laptop (``count 1``) and the cluster (``count 3``).  Each
shard writes ``artifacts/<leg>.<key>.json`` plus (for the runs whose fields the
report figures need) ``artifacts/fields/<key>.npz``.  Shards are independent:
a killed shard costs one configuration, and ``merge`` rebuilds the acceptance
record from whatever exists.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(max(1, int(os.environ.get("TP_THREADS", "16"))))

from tensorfvm.vof_threecase.dambreak import DamBreakMMConfig, run_dam_break
from tensorfvm.vof_threecase.statics import StaticBubbleConfig, run_static_bubble
from tensorfvm.vof_threecase.zalesak import NotchConfig, notch_mask, run_long_rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIELDS = os.path.join(ART, "fields")

#: 50x100 / 100x200 / 200x400 square cells in a 2:1 tank (the node's grid sequence)
DAM_GRIDS = ((50, 100), (100, 200), (200, 400))


def dam_configs():
    return [DamBreakMMConfig(ny=ny, nx=nx, aspect=2.0, column_width=0.05715,
                             length=0.4, height=0.2) for ny, nx in DAM_GRIDS]


def static_configs():
    """Spurious-current matrix.

    ``mu`` sets the viscous time at the grid scale, ``tau_mu = rho h^2/mu``,
    which is the time the current needs to saturate; each point is run for
    ``damping_times * tau_mu`` (capped), so a point either saturates inside its
    budget (``saturated = True``) or is reported as a transient.  The matrix is
    organised as three sweeps plus two controls:

    * resolution sweep at ``mu = 0.3 Pa s`` -- the spurious current is a
      grid-scale phenomenon, so this is the sweep the criterion is judged on;
    * viscosity (Laplace-number) sweep at ``N = 32/64``;
    * one high-La point per resolution, run inside ``damping_times < 1`` so the
      transient is visible and labelled as such rather than quoted as steady;
    * controls: cell-centred (legacy) CSF and ``kappa = div(n)`` curvature.
    """
    cfgs = []
    # resolution sweep at mu = 0.3 Pa s (tau_mu = 0.2 .. 3.3 s)
    for cells in (32, 64, 96, 128):
        cfgs.append(StaticBubbleConfig(cells=cells, mu_liquid=3.0e-1, mu_gas=3.0e-1,
                                       damping_times=3.0))
    # viscosity sweep at N = 32 / 64
    cfgs.append(StaticBubbleConfig(cells=32, mu_liquid=1.0, mu_gas=1.0, damping_times=3.0))
    cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=1.0, mu_gas=1.0, damping_times=3.0))
    cfgs.append(StaticBubbleConfig(cells=32, mu_liquid=1.0e-2, mu_gas=1.0e-2,
                                   damping_times=0.2, max_steps=1500))
    cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=1.0e-2, mu_gas=1.0e-2,
                                   damping_times=0.1, max_steps=1200))
    # controls: cell-centred (legacy) CSF and kappa = div(n), same metric
    cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=3.0e-1, mu_gas=3.0e-1,
                                   csf_balance="legacy", damping_times=3.0))
    cfgs.append(StaticBubbleConfig(cells=64, mu_liquid=3.0e-1, mu_gas=3.0e-1,
                                   csf_curvature="divergence", damping_times=3.0))
    return cfgs


def zalesak_configs():
    cfgs = []
    for cells in (64, 128, 256):
        for deg in (360.0, 1000.0):
            cfgs.append(NotchConfig(cells=cells, scheme="cicsam", degrees=deg))
    for cells in (64, 128):
        cfgs.append(NotchConfig(cells=cells, scheme="plic", degrees=1000.0))
        cfgs.append(NotchConfig(cells=cells, scheme="plic", degrees=360.0))
        cfgs.append(NotchConfig(cells=cells, scheme="upwind", degrees=360.0))
    return cfgs


def static_key(c):
    tail = "" if c.profile == "ramp" else f"_{c.profile}"
    return f"N{c.cells}_mu{c.mu_liquid:g}_{c.csf_balance}_{c.csf_curvature}{tail}"


def zalesak_key(c):
    return f"{c.scheme}_N{c.cells}_{c.degrees:g}deg"


def dam_key(c):
    return f"{c.ny}x{c.nx}"


def _dump(name, payload):
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, name + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[write] {path}", flush=True)


def run_static(cfg):
    t0 = time.time()
    rec = run_static_bubble(cfg)
    solver = rec.pop("_solver")
    key = static_key(cfg)
    os.makedirs(FIELDS, exist_ok=True)
    u = solver.u.detach().numpy()
    v = solver.v.detach().numpy()
    np.savez_compressed(
        os.path.join(FIELDS, f"static_{key}.npz"),
        alpha=solver.alpha.detach().numpy(),
        speed=np.sqrt((0.5 * (u[:, :-1] + u[:, 1:])) ** 2
                      + (0.5 * (v[:-1, :] + v[1:, :])) ** 2))
    print(f"[static] {key} ratio={rec['spurious_ratio']:.3e} "
          f"saturated={rec['saturated']} jump={rec['jump_error_relative']:+.3f} "
          f"steps={rec['steps']} wall={time.time() - t0:.0f}s", flush=True)
    _dump(f"static.{key}", rec)


def run_zalesak(cfg):
    t0 = time.time()
    rec = run_long_rotation(cfg)
    alpha = rec.pop("_field")
    exact = rec.pop("_exact")
    key = zalesak_key(cfg)
    os.makedirs(FIELDS, exist_ok=True)
    np.savez_compressed(os.path.join(FIELDS, f"zalesak_{key}.npz"),
                        alpha=alpha.numpy(), exact=exact.numpy(),
                        notch=notch_mask(cfg.cells).numpy())
    print(f"[zalesak] {key} notch={rec['notch_error_relative'] * 100:.2f}% "
          f"area={rec['area_error_contour_vs_initial'] * 100:.2f}% "
          f"drift={rec['mass_drift_relative']:.2e} wall={time.time() - t0:.0f}s", flush=True)
    _dump(f"zalesak.{key}", rec)


def run_dam(cfg):
    t0 = time.time()
    rec = run_dam_break(cfg)
    alpha = rec.pop("_alpha_final")
    rec.pop("_solver")
    rec["wall_seconds"] = time.time() - t0
    key = dam_key(cfg)
    os.makedirs(FIELDS, exist_ok=True)
    np.savez_compressed(os.path.join(FIELDS, f"dam_{key}.npz"), alpha=alpha.numpy())
    head = " ".join(f"t={s['t']:.2f}:{s['relative_error'] * 100:+.1f}%"
                    for s in rec["samples"])
    print(f"[dam] {key} {head} drift={rec['mass_drift_max']:.2e} "
          f"steps={rec['steps']} wall={time.time() - t0:.0f}s", flush=True)
    _dump(f"dam.{key}", rec)


LEGS = {"static": (static_configs, run_static, static_key),
        "zalesak": (zalesak_configs, run_zalesak, zalesak_key),
        "dam": (dam_configs, run_dam, dam_key)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=sorted(LEGS))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--only", default=None, help="run only this configuration key")
    args = ap.parse_args()
    cfgs_fn, runner, key_fn = LEGS[args.leg]
    for i, cfg in enumerate(cfgs_fn()):
        if args.only is not None:
            if key_fn(cfg) != args.only:
                continue
        elif i % args.count != args.index:
            continue
        runner(cfg)


if __name__ == "__main__":
    main()