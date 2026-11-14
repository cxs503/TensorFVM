"""Development probe (m21018): immersed-cylinder Re=100 force history.

Not part of the delivered capability surface; kept as the driver used to size the
production runs (force sign, shedding onset time, time-step stability).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.immersed.forces import ForceRecorder, control_box
from tensorfvm.immersed.geometry import circular_cylinder
from tensorfvm.immersed.solver import ImmersedConfig, ImmersedFlow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpd", type=int, default=20)
    ap.add_argument("--t-end", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--perturb", type=float, default=0.1)
    a = ap.parse_args()
    D = U = 1.0
    Re = 100.0
    g = circular_cylinder(a.cpd, diameter=D)
    cfg = ImmersedConfig(grid=g, viscosity=U * D / Re, density=1.0, projection="masked",
                         inlet_mean_velocity=U, inlet_bounds=(-10 * D, 10 * D),
                         reference_length=D, dt=a.dt, t_end=a.t_end,
                         max_steps=10 ** 7, poisson_tol=1e-10)
    f = ImmersedFlow(cfg)
    f.inlet_profile = torch.ones_like(f.inlet_profile)  # uniform freestream
    if a.perturb:
        xc = g.x0 + (torch.arange(g.nx, dtype=torch.float64) + 0.5) * g.dx
        yf = g.y0 + torch.arange(g.ny + 1, dtype=torch.float64) * g.dy
        X = xc[None, :]
        Y = yf[:, None]
        f.v = f.v + a.perturb * torch.exp(-(((X - 0.8 * D) / (0.4 * D)) ** 2
                                           + ((Y - 0.25 * D) / (0.25 * D)) ** 2))
    box = control_box(g, x_min=-2 * D, x_max=3 * D, y_min=-1.5 * D, y_max=1.5 * D)
    rec = ForceRecorder(f, box)
    q = 0.5 * cfg.density * U * U * D
    t0 = time.perf_counter()
    nxt = 1.0
    while f.ledger.time < a.t_end - 1e-12:
        f.step()
        s = rec.sample(f.ledger.time)
        t = f.ledger.time
        if t >= nxt:
            print(f"t={t:6.2f} Cd={s.fx / q:+.4f} Cl={s.fy / q:+.4f} "
                  f"unst={s.fx_unsteady / q:+.4f} steps={f.ledger.steps} "
                  f"umax={f.snapshot()['u_max']:.3f}", flush=True)
            nxt += 2.0
    wall = time.perf_counter() - t0
    print(f"wall {wall:.1f}s, per step {wall / max(f.ledger.steps, 1):.4f}s, "
          f"grid {g.nx}x{g.ny}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())