"""Field snapshot figure for the floating-section node (forced heave, calm water)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.wave_tank_td import seakeeping as sk
from tensorfvm.wave_tank_td.sigma_tank import SigmaWaveTank, TankConfig

FIG = Path("docs/wave-tank-td-body/figures")
SEC = sk.floating_strip(1.2, 0.2, freeboard=0.05, x0=4.0)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    period = 1.0
    omega = 2 * math.pi / period
    t = SigmaWaveTank(TankConfig(length=12.0, nx=220, ns=24))
    t.set_sponge(period)
    t.set_body(SEC.polygon, mass=SEC.mass, inertia=SEC.inertia, ref=SEC.ref,
               motion="forced", forced={"dof": 0, "amp": 0.01, "omega": omega})
    dt = period / 60
    etas = []
    for k in range(int(2.5 * 60)):
        t.step(dt)
        if k % 30 == 0:
            etas.append((float(t.t), t.eta.copy()))
    # sigma (vertical) structure of the field: q on the physical cells
    y = t.yc
    fig, ax = plt.subplots(2, 1, figsize=(11.0, 6.6))
    x2 = np.broadcast_to(t.x[None, :], t.p.T.shape)
    mesh = ax[0].pcolormesh(x2, y.T, t.p.T, shading="nearest", cmap="RdBu_r")
    ax[0].plot(*np.vstack([t.body_polygon(), t.body_polygon()[0]]).T, "k-", lw=1.6)
    ax[0].set_ylim(-1.0, 0.05)
    ax[0].set_xlim(2.0, 6.5)
    ax[0].set_ylabel("y [m]")
    ax[0].set_xlabel("x [m]")
    ax[0].set_title("non-hydrostatic pressure $q/\\rho$ [m$^2$/s$^2$], forced heave "
                    "$a=1$ cm, $T=%.1f$ s" % period)
    fig.colorbar(mesh, ax=ax[0], label="$q/\\rho$")
    for tt, eta in etas:
        ax[1].plot(t.x, eta, lw=1.0, label="t=%.2f s" % tt)
    ax[1].plot(t.x, np.full_like(t.x, 0.0), color="0.7", lw=0.8)
    ax[1].set_xlim(2.0, 6.5)
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("$\\eta$ [m]")
    ax[1].set_title("free-surface elevation around the section (deck region is dry)")
    ax[1].legend(fontsize=7, ncol=3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "field_snapshot.png", dpi=140)
    print("wrote", FIG / "field_snapshot.png")


if __name__ == "__main__":
    main()
