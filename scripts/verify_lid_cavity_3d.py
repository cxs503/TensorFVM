"""Verification of the 3-D lid-driven cavity solver against the 2-D Ghia benchmark.

The 3-D solver (``tensorfvm.lid_cavity_3d``) extends the 2-D staggered grid to
a prism ``n_x \u00d7 n_y \u00d7 n_z`` with periodic boundaries in the spanwise
(z) direction.  For sufficiently large ``n_z`` the flow should converge to the
2-D solution, so we can reuse the Ghia reference tables as an independent
accuracy anchor.

This module measures the L2 error of the centreline velocity profiles against
the published Ghia data for Re = 100, 400, 1000 on a ``32 \u00d7 32 \u00d7 4``
mesh, and verifies that the spanwise velocity component ``w`` remains at machine
zero.
"""
from __future__ import annotations

import numpy as np
import torch

from tensorfvm.lid_cavity_3d import LidCavity3DConfig, LidCavity3DSolver
from tensorfvm.lid_cavity_reference import table, extrema


def centreline_error(n_x: int = 32, reynolds: float = 100.0, n_z: int = 4) -> dict:
    """Run the 3-D solver and compare centreline profiles with the 2-D reference."""
    config = LidCavity3DConfig(
        n_x=n_x, n_y=n_x, n_z=n_z,
        reynolds=reynolds,
        lid_velocity=1.0,
        tolerance=1e-7,
        max_steps=20_000,
        check_every=250,
    )
    solver = LidCavity3DSolver(config)
    info = solver.run()
    assert info["converged"], info["history"][-3:]
    assert info["max_divergence"] < 1e-11

    # Extract centreline profiles (y = 0.5)
    y_idx = n_x // 2
    u_profile = solver.u[y_idx, :].cpu().numpy()
    v_profile = solver.v[y_idx, :].cpu().numpy()
    w_profile = solver.w[y_idx, :].cpu().numpy()

    # Compare with 2-D reference
    ref_u = np.array([p[1] for p in table(reynolds, "u")])
    ref_v = np.array([p[1] for p in table(reynolds, "v")])
    coord = np.array([p[0] for p in table(reynolds, "u")])

    u_error = np.sqrt(np.mean((u_profile - ref_u) ** 2))
    v_error = np.sqrt(np.mean((v_profile - ref_v) ** 2))
    w_max = np.max(np.abs(w_profile))

    return dict(
        reynolds=reynolds,
        n_x=n_x,
        n_z=n_z,
        u_l2_error=float(u_error),
        v_l2_error=float(v_error),
        w_max=float(w_max),
        converged=info["converged"],
        iterations=info["iterations"],
    )


def sweep() -> list:
    """Run the verification sweep for Re = 100, 400, 1000."""
    results = []
    for re in [100.0, 400.0, 1000.0]:
        results.append(centreline_error(n_x=32, reynolds=re, n_z=4))
    return results


def main() -> list:
    results = sweep()
    for r in results:
        print(f"Re={r['reynolds']}: u_l2={r['u_l2_error']:.6f}, v_l2={r['v_l2_error']:.6f}, w_max={r['w_max']:.2e}")
    return results


if __name__ == "__main__":
    main()
