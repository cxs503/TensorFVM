#!/usr/bin/env python3
"""Write the annular grid-convergence evidence bundle.

Runs the production annular Poiseuille solve over a refinement sequence, checks
the closed form against an independent finite-difference solution of the
differential statement, and writes

    src/tensorfvm/verification/artifacts/annular_grid_convergence.json   machine-readable evidence
    src/tensorfvm/verification/artifacts/annular_grid_convergence.png    error-vs-spacing figure

Usage:  OMP_NUM_THREADS=1 python scripts/verify_annular_grid_convergence.py [--levels 4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from tensorfvm.annular import AnnularConfig, analytic_flow, analytic_velocity  # noqa: E402
from tensorfvm.verification.annular_convergence import study  # noqa: E402


def figure(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    h = np.array([row["spacing_m"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for key, label in (("relative_l2", r"$\|u-u_{exact}\|_{L^2(V)}/\|u_{exact}\|$"),
                       ("relative_linf", r"$\|u-u_{exact}\|_\infty/\|u_{exact}\|_\infty$"),
                       ("flow_rate_relative_error", "flow-rate relative error")):
        ax.loglog(h, [row[key] for row in rows], "o-", label=label)
    reference = h ** 2
    ax.loglog(h, reference * rows[0]["relative_l2"] / reference[0], "k--", lw=1, label=r"$\propto h^2$")
    ax.set_xlabel("cell spacing h (m)")
    ax.set_ylabel("relative error")
    ax.set_title(
        "Annular Poiseuille: observed order "
        f"p={report['l2_loglog_slope']['slope']:.3f} (L2), "
        f"{report['flow_rate_order']['order']:.3f} (flow rate)")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    config = AnnularConfig()
    from tensorfvm.verification.annular_convergence import _fd_annular_profile

    r, u, _ = _fd_annular_profile(config, 1201)
    ax.plot(r * 1e3, analytic_velocity(r, config), "k-", lw=2, label="closed form (backend)")
    ax.plot(r * 1e3, u, "r--", lw=1, label="independent 2nd-order FD BVP")
    finest = rows[-1]
    centre = np.linalg.norm(
        np.array(np.meshgrid(
            np.linspace(config.inner_radius_m, config.outer_radius_m, finest["grid"][1] + 1),
            np.linspace(0, 2 * math.pi, finest["grid"][0] + 1), indexing="ij"))[0], axis=0)
    ax.set_xlabel("radius r (mm)")
    ax.set_ylabel("axial velocity u (m/s)")
    ax.set_title("Exact profile for the benchmark (G = 0.01 Pa/m, mu = 1 mPa s)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("src/tensorfvm/verification/artifacts"))
    args = parser.parse_args(argv)

    report = study(levels=args.levels)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "annular_grid_convergence.json"
    json_path.write_text(json.dumps(report, indent=2))

    png_path = args.out / "annular_grid_convergence.png"
    figure(report, png_path)

    flow = report["flow_rate_order"]
    print(f"levels           : {report['levels']}")
    print(f"L2 log-log slope : {report['l2_loglog_slope']['slope']:.4f}")
    print(f"flow-rate order p: {flow['order']:.4f}  GCI_fine={flow['gci_fine_relative']:.3e}")
    print(f"Richardson vs analytic: {report['extrapolated_vs_analytic_relative_deviation']:.3e} "
          f"(band {report['extrapolation_gci_band_relative']:.3e}, "
          f"agrees={report['extrapolation_agrees_with_analytic']})")
    print(f"analytic baseline (independent FD): "
          f"{report['analytic_reference']['finest_fd_max_relative_deviation']:.3e}, "
          f"FD orders {report['analytic_reference']['fd_observed_orders']}")
    print(f"wrote {json_path} and {png_path}")
    ok = (
        abs(report["l2_loglog_slope"]["slope"] - 2.0) < 0.1
        and report["extrapolation_agrees_with_analytic"]
        and report["analytic_reference"]["finest_fd_max_relative_deviation"] < 1e-6
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())