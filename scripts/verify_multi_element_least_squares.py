"""Produce the verification artifact for the shared least-squares pressure gradient.

Runs the operator-level checks of :mod:`tensorfvm.verification.multi_element_ls`
on a mesh-doubling family (uniform and exponentially clustered), writes a machine
readable record and two figures, and exits non-zero if a gate fails.

Usage::

    python scripts/verify_multi_element_least_squares.py [--output DIR]
"""
import argparse
import json
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorfvm.verification.multi_element_ls import (linear_exactness, operator_consistency,
                                                    order_study, smooth_field, smooth_gradient,
                                                    solver_for, solver_step_defect)

ARTIFACT = pathlib.Path("src/tensorfvm/verification/artifacts/multi_element_least_squares")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/figures")
    parser.add_argument("--levels", default="8,16,32")
    args = parser.parse_args()
    torch.set_num_threads(1)
    levels = tuple(int(v) for v in args.levels.split(","))
    figures = pathlib.Path(args.output)
    figures.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as directory:
        stretched = solver_for(directory, 24, 8, 6.0)
        measured = dict(
            linear=linear_exactness(stretched),
            consistency=operator_consistency(stretched),
            orders={str(k): v for k, v in
                    order_study(directory, levels=levels).items()},
            step=solver_step_defect(directory),
        )

        # Figure 1: volume-weighted L2 gradient error versus mean cell size, with the
        # second-order reference slope, one curve per exponential clustering exponent.
        fig, axes = plt.subplots(figsize=(5.2, 4.0))
        for stretch, rows in measured["orders"].items():
            spacing = [row["spacing"] for row in rows]
            error = [row["error"] for row in rows]
            axes.loglog(spacing, error, "o-", label=f"clustering exp($\\beta$={stretch})")
            if len(rows) > 1:
                orders = [row["observed_order"] for row in rows[1:]]
                axes.annotate("p=" + ", ".join(f"{p:.2f}" for p in orders),
                              xy=(spacing[-1], error[-1]), textcoords="offset points",
                              xytext=(6, -10), fontsize=7)
        reference_x = np.array(axes.get_xlim())
        curve = measured["orders"][list(measured["orders"])[0]]
        axes.loglog(reference_x, curve[0]["error"] * (reference_x / curve[0]["spacing"]) ** 2,
                    "k--", linewidth=1, label="second order")
        axes.set_xlabel("mean cell size $\\sqrt{\\Omega}$ [m]")
        axes.set_ylabel("volume-weighted $L_2$ gradient error [-]")
        axes.set_title("Least-squares gradient on clustered multi-element meshes")
        axes.grid(True, which="both", alpha=0.3)
        axes.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "least_squares_order.png", dpi=105)
        plt.close(fig)

        # Figure 2: pointwise gradient-error field on the clustered mesh, showing the
        # localisation of the error at the coarse far-field cells and the wall layer.
        mesh = stretched.mesh
        cells = mesh.centers.reshape(-1, 2).numpy()
        approximated = stretched._gradient(
            torch.as_tensor(smooth_field(cells)), pressure=True).numpy()
        local = np.linalg.norm(approximated - smooth_gradient(cells), axis=1)
        vertices = mesh.vertices[:, :2].numpy()
        triangles = [[cell[0], cell[1], cell[2]] for cell in mesh.cell_nodes]
        triangles += [[cell[0], cell[2], cell[3]] for cell in mesh.cell_nodes]
        fig, axes = plt.subplots(figsize=(6.0, 3.0))
        cloud = axes.tripcolor(vertices[:, 0], vertices[:, 1], triangles, np.repeat(local, 2),
                               shading="flat", cmap="magma")
        axes.set_aspect("equal")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_title("Pointwise least-squares gradient error, 24x8, exp(=6) clustering")
        fig.colorbar(cloud, ax=axes, label="$|\\nabla q_h-\\nabla q|$ [-]")
        fig.tight_layout()
        fig.savefig(figures / "least_squares_error_field.png", dpi=105)
        plt.close(fig)

    (ARTIFACT / "study.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps({k: (v if k != "orders" else {s: [{kk: vv for kk, vv in row.items()}
                                                     for row in rows]
                                                   for s, rows in v.items()})
                      for k, v in measured.items()}, indent=2))
    orders = [row["observed_order"] for rows in measured["orders"].values()
              for row in rows if "observed_order" in row]
    ok = (measured["linear"]["interior_error"] < 1e-12
          and measured["consistency"]["gradient_difference"] < 1e-11
          and min(orders) > 1.5)
    print("gate", ok, "min observed order", min(orders))
    assert ok, "verification gate failed"


if __name__ == "__main__":
    main()
