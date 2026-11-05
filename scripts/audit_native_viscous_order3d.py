"""Audit: order of accuracy of the native 3-D viscous operator (NativeViscous3D).

Runs the manufactured-field study in :mod:`tensorfvm.verification.native_order3d`
on uniform and warped hexahedral families and writes JSON + figures.

    python scripts/audit_native_viscous_order3d.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from tensorfvm.native_viscous3d import NativeViscous3D  # noqa: E402
from tensorfvm.verification import native_order3d as study  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
GRIDS = (4, 8, 16)
WARPS = (0.0, 2.0)
VISCOSITY = 1.3


def _finite_difference_check(points, viscosity):
    """Verify the analytic divergence_of_stress against central differences."""
    step = 1e-6

    def stress_at(p):
        g = np.zeros((len(p), 3, 3))
        for b in range(3):
            unit = np.zeros(3)
            unit[b] = step
            g[:, :, b] = (
                study.manufactured(p + unit) - study.manufactured(p - unit)
            ) / (2 * step)
        return viscosity * (
            g
            + g.transpose(0, 2, 1)
            - 2 / 3 * np.trace(g, axis1=1, axis2=2)[:, None, None] * np.eye(3)
        )

    numeric = np.zeros((len(points), 3))
    for j in range(3):
        unit = np.zeros(3)
        unit[j] = step
        derivative = (stress_at(points + unit) - stress_at(points - unit)) / (2 * step)
        numeric += derivative[:, :, j]
    analytic = study.divergence_of_stress(points, viscosity)
    return float(np.abs(numeric - analytic).max() / max(np.abs(analytic).max(), 1e-30))


def _figure(truncation, faces, solutions, mesh, op, data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for warp, marker in ((0.0, "o"), (2.0, "s")):
        rows = [r for r in truncation if r["warp"] == warp]
        h = [r["h"] for r in rows]
        axes[0].loglog(
            h,
            [r["l2_global"] for r in rows],
            marker + "-",
            label=f"global, warp={warp}",
        )
        axes[0].loglog(
            h,
            [r["l2_interior"] for r in rows],
            marker + "--",
            label=f"interior, warp={warp}",
        )
        axes[0].loglog(
            h,
            [r["l2_wall"] for r in rows],
            marker + ":",
            label=f"wall cells, warp={warp}",
        )
        fr = [r for r in faces if r["warp"] == warp]
        axes[1].loglog(
            h,
            [r["rel_internal_per_area"] for r in fr],
            marker + "-",
            label=f"internal face, warp={warp}",
        )
        axes[1].loglog(
            h,
            [r["rel_boundary_per_area"] for r in fr],
            marker + "--",
            label=f"wall face, warp={warp}",
        )
        axes[1].loglog(
            h,
            [r["rel_wall_normal_derivative"] for r in fr],
            marker + ":",
            label=f"wall du/dn, warp={warp}",
        )
        sr = [r for r in solutions if r["warp"] == warp]
        axes[2].loglog(
            h, [r["l2_global"] for r in sr], marker + "-", label=f"global, warp={warp}"
        )
        axes[2].loglog(
            h,
            [r["l2_interior"] for r in sr],
            marker + "--",
            label=f"interior, warp={warp}",
        )
    axes[0].set_title("Cell divergence error of div(tau)")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("L2 error")
    axes[1].set_title("Face traction error / area, wall du/dn")
    axes[1].set_xlabel("h")
    axes[2].set_title("Dirichlet diffusion solve: |u_h - u_exact|")
    axes[2].set_xlabel("h")
    for ax in axes[:3]:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    # defect localisation: mid-plane error cloud (log10)
    n = getattr(_figure, "_n")
    plane = np.log10(np.maximum(getattr(_figure, "_cloud"), 1e-18)).reshape(n, n, n)[
        :, n // 2, :
    ]
    axis = (np.arange(n) + 0.5) / n
    im = axes[3].pcolormesh(axis, axis, plane, shading="nearest")
    axes[3].set_title("log10 |div(tau)_h - div(tau)| (all cells, n=16 warp=0)")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("z")
    fig.colorbar(im, ax=axes[3])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    probes = np.array(
        [[0.13, 0.27, 0.41], [0.71, 0.12, 0.88], [0.5, 0.5, 0.5], [0.93, 0.85, 0.07]]
    )
    fd_rel = _finite_difference_check(probes, VISCOSITY)

    truncation, faces, solutions = [], [], []
    for warp in WARPS:
        for n in GRIDS:
            truncation.append(study.truncation_study(n, warp, VISCOSITY))
            faces.append(study.face_traction_study(n, warp, VISCOSITY))
            solutions.append(study.solution_study(n, warp, VISCOSITY))
            print(json.dumps(truncation[-1]), flush=True)
            print(json.dumps(faces[-1]), flush=True)
            print(json.dumps(solutions[-1]), flush=True)

    summary = {}
    for warp in WARPS:
        tag = f"warp{warp}"
        tr = [r for r in truncation if r["warp"] == warp]
        sr = [r for r in solutions if r["warp"] == warp]
        fr = [r for r in faces if r["warp"] == warp]
        summary[tag] = dict(
            truncation_l2_global=study.orders(tr, "l2_global"),
            truncation_l2_interior=study.orders(tr, "l2_interior"),
            truncation_l2_wall=study.orders(tr, "l2_wall"),
            traction_per_area_internal=study.orders(fr, "rel_internal_per_area"),
            traction_per_area_wall=study.orders(fr, "rel_boundary_per_area"),
            wall_normal_derivative=study.orders(fr, "rel_wall_normal_derivative"),
            interior_normal_derivative=study.orders(
                fr, "interior_normal_derivative_error"
            ),
            solution_l2_global=study.orders(sr, "l2_global"),
            solution_l2_interior=study.orders(sr, "l2_interior"),
        )

    # affine exactness (machine-precision anchor)
    mesh = study.hex_box(4, 0.0)
    op = NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY)
    grad = np.array([[0.7, 0.2, -0.1], [-0.3, 0.4, 0.6], [0.15, -0.25, 0.1]])
    base = np.array([1.2, 0.8, -0.4])
    affine_faces = np.zeros((op.F, 3))
    affine_faces[op.dirichlet] = base + mesh.face_centers[op.dirichlet] @ grad.T
    flux = op.momentum_flux(
        base + mesh.cell_centers @ grad.T, dirichlet_values=affine_faces
    )
    sym = grad + grad.T - 2 / 3 * np.trace(grad) * np.eye(3)
    expect = VISCOSITY * mesh.area_vectors @ sym.T
    affine_error = float(np.abs(flux - expect).max() / np.abs(expect).max())
    div_affine = float(np.abs(op.op.integrate(flux) / mesh.cell_volumes[:, None]).max())

    # cloud figure uses n=16 uniform mesh divergence error
    big = study.hex_box(16, 0.0)
    big_op = NativeViscous3D(big, big.neighbor < 0, VISCOSITY)
    big_data = study._boundary_data(big_op)
    big_flux = big_op.momentum_flux(
        study.manufactured(big.cell_centers), dirichlet_values=big_data
    )
    error = np.linalg.norm(
        big_op.op.integrate(big_flux) / big.cell_volumes[:, None]
        - study.divergence_of_stress(big.cell_centers, VISCOSITY),
        axis=1,
    )
    _figure._cloud = error
    _figure._n = 16
    figure = os.path.join(OUT, "native_viscous_order3d.png")
    _figure(truncation, faces, solutions, big, big_op, big_data, figure)

    payload = dict(
        operator="NativeViscous3D",
        viscosity=VISCOSITY,
        grids=list(GRIDS),
        warps=list(WARPS),
        analytic_check_finite_difference_relative_error=fd_rel,
        affine_traction_relative_error=affine_error,
        affine_cell_divergence=div_affine,
        truncation=truncation,
        face_traction=faces,
        solution=solutions,
        orders=summary,
        figure=os.path.basename(figure),
    )
    target = os.path.join(OUT, "native_viscous_order3d.json")
    json.dump(payload, open(target, "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print(
        "affine_traction_relative_error",
        affine_error,
        "analytic_fd_check",
        fd_rel,
        flush=True,
    )
    print("wrote", target, figure, flush=True)


if __name__ == "__main__":
    main()
