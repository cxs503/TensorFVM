"""Independent verification of the ``wall_closure='three_point'`` option of
:class:`tensorfvm.native_viscous3d.NativeViscous3D`.

The option was merged without any test or measurement.  This driver answers
three questions with numbers only:

1. does the option engage at all (how many Dirichlet faces get the one-sided
   three-point stencil, versus the count of Dirichlet faces)?
2. is it exact on affine fields (the property the docstring claims)?
3. does it improve the quantified errors (Dirichlet wall ``du/dn``, and the
   Dirichlet solution error on a field that *is* an exact solution) and at what
   observed order, relative to the historical ``'two_point'`` closure?

Everything is written to ``docs/native-viscous-three-point-closure/``
(``results.json`` + two PNGs).  Pass ``--publish-path`` to also emit a platform
report payload with every number injected programmatically.

Run::

    python3 scripts/verify_native_viscous_three_point_closure.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.native_viscous3d import NativeViscous3D  # noqa: E402
from tensorfvm.verification import native_order3d as nv  # noqa: E402
from tensorfvm.verification import native_viscous_order3d as nvw  # noqa: E402

CLOSURES = ("two_point", "three_point")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs",
                       "native-viscous-three-point-closure")


def affine_exactness(divisions, closure, viscosity=1.3):
    """Relative Dirichlet-flux error on an affine field (machine zero expected)."""
    mesh = nvw.cube_mesh(divisions)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, viscosity, wall_closure=closure)
    gradient = np.array([0.7, -0.3, 0.2])
    base = 1.2
    cell_values = base + mesh.cell_centers @ gradient
    boundary_values = base + mesh.face_centers @ gradient
    exact = viscosity * (mesh.area_vectors @ gradient)
    computed = op.scalar_flux(cell_values, 0.0, boundary_values)
    error = float(np.abs(computed[boundary] - exact[boundary]).max()
                  / np.linalg.norm(exact[boundary]))
    return dict(divisions=list(divisions), closure=closure, affine_error=error,
                upgraded_faces=op.wall_second_order_faces,
                dirichlet_faces=int(boundary.sum()),
                h=1.0 / max(divisions))


def wall_flux(divisions, closure, mode, viscosity=1.3):
    """Relative L2 error of the Dirichlet-face normal derivative."""
    mesh = nvw.cube_mesh(divisions)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, viscosity, wall_closure=closure)
    cell_values, _ = nvw.scalar_field(mesh.cell_centers, mode)
    boundary_values, boundary_gradient = nvw.scalar_field(mesh.face_centers, mode)
    computed = op.scalar_flux(cell_values, 0.0, boundary_values) / (viscosity * op.area)
    exact = np.einsum("ij,ij->i", boundary_gradient, op.normal)
    return dict(divisions=list(divisions), closure=closure, mode=mode,
                wall_flux=float(np.linalg.norm(computed[boundary] - exact[boundary])
                                / np.linalg.norm(exact[boundary])),
                upgraded_faces=op.wall_second_order_faces,
                dirichlet_faces=int(boundary.sum()),
                h=1.0 / max(divisions))


def harmonic_solution(n, closure, viscosity=1.3, warp=0.0):
    """Dirichlet solve against a field that satisfies ``div(tau)=0`` exactly."""
    from scipy.sparse.linalg import spsolve

    mesh = nv.hex_box(n, warp)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, viscosity, wall_closure=closure)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = nv.harmonic_velocity(mesh.face_centers[op.dirichlet])
    rhs = -(op.vector_divergence_matrix
            @ op.momentum_flux_source(dirichlet_values=data).ravel())
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    values = solved.reshape(len(mesh.cells), 3)
    error = np.linalg.norm(values - nv.harmonic_velocity(mesh.cell_centers), axis=1)
    wall = nv._wall_mask(mesh, n)
    return dict(n=n, h=1.0 / n, closure=closure, warp=warp,
                l2_global=float(np.sqrt(np.mean(error ** 2))),
                l2_interior=float(np.sqrt(np.mean(error[~wall] ** 2))),
                l2_wall=float(np.sqrt(np.mean(error[wall] ** 2))),
                linf_global=float(error.max()),
                residual=float(np.linalg.norm(op.momentum_diffusion_matrix @ solved - rhs)
                               / max(np.linalg.norm(rhs), 1e-30)),
                upgraded_faces=op.wall_second_order_faces,
                dirichlet_faces=int(boundary.sum()),
                nonzeros=int(op.momentum_diffusion_matrix.nnz))


def observed_orders(rows, key):
    return [float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
            for a, b in zip(rows, rows[1:])]


TRIG1D = [(8, 4, 4), (16, 4, 4), (32, 4, 4)]
TRIG3D = [(4, 4, 4), (6, 6, 6), (8, 8, 8)]
HARMONIC = (4, 6, 8)


def run_study():
    started = time.time()
    results = dict(
        closures=list(CLOSURES),
        affine=[affine_exactness((6, 6, 6), c) for c in CLOSURES],
        trig1d={c: [wall_flux(d, c, "trig1d") for d in TRIG1D] for c in CLOSURES},
        trig3d={c: [wall_flux(d, c, "trig3d") for d in TRIG3D] for c in CLOSURES},
        harmonic={c: [harmonic_solution(n, c) for n in HARMONIC] for c in CLOSURES},
    )
    results["orders"] = {
        mode: {c: observed_orders(results[mode][c], "wall_flux") for c in CLOSURES}
        for mode in ("trig1d", "trig3d")
    }
    results["orders"]["harmonic_l2_global"] = {
        c: observed_orders(results["harmonic"][c], "l2_global") for c in CLOSURES}
    results["orders"]["harmonic_l2_wall"] = {
        c: observed_orders(results["harmonic"][c], "l2_wall") for c in CLOSURES}
    results["orders"]["harmonic_l2_interior"] = {
        c: observed_orders(results["harmonic"][c], "l2_interior") for c in CLOSURES}
    results["seconds"] = round(time.time() - started, 1)
    return results


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for closure, marker in zip(CLOSURES, ("o", "s")):
        rows = results["trig3d"][closure]
        ax.loglog([r["h"] for r in rows], [r["wall_flux"] for r in rows],
                  marker + "-", label=f"{closure} (order {', '.join('%.2f' % o for o in results['orders']['trig3d'][closure])})")
    ax.set_xlabel("h")
    ax.set_ylabel("relative L2 error of wall du/dn")
    ax.set_title("3-D trig field: Dirichlet wall normal derivative")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for closure, marker in zip(CLOSURES, ("o", "s")):
        rows = results["trig1d"][closure]
        ax.loglog([r["h"] for r in rows], [r["wall_flux"] for r in rows],
                  marker + "-", label=f"{closure} (order {', '.join('%.2f' % o for o in results['orders']['trig1d'][closure])})")
    ax.set_xlabel("h (x direction)")
    ax.set_ylabel("relative L2 error of wall du/dn")
    ax.set_title("1-D control sin(pi x): closure isolated in x")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    for closure, marker in zip(CLOSURES, ("o", "s")):
        rows = results["harmonic"][closure]
        ax.loglog([r["h"] for r in rows], [r["l2_global"] for r in rows],
                  marker + "-", label=f"{closure} (order {', '.join('%.2f' % o for o in results['orders']['harmonic_l2_global'][closure])})")
    ax.set_xlabel("h")
    ax.set_ylabel("L2 error of the Dirichlet solution")
    ax.set_title("harmonic field (exact solution): solution error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "three-point-closure.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    with open(path, "rb") as handle:
        figures.append(dict(name="three-point-closure.png",
                            caption="Left: wall du/dn error on the 3-D trig field; middle: "
                                    "the 1-D control isolating the x closure; right: Dirichlet "
                                    "solution error on the harmonic field. markers = observed "
                                    "convergence order.",
                            png_base64=base64.b64encode(handle.read()).decode()))

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r["closure"] for r in results["affine"]]
    values = [max(r["affine_error"], 1e-17) for r in results["affine"]]
    ax.bar(names, values)
    ax.set_yscale("log")
    ax.set_ylabel("relative Dirichlet-flux error on an affine field")
    ax.set_title("affine exactness (machine zero = correct stencil signs)")
    for index, value in enumerate(values):
        ax.text(index, value * 1.6, "%.1e" % value, ha="center", fontsize=8)
    ax.set_ylim(1e-18, 1e-8)
    path = os.path.join(OUT_DIR, "affine-exactness.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    with open(path, "rb") as handle:
        figures.append(dict(name="affine-exactness.png",
                            caption="Affine-field exactness per closure; a sign error in the "
                                    "one-sided stencil would show up here as O(1e-2).",
                            png_base64=base64.b64encode(handle.read()).decode()))
    return figures


def markdown_body(results):
    lines = []
    lines.append("| closure | affine error | upgraded faces / Dirichlet faces |")
    lines.append("|---|---|---|")
    for row in results["affine"]:
        lines.append("| %s | %.2e | %d / %d |" % (row["closure"], row["affine_error"],
                                                  row["upgraded_faces"], row["dirichlet_faces"]))
    lines.append("")
    lines.append("3-D trig field, wall $\\partial u/\\partial n$:")
    lines.append("")
    lines.append("| closure | " + " | ".join("h=1/%d" % max(r["divisions"]) for r in results["trig3d"]["two_point"]) + " | orders |")
    lines.append("|---" * (len(results["trig3d"]["two_point"]) + 2) + "|")
    for closure in CLOSURES:
        rows = results["trig3d"][closure]
        lines.append("| %s | %s | %s |" % (closure,
                                           " | ".join("%.4e" % r["wall_flux"] for r in rows),
                                           " / ".join("%.2f" % o for o in results["orders"]["trig3d"][closure])))
    lines.append("")
    lines.append("Harmonic field, Dirichlet solution error:")
    lines.append("")
    lines.append("| closure | " + " | ".join("n=%d" % r["n"] for r in results["harmonic"]["two_point"]) + " | orders |")
    lines.append("|---" * (len(results["harmonic"]["two_point"]) + 2) + "|")
    for closure in CLOSURES:
        rows = results["harmonic"][closure]
        lines.append("| %s | %s | %s |" % (closure,
                                           " | ".join("%.5f" % r["l2_global"] for r in rows),
                                           " / ".join("%.2f" % o for o in results["orders"]["harmonic_l2_global"][closure])))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-path")
    args = parser.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = run_study()
    results["markdown"] = markdown_body(results)
    figures = make_figures(results)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    if args.publish_path:
        payload = dict(
            title="TensorFVM · wall_closure='three_point' 复核：合并即死代码（0 面启用）+ COO 长度崩溃 + 仿射不精确，修复后量化增益与残余",
            summary=("对已合并但零测试的 NativeViscous3D wall_closure='three_point' 做独立判定："
                     "原码在正交/贴体网格上启用 0 个 Dirichlet 面（法向约定反号），强制启用后 "
                     "COO 长度不匹配崩溃，修复后又不满足 docstring 声称的仿射精确（横向质心偏移）。"
                     "本单给出最小修复与三族网格量化复核。"),
            scenario_key="tensorfvm-verification",
            mission_id=3244,
            tags=["TensorFVM", "verification-debt", "wall-closure", "second-order", "affine-exactness"],
            markdown=results["markdown"],
            figures=figures,
        )
        with open(args.publish_path, "w") as handle:
            json.dump(payload, handle)
    print(results["markdown"])
    print("seconds", results["seconds"])


if __name__ == "__main__":
    main()