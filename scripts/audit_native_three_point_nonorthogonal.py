"""Audit: is ``wall_closure='three_point'`` affine exact away from orthogonal meshes?

``docs/native-viscous-three-point-closure/report.md`` measured the option on
orthogonal (and sinusoidally warped) cube meshes and stated the closure is
"exact for affine fields".  This driver re-tests that claim on a mesh family
the earlier study did not cover -- meshes whose *cells* are laterally
displaced (interior warp), i.e. non-orthogonal cell lines -- and attributes the
measured residual to a specific geometric quantity, so that the claim is
either confirmed or bounded with numbers.

Everything below is analytic: the manufactured field is affine (closed-form
flux) or a trigonometric field (closed-form gradient), no reference data and no
fitted parameter are involved.  Outputs:

* ``docs/native-viscous-three-point-closure/nonorthogonal-audit.json``
* ``docs/native-viscous-three-point-closure/nonorthogonal-audit.png``

Run::

    python3 scripts/audit_native_three_point_nonorthogonal.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.native_viscous3d import NativeViscous3D          # noqa: E402
from tensorfvm.verification import native_viscous_order3d as nvw  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs",
                   "native-viscous-three-point-closure")
CLOSURES = ("two_point", "three_point")
WARPS = (0.0, 0.15, 0.3)                 # peak interior vertex shift / h
DIVISIONS = ((4, 4, 4), (6, 6, 6))
GRADIENT = np.array([0.7, -0.3, 0.2])
MU = 1.3


def _operator(divisions, warp, closure):
    mesh = nvw.cube_mesh(divisions, warp=warp)
    boundary = mesh.neighbor < 0
    return mesh, boundary, NativeViscous3D(mesh, boundary, MU, wall_closure=closure)


def affine_study(divisions, warp, closure):
    """Relative Dirichlet-flux error on an affine field (machine zero expected)."""
    mesh, boundary, op = _operator(divisions, warp, closure)
    cell = 1.2 + mesh.cell_centers @ GRADIENT
    face = 1.2 + mesh.face_centers @ GRADIENT
    exact = MU * (mesh.area_vectors @ GRADIENT)
    residual = op.scalar_flux(cell, 0.0, face)[boundary] - exact[boundary]
    scale = np.linalg.norm(exact[boundary])
    scaled = residual / (MU * op.area[boundary])          # du/dn units
    # predicted residual of the three-point stencil, from the stencil geometry
    coeff = op.wall_coeff[boundary]
    offset = op.wall_second_offset[boundary]
    predicted = coeff[:, 2] * (offset @ GRADIENT)
    live = np.abs(predicted).max() > 0.0
    return dict(
        divisions=list(divisions), warp=warp, closure=closure, h=1.0 / max(divisions),
        affine_error=float(np.abs(residual).max() / scale),
        affine_rms=float(np.linalg.norm(residual) / scale),
        residual_max_du_dn=float(np.abs(scaled).max()),
        predicted_max_du_dn=float(np.abs(predicted).max()),
        predicted_correlation=float(np.corrcoef(scaled, predicted)[0, 1]) if live else None,
        second_cell_offset_max=float(
            np.abs(offset - np.einsum("ij,ij->i", offset, op.normal[boundary])[:, None]
                   * op.normal[boundary]).max()),
        upgraded_faces=op.wall_second_order_faces,
        dirichlet_faces=int(boundary.sum()),
    )


def wall_derivative_study(divisions, warp, closure, mode="trig3d"):
    """Relative L2 error of the Dirichlet-face normal derivative (analytic)."""
    mesh, boundary, op = _operator(divisions, warp, closure)
    cell, _ = nvw.scalar_field(mesh.cell_centers, mode)
    face, gradient = nvw.scalar_field(mesh.face_centers, mode)
    computed = op.scalar_flux(cell, 0.0, face)[boundary] / (MU * op.area[boundary])
    exact = np.einsum("ij,ij->i", gradient[boundary], op.normal[boundary])
    return dict(divisions=list(divisions), warp=warp, closure=closure,
                mode=mode, h=1.0 / max(divisions),
                wall_derivative=float(np.linalg.norm(computed - exact)
                                      / np.linalg.norm(exact)))


def gradient_estimate_defect(divisions=(4, 4, 4), warp=0.3):
    """Why a matrix-only second-cell correction cannot work.

    The assembled two-point face-gradient rows are not self-contained on a
    Dirichlet face: the prescribed datum enters through the flux *source* with
    the opposite ``owner_response`` weight.  A correction built from the matrix
    part alone therefore sees ``t . grad u`` with an O(1) spurious offset.
    """
    mesh = nvw.cube_mesh(divisions, warp=warp)
    mask = mesh.neighbor < 0
    boundary = np.where(mask)[0]
    plain = NativeViscous3D(mesh, mask, MU, wall_closure="two_point")
    three = NativeViscous3D(mesh, mask, MU, wall_closure="three_point")
    cell = 1.2 + mesh.cell_centers @ GRADIENT
    offset = three.wall_offset
    directional = np.zeros(len(offset))
    for a in range(3):
        directional += offset[:, a] * (plain.face_gradient_matrices[a] @ cell)
    target = offset @ GRADIENT
    defect = float(np.abs(directional[boundary] - target[boundary]).max())
    response = offset[boundary] * plain.owner_response[boundary]
    return dict(divisions=list(divisions), warp=warp, defect=defect,
                target_scale=float(np.abs(target[boundary]).max()),
                response_scale=float(np.abs(response.sum(axis=1)).max()))


def selection_study(divisions=(4, 4, 4), warp=0.3):
    """Is the chosen second cell the best available one?

    The choice currently maximises the face-normal alignment.  Score every
    admissible candidate by the predicted stencil truncation
    ``|coeff2| |t_cell|`` and compare with the chosen one.
    """
    mesh = nvw.cube_mesh(divisions, warp=warp)
    boundary = mesh.neighbor < 0
    internal = mesh.neighbor >= 0
    o, n = mesh.owner, mesh.neighbor
    fc, cc = mesh.face_centers, mesh.cell_centers
    nrm = mesh.area_vectors / np.linalg.norm(mesh.area_vectors, axis=1)[:, None]
    cand = {}
    for gf in np.where(internal)[0]:
        cand.setdefault(o[gf], []).append((gf, 1))
        cand.setdefault(n[gf], []).append((gf, -1))
    chosen, best = [], []
    for face in np.where(boundary)[0]:
        c0, nf = o[face], nrm[face]
        delta0 = (fc[face] - cc[c0]) @ nf
        rows = []
        for gf, sig in cand.get(c0, ()):
            align = sig * (nrm[gf] @ -nf)
            c1 = n[gf] if o[gf] == c0 else o[gf]
            if c1 < 0 or align < 0.5:
                continue
            delta1 = (fc[face] - cc[c1]) @ nf
            if delta0 <= 0 or delta1 <= delta0 * (1 + 1e-9):
                continue
            coeff2 = delta0 / (delta1 * (delta1 - delta0))
            t = (cc[c1] - cc[c0]) - ((cc[c1] - cc[c0]) @ nf) * nf
            rows.append((abs(coeff2) * np.linalg.norm(t), align))
        if rows:
            chosen.append(max(rows, key=lambda r: r[1])[0])
            best.append(min(r[0] for r in rows))
    return dict(divisions=list(divisions), warp=warp, faces=len(chosen),
                truncation_chosen_max=float(max(chosen)),
                truncation_best_max=float(max(best)),
                truncation_chosen_mean=float(np.mean(chosen)),
                truncation_best_mean=float(np.mean(best)))


def observed_orders(rows, key):
    return [float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
            for a, b in zip(rows, rows[1:])]


def figure(affine, wall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    styles = {"two_point": ("o", "-"), "three_point": ("s", "--")}
    for closure in CLOSURES:
        marker, line = styles[closure]
        for warp, colour in zip(WARPS, ("#1f77b4", "#ff7f0e", "#2ca02c")):
            rows = [r for r in affine if r["closure"] == closure and r["warp"] == warp]
            axes[0].loglog([r["h"] for r in rows], [r["affine_error"] for r in rows],
                           marker=marker, ls=line, color=colour,
                           label=f"{closure}, warp={warp}")
            rows = [r for r in wall if r["closure"] == closure and r["warp"] == warp]
            axes[1].loglog([r["h"] for r in rows], [r["wall_derivative"] for r in rows],
                           marker=marker, ls=line, color=colour,
                           label=f"{closure}, warp={warp}")
    axes[0].axhline(1e-12, color="k", lw=0.6, ls=":")
    axes[0].set_xlabel("h = 1/n")
    axes[0].set_ylabel("relative affine flux error")
    axes[0].set_title("affine exactness (machine zero expected)")
    axes[1].set_xlabel("h = 1/n")
    axes[1].set_ylabel(r"relative $L_2$ error of $\partial u/\partial n$")
    axes[1].set_title("analytic trigonometric field")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.suptitle("three_point wall closure: orthogonal vs warped (non-orthogonal) meshes",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "nonorthogonal-audit.png")
    fig.savefig(path, dpi=140)
    return path


def main():
    affine, wall = [], []
    for warp in WARPS:
        for divisions in DIVISIONS:
            for closure in CLOSURES:
                affine.append(affine_study(divisions, warp, closure))
                wall.append(wall_derivative_study(divisions, warp, closure))
    payload = dict(
        comment="three_point wall closure on warped (non-orthogonal) cube meshes",
        viscosity=MU, gradient=list(GRADIENT), walls=WARPS,
        divisions=[list(d) for d in DIVISIONS],
        affine=affine, wall_derivative=wall,
        orders={f"{r['closure']}|warp={r['warp']}": observed_orders(
            [x for x in affine if x["closure"] == r["closure"] and x["warp"] == r["warp"]],
            "affine_error") for r in affine if r["divisions"] == [4, 4, 4]},
        order_wall={f"{r['closure']}|warp={r['warp']}": observed_orders(
            [x for x in wall if x["closure"] == r["closure"] and x["warp"] == r["warp"]],
            "wall_derivative") for r in wall if r["divisions"] == [4, 4, 4]},
        gradient_estimate_defect=gradient_estimate_defect(),
        selection=selection_study(),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nonorthogonal-audit.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    png = figure(affine, wall)
    for row in affine:
        print(f"affine  warp={row['warp']:<5} n={row['divisions'][0]:<2} "
              f"{row['closure']:<11} err={row['affine_error']:.3e} "
              f"pred_max={row['predicted_max_du_dn']:.3e} "
              f"corr={row['predicted_correlation']} "
              f"up={row['upgraded_faces']}/{row['dirichlet_faces']}")
    for row in wall:
        print(f"du/dn   warp={row['warp']:<5} n={row['divisions'][0]:<2} "
              f"{row['closure']:<11} err={row['wall_derivative']:.4e}")
    print("orders(affine)", json.dumps(payload["orders"]))
    print("orders(du/dn)", json.dumps(payload["order_wall"]))
    print("gradient_estimate_defect", payload["gradient_estimate_defect"])
    print("selection", payload["selection"])
    print("wrote", png)


if __name__ == "__main__":
    main()