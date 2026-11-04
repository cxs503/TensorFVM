"""Qualification of the ``three_point`` Dirichlet wall closure.

The operator ``tensorfvm.native_viscous3d.NativeViscous3D`` offers two wall
closures.  ``two_point`` (default, historical) is first order in the wall
traction and leaves the wall-adjacent cell divergence O(1); ``three_point``
was documented as second order but had never been exercised by any test,
script or document -- the alignment search compared the candidate face
normal with the outward wall normal using the *same* sign, so it selected
nothing and the option silently degraded to ``two_point``.

This script measures both closures against analytic fields on the same
``n^3`` hex boxes used by :mod:`tensorfvm.verification.native_order3d`:

* ``rel_wall_normal_derivative`` -- max relative error of ``grad(u).n`` on
  Dirichlet faces, from the assembled face gradient (matrix + boundary data).
* ``l2_wall`` -- L2 error of the cell-integrated ``div(tau(u))`` on cells
  touching a box face, i.e. the quantity that stayed O(1) before.
* ``l2_interior`` -- the same on interior cells: the wall work must not
  disturb the interior operator.
* ``affine_flux`` -- max deviation from the exact affine traction, which has
  to stay at round-off for *both* closures and both mesh families.

Everything is written to ``runs/three_point_wall_closure.json`` together with
the observed orders; ``--plot`` renders ``runs/three_point_wall_closure.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.native_viscous3d import NativeViscous3D          # noqa: E402
from tensorfvm.verification import native_order3d as study      # noqa: E402

CLOSURES = ("two_point", "three_point")
VISCOSITY = 1.3
GRADIENT = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
BASE = np.array([1.2, .8, -.4])


def affine_flux_error(n, warp, closure):
    """Affine fields must give the exact traction: round-off only."""
    mesh = study.hex_box(n, warp)
    op = NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure=closure)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = BASE + mesh.face_centers[op.dirichlet] @ GRADIENT.T
    flux = op.momentum_flux(BASE + mesh.cell_centers @ GRADIENT.T, dirichlet_values=data)
    symmetric = GRADIENT + GRADIENT.T - 2/3*np.trace(GRADIENT)*np.eye(3)
    expected = VISCOSITY*mesh.area_vectors @ symmetric.T
    return float(np.abs(flux - expected).max())


def quadratic_normal_derivative(n, warp, closure):
    """``u_x = x^2``: analytic ``du/dn`` on every Dirichlet face is ``2 x n_x``."""
    mesh = study.hex_box(n, warp)
    op = NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure=closure)
    field = mesh.cell_centers[:, 0]**2
    datum = mesh.face_centers[:, 0]**2
    gradient = np.stack([op.face_gradient_matrices[a] @ field
                         + op.face_gradient_source(0.0, datum)[:, a] for a in range(3)], axis=1)
    numeric = np.einsum('fa,fa->f', gradient, op.normal)
    exact = 2*mesh.face_centers[:, 0]*op.normal[:, 0]
    error = np.abs(numeric - exact)[op.dirichlet]
    return float(error.max())


def run(levels, warps):
    records = []
    last = {}
    for closure in CLOSURES:
        for warp in warps:
            def factory(mesh, mu, closure=closure, last=last):
                operator = NativeViscous3D(mesh, mesh.neighbor < 0, mu, wall_closure=closure)
                last["op"] = operator
                return operator

            study._operator = factory
            for n in levels:
                started = time.time()
                truncation = study.truncation_study(n, warp, VISCOSITY)
                traction = study.face_traction_study(n, warp, VISCOSITY)
                record = dict(closure=closure, warp=float(warp), n=int(n),
                              cells=int(truncation["cells"]),
                              faces=int(truncation["faces"]),
                              wall_cells=int(truncation["wall_cells"]),
                              upgraded_faces=int(last["op"].wall_second_order_faces),
                              l2_wall=truncation["l2_wall"],
                              linf_wall=truncation["linf_wall"],
                              l2_interior=truncation["l2_interior"],
                              rel_wall_normal_derivative=traction["rel_wall_normal_derivative"],
                              rel_boundary_per_area=traction["rel_boundary_per_area"],
                              affine_flux=affine_flux_error(n, warp, closure),
                              quadratic_normal_derivative=quadratic_normal_derivative(n, warp, closure),
                              wall_seconds=time.time() - started)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    return records


def orders(records, key):
    """Observed orders ``log(e_i/e_{i+1}) / log(n_{i+1}/n_i)`` (no doubling assumed)."""
    table = {}
    for closure in CLOSURES:
        for warp in sorted({r["warp"] for r in records}):
            rows = sorted([r for r in records if r["closure"] == closure and r["warp"] == warp],
                          key=lambda r: r["n"])
            values = [row[key] for row in rows]
            table[f"{closure}|warp={warp:g}"] = [
                float(np.log(values[i]/values[i + 1])/np.log(rows[i + 1]["n"]/rows[i]["n"]))
                for i in range(len(values) - 1)]
    return table


def plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = (("rel_wall_normal_derivative", "rel. error of du/dn on Dirichlet faces"),
              ("l2_wall", "L2 error of cell-integrated div(tau) (wall cells)"),
              ("l2_interior", "L2 error of cell-integrated div(tau) (interior cells)"))
    style = {("two_point", 0.0): ("o", "-", "tab:red"), ("three_point", 0.0): ("s", "-", "tab:blue"),
             ("two_point", 2.0): ("o", "--", "tab:orange"), ("three_point", 2.0): ("s", "--", "tab:green")}
    for axis, (key, title) in zip(axes, panels):
        for closure in CLOSURES:
            for warp in sorted({r["warp"] for r in records}):
                rows = sorted([r for r in records if r["closure"] == closure and r["warp"] == warp],
                              key=lambda r: r["n"])
                marker, line, colour = style[(closure, warp)]
                axis.loglog([r["n"] for r in rows], [r[key] for r in rows], marker + line,
                            color=colour, label=f"{closure}, warp={warp:g}")
        axis.set_xlabel("cells per direction n")
        axis.set_title(title, fontsize=9)
        axis.grid(True, which="both", alpha=.3)
    axes[0].set_ylabel("error")
    axes[0].legend(fontsize=7)
    figure.suptitle("Dirichlet wall closure: two-point vs three-point (analytic fields, unit box)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--warps", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--output", default="runs/three_point_wall_closure.json")
    parser.add_argument("--plot", action="store_true")
    options = parser.parse_args()
    records = run(options.levels, options.warps)
    orders_table = {key: orders(records, key) for key in
                    ("rel_wall_normal_derivative", "l2_wall", "l2_interior", "rel_boundary_per_area")}
    payload = dict(levels=options.levels, warps=options.warps, viscosity=VISCOSITY,
                   closures=list(CLOSURES), records=records, orders=orders_table,
                   generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   script="scripts/verify_three_point_wall_closure.py")
    out = pathlib.Path(options.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    for key, table in orders_table.items():
        for name, values in table.items():
            print(f"order {key:28s} {name:24s} " + " ".join(f"{v:.3f}" for v in values), flush=True)
    if options.plot:
        plot(records, str(out.with_suffix(".png")))


if __name__ == "__main__":
    main()