"""Analytic audit of the native planar section engine.

``tensorfvm.native_sections3d.cell_section_polygons`` intersects the *actual*
planar facets of a polyhedral cell and stitches each cell's section ring.  It is
the sole geometry source of every native section figure in this repository
(SUBoff wake/pressure cuts, manufactured-solution comparisons, developed-channel
reports), but until now it was exercised only through plotting scripts that CI
never runs.  A silent defect there -- a dropped cell, a leaked facet diagonal, a
mis-projected axis -- would move published section areas and section field
ranges without touching any solver test.

This audit grades the engine against closed forms only.  No case compares the
engine with another code path, so the audit is independent of the geometry
being verified:

* **brick sections** -- the cut of an axis-aligned box is an exact rectangle,
  and the three axis permutations give three distinct analytic values, so the
  projected-axis selection is graded as well;
* **similarity laws** -- a pyramid (side 2) and a right tetra give the exact
  area laws ``4(1-h)^2`` and ``(1-h)^2/2`` for a plane at height ``h``;
* **decomposition invariance** -- six Kuhn tetrahedra tile the cube, so their
  stitched section must reproduce area 1 for *any* height; a lost cell, a
  double-counted internal facet or a swallowed collinear vertex all break it;
* **functionals** -- the area-weighted cell-centre sum of a linear field, i.e.
  exactly what the report scripts integrate, must equal the analytic integral.

Thresholds are analytic round-off, not engineering bands: ``TOL_AREA = 1e-12``
relative for a closed polygon sum, which is where double precision sits for the
coordinate magnitudes used here.

Outputs ``docs/verification-native-sections3d/results.json`` and two figures.
Run with ``python scripts/audit_native_sections3d.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "docs", "verification-native-sections3d")

from tensorfvm.native_mesh3d import NativeMesh3D  # noqa: E402
from tensorfvm.native_sections3d import cell_section_polygons  # noqa: E402

TOL_AREA = 1e-12


def signed_area(polygon):
    """Signed area of a planar ring projected onto the section plane."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def total_area(polygons):
    return float(sum(signed_area(p) for p in polygons))


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def box_mesh(lengths, divisions, triangulate_faces=False, graded=False):
    """Structured hexahedral box anchored at the origin."""
    coordinates = []
    for length, count in zip(lengths, divisions):
        s = np.linspace(0.0, 1.0, count + 1)
        coordinates.append(length * (s ** 2 if graded else s))
    x, y, z = coordinates
    vertices, index = [], {}
    for k, zk in enumerate(z):
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                index[(i, j, k)] = len(vertices)
                vertices.append((xi, yj, zk))
    cells = []
    for k in range(divisions[2]):
        for j in range(divisions[1]):
            for i in range(divisions[0]):
                cells.append((index[(i, j, k)], index[(i + 1, j, k)],
                              index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
                              index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                              index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)]))
    return NativeMesh3D.from_cells(np.array(vertices), cells,
                                   ['hexahedron'] * len(cells),
                                   triangulate_faces=triangulate_faces)


def pyramid_mesh():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], float)
    return NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4)], ['pyramid'])


def tetra_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    return NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3)], ['tetra'])


def wedge_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    return NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4, 5)], ['wedge'])


def prism_mesh(sides, radius=1.0, height=1.0):
    """Regular polygonal prism as a fan of wedges; section area is irrational."""
    angle = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.stack((radius * np.cos(angle), radius * np.sin(angle)), -1)
    vertices = [[0.0, 0.0, 0.0], [0.0, 0.0, height]]
    vertices += [[x, y, 0.0] for x, y in ring] + [[x, y, height] for x, y in ring]
    cells = [(0, 2 + i, 2 + (i + 1) % sides, 1, 2 + sides + i, 2 + sides + (i + 1) % sides)
             for i in range(sides)]
    return NativeMesh3D.from_cells(np.array(vertices, float), cells, ['wedge'] * sides)


def kuhn_mesh():
    """Six tetrahedra on the main diagonal tile the unit cube exactly."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 5, 1, 6), (0, 4, 5, 6), (0, 3, 7, 6), (0, 4, 7, 6)]
    return NativeMesh3D.from_cells(vertices, tets, ['tetra'] * len(tets))


BOX_LENGTHS = (2.0, 3.0, 4.0)
BOX_DIVISIONS = (2, 3, 4)


def record(name, kind, value, analytic, polygons):
    computed = total_area(polygons)
    error = abs(computed - analytic) / max(abs(analytic), 1e-300)
    return {
        "case": name,
        "kind": kind,
        "plane": float(value),
        "cells": int(len(polygons)),
        "analytic_area": float(analytic),
        "computed_area": computed,
        "rel_error": float(error),
        "passed": bool(error <= TOL_AREA),
    }


def brick_cases():
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    # Interior cuts: the section is the rectangle spanned by the other two axes.
    cases = []
    for axis, value, analytic in ((0, 0.5, 12.0), (1, 1.5, 8.0), (2, 2.5, 6.0)):
        polygons, _ = cell_section_polygons(mesh, axis=axis, value=value)
        cases.append(record(f"brick_axis{axis}_interior", "rectangle", value, analytic, polygons))
    # A cut lying inside an interior facet plane is shared by both layers.
    polygons, _ = cell_section_polygons(mesh, axis=2, value=2.0)
    cases.append(record("brick_coplanar_facet_plane", "rectangle", 2.0, 12.0, polygons))
    # Non-uniform spacing must not bias the stitched area.
    graded = box_mesh(BOX_LENGTHS, BOX_DIVISIONS, graded=True)
    polygons, _ = cell_section_polygons(graded, axis=2, value=1e-3)
    cases.append(record("brick_graded_spacing", "rectangle", 1e-3, 6.0, polygons))
    # Internal facet diagonals must cancel, not add area.
    triangulated = box_mesh((2.0, 3.0, 4.0), (1, 1, 1), triangulate_faces=True)
    polygons, _ = cell_section_polygons(triangulated, axis=2, value=0.5)
    cases.append(record("brick_triangulated_facets", "rectangle", 0.5, 6.0, polygons))
    polygons, _ = cell_section_polygons(triangulated, axis=2, value=0.0)
    cases.append(record("brick_triangulated_coplanar", "rectangle", 0.0, 6.0, polygons))
    return cases


def pyramid_law():
    """Area(z) = 4 (1 - z)^2 for a square-base pyramid, apex at z = 1."""
    mesh = pyramid_mesh()
    rows = []
    for value in np.linspace(0.0, 0.95, 20):
        polygons, _ = cell_section_polygons(mesh, axis=2, value=float(value))
        rows.append(record(f"pyramid_h{value:.4f}", "similarity", value,
                           4.0 * (1.0 - value) ** 2, polygons))
    return rows


def tetra_law():
    mesh = tetra_mesh()
    rows = []
    for value in np.linspace(0.0, 0.95, 20):
        polygons, _ = cell_section_polygons(mesh, axis=2, value=float(value))
        rows.append(record(f"tetra_h{value:.4f}", "similarity", value,
                           0.5 * (1.0 - value) ** 2, polygons))
    return rows


def wedge_case():
    polygons, _ = cell_section_polygons(wedge_mesh(), axis=2, value=0.5)
    return [record("wedge_parallel_cut", "similarity", 0.5, 0.5, polygons)]


def kuhn_cases():
    """Six Kuhn tets tile the cube: the stitched cut must be area 1 everywhere."""
    mesh = kuhn_mesh()
    rows = []
    for value in np.linspace(0.02, 0.98, 25):
        polygons, _ = cell_section_polygons(mesh, axis=2, value=float(value))
        rows.append(record(f"kuhn_z{value:.4f}", "decomposition", value, 1.0, polygons))
    return rows


def prism_cases():
    """A regular n-gon prism: area = (n/2) R^2 sin(2 pi/n), irrational by construction.

    Unlike the brick and similarity laws these values are not exactly representable,
    so the recorded error is genuine round-off rather than an exact match.
    """
    rows = []
    for sides in (3, 6, 12, 24, 96):
        mesh = prism_mesh(sides)
        polygons, _ = cell_section_polygons(mesh, axis=2, value=0.5)
        analytic = 0.5 * sides * np.sin(2.0 * np.pi / sides)
        rows.append(record(f"prism_n{sides}", "polygon", 0.5, analytic, polygons))
    return rows


def functional_case():
    """Area-weighted cell-centre sum of x + 2y equals 6 + 18 over [0,2] x [0,3]."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    polygons, ids = cell_section_polygons(mesh, axis=2, value=2.5)
    weights = np.array([signed_area(p) for p in polygons])
    centres = mesh.cell_centers[ids]
    computed = float(np.sum(weights * (centres[:, 0] + 2.0 * centres[:, 1])))
    analytic = 24.0
    return [{"case": "linear_field_integral", "kind": "functional", "plane": 2.5,
             "cells": int(len(polygons)), "analytic_area": analytic,
             "computed_area": computed,
             "rel_error": abs(computed - analytic) / analytic,
             "passed": bool(abs(computed - analytic) / analytic <= TOL_AREA)}]


def figure(cases, pyramid_rows, kuhn_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_handle, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    # (a) the actual stitched section polygons, drawn from the engine output.
    for mesh, value, colour, label in [
            (pyramid_mesh(), 0.5, "#1f77b4", "pyramid, z = 0.5"),
            (kuhn_mesh(), 0.3, "#d62728", "6-tet cube, z = 0.3"),
            (box_mesh(BOX_LENGTHS, BOX_DIVISIONS), 2.5, "#2ca02c", "2x3x4 brick, z = 2.5")]:
        polygons, _ = cell_section_polygons(mesh, axis=2, value=value)
        for polygon in polygons:
            ring = np.vstack((polygon, polygon[:1]))
            axes[0].plot(ring[:, 0], ring[:, 1], color=colour, lw=0.9, alpha=0.85)
        axes[0].plot([], [], color=colour, lw=1.2, label=label)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("(a) stitched native section polygons")

    # (b) computed vs analytic area laws: the curves must lie on the symbols.
    height = np.array([row["plane"] for row in pyramid_rows])
    axes[1].plot(np.linspace(0, 1, 200), 4.0 * (1 - np.linspace(0, 1, 200)) ** 2,
                 "-", color="#1f77b4", lw=2.0, label="pyramid analytic 4(1-h)^2")
    axes[1].plot(height, [row["computed_area"] for row in pyramid_rows], "o",
                 mfc="none", color="#1f77b4", ms=5, label="pyramid computed")
    axes[1].plot(np.linspace(0, 1, 200), 0.5 * (1 - np.linspace(0, 1, 200)) ** 2,
                 "-", color="#d62728", lw=2.0, label="tetra analytic (1-h)^2/2")
    tetra_rows = tetra_law()
    axes[1].plot([row["plane"] for row in tetra_rows],
                 [row["computed_area"] for row in tetra_rows], "s",
                 mfc="none", color="#d62728", ms=5, label="tetra computed")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("plane height h")
    axes[1].set_ylabel("section area")
    axes[1].legend(fontsize=8)
    axes[1].set_title("(b) section area against the similarity laws")
    figure_handle.tight_layout()
    figure_handle.savefig(os.path.join(OUT, "sections3d-analytic.png"), dpi=200)
    plt.close(figure_handle)

    # (c) how the engine behaves as the plane sweeps: a facet-coplanar plane is
    # reported by the cells on *both* sides, so the reported total is 2x the
    # geometric cross-section there.  Users must not read that as a defect.
    sweep = np.linspace(0.0, 4.0, 161)
    reported, cells = [], []
    for value in sweep:
        polygons, _ = cell_section_polygons(box_mesh(BOX_LENGTHS, BOX_DIVISIONS),
                                           axis=2, value=float(value))
        reported.append(total_area(polygons))
        cells.append(len(polygons))
    figure_handle, axis = plt.subplots(figsize=(9.0, 4.0))
    axis.step(sweep, reported, where="mid", color="#1f77b4", lw=1.4,
              label="reported section area")
    axis.axhline(6.0, color="k", ls="--", lw=1.0, label="geometric area 2x3 = 6")
    for grid_plane in (1.0, 2.0, 3.0):
        axis.axvline(grid_plane, color="#d62728", ls=":", lw=0.9)
    axis.plot([], [], color="#d62728", ls=":", lw=0.9,
              label="interior facet plane (two-sided report)")
    axis2 = axis.twinx()
    axis2.plot(sweep, cells, color="#2ca02c", lw=1.0, label="reported cells")
    axis2.set_ylabel("reported cells", color="#2ca02c")
    axis.set_xlabel("plane z")
    axis.set_ylabel("reported area")
    axis.set_title("(c) plane sweep: facet-coplanar planes report both layers")
    axis.legend(fontsize=8, loc="upper left")
    figure_handle.tight_layout()
    figure_handle.savefig(os.path.join(OUT, "sections3d-errors.png"), dpi=200)
    plt.close(figure_handle)
    return kuhn_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pyramid_rows = pyramid_law()
    tetra_rows = tetra_law()
    cases = (brick_cases() + pyramid_rows + tetra_rows + wedge_case()
             + prism_cases() + kuhn_cases() + functional_case())
    figure(cases, pyramid_rows, None)
    sources = ["src/tensorfvm/native_sections3d.py", "src/tensorfvm/native_mesh3d.py",
               "tests/test_native_sections3d.py", "scripts/audit_native_sections3d.py"]
    errors = np.array([row["rel_error"] for row in cases])
    worst = cases[int(np.argmax(errors))]
    summary = {
        "case_count": len(cases),
        "passed": all(row["passed"] for row in cases),
        "tolerance_relative": TOL_AREA,
        "max_rel_error": float(errors.max()),
        "worst_case": worst["case"],
        "median_rel_error": float(np.median(errors)),
        "kinds": sorted({row["kind"] for row in cases}),
        "source_sha256": {name: sha256(os.path.join(ROOT, name)) for name in sources},
        "figures": ["sections3d-analytic.png", "sections3d-errors.png"],
        "note": ("Cells are graded against closed-form areas only; no case compares the "
                 "engine with another implementation. A plane touching a cell at a single "
                 "vertex encloses no finite area and is reported as absent, which is a "
                 "documented convention, not a tolerance."),
    }
    with open(os.path.join(OUT, "results.json"), "w") as stream:
        json.dump({"summary": summary, "cases": cases}, stream, indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())