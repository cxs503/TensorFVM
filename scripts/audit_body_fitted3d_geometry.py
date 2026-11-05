"""Exact geometric qualification of the extruded body-fitted 3-D cylinder O-grid.

Verification debt addressed
---------------------------
``tensorfvm/body_fitted3d.py`` is a merged production mesh backend, but its only
geometric check (``tests/test_body_fitted3d.py``) accepts
``|A_h - 2 pi R L| / (2 pi R L) < 1e-2`` on one coarse mesh.  A 1 % tolerance on
a single grid cannot tell a correct chord quadrature from a biased one, and it
never pins the stored volumes/areas to the actual vertex array.  This audit
replaces that with exact recomputations from the raw ``vertices`` array (no
helper of ``body_fitted3d`` is used as truth) plus negative controls:

  G1  stored wetted area == sum of chord lengths x span recomputed from vertices
      (round-off), and every cylinder-arc vertex has ``|r - c| = R`` exactly,
      where ``c`` is the circum-centre recovered from the arc vertices;
  G2  the arc chord sum equals the *inscribed regular polygon* value
      ``2 n_a R sin(pi/n_a) L`` -- the exact discrete area of the polygon the
      O-grid builds, i.e. the 2 pi R L deficit is a known quadrature term, not an
      unknown bias;
  G3  the stored cross-section area equals an independent Green/shoelace
      recomputation of the outer and inner vertex rings, and
      ``sum(cell_volumes) = A_cross * L`` to round-off;
  G4  the deficit against the smooth cylinder decays at the inscribed-polygon
      rate, i.e. observed order 2 in ``1/n_theta`` (measured, not assumed);
  G5  negative controls: a 1 % radius inflation and a 1 % radial shift of one
      wall vertex must both be detected, otherwise the metric measures nothing.

Structural observation (not an error, but not documented in the repo either):
the angular sweep is *not* a full circle -- at ``nx=32`` it holds 24 genuine arc
faces plus 8 near-zero-length wake-plate faces.  ``analytic_cylinder_surface_area``
therefore compares ``2 pi R L`` against ``arc + wake`` wall area; the wake faces
are ~1e-2 of the arc faces, so the comparison is dominated by the arc, but the
identity that actually holds is the per-arc chord sum, which is what G1/G2 pin.

Everything below is mesh data only, single-threaded, sub-second.
Run: ``PYTHONPATH=src python scripts/audit_body_fitted3d_geometry.py``
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensorfvm.body_fitted3d import BodyFittedCylinderMesh3D
from tensorfvm.solver3d import Cylinder3DConfig

STRETCHING = 2.5


def _mesh(nx: int, ny: int, nz: int):
    config = Cylinder3DConfig(nx=nx, ny=ny, nz=nz, mesh_type="body-fitted",
                              body_fitted_stretching=STRETCHING)
    return config, BodyFittedCylinderMesh3D.build(config, torch.device("cpu"))


def _shoelace(ring: np.ndarray) -> float:
    """Absolute area of the closed polygon through ``ring`` (last point repeated)."""
    x, y = ring[:-1, 0], ring[:-1, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _wall_rings(mesh):
    """Cross-section rings at z=0: (inner/wall ring, outer ring) from raw vertices."""
    v = mesh.vertices.detach().cpu().numpy()
    return v, v[0, 0, :, :2], v[0, -1, :, :2]


def _arc_split(wall_ring: np.ndarray):
    """Split the wall ring into genuine arc chords and wake-plate chords.

    The circum-centre is *recovered from the arc vertices themselves* (mean of the
    arc chord endpoints), so nothing about the mesh's own centre or radius helper
    enters the identity.
    """
    chords = np.linalg.norm(np.diff(wall_ring, axis=0), axis=1)
    arc = chords > 0.5 * chords.max()
    endpoints = wall_ring[:-1][arc]                      # genuine arc vertices
    # algebraic (Kasa) circle fit: centre from the arc vertices only
    A = np.column_stack([endpoints[:, 0], endpoints[:, 1], np.ones(len(endpoints))])
    b = -(endpoints ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = np.array([-0.5 * sol[0], -0.5 * sol[1]])
    radii = np.linalg.norm(wall_ring - centre, axis=1)
    return chords, arc, centre, radii


def geometry_report(nx: int = 32, ny: int = 8, nz: int = 6) -> dict:
    _, mesh = _mesh(nx, ny, nz)
    v, wall_ring, outer_ring = _wall_rings(mesh)
    span = float(mesh.local_span)
    radius = float(mesh.cylinder_radius)
    chords, arc, centre, radii = _arc_split(wall_ring)
    n_a = int(arc.sum())

    wall_area_vertices = float(chords.sum() * span)
    arc_area_vertices = float(chords[arc].sum() * span)
    arc_area_inscribed = float(2.0 * n_a * radius * math.sin(math.pi / n_a) * span)
    analytic = float(2.0 * math.pi * radius * span)
    stored = float(mesh.cylinder_surface_area)

    fluid_area = _shoelace(outer_ring) - _shoelace(wall_ring)
    stored_area = float(mesh.cross_section_area)
    volume_area = float(mesh.cell_volumes.sum()) / span
    arc_radii = radii[:-1][arc]

    return {
        "nx": nx, "ny": ny, "nz": nz, "cells": int(mesh.cell_count),
        "span_m": span, "radius_m": radius,
        "cylinder_centre_recovered": [float(centre[0]), float(centre[1])],
        "arc_faces": n_a, "wake_faces": int((~arc).sum()),
        "wake_to_arc_area_ratio": float(chords[~arc].sum() / chords[arc].sum()),
        "cylinder_area_analytic": analytic,
        "cylinder_area_inscribed_arc_exact": arc_area_inscribed,
        "cylinder_area_from_vertices": wall_area_vertices,
        "cylinder_area_arc_from_vertices": arc_area_vertices,
        "cylinder_area_stored": stored,
        "identity_stored_vs_vertices_rel": abs(stored - wall_area_vertices) / analytic,
        "identity_arc_vs_inscribed_rel": abs(arc_area_vertices - arc_area_inscribed) / analytic,
        "deficit_vs_analytic_rel": (analytic - stored) / analytic,
        "arc_vertex_radius_rel_error": float(np.max(np.abs(arc_radii - radius)) / radius),
        "cross_section_from_vertices": fluid_area,
        "cross_section_stored": stored_area,
        "cross_section_stored_rel": abs(stored_area - fluid_area) / fluid_area,
        "cross_section_volume_rel": abs(volume_area - fluid_area) / fluid_area,
    }


def negative_controls(nx: int = 32, ny: int = 8, nz: int = 4) -> dict:
    """Perturb the vertex array; the identities must fire."""
    _, mesh = _mesh(nx, ny, nz)
    v0, wall_ring, outer_ring = _wall_rings(mesh)
    span = float(mesh.local_span)
    radius = float(mesh.cylinder_radius)
    chords, arc, centre, _ = _arc_split(wall_ring)

    def arc_chord_sum(ring):
        c = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        return float(c[arc].sum() * span)

    baseline = arc_chord_sum(wall_ring)
    # (a) uniform 1 % radius inflation: area is quadratic in R -> +2 %
    inflated_ring = centre + 1.01 * (wall_ring - centre)
    inflated = arc_chord_sum(inflated_ring)
    # (b) one arc vertex shifted radially by 1 % of R
    perturbed = wall_ring.copy()
    idx = int(np.argmax(np.linalg.norm(wall_ring[:-1] - centre, axis=1)))
    direction = wall_ring[idx] - centre
    direction = direction / np.linalg.norm(direction)
    perturbed[idx] = wall_ring[idx] + 0.01 * radius * direction
    shifted = arc_chord_sum(perturbed)
    return {
        "nx": nx,
        "baseline_arc_area": baseline,
        "radius_inflation_1pct_area_rel_change": abs(inflated - baseline) / baseline,
        "predicted_radius_inflation_change": 0.01,  # chord sum is linear in R
        "single_vertex_shift_1pct_area_rel_change": abs(shifted - baseline) / baseline,
        "single_vertex_shift_detected": bool(abs(shifted - baseline) / baseline > 1e-5),
    }


def refinement_ladder(nx_values=(8, 16, 32, 64, 128), ny=8, nz=4) -> dict:
    rows = []
    for nx in nx_values:
        case = geometry_report(nx=nx, ny=ny, nz=nz)
        rows.append({"nx": nx, "cells": case["cells"],
                     "arc_faces": case["arc_faces"],
                     "deficit_vs_analytic_rel": case["deficit_vs_analytic_rel"],
                     "identity_arc_vs_inscribed_rel": case["identity_arc_vs_inscribed_rel"],
                     "cross_section_stored_rel": case["cross_section_stored_rel"]})

    def slope(key):
        n = np.array([r["nx"] for r in rows], float)
        e = np.array([abs(r[key]) for r in rows], float)
        return float(np.polyfit(np.log(1 / n), np.log(e), 1)[0])

    return {"rows": rows,
            "deficit_observed_order": slope("deficit_vs_analytic_rel"),
            "identity_observed_order": slope("identity_arc_vs_inscribed_rel")}


def main() -> dict:
    torch.set_num_threads(1)
    return {
        "description": "Exact vertex-level geometric identities, inscribed-polygon area law, "
                       "refinement order and negative controls for the extruded body-fitted "
                       "cylinder O-grid (TensorFVM).",
        "stretching": STRETCHING,
        "cases": {f"nx{nx}": geometry_report(nx=nx, ny=8, nz=6) for nx in (16, 32, 64)},
        "refinement": refinement_ladder(),
        "negative_controls": negative_controls(),
    }


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["refinement"]["rows"]
    n = np.array([r["nx"] for r in rows], float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    deficit = np.array([r["deficit_vs_analytic_rel"] for r in rows])
    ax.loglog(1 / n, deficit, "o-", label="wetted area deficit vs $2\\pi R L$")
    ax.loglog(1 / n, deficit[0] * (n[0] / n) ** 2, "k--", lw=1, label="slope 2 reference")
    ax.set_xlabel("angular spacing $1/n_\\theta$")
    ax.set_ylabel("relative deficit")
    ax.set_title(f"Deficit is O($n_\\theta^{{-2}}$), observed "
                 f"{out['refinement']['deficit_observed_order']:.3f}")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    nc = out["negative_controls"]
    case = out["cases"]["nx32"]
    labels = ["stored vs\nvertices", "arc vs\npolygon law", "cross-section\nrecompute",
              "1 % radius\ninflation", "1 % vertex\nshift"]
    values = [case["identity_stored_vs_vertices_rel"], case["identity_arc_vs_inscribed_rel"],
              case["cross_section_stored_rel"],
              nc["radius_inflation_1pct_area_rel_change"],
              nc["single_vertex_shift_1pct_area_rel_change"]]
    colors = ["#2b8cbe"] * 3 + ["#de2d26"] * 2
    ax.bar(labels, [max(v, 1e-18) for v in values], color=colors)
    ax.set_yscale("log")
    ax.axhline(1e-12, color="k", ls=":", lw=1, label="round-off gate $10^{-12}$")
    ax.set_ylabel("relative discrepancy")
    ax.set_title(f"Exact identities ($n_\\theta$={case['arc_faces']} arc faces) vs negative controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/verification-body-fitted3d-geometry")
    args = parser.parse_args()
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    results = main()
    (target / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    plot(results, target / "geometry-convergence.png")
    case = results["cases"]["nx32"]
    print(json.dumps({
        "deficit_observed_order": results["refinement"]["deficit_observed_order"],
        "identity_observed_order": results["refinement"]["identity_observed_order"],
        "arc_faces_nx32": case["arc_faces"], "wake_faces_nx32": case["wake_faces"],
        "identity_stored_vs_vertices_rel": case["identity_stored_vs_vertices_rel"],
        "identity_arc_vs_inscribed_rel": case["identity_arc_vs_inscribed_rel"],
        "cross_section_recompute_rel": case["cross_section_stored_rel"],
        "cross_section_volume_rel": case["cross_section_volume_rel"],
        "arc_vertex_radius_rel_error": case["arc_vertex_radius_rel_error"],
        "deficit_vs_analytic_nx32": case["deficit_vs_analytic_rel"],
        "negative_radius_inflation": results["negative_controls"]["radius_inflation_1pct_area_rel_change"],
        "negative_vertex_shift": results["negative_controls"]["single_vertex_shift_1pct_area_rel_change"],
    }, indent=1))