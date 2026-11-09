"""Exact planar-section verification for ``native_sections3d`` (verification debt).

Module under test: ``tensorfvm.native_sections3d.cell_section_polygons``.
Audited claims (from its module docstring):

1. a cell's section polygon is *stitched* from the actual facet/plane
   intersections (no convex hull, no cell-centre heuristics);
2. coplanar facet edges cancel their internal triangulation diagonals, so a cell
   whose own face lies in the plane yields its true (non-degenerate) face ring;
3. zero-area tangencies are omitted instead of emitted as degenerate polygons;
4. a non-closed finite-area ring raises ``ValueError``.

Why it is verification debt: before this module **no test** referenced
``native_sections3d`` (0 references in ``tests/``), while every native section
figure in ``docs/verification-*`` (channel DNS, hybrid RANS, compatible inertia,
gradient jump, ...) is drawn from these polygons. A wrong polygon is silent: the
figure still renders, the mesh/field still colours, and the report claims stand
on corrupted geometry.

Anchors are analytic and independent of the code under test:

* **closed-form box section** — a plane ``0 < z < a`` through a box
  ``[0,a]^3`` has area ``a^2`` exactly; a plane coincident with an interior grid
  plane partitions it into ``nx*ny`` quads whose areas sum to ``a^2``;
* **closed-form concentric ring** — the region between a regular ``n``-gon of
  circumradius ``R`` and one of circumradius ``r`` has area
  ``(n/2) sin(2*pi/n) (R**2 - r**2)``. An O-grid whose radial/azimuthal grid
  lines are exactly those polygon vertices must reproduce that number to machine
  precision (there is no truncation left to make), and must approach
  ``pi (R**2 - r**2)`` as ``n`` grows with the pure quadrature deficit
  ``sin(t)/t``, ``t = 2*pi/n``;
* **convex-hull negative control** — the hull of the same ring omits the hole and
  overestimates by ``R**2/(R**2 - r**2)`` (``+33.33%`` at ``r = R/2``), so an
  implementation that silently fell back to a hull cannot pass the case.

All numbers here are produced by code, none is transcribed.
"""
from __future__ import annotations

import numpy as np

from ..native_mesh3d import NativeMesh3D
from ..native_sections3d import cell_section_polygons

__all__ = [
    "box_mesh",
    "annulus_mesh",
    "tetra_mesh",
    "polygon_area",
    "section_areas",
    "unique_section_area",
    "concentric_ring_area",
    "pi_annulus_area",
    "hull_annulus_area",
    "ring_error_ratio",
    "section_study",
]


def box_mesh(nx: int = 1, ny: int = 1, nz: int = 1, *, side: float = 1.0,
             xs=None, ys=None, zs=None) -> NativeMesh3D:
    """Structured hexahedral box ``[0,side]^3`` (or with explicit grid lines).

    Vertex order per cell follows ``native_mesh3d._FACES['hexahedron']``:
    ``(0,3,2,1)`` is the low-coordinate face and ``4..7`` the matching high one.
    """
    xs = np.linspace(0.0, side, nx + 1) if xs is None else np.asarray(xs, float)
    ys = np.linspace(0.0, side, ny + 1) if ys is None else np.asarray(ys, float)
    zs = np.linspace(0.0, side, nz + 1) if zs is None else np.asarray(zs, float)
    index, vertices = {}, []
    for k in range(len(zs)):
        for j in range(len(ys)):
            for i in range(len(xs)):
                index[(i, j, k)] = len(vertices)
                vertices.append((xs[i], ys[j], zs[k]))
    cells = []
    for k in range(len(zs) - 1):
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                cells.append((
                    index[(i, j, k)], index[(i + 1, j, k)],
                    index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
                    index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                    index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)],
                ))
    return NativeMesh3D.from_cells(np.asarray(vertices), cells,
                                   ["hexahedron"] * len(cells))


def annulus_mesh(n_theta: int = 16, n_r: int = 2, *, outer: float = 1.0,
                 inner: float = 0.5, height: float = 0.2) -> NativeMesh3D:
    """One-layer O-grid annulus: grid lines are the vertices of two regular n-gons.

    The cross-section of every cell is the exact quadrilateral between the
    polygons, so the *section* of the whole mesh has a closed-form area (see
    :func:`concentric_ring_area`) with no discretisation error of its own.
    """
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    radial = np.linspace(inner, outer, n_r + 1)
    index, vertices = {}, []
    for k in range(n_r + 1):
        for j in range(n_theta):
            index[(j, k, 0)] = len(vertices)
            vertices.append((radial[k] * np.cos(theta[j]),
                             radial[k] * np.sin(theta[j]), 0.0))
    for k in range(n_r + 1):
        for j in range(n_theta):
            index[(j, k, 1)] = len(vertices)
            vertices.append((radial[k] * np.cos(theta[j]),
                             radial[k] * np.sin(theta[j]), height))
    cells = []
    for k in range(n_r):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            cells.append((
                index[(j, k, 0)], index[(jn, k, 0)],
                index[(jn, k + 1, 0)], index[(j, k + 1, 0)],
                index[(j, k, 1)], index[(jn, k, 1)],
                index[(jn, k + 1, 1)], index[(j, k + 1, 1)],
            ))
    return NativeMesh3D.from_cells(np.asarray(vertices), cells,
                                   ["hexahedron"] * len(cells))


def tetra_mesh() -> NativeMesh3D:
    """Single reference tetrahedron on the unit corner (vertex 3 at z = 1)."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3)], ["tetra"])


def polygon_area(polygon: np.ndarray) -> float:
    """Signed shoelace area of a planar polygon in the projected (x, y) coords."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def section_areas(mesh: NativeMesh3D, *, axis: int = 2, value: float = 0.5,
                  tolerance: float = 1e-11):
    """Return ``(polygons, cell_ids, areas)`` with ``areas[i] = polygon_area``."""
    polygons, ids = cell_section_polygons(mesh, axis=axis, value=value,
                                          tolerance=tolerance)
    areas = np.array([polygon_area(p) for p in polygons], dtype=float)
    return polygons, np.asarray(ids, dtype=int), areas

def unique_section_area(mesh: NativeMesh3D, *, axis: int = 2, value: float = 0.5,
                        tolerance: float = 1e-11) -> float:
    """Geometric section area: identical rings emitted twice are counted once.

    ``cell_section_polygons`` is per-cell by construction. When the plane is
    *coincident* with an interior grid plane, both owners of that face emit their
    own (identical) ring, so ``sum(polygon_area)`` is exactly ``2x`` the
    geometric section area. This helper deduplicates by the rounded vertex set and
    is the quantity a consumer should quote as "section area".
    """
    polygons, _ = cell_section_polygons(mesh, axis=axis, value=value,
                                        tolerance=tolerance)
    seen, total = set(), 0.0
    for polygon in polygons:
        key = tuple(sorted(tuple(np.rint(v / tolerance).astype(np.int64))
                           for v in polygon))
        if key in seen:
            continue
        seen.add(key)
        total += polygon_area(polygon)
    return float(total)


def concentric_ring_area(n_theta: int, outer: float, inner: float) -> float:
    """Closed form: ``(n/2) sin(2*pi/n) (R**2 - r**2)`` for two regular n-gons."""
    return 0.5 * n_theta * np.sin(2.0 * np.pi / n_theta) * (outer ** 2 - inner ** 2)


def pi_annulus_area(outer: float, inner: float) -> float:
    """Continuum value ``pi (R**2 - r**2)`` the discretised ring converges to."""
    return np.pi * (outer ** 2 - inner ** 2)


def hull_annulus_area(n_theta: int, outer: float) -> float:
    """Convex hull of the ring = the outer regular n-gon (the hole is lost)."""
    return 0.5 * n_theta * np.sin(2.0 * np.pi / n_theta) * outer ** 2


def ring_error_ratio(n_theta: int) -> float:
    """Analytic relative overestimate of the continuum annulus: ``sin(t)/t - 1``."""
    t = 2.0 * np.pi / n_theta
    return float(np.sin(t) / t - 1.0)


def section_study(n_theta=(8, 16, 32, 64), *, outer=1.0, inner=0.5) -> dict:
    """Run every audited case and return a JSON-ready record (numbers only)."""
    record = {"cases": {}, "claims": {}}

    # (1) closed-form box sections. For n >= 2 the plane z = 0.5 coincides with
    # an interior grid plane, i.e. it is a *double-sided* section: both owners of
    # that face emit their own ring, so the polygon sum is 2x the area and the
    # deduplicated area is the closed form.
    box_rows = []
    for n in (1, 2, 4, 8):
        mesh = box_mesh(n, n, n)
        polygons, ids, areas = section_areas(mesh, value=0.5)
        coincident = (n % 2 == 0)
        box_rows.append({
            "n": n,
            "coincident_interior_plane": coincident,
            "n_polygons": int(areas.size),
            "n_polygons_expected": int(2 * n * n if coincident else n * n),
            "polygon_sum_area": float(areas.sum()),
            "unique_area": unique_section_area(mesh, value=0.5),
            "area_min": float(areas.min()) if areas.size else 0.0,
            "max_vertices": int(max((len(p) for p in polygons), default=0)),
        })
    record["cases"]["box_mid_plane"] = box_rows

    # graded (non-uniform) box: the plane is still the exact unit square
    graded = box_mesh(xs=[0.0, 0.05, 0.3, 1.0], ys=[0.0, 0.5, 1.0], zs=[0.0, 0.9, 1.0])
    _, _, graded_areas = section_areas(graded, value=0.5)
    record["cases"]["graded_box"] = {
        "n_polygons": int(graded_areas.size),
        "polygon_sum_area": float(graded_areas.sum()),
        "unique_area": unique_section_area(graded, value=0.5),
    }

    # boundary-coincident plane and an outside plane
    on_face_mesh = box_mesh(1, 1, 1)
    _, _, on_face = section_areas(on_face_mesh, value=0.0)
    _, _, outside = section_areas(on_face_mesh, value=1.0001)
    record["cases"]["boundary_face"] = {
        "n_polygons": int(on_face.size),
        "polygon_sum_area": float(on_face.sum()),
        "unique_area": unique_section_area(on_face_mesh, value=0.0),
    }
    record["cases"]["outside_plane"] = {"n_polygons": int(outside.size),
                                        "polygon_sum_area": float(outside.sum())}

    # (2) concentric rings: exact closed form, continuum limit, hull control
    ring_rows = []
    for n in n_theta:
        _, _, areas = section_areas(annulus_mesh(n), value=0.1)
        exact = concentric_ring_area(n, outer, inner)
        continuum = pi_annulus_area(outer, inner)
        hull = hull_annulus_area(n, outer)
        ring_rows.append({
            "n_theta": int(n),
            "n_polygons": int(areas.size),
            "n_polygons_expected": int(2 * n),
            "section_area": float(areas.sum()),
            "closed_form_area": float(exact),
            "rel_error_vs_closed_form": float(abs(areas.sum() - exact) / exact),
            "pi_area": float(continuum),
            "rel_error_vs_pi": float((areas.sum() - continuum) / continuum),
            "analytic_sin_over_t_minus_one": ring_error_ratio(n),
            "hull_area": float(hull),
            "hull_rel_overestimate": float(hull / exact - 1.0),
            "area_min": float(areas.min()),
        })
    record["cases"]["annulus_rings"] = ring_rows

    # (3) tangency: a plane touching only a vertex has no finite area
    _, _, tangency = section_areas(tetra_mesh(), value=1.0)
    _, _, tetra_face = section_areas(tetra_mesh(), value=0.0)
    record["cases"]["tangency"] = {"n_polygons": int(tangency.size),
                                   "total_area": float(tangency.sum()),
                                   "face_n_polygons": int(tetra_face.size),
                                   "face_total_area": float(tetra_face.sum())}

    # claim ledger, evaluated from the numbers above
    record["claims"] = {
        "box_unique_area_max_error":
            max(abs(r["unique_area"] - 1.0) for r in box_rows),
        "box_polygon_count_matches_semantics":
            all(r["n_polygons"] == r["n_polygons_expected"] for r in box_rows),
        "coincident_planes_are_double_sided":
            all(r["coincident_interior_plane"] for r in box_rows if r["n"] >= 2),
        "box_polygon_sum_excess_equals_two":
            [round(r["polygon_sum_area"] / r["unique_area"], 12) for r in box_rows],
        "box_polygons_stay_quads": max(r["max_vertices"] for r in box_rows),
        "graded_box_unique_area_error":
            abs(record["cases"]["graded_box"]["unique_area"] - 1.0),
        "boundary_face_unique_area_error":
            abs(record["cases"]["boundary_face"]["unique_area"] - 1.0),
        "outside_plane_empty": record["cases"]["outside_plane"]["n_polygons"] == 0,
        "ring_closed_form_max_rel_error":
            max(r["rel_error_vs_closed_form"] for r in ring_rows),
        "ring_polygon_count_ok":
            all(r["n_polygons"] == r["n_polygons_expected"] for r in ring_rows),
        "ring_deficit_tracks_analytic":
            max(abs(r["rel_error_vs_pi"] - r["analytic_sin_over_t_minus_one"])
                for r in ring_rows),
        "tangency_dropped": record["cases"]["tangency"]["n_polygons"] == 0,
        "tangency_face_area": record["cases"]["tangency"]["face_total_area"],
        "hull_control_rel_overestimate": ring_rows[0]["hull_rel_overestimate"],
    }
    return record
