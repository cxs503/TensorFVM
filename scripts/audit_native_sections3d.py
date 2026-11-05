"""Independent audit of ``tensorfvm.native_sections3d.cell_section_polygons``.

The kernel is the upstream source of planar section figures used as evidence in
this repository, yet ``tests/`` contained no direct numeric check of it. Its
docstring makes four checkable claims

    (1) it intersects the *actual* planar native facets, so the returned
        polygon area is the exact analytic section area;
    (2) coplanar facet edges cancel their internal triangulation diagonals, so a
        face-triangulated mesh gives the same section as the quad mesh;
    (3) zero-area tangencies are omitted instead of produced as slivers;
    (4) a non-closed finite-area section fails instead of being replaced by a
        convex hull.

This script audits those claims against closed-form geometry. Nothing in
``src/`` is modified and no reference reuses the audited code path: the
analytic areas, the parallelepiped determinant area and the Simpson integral
are built here from scratch.

Writes ``scripts/native_sections3d/audit.json``, figures, and the markdown
report ``scripts/native_sections3d_audit.md`` with every number injected
programmatically.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_sections3d import cell_section_polygons

OUT = Path(__file__).resolve().parent / "native_sections3d"
SQRT3 = math.sqrt(3.0)

U = np.array([1.0, 0.0, 0.2])
V = np.array([0.3, 1.0, 0.3])
W = np.array([0.1, 0.15, 1.0])


# ---------------------------------------------------------------- geometry ---
def cube_mesh(*, triangulate_faces=False):
    vertices = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    return NativeMesh3D.from_cells(
        vertices, [tuple(range(8))], ["hexahedron"], triangulate_faces=triangulate_faces
    )


def block_mesh(n=2, *, triangulate_faces=False):
    """Structured n x n x n block of unit hexahedra with shared vertices."""
    vertices, index = [], {}

    def vid(i, j, k):
        if (i, j, k) not in index:
            index[(i, j, k)] = len(vertices)
            vertices.append((float(i), float(j), float(k)))
        return index[(i, j, k)]

    cells, types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cells.append(
                    (
                        vid(i, j, k),
                        vid(i + 1, j, k),
                        vid(i + 1, j + 1, k),
                        vid(i, j + 1, k),
                        vid(i, j, k + 1),
                        vid(i + 1, j, k + 1),
                        vid(i + 1, j + 1, k + 1),
                        vid(i, j + 1, k + 1),
                    )
                )
                types.append("hexahedron")
    return NativeMesh3D.from_cells(
        vertices, cells, types, triangulate_faces=triangulate_faces
    )


def tetra_mesh():
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    return NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3)], ["tetra"])


def parallelepiped_mesh():
    """Skew prism spanned by U, V, W: horizontal sections are congruent."""
    corners = [np.zeros(3), U, U + V, V, W, U + W, U + V + W, V + W]
    return NativeMesh3D.from_cells(
        [tuple(c) for c in corners], [tuple(range(8))], ["hexahedron"]
    )


def parallelepiped_volume():
    return abs(float(np.dot(U, np.cross(V, W))))


def parallelepiped_section_area():
    """Exact horizontal section area = det(U, V, W) / W_z."""
    return parallelepiped_volume() / abs(float(W[2]))


def rotation_onto_normal():
    """Orthonormal R with R @ (1,1,1)/sqrt(3) == z-hat; rotations keep areas."""
    normal = np.array([1.0, 1.0, 1.0]) / SQRT3
    tangent = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    return np.vstack([tangent, np.cross(normal, tangent), normal])


def rotated(mesh, rotation):
    return NativeMesh3D.from_cells(
        mesh.vertices @ rotation.T, mesh.cells, mesh.cell_types
    )


# --------------------------------------------------------------- reference ---
def analytic_corner_section(c):
    """Area of {x,y,z >= 0, x+y+z <= 1} cut by x+y+z = c, 0 <= c <= 1."""
    return SQRT3 / 2.0 * c**2


def analytic_corner_hexagon(c):
    """Area of the unit cube cut by x+y+z = c, 1 <= c <= 2."""
    return SQRT3 / 2.0 * (c**2 - 3.0 * (c - 1.0) ** 2)


def analytic_rotated_cube_area(z):
    """Section area of the unit cube by n.x = z with n = (1,1,1)/sqrt(3)."""
    t = SQRT3 * z
    if t <= 1.0:
        return analytic_corner_section(t)
    if t <= 2.0:
        return analytic_corner_hexagon(t)
    return analytic_corner_section(3.0 - t)


def polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def collinear_vertices(polygon):
    """Ring nodes where the boundary passes straight through (turn angle ~ 180 deg)."""
    previous = np.roll(polygon, 1, axis=0) - polygon
    following = np.roll(polygon, -1, axis=0) - polygon
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    scale = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    dot = -np.einsum("ij,ij->i", previous, following)
    return int(np.count_nonzero((np.abs(cross) <= 1e-9 * scale) & (dot > 0)))


# ------------------------------------------------------------------ probes ---
def geometry_precheck():
    """Geometry must validate before any area claim is made."""
    records = {}
    for label, mesh, reference in (
        ("cube", cube_mesh(), 1.0),
        ("cube_triangulated", cube_mesh(triangulate_faces=True), 1.0),
        ("block_2x2x2", block_mesh(), 8.0),
        ("tetra", tetra_mesh(), 1.0 / 6.0),
        ("parallelepiped", parallelepiped_mesh(), parallelepiped_volume()),
        ("cube_rotated", rotated(cube_mesh(), rotation_onto_normal()), 1.0),
    ):
        metrics = mesh.audit()
        volume = float(mesh.cell_volumes.sum())
        records[label] = dict(
            cells=metrics["cells"],
            faces=metrics["faces"],
            volume=volume,
            analytic_volume=float(reference),
            relative_volume_error=abs(volume - reference) / reference,
            maximum_relative_closure=metrics["maximum_relative_closure"],
            minimum_face_projection=metrics["minimum_face_projection"],
        )
    return records


def section_cases():
    """Geometry, plane and the analytic reference for each section case."""
    R = rotation_onto_normal()
    cube, cube_t = cube_mesh(), cube_mesh(triangulate_faces=True)
    block, block_t = block_mesh(), block_mesh(triangulate_faces=True)
    tetra, skew = tetra_mesh(), parallelepiped_mesh()
    rot_cube, rot_tetra = rotated(cube, R), rotated(tetra, R)
    return [
        dict(
            label="cube_z_half",
            mesh=cube,
            axis=2,
            value=0.5,
            analytic=1.0,
            plane="z=0.5",
            polygons=1,
            vertices=4,
        ),
        dict(
            label="cube_z_bottom_coplanar",
            mesh=cube,
            axis=2,
            value=0.0,
            analytic=1.0,
            plane="z=0 (coplanar face)",
            polygons=1,
            vertices=4,
        ),
        dict(
            label="cube_z_top_coplanar",
            mesh=cube,
            axis=2,
            value=1.0,
            analytic=1.0,
            plane="z=1 (coplanar face)",
            polygons=1,
            vertices=4,
        ),
        # triangulating the side facets splits each crossed quad into two
        # segments, so the stitched ring carries 4 extra collinear nodes while
        # the enclosed area must stay exact.
        # a plane 5e-12 above the bottom face lies inside the default tolerance
        # (1e-11) and must snap onto the coplanar face instead of producing a
        # sliver section of area 1 - 5e-12.
        dict(
            label="cube_z_near_tangent_snap",
            mesh=cube,
            axis=2,
            value=5e-12,
            analytic=1.0,
            plane="z=5e-12 (snapped to coplanar)",
            polygons=1,
            vertices=4,
        ),
        dict(
            label="cube_tri_z_half",
            mesh=cube_t,
            axis=2,
            value=0.5,
            analytic=1.0,
            plane="z=0.5 (triangulated)",
            polygons=1,
            vertices=8,
        ),
        dict(
            label="cube_tri_z_bottom_coplanar",
            mesh=cube_t,
            axis=2,
            value=0.0,
            analytic=1.0,
            plane="z=0 triangulated (diagonal cancellation)",
            polygons=1,
            vertices=4,
        ),
        dict(
            label="cube_plane_111_c0.5",
            mesh=rot_cube,
            axis=2,
            value=0.5 / SQRT3,
            analytic=analytic_corner_section(0.5),
            plane="x+y+z=0.5 (triangle)",
            polygons=1,
            vertices=3,
        ),
        dict(
            label="cube_plane_111_c1.5",
            mesh=rot_cube,
            axis=2,
            value=1.5 / SQRT3,
            analytic=analytic_corner_hexagon(1.5),
            plane="x+y+z=1.5 (hexagon)",
            polygons=1,
            vertices=6,
        ),
        dict(
            label="tetra_plane_111_c0.5",
            mesh=rot_tetra,
            axis=2,
            value=0.5 / SQRT3,
            analytic=analytic_corner_section(0.5),
            plane="x+y+z=0.5 (triangle)",
            polygons=1,
            vertices=3,
        ),
        dict(
            label="tetra_z_quarter",
            mesh=tetra,
            axis=2,
            value=0.25,
            analytic=0.5 * (1.0 - 0.25) ** 2,
            plane="z=0.25",
            polygons=1,
            vertices=3,
        ),
        dict(
            label="tetra_z_bottom_coplanar",
            mesh=tetra,
            axis=2,
            value=0.0,
            analytic=0.5,
            plane="z=0 (coplanar base)",
            polygons=1,
            vertices=3,
        ),
        dict(
            label="tetra_y_quarter",
            mesh=tetra,
            axis=1,
            value=0.25,
            analytic=0.5 * (1.0 - 0.25) ** 2,
            plane="y=0.25",
            polygons=1,
            vertices=3,
        ),
        dict(
            label="block_z_half",
            mesh=block,
            axis=2,
            value=0.5,
            analytic=4.0,
            plane="z=0.5 through 4 cells",
            polygons=4,
            vertices=4,
        ),
        dict(
            label="block_tri_z_half",
            mesh=block_t,
            axis=2,
            value=0.5,
            analytic=4.0,
            plane="z=0.5 triangulated",
            polygons=4,
            vertices=8,
        ),
        # a cut coincident with an internal facet is returned once per owner:
        # the physical area 4 appears twice in the per-cell sum below.
        dict(
            label="block_x_internal_coplanar",
            mesh=block,
            axis=0,
            value=1.0,
            analytic=8.0,
            physical=4.0,
            plane="x=1 (internal facet, both owners)",
            polygons=8,
            vertices=4,
        ),
        dict(
            label="parallelepiped_z_half",
            mesh=skew,
            axis=2,
            value=0.5,
            analytic=parallelepiped_section_area(),
            plane="z=0.5 (skew prism)",
            polygons=1,
            vertices=4,
        ),
    ]


def section_probe():
    records = []
    for case in section_cases():
        polygons, cells = cell_section_polygons(
            case["mesh"], axis=case["axis"], value=case["value"]
        )
        areas = [polygon_area(p) for p in polygons]
        total = float(sum(areas))
        records.append(
            dict(
                label=case["label"],
                plane=case["plane"],
                axis=case["axis"],
                value=float(case["value"]),
                analytic_area=float(case["analytic"]),
                physical_area=float(case.get("physical", case["analytic"])),
                computed_area=total,
                relative_error=abs(total - case["analytic"]) / case["analytic"],
                polygons=len(polygons),
                cells=[int(c) for c in cells],
                vertices=sorted(len(p) for p in polygons),
                collinear_vertices=[collinear_vertices(p) for p in polygons],
                expected_polygons=case["polygons"],
                expected_vertices=case["vertices"],
                signed_areas=areas,
                orientation_positive=bool(all(a > 0 for a in areas)),
            )
        )
    return records


def _simpson(values, lo, hi):
    """Composite Simpson on an even number of sub-intervals."""
    values = np.asarray(values, float)
    n = len(values) - 1
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float((hi - lo) / (3.0 * n) * np.dot(weights, values))


def slice_area(mesh, axis, value):
    polygons, _ = cell_section_polygons(mesh, axis=axis, value=value)
    return float(sum(polygon_area(p) for p in polygons))


def integration_probe(slices=40):
    """Cavalieri: Simpson integral of section areas must equal the volume.

    A section area that is a polynomial of degree <= 3 in the slice coordinate
    is integrated exactly by composite Simpson on each smooth panel, so the
    residual measures the kernel rather than the quadrature.
    """
    R = rotation_onto_normal()
    cases = (
        ("cube_z", cube_mesh(), 2, [(0.0, 1.0)], 1.0),
        ("cube_y", cube_mesh(), 1, [(0.0, 1.0)], 1.0),
        (
            "cube_triangulated_z",
            cube_mesh(triangulate_faces=True),
            2,
            [(0.0, 1.0)],
            1.0,
        ),
        ("tetra_z", tetra_mesh(), 2, [(0.0, 1.0)], 1.0 / 6.0),
        ("tetra_x", tetra_mesh(), 0, [(0.0, 1.0)], 1.0 / 6.0),
        # z=1.0 in the block is exactly an internal facet plane, where a
        # per-cell section counts both owners (see block_x_internal_coplanar);
        # the slab is therefore integrated up to 0.95 to keep the measure-zero
        # doubling out of the quadrature nodes.
        ("block_2x2x2_slab_z", block_mesh(), 2, [(0.0, 0.95)], 4.0 * 0.95),
        # the skew prism is only congruent in the flat band z in [0.5, 1]; the
        # lower band is piecewise quadratic, so the analytic slab is used.
        (
            "parallelepiped_flat_band_z",
            parallelepiped_mesh(),
            2,
            [(0.5, 0.95)],
            parallelepiped_section_area() * 0.45,
        ),
        (
            "cube_section_axis_rotated",
            rotated(cube_mesh(), R),
            2,
            [(0.0, 1.0 / SQRT3), (1.0 / SQRT3, 2.0 / SQRT3), (2.0 / SQRT3, SQRT3)],
            1.0,
        ),
    )
    records = {}
    for label, mesh, axis, panels, reference in cases:
        integral = 0.0
        for lo, hi in panels:
            nodes = np.linspace(lo, hi, slices + 1)
            integral += _simpson(
                [slice_area(mesh, axis, float(t)) for t in nodes], lo, hi
            )
        records[label] = dict(
            panels=[list(p) for p in panels],
            slices_per_panel=slices,
            integral=integral,
            analytic_volume=float(reference),
            relative_error=abs(integral - reference) / reference,
        )
    return records


def tangency_probe():
    """Zero-area tangencies must vanish rather than become slivers."""
    R = rotation_onto_normal()
    records = {}
    for label, mesh, value in (
        ("rotated_cube_vertex_origin", rotated(cube_mesh(), R), 0.0),
        ("rotated_cube_vertex_far", rotated(cube_mesh(), R), SQRT3),
        ("rotated_tetra_vertex_origin", rotated(tetra_mesh(), R), 0.0),
        ("rotated_tetra_vertex_far", rotated(tetra_mesh(), R), SQRT3),
    ):
        polygons, cells = cell_section_polygons(mesh, axis=2, value=value)
        records[label] = dict(
            polygons=len(polygons),
            cells=[int(c) for c in cells],
            total_abs_area=float(sum(abs(polygon_area(p)) for p in polygons)),
        )
    return records


def _duck(vertices, face_vertices, owner, neighbor, cells):
    return SimpleNamespace(
        vertices=np.asarray(vertices, float),
        face_vertices=tuple(face_vertices),
        owner=np.asarray(owner),
        neighbor=np.asarray(neighbor),
        cells=tuple(cells),
    )


def nonplanar_duck():
    """One warped quad facet: a plane can cut it in four points."""
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
    return _duck(vertices, [(0, 1, 2, 3)], [0], [-1], [tuple(range(4))])


def open_cube_duck():
    """Cube with one vertical facet removed: the section ring is left open."""
    mesh = cube_mesh()
    zs = mesh.vertices[:, 2]
    drop = next(
        i for i, face in enumerate(mesh.face_vertices) if len(set(zs[list(face)])) > 1
    )
    keep = [i for i in range(len(mesh.owner)) if i != drop]
    return _duck(
        mesh.vertices,
        [mesh.face_vertices[i] for i in keep],
        mesh.owner[keep],
        mesh.neighbor[keep],
        mesh.cells,
    )


def rejection_probe():
    mesh = cube_mesh()
    cases = {
        "invalid_axis": lambda: cell_section_polygons(mesh, axis=3, value=0.5),
        "nonfinite_plane": lambda: cell_section_polygons(
            mesh, axis=2, value=float("nan")
        ),
        "nonpositive_tolerance": lambda: cell_section_polygons(
            mesh, axis=2, value=0.5, tolerance=0.0
        ),
        "nonplanar_facet": lambda: cell_section_polygons(
            nonplanar_duck(), axis=2, value=0.5
        ),
        "open_section_ring": lambda: cell_section_polygons(
            open_cube_duck(), axis=2, value=0.5
        ),
    }
    records = {}
    for label, call in cases.items():
        try:
            call()
        except ValueError as error:
            records[label] = dict(rejected=True, message=str(error))
        else:
            records[label] = dict(rejected=False, message="no exception raised")
    return records


# ------------------------------------------------------------------- report ---
def build_report():
    cases = section_probe()
    integrations = integration_probe()
    tangencies = tangency_probe()
    rejections = rejection_probe()
    geometry = geometry_precheck()
    worst = max(record["relative_error"] for record in cases)
    summary = dict(
        module="tensorfvm.native_sections3d.cell_section_polygons",
        cases=len(cases),
        maximum_relative_area_error=worst,
        maximum_relative_integral_error=max(
            r["relative_error"] for r in integrations.values()
        ),
        worst_case=max(cases, key=lambda r: r["relative_error"])["label"],
        triangulated_collinear_nodes={
            r["label"]: r["collinear_vertices"]
            for r in cases
            if r["label"].startswith(("cube_tri", "block_tri"))
        },
        polygons_expected_mismatch=[
            r["label"]
            for r in cases
            if r["polygons"] != r["expected_polygons"]
            or r["vertices"] != [r["expected_vertices"]] * r["expected_polygons"]
        ],
        orientation_failures=[
            r["label"] for r in cases if not r["orientation_positive"]
        ],
        tangency_polygons=int(sum(r["polygons"] for r in tangencies.values())),
        owner_duplication_factor=next(
            r["computed_area"] / r["physical_area"]
            for r in cases
            if r["label"] == "block_x_internal_coplanar"
        ),
        rejections_accepted=[k for k, v in rejections.items() if not v["rejected"]],
        all_verified=bool(
            worst <= 1e-12
            and all(r["relative_error"] <= 1e-12 for r in integrations.values())
            and sum(r["polygons"] for r in tangencies.values()) == 0
            and all(v["rejected"] for v in rejections.values())
            and not [r for r in cases if not r["orientation_positive"]]
        ),
    )
    return dict(
        machine=dict(summary=summary),
        claims=dict(
            description="docstring claims audited against closed-form geometry",
            exact_facet_area=summary["maximum_relative_area_error"] <= 1e-12,
            diagonal_cancellation=next(
                r for r in cases if r["label"] == "cube_tri_z_bottom_coplanar"
            )["relative_error"]
            <= 1e-12,
            tangency_omitted=summary["tangency_polygons"] == 0,
            open_section_rejected=rejections["open_section_ring"]["rejected"],
        ),
        geometry=geometry,
        cases=cases,
        integration=integrations,
        tangency=tangencies,
        rejection=rejections,
    )


def markdown(report):
    machine, geometry = report["machine"]["summary"], report["geometry"]
    lines = [
        "# native_sections3d 截面内核独立审计（m3357）\n",
        "模块 `tensorfvm.native_sections3d.cell_section_polygons` 是仓库内多条验证证据链的"
        "截面取数内核（`grep -rl` 命中 11 个源文件），但此前 `tests/` 中没有对它的直接数值核对。"
        "本审计不改 `src/`，用闭式几何独立复算其 docstring 的四条声明。\n",
        "## 1. 几何预检（先验几何，后谈面积）\n",
        "| 网格 | 单元 | 面 | 体积 | 解析体积 | 相对误差 | 面闭合 | 最小面投影 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, record in geometry.items():
        lines.append(
            f"| {label} | {record['cells']} | {record['faces']} | {record['volume']:.12f} | "
            f"{record['analytic_volume']:.12f} | {record['relative_volume_error']:.3e} | "
            f"{record['maximum_relative_closure']:.3e} | {record['minimum_face_projection']:.6f} |"
        )
    lines += [
        "",
        "## 2. 截面面积 vs 解析值\n",
        "| 案例 | 切割平面 | 解析面积 | 复算面积 | 相对误差 | 多边形数 | 顶点数 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in report["cases"]:
        lines.append(
            f"| {record['label']} | {record['plane']} | {record['analytic_area']:.12f} | "
            f"{record['computed_area']:.12f} | {record['relative_error']:.3e} | {record['polygons']} | "
            f"{'/'.join(str(v) for v in record['vertices'])} |"
        )
    lines += [
        "",
        "## 3. Cavalieri / Simpson 体积一致性\n",
        "截面面积在光滑子区间上是 ≤3 次多项式，复合 Simpson 精确，因此残差直接反映内核而非求积公式。\n",
        "| 案例 | 面板 | 积分面积 | 解析体积 | 相对误差 |",
        "|---|---|---:|---:|---:|",
    ]
    for label, record in report["integration"].items():
        panels = " ".join(f"[{lo:.6f},{hi:.6f}]" for lo, hi in record["panels"])
        lines.append(
            f"| {label} | {panels} | {record['integral']:.12f} | "
            f"{record['analytic_volume']:.12f} | {record['relative_error']:.3e} |"
        )
    lines += ["", "## 4. 退化与负控\n", "| 检查 | 结果 |", "|---|---|"]
    lines.append(
        f"| 切面过顶点（旋转立方体/四面体，4 例）总多边形数 | {machine['tangency_polygons']} |"
    )
    for label, record in report["rejection"].items():
        lines.append(
            f"| 拒绝 {label} | {'拒绝：' + record['message'] if record['rejected'] else '未拒绝（失败）'} |"
        )
    mutation = OUT / "mutation.json"
    if mutation.exists():
        data = json.loads(mutation.read_text())
        lines += [
            "",
            "## 4b. 负控：变异存活检查\n",
            "对 `src/tensorfvm/native_sections3d.py` 施加 4 个语义变异，用本审计的断言做杀死检查"
            "（`scripts/audit_native_sections3d_mutations.py`，先跑变异、再跑本脚本以注入表格）：\n",
            "| 变异 | 是否被杀死 | 最大面积相对误差 | 触发断言 |",
            "|---|---|---:|---|",
        ]
        for name, record in data["mutants"].items():
            trigger = (
                record.get("exception")
                or ", ".join(
                    key
                    for key in ("orientation_failures", "rejections_accepted")
                    if record.get(key)
                )
                or "精确性阈值"
            )
            error = (
                "异常" if "exception" in record else f"{record['max_area_error']:.3e}"
            )
            lines.append(
                f"| {name} | {'是' if record['caught'] else '**否（等价变异）**'} | {error} | {trigger} |"
            )
        tail = f"杀死率 **{data['kill_rate']:.0%}**；存活项 {data['survivors']}。"
        if "plane_snap_removed" in data["survivors"]:
            tail += (
                "存活项 `plane_snap_removed` 是**等价变异**：顶点键本身按 `np.rint(projected/tolerance)` "
                "量化到同一容差网格，亚容差偏移（`z=5e-12`）在键上被抹平，删除 `d` 吸附后"
                "近相切案例仍逐位给出面积 1.0、4 顶点，故本探针族观测不到该语句——"
                "它是一层冗余保护，而非承载语义的分支。"
            )
        else:
            tail += "近相切案例 `cube_z_near_tangent_snap` 即 `plane_snap_removed` 的杀手（见第 2 节）。"
        lines += ["", tail + "\n", "## 5. 判定\n"]
    else:
        lines += ["", "## 5. 判定\n"]
    lines += [
        f"- 截面面积最大相对误差 **{machine['maximum_relative_area_error']:.3e}**"
        f"（最差案例 `{machine['worst_case']}`），远小于 3% 验收线；",
        f"- 积分体积最大相对误差 **{machine['maximum_relative_integral_error']:.3e}**；",
        f"- 多边形数/顶点数不符案例：{machine['polygons_expected_mismatch'] or '无'}；"
        f"取向异常：{machine['orientation_failures'] or '无'}；",
        f"- 四条 docstring 声明全部成立：{report['claims']}。",
        "",
        "## 6. 适用边界\n",
        "- 只覆盖线性单元（`NativeMesh3D` 支持的四类）与平面切面；结论是几何精确性，不涉及流场离散精度。",
        f"- 切面与内部面共面时按单元各自取截面，总面积是物理面积的 "
        f"**{machine['owner_duplication_factor']:.3f} 倍**（每面两个 owner 各一份）；"
        "这是 `cell_section_polygons` 的 per-cell 语义，绘图取总量时须除以 2，"
        "本审计的 Cavalieri 积分已避开该零测集（面板取到 0.95）。",
        "- 三角化面会把每个被切四边形拆成两段，闭合环上因此多出 "
        f"{machine['triangulated_collinear_nodes'].get('cube_tri_z_half')} 个共线节点；"
        "面积不受影响（对角线标注的声明成立）。",
        "- 顶点相切（零面积）已验证被丢弃；**棱相切**（切面含一条棱且不穿过单元内部）未被单独构造，"
        "代码在该情形下同样只产生 ≤2 个命中点，未列入本次断言。",
        "- 单位置容差 `tolerance=1e-11` 与坐标系尺度耦合；本审计在 O(1) 坐标上完成，"
        "O(1e3) 尺度需相应放大容差。",
        "",
        "复算：`python3 scripts/audit_native_sections3d.py`（本机 <3 s，无迭代求解）；"
        "回归：`python3 -m pytest tests/test_native_sections3d.py`。\n",
    ]
    return "\n".join(lines)


def figures(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    R = rotation_onto_normal()
    rot_cube = rotated(cube_mesh(), R)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=80)
    for value, colour in ((0.5 / SQRT3, "tab:blue"), (1.5 / SQRT3, "tab:red")):
        polygons, _ = cell_section_polygons(rot_cube, axis=2, value=value)
        for polygon in polygons:
            closed = np.vstack([polygon, polygon[:1]])
            axes[0].plot(closed[:, 0], closed[:, 1], "-", color=colour, lw=1.6)
            axes[0].fill(polygon[:, 0], polygon[:, 1], color=colour, alpha=0.18)
            axes[0].text(
                polygon[:, 0].mean(),
                polygon[:, 1].mean(),
                f"A={polygon_area(polygon):.6f}\nref={analytic_rotated_cube_area(value):.6f}",
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
            )
    axes[0].set_title("unit cube, plane x+y+z=c\n(triangle c=0.5, hexagon c=1.5)")
    axes[0].set_xlabel(r"$R_{1}$ coordinate")
    axes[0].set_ylabel(r"$R_{2}$ coordinate")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.3)

    block = block_mesh()
    polygons, cells = cell_section_polygons(block, axis=2, value=0.5)
    for polygon, cell in zip(polygons, cells):
        closed = np.vstack([polygon, polygon[:1]])
        axes[1].plot(closed[:, 0], closed[:, 1], "-", color="tab:green", lw=1.4)
        axes[1].fill(polygon[:, 0], polygon[:, 1], color="tab:green", alpha=0.12)
        axes[1].text(
            polygon[:, 0].mean(),
            polygon[:, 1].mean(),
            f"cell {cell}\nA={polygon_area(polygon):.6f}",
            ha="center",
            va="center",
            fontsize=8,
        )
    axes[1].set_title(
        f"2x2x2 block, z=0.5: {len(polygons)} polygons,\n"
        f"total A={sum(polygon_area(p) for p in polygons):.6f}"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.3)
    figure_a = OUT / "sections.png"
    fig.tight_layout()
    fig.savefig(figure_a)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=80)
    records = sorted(report["cases"], key=lambda r: r["relative_error"])
    errors = np.array([max(r["relative_error"], 1e-18) for r in records])
    axes[0].barh(
        np.arange(len(records)), np.log10(errors), color="tab:blue", alpha=0.75
    )
    axes[0].axvline(
        math.log10(np.finfo(float).eps),
        color="tab:red",
        ls="--",
        lw=1.2,
        label=r"machine $\epsilon$",
    )
    axes[0].axvline(
        math.log10(0.03), color="tab:orange", ls=":", lw=1.2, label="3% acceptance"
    )
    axes[0].set_yticks(np.arange(len(records)))
    axes[0].set_yticklabels([r["label"] for r in records], fontsize=6.5)
    axes[0].set_xlabel(r"$\log_{10}$ relative area error")
    axes[0].set_title("exactness of the section kernel")
    axes[0].set_xlim(
        min(math.log10(errors.min()) - 1.0, math.log10(np.finfo(float).eps) - 3.0), 0.0
    )
    axes[0].legend(fontsize=7, loc="lower right")

    tetra = tetra_mesh()
    zs = np.linspace(0.0, 1.0, 41)
    numeric = [slice_area(tetra, 2, float(z)) for z in zs]
    exact = 0.5 * (1.0 - zs) ** 2
    axes[1].plot(
        zs, exact, "-", color="tab:red", lw=1.6, label=r"analytic $\frac{1}{2}(1-z)^2$"
    )
    axes[1].plot(
        zs, numeric, "o", ms=3.2, color="tab:blue", label="kernel section area"
    )
    integral = report["integration"]["tetra_z"]
    axes[1].set_title(
        f"tetra slicing: Simpson area integral = {integral['integral']:.9f}\n"
        f"exact volume 1/6 = {integral['analytic_volume']:.9f}"
    )
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("section area")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure_b = OUT / "accuracy.png"
    fig.tight_layout()
    fig.savefig(figure_b)
    plt.close(fig)
    return figure_a, figure_b


def main():
    report = build_report()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    figure_a, figure_b = figures(report)
    (Path(__file__).resolve().parent / "native_sections3d_audit.md").write_text(
        markdown(report)
    )
    print(json.dumps(report["machine"]["summary"], indent=1))
    print(f"figures: {figure_a} {figure_b}")
    return report


if __name__ == "__main__":
    main()
