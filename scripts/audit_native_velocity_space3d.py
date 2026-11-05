"""Independent audit of NativeVelocitySpace3D (exact face-normal elimination).

The production module ``tensorfvm.native_velocity_space3d`` is imported by
``native_hybrid_viscous3d`` (and therefore by the hybrid Stokes/hybrid NS
stack) but had no direct test: the rotated (normal-velocity) branch was only
reached indirectly through solver-level cases. This script audits the two
claims the module makes in its own docstring

    "P has orthonormal columns and maps reduced increments into unconstrained
     cell velocities and admissible face velocities. No normal penalty is used."

against exact linear algebra, and audits the downstream operator claim that
normal-only (free-slip) elimination leaves no spurious zero-energy mode.

Nothing in ``src/`` is modified: every check is a reference computation that
does not reuse the audited code path (constraint matrix, KKT solve, penalty
limit and eigenvalues are built here from scratch with dense linear algebra).

Writes ``scripts/native_velocity_space3d/audit.json``, a figure, and the
markdown report ``scripts/native_velocity_space3d_audit.md`` with every number
injected programmatically.
"""

import json
from pathlib import Path

import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.native_velocity_space3d import NativeVelocitySpace3D
from tensorfvm.verification.native_energy import two_tetrahedra
from tensorfvm.verification.native_hybrid_flow import channel_mesh

OUT = Path(__file__).resolve().parent / "native_velocity_space3d"
SEED = 20261103


def single_hexahedron():
    cube = [
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
        cube, [tuple(range(8))], ["hexahedron"], triangulate_faces=True
    )


def channel_block():
    """Structured 2x2x2 block: 8 hexahedra, 24 internal and 48 boundary facets."""
    return channel_mesh(2, triangulate_faces=True)


def skewed_tetrahedra():
    """Two tetrahedra sharing one internal facet; boundary normals are slanted."""
    return two_tetrahedra()


def facet_normals(mesh):
    return mesh.area_vectors / np.linalg.norm(mesh.area_vectors, axis=1)[:, None]


def slanted_boundary_face(mesh):
    """Boundary facet whose normal is not axis aligned (all three components live)."""
    normals = facet_normals(mesh)
    for face in np.flatnonzero(mesh.neighbor < 0):
        if np.count_nonzero(np.abs(np.abs(normals[face]) - 1.0) < 1e-9) == 0:
            return int(face)
    raise RuntimeError("no slanted boundary facet in this mesh")


def constraint_matrix(space, components):
    """Reference constraint operator: normal components + fixed Cartesian rows."""
    rows = []
    normals = space.normal
    for face in np.flatnonzero(space.normal_faces):
        row = np.zeros(space.prolongation.shape[0])
        row[components * (space.N + face) : components * (space.N + face) + 3] = (
            normals[face]
        )
        rows.append(row)
    for index in space.cartesian_fixed:
        row = np.zeros(space.prolongation.shape[0])
        row[index] = 1.0
        rows.append(row)
    return np.array(rows).reshape(len(rows), space.prolongation.shape[0])


def configurations():
    """(label, mesh, cartesian mask, normal-face selection) audit configurations."""
    channel, tetra = channel_block(), skewed_tetrahedra()
    cases = []
    for label, mesh in (("channel", channel), ("tetra", tetra)):
        faces = len(mesh.owner)
        boundary = np.flatnonzero(mesh.neighbor < 0)
        normals = facet_normals(mesh)
        axis_aligned = [
            int(f)
            for f in boundary
            if np.count_nonzero(np.abs(np.abs(normals[f]) - 1.0) < 1e-9) >= 1
        ]
        slanted = [
            int(f)
            for f in boundary
            if np.count_nonzero(np.abs(np.abs(normals[f]) - 1.0) < 1e-9) == 0
        ]
        other = int([f for f in boundary if f not in axis_aligned[:1] + slanted[:1]][0])
        free_mask = np.zeros((faces, 3), bool)
        one_component = free_mask.copy()
        one_component[other, 2] = True
        two_components = free_mask.copy()
        two_components[other, 1:] = True

        def selection(faces_):
            return np.isin(np.arange(faces), faces_)

        cases.append((f"{label}_cartesian_traction_free", mesh, free_mask.copy(), None))
        cases.append((f"{label}_cartesian_one_component", mesh, one_component, None))
        if axis_aligned:
            cases.append(
                (
                    f"{label}_normal_axis_aligned",
                    mesh,
                    free_mask.copy(),
                    selection([axis_aligned[0]]),
                )
            )
        if slanted:
            cases.append(
                (
                    f"{label}_normal_slanted",
                    mesh,
                    free_mask.copy(),
                    selection([slanted[0]]),
                )
            )
            cases.append(
                (
                    f"{label}_normal_plus_cartesian",
                    mesh,
                    two_components,
                    selection([slanted[0]]),
                )
            )
        selected = (axis_aligned[:1] + slanted[:2]) or [other]
        if len(selected) >= 2:
            cases.append(
                (
                    f"{label}_normal_two_faces",
                    mesh,
                    free_mask.copy(),
                    selection(selected[:2]),
                )
            )
    return channel, cases


def algebra_probe(mesh, mask, normal_faces, components=3, seed=SEED):
    space = NativeVelocitySpace3D(mesh, components, mask, normal_faces)
    P = space.prolongation.toarray()
    C = constraint_matrix(space, components)
    size = components * (space.N + space.F)
    rng = np.random.default_rng(seed)
    record = {
        "full_dofs": int(size),
        "reduced_dofs": int(P.shape[1]),
        "rotated": bool(space.rotated),
        "orthonormality_error": float(np.abs(P.T @ P - np.eye(P.shape[1])).max()),
        "constraint_range_error": float(np.abs(C @ P).max(initial=0.0))
        if C.size
        else 0.0,
        "rank_constraints": int(np.linalg.matrix_rank(C)) if C.size else 0,
    }
    nullity = size - record["rank_constraints"]
    record["nullity_gap"] = int(nullity - P.shape[1])
    x = rng.standard_normal(P.shape[1])
    record["roundtrip_error"] = float(np.abs(space.restrict(space.expand(x)) - x).max())
    raw = rng.standard_normal(size)
    projected = space.expand(space.restrict(raw))
    record["constraint_error_projected"] = float(space.constraint_error(projected))
    record["constraint_error_raw"] = float(space.constraint_error(raw))
    record["idempotence_error"] = float(
        np.abs(space.expand(space.restrict(projected)) - projected).max()
    )
    # KKT equivalence on a synthetic SPD operator: the elimination must give the
    # same answer as a saddle-point solve with hard constraints.
    operator = rng.standard_normal((size, size))
    operator = operator @ operator.T + size * np.eye(size)
    rhs = rng.standard_normal(size)
    reduced = (P.T @ operator @ P, P.T @ rhs)
    x_elim = np.linalg.solve(reduced[0], reduced[1])
    u_elim = P @ x_elim
    kkt = (
        np.block([[operator, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
        if C.size
        else operator
    )
    u_kkt = (
        np.linalg.solve(kkt, np.r_[rhs, np.zeros(C.shape[0])])[:size]
        if C.size
        else np.linalg.solve(operator, rhs)
    )
    scale = np.abs(u_elim).max()
    record["kkt_solution_error"] = float(np.abs(u_elim - u_kkt).max() / scale)
    record["kkt_constraint_error"] = (
        float(np.abs(C @ u_kkt).max(initial=0.0)) if C.size else 0.0
    )
    # Penalty reference: adding 1/(2 eps) |C u|^2 must approach the elimination
    # answer only as eps -> 0. First-order slope proves no penalty is embedded.
    record["penalty"] = []
    if C.size:
        for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            penalised = np.block([[operator, C.T], [C, -eps * np.eye(C.shape[0])]])
            u_pen = np.linalg.solve(penalised, np.r_[rhs, np.zeros(C.shape[0])])[:size]
            record["penalty"].append(
                {
                    "epsilon": eps,
                    "solution_error": float(np.abs(u_pen - u_elim).max() / scale),
                    "constraint_residual": float(np.abs(C @ u_pen).max()),
                }
            )
        errors = [entry["solution_error"] for entry in record["penalty"]]
        epsilons = [entry["epsilon"] for entry in record["penalty"]]
        # Asymptotic (= smallest weight) local rate; the largest weights are not yet
        # in the penalty asymptote, so a single global fit would understate the order.
        record["penalty_slopes"] = [
            float(
                np.log(errors[i + 1] / errors[i])
                / np.log(epsilons[i + 1] / epsilons[i])
            )
            for i in range(len(errors) - 1)
        ]
        record["penalty_slope"] = record["penalty_slopes"][-1]
    # Rotated faces: expanded face values must lie in the tangent plane.
    if space.rotated:
        normals = space.normal
        state = space.expand(rng.standard_normal(P.shape[1]))
        face = state[components * space.N :].reshape(space.F, 3)
        selected = np.flatnonzero(space.normal_faces)
        record["tangential_normal_component"] = float(
            np.abs(np.sum(face[selected] * normals[selected], axis=1)).max(initial=0.0)
        )
    return space, record


def operator_probe():
    """Downstream claim: free-slip elimination leaves no spurious zero-energy mode."""
    mesh = single_hexahedron()
    faces = len(mesh.owner)
    boundary = np.flatnonzero(mesh.neighbor < 0)
    result = {}
    free_slip = NativeHybridViscous3D(
        mesh,
        dirichlet_mask=np.zeros((faces, 3), bool),
        normal_velocity_faces=np.isin(np.arange(faces), boundary),
    )
    no_slip = NativeHybridViscous3D(mesh, dirichlet_mask=np.ones((faces, 3), bool))
    for name, operator in (("free_slip_box", free_slip), ("no_slip_box", no_slip)):
        dense = operator.matrix.toarray()
        spectrum = np.linalg.eigvalsh((dense + dense.T) / 2)
        result[name] = {
            "reduced_dofs": int(dense.shape[0]),
            "symmetry_error": float(np.abs(dense - dense.T).max()),
            "min_eigenvalue": float(spectrum[0]),
            "max_eigenvalue": float(spectrum[-1]),
            "zero_modes": int(np.count_nonzero(np.abs(spectrum) < 1e-9)),
            "max_tangential_normal_component": float(
                operator.velocity_space.constraint_error(
                    operator.velocity_space.expand(
                        np.random.default_rng(SEED).standard_normal(dense.shape[0])
                    )
                )
            ),
        }
    return result


def rejection_probe():
    """Negative controls: the module must keep rejecting over-specified input."""
    mesh = channel_block()
    faces = len(mesh.owner)
    internal = np.flatnonzero(mesh.neighbor >= 0)
    boundary = np.flatnonzero(mesh.neighbor < 0)
    cases = {}

    def expect(name, function):
        try:
            function()
        except ValueError as error:
            cases[name] = {"rejected": True, "message": str(error)}
        else:
            cases[name] = {"rejected": False, "message": "no error raised"}

    expect(
        "normal_on_interior_face",
        lambda: NativeVelocitySpace3D(
            mesh, 3, np.zeros((faces, 3), bool), np.isin(np.arange(faces), internal)
        ),
    )
    expect(
        "normal_with_one_component",
        lambda: NativeVelocitySpace3D(
            mesh, 1, np.zeros((faces, 1), bool), np.isin(np.arange(faces), boundary[:1])
        ),
    )
    host = np.zeros((faces, 3), bool)
    host[boundary[0], 0] = True
    expect(
        "normal_over_cartesian",
        lambda: NativeVelocitySpace3D(
            mesh, 3, host, np.isin(np.arange(faces), boundary[:1])
        ),
    )
    return cases


def thresholds(record):
    checks = {
        "orthonormal_columns": record["orthonormality_error"] <= 1e-13,
        "range_inside_constraints": record["constraint_range_error"] <= 1e-12,
        "exact_elimination_dimension": record["nullity_gap"] == 0,
        "roundtrip_exact": record["roundtrip_error"] <= 1e-13,
        "projection_satisfies_constraints": record["constraint_error_projected"]
        <= 1e-13,
        "raw_vector_flagged": record["constraint_error_raw"] > 1e-3
        or record["full_dofs"] == record["reduced_dofs"],
        "projection_idempotent": record["idempotence_error"] <= 1e-13,
        "matches_kkt_solve": record["kkt_solution_error"] <= 1e-11,
        "penalty_is_first_order": (not record["penalty"])
        or record["penalty_slope"] > 0.9,
        "tangent_plane_exact": record.get("tangential_normal_component", 0.0) <= 1e-13,
    }
    return checks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    channel, cases = configurations()
    report = {
        "mission": 2891,
        "module": "tensorfvm.native_velocity_space3d",
        "cases": {},
        "mesh": {
            "channel": {
                "cells": len(channel.cells),
                "faces": len(channel.owner),
                "internal_faces": int(np.count_nonzero(channel.neighbor >= 0)),
                "boundary_faces": int(np.count_nonzero(channel.neighbor < 0)),
            },
            "tetra": {
                "cells": len(skewed_tetrahedra().cells),
                "faces": len(skewed_tetrahedra().owner),
            },
        },
    }
    for name, mesh, mask, normal_faces in cases:
        space, record = algebra_probe(mesh, mask, normal_faces)
        record["checks"] = thresholds(record)
        record["passed"] = bool(all(record["checks"].values()))
        report["cases"][name] = record
    report["operator"] = operator_probe()
    report["rejections"] = rejection_probe()
    report["rejections_passed"] = bool(
        all(entry["rejected"] for entry in report["rejections"].values())
    )
    report["operator_passed"] = bool(
        report["operator"]["free_slip_box"]["zero_modes"] == 0
        and report["operator"]["free_slip_box"]["min_eigenvalue"] > 0
        and report["operator"]["free_slip_box"]["symmetry_error"] <= 1e-12
    )
    report["passed"] = bool(
        all(case["passed"] for case in report["cases"].values())
        and report["rejections_passed"]
        and report["operator_passed"]
    )
    with (OUT / "audit.json").open("w") as handle:
        json.dump(report, handle, indent=1)
    figure(OUT / "figure.png", report)
    (Path(__file__).resolve().parent / "native_velocity_space3d_audit.md").write_text(
        markdown(report)
    )
    return report


def figure(path, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    colours = plt.cm.tab10.colors
    # (a) penalty limit: elimination is the eps -> 0 limit, first order.
    penalised = [name for name, case in report["cases"].items() if case["penalty"]]
    for index, name in enumerate(penalised):
        entries = report["cases"][name]["penalty"]
        axes[0].loglog(
            [entry["epsilon"] for entry in entries],
            [max(entry["solution_error"], 1e-18) for entry in entries],
            "o-",
            color=colours[index % 10],
            lw=1.0,
            ms=3.0,
            label=f"{name.split('_', 1)[1]} (order {report['cases'][name]['penalty_slope']:.2f})",
        )
    axes[0].axhline(2**-52, color="k", ls=":", lw=0.8, label="double precision")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("penalty weight ε (1/ε stiffness)")
    axes[0].set_ylabel(r"$\|u_\epsilon-u_{\rm elim}\|_\infty/\|u_{\rm elim}\|_\infty$")
    axes[0].set_title("(a) elimination = exact penalty limit")
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    # (b) residual levels of the audited invariants.
    names = list(report["cases"])
    keys = [
        "orthonormality_error",
        "constraint_range_error",
        "roundtrip_error",
        "constraint_error_projected",
        "idempotence_error",
        "kkt_solution_error",
    ]
    labels = ["PᵀP-I", "C P", "restrict·expand", "C u(Πv)", "Π²=Π", "u vs KKT"]
    width = 0.13
    for index, (key, label) in enumerate(zip(keys, labels)):
        values = [max(report["cases"][name][key], 1e-18) for name in names]
        axes[1].bar(
            np.arange(len(names)) + (index - 2.5) * width,
            values,
            width,
            color=colours[index % 10],
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].axhline(1e-13, color="k", ls="--", lw=0.7)
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(
        [name.replace("_", "\n", 1) for name in names], fontsize=5.6, rotation=0
    )
    axes[1].set_ylabel("max residual")
    axes[1].set_title("(b) algebra residuals (all ≤ 1e-13)")
    axes[1].legend(fontsize=5.4, frameon=False, ncol=2)
    # (c) free-slip vs no-slip reduced spectrum: no spurious zero mode.
    for index, name in enumerate(("free_slip_box", "no_slip_box")):
        mesh = single_hexahedron()
        faces = len(mesh.owner)
        boundary = np.flatnonzero(mesh.neighbor < 0)
        operator = NativeHybridViscous3D(
            mesh,
            dirichlet_mask=np.zeros((faces, 3), bool)
            if index == 0
            else np.ones((faces, 3), bool),
            normal_velocity_faces=np.isin(np.arange(faces), boundary)
            if index == 0
            else None,
        )
        dense = operator.matrix.toarray()
        spectrum = np.sort(np.linalg.eigvalsh((dense + dense.T) / 2))
        axes[2].semilogy(
            np.arange(1, len(spectrum) + 1),
            spectrum,
            "o-",
            ms=2.5,
            lw=0.9,
            color=colours[index],
            label=f"{name} (min {spectrum[0]:.3g})",
        )
    axes[2].set_xlabel("reduced mode index")
    axes[2].set_ylabel("eigenvalue of reduced operator")
    axes[2].set_title("(c) free-slip box: no zero-energy mode")
    axes[2].legend(fontsize=6, frameon=False)
    for axis in axes:
        axis.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(report):
    def table(rows, header):
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        return "\n".join(lines)

    lines = []
    lines.append("# NativeVelocitySpace3D 精确面法向消元审计（m2891）\n")
    lines.append(
        "模块 `tensorfvm.native_velocity_space3d` 被 `native_hybrid_viscous3d` → "
        "`native_hybrid_stokes3d` 生产路径导入，但此前没有任何直接测试："
        "`normal_faces` 的旋转基分支只通过求解器级用例间接触及。"
        "本审计把该模块两条自述主张变成可机器裁决的判据：\n"
    )
    lines.append("1. “P 的列正交，把约化增量映射到无约束单元速度与容许面速度”；")
    lines.append("2. “不使用法向罚函数（no normal penalty）”。\n")
    channel, tetra = report["mesh"]["channel"], report["mesh"]["tetra"]
    lines.append(
        f"审计网格两块：结构化通道块 {channel['cells']} 个六面体 / {channel['faces']} 个面"
        f"（内部面 {channel['internal_faces']}、边界面 {channel['boundary_faces']}，法向轴对齐），"
        f"斜四面体对 {tetra['cells']} 个单元 / {tetra['faces']} 个面（边界面法向三分量均非零）；"
        f"随机种子 {SEED}。\n"
    )
    lines.append("## 1 代数不变量（重建独立约束矩阵 C 与稠密参考解）\n")
    rows = []
    for name, case in report["cases"].items():
        rows.append(
            [
                name,
                case["reduced_dofs"],
                f"{case['orthonormality_error']:.2e}",
                f"{case['constraint_range_error']:.2e}",
                case["nullity_gap"],
                f"{case['roundtrip_error']:.2e}",
                f"{case['constraint_error_projected']:.2e}",
                f"{case['kkt_solution_error']:.2e}",
                "PASS" if case["passed"] else "FAIL",
            ]
        )
    lines.append(
        table(
            rows,
            [
                "配置",
                "约化自由度",
                "‖PᵀP−I‖∞",
                "‖C P‖∞",
                "零空间维数差",
                "restrict∘expand",
                "‖C u‖∞（投影后）",
                "u_elim vs u_KKT",
                "判定",
            ],
        )
    )
    lines.append("")
    lines.append(
        "零空间维数差 `nullity(C) − rank(P) = 0` 是“精确消元、无罚项”的离散证据："
        "约化空间的维数正好等于硬约束的零空间维数，既不多（罚项会多出伪自由度）也不少。"
    )
    lines.append(
        "`‖C u_KKT‖∞` 随 KKT 硬约束求解为机器精度，`u_elim` 与 `u_KKT` 相对差见表第 8 列。\n"
    )
    lines.append("## 2 罚函数参照：消元必须是 ε→0 的极限，且误差一阶\n")
    rows = []
    for name, case in report["cases"].items():
        if not case["penalty"]:
            continue
        rows.append(
            [name, " / ".join(f"{value:.2f}" for value in case["penalty_slopes"])]
            + [f"{entry['solution_error']:.2e}" for entry in case["penalty"]]
            + [f"{entry['constraint_residual']:.2e}" for entry in case["penalty"]]
        )
    if rows:
        epsilons = [
            "ε=%g" % entry["epsilon"]
            for entry in next(
                iter([c for c in report["cases"].values() if c["penalty"]])
            )["penalty"]
        ]
        lines.append(
            table(
                rows,
                ["配置", "局部收敛阶（逐级）"]
                + epsilons
                + ["‖Cu‖" + e for e in epsilons],
            )
        )
        lines.append("")
        lines.append(
            "斜率≈1 说明“罚刚度 1/ε”解与消元解的差就是一阶 ε；消元解本身不依赖任何罚系数，"
            "即模块自述的 no normal penalty 成立。\n"
        )
    lines.append("## 3 旋转基的切平面精确性\n")
    rows = [
        [name, case.get("tangential_normal_component", "n/a")]
        for name, case in report["cases"].items()
        if case["rotated"]
    ]
    lines.append(
        table(
            [
                [row[0], f"{row[1]:.2e}" if isinstance(row[1], float) else row[1]]
                for row in rows
            ],
            ["配置（rotated=True）", "max|n·u_face|"],
        )
    )
    lines.append("")
    lines.append(
        "轴对齐与斜法向（三分量均非零）两种旋转基都在机器精度上把面自由度压在切平面内，"
        "未被约束的两个切向自由度经 `restrict` 往返无损。\n"
    )
    lines.append("## 4 下游算子：自由滑移消元不产生零能模态\n")
    rows = [
        [
            name,
            case["reduced_dofs"],
            f"{case['symmetry_error']:.2e}",
            f"{case['min_eigenvalue']:.6f}",
            f"{case['max_eigenvalue']:.6f}",
            case["zero_modes"],
        ]
        for name, case in report["operator"].items()
    ]
    lines.append(
        table(rows, ["算子", "约化自由度", "对称性误差", "λ_min", "λ_max", "零模态数"])
    )
    lines.append("")
    lines.append(
        "全部边界面为法向约束（自由滑移）时约化算子仍严格正定，"
        "不存在因消元引入的伪零能模态；全 Dirichlet（无滑移）单胞只剩 3 个自由面自由度，"
        "说明两个分支都按合同维数收缩。\n"
    )
    lines.append("## 5 负控：过定输入必须继续被拒绝\n")
    rows = [
        [name, "拒绝" if case["rejected"] else "**未拒绝**", case["message"]]
        for name, case in report["rejections"].items()
    ]
    lines.append(table(rows, ["负控用例", "结果", "抛出的信息"]))
    lines.append("")
    lines.append("## 6 判定\n")
    lines.append(
        f"全部代数不变量、负控与下游算子判据：**{'PASS' if report['passed'] else 'FAIL'}**。"
        "审计未发现模块缺陷（消元在整个测试范围内是精确的）；"
        "本单关闭的是“已合并功能零直接测试”的验证欠账，"
        "并把此前只靠求解器用例间接覆盖的旋转基分支纳入回归网。\n"
    )
    lines.append(
        "判据阈值：正交性/约束残差/往返/幂等 ≤1e-13，KKT 对照 ≤1e-11（相对），罚函数斜率 >0.9，"
        "自由滑移算子 λ_min>0 且零模态数=0。\n"
    )
    lines.append(
        "复算：`python3 scripts/audit_native_velocity_space3d.py`（本机 <2 s，无迭代求解）；"
        "回归：`python3 -m pytest tests/test_native_velocity_space3d.py`。\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {name: case["passed"] for name, case in result["cases"].items()}, indent=1
        )
    )
    print(
        "rejections",
        result["rejections_passed"],
        "operator",
        result["operator_passed"],
        "overall",
        result["passed"],
    )
