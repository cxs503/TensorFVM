"""Verification audit of the viscous momentum assemblies: sign convention, scaling, defect boundary.

Three machine-checkable questions, none of which the merged dissipativity audit
(``audit_compatible_viscous_dissipativity.py``) answered:

A. Sign convention.  ``NativeViscous3D.momentum_diffusion_matrix`` (``D A G``) is
   the operator slot of the production solvers: ``u^T (D A G) u <= 0`` is the
   dissipative sign.  ``compatible_momentum_operators['matrix']`` (``G^T 2 mu V G``)
   is PSD, i.e. ``u^T K u >= 0``.  The two assemblies therefore live in *opposite*
   sign conventions even though the compatible docstring claims the momentum matrix
   is returned "same sign convention as NativeViscous3D.momentum_diffusion_matrix".

B. Absolute scaling (the follow-up explicitly reserved by
   ``tests/test_native_compatible_viscous3d.py``).  Is the quadratic form of the
   compatible operator the volume-weighted discrete strain norm?
   ``u^T K u = 2 mu sum_c V_c dev(S_h(c)):dev(S_h(c))``.

C. Boundary of the open legacy defect.  Where does the production ``D A G`` stop
   being dissipative (``lambda_max(sym M) > 0``)?

D. Residual order of the viscous operator against ``-div(2 mu S)`` for a smooth
   solenoidal field, with a scalar-diffusion control that isolates the one-sided
   boundary closure.

Outputs (default ``scripts/verification-native-compatible-sign``):
``sweep.csv``, ``sweep.json``, ``report-table.md``, ``sign-sweep.png``,
``energy-identity.png``, ``residual-order.png``.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tensorfvm.native_compatible_viscous3d import (
    compatible_momentum_operators,
    compatible_momentum_matrix_in_legacy_sign,
)
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra

PI = np.pi


def sheared_cube(n=2, shear=0.18, tilt=0.12):
    """Affine-sheared hexahedral cube (identical to the merged dissipativity audit)."""
    from numpy import array, linspace

    from tensorfvm.native_mesh3d import NativeMesh3D

    points = array(
        [
            [x, y, z]
            for z in linspace(0, 1, n + 1)
            for y in linspace(0, 1, n + 1)
            for x in linspace(0, 1, n + 1)
        ]
    )
    points = points @ array([[1, shear, tilt], [0, 1, 0.2], [0, 0, 1]]).T

    def index(x, y, z):
        return z * (n + 1) ** 2 + y * (n + 1) + x

    cells = [
        [
            index(x, y, z),
            index(x + 1, y, z),
            index(x + 1, y + 1, z),
            index(x, y + 1, z),
            index(x, y, z + 1),
            index(x + 1, y, z + 1),
            index(x + 1, y + 1, z + 1),
            index(x, y + 1, z + 1),
        ]
        for z in range(n)
        for y in range(n)
        for x in range(n)
    ]
    return NativeMesh3D.from_cells(
        points, cells, ["hexahedron"] * len(cells), triangulate_faces=True
    )


def symmetric_eigenvalue_bounds(matrix):
    dense = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (dense + dense.T))
    return float(eigenvalues[0]), float(eigenvalues[-1])


def strain_norm(velocity, base, deviatoric=True, dynamic_viscosity=1.0):
    """Volume-weighted discrete strain energy ``2 mu sum_c V_c S:S`` of the matrix part.

    Boundary data are zero here on purpose: the matrices act on interior degrees of
    freedom only, so the matching strain field is the homogeneous-Dirichlet one.
    """
    gradient = np.stack(
        [operator @ velocity for operator in base.gradient_matrices], axis=2
    )
    strain = 0.5 * (gradient + gradient.transpose(0, 2, 1))
    if deviatoric:
        strain = (
            strain - np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
        )
    return (
        2
        * dynamic_viscosity
        * float(
            np.sum(base.mesh.cell_volumes * np.einsum("cij,cij->c", strain, strain))
        )
    )


def scaling_row(mesh, velocity, mu=1.0):
    mask = mesh.neighbor < 0
    base = NativeViscous3D(mesh, mask, mu)
    operators = compatible_momentum_operators(mesh, mask, mu)
    quadratic = float(velocity.ravel() @ (operators["matrix"] @ velocity.ravel()))
    reference = strain_norm(velocity, base, operators["deviatoric"], mu)
    return dict(
        quadratic_form=quadratic,
        strain_reference=reference,
        relative_error=abs(quadratic - reference) / abs(reference),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="scripts/verification-native-compatible-sign")
    parser.add_argument("--skew-points", type=int, default=21)
    parser.add_argument("--refinements", default="2,4,8,16")
    arguments = parser.parse_args()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261103)
    rows = []

    # A + C: eigen-bound sweep over the two-tetrahedron family and the sheared cube family.
    for skew in np.linspace(0, 1, arguments.skew_points):
        mesh = two_tetrahedra(float(skew))
        mask = mesh.neighbor < 0
        legacy = NativeViscous3D(mesh, mask, 1.0)
        operators = compatible_momentum_operators(mesh, mask)
        low, high = symmetric_eigenvalue_bounds(
            legacy.momentum_diffusion_matrix.toarray()
        )
        p_low, p_high = symmetric_eigenvalue_bounds(operators["matrix"].toarray())
        cosine = np.sum(mesh.area_vectors * legacy.op.d, axis=1) / (
            np.linalg.norm(mesh.area_vectors, axis=1)
            * np.linalg.norm(legacy.op.d, axis=1)
        )
        velocity = rng.normal(size=(len(mesh.cells), 3))
        rows.append(
            dict(
                family="two_tetrahedra",
                parameter=float(skew),
                max_nonorthogonality_degrees=float(
                    np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()
                ),
                legacy_lambda_min=low,
                legacy_lambda_max=high,
                compatible_lambda_min=p_low,
                compatible_lambda_max=p_high,
                legacy_form=float(
                    velocity.ravel()
                    @ (legacy.momentum_diffusion_matrix @ velocity.ravel())
                ),
                compatible_form=float(
                    velocity.ravel() @ (operators["matrix"] @ velocity.ravel())
                ),
            )
        )
    for shear in (0.0, 0.18, 0.35, 0.5):
        for n in (2, 4):
            mesh = sheared_cube(n, shear=shear)
            mask = mesh.neighbor < 0
            legacy = NativeViscous3D(mesh, mask, 1.0)
            operators = compatible_momentum_operators(mesh, mask)
            low, high = symmetric_eigenvalue_bounds(
                legacy.momentum_diffusion_matrix.toarray()
            )
            p_low, p_high = symmetric_eigenvalue_bounds(operators["matrix"].toarray())
            velocity = rng.normal(size=(len(mesh.cells), 3))
            rows.append(
                dict(
                    family="sheared_cube_shear_%.2f_n%d" % (shear, n),
                    parameter=float(shear),
                    max_nonorthogonality_degrees=float("nan"),
                    legacy_lambda_min=low,
                    legacy_lambda_max=high,
                    compatible_lambda_min=p_low,
                    compatible_lambda_max=p_high,
                    legacy_form=float(
                        velocity.ravel()
                        @ (legacy.momentum_diffusion_matrix @ velocity.ravel())
                    ),
                    compatible_form=float(
                        velocity.ravel() @ (operators["matrix"] @ velocity.ravel())
                    ),
                )
            )
    (out / "sweep.json").write_text(json.dumps(rows, indent=1))
    with (out / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A: the sign-convention statement, on a random field, both families.
    sign_rows = []
    for label, mesh in (
        ("two_tetrahedra_skew_0.0", two_tetrahedra(0.0)),
        ("two_tetrahedra_skew_1.0", two_tetrahedra(1.0)),
        ("sheared_cube_n2", sheared_cube(2)),
        ("sheared_cube_n4_shear_0.35", sheared_cube(4, shear=0.35)),
    ):
        mask = mesh.neighbor < 0
        legacy = NativeViscous3D(mesh, mask, 1.0)
        operators = compatible_momentum_operators(mesh, mask)
        flipped = compatible_momentum_matrix_in_legacy_sign(operators)
        velocity = rng.normal(size=(len(mesh.cells), 3))
        flat = velocity.ravel()
        good = bad = 0
        for _ in range(64):
            probe = rng.normal(size=flat.shape)
            legacy_form = float(probe @ (legacy.momentum_diffusion_matrix @ probe))
            compatible_form = float(probe @ (operators["matrix"] @ probe))
            flipped_form = float(probe @ (flipped @ probe))
            good += int(legacy_form <= 0 <= compatible_form)
            bad += int(-legacy_form <= 0 <= -compatible_form)
            sign_rows.append(
                dict(
                    mesh=label,
                    legacy_form=legacy_form,
                    compatible_form=compatible_form,
                    flipped_form=flipped_form,
                    opposite_signs=bool(legacy_form <= 0 <= compatible_form),
                )
            )
        del velocity
        print(
            "%-28s random probes with legacy<=0<=compatible: %d/64 ; "
            "compatible<=0<=legacy: %d/64" % (label, good, bad)
        )

    # B: absolute scaling identity.
    identity_rows = []
    for mu in (1.0, 2.5):
        for label, mesh in (
            ("two_tetrahedra_skew_0.0", two_tetrahedra(0.0)),
            ("two_tetrahedra_skew_1.0", two_tetrahedra(1.0)),
            ("sheared_cube_n2", sheared_cube(2)),
            ("sheared_cube_n4_shear_0.35", sheared_cube(4, shear=0.35)),
        ):
            velocity = rng.normal(size=(len(mesh.cells), 3))
            row = scaling_row(mesh, velocity, mu)
            row.update(mesh=label, dynamic_viscosity=mu)
            identity_rows.append(row)

    # D: residual against -div(2 mu S) plus the scalar-diffusion control.
    refinements = [int(value) for value in arguments.refinements.split(",")]

    def field(x):
        return np.stack(
            [
                np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1]),  # noqa: E731
                -np.cos(PI * x[:, 0]) * np.sin(PI * x[:, 1]),
                np.zeros(len(x)),
            ],
            axis=1,
        )

    laplacian = lambda x: -2 * PI**2 * field(x)  # noqa: E731
    scalar = lambda x: np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1])  # noqa: E731
    residual_rows = []
    for n in refinements:
        mesh = sheared_cube(n)
        cells = len(mesh.cells)
        faces = len(mesh.owner)
        centers = mesh.cell_centers
        legacy = NativeViscous3D(mesh, mesh.neighbor < 0, 1.0)
        velocity = field(centers)
        exact = laplacian(centers)
        relative = np.linalg.norm(exact)
        source = (
            legacy.vector_divergence_matrix
            @ legacy.momentum_flux_source(
                np.zeros((faces, 3)), field(mesh.face_centers)
            ).ravel()
        )
        momentum = (
            (legacy.momentum_diffusion_matrix @ velocity.ravel() + source).reshape(
                -1, 3
            )
        ) / mesh.cell_volumes[:, None]
        phi = scalar(centers)
        scalar_residual = (
            legacy.scalar_diffusion_matrix @ phi
            + legacy.divergence_matrix
            @ legacy.scalar_flux_source(np.zeros(faces), scalar(mesh.face_centers))
        ) / mesh.cell_volumes
        row = dict(
            cells=cells,
            h=1.0 / n,
            momentum_relative_error=float(np.linalg.norm(momentum - exact) / relative),
            scalar_relative_error=float(
                np.linalg.norm(scalar_residual - (-2 * PI**2 * phi))
                / np.linalg.norm(-2 * PI**2 * phi)
            ),
        )
        if n <= 8:
            operators = compatible_momentum_operators(mesh, mesh.neighbor < 0)
            from tensorfvm.native_compatible_viscous3d import compatible_boundary_source

            compatible_source = compatible_boundary_source(
                operators, np.zeros((faces, 3)), field(mesh.face_centers)
            )
            compatible_residual = (
                -(
                    (
                        operators["matrix"] @ velocity.ravel() + compatible_source
                    ).reshape(-1, 3)
                )
                / mesh.cell_volumes[:, None]
            )
            row["compatible_relative_error"] = float(
                np.linalg.norm(compatible_residual - exact) / relative
            )
        else:
            row["compatible_relative_error"] = float("nan")
        residual_rows.append(row)

    # Table + figures.
    lines = [
        "| family | parameter | max non-orthogonality deg | legacy lambda_min | legacy lambda_max "
        "| compatible lambda_min | compatible lambda_max | legacy form | compatible form |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.3f} | {:.2f} | {:+.4e} | {:+.4e} | {:+.4e} | {:+.4e} | {:+.4e} | {:+.4e} |".format(
                row["family"],
                row["parameter"],
                row["max_nonorthogonality_degrees"],
                row["legacy_lambda_min"],
                row["legacy_lambda_max"],
                row["compatible_lambda_min"],
                row["compatible_lambda_max"],
                row["legacy_form"],
                row["compatible_form"],
            )
        )
    lines += [
        "",
        "| mesh | mu | u^T K u | 2 mu sum V dev(S_h):dev(S_h) | relative error |",
        "|---|---|---|---|---|",
    ]
    for row in identity_rows:
        lines.append(
            "| {} | {:.1f} | {:+.10e} | {:+.10e} | {:.2e} |".format(
                row["mesh"],
                row["dynamic_viscosity"],
                row["quadratic_form"],
                row["strain_reference"],
                row["relative_error"],
            )
        )
    lines += [
        "",
        "| cells | h | momentum residual rel. err | scalar residual rel. err | compatible residual rel. err |",
        "|---|---|---|---|---|",
    ]
    for row in residual_rows:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["cells"],
                row["h"],
                row["momentum_relative_error"],
                row["scalar_relative_error"],
                row["compatible_relative_error"],
            )
        )
    (out / "report-table.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tetra = [row for row in rows if row["family"] == "two_tetrahedra"]
    axes[0].plot(
        [row["parameter"] for row in tetra],
        [row["legacy_lambda_max"] for row in tetra],
        "o-",
        label="legacy lambda_max(sym M)",
    )
    axes[0].plot(
        [row["parameter"] for row in tetra],
        [row["compatible_lambda_min"] for row in tetra],
        "s-",
        label="compatible lambda_min(K)",
    )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("two-tetrahedron skew parameter")
    axes[0].set_ylabel("symmetric-part eigenvalue")
    axes[0].set_title("Positivity: legacy indefinite, compatible PSD")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        [row["parameter"] for row in tetra],
        [row["legacy_form"] / max(abs(row["compatible_form"]), 1e-30) for row in tetra],
        "o-",
        color="tab:red",
    )
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("two-tetrahedron skew parameter")
    axes[1].set_ylabel("u^T M u / |u^T K u|")
    axes[1].set_title("Sign convention of the same random field")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "sign-sweep.png", dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(6, 4.2))
    scales = [abs(row["strain_reference"]) for row in identity_rows]
    axes.loglog(
        scales,
        [abs(row["quadratic_form"]) for row in identity_rows],
        "o",
        label="|u^T K u|",
    )
    limits = [min(scales) * 0.5, max(scales) * 2]
    axes.plot(limits, limits, "k--", lw=0.8, label="identity")
    axes.set_xlabel("2 mu sum_c V_c dev(S_h):dev(S_h)")
    axes.set_ylabel("u^T K u")
    axes.set_title(
        "Absolute scaling identity (relative error <= %.1e)"
        % max(row["relative_error"] for row in identity_rows)
    )
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(out / "energy-identity.png", dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(6, 4.2))
    axes.loglog(
        [row["h"] for row in residual_rows],
        [row["momentum_relative_error"] for row in residual_rows],
        "o-",
        label="momentum residual (legacy)",
    )
    axes.loglog(
        [row["h"] for row in residual_rows],
        [row["scalar_relative_error"] for row in residual_rows],
        "s-",
        label="scalar residual (control)",
    )
    finite = [
        (row["h"], row["compatible_relative_error"])
        for row in residual_rows
        if np.isfinite(row["compatible_relative_error"])
    ]
    axes.loglog(
        [h for h, _ in finite],
        [value for _, value in finite],
        "^-",
        label="momentum residual (compatible, sign corrected)",
    )
    axes.set_xlabel("h")
    axes.set_ylabel("relative L2 residual")
    axes.set_title("Residual order against -div(2 mu S)")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, which="both")
    axes.invert_xaxis()
    figure.tight_layout()
    figure.savefig(out / "residual-order.png", dpi=130)
    plt.close(figure)

    print(
        json.dumps(
            dict(
                rows=len(rows),
                identity=identity_rows,
                residual=residual_rows,
                sign_probes=len(sign_rows),
            ),
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
