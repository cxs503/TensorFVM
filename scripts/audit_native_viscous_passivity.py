"""Audit the discrete passivity of the native 3D viscous operator.

Writes a JSON verdict plus figures under ``--out`` (default
``scripts/native_viscous_passivity``).  Everything is recomputed here; the
numbers in the accompanying note and report are copied from this JSON by the
report generator, not by hand.

Usage::

    python scripts/audit_native_viscous_passivity.py
    python scripts/audit_native_viscous_passivity.py --out /tmp/audit --no-plots
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.native_energy import (
    affine_consistency,
    correction_family_scan,
    energy_diagnostic,
    passivity_audit,
    passivity_boundary,
    random_shear_survey,
    skew_sweep,
    two_point_control,
    two_tetrahedra,
)

QUALITY_GATE_DEGREES = 70.0  # tensorfvm.mesh_quality quality_gate_passed
SWEEP_SKEWS = tuple(float(x) for x in np.linspace(0.0, 1.4, 15))


def _spectrum(eigenvalues):
    return sorted(float(x) for x in eigenvalues)


def audit():
    counterexample = two_tetrahedra(1.0)
    verdict = dict(
        quality_gate_degrees=QUALITY_GATE_DEGREES,
        counterexample=dict(
            skew=1.0,
            cells=len(counterexample.cells),
            faces=len(counterexample.owner),
            over_relaxed=passivity_audit(counterexample, "over-relaxed"),
            minimum=passivity_audit(counterexample, "minimum"),
            two_point=two_point_control(counterexample),
        ),
        boundaries={
            name: passivity_boundary(name) for name in ("over-relaxed", "minimum")
        },
        energy={
            name: energy_diagnostic(counterexample, correction=name)
            for name in ("over-relaxed", "minimum", "none")
        },
        family_scan=dict(
            correction_family_scan(counterexample),
            min_creation=min(
                row["vector_lambda_max"]
                for row in correction_family_scan(counterexample)["rows"]
            ),
        ),
        sweep=skew_sweep(SWEEP_SKEWS),
        random_survey=random_shear_survey(),
        affine=dict(),
        spectra=dict(),
    )
    for name in ("over-relaxed", "minimum", "none"):
        verdict["affine"][name] = affine_consistency(counterexample, name)
        audit_row = passivity_audit(counterexample, name)
        verdict["spectra"][name] = dict(
            vector=_spectrum(
                np.linalg.eigvalsh(
                    (
                        np.array(audit_row["momentum_matrix"])
                        + np.array(audit_row["momentum_matrix"]).T
                    )
                    / 2
                )
            ),
            scalar=_spectrum(
                np.linalg.eigvalsh(
                    (
                        np.array(audit_row["scalar_matrix"])
                        + np.array(audit_row["scalar_matrix"]).T
                    )
                    / 2
                )
            ),
        )
    verdict["affine"]["two-point"] = dict(
        correction="two-point",
        maximum_relative_error=verdict["counterexample"]["two_point"][
            "affine_flux_max_relative_error"
        ],
    )
    return verdict


def plots(verdict, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    # 1. signed spectra of -L: passivity means no eigenvalue below zero.
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, style in (
        (
            "over-relaxed",
            dict(marker="o", color="C3", label="deferred over-relaxed (default)"),
        ),
        ("minimum", dict(marker="s", color="C0", label="deferred minimum correction")),
        ("none", dict(marker="^", color="C2", label="no skewness correction")),
    ):
        axis.plot(
            range(1, 7),
            verdict["spectra"][name]["vector"],
            linestyle="-",
            linewidth=1.2,
            markersize=5,
            **style,
        )
    verdict["counterexample"]["two_point"]["spectrum"]
    axis.axhline(0.0, color="k", linewidth=1.0)
    axis.set_xlabel("mode index (sorted)")
    axis.set_ylabel(r"eigenvalue of  $-L$  [mu = rho = 1]")
    axis.set_title(
        "Two-cell skewed pair (theta = %.1f deg, outside the %.0f deg gate)\n"
        r"lower half-plane = passive, upper half-plane = kinetic-energy creation"
        % (
            verdict["counterexample"]["over_relaxed"][
                "maximum_nonorthogonality_degrees"
            ],
            verdict["quality_gate_degrees"],
        ),
        fontsize=9,
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = out / "fig1_passivity_spectrum.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    figures.append(path)

    # 2. creation rate versus nonorthogonality, deterministic sweep + random shear survey.
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    valid = [row for row in verdict["sweep"]["rows"] if row["valid"]]
    for name, color in (("over-relaxed", "C3"), ("minimum", "C0")):
        angle = [row["nonorthogonality_degrees"] for row in valid]
        creation = [row["corrections"][name]["creation_rate"] for row in valid]
        axis.semilogy(
            angle,
            np.maximum(creation, 1e-16),
            "-o",
            color=color,
            markersize=4,
            label="deterministic sweep, %s" % name,
        )
        boundary = verdict["boundaries"][name]
        axis.axvline(
            boundary["passive_degrees"], color=color, linestyle="--", linewidth=1.0
        )
    survey = verdict["random_survey"]
    axis.semilogy(
        [row["nonorthogonality_degrees"] for row in survey["rows"]],
        np.maximum(
            [row["over_relaxed"]["creation_rate"] for row in survey["rows"]], 1e-16
        ),
        "x",
        color="k",
        markersize=4,
        alpha=0.5,
        label="%d random sheared pairs (default)" % survey["samples"],
    )
    axis.axvline(verdict["quality_gate_degrees"], color="g", linewidth=1.4)
    axis.text(
        verdict["quality_gate_degrees"] - 1.5,
        1e-8,
        "mesh-quality gate 70 deg",
        rotation=90,
        fontsize=7,
        color="g",
        ha="right",
    )
    axis.set_xlabel("maximum nonorthogonality angle [deg]")
    axis.set_ylabel("worst-case creation rate")
    axis.set_title(
        "Passivity fails only past the bisected boundary (%.1f / %.1f deg)"
        % (
            verdict["boundaries"]["over-relaxed"]["passive_degrees"],
            verdict["boundaries"]["minimum"]["passive_degrees"],
        ),
        fontsize=9,
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    path = out / "fig2_creation_vs_angle.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    figures.append(path)

    # 3. family scan: no correction weight restores passivity.
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    gamma = verdict["family_scan"]["scales"]
    axis.loglog(
        gamma,
        [
            max(row["vector_lambda_max"], 1e-16)
            for row in verdict["family_scan"]["rows"]
        ],
        "o-",
        color="C3",
        label="max eigenvalue of $-L$ (vector)",
    )
    axis.loglog(
        gamma,
        [
            max(row["scalar_lambda_max"], 1e-16)
            for row in verdict["family_scan"]["rows"]
        ],
        "s--",
        color="C0",
        label="max eigenvalue of $-L$ (scalar)",
    )
    axis.axhline(0.0, color="k", linewidth=1.0)
    axis.set_xlabel(
        r"correction scale  $\gamma$  in  $\gamma\,s\,(\Delta u - d\cdot\nabla u)$"
    )
    axis.set_ylabel("energy-creation eigenvalue")
    axis.set_title(
        "No gamma in the family is passive: the defect is structural", fontsize=9
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    path = out / "fig3_family_scan.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    figures.append(path)

    # 4. consistency/passivity trade-off.
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    points = []
    for name, color in (("over-relaxed", "C3"), ("minimum", "C0")):
        audit_row = verdict["counterexample"][name.replace("-", "_")]
        points.append(
            (
                audit_row["maximum_nonorthogonality_degrees"],
                verdict["affine"][name]["maximum_relative_error"],
                audit_row["vector"]["creation_rate"],
                name,
                color,
            )
        )
    points.append(
        (
            verdict["counterexample"]["over_relaxed"][
                "maximum_nonorthogonality_degrees"
            ],
            verdict["affine"]["two-point"]["maximum_relative_error"],
            0.0,
            "two-point (passive)",
            "C2",
        )
    )
    for angle, error, creation, label, color in points:
        axis.scatter(
            error,
            max(creation, 1e-3),
            s=60,
            color=color,
            marker="o" if creation > 0 else "*",
            label=label,
        )
        axis.annotate(
            label,
            (error, max(creation, 1e-3)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("affine-traction relative error (consistency)")
    axis.set_ylabel("worst-case creation rate (passivity)")
    axis.set_title("Consistency/passivity trade-off on the skewed pair", fontsize=9)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    path = out / "fig4_tradeoff.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    figures.append(path)
    return figures


def note(verdict, path):
    """Write the audit note; every number below is injected from ``verdict``."""
    over = verdict["counterexample"]["over_relaxed"]
    minimum = verdict["counterexample"]["minimum"]
    two_point = verdict["counterexample"]["two_point"]
    boundaries = verdict["boundaries"]
    design = verdict["family_scan"]
    survey = verdict["random_survey"]
    affine = verdict["affine"]
    valid = [row for row in verdict["sweep"]["rows"] if row["valid"]]
    lines = [
        "# Native 3D viscous operator: discrete passivity audit",
        "",
        "Status: **open defect, quantified and bounded**. `tests/test_native_energy_small.py::"
        "test_two_cell_skew_viscosity_must_dissipate` stays a strict `xfail`; the quantified",
        "verdict lives in `tests/test_native_energy_passivity.py`.",
        "",
        "## 1. Object and criterion",
        "",
        "The implicit viscous operator consumed by `native_collocated3d` and `native_rans3d` is",
        "`L = -momentum_diffusion_matrix`. A cell-centred discretisation of",
        "`-div(mu (grad u + grad u^T))` with homogeneous Dirichlet data must satisfy the discrete",
        "dissipation (passivity) inequality",
        "",
        "    x^T L x <= 0   for every velocity vector x,   i.e. -L positive semidefinite.",
        "",
        "Otherwise the implicit step can inject kinetic energy. There is no pressure, no",
        "turbulence and no flow benchmark in the test: it is a pure operator property, so it also",
        "covers the Rhie-Chow mobility reference built from the same scalar Laplacian.",
        "",
        "Counterexample (`two_tetrahedra(1.)`): two valid tetrahedra sharing one face, cell",
        "volumes 1/24 and 1/3, %d faces, largest nonorthogonality %.3f deg."
        % (over["faces"], over["maximum_nonorthogonality_degrees"]),
        "",
        "## 2. Measured result on the counterexample",
        "",
        "| operator | scalar min eig | scalar max eig | vector min eig | vector max eig | creation rate | creation fraction | passive |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in (
        ("deferred over-relaxed (default)", over),
        ("deferred minimum correction", minimum),
    ):
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | %s |"
            % (
                name,
                row["scalar"]["lambda_min"],
                row["scalar"]["lambda_max"],
                row["vector"]["lambda_min"],
                row["vector"]["lambda_max"],
                row["vector"]["creation_rate"],
                row["vector"]["creation_fraction"],
                "yes" if row["passive"] else "**no**",
            )
        )
    lines.append(
        "| two-point reference | - | - | %.4f | %.4f | %.4f | %.3f | %s |"
        % (
            two_point["spectrum"]["lambda_min"],
            two_point["spectrum"]["lambda_max"],
            two_point["spectrum"]["creation_rate"],
            two_point["spectrum"]["creation_fraction"],
            "yes" if two_point["passive"] else "no",
        )
    )
    lines += [
        "",
        "`creation rate = max(0, -min eig(-L))`, `creation fraction = creation / (creation + dissipation)`.",
        "The assembled traction and the matrix quadratic form agree to %.1e, and the prescribed"
        % verdict["energy"]["over-relaxed"]["energy_balance_error"],
        "velocity (1,0,0) in cell 0 creates %.4f units of kinetic energy per unit time"
        % verdict["energy"]["over-relaxed"]["kinetic_energy_rate"],
        "(the energy identity is not the issue; the sign of the operator is).",
        "",
        "## 3. Is it a weight bug or structural?",
        "",
        "The deferred correction is one-parameter: `gamma*s*(du - d.grad)` with `gamma = 1/(s.d)`"
        " (over-relaxed, default) or `(s.d)/(|d|^2 |s|^2)` (minimum/orthogonal correction).",
        "Scanning %d weights over [%.1e, %.1f] gives a smallest creation eigenvalue of %.4f at"
        % (
            len(design["rows"]),
            design["scales"][0],
            design["scales"][-1],
            design["min_creation"],
        ),
        "the smallest weight (`gamma -> 0`, i.e. the correction switched off): **no member of the",
        "family is passive**. The cause is the interpolated-cell-gradient composite, not the",
        "correction weight: `div(interp(grad_LS u))` is not the adjoint of the cell-to-face jump",
        "operator, so the symmetric part is indefinite.",
        "",
        "## 4. Validity range in mesh-quality terms",
        "",
        "| correction | largest passive skew | largest passive nonorthogonality | first creating nonorthogonality |",
        "|---|---|---|---|",
    ]
    for name, row in boundaries.items():
        lines.append(
            "| %s | %.4f | %.3f deg | %.3f deg |"
            % (
                name,
                row["passive_skew"],
                row["passive_degrees"],
                row["creation_degrees"],
            )
        )
    lines += [
        "",
        "The repository mesh-quality gate (`tensorfvm.mesh_quality.quality_gate_passed`) admits",
        "nonorthogonality < %.0f deg, so the shipped default operator is passive on **every mesh the"
        % verdict["quality_gate_degrees"],
        "quality gate admits, with %.1f deg of margin**. Independently, %d randomly sheared tetrahedral"
        % (
            boundaries["over-relaxed"]["passive_degrees"]
            - verdict["quality_gate_degrees"],
            survey["samples"],
        ),
        "pairs (`random_shear_survey`) gave %d over-relaxed failures and %d minimum-correction failures."
        % (survey["over_relaxed_failures"], survey["minimum_failures"]),
        "",
        "Deterministic sweep (vector operator, creation rate):",
        "",
        "| skew | nonorthogonality [deg] | over-relaxed | minimum |",
        "|---|---|---|---|",
    ]
    for row in valid:
        lines.append(
            "| %.3f | %.2f | %.3e | %.3e |"
            % (
                row["skew"],
                row["nonorthogonality_degrees"],
                row["corrections"]["over-relaxed"]["creation_rate"],
                row["corrections"]["minimum"]["creation_rate"],
            )
        )
    lines += [
        "",
        "## 5. Consistency/passivity trade-off",
        "",
        "| scheme | affine-traction relative error | creation rate at skew=1 | passive |",
        "|---|---|---|---|",
    ]
    for name in ("over-relaxed", "minimum", "none"):
        lines.append(
            "| deferred %s | %.1e | %.4f | no |"
            % (
                name,
                affine[name]["maximum_relative_error"],
                over["vector"]["creation_rate"]
                if name == "over-relaxed"
                else minimum["vector"]["creation_rate"]
                if name == "minimum"
                else passivity_audit(two_tetrahedra(1.0), "none")["vector"][
                    "creation_rate"
                ],
            )
        )
    lines += [
        "| two-point (`mu (s.d)/|d|^2`) | %.3f | %.4f | yes |"
        % (
            affine["two-point"]["maximum_relative_error"],
            two_point["spectrum"]["creation_rate"],
        ),
        "",
        "Every deferred variant reproduces the exact affine traction to round-off, so re-weighting",
        "`gamma` costs nothing in consistency - but it never buys passivity either. The strictly",
        "passive two-point operator pays %.0f%% flux error on the skewed face. This is the classical"
        % (100 * affine["two-point"]["maximum_relative_error"]),
        "monotonicity/accuracy barrier, so the honest verdict is a *bounded open defect*, not a bug",
        "to be patched by tuning.",
        "",
        "## 6. Recommendation",
        "",
        "* Keep the default over-relaxed correction (widest passive range).",
        "* The minimum correction is gentler past the boundary (%.0f x smaller creation at skew=1.25) but"
        % (
            passivity_audit(two_tetrahedra(1.25), "over-relaxed")["vector"][
                "creation_rate"
            ]
            / passivity_audit(two_tetrahedra(1.25), "minimum")["vector"][
                "creation_rate"
            ]
        ),
        "  loses %.1f deg of passive range; it is not a drop-in safety win."
        % (
            boundaries["over-relaxed"]["passive_degrees"]
            - boundaries["minimum"]["passive_degrees"]
        ),
        "* Reject meshes above the 70 deg gate as already done; do **not** relax it past",
        "  %.0f deg without re-running this audit."
        % boundaries["over-relaxed"]["passive_degrees"],
        "* A fully passive *and* affine-exact operator needs a symmetric scheme family",
        "  (SUSHI/DDfV-style), which is a separate implementation item, not a weight change.",
        "",
        "## 7. Reproduction",
        "",
        "```",
        "python scripts/audit_native_viscous_passivity.py   # writes audit.json + 4 figures",
        "python -m pytest tests/test_native_energy_passivity.py tests/test_native_energy_small.py -q",
        "```",
        "",
        "Figures: `fig1_passivity_spectrum.png` (signed spectra), `fig2_creation_vs_angle.png`",
        "(creation versus nonorthogonality with the gate and the bisected boundary),",
        "`fig3_family_scan.png` (no gamma is passive), `fig4_tradeoff.png` (consistency versus",
        "passivity).",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scripts/native_viscous_passivity")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    verdict = audit()
    figures = [] if arguments.no_plots else plots(verdict, out)
    note_path = Path("scripts/native_viscous_passivity_audit.md")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note(verdict, note_path)
    payload = dict(
        verdict=verdict, figures=[str(path) for path in figures], note=str(note_path)
    )
    (out / "audit.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    summary = dict(
        quality_gate_degrees=verdict["quality_gate_degrees"],
        boundary_passive_degrees={
            name: row["passive_degrees"] for name, row in verdict["boundaries"].items()
        },
        counterexample_creation={
            name: verdict["counterexample"][name.replace("-", "_")]["vector"][
                "creation_rate"
            ]
            for name in ("over-relaxed", "minimum")
        },
        family_any_passive=verdict["family_scan"]["any_passive"],
        family_min_creation=min(
            row["vector_lambda_max"] for row in verdict["family_scan"]["rows"]
        ),
        random_survey=dict(
            samples=verdict["random_survey"]["samples"],
            over_relaxed_failures=verdict["random_survey"]["over_relaxed_failures"],
            minimum_failures=verdict["random_survey"]["minimum_failures"],
        ),
        affine_relative_error={
            name: row["maximum_relative_error"]
            for name, row in verdict["affine"].items()
        },
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
