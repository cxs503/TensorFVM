"""Ethier--Steinman 3-D manufactured-solution verification study.

Runs the frozen-time steady problem of
:mod:`tensorfvm.verification.ethier_steinman` on a grid ladder for the two
prescribed conditions (Re = 1 and Re = 100), writes ``study.json``,
``RESULTS.md`` and the per-case field artefacts next to this script.

    PYTHONPATH=src python3 scripts/ethier-steinman-3d/run_study.py \
        --n-list 8,12,16 --cases re1,re100 --time 0.25

The second condition is started from the converged field of the first one at
the same resolution (viscosity continuation); a failed or unconverged solve is
recorded as an honest failure and never contributes a metric.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from tensorfvm.verification.ethier_steinman import (
    EthierSteinmanCase,
    es_metrics,
    es_solve,
    es_solver,
    observed_order,
    reference_audit,
)

HERE = Path(__file__).resolve().parent
# nu = d^2/a^2 is the reference relation; Re = 1/nu follows from it, while the
# exponent a only sets the wavenumber.  re100 is posed at the reference scale
# (a = 1) and re100_resolved at the wavenumber whose *cell* Reynolds number
# a*h/nu is resolvable by a second-order central scheme.
CASES = {
    "re1": EthierSteinmanCase(a=1.0, d=1.0, viscosity=1.0, half_width=1.0),
    "re100": EthierSteinmanCase(a=1.0, d=0.1, viscosity=0.01, half_width=1.0),
    "re100_resolved": EthierSteinmanCase(a=0.2, d=0.02, viscosity=0.01, half_width=1.0),
}
GATE = 0.01  # velocity L_inf relative error requested by the case sheet
OMEGA_GATE = 1e-12  # discrete divergence of the accepted face flux (1/s)


def solve_case(name: str, n: int, *, time_s: float, convection_order: int,
               tolerance: int, max_iterations: int, continuation=None):
    """Solve one (case, resolution) and return (record, state) with state for continuation."""
    case = CASES[name]
    started = time.perf_counter()
    mesh, solver, exact = es_solver(
        n, case, time_s=time_s, convection_order=convection_order
    )
    initialised_from = "zero"
    if continuation is not None:
        solver.state = continuation[0].copy()
        solver.p = continuation[1].copy()
        initialised_from = "previous_case"
    try:
        result = solver.solve_newton(max_iterations=max_iterations, tolerance=tolerance)
    except Exception:  # pragma: no cover - recorded, never masked
        return (
            dict(
                case=name,
                n=int(n),
                reynolds_number=float(case.reynolds_number),
                viscosity=float(case.viscosity),
                converged=False,
                failure_stage="newton_raised",
                failure_reason=traceback.format_exc().strip().splitlines()[-1],
                elapsed_seconds=time.perf_counter() - started,
            ),
            None,
        )
    if not result["numerically_converged"]:
        return (
            dict(
                case=name,
                n=int(n),
                reynolds_number=float(case.reynolds_number),
                viscosity=float(case.viscosity),
                converged=False,
                failure_stage="newton_not_converged",
                failure_reason=str(result.get("newton_failure")),
                initialised_from=initialised_from,
                elapsed_seconds=time.perf_counter() - started,
            ),
            None,
        )
    metrics = es_metrics(mesh, result, exact)
    metrics.update(
        case=name,
        n=int(n),
        resolution=f"{n}x{n}x{n}",
        cells=int(len(mesh.cells)),
        spacing_m=float(2.0 * case.half_width / n),
        time_s=float(time_s),
        reynolds_number=float(case.reynolds_number),
        viscosity=float(case.viscosity),
        a=float(case.a),
        d=float(case.d),
        is_unforced=bool(case.is_unforced),
        converged=True,
        initialised_from=initialised_from,
        newton_iterations=int(len(result.get("newton_history") or [])),
        velocity_linf_gate_passed=bool(metrics["velocity_relative_linf"] < GATE),
        divergence_gate_passed=bool(metrics["max_divergence_s_inv"] < OMEGA_GATE),
        elapsed_seconds=time.perf_counter() - started,
    )
    return metrics, (solver.state.copy(), solver.p.copy(), mesh, result, exact)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-list", default="8,12")
    parser.add_argument("--cases", default="re1,re100")
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--convection-order", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else HERE
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    n_list = [int(item) for item in args.n_list.split(",") if item.strip()]
    names = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = set(names) - set(CASES)
    if unknown:
        raise SystemExit(f"unknown cases: {sorted(unknown)}")

    audit = {name: reference_audit(CASES[name]) for name in names}
    records = []
    for n in n_list:
        continuation = None
        for name in names:
            record, state = solve_case(
                name,
                n,
                time_s=args.time,
                convection_order=args.convection_order,
                tolerance=args.tolerance,
                max_iterations=args.max_iterations,
                continuation=continuation,
            )
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)
            if state is None:
                continuation = None
                continue
            face_state, pressure, mesh, result, exact = state
            continuation = (face_state, pressure)
            if not args.skip_artifacts:
                np.savez_compressed(
                    out / "artifacts" / f"{name}-n{n}.npz",
                    velocity=result["velocity"],
                    pressure=result["pressure"],
                    reference_velocity=exact["velocity"],
                    manufactured_force=exact["force"],
                    cell_centers=mesh.cell_centers,
                    cell_volumes=mesh.cell_volumes,
                    errors=result["velocity"] - exact["velocity"],
                )

    orders = {}
    for name in names:
        rows = [r for r in records if r.get("case") == name and r.get("converged")]
        orders[name] = [
            observed_order(coarse, fine) for coarse, fine in zip(rows, rows[1:])
        ]
    payload = dict(
        schema="tensorfvm.ethier-steinman-3d-study/1",
        backend="native_hybrid_ns3d.NativeHybridNavierStokes3D",
        reference=dict(
            doi="10.1002/fld.1650190502",
            citation="Ethier D.L., Steinman D.A., Int. J. Numer. Methods Fluids 19 (1994) 369-375",
            access="publisher paywall (Semantic Scholar openAccess = CLOSED); closed form re-derived and self-audited, see 'reference_audit'",
        ),
        formulation="frozen-time steady problem with manufactured force f = du/dt + (u.grad)u - nu lap(u)",
        conditions=dict(
            frozen_time_s=args.time,
            convection_order=args.convection_order,
            newton_tolerance=args.tolerance,
            velocity_linf_gate=GATE,
            divergence_gate=OMEGA_GATE,
            domain="[-1,1]^3, all six faces Dirichlet (exact area-mean velocity)",
        ),
        cases={name: dict(a=CASES[name].a, d=CASES[name].d,
                          viscosity=CASES[name].viscosity,
                          reynolds_number=CASES[name].reynolds_number,
                          unforced=bool(CASES[name].is_unforced)) for name in names},
        reference_audit=audit,
        records=records,
        observed_order_velocity_linf=orders,
    )
    (out / "study.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "RESULTS.md").write_text(render_markdown(payload), encoding="utf-8")
    write_figures(out, n_list, names)
    failed = [r for r in records if not r.get("converged")]
    return 0 if not failed else 3


def render_markdown(payload) -> str:
    lines = [
        "# Ethier--Steinman 3-D manufactured solution: native hybrid NS ladder",
        "",
        "Machine-generated by `scripts/ethier-steinman-3d/run_study.py` -- do not edit by hand.",
        "",
        "Reference: " + payload["reference"]["citation"] + " (DOI "
        + payload["reference"]["doi"] + ").",
        "Access status: " + payload["reference"]["access"] + ".",
        "",
        "## Exact field (rho = 1)",
        "",
        "u_i = -a [ e^{a x_i} sin(a (x_j + x_k)) + e^{a x_k} cos(a (x_i + x_j)) ] e^{-d^2 t} "
        "with (i,j,k) cyclic;",
        "the closed form is divergence-free (analytic max "
        f"{payload['reference_audit'][list(payload['cases'])[0]]['analytic_divergence_max']:.1e}), "
        "is an exact Laplacian eigenfunction (lap(u) = -a^2 u) and satisfies the unforced "
        "Navier-Stokes equations iff nu = d^2/a^2 (curl of the momentum residual vanishes there).",
        "",
        "## Conditions",
        "",
        "| tag | a | d | nu | Re = 1/nu | unforced NS solution |",
        "|---|---|---|---|---|---|",
    ]
    for name, case in payload["cases"].items():
        lines.append(
            f"| {name} | {case['a']:g} | {case['d']:g} | {case['viscosity']:g} | "
            f"{case['reynolds_number']:g} | {case['unforced']} |"
        )
    lines += [
        "",
        f"Frozen time t = {payload['conditions']['frozen_time_s']} s, convection order "
        f"{payload['conditions']['convection_order']}, Newton tolerance "
        f"{payload['conditions']['newton_tolerance']:g}, domain ["
        + payload["conditions"]["domain"] + "].",
        "",
        "## Reference self-audit",
        "",
        "| case | div u (analytic) | div u (FD) | grad err | lap err | curl(momentum residual) | curl(f) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, audit in payload["reference_audit"].items():
        lines.append(
            f"| {name} | {audit['analytic_divergence_max']:.2e} | "
            f"{audit['finite_difference_divergence_max']:.2e} | "
            f"{audit['gradient_max_error']:.2e} | {audit['laplacian_max_error']:.2e} | "
            f"{audit['unforced_momentum_curl_max']:.2e} | {audit['force_curl_max']:.2e} |"
        )
    lines += [
        "",
        "## Ladder (velocity L_inf relative error is the gated metric)",
        "",
        "| case | n | dx | velocity L_inf rel | gate <1% | velocity L2 rel | max div | newton it | s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in payload["records"]:
        if record.get("converged"):
            lines.append(
                f"| {record['case']} | {record['n']} | {record['spacing_m']:.4f} | "
                f"{record['velocity_relative_linf']:.3e} | "
                f"{'PASS' if record['velocity_linf_gate_passed'] else 'FAIL'} | "
                f"{record['velocity_relative_l2']:.3e} | {record['max_divergence_s_inv']:.1e} | "
                f"{record['newton_iterations']} | {record['elapsed_seconds']:.1f} |"
            )
        else:
            lines.append(
                f"| {record['case']} | {record['n']} | n/a | FAIL | FAIL | n/a | n/a | n/a | "
                f"{record['elapsed_seconds']:.1f} |"
            )
    lines += ["", "## Observed order of the gated metric", ""]
    for name, order in payload["observed_order_velocity_linf"].items():
        lines.append(f"* {name}: " + ", ".join(f"{o:.2f}" for o in order))
    failures = [r for r in payload["records"] if not r.get("converged")]
    lines += ["", "## Failures", ""]
    if failures:
        for record in failures:
            lines.append(
                f"* {record['case']} n={record['n']}: {record.get('failure_stage')} -- "
                f"{record.get('failure_reason')}"
            )
    else:
        lines.append("* none")
    lines.append("")
    return "\n".join(lines)


def write_figures(out: Path, n_list, names) -> None:
    """Slice fields and the convergence ladder as PNG figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - figures are optional for the ladder
        return
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    finest = max(n_list)
    for name in names:
        path = out / "artifacts" / f"{name}-n{finest}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        centres = data["cell_centers"]
        n = finest
        shape = (n, n, n)  # z, y, x ordering of es_mesh
        mid = n // 2
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        panels = [
            ("reference_velocity", 0, "exact u_x"),
            ("velocity", 0, "computed u_x"),
            ("errors", 0, "u_x error"),
            ("reference_velocity", 1, "exact u_y"),
            ("velocity", 1, "computed u_y"),
            ("errors", 1, "u_y error"),
        ]
        for axis, (key, component, title) in zip(axes.ravel(), panels):
            field = data[key][:, component].reshape(shape)[mid, :, :]
            limit = np.max(np.abs(data[key][:, component]))
            if key == "errors":
                limit = np.max(np.abs(data["errors"][:, component])) or 1.0
                image = axis.imshow(field, origin="lower", cmap="RdBu_r",
                                    extent=[-1, 1, -1, 1], vmin=-limit, vmax=limit)
            else:
                image = axis.imshow(field, origin="lower", cmap="viridis",
                                    extent=[-1, 1, -1, 1], vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.set_xlabel("y (z = 0 slice)" if False else "y")
            axis.set_ylabel("z")
            fig.colorbar(image, ax=axis, fraction=0.046)
        fig.suptitle(f"Ethier--Steinman {name}: mid-plane x slice, n={n}")
        fig.tight_layout()
        fig.savefig(figures / f"{name}-slice-n{finest}.png", dpi=130)
        plt.close(fig)

    payload = json.loads((out / "study.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        rows = [r for r in payload["records"] if r.get("case") == name and r.get("converged")]
        if not rows:
            continue
        h = [r["spacing_m"] for r in rows]
        axes[0].loglog(h, [r["velocity_relative_linf"] for r in rows], "o-", label=f"{name} L_inf")
        axes[0].loglog(h, [r["velocity_relative_l2"] for r in rows], "s--", label=f"{name} L2")
        axes[1].loglog(h, [max(r["max_divergence_s_inv"], 1e-16) for r in rows], "o-", label=name)
    axes[0].axhline(0.01, color="k", ls=":", label="1% gate")
    axes[0].set_xlabel("dx"); axes[0].set_ylabel("relative velocity error")
    axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_title("velocity error vs mesh size")
    axes[1].set_xlabel("dx"); axes[1].set_ylabel("max |div u| (1/s)")
    axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_title("discrete divergence of accepted flux")
    fig.tight_layout()
    fig.savefig(figures / "convergence.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())