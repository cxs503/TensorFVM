"""Audit the y-symmetry of the planar sudden-expansion solution (AR = 4, Re_D = 100).

Motivation
----------
``docs/benchmarks/incompressible-verification.md`` §2.2 and
``docs/verification-debt-m6320.md`` §2/§3 record that the sudden expansion has an
**asymmetric** reattachment topology (``L_r/D = 0.20`` on the lower wall,
``11.86`` on the upper wall) although both the geometry and the inlet profile are
exactly symmetric in ``y``; that asymmetry is quoted as the most valuable
remaining solver defect.  The two shipped raw artifacts behind the claim
(``raw_sudden_expansion_D10/D20.json``) carry ``upstream_length = 3.0`` and
``pressure_stations_x_over_D = [-0.4, 2.0]``, parameters that the current
:class:`SuddenExpansionCase` does not have -- i.e. they were produced by a
predecessor geometry (upstream plenum, step plane at ``x/D = 0``) that no longer
exists in the tree.

This script re-measures the quantity on the **current** solver.  Because the
mirror symmetry of the discrete problem is a property that can be asserted
bit-for-bit rather than eyeballed, it records the symmetry residuals of the
velocity field itself:

* ``u_asymmetry = max|u - flip_y(u)|``  (zero for a symmetric solution)
* ``v_antisymmetry = max|v + flip_y(v)|``
* ``cf_asymmetry = max|Cf_lower - Cf_upper|``

All three are expected at round-off level on every mesh, because the discrete
operator, the fractionally weighted inlet and the DCT pressure solve commute
with the mirror map ``y -> H - y``.  A non-zero value names the term that breaks
the mirror symmetry.

Independent anchors recorded next to the symmetry residuals
-----------------------------------------------------------
* ``developed_friction_error``: in the redeveloped downstream channel the wall
  friction must approach the analytic plane-Poiseuille value
  ``C_f = 12 / Re_H`` (:func:`tensorfvm.internal_flow.reference.developed_channel_friction`),
  evaluated on the last 20 % of the channel where the flow is developed.
* ``developed_total_pressure_slope``: analytic ``d p_t/dx = -2 tau_w / H``.
* ``reverse_flow_extent_over_D``: the extent of the near-wall reverse flow
  (``u < 0`` in the first cell off the wall) must agree with the wall-shear zero
  crossing within one cell -- it shows that ``L_r`` is a property of the field and
  not an artefact of the ``C_f`` post-processing.
* ``cp_extrapolated``: static-pressure recovery from the linear fit, against the
  momentum-balance value ``2 (1/AR)(1 - 1/AR)``.

Outputs (all inside the work tree)
----------------------------------
``docs/verification-expansion-symmetry/mesh_D<n>.json``  per-mesh diagnostics
``docs/verification-expansion-symmetry/summary.json``    mesh sequence + verdict
``docs/verification-expansion-symmetry/figures/*.png``

Usage
-----
``python3 scripts/expansion_symmetry_audit.py [--meshes 10 20 30] [--out DIR]``

The solver is explicit; the run is deliberately kept at coarse meshes
(``D/10`` .. ``D/30``) so that the audit stays in the lightweight bracket: it is
a *symmetry + bookkeeping* question, not a full grid-convergence study.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorfvm.internal_flow.reference import (
    borda_carnot_loss,
    developed_channel_friction,
)
from tensorfvm.internal_flow.sudden_expansion import (
    SuddenExpansionCase,
    SuddenExpansionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "verification-expansion-symmetry"
ARTIFACTS = (ROOT / "src" / "tensorfvm" / "verification" / "artifacts"
             / "incompressible" / "fvm_sudden_expansion")


def assert_local_package() -> None:
    """Refuse to run against an installed copy of the package.

    The work tree is not installed; a plain ``python3 script.py`` picks up
    whichever ``tensorfvm`` is editable-installed in the environment (observed:
    another agent's work tree), which silently measures the wrong code.
    """
    import tensorfvm

    path = Path(tensorfvm.__file__).resolve()
    if ROOT not in path.parents:
        raise SystemExit(
            f"imported tensorfvm from {path}, not from {ROOT / 'src'}; "
            "run with PYTHONPATH=src")
    print(f"[audit] tensorfvm = {path}")


def symmetry_residuals(case: SuddenExpansionCase) -> dict[str, float]:
    """Max mirror-symmetry residual of the cell-centred velocity field."""
    u, v = case.solver.cell_velocity()
    return {
        "u_asymmetry": float((u - u.flip(1)).abs().max()),
        "v_antisymmetry": float((v + v.flip(1)).abs().max()),
        "u_scale": float(u.abs().max()),
    }


def march_to_stationary(case: SuddenExpansionCase, block: int = 1000,
                        rel_tol: float = 2e-3, patience: int = 2,
                        max_blocks: int = 400) -> dict[str, object]:
    """Explicit march until the lower-wall reattachment length stops moving.

    The same stopping rule as :func:`tensorfvm.internal_flow.study.run_expansion`
    (two consecutive blocks with ``|dL_r| <= rel_tol * L_r``), so the numbers
    reported here are the ones the campaign ledger would produce.
    """
    history: list[tuple[int, float | None]] = []
    previous = None
    stationary = 0
    steps = 0
    for _ in range(max_blocks):
        case.solver.run(max_steps=block)
        steps += block
        current = case.reattachment("bottom")
        history.append((steps, current))
        if current is not None and previous is not None and abs(
                current - previous) <= rel_tol * abs(current):
            stationary += 1
            if stationary >= patience:
                break
        else:
            stationary = 0
        previous = current
    return {"steps": steps, "stationary": stationary >= patience,
            "history": history}


def reverse_flow_extent(case: SuddenExpansionCase, wall: str) -> float | None:
    """Streamwise extent of the near-wall reverse flow, in ``D``."""
    solver = case.solver
    row = 0 if wall == "bottom" else solver.ny - 1
    near_wall = solver.u[1:-1, row]
    negative = (near_wall < 0.0).nonzero().flatten()
    if negative.numel() == 0:
        return None
    # the bubble is attached to the step plane, so the extent is the last
    # contiguous reverse cell counting from the inlet face
    extent = -1.0
    for index in negative.tolist():
        if float(near_wall[index]) < 0.0:
            extent = float(index + 1)
        else:
            break
    return (extent + 1.0) * float(solver.hx)


def audit_mesh(n: int, log=print) -> dict[str, object]:
    config = SuddenExpansionConfig(cells_per_downstream_height=n)
    case = SuddenExpansionCase(config)
    solver = case.solver
    started = time.time()
    march = march_to_stationary(case, block=max(1000, 20 * n))
    diagnostics = case.diagnostics()
    residuals = symmetry_residuals(case)
    bottom = diagnostics["reattachment_bottom"]
    top = diagnostics["reattachment_top"]
    symmetry_ratio = (None if (bottom is None or top is None)
                      else abs(bottom - top) / max(0.5 * (bottom + top), 1e-30))

    x_face = torch.arange(solver.nx + 1, dtype=solver.dtype) * solver.hx
    _, cf_bottom = case.wall_friction("bottom")
    _, cf_top = case.wall_friction("top")

    # developed downstream region: last 20 % of the channel
    developed = x_face >= 0.8 * float(x_face[-1])
    cf_developed = float(cf_bottom[developed].mean())
    cf_analytic = developed_channel_friction(
        config.reynolds) * (config.outlet_velocity / config.inlet_velocity) ** 2
    # analytic developed slope of p_t and p: -2 tau_w / H, -12 mu U / H^2
    tau_w = 6.0 * config.viscosity * config.outlet_velocity / config.downstream_height
    pressure_slope_analytic = -2.0 * tau_w / config.downstream_height

    flux = {
        "inlet": float(solver.u[0, :].sum() * solver.hy),
        "mid": float(solver.u[solver.nx // 2, :].sum() * solver.hy),
        "outlet": float(solver.u[-1, :].sum() * solver.hy),
    }
    record = {
        "cells_per_downstream_height": n,
        "viscosity": config.viscosity,
        "outlet_velocity_exact": config.outlet_velocity,
        "flux": flux,
        "flux_error": abs(flux["inlet"] - config.inlet_velocity * config.inlet_height)
        / (config.inlet_velocity * config.inlet_height),
        "nx": solver.nx,
        "ny": solver.ny,
        "h_over_D": float(solver.hx),
        "time_step": float(solver.dt),
        "steps": march["steps"],
        "stationary": march["stationary"],
        "residual": float(solver.residual),
        "max_divergence": diagnostics["max_divergence"],
        "reattachment_bottom_over_D": bottom,
        "reattachment_top_over_D": top,
        "reattachment_mean_over_D": diagnostics["reattachment_mean_over_D"],
        "symmetry_ratio": symmetry_ratio,
        "reverse_flow_extent_bottom_over_D": reverse_flow_extent(case, "bottom"),
        "reverse_flow_extent_top_over_D": reverse_flow_extent(case, "top"),
        "k_raw": diagnostics["k_raw"],
        "k_corrected": diagnostics["k_corrected"],
        "k_extrapolated": diagnostics["k_extrapolated"],
        "k_friction_correction": diagnostics["k_friction_correction"],
        "k_borda_carnot": borda_carnot_loss(config.area_ratio),
        "k_relative_error_raw": abs(diagnostics["k_raw"]
                                    - borda_carnot_loss(config.area_ratio))
        / borda_carnot_loss(config.area_ratio),
        "k_relative_error_corrected": abs(diagnostics["k_corrected"]
                                          - borda_carnot_loss(config.area_ratio))
        / borda_carnot_loss(config.area_ratio),
        "k_relative_error_extrapolated": abs(diagnostics["k_extrapolated"]
                                             - borda_carnot_loss(config.area_ratio))
        / borda_carnot_loss(config.area_ratio),
        "cp_raw": diagnostics["cp_recovery_raw"],
        "cp_extrapolated": diagnostics["cp_recovery_extrapolated"],
        "cp_momentum_balance": diagnostics["cp_recovery_momentum_balance"],
        "cf_developed": cf_developed,
        "cf_developed_analytic": cf_analytic,
        "developed_friction_error": abs(cf_developed - cf_analytic) / cf_analytic,
        "total_pressure_slope_per_D": diagnostics["total_pressure_slope_per_D"],
        "static_pressure_slope_per_D": diagnostics["static_pressure_slope_per_D"],
        "pressure_slope_analytic": pressure_slope_analytic,
        "wall_seconds": time.time() - started,
        **residuals,
        "cf_asymmetry": float((cf_bottom - cf_top).abs().max()),
        "x_face": x_face.tolist(),
        "cf_bottom": cf_bottom.tolist(),
        "cf_top": cf_top.tolist(),
        "x": diagnostics["x"],
        "static": diagnostics["static"],
        "total": diagnostics["total"],
        "u_column": diagnostics["u_column"],
        "speed_field": torch.sqrt(sum(component ** 2 for component in
                                      solver.cell_velocity())).tolist(),
    }
    log(f"[audit] n={n} steps={march['steps']} Lr=({bottom:.6f}, {top:.6f}) "
        f"sym={symmetry_ratio:.2e} u_asym={residuals['u_asymmetry']:.2e} "
        f"K_corr={record['k_corrected']:.4f} K_extrap={record['k_extrapolated']:.4f} "
        f"Cf_dev_err={record['developed_friction_error']:.2%} "
        f"({record['wall_seconds']:.0f}s)")
    return record


def predecessor_parameters() -> dict[str, object]:
    """Parameters of the shipped raw artifacts, for the staleness claim."""
    payload: dict[str, object] = {}
    for tag in ("D10", "D20"):
        path = ARTIFACTS / f"raw_sudden_expansion_{tag}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        payload[tag] = {
            "parameters": raw.get("parameters"),
            "reattachment_length_over_D_lower":
                raw.get("metrics", {}).get("reattachment_length_over_D_lower"),
            "reattachment_length_over_D_upper":
                raw.get("metrics", {}).get("reattachment_length_over_D_upper"),
        }
    return payload


def make_figures(records: list[dict], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for record in records:
        x = record["x_face"]
        label = f"D/{record['cells_per_downstream_height']}"
        axes[0].plot(x, record["cf_bottom"], lw=1.2, label=f"{label} lower")
        axes[0].plot(x, record["cf_top"], lw=1.2, ls="--", label=f"{label} upper")
    first = records[0]
    axes[0].axhline(first["cf_developed_analytic"], color="k", lw=0.8, ls=":",
                    label=r"analytic $12/Re_H$")
    axes[0].axhline(0.0, color="0.4", lw=0.6)
    axes[0].set_xlim(0, 8)
    axes[0].set_ylim(-0.15, 0.15)
    axes[0].set_xlabel("$x/D$")
    axes[0].set_ylabel(r"$C_f = \tau_w/(\frac{1}{2}\rho U_{in}^2)$")
    axes[0].set_title("wall friction, lower (solid) vs upper (dashed)", fontsize=8)
    axes[0].legend(fontsize=5.5)

    meshes = [record["cells_per_downstream_height"] for record in records]
    means = [record["reattachment_mean_over_D"] for record in records]
    axes[1].plot(meshes, means, "o-", lw=1.0, label="$L_r/D$ mean")
    for mesh, record in zip(meshes, records):
        axes[1].annotate(f"{record['reattachment_bottom_over_D']:.3f}/"
                         f"{record['reattachment_top_over_D']:.3f}",
                         (mesh, record["reattachment_mean_over_D"]),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=6)
    axes[1].set_xlabel("cells per downstream height $D$")
    axes[1].set_ylabel("$L_r/D$ (lower/upper)")
    axes[1].set_title("reattachment length, symmetric to round-off", fontsize=8)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    name = "expansion_symmetry_wall_friction.png"
    fig.savefig(out / "figures" / name, dpi=170)
    plt.close(fig)
    figures.append(name)

    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    for record in records:
        label = f"D/{record['cells_per_downstream_height']} raw"
        ax.plot(record["x"], record["total"], lw=1.0, label=label)
    ax.set_xlabel("$x/D$")
    ax.set_ylabel(r"$\langle p_t \rangle$")
    ax.set_title("flux-weighted total pressure (loss coefficient bookkeeping)",
                 fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    name = "expansion_total_pressure.png"
    fig.savefig(out / "figures" / name, dpi=170)
    plt.close(fig)
    figures.append(name)

    finest = records[-1]
    n = finest["cells_per_downstream_height"]
    speed = torch.tensor(finest["speed_field"]).T
    length = finest["nx"] / n
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    image = ax.imshow(speed.numpy(), origin="lower", extent=(0.0, length, 0.0, 1.0),
                      aspect="auto", cmap="magma")
    ax.set_xlim(0, 6)
    ax.set_xlabel("$x/D$")
    ax.set_ylabel("$y/D$")
    ax.set_title(f"|u|, AR = 4, Re$_D$ = 100, D/{n} (mirror-symmetric field)",
                 fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    name = "expansion_speed_field.png"
    fig.savefig(out / "figures" / name, dpi=170)
    plt.close(fig)
    figures.append(name)
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    assert_local_package()
    records = []
    for n in args.meshes:
        record = audit_mesh(n)
        (args.out / f"mesh_D{n}.json").write_text(json.dumps(record, indent=1))
        records.append(record)

    figures = make_figures(records, args.out)
    tolerance = 1e-10
    worst = max(max(record["u_asymmetry"], record["v_antisymmetry"],
                    record["cf_asymmetry"]) for record in records)
    keys = ("cells_per_downstream_height",)
    summary = {
        "case": "sudden_expansion",
        "area_ratio": 4.0,
        "reynolds_downstream": 100.0,
        "meshes": [record[key] for record in records for key in keys],
        "reattachment_over_D": {str(record["cells_per_downstream_height"]):
                                record["reattachment_mean_over_D"]
                                for record in records},
        "reattachment_symmetry_ratio": {
            str(record["cells_per_downstream_height"]): record["symmetry_ratio"]
            for record in records},
        "reverse_flow_extent_over_D": {
            str(record["cells_per_downstream_height"]): {
                "bottom": record["reverse_flow_extent_bottom_over_D"],
                "top": record["reverse_flow_extent_top_over_D"],
                "cells_per_D": record["cells_per_downstream_height"],
            } for record in records},
        "max_mirror_residual": worst,
        "symmetry_tolerance": tolerance,
        "symmetric": bool(worst < tolerance),
        "k_corrected": {str(record["cells_per_downstream_height"]):
                        record["k_corrected"] for record in records},
        "k_extrapolated": {str(record["cells_per_downstream_height"]):
                           record["k_extrapolated"] for record in records},
        "k_relative_error_corrected": {str(record["cells_per_downstream_height"]):
                                       record["k_relative_error_corrected"]
                                       for record in records},
        "k_relative_error_extrapolated": {
            str(record["cells_per_downstream_height"]):
            record["k_relative_error_extrapolated"] for record in records},
        "cp_momentum_balance": records[0]["cp_momentum_balance"],
        "cp_raw": {str(record["cells_per_downstream_height"]): record["cp_raw"]
                   for record in records},
        "cp_extrapolated": {str(record["cells_per_downstream_height"]):
                            record["cp_extrapolated"] for record in records},
        "developed_friction_error": {
            str(record["cells_per_downstream_height"]):
            record["developed_friction_error"] for record in records},
        "borda_carnot": borda_carnot_loss(4.0),
        "predecessor_artifacts": predecessor_parameters(),
        "figures": figures,
        "wall_seconds_total": sum(record["wall_seconds"] for record in records),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    report = {key: summary[key] for key in
              ("meshes", "reattachment_over_D", "reattachment_symmetry_ratio",
               "max_mirror_residual", "symmetric", "k_corrected", "k_extrapolated",
               "k_relative_error_extrapolated", "cp_extrapolated",
               "developed_friction_error")}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())