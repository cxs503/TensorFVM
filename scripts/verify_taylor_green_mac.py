"""Taylor--Green vortex verification runner for the periodic MAC kernel.

Producer of ``scripts/verification-taylor-green-mac/``: study JSON, figures and a
machine-generated results table.  Every number in the published artefacts comes
from this script; nothing is transcribed by hand.

Run::

    PYTHONPATH=src python3 scripts/verify_taylor_green_mac.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorfvm.periodic_mac import PeriodicMACConfig, PeriodicMACSolver
from tensorfvm.verification.taylor_green_mac import (
    TaylorGreenCase,
    compare,
    exact_fields,
    initialise,
    run_to,
)

BOX_M = 2 * math.pi
NZ = 4                       # z-invariant: nz is a plane count, dz is irrelevant
BASE_VISCOSITY = 0.05        # Re = A L / nu = 1 * 2 pi / 0.05 = 125.7
SPATIAL_STEPS_S = 0.01
SPATIAL_FINAL_S = 1.0
TEMPORAL_STEP_LADDER_S = (0.2, 0.1, 0.05, 0.01)
HORIZON_S = 3.0
GRID_LADDER = (8, 16, 24, 32)


def solver(n: int, dt: float, viscosity: float = BASE_VISCOSITY) -> PeriodicMACSolver:
    return PeriodicMACSolver(
        PeriodicMACConfig(
            nx=n,
            ny=n,
            nz=NZ,
            lengths_m=(BOX_M, BOX_M, BOX_M),
            viscosity_pa_s=viscosity,
            time_step_s=dt,
            nonlinear_tolerance=1e-12,
            nonlinear_iterations=100,
        )
    )


class NoConvection(PeriodicMACSolver):
    """Control: drop the nonlinear momentum flux, keep everything else.

    The Taylor--Green velocity field is *also* the exact solution of the linear
    diffusion problem, because ``(u.grad)u`` happens to be a pure gradient for
    it.  A kernel with the convective term removed therefore reproduces the
    velocity while producing no pressure at all -- which is exactly the point:
    only the pressure comparison can certify the nonlinear momentum flux.
    """

    def convective_fluxes(self, velocity):
        return torch.zeros((3, 3, *self.shape), dtype=velocity.dtype, device=velocity.device)


def order(fine: float, coarse: float, ratio: float) -> float:
    return math.log(coarse / fine, ratio)


def run_case(n: int, dt: float, final_time: float, case: TaylorGreenCase,
             wrapper=None) -> dict:
    build = solver(n, dt, case.viscosity)
    if wrapper is not None:
        build = wrapper(build)
    init = initialise(build, case, 0.0)
    run_to(build, final_time)
    record = compare(build, case, init["kinetic_energy_J"])
    record["steps"] = len(build.history)
    record["initialisation"] = init
    record["max_nonlinear_iterations"] = max(r["nonlinear_iterations"] for r in build.history)
    record["nonlinear_all_converged"] = all(r["nonlinear_converged"] for r in build.history)
    record["max_energy_balance_defect_J"] = max(abs(r["energy_balance_defect_J"]) for r in build.history)
    return {"solver": build, "record": record}


def vorticity_z(s: PeriodicMACSolver, velocity: torch.Tensor) -> torch.Tensor:
    """Cell-centred ``dv/dx - du/dy`` from the staggered faces (2nd order)."""
    dvdx = (s.next(velocity[1], 2) - velocity[1]) / s.spacing[0]
    dudy = (s.next(velocity[0], 1) - velocity[0]) / s.spacing[1]
    return dvdx - dudy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scripts/verification-taylor-green-mac")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = TaylorGreenCase(amplitude=1.0, viscosity=BASE_VISCOSITY, density=1.0)
    study: dict = {
        "schema": "tensorfvm.taylor-green-mac-study/1",
        "exact_solution": {
            "u": "A cos(x) sin(y) exp(-2 nu t)",
            "v": "-A sin(x) cos(y) exp(-2 nu t)",
            "w": "0",
            "p": "-(A^2/4)(cos 2x + cos 2y) exp(-4 nu t)   (rho = 1)",
            "kinetic_energy": "rho Lz A^2 pi^2 exp(-4 nu t)",
            "source": "Taylor & Green (1937) single-mode vortex; exact unforced NS solution",
        },
        "case": {
            "amplitude": case.amplitude,
            "viscosity_pa_s": case.viscosity,
            "density_kg_m3": case.density,
            "amplitude_decay_exponent_1_s": case.decay_exponent,
            "energy_decay_exponent_1_s": case.energy_decay_exponent,
            "box_m": [BOX_M, BOX_M, BOX_M],
            "nz": NZ,
            "reynolds_amplitude_based": case.amplitude * BOX_M / case.viscosity,
            "theoretical_initial_energy_J": case.density * BOX_M * case.amplitude**2 * math.pi**2,
        },
    }

    # ---- spatial ladder -------------------------------------------------------
    spatial = []
    for n in GRID_LADDER:
        run = run_case(n, SPATIAL_STEPS_S, SPATIAL_FINAL_S, case)
        rec = run["record"]
        spatial.append({
            "n": n,
            "dx_m": rec["dx_m"],
            "steps": rec["steps"],
            "velocity_relative_l2": rec["velocity_relative_l2"],
            "pressure_relative_l2": rec["pressure_relative_l2"],
            "velocity_x_relative_l2": rec["velocity_x_relative_l2"],
            "velocity_y_relative_l2": rec["velocity_y_relative_l2"],
            "velocity_z_max_abs_m_s": rec["velocity_z_max_abs_m_s"],
            "energy_relative_error": rec["energy"]["relative_error"],
            "measured_energy_decay_rate_1_s": rec["energy"]["measured_decay_rate_1_s"],
            "decay_rate_relative_error": rec["energy"]["decay_rate_relative_error"],
            "max_divergence_s_inv": rec["max_divergence_s_inv"],
            "sampled_max_divergence_s_inv": rec["initialisation"]["sampled_max_divergence_s_inv"],
            "sampled_energy_relative_error": rec["initialisation"]["energy_sampling_relative_error"],
            "max_energy_balance_defect_over_initial_J": rec["max_energy_balance_defect_J"]
            / rec["initialisation"]["kinetic_energy_J"],
            "nonlinear_all_converged": rec["nonlinear_all_converged"],
            "max_nonlinear_iterations": rec["max_nonlinear_iterations"],
        })
        print(f"spatial n={n:3d} u={rec['velocity_relative_l2']:.3e} "
              f"p={rec['pressure_relative_l2']:.3e} E={rec['energy']['relative_error']:.3e}",
              flush=True)
    for i in range(1, len(spatial)):
        ratio = spatial[i]["n"] / spatial[i - 1]["n"]
        spatial[i]["velocity_observed_order"] = order(
            spatial[i]["velocity_relative_l2"], spatial[i - 1]["velocity_relative_l2"], ratio)
        spatial[i]["pressure_observed_order"] = order(
            spatial[i]["pressure_relative_l2"], spatial[i - 1]["pressure_relative_l2"], ratio)
        spatial[i]["energy_observed_order"] = order(
            spatial[i]["energy_relative_error"], spatial[i - 1]["energy_relative_error"], ratio)
    study["spatial_ladder"] = spatial

    # ---- temporal ladder on the finest grid ------------------------------------
    temporal = []
    for dt in TEMPORAL_STEP_LADDER_S:
        run = run_case(GRID_LADDER[-1], dt, SPATIAL_FINAL_S, case)
        rec = run["record"]
        temporal.append({
            "time_step_s": dt,
            "steps": rec["steps"],
            "velocity_relative_l2": rec["velocity_relative_l2"],
            "pressure_relative_l2": rec["pressure_relative_l2"],
            "energy_relative_error": rec["energy"]["relative_error"],
            "max_nonlinear_iterations": rec["max_nonlinear_iterations"],
            "nonlinear_all_converged": rec["nonlinear_all_converged"],
        })
        print(f"temporal dt={dt:5.3f} u={rec['velocity_relative_l2']:.3e} "
              f"p={rec['pressure_relative_l2']:.3e}", flush=True)
    spatial_floor = min(row["velocity_relative_l2"] for row in temporal)
    spread = (max(row["velocity_relative_l2"] for row in temporal) - spatial_floor) / spatial_floor
    study["temporal_ladder"] = {
        "grid_n": GRID_LADDER[-1],
        "rows": temporal,
        "velocity_error_spread_over_floor": spread,
        "interpretation": (
            "Refining dt from 0.2 s to 0.01 s moves the velocity error by "
            f"{100 * spread:.2f}% around its floor, i.e. the error is dominated by the "
            "spatial truncation and no temporal observed order can be extracted from this "
            "case at these resolutions. This is a negative result, reported instead of a "
            "fitted order."
        ),
        "temporal_order_resolved": False,
    }

    # ---- long horizon ---------------------------------------------------------
    horizon = []
    for n in (16, 24, 32):
        run = run_case(n, SPATIAL_STEPS_S, HORIZON_S, case)
        rec = run["record"]
        horizon.append({
            "n": n,
            "final_time_s": rec["time_s"],
            "steps": rec["steps"],
            "velocity_relative_l2": rec["velocity_relative_l2"],
            "pressure_relative_l2": rec["pressure_relative_l2"],
            "energy_relative_error": rec["energy"]["relative_error"],
            "measured_energy_decay_rate_1_s": rec["energy"]["measured_decay_rate_1_s"],
            "decay_rate_relative_error": rec["energy"]["decay_rate_relative_error"],
            "max_divergence_s_inv": rec["max_divergence_s_inv"],
            "nonlinear_all_converged": rec["nonlinear_all_converged"],
        })
        print(f"horizon n={n:3d} t={rec['time_s']:.1f} u={rec['velocity_relative_l2']:.3e} "
              f"p={rec['pressure_relative_l2']:.3e}", flush=True)
    study["long_horizon"] = horizon

    # ---- discrimination controls ---------------------------------------------
    baseline = run_case(GRID_LADDER[-1], SPATIAL_STEPS_S, SPATIAL_FINAL_S, case)
    base_rec = baseline["record"]
    no_conv = run_case(GRID_LADDER[-1], SPATIAL_STEPS_S, SPATIAL_FINAL_S, case,
                       wrapper=lambda s: _rewrap(s, NoConvection))
    nc_rec = no_conv["record"]
    wrong = compare(baseline["solver"], TaylorGreenCase(amplitude=case.amplitude,
                                                        viscosity=1.2 * case.viscosity),
                    base_rec["initialisation"]["kinetic_energy_J"])
    exact_pressure_max = float(exact_fields(baseline["solver"], case, SPATIAL_FINAL_S)["pressure"].abs().max())
    study["discrimination"] = {
        "grid_n": GRID_LADDER[-1],
        "note": (
            "Each control breaks one physical ingredient on purpose. A gate that does not "
            "move is a gate that cannot see that ingredient."
        ),
        "baseline": {
            "velocity_relative_l2": base_rec["velocity_relative_l2"],
            "pressure_relative_l2": base_rec["pressure_relative_l2"],
            "decay_rate_relative_error": base_rec["energy"]["decay_rate_relative_error"],
        },
        "no_convection": {
            "velocity_relative_l2": nc_rec["velocity_relative_l2"],
            "pressure_relative_l2": nc_rec["pressure_relative_l2"],
            "pressure_max_abs_pa": float(no_conv["solver"].pressure.abs().max()),
            "exact_pressure_max_abs_pa": exact_pressure_max,
            "decay_rate_relative_error": nc_rec["energy"]["decay_rate_relative_error"],
        },
        "reference_viscosity_20_percent_high": {
            "velocity_relative_l2": wrong["velocity_relative_l2"],
            "pressure_relative_l2": wrong["pressure_relative_l2"],
            "decay_rate_relative_error": wrong["energy"]["decay_rate_relative_error"],
            "energy_relative_error": wrong["energy"]["relative_error"],
        },
    }
    print("controls", json.dumps(study["discrimination"], indent=1)[:400], flush=True)

    # ---- gates ------------------------------------------------------------------
    gate = 0.03
    finest = spatial[-1]
    study["gates"] = {
        "criterion": "relative L2 error against the exact solution below 3% on the finest grid",
        "velocity": {
            "value": finest["velocity_relative_l2"],
            "passed": finest["velocity_relative_l2"] < gate,
        },
        "pressure": {
            "value": finest["pressure_relative_l2"],
            "passed": finest["pressure_relative_l2"] < gate,
        },
        "energy_decay_rate": {
            "value": finest["decay_rate_relative_error"],
            "passed": finest["decay_rate_relative_error"] < gate,
        },
        "energy_at_final_time": {
            "value": finest["energy_relative_error"],
            "passed": finest["energy_relative_error"] < gate,
        },
        "discrete_divergence_free_initial_condition": {
            "value": finest["sampled_max_divergence_s_inv"],
            "passed": finest["sampled_max_divergence_s_inv"] < 1e-12,
        },
        "observed_order_velocity_in_1p8_2p2": {
            "value": finest.get("velocity_observed_order"),
            "passed": 1.8 <= finest.get("velocity_observed_order", 0) <= 2.2,
        },
        "observed_order_pressure_in_1p8_2p2": {
            "value": finest.get("pressure_observed_order"),
            "passed": 1.8 <= finest.get("pressure_observed_order", 0) <= 2.2,
        },
    }
    study["gate_summary"] = {
        "physical_accuracy_qualified": all(v["passed"] for k, v in study["gates"].items()
                                           if k != "criterion"),
        "informative_gates": {
            "velocity_gate_blind_to_convection": nc_rec["pressure_relative_l2"] > 0.99
            and nc_rec["velocity_relative_l2"] < gate,
            "velocity_gate_blind_to_20_percent_viscosity_error": wrong["velocity_relative_l2"] < gate,
            "decay_rate_gate_sees_20_percent_viscosity_error":
                wrong["energy"]["decay_rate_relative_error"] > gate,
        },
        "honest_limitations": [
            "The velocity field is also the exact solution of the linear diffusion problem, "
            "so the velocity gate alone cannot certify the nonlinear momentum flux; the "
            "pressure comparison is the discriminating witness (control no_convection).",
            "The whole movement is a single Fourier mode on a periodic box: no wall, no "
            "shear layer, no inhomogeneous direction. This qualifies the periodic kernel "
            "against a nonlinear NS solution; it does not qualify wall-bounded backends.",
            "No temporal observed order is reported: at n=32 the velocity error is "
            "spatial-error dominated for dt in [0.01, 0.2] s.",
        ],
    }

    (out / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    _figures(out, study, baseline["solver"], case)
    _results_table(out, study)
    print("verdict:", study["gate_summary"]["physical_accuracy_qualified"], flush=True)
    return 0


def _rewrap(original: PeriodicMACSolver, cls) -> PeriodicMACSolver:
    """Rebuild a solver of ``cls`` with the configuration of ``original``."""
    fresh = cls(original.config)
    return fresh


def _figures(out: Path, study: dict, s: PeriodicMACSolver, case: TaylorGreenCase) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # figure 1: convergence ladder
    rows = study["spatial_ladder"]
    dx = np.array([r["dx_m"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(dx, [r["velocity_relative_l2"] for r in rows], "o-", label="velocity $L_2$")
    ax.loglog(dx, [r["pressure_relative_l2"] for r in rows], "s-", label="pressure $L_2$")
    ax.loglog(dx, [r["energy_relative_error"] for r in rows], "^-", label="kinetic energy")
    ref = [r["velocity_relative_l2"] for r in rows][0] * (dx / dx[0]) ** 2
    ax.loglog(dx, ref, "k--", lw=1, label=r"$h^2$")
    ax.axhline(0.03, color="crimson", lw=1, ls=":", label="3% gate")
    ax.set_xlabel("$h = 2\\pi/n$ [m]")
    ax.set_ylabel("relative error vs exact")
    ax.set_title("Taylor--Green decay, exact nonlinear NS reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    t = np.linspace(0, 3, 200)
    ax.plot(t, np.exp(-4 * case.viscosity * t), "k-", lw=1.5, label=r"exact $e^{-4\nu t}$")
    lat = study["long_horizon"]
    ax.plot([r["final_time_s"] for r in lat],
            [1 - r["energy_relative_error"] for r in lat], "o", label="measured $E/E_0$, t=3 s")
    ax.plot([SPATIAL_FINAL_S] * len(rows),
            [1 - r["energy_relative_error"] for r in rows], "x",
            label=f"measured $E/E_0$, t={SPATIAL_FINAL_S:.0f} s")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$E(t)/E(0)$")
    ax.set_title("Kinetic-energy decay vs analytic $e^{-4\\nu t}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    plt.close(fig)

    # figure 2: vorticity field, exact field, error map
    numeric = vorticity_z(s, s.velocity)
    exact = exact_fields(s, case, s.time)
    exact_vort = vorticity_z(s, exact["velocity"])
    err = numeric - exact_vort
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panel = [(numeric, "numerical $\\omega_z$"), (exact_vort, "exact $\\omega_z$"),
             (err, "$\\omega_z$ error (numerical - exact)")]
    extent = [0, BOX_M, 0, BOX_M]
    vmax = float(numeric.abs().max())
    for ax, (field, title) in zip(axes, panel):
        limit = vmax if not title.startswith("$\\omega_z$ error") else float(err.abs().max())
        im = ax.imshow(field[0].numpy(), origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-limit, vmax=limit)
        ax.set_title(f"{title}\nmax={float(field.abs().max()):.3e}", fontsize=9)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"n={s.config.nx}, t={s.time:.2f} s, "
                 f"velocity $L_2$ error {study['gates']['velocity']['value']:.2e}")
    fig.tight_layout()
    fig.savefig(out / "fields.png", dpi=130)
    plt.close(fig)

    # figure 3: gate discrimination
    d = study["discrimination"]
    labels = ["baseline", "no convection", "reference $\\nu\\times1.2$"]
    velocity = [d["baseline"]["velocity_relative_l2"],
                d["no_convection"]["velocity_relative_l2"],
                d["reference_viscosity_20_percent_high"]["velocity_relative_l2"]]
    pressure = [d["baseline"]["pressure_relative_l2"],
                d["no_convection"]["pressure_relative_l2"],
                d["reference_viscosity_20_percent_high"]["pressure_relative_l2"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.2, velocity, 0.4, label="velocity $L_2$")
    ax.bar(x + 0.2, pressure, 0.4, label="pressure $L_2$")
    ax.axhline(0.03, color="crimson", ls=":", lw=1.2, label="3% gate")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative error")
    ax.set_title("Which gate can see which broken ingredient")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "discrimination.png", dpi=130)
    plt.close(fig)


def _results_table(out: Path, study: dict) -> None:
    lines = [
        "# Taylor--Green vortex decay: exact nonlinear Navier--Stokes reference",
        "",
        "Machine-generated by `scripts/verify_taylor_green_mac.py` -- do not edit by hand.",
        "",
        "Exact solution (periodic box $2\\pi^3$, $\\rho=1$): "
        "$u=A\\cos x\\sin y\\,e^{-2\\nu t}$, $v=-A\\sin x\\cos y\\,e^{-2\\nu t}$, "
        "$p=-\\frac{A^2}{4}(\\cos 2x+\\cos 2y)e^{-4\\nu t}$.",
        f"Case: A={study['case']['amplitude']}, nu={study['case']['viscosity_pa_s']}, "
        f"Re={study['case']['reynolds_amplitude_based']:.1f}, "
        f"final time {SPATIAL_FINAL_S} s.",
        "",
        "## Spatial ladder (dt = 0.01 s)",
        "",
        "| n | dx | velocity L2 | order | pressure L2 | order | energy err | order | max div |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study["spatial_ladder"]:
        lines.append(
            f"| {r['n']} | {r['dx_m']:.4f} | {r['velocity_relative_l2']:.3e} | "
            f"{r.get('velocity_observed_order', float('nan')):.2f} | "
            f"{r['pressure_relative_l2']:.3e} | "
            f"{r.get('pressure_observed_order', float('nan')):.2f} | "
            f"{r['energy_relative_error']:.3e} | "
            f"{r.get('energy_observed_order', float('nan')):.2f} | "
            f"{r['max_divergence_s_inv']:.1e} |")
    lines += ["", f"## Temporal ladder (n={study['temporal_ladder']['grid_n']})", "",
              "| dt [s] | velocity L2 | pressure L2 | energy err |", "|---|---|---|---|"]
    for r in study["temporal_ladder"]["rows"]:
        lines.append(f"| {r['time_step_s']} | {r['velocity_relative_l2']:.3e} | "
                     f"{r['pressure_relative_l2']:.3e} | {r['energy_relative_error']:.3e} |")
    lines += ["", study["temporal_ladder"]["interpretation"], "",
              f"## Long horizon (t = {HORIZON_S} s)", "",
              "| n | velocity L2 | pressure L2 | decay rate measured | exact | rate err |",
              "|---|---|---|---|---|---|"]
    for r in study["long_horizon"]:
        lines.append(f"| {r['n']} | {r['velocity_relative_l2']:.3e} | "
                     f"{r['pressure_relative_l2']:.3e} | "
                     f"{r['measured_energy_decay_rate_1_s']:.5f} | 0.2 | "
                     f"{r['decay_rate_relative_error']:.3e} |")
    lines += ["", "## Gate discrimination", "",
              "| variant | velocity L2 | vs 3% gate | pressure L2 | vs 3% gate | decay-rate err |",
              "|---|---|---|---|---|---|"]
    d = study["discrimination"]
    for label, key in (("baseline", "baseline"),
                       ("no convection", "no_convection"),
                       ("reference nu x1.2", "reference_viscosity_20_percent_high")):
        row = d[key]
        lines.append(
            f"| {label} | {row['velocity_relative_l2']:.3e} | "
            f"{'PASS' if row['velocity_relative_l2'] < 0.03 else 'FAIL'} | "
            f"{row['pressure_relative_l2']:.3e} | "
            f"{'PASS' if row['pressure_relative_l2'] < 0.03 else 'FAIL'} | "
            f"{row['decay_rate_relative_error']:.3e} |")
    lines += ["", "## Verdict", "",
              f"physical_accuracy_qualified = {study['gate_summary']['physical_accuracy_qualified']}",
              ""]
    for k in ("velocity", "pressure", "energy_decay_rate", "energy_at_final_time",
              "discrete_divergence_free_initial_condition",
              "observed_order_velocity_in_1p8_2p2", "observed_order_pressure_in_1p8_2p2"):
        g = study["gates"][k]
        lines.append(f"- {k}: {g['value']} -> {'PASS' if g['passed'] else 'FAIL'}")
    lines += ["", "### Honest limitations", ""]
    lines += [f"- {item}" for item in study["gate_summary"]["honest_limitations"]]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())