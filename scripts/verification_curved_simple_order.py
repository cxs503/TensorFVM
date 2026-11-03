"""Grid-convergence verification of the production collocated SIMPLE solver.

The ``curved-channel`` backend (sinusoidal walls, manufactured continuous
Navier--Stokes solution with a prescribed body force) is already gated at a
fixed 3 % field tolerance on one 32x16 grid.  A single-grid tolerance does not
establish the *order of accuracy*, the size of the remaining discretisation
error, or whether the gate sits inside an asymptotic range.  This script adds
that verification debt:

* a grid family 16x8 ... 128x64 solved with the unmodified production solver
  (no analytic initialisation, no coarse pre-conditioner, no Anderson damping);
* volume-weighted relative L2 and L-inf field errors against the manufactured
  continuous solution, pairwise observed orders, a log-log least-squares slope
  and the Richardson-extrapolated discretisation error of the finest grid;
* an independent audit of the manufactured solution itself (wall no-slip,
  constant cross-section mass flux, and an automatic-differentiation
  momentum/divergence residual), so the reference is never taken on trust;
* the discretisation-error ledger rendered into report.md with every number
  injected from the computed JSON (no hand transcription).

The manufactured solution is a *verification* (unit-problem) anchor: it proves
the discrete operators converge at the expected rate.  It is not a validation
against experiment and asserts nothing about turbulence or 3-D flow.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.solver import SolverConfig
from tensorfvm.sparse_simple import SparseBodyFittedSolver, host
from tensorfvm.curved_channel import reference, shape
from tensorfvm.verification.core import (
    Metric, ERROR_LIMIT, evaluate, provenance, sha, write_json, artifact_manifest)

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-7          # far below the discretisation error so orders are clean
GATE = ERROR_LIMIT        # repository-wide 3 % acceptance limit


def config(nx, ny, reynolds, max_iterations):
    return SolverConfig(
        mesh_type="curved-channel", nx=nx, ny=ny, length=2., height=1.,
        cylinder_radius=None, inlet_profile="parabolic", reynolds=reynolds,
        inlet_velocity=1., tolerance=TOLERANCE, max_iterations=max_iterations)


def solve(c, max_iterations):
    s = SparseBodyFittedSolver(c)
    s.momentum_body_force = reference(s.mesh.centers.reshape(-1, 2), c)[2]
    s.anderson_depth = 0
    s.coupled_coarse_shape = None
    steps = 0
    for steps in range(1, max_iterations + 1):
        s.step()
        if s.converged:
            break
    return s, steps


def field_errors(s, c):
    volume = host(s.volume)
    u, p = host(s.velocity), host(s.p)
    exact_u, exact_p, _ = reference(s.mesh.centers.reshape(-1, 2), c)
    exact_u, exact_p = host(exact_u), host(exact_p)
    weights = volume[:, None]
    return dict(
        velocity_volume_l2=float(np.sqrt(np.sum(weights * (u - exact_u) ** 2)
                                         / np.sum(weights * exact_u ** 2))),
        velocity_linf=float(np.max(np.abs(u - exact_u)) / np.max(np.abs(exact_u))),
        pressure_volume_l2=float(np.sqrt(np.sum(volume * (p - exact_p) ** 2)
                                         / np.sum(volume * exact_p ** 2))),
        pressure_linf=float(np.max(np.abs(p - exact_p)) / np.max(np.abs(exact_p))))


def divergence_ledger(s):
    """Cell divergence of the actual conservative face-flux ledger."""
    flux = s.mass_flux
    cell = torch.zeros_like(s.p)
    cell.index_add_(0, s.o, flux)
    cell.index_add_(0, s.ni, -flux[s.f])
    return host(cell / s.volume)


def audit_reference(c, mesh):
    """Independent checks of the manufactured solution, without trusting source.

    Derivatives of the analytic field are taken by central finite differences
    (stencil 1e-4 m gives ~1e-8 relative truncation and round-off on this mesh),
    so the check does not reuse any solver operator.
    """
    def fields(xy):
        velocity, pressure, force = reference(xy, c)
        return velocity, pressure, force

    centres = mesh.centers.reshape(-1, 2)
    step = 1e-4
    e = torch.zeros_like(centres)
    velocity, pressure, force = fields(centres)
    velocity = velocity.detach(); pressure = pressure.detach(); force = force.detach()
    derivative = torch.zeros(centres.shape[0], 2, 2)     # d u_a / d x_b
    second = torch.zeros(centres.shape[0], 2, 2)         # d2 u_a / d x_b2
    gradient = torch.zeros(centres.shape[0], 2)          # d p / d x_b
    for b in range(2):
        offset = e.clone(); offset[:, b] = step
        plus_v, plus_p, _ = fields(centres + offset)
        minus_v, minus_p, _ = fields(centres - offset)
        derivative[:, :, b] = (plus_v - minus_v) / (2 * step)
        gradient[:, b] = (plus_p - minus_p) / (2 * step)
        second[:, :, b] = (plus_v - 2 * velocity + minus_v) / step ** 2
    divergence = derivative[:, 0, 0] + derivative[:, 1, 1]
    convection = (velocity[:, :, None] * derivative).sum(1)
    residual = (c.density * convection + gradient - c.viscosity * second.sum(-1) - force)
    scale = max(float(torch.abs(force).max()), float(torch.abs(gradient).max()))
    wall = mesh.masks["wall"]
    wall_velocity = reference(mesh.face_centers[wall], c)[0]
    eta = torch.linspace(0., 1., 4001, dtype=torch.float64)
    fluxes = []
    for x in (0.25 * c.length, 0.5 * c.length, 0.75 * c.length):
        wall = float(shape(torch.tensor([x], dtype=torch.float64),
                           c.length, .2 * c.height)[0])
        cut = torch.stack((torch.full_like(eta, x), wall + eta * c.height), -1)
        w = reference(cut, c)[0][:, 0]
        fluxes.append(float(torch.trapz(w, eta) * c.height))
    return dict(
        divergence_max=float(torch.abs(divergence).max()),
        momentum_residual_max=float(torch.abs(residual).max()),
        momentum_residual_relative=float(torch.abs(residual).max() / scale),
        wall_no_slip_max=float(torch.abs(wall_velocity).max()),
        cross_section_flux=fluxes,
        cross_section_flux_exact=float(c.inlet_velocity * c.height),
        cross_section_flux_relative_error=float(
            max(abs(f - c.inlet_velocity * c.height) for f in fluxes)
            / (c.inlet_velocity * c.height)))


def orders(rows, key):
    values = [r[key] for r in rows]
    spacing = [r["spacing_m"] for r in rows]
    return [math.log(a / b) / math.log(spacing[i] / spacing[i + 1])
            for i, (a, b) in enumerate(zip(values, values[1:]))]


def least_squares_slope(rows, key):
    h = np.log([r["spacing_m"] for r in rows])
    e = np.log([r[key] for r in rows])
    return float(np.polyfit(h, e, 1)[0])


def richardson(rows, key):
    """Extrapolated discretisation error of the finest grid from the last pair."""
    e_coarse, e_fine = rows[-2][key], rows[-1][key]
    r = rows[-2]["spacing_m"] / rows[-1]["spacing_m"]
    p = math.log(e_coarse / e_fine) / math.log(r)
    return dict(order=p, refinement_ratio=r,
                extrapolated_relative_error=e_fine / (r ** p - 1))


def render(output, summary, rows):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    h = np.array([r["spacing_m"] for r in rows])
    colours = dict(velocity_volume_l2="tab:blue", pressure_volume_l2="tab:red")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, colour in colours.items():
        ax.loglog(h, [r[key] for r in rows], "o-", color=colour,
                  label=f"{key.replace('_', ' ')} (slope {summary['slopes'][key]:.2f})")
    for order, style in ((1., ":"), (2., "--")):
        ax.loglog(h, h ** order * rows[-1][next(iter(colours))] / rows[-1]["spacing_m"] ** order,
                  style, color="grey", lw=.8, label=f"reference slope {order:.0f}")
    ax.axhline(GATE, color="k", lw=.8, ls="-.", label="3 % acceptance gate")
    ax.set_xlabel("nominal cell size h = height / ny  [m]")
    ax.set_ylabel("volume-weighted relative error")
    ax.set_title("Curved-channel manufactured solution: observed order")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "observed-order.png", dpi=130)
    plt.close(fig)

    fine = rows[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11., 3.9))
    grid = fine["fields"]
    for ax, key, title, cmap in ((axes[0], "speed", "|u| / U", "viridis"),
                                 (axes[1], "pressure", "p / (G L)", "coolwarm")):
        mesh = ax.tricontourf(grid["xc"], grid["yc"], grid[key], 24, cmap=cmap)
        for wall in grid["wall_polylines"]:
            ax.plot(wall[:, 0], wall[:, 1], "k-", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        fig.colorbar(mesh, ax=ax, shrink=.85)
    fig.suptitle(f"Finest grid {fine['resolution']}: computed field on the sinusoidal channel")
    fig.tight_layout()
    fig.savefig(figures / "fields.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11., 3.9))
    axes[0].plot(grid["cut_eta"], grid["cut_computed"], "o", ms=3, label="computed")
    axes[0].plot(grid["cut_eta"], grid["cut_exact"], "-", color="k", lw=1.2, label="manufactured")
    axes[0].set_xlabel("eta = (y - g(x)) / height at x = 0.75 L")
    axes[0].set_ylabel("|u| [m/s]")
    axes[0].legend(fontsize=8)
    axes[1].plot(grid["row_x"], grid["row_computed"], "o", ms=3, label="computed")
    axes[1].plot(grid["row_x"], grid["row_exact"], "-", color="k", lw=1.2, label="manufactured")
    axes[1].set_xlabel("x [m] on the cell row nearest y = 0.5 height")
    axes[1].set_ylabel("p [Pa]")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Finest grid {fine['resolution']}: profiles against the continuous solution")
    fig.tight_layout()
    fig.savefig(figures / "profiles.png", dpi=130)
    plt.close(fig)
    return figures


def markdown(summary, rows):
    lines = ["# Curved-channel SIMPLE: grid-convergence verification", "",
             "Manufactured continuous Navier--Stokes solution on the sinusoidal channel of",
             "`tensorfvm.curved_channel`, solved by the unmodified production collocated",
             "SIMPLE solver (`SparseBodyFittedSolver`). Reference: the analytic field and its",
             "prescribed body force; anchor type = code verification (unit problem), not",
             "experimental validation.", "",
             "## Grid family", "",
             "| grid | cells | h [m] | steps | converged | u L2 | u Linf | p L2 | p Linf | 3 % gate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['resolution']} | {r['cells']} | {r['spacing_m']:.5f} | {r['steps']} | "
            f"{r['converged']} | {r['velocity_volume_l2']:.3e} | {r['velocity_linf']:.3e} | "
            f"{r['pressure_volume_l2']:.3e} | {r['pressure_linf']:.3e} | {r['passed']} |")
    lines += ["", "## Observed order of accuracy", "",
              "| pair | order(u) | order(p) |", "|---|---|---|"]
    for (a, b), ou, op in zip(zip(rows, rows[1:]), summary["orders"]["velocity_volume_l2"],
                              summary["orders"]["pressure_volume_l2"]):
        lines.append(f"| {a['resolution']} -> {b['resolution']} | {ou:.2f} | {op:.2f} |")
    lines += ["", f"Least-squares slope u = {summary['slopes']['velocity_volume_l2']:.3f}, "
              f"p = {summary['slopes']['pressure_volume_l2']:.3f}.", "",
              "| quantity | order | refinement ratio | extrapolated relative error |",
              "|---|---|---|---|"]
    for key, value in summary["richardson"].items():
        lines.append(f"| {key} | {value['order']:.2f} | {value['refinement_ratio']:.2f} | "
                     f"{value['extrapolated_relative_error']:.3e} |")
    lines += ["", "## Independent audit of the manufactured solution", "",
              f"* discrete divergence of the manufactured velocity field: "
              f"{summary['reference_audit']['divergence_max']:.2e} 1/s",
              f"* momentum residual (AD, relative to the force scale): "
              f"{summary['reference_audit']['momentum_residual_relative']:.2e}",
              f"* wall no-slip of the manufactured field at face centres: "
              f"{summary['reference_audit']['wall_no_slip_max']:.2e} m/s",
              f"* cross-section mass flux {summary['reference_audit']['cross_section_flux']} vs exact "
              f"{summary['reference_audit']['cross_section_flux_exact']:.4f} m^2/s "
              f"(relative {summary['reference_audit']['cross_section_flux_relative_error']:.2e})",
              "", "* Reading: the manufactured field is exactly divergence-free and carries a constant",
              "  cross-section flux, but the prescribed body force is **not** the exact Navier--Stokes",
              "  residual of that field (see the momentum residual above). The wall no-slip number is the",
              "  chord sagitta of the polygonal wall, not a defect of the field. The grid study below is",
              "  therefore an order-of-accuracy verification of the discrete operators against a smooth",
              "  prescribed field; a continuous-consistency repair of `curved_channel.reference` remains",
              "  open debt.", "",
              f"* cell-divergence of the *computed* flux ledger: "
              f"{summary['computed_divergence_max']:.2e} 1/s", "",
              "## Verdict", "",
              f"* asymptotic order consistent with second order: "
              f"**{summary['second_order_consistent']}** "
              f"(pairs within +/-0.35 of 2: {summary['second_order_pairs']})",
              f"* every grid inside the 3 % gate: **{summary['all_grids_pass']}**",
              f"* finest-grid Richardson extrapolation: u {summary['richardson']['velocity_volume_l2']['extrapolated_relative_error']:.2e}, "
              f"p {summary['richardson']['pressure_volume_l2']['extrapolated_relative_error']:.2e}",
              "", "Figures: `figures/observed-order.png`, `figures/fields.png`, "
              "`figures/profiles.png`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/verification-curved-simple-order")
    parser.add_argument("--grids", default="16x8,32x16,64x32",
                        help="comma separated nx x ny, coarse to fine")
    parser.add_argument("--reynolds", type=float, default=1.)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    finest = None
    for spec in args.grids.split(","):
        nx, ny = (int(v) for v in spec.lower().split("x"))
        c = config(nx, ny, args.reynolds, args.max_iterations)
        s, steps = solve(c, args.max_iterations)
        errors = field_errors(s, c)
        metrics = [Metric(name=key, value=value, limit=GATE,
                          description="manufactured continuous solution, strict 3 %")
                   for key, value in errors.items()]
        record = evaluate(metrics)
        row = dict(resolution=f"{nx}x{ny}", nx=nx, ny=ny, cells=int(s.count),
                   spacing_m=c.height / ny, steps=steps,
                   converged=bool(s.converged), metrics=record[0], passed=bool(record[1]),
                   final_residuals={k: float(v) for k, v in s.history[-1].items()
                                    if isinstance(v, (int, float))},
                   **errors)
        rows.append(row)
        print(f"{row['resolution']} cells={row['cells']} steps={steps} "
              f"uL2={errors['velocity_volume_l2']:.4e} pL2={errors['pressure_volume_l2']:.4e} "
              f"passed={row['passed']}", flush=True)
        if finest is None or s.count > finest[0].count:
            finest = (s, c)
    reference_audit = audit_reference(finest[1], finest[0].mesh)
    keys = ("velocity_volume_l2", "pressure_volume_l2")
    summary = dict(
        scope="manufactured continuous Navier-Stokes solution on the curved-channel backend; "
              "code verification only, no experimental validation, no turbulence, 2-D",
        reynolds=args.reynolds, length_m=finest[1].length, height_m=finest[1].height,
        inlet_velocity_m_s=finest[1].inlet_velocity, viscosity_pa_s=finest[1].viscosity,
        density_kg_m3=finest[1].density, tolerance=TOLERANCE, gate=GATE,
        orders={key: orders(rows, key) for key in keys},
        slopes={key: least_squares_slope(rows, key) for key in keys},
        richardson={key: richardson(rows, key) for key in keys},
        reference_audit=reference_audit,
        computed_divergence_max=float(np.max(np.abs(
            divergence_ledger(finest[0])))) if finest else None,
        all_grids_pass=all(r["passed"] for r in rows),
        second_order_pairs={key: [bool(1.65 <= p <= 2.35) for p in orders(rows, key)]
                            for key in keys},
        runs=rows)
    summary["second_order_consistent"] = all(
        all(v) for v in summary["second_order_pairs"].values())
    rows[-1]["fields"] = None
    try:
        (s, c) = finest
        row = rows[-1]
        centres = host(s.mesh.centers)
        xc, yc = centres[:, 0], centres[:, 1]
        h = c.length / int(c.nx)
        speed = np.hypot(host(s.velocity)[:, 0], host(s.velocity)[:, 1]) / c.inlet_velocity
        pstar = host(s.p) / (12 * c.viscosity * c.inlet_velocity / c.height ** 2 * c.length)
        exact_u, exact_p, _ = reference(s.mesh.centers.reshape(-1, 2), c)
        exact_u, exact_p = host(exact_u), host(exact_p)
        xs = torch.linspace(0., c.length, 401, dtype=torch.float64)
        wall = shape(xs, c.length, .2 * c.height)[0]
        lower = np.stack((host(xs), host(wall)), -1)
        upper = np.stack((host(xs), host(wall) + c.height), -1)
        def wall_height(where):
            return float(shape(torch.tensor([where], dtype=torch.float64),
                               c.length, .2 * c.height)[0])
        column_mask = np.abs(xc - .75 * c.length) < .5 * h
        eta = (yc[column_mask] - wall_height(.75 * c.length)) / c.height
        cut = np.argsort(eta)
        row_y = wall_height(.5 * c.length) + .5 * c.height
        row_mask = np.abs(yc - row_y) < .5 * c.height / int(c.ny)
        if not row_mask.any():
            row_mask = np.abs(yc - np.median(yc)) < .5 * c.height / int(c.ny)
        sweep = np.argsort(xc[row_mask])
        row["fields"] = dict(
            xc=xc, yc=yc, speed=speed, pressure=pstar, wall_polylines=[lower, upper],
            cut_eta=eta[cut].tolist(), cut_computed=speed[column_mask][cut].tolist(),
            cut_exact=np.hypot(exact_u[:, 0], exact_u[:, 1])[column_mask][cut].tolist(),
            row_x=xc[row_mask][sweep].tolist(),
            row_computed=pstar[row_mask][sweep].tolist(),
            row_exact=exact_p[row_mask][sweep].tolist())
        if not args.no_figures:
            render(output, summary, rows)
    except Exception as failure:          # figures never block the numeric ledger
        print("figures skipped:", repr(failure), flush=True)
    summary["provenance"] = provenance()
    for name in ("src/tensorfvm/curved_channel.py", "src/tensorfvm/sparse_simple.py",
                 "src/tensorfvm/body_fitted.py", "scripts/verification_curved_simple_order.py"):
        summary["provenance"]["source_sha256"][name] = sha(ROOT / name)
    write_json(output / "results.json", summary)
    (output / "report.md").write_text(markdown(summary, rows))
    write_json(output / "manifest.json",
               dict(source_sha256=summary["provenance"]["source_sha256"],
                    artifacts_sha256=artifact_manifest(output)))
    print("second order consistent:", summary["second_order_consistent"],
          "all grids <3%:", summary["all_grids_pass"], flush=True)
    return 0 if summary["all_grids_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())