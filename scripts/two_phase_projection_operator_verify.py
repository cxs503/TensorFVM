"""Verification report generator: two-phase pressure-Poisson operator.

Every number in ``docs/two-phase-projection-operator-verification/report.md``
comes from this script (``summary.json`` + ``operator-verification.png``).

Checks (all against independent, first-principles assembly — see
``tests/test_two_phase_projection_operator_matrix.py``):

1. relative residual ``||A dp - b||_inf / ||b||_inf`` on three configurations
   (isotropic/uniform, anisotropic/uniform, anisotropic/1000:1 stratified);
2. discrete divergence removal ``max|div(u_new)| / max|div(u*)|`` through the
   production correction routine ``_velocity_correction``;
3. two *mutation controls* (swapped ``dx/dy`` face geometry, dropped top
   atmospheric datum) proving the residual gate is not vacuous;
4. boundary-condition sensitivity: the same correction with the atmospheric top
   face frozen, i.e. what omitting the half-cell datum costs.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_two_phase_projection_operator_matrix import (  # noqa: E402
    _case,
    _divergent_field,
    assemble_poisson,
    divergence,
)

OUT = Path("docs/two-phase-projection-operator-verification")
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("isotropic_uniform", dict(anisotropic=False, stratified=False)),
    ("anisotropic_uniform", dict(anisotropic=True, stratified=False)),
    ("anisotropic_stratified", dict(anisotropic=True, stratified=True)),
]


def stem(name):
    return (name, _case(**dict(CASES)[name]))


def study(name, kwargs, dt=1e-5, seed=0):
    solver = _case(**kwargs)
    u_star, v_star = _divergent_field(solver, seed=seed)
    dp = solver._projection_pressure(u_star, v_star, dt)
    u_new, v_new = solver._velocity_correction(u_star, v_star, dp, dt)

    volume = solver.dx * solver.dy
    b = (-divergence(solver, u_star, v_star) * volume / dt).reshape(-1).numpy()
    a = assemble_poisson(solver)
    residual = float(np.abs(a @ dp.reshape(-1).numpy() - b).max() / np.abs(b).max())

    before = divergence(solver, u_star, v_star)
    after = divergence(solver, u_new, v_new)
    # boundary-condition sensitivity: atmospheric face left uncorrected
    v_top = v_new.clone()
    v_top[-1, :] = v_star[-1, :]
    after_top = divergence(solver, u_new, v_top)

    w = assemble_poisson(solver, swap_geometry=True)
    r_swap = float(np.abs(w @ dp.reshape(-1).numpy() - b).max() / np.abs(b).max())
    d = assemble_poisson(solver, drop_datums=True)
    r_drop = float(np.abs(d @ dp.reshape(-1).numpy() - b).max() / np.abs(b).max())

    return dict(
        case=name,
        nx=solver.nx,
        ny=solver.ny,
        dx=solver.dx,
        dy=solver.dy,
        density_ratio=float(solver.rho_field.max() / solver.rho_field.min()),
        operator_residual=residual,
        div_before=float(before.abs().max()),
        div_after=float(after.abs().max()),
        removal_ratio=float(after.abs().max() / before.abs().max()),
        div_after_top_frozen=float(after_top.abs().max()),
        top_frozen_ratio=float(after_top.abs().max() / before.abs().max()),
        mutation_residual_swap_geometry=r_swap,
        mutation_residual_drop_datum=r_drop,
        notes="anisotropic" if solver.dx != solver.dy else "isotropic",
    )


def main():
    rows = [study(name, kwargs) for name, kwargs in CASES]
    gate = 1e-8
    summary = dict(
        schema="tensorfvm.two-phase-projection-operator/1",
        gate_operator_residual=gate,
        gate_removal_ratio=1e-8,
        gate_mutation_min=1e-3,
        cases=rows,
        all_operator_residuals_pass=all(r["operator_residual"] < gate for r in rows),
        all_removal_ratios_pass=all(r["removal_ratio"] < 1e-8 for r in rows),
        # The geometry swap is only observable when dx != dy: on the isotropic
        # case it is a no-op, which is exactly why the anisotropic case exists.
        all_mutations_detected=all(
            r["mutation_residual_swap_geometry"] > 1e-3
            and r["mutation_residual_drop_datum"] > 1e-3
            for r in rows
            if r["notes"] == "anisotropic"
        ),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [r["case"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    width = 0.28
    axes[0].bar(x - width, [r["operator_residual"] for r in rows], width,
                label="A dp vs b (assembled operator)")
    axes[0].bar(x, [r["removal_ratio"] for r in rows], width,
                label="max|div| after / before")
    axes[0].bar(x + width, [r["top_frozen_ratio"] for r in rows], width,
                label="same, atmosphere datum frozen")
    axes[0].axhline(gate, ls="--", c="k", lw=1, label=f"gate {gate:g}")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("relative quantity [-]")
    axes[0].set_title("Operator consistency and divergence removal")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, [r["mutation_residual_swap_geometry"] for r in rows],
                width, label="mutation: dx/dy swapped")
    axes[1].bar(x + width / 2, [r["mutation_residual_drop_datum"] for r in rows],
                width, label="mutation: datum dropped")
    axes[1].axhline(1e-3, ls="--", c="k", lw=1, label="detection floor 1e-3")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("relative residual [-]")
    axes[1].set_title("Negative controls: wrong operators must be rejected")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "operator-verification.png", dpi=130)

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()