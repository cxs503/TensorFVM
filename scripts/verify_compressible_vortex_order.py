#!/usr/bin/env python3
"""Order-of-accuracy verification of the compressible Euler surface (isentropic vortex).

Why this script exists
----------------------
``docs/RANS_verification.md`` and the m7555 verification-debt audit recorded the
compressible path as verified only against *discontinuous* exact solutions
(Sod shock tube, two-gas contact, Richtmyer--Meshkov, double blast).  None of
those can measure the truncation error of a smooth flow, so the module docstring
claim "second order in space and time for smooth flow"
(``src/tensorfvm/compressible/reconstruct.py``) had never been tested.

This script builds the missing test: the isentropic vortex of Shu (1998) on a
periodic box, where the exact solution is known analytically at every time, plus
two *directional controls* (the same smooth sine field advected along x and along
y) that separate a flux/integrator defect from a direction-specific one.

Outputs (all numbers below are produced by this script, nothing hand-typed):

``docs/verification-compressible-isentropic-vortex/study.json``
    grids, per-grid errors, observed orders, invariants, controls.
``docs/verification-compressible-isentropic-vortex/fields.npz``
    the n=64 numerical and exact density fields behind the cloud figure.
``docs/verification-compressible-isentropic-vortex/vortex_field.png``
    density field + pointwise error map.
``docs/verification-compressible-isentropic-vortex/order_convergence.png``
    log-log error vs h with the second-order reference slope.

Usage::

    PYTHONPATH=src python3 scripts/verify_compressible_vortex_order.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.compressible.isentropic_vortex import (  # noqa: E402
    cell_centres,
    error_norms,
    initial_state,
    mass_and_momentum,
    observed_order,
    run_vortex,
    single_gas_eos,
)
from tensorfvm.compressible.solver import EulerSolver  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "verification-compressible-isentropic-vortex"

L = 10.0
A = 0.2
LAMBDA = 2.0
T_END = 1.0


def advection_control(n: int, axis: str, limiter: str = "none") -> float:
    """L1 error of one full sine period-half advected along ``axis`` at unit speed.

    The field is an exact *entropy wave* of the Euler equations (uniform ``p``
    and uniform velocity), so the exact solution is the initial sine shifted by
    ``t``.  Running the same field along x and along y on the *same* grid isolates
    a direction-specific truncation defect from a flux/integrator one.
    """
    solver = EulerSolver(
        nx=n, ny=n, lx=L, ly=L, eos=single_gas_eos(), bc=("periodic",) * 4,
        flux_mode="hllc", order=2, limiter=limiter, cfl=0.3,
    )
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * (L / n)
    profile = 1.0 + A * torch.sin(2.0 * math.pi * c / LAMBDA)

    def field(shift: float) -> torch.Tensor:
        base = 1.0 + A * torch.sin(2.0 * math.pi * (c - shift) / LAMBDA)
        return (base.view(-1, 1) if axis == "x" else base.view(1, -1)).expand(n, n).contiguous()

    zeros = torch.zeros((n, n), dtype=torch.float64)
    ones = torch.ones((n, n), dtype=torch.float64)
    solver.set_primitive({
        "rho": field(0.0),
        "u": field(0.0) * 0.0 + (ones if axis == "x" else zeros),
        "v": ones if axis == "y" else zeros,
        "p": ones.clone(),
        "Y": zeros.clone(),
    })
    t = 0.0
    while t < T_END - 1e-15:
        dt = min(solver.stable_dt(), T_END - t)
        solver.step(dt)
        t += dt
    return float((solver.primitive()["rho"] - field(T_END)).abs().mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    study: dict = {
        "case": "isentropic_vortex",
        "reference": "Shu (1998) SIAM Rev. 40:743 isentropic vortex, exact at every t on a "
                     "periodic box; advected at (u_inf, v_inf) = (1, 0)",
        "settings": {"lx": L, "ly": L, "t_end": T_END, "cfl": 0.3, "flux": "HLLC",
                     "integrator": "MUSCL-Hancock", "limiter": "mc"},
        "vortex": {},
        "controls": {},
    }

    # ---- 1. vortex convergence for two swirl strengths ------------------
    for beta in (5.0, 1.0):
        grids = (32, 64, 128) if beta == 5.0 else (32, 64)
        rows = []
        for n in grids:
            solver, exact = run_vortex(n, beta=beta)
            num = solver.primitive()
            dx = solver.dx
            init = initial_state(n, n, lx=L, ly=L, beta=beta)
            inv0 = mass_and_momentum(init, dx, dx)
            inv1 = mass_and_momentum(num, dx, dx)
            x, y = cell_centres(n, n, L, L)
            w = 1.0 - exact["rho"]
            row = {
                "n": n, "h": dx, "steps": solver.nsteps,
                "rho": error_norms(num["rho"], exact["rho"]),
                "u": error_norms(num["u"], exact["u"]),
                "v": error_norms(num["v"], exact["v"]),
                "p": error_norms(num["p"], exact["p"]),
                "mass_drift": (inv1["mass"] - inv0["mass"]) / inv0["mass"],
                "momentum_x_drift": (inv1["momentum_x"] - inv0["momentum_x"])
                / abs(inv0["momentum_x"]),
                "core_x_error": float((x * num["rho"] * w).sum() / (num["rho"] * w).sum()
                                      - (x * exact["rho"] * w).sum() / (exact["rho"] * w).sum()),
                "core_amplitude_error": float(num["rho"].min() - exact["rho"].min()),
                "min_rho": float(num["rho"].min()),
                "min_p": float(num["p"].min()),
            }
            rows.append(row)
            if n == 64:
                torch.save(
                    {"rho": num["rho"], "rho_exact": exact["rho"], "h": torch.tensor(dx)},
                    OUT / "fields.npz",
                )
        orders = {
            name: [observed_order(rows[i][name]["L1"], rows[i + 1][name]["L1"])
                   for i in range(len(rows) - 1)]
            for name in ("rho", "u", "v", "p")
        }
        orders["core_amplitude"] = [
            observed_order(abs(rows[i]["core_amplitude_error"]),
                           abs(rows[i + 1]["core_amplitude_error"]))
            for i in range(len(rows) - 1)
        ]
        study["vortex"][f"beta={beta}"] = {"grids": list(grids), "rows": rows, "orders": orders}

    # ---- 2. temporal refinement at fixed grid (is time order >= 2?) -----
    temporal = []
    prev = None
    for cfl in (0.4, 0.2, 0.1):
        solver, exact = run_vortex(32, cfl=cfl)
        err = error_norms(solver.primitive()["rho"], exact["rho"])["L1"]
        temporal.append({"cfl": cfl, "steps": solver.nsteps, "rho_L1": err,
                         "ratio_vs_previous": (prev / err) if prev else None})
        prev = err
    study["temporal_refinement"] = temporal

    # ---- 3. directional controls ---------------------------------------
    controls = {}
    for axis in ("x", "y"):
        errs = [advection_control(n, axis) for n in (32, 64)]
        controls[axis] = {
            "grids": [32, 64],
            "L1": errs,
            "order": observed_order(errs[0], errs[1]),
        }
    study["controls"] = controls

    (OUT / "study.json").write_text(json.dumps(study, indent=2) + "\n")

    # ---- 4. figures -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fields = torch.load(OUT / "fields.npz")
        rho, exact = fields["rho"].numpy(), fields["rho_exact"].numpy()
        n = rho.shape[0]
        extent = (0.0, L, 0.0, L)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, data, title in (
            (axes[0], rho, "numerical $\\rho$ (n=%d)" % n),
            (axes[1], exact, "exact $\\rho$ (Shu 1998)"),
            (axes[2], np.abs(rho - exact), "pointwise $|\\rho-\\rho_{exact}|$"),
        ):
            im = ax.imshow(data.T, origin="lower", extent=extent, cmap="viridis")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Isentropic vortex, t=1, HLLC + MUSCL-Hancock (mc)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "vortex_field.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for key in study["vortex"]:
            rows = study["vortex"][key]["rows"]
            h = [r["h"] for r in rows]
            e = [r["rho"]["L1"] for r in rows]
            ax.loglog(h, e, "o-", label=f"vortex {key} (order "
                      f"{study['vortex'][key]['orders']['rho'][-1]:.2f})")
        h = [r["h"] for r in study["vortex"]["beta=5.0"]["rows"]]
        ref = [study["vortex"]["beta=5.0"]["rows"][0]["rho"]["L1"] * (x / h[0]) ** 2 for x in h]
        ax.loglog(h, ref, "k--", lw=1, label="second-order reference")
        ax.set_xlabel("h"); ax.set_ylabel(r"relative $L_1(\rho)$")
        ax.set_title("Smooth-flow convergence, compressible Euler surface", fontsize=10)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "order_convergence.png", dpi=130)
        plt.close(fig)
        study["figures"] = ["vortex_field.png", "order_convergence.png"]
    except Exception as exc:  # pragma: no cover - figures are not the claim
        study["figures"] = [f"figure generation failed: {exc}"]

    print(json.dumps({
        "vortex_beta5_orders": study["vortex"]["beta=5.0"]["orders"],
        "vortex_beta1_orders": study["vortex"]["beta=1.0"]["orders"],
        "controls": {k: v["order"] for k, v in study["controls"].items()},
        "temporal": [t["rho_L1"] for t in study["temporal_refinement"]],
    }, indent=2))
    print(f"wrote {OUT}/study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())