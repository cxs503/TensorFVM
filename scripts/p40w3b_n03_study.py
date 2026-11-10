#!/usr/bin/env python3
"""P40W3B-N03: Quick demonstration of momentum budget closure."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorfvm.internal_flow.backward_step import BackwardStepCase, BackwardStepConfig
from tensorfvm.internal_flow.natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from tensorfvm.internal_flow.sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig
from tensorfvm.internal_flow.momentum_budget import global_momentum_closure
from tensorfvm.internal_flow.reference import DE_VAHL_DAVIS_NU, borda_carnot_loss

torch.set_num_threads(4)
ARTIFACTS = Path("artifacts/internal-flow-p40w3b-n03")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_single_case(name, case, max_steps=5000):
    """Run a single case and return diagnostics."""
    print(f"\n{name}")
    t0 = time.time()
    case.solver.run(max_steps=max_steps, tolerance=1e-5)
    wt = time.time() - t0
    closure = global_momentum_closure(case.solver)
    print(f"  Steps: {case.solver.steps}, Time: {wt:.1f}s")
    print(f"  x-closure: {closure['x_closure_pct']:.4f}%")
    print(f"  y-closure: {closure['y_closure_pct']:.4f}%")
    print(f"  mass-div: {closure['mass_divergence_max']:.2e}")
    return {"steps": case.solver.steps, "wall_time_s": wt, "momentum_closure": closure}


def main():
    print("=" * 60)
    print("P40W3B-N03: Momentum Budget Closure Demonstration")
    print("=" * 60)

    results = {}

    # BFS Re=100, n=12
    cfg = BackwardStepConfig(cells_per_step_height=12, reynolds=100.0, cfl=0.4, max_steps=8000, tolerance=1e-6)
    case = BackwardStepCase(cfg)
    res = run_single_case("BFS Re=100, n=12", case, 8000)
    res["xr_over_H"] = case.reattachment()
    diag = case.diagnostics()
    res["x"] = diag["x"]; res["cf"] = diag["cf"]
    results["bfs_re100"] = res

    # BFS Re=400, n=12
    cfg = BackwardStepConfig(cells_per_step_height=12, reynolds=400.0, cfl=0.4, max_steps=10000, tolerance=1e-6)
    case = BackwardStepCase(cfg)
    res = run_single_case("BFS Re=400, n=12", case, 10000)
    res["xr_over_H"] = case.reattachment()
    diag = case.diagnostics()
    res["x"] = diag["x"]; res["cf"] = diag["cf"]
    results["bfs_re400"] = res

    # Expansion AR=2, n=15
    cfg = SuddenExpansionConfig(area_ratio=2.0, reynolds=100.0, cells_per_downstream_height=15,
                                 cfl=0.4, max_steps=8000, tolerance=1e-6)
    case = SuddenExpansionCase(cfg)
    res = run_single_case("Expansion AR=2, n=15", case, 8000)
    diag = case.diagnostics()
    res["k_extrapolated"] = diag["k_extrapolated"]
    res["k_borda_carnot"] = diag["k_borda_carnot"]
    res["cp_recovery"] = diag["cp_recovery_extrapolated"]
    res["x"] = diag["x"]; res["static"] = diag["static"]; res["total"] = diag["total"]
    results["expansion_ar2"] = res

    # Expansion AR=3, n=15
    cfg = SuddenExpansionConfig(area_ratio=3.0, reynolds=100.0, cells_per_downstream_height=15,
                                 cfl=0.4, max_steps=8000, tolerance=1e-6)
    case = SuddenExpansionCase(cfg)
    res = run_single_case("Expansion AR=3, n=15", case, 8000)
    diag = case.diagnostics()
    res["k_extrapolated"] = diag["k_extrapolated"]
    res["k_borda_carnot"] = diag["k_borda_carnot"]
    res["cp_recovery"] = diag["cp_recovery_extrapolated"]
    res["x"] = diag["x"]; res["static"] = diag["static"]; res["total"] = diag["total"]
    results["expansion_ar3"] = res

    # Cavity Ra=1e4, n=32
    cfg = NaturalConvectionConfig(n=32, rayleigh=1e4, prandtl=0.71, cfl=0.4, max_steps=10000, tolerance=1e-6)
    case = NaturalConvectionCase(cfg)
    res = run_single_case("Cavity Ra=1e4, n=32", case, 10000)
    diag = case.diagnostics()
    res["nu_mean"] = diag["nu_mean"]; res["nu_reference"] = diag["nu_reference"]
    res["nu_error_percent"] = diag["nu_error_percent"]; res["psi_max"] = diag["psi_max"]
    results["cavity_ra1e4"] = res

    # Cavity Ra=1e5, n=32
    cfg = NaturalConvectionConfig(n=32, rayleigh=1e5, prandtl=0.71, cfl=0.4, max_steps=15000, tolerance=1e-6)
    case = NaturalConvectionCase(cfg)
    res = run_single_case("Cavity Ra=1e5, n=32", case, 15000)
    diag = case.diagnostics()
    res["nu_mean"] = diag["nu_mean"]; res["nu_reference"] = diag["nu_reference"]
    res["nu_error_percent"] = diag["nu_error_percent"]; res["psi_max"] = diag["psi_max"]
    results["cavity_ra1e5"] = res

    # Cavity Ra=1e6, n=32
    cfg = NaturalConvectionConfig(n=32, rayleigh=1e6, prandtl=0.71, cfl=0.4, max_steps=20000, tolerance=1e-6)
    case = NaturalConvectionCase(cfg)
    res = run_single_case("Cavity Ra=1e6, n=32", case, 20000)
    diag = case.diagnostics()
    res["nu_mean"] = diag["nu_mean"]; res["nu_reference"] = diag["nu_reference"]
    res["nu_error_percent"] = diag["nu_error_percent"]; res["psi_max"] = diag["psi_max"]
    results["cavity_ra1e6"] = res

    # Save JSON
    def to_json(obj):
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_json(v) for v in obj]
        return obj

    ledger = ARTIFACTS / "summary.json"
    ledger.write_text(json.dumps(to_json(results), indent=2))
    print(f"\nLedger: {ledger}")

    # Generate figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = ARTIFACTS / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Fig 1: BFS Cf
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, key in enumerate(["bfs_re100", "bfs_re400"]):
        ax = axes[idx]
        data = results[key]
        ax.plot(data["x"], data["cf"], 'b-', lw=1.5)
        ax.axhline(0, color='k', ls='--', alpha=0.3)
        ax.set_xlabel("x/H"); ax.set_ylabel("$C_f$")
        ax.set_title(f"BFS Re={100 if idx==0 else 400}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_bfs_cf.png", dpi=150)
    plt.close()

    # Fig 2: Expansion Cp
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, key in enumerate(["expansion_ar2", "expansion_ar3"]):
        ax = axes[idx]
        data = results[key]
        x = np.array(data["x"]); p = np.array(data["static"])
        cp = (p - p[0]) / 0.5
        ax.plot(x, cp, 'b-', lw=1.5)
        ax.set_xlabel("x/D"); ax.set_ylabel("$C_p$")
        ax.set_title(f"Expansion AR={2 if idx==0 else 3}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_expansion_cp.png", dpi=150)
    plt.close()

    # Fig 3: Cavity Nu
    fig, ax = plt.subplots(figsize=(8, 5))
    ras = [1e4, 1e5, 1e6]
    nus = [results[f"cavity_ra{int(ra):.0e}".replace("+", "").replace("0", "")]["nu_mean"] for ra in ras]
    nu_refs = [DE_VAHL_DAVIS_NU[ra] for ra in ras]
    x = np.arange(3)
    ax.bar(x - 0.2, nus, 0.4, label='Computed', color='steelblue')
    ax.bar(x + 0.2, nu_refs, 0.4, label='de Vahl Davis', color='indianred')
    ax.set_xticks(x); ax.set_xticklabels([f"Ra={ra:.0e}" for ra in ras])
    ax.set_ylabel("$\\overline{Nu}$"); ax.set_title("Natural Convection Cavity")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_cavity_nu.png", dpi=150)
    plt.close()

    # Fig 4: Momentum closure
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [k.replace("_", "\n") for k in results.keys() if "momentum_closure" in results[k]]
    x_cl = [results[k]["momentum_closure"]["x_closure_pct"] for k in labels if k.replace("\n", "_") in results]
    y_cl = [results[k.replace("\n", "_")]["momentum_closure"]["y_closure_pct"] for k in labels]
    x_cl = [max(v, 1e-6) for v in x_cl]
    y_cl = [max(v, 1e-6) for v in y_cl]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos - 0.2, x_cl, 0.4, label='x-mom', color='steelblue')
    ax.bar(x_pos + 0.2, y_cl, 0.4, label='y-mom', color='indianred')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='1% target')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Closure error (%)"); ax.set_yscale('log')
    ax.set_title("Momentum Budget Closure"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_closure.png", dpi=150)
    plt.close()

    print(f"\nFigures: {fig_dir}/")
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
