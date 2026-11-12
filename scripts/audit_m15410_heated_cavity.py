#!/usr/bin/env python
"""Board audit of m15410 — CASE-HEATED-CAVITY-FVM (TensorFVM).

Cross-audit of the de Vahl Davis (1983) differentially-heated-cavity benchmark
delivered by mission m15410.  Everything printed here is read from the mission's
run artifacts and from an **independent** re-run of the same committed driver in
an isolated scratch directory (never from the report tables, which are hand
typed in ``scripts/make_report.py``).

Outputs (all inside the auditing work tree):

* ``docs/audit-m15410-heated-cavity/audit-evidence.json`` — machine checkable
  numbers: reference vs computed per Ra and mesh, tolerances, test verdicts.
* ``docs/audit-m15410-heated-cavity/original-artifacts/`` — verbatim copy of the
  mission's run artifacts (they were left untracked in the mission work tree, so
  this copy is what makes the evidence durable in the repository).
* ``docs/audit-m15410-heated-cavity/fig_*.png`` — audit figures.

Usage::

    python scripts/audit_m15410_heated_cavity.py \
        --mission-tree /root/tp-agent/work/m15410 \
        --repro-tree /tmp/audit_m15410
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/audit-m15410-heated-cavity")
NU_PLATFORM_LIMIT = 0.15   # owner standard: >15 % against a reference = unacceptable
NU_CODE_GATE = 0.03        # tests/test_fvm_internal_flow_verified.py asserts < 3 %
RA_LIST = ("1000", "10000", "100000", "1000000")


def load_ledger(mission_tree: Path) -> dict:
    path = mission_tree / "artifacts/heated-cavity-benchmark/summary.json"
    if not path.exists():
        raise SystemExit(f"ledger missing: {path}")
    return json.loads(path.read_text())


def copy_artifacts(mission_tree: Path) -> list[str]:
    src = mission_tree / "artifacts/heated-cavity-benchmark"
    dst = OUT / "original-artifacts"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            copied.append(str((dst / f.name)))
    return copied


def consistency(ledger: dict) -> dict:
    """Internal invariants of the ledger itself (cheap, no re-run)."""
    out = {}
    for ra, case in ledger["cases"].items():
        rows = {}
        for n, blk in case["meshes"].items():
            nu_local = blk["nu_local"]
            rows[n] = {
                "nu_mean": blk["nu_mean"],
                "mean_of_nu_local": float(np.mean(nu_local)),
                "recomputed_mean_matches": bool(
                    abs(float(np.mean(nu_local)) - blk["nu_mean"]) < 1e-9),
                "nu_local_peak_location_y": float(
                    (int(np.argmax(nu_local)) + 0.5) / len(nu_local)),
                "nu_local_decays_after_peak": bool(
                    np.all(np.diff(nu_local[int(np.argmax(nu_local)):]) <= 1e-9)),
                "nu_local_first_last": [float(nu_local[0]), float(nu_local[-1])],
                "steps": blk["steps"],
                "converged": blk["converged"],
                "max_divergence": blk["max_divergence"],
                "momentum_residual_max": float(
                    blk["momentum_residual"]["max"]
                    if isinstance(blk["momentum_residual"], dict)
                    else blk["momentum_residual"]),
                "psi_max": blk["psi_max"],
            }
        nus = [rows[n]["nu_mean"] for n in sorted(rows, key=int)]
        rows["_convergence"] = {
            "nu_by_mesh": nus,
            "monotone_in_mesh": bool(all(np.diff(nus) * np.sign(nus[0] - case["meshes"][sorted(
                case["meshes"], key=int)[0]]["nu_reference"]) <= 1e-12)),
            "error_vs_reference": [abs(nu - case["meshes"][sorted(case["meshes"], key=int)[0]]
                                       ["nu_reference"]) / case["meshes"][sorted(
                                           case["meshes"], key=int)[0]]["nu_reference"]
                                   for nu in nus],
        }
        out[ra] = rows
    return out


def acceptance_table(ledger: dict) -> dict:
    """Reference vs computed at the finest mesh, with the two tolerance bars."""
    table = {}
    for ra in RA_LIST:
        acc = ledger["acceptance"][ra]
        finest = max(ledger["cases"][ra]["meshes"], key=int)
        blk = ledger["cases"][ra]["meshes"][finest]
        row = {
            "finest_mesh": int(finest),
            "nu_ref": acc["nu_reference"],
            "nu_got": acc["nu_computed"],
            "nu_rel_err": abs(acc["nu_computed"] - acc["nu_reference"]) / acc["nu_reference"],
            "u_ref": acc["u_max_reference"], "u_got": acc["u_max_computed"],
            "u_rel_err": abs(acc["u_max_computed"] - acc["u_max_reference"]) / acc["u_max_reference"],
            "v_ref": acc["v_max_reference"], "v_got": acc["v_max_computed"],
            "v_rel_err": abs(acc["v_max_computed"] - acc["v_max_reference"]) / acc["v_max_reference"],
            "u_got_matches_ledger_block": bool(
                abs(blk["u_max_over_alpha"] - acc["u_max_computed"]) < 1e-9),
            "converged": acc["converged"],
        }
        row["within_code_gate_3pct"] = bool(max(row["nu_rel_err"], row["u_rel_err"],
                                                row["v_rel_err"]) < NU_CODE_GATE)
        row["within_platform_limit_15pct"] = bool(max(row["nu_rel_err"], row["u_rel_err"],
                                                      row["v_rel_err"]) < NU_PLATFORM_LIMIT)
        table[ra] = row
    return table


def independent_repro(ledger: dict, repro_tree: Path) -> dict:
    """Compare the audit re-run against the mission ledger, mesh by mesh."""
    path = repro_tree / "artifacts/heated-cavity-benchmark/case_Ra10000.json"
    if not path.exists():
        return {"available": False, "note": f"re-run artifact missing: {path}"}
    repro = json.loads(path.read_text())
    res = {"available": True, "command":
           "python <m15410>/scripts/run_cavity_case.py --rayleigh 10000 --meshes 32,64 "
           "(cwd=/tmp/audit_m15410, imports <m15410>/src)", "meshes": {}}
    for n, blk in repro["case"]["meshes"].items():
        led = ledger["cases"]["10000"]["meshes"][n]
        res["meshes"][n] = {
            "nu_ledger": led["nu_mean"], "nu_rerun": blk["nu_mean"],
            "nu_abs_delta": abs(blk["nu_mean"] - led["nu_mean"]),
            "u_ledger": led["u_max_over_alpha"], "u_rerun": blk["u_max_over_alpha"],
            "u_abs_delta": abs(blk["u_max_over_alpha"] - led["u_max_over_alpha"]),
            "v_ledger": led["v_max_over_alpha"], "v_rerun": blk["v_max_over_alpha"],
            "v_abs_delta": abs(blk["v_max_over_alpha"] - led["v_max_over_alpha"]),
            "steps_ledger": led["steps"], "steps_rerun": blk["steps"],
            "reproduces_within_1e-6": bool(
                abs(blk["nu_mean"] - led["nu_mean"]) < 1e-6
                and abs(blk["u_max_over_alpha"] - led["u_max_over_alpha"]) < 1e-6
                and abs(blk["v_max_over_alpha"] - led["v_max_over_alpha"]) < 1e-6),
        }
    return res


def figures(ledger: dict, table: dict) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # fig 1: Nu vs mesh for every Ra, ledger lines + reference markers
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ra in zip(axes.ravel(), RA_LIST):
        meshes = sorted(ledger["cases"][ra]["meshes"], key=int)
        nus = [ledger["cases"][ra]["meshes"][n]["nu_mean"] for n in meshes]
        ref = ledger["acceptance"][ra]["nu_reference"]
        ax.plot([int(n) for n in meshes], nus, "o-", label="TensorFVM (ledger)")
        ax.axhline(ref, color="k", ls="--", label=f"de Vahl Davis Nu={ref}")
        ax.axhspan(ref * 0.97, ref * 1.03, color="grey", alpha=0.2, label="±3 % band")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("mesh n (n×n cells)")
        ax.set_ylabel("mean Nu (hot wall)")
        ax.set_title(f"Ra = {ra}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("m15410 heated-cavity benchmark — mesh convergence against de Vahl Davis (1983)")
    p = OUT / "fig_nu_vs_mesh.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    # fig 2: relative error (Nu, u_max, v_max) at the finest mesh, log scale + bars
    ras = list(RA_LIST)
    x = np.arange(len(ras))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (key, lbl) in enumerate([("nu_rel_err", "Nu"), ("u_rel_err", "|u|max"),
                                    ("v_rel_err", "|v|max")]):
        vals = [100.0 * table[ra][key] for ra in ras]
        ax.bar(x + (k - 1) * w, vals, w, label=lbl)
    ax.axhline(3.0, color="tab:red", ls="--", label="code gate 3 %")
    ax.axhline(15.0, color="darkred", ls=":", label="platform limit 15 %")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ra={r}" for r in ras])
    ax.set_ylabel("relative error at finest mesh (%, log)")
    ax.set_title("m15410: independent tolerance check against de Vahl Davis (1983)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    p = OUT / "fig_error_budget.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission-tree", default="/root/tp-agent/work/m15410")
    ap.add_argument("--repro-tree", default="/tmp/audit_m15410")
    args = ap.parse_args()
    mission_tree = Path(args.mission_tree)
    OUT.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(mission_tree)
    copied = copy_artifacts(mission_tree)
    table = acceptance_table(ledger)
    cons = consistency(ledger)
    repro = independent_repro(ledger, Path(args.repro_tree))
    figs = figures(ledger, table)

    evidence = {
        "mission": "m15410",
        "anchor_slug": "CASE-HEATED-CAVITY-FVM",
        "repo": "TensorFVM",
        "reference": "de Vahl Davis (1983), Int. J. Numer. Methods Fluids 3, 249-264",
        "tolerances": {"code_gate": NU_CODE_GATE, "platform_limit": NU_PLATFORM_LIMIT},
        "acceptance_finest_mesh": table,
        "ledger_consistency": cons,
        "independent_repro": repro,
        "artifacts_copied": copied,
        "figures": figs,
    }

    # verdict-relevant tallies
    all_pass_3 = all(table[ra]["within_code_gate_3pct"] for ra in RA_LIST)
    all_conv = all(cons[ra]["_convergence"]["monotone_in_mesh"] for ra in RA_LIST)
    evidence["summary_flags"] = {
        "all_ra_within_code_gate_3pct": bool(all_pass_3),
        "all_ra_monotone_mesh_convergence": bool(all_conv),
        "repro_bitwise_identical": bool(
            repro.get("available")
            and all(v["reproduces_within_1e-6"] for v in repro["meshes"].values())),
        # a buoyant cavity has |psi|_max of order 1 (de Vahl Davis: 1.94 at Ra=1e5);
        # values below 0.05 mean the streamfunction diagnostic is broken
        "psi_diagnostic_degenerate": bool(
            all(abs(cons[ra][n]["psi_max"]) < 0.05
                for ra in RA_LIST for n in cons[ra] if n != "_convergence")),
        "hot_wall_nu_peaks_near_bottom_corner_then_decays": bool(
            all(cons[ra][n]["nu_local_peak_location_y"] <= 0.30
                and cons[ra][n]["nu_local_decays_after_peak"]
                for ra in RA_LIST for n in cons[ra] if n != "_convergence")),
    }
    (OUT / "audit-evidence.json").write_text(json.dumps(evidence, indent=2))

    for ra in RA_LIST:
        r = table[ra]
        print(f"Ra={ra:>8} n={r['finest_mesh']}: Nu {r['nu_got']:.4f} vs {r['nu_ref']:.3f} "
              f"({100*r['nu_rel_err']:.2f}%)  u {100*r['u_rel_err']:.2f}%  "
              f"v {100*r['v_rel_err']:.2f}%  <3%={r['within_code_gate_3pct']}")
    print(json.dumps(evidence["summary_flags"], indent=2))
    if repro.get("available"):
        for n, v in repro["meshes"].items():
            print(f"repro n={n}: dNu={v['nu_abs_delta']:.2e} dU={v['u_abs_delta']:.2e} "
                  f"dV={v['v_abs_delta']:.2e} identical={v['reproduces_within_1e-6']}")
    print(f"evidence -> {OUT/'audit-evidence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())