"""Build the artifacts and figures for CASE-FVM-BLASIUS-PLATE.

* runs/blasius-plate/march.json                baseline marching-rung result
* runs/blasius-plate/fvm-partial-traces.json   whatever the FVM run reached
* docs/case-blasius-plate/figures/*.png        report figures

The FVM trace file is parsed from the run logs, so a partially converged (or
killed) FVM run still yields auditable numbers in the report.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.case_blasius_plate import (
    ERROR_LIMIT, LITERATURE, REFERENCE_RE_X, blasius_reference, run_march_case,
)
from tensorfvm.rans_ship_resistance.anchors import blasius_solution

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "blasius-plate"
FIGS = ROOT / "docs" / "case-blasius-plate" / "figures"
TRACE_RE = re.compile(r"it=\s*(\d+) cf=([-0-9.e+]+) H=([-0-9.e+]+) "
                      r"mom=([-0-9.e+]+) t=(\d+)s")


def baseline_march():
    result = run_march_case()
    result["tag"] = "march"
    (RUNS / "march.json").write_text(json.dumps(result, indent=2) + "\n",
                                     encoding="utf-8")
    return result


def fvm_traces():
    traces = {}
    for path in sorted((ROOT / "runs" / "logs").glob("fvm-*.log")):
        entries = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = TRACE_RE.search(line)
            if m:
                entries.append({"iteration": int(m.group(1)),
                                "cf_trailing": float(m.group(2)),
                                "H_trailing": float(m.group(3)),
                                "momentum": float(m.group(4)),
                                "elapsed_s": float(m.group(5))})
        if entries:
            traces[path.stem] = entries
    payload = {"case": "CASE-FVM-BLASIUS-PLATE", "kind": "partial-traces",
               "source": "runs/logs/fvm-*.log",
               "note": ("iterations actually completed by the 2-D FVM device in this "
                        "session; the Blasius target is cf = %.6g"
                        % blasius_reference()["cf"]),
               "traces": traces}
    (RUNS / "fvm-partial-traces.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                  encoding="utf-8")
    return traces


def figure_local_cf(sensitivity):
    """Local c_f along the plate: marching rung vs the Blasius line."""
    from tensorfvm.rans_ship_resistance.boundary_layer import march_boundary_layer

    stations = 401
    x = np.linspace(1.0 / stations, 1.0, stations)
    ue = np.ones_like(x)
    nu = 1.0 / REFERENCE_RE_X
    result = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None,
                                  ny=512, stretching=8.0)
    cf = np.asarray(result.cf)
    re_x = x / nu
    blasius = LITERATURE["cf_sqrt_re_x"] / np.sqrt(re_x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.plot(re_x, cf, "-", lw=1.6, label="TensorFVM thin-shear-layer march (laminar)")
    ax.plot(re_x, blasius, "--", lw=1.4, label=r"Blasius $c_f=0.664/\sqrt{Re_x}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$Re_x$")
    ax.set_ylabel(r"local $c_f$")
    ax.set_title("local skin friction along the plate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.axvline(REFERENCE_RE_X, color="0.6", lw=0.8)

    ax = axes[1]
    err = 100.0 * (cf - blasius) / blasius
    ax.plot(re_x, err, "-", lw=1.6, label="march vs Blasius")
    ax.axhline(100 * ERROR_LIMIT, color="r", ls=":", lw=1.0, label="+3 % gate")
    ax.axhline(-100 * ERROR_LIMIT, color="r", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$Re_x$")
    ax.set_ylabel("relative error [%]")
    ax.set_title(r"error against the similarity solution ($Re_x=10^5$ marked)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.axvline(REFERENCE_RE_X, color="0.6", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_local_cf_march_vs_blasius.png", dpi=150)
    plt.close(fig)
    return {"cf_at_rex_1e5": float(cf[-1]),
            "cf_error_percent": float(err[-1]),
            "max_abs_error_percent": float(np.abs(err).max())}


def figure_profile():
    """Wall-normal velocity profile at Re_x = 1e5 vs the Blasius similarity profile."""
    from tensorfvm.rans_ship_resistance.boundary_layer import march_boundary_layer

    stations = 401
    x = np.linspace(1.0 / stations, 1.0, stations)
    nu = 1.0 / REFERENCE_RE_X
    result = march_boundary_layer(x, np.ones_like(x), nu, model="laminar",
                                  trip_x=None, ny=512, stretching=8.0)
    y = np.asarray(result.y)          # wall-normal grid of the last station
    u = np.asarray(result.u)          # profile at the last station
    ue = float(np.asarray(result.ue)[-1])
    eta = y * math.sqrt(ue / (nu * x[-1]))
    sol = blasius_solution(eta_max=12.0, n=601)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(np.asarray(sol["fp"]), np.asarray(sol["eta"]), "--", lw=1.4,
            label="Blasius similarity")
    ax.plot(u / ue, eta, "-", lw=1.6, label="TensorFVM march")
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$u/U_e$")
    ax.set_ylabel(r"$\eta = y\sqrt{U_e/\nu x}$")
    ax.set_title(r"boundary-layer profile at $Re_x=10^5$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_profile_vs_blasius.png", dpi=150)
    plt.close(fig)
    interp = np.interp(eta, np.asarray(sol["eta"]), np.asarray(sol["fp"]))
    inside = eta <= 4.0
    if inside.sum() < 2:            # march grid may start above eta = 4
        inside = eta <= eta.max()
    dev = float(np.max(np.abs(u[inside] / ue - interp[inside])))
    return {"max_profile_deviation_eta_le_4": dev,
            "eta_range": [float(eta.min()), float(eta.max())]}


def figure_sensitivity(sensitivity):
    rows = sensitivity["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ny = [r["ny"] for r in rows[:3]]
    cf_err = [100 * r["cf_relative_error"] for r in rows[:3]]
    h_err = [100 * r["H_relative_error"] for r in rows[:3]]
    ax.plot(ny, cf_err, "o-", label=r"$c_f$ error")
    ax.plot(ny, h_err, "s-", label=r"$H$ error")
    ax.axhspan(-3, 3, color="0.85", zorder=0)
    ax.set_xlabel("wall-normal nodes")
    ax.set_ylabel("error vs Blasius [%]")
    ax.set_title("wall-normal refinement (march)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    st = [r["stations"] for r in rows]
    ax.plot(st[3], 100 * rows[3]["cf_relative_error"], "o",
            label="stations = 101")
    ax.plot(st[1], 100 * rows[1]["cf_relative_error"], "o", label="stations = 401")
    ax.plot(st[4], 100 * rows[4]["cf_relative_error"], "o",
            label="stations = 1601")
    ax.axhspan(-3, 3, color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("streamwise stations")
    ax.set_ylabel(r"$c_f$ error vs Blasius [%]")
    ax.set_title("streamwise refinement (march, ny = 256)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_march_sensitivity.png", dpi=150)
    plt.close(fig)


def figure_fvm_traces(traces):
    if not traces:
        return None
    ref = blasius_reference()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for tag, entries in traces.items():
        it = [e["iteration"] for e in entries]
        ax = axes[0]
        ax.plot(it, [e["cf_trailing"] for e in entries], "o-", ms=3, label=tag)
        axes[1].plot(it, [e["momentum"] for e in entries], "o-", ms=3, label=tag)
    ax = axes[0]
    ax.axhline(ref["cf"], color="k", ls="--", lw=1.2,
               label="Blasius target %.6g" % ref["cf"])
    ax.set_yscale("log")
    ax.set_xlabel("SIMPLE iteration")
    ax.set_ylabel(r"trailing-station $c_f$")
    ax.set_title("2-D FVM: local $c_f$ history (not converged)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.set_yscale("log")
    ax.set_xlabel("SIMPLE iteration")
    ax.set_ylabel("momentum residual")
    ax.set_title("2-D FVM: residual history")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_fvm_partial_convergence.png", dpi=150)
    plt.close(fig)
    return True


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    baseline_march()
    sensitivity = json.loads((RUNS / "march-sensitivity.json").read_text())
    traces = fvm_traces()
    summary = {
        "figure1": figure_local_cf(sensitivity),
        "figure2": figure_profile(),
        "figure4_written": figure_fvm_traces(traces),
    }
    figure_sensitivity(sensitivity)
    (RUNS / "figure-summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()