"""Unbounded cylinder at Re=100: vortex shedding (Cd, St) against literature.

This module runs a transient body-fitted SIMPLE/URANS simulation of 2D flow
past a circular cylinder at Re=100 (based on diameter D and free-stream velocity
U_inf).  The flow is unsteady and exhibits periodic von Karman vortex shedding.

Key quantities:
  - Cd_mean : time-averaged drag coefficient
  - St      : Strouhal number = f * D / U_inf (from lift-coefficient FFT)

Literature reference band (Braza 1986; Schafer & Turek 1996 DFG 2D-2):
  Cd_mean = 1.34 +/- 0.07  (~5 % band)
  St      = 0.164 +/- 0.005 (~3 % band)

Usage
  python -m tensorfvm.verification.cylinder_re100 --output results/cyl-re100 \
      --grid 128x48 --steps 2000 --dt 0.01
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..solver import SolverConfig
from ..body_fitted import BodyFittedSolver

# Literature reference band
# Braza et al. JFM 165 (1986) 79-130, doi:10.1017/S0022112086003014
# Schafer & Turek, NNMFM 52 (1996), doi:10.1007/978-3-322-89849-4_39
REFERENCE = dict(cd_mean=1.34, st=0.164)
REFERENCE_BAND = dict(cd_mean=0.07, st=0.005)
REFERENCE_CITATION = (
    "Braza et al. JFM 165 (1986) 79-130, doi:10.1017/S0022112086003014; "
    "Schafer & Turek, NNMFM 52 (1996), doi:10.1007/978-3-322-89849-4_39"
)
REFERENCE_URL = "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html"


def _build_config(grid, dt, max_steps, device="cpu"):
    nx, ny = (int(v) for v in grid.split("x"))
    return SolverConfig(
        mesh_type="body-fitted", outer_boundary="far-field",
        nx=nx, ny=ny, length=20.0, height=20.0,
        cylinder_x=10.0, cylinder_y=10.0, cylinder_radius=0.5,
        inlet_velocity=1.0, reynolds=100.0, density=1.0,
        time_step=dt, max_iterations=max_steps, inner_iterations=3,
        tolerance=1e-5, velocity_relaxation=0.6, pressure_relaxation=0.4,
        device=device, inlet_profile="uniform",
        initial_perturbation=0.01,
    )


def run_transient(config, warmup_steps=200):
    solver = BodyFittedSolver(config)
    start = time.perf_counter()
    total_steps = config.max_iterations
    for step in range(total_steps):
        solver.step()
        if (step + 1) % 200 == 0:
            h = solver.history[-1]
            fh = solver.force_history[-1]
            print(f"step {step+1}/{total_steps}: t={h['time']:.4f}, "
                  f"Cd={fh['drag']:.4f}, Cl={fh['lift']:.4f}, "
                  f"res={h['momentum']:.2e}", flush=True)
    elapsed = time.perf_counter() - start
    force_hist = solver.force_history
    times = np.array([f["time"] for f in force_hist])
    cd_vals = np.array([f["drag"] for f in force_hist])
    cl_vals = np.array([f["lift"] for f in force_hist])
    if warmup_steps < len(force_hist):
        cd_stat = cd_vals[warmup_steps:]
        cl_stat = cl_vals[warmup_steps:]
        t_stat = times[warmup_steps:]
    else:
        cd_stat = cd_vals
        cl_stat = cl_vals
        t_stat = times
    cd_mean = float(np.mean(cd_stat))
    dt_phys = config.time_step
    st = 0.0
    if len(cl_stat) > 4:
        cl_detrend = cl_stat - np.mean(cl_stat)
        spectrum = np.fft.rfft(cl_detrend)
        freqs = np.fft.rfftfreq(len(cl_detrend), d=dt_phys)
        mask = freqs > 0.01
        if mask.any():
            peak_idx = np.argmax(np.abs(spectrum[mask]))
            freq = freqs[mask][peak_idx]
            st = freq * (2 * config.cylinder_radius) / config.inlet_velocity
    cl_amp = float(0.5 * (np.max(cl_stat) - np.min(cl_stat)))
    result = dict(
        cd_mean=cd_mean, st=st, cl_amplitude=cl_amp,
        cd_min=float(np.min(cd_stat)), cd_max=float(np.max(cd_stat)),
        cl_min=float(np.min(cl_stat)), cl_max=float(np.max(cl_stat)),
        n_samples=len(cd_stat), t_start=float(t_stat[0]), t_end=float(t_stat[-1]),
        elapsed_s=elapsed, config=asdict(config),
        force_history=[{"step": f["step"], "time": f["time"],
                        "drag": f["drag"], "lift": f["lift"]} for f in force_hist],
    )
    return result, solver


def compute_errors(result):
    errors = {}
    for key in ("cd_mean", "st"):
        computed, ref, band = result[key], REFERENCE[key], REFERENCE_BAND[key]
        rel_error = abs(computed - ref) / ref
        errors[key] = dict(computed=computed, reference=ref, tolerance_band=band,
                           relative_error=rel_error, passed=bool(rel_error <= band / ref))
    return errors


def publish(output, result, errors):
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    (output / "errors.json").write_text(json.dumps(errors, indent=2))
    hist = result["force_history"]
    with (output / "force_history.csv").open("w") as f:
        f.write("step,time,drag,lift\n")
        for row in hist:
            f.write(f"{row['step']},{row['time']:.6f},{row['drag']:.6f},{row['lift']:.6f}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    times = [h["time"] for h in hist]
    ax1.plot(times, [h["drag"] for h in hist], "b-", lw=0.8)
    ax1.axhline(REFERENCE["cd_mean"], color="r", ls="--", label=f"ref Cd={REFERENCE['cd_mean']}")
    ax1.set_ylabel("Cd"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(times, [h["lift"] for h in hist], "r-", lw=0.8)
    ax2.set_ylabel("Cl"); ax2.set_xlabel("time (s)"); ax2.grid(alpha=0.3)
    fig.suptitle("Cylinder Re=100: force coefficients vs time")
    fig.tight_layout(); fig.savefig(output / "force_time_series.png", dpi=160); plt.close(fig)
    cl_vals = np.array([h["lift"] for h in hist])
    dt_phys = result["config"]["time_step"]
    cl_detrend = cl_vals - np.mean(cl_vals)
    spectrum = np.fft.rfft(cl_detrend)
    freqs = np.fft.rfftfreq(len(cl_detrend), d=dt_phys)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, np.abs(spectrum), "b-", lw=0.8)
    ax.axvline(REFERENCE["st"] / (2 * result["config"]["cylinder_radius"]),
               color="r", ls="--", label=f"ref St={REFERENCE['st']}")
    ax.set(xlabel="frequency (Hz)", ylabel="|FFT(Cl)|", title="Lift coefficient spectrum")
    ax.set_xlim(0, 0.5); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "lift_spectrum.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["cd_mean", "st"]; labels = ["Cd_mean", "St"]
    rel_errors = [100 * errors[k]["relative_error"] for k in keys]
    tolerances = [100 * errors[k]["tolerance_band"] / errors[k]["reference"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, rel_errors, 0.4, label="relative error (%)", color="tab:blue")
    ax.bar(x + 0.4, tolerances, 0.4, label="tolerance band (%)", color="tab:orange")
    ax.set_xticks(x + 0.2); ax.set_xticklabels(labels)
    ax.set_ylabel("percentage (%)"); ax.set_title("Re=100 cylinder: error vs literature")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(output / "error_bars.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([h["lift"] for h in hist], [h["drag"] for h in hist], "b-", lw=0.5, alpha=0.7)
    ax.set(xlabel="Cl", ylabel="Cd", title="Cd-Cl phase plot"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "phase_plot.png", dpi=160); plt.close(fig)
    passed_all = all(errors[k]["passed"] for k in keys)
    pm = lambda k: "PASS" if errors[k]["passed"] else "FAIL"
    report = (
        f"# Cylinder Re=100: Vortex Shedding Verification\n\n"
        f"## Physical Problem\n\n"
        f"2D incompressible flow past a circular cylinder at Re=100.\n"
        f"Domain: 20D x 20D, cylinder at (10,10), D=1.0 m, U=1.0 m/s, nu=0.01 m2/s.\n"
        f"Solver: body-fitted O-grid SIMPLE/URANS, implicit Euler.\n\n"
        f"## Literature Reference\n\n"
        f"| Quantity | Reference | Tolerance | Source |\n"
        f"|----------|-----------|-----------|--------|\n"
        f"| Cd_mean  | {REFERENCE['cd_mean']:.3f} | +/-{REFERENCE_BAND['cd_mean']:.3f} | Braza 1986; Schafer-Turek 1996 |\n"
        f"| St       | {REFERENCE['st']:.3f} | +/-{REFERENCE_BAND['st']:.3f} | Braza 1986; Schafer-Turek 1996 |\n\n"
        f"Citation: {REFERENCE_CITATION}\n\n"
        f"## Results\n\n"
        f"| Quantity | Computed | Reference | Rel.Error | Tolerance | Status |\n"
        f"|----------|----------|-----------|-----------|-----------|--------|\n"
        f"| Cd_mean  | {result['cd_mean']:.4f} | {REFERENCE['cd_mean']:.4f} | {100*errors['cd_mean']['relative_error']:.2f}% | +/-{100*errors['cd_mean']['tolerance_band']/REFERENCE['cd_mean']:.1f}% | {pm('cd_mean')} |\n"
        f"| St       | {result['st']:.4f} | {REFERENCE['st']:.4f} | {100*errors['st']['relative_error']:.2f}% | +/-{100*errors['st']['tolerance_band']/REFERENCE['st']:.1f}% | {pm('st')} |\n\n"
        f"**Overall: {'PASS' if passed_all else 'FAIL'}**\n\n"
        f"## Force Statistics\n\n"
        f"- Cd range: [{result['cd_min']:.4f}, {result['cd_max']:.4f}]\n"
        f"- Cl range: [{result['cl_min']:.4f}, {result['cl_max']:.4f}]\n"
        f"- Cl amplitude: {result['cl_amplitude']:.4f}\n\n"
        f"## Figures\n\n"
        f"![Force time series](force_time_series.png)\n"
        f"![Lift spectrum](lift_spectrum.png)\n"
        f"![Error bars](error_bars.png)\n"
        f"![Phase plot](phase_plot.png)\n"
    )
    (output / "report.md").write_text(report)
    return passed_all


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid", default="128x48")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    config = _build_config(args.grid, args.dt, args.steps, args.device)
    result, solver = run_transient(config, warmup_steps=args.warmup)
    errors = compute_errors(result)
    passed = publish(args.output, result, errors)
    print(f"\n{'='*60}")
    print(f"Cylinder Re=100 verification: {'PASS' if passed else 'FAIL'}")
    for key in ("cd_mean", "st"):
        e = errors[key]
        print(f"  {key}: {e['computed']:.4f} (ref={e['reference']:.4f}, "
              f"err={100*e['relative_error']:.2f}%, {'PASS' if e['passed'] else 'FAIL'})")
    print(f"{'='*60}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
