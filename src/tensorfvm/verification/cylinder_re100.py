"""Unbounded circular cylinder at Re = 100: von Karman shedding, Cd and St.

Two solver backends produce the same workpiece:

``immersed`` (production)
    ``tensorfvm.immersed`` masked MAC fractional step (geometric-multigrid
    preconditioned Poisson) on a Cartesian grid whose cylinder is the mask
    staircase from :func:`tensorfvm.immersed.geometry.circular_cylinder`.
    The hydrodynamic force comes from a control-volume momentum budget
    (:mod:`tensorfvm.immersed.forces`): the masked projection never applies an
    explicit body force, so this is the only *consistent* way to read the force
    off the field.  Time to solution is ~0.14 s/step at 20 cells per diameter
    (600 x 400), i.e. minutes instead of hours.

``body-fitted`` (legacy)
    ``tensorfvm.body_fitted.BodyFittedSolver`` SIMPLE/URANS on a 20D x 20D
    O-grid (the module delivered by mission m20955).  Kept for continuity; it
    is ~50x more expensive per step and is not used for the production numbers.

Statistics are computed by one code path for both backends
(:func:`shedding_statistics`) and are *gated*: if the lift signal never grows
beyond ``CL_AMPLITUDE_GATE`` no Strouhal number is reported at all (the
predecessor module happily read a St = 0.267 out of 1e-7 rounding noise).  The
primary St is the zero-crossing estimate; an independent zero-padded FFT
estimate must agree with it to ``CROSS_CHECK_TOL`` or the run is flagged.

Literature anchor (``artifacts/case-cylinder-re100/reference/reference.json``)
    Cd_mean = 1.34 +/- 5 %, St = 0.164 +/- 3 %
    Braza et al., JFM 165 (1986) 79-130, doi:10.1017/S0022112086003014
    Schaefer & Turek, NNFM 52 (1996), doi:10.1007/978-3-322-89849-4_39

Usage
  PYTHONPATH=src python -m tensorfvm.verification.cylinder_re100 \\
      --backend immersed --cells-per-diameter 20 --dt 0.02 --t-end 100 \\
      --output runs/case-cylinder-re100/cpd20
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..body_fitted import BodyFittedSolver
from ..solver import SolverConfig

# --------------------------------------------------------------- reference band
#: Literature anchor.  The JSON file below is the single source of truth; these
#: constants are the fallback used when the module is imported outside a checkout
#: and are kept bit-identical to the file by
#: ``tests/test_case_cylinder_reference_binding.py``.
REFERENCE = dict(cd_mean=1.34, st=0.164, cl_amplitude=0.32)
REFERENCE_BAND = dict(cd_mean=0.07, st=0.005)
REFERENCE_CITATION = (
    "Braza et al. JFM 165 (1986) 79-130, doi:10.1017/S0022112086003014; "
    "Schafer & Turek, NNFM 52 (1996), doi:10.1007/978-3-322-89849-4_39"
)
REFERENCE_URL = ("https://wwwold.mathematik.tu-dortmund.de/~featflow/en/"
                 "benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html")
ANCHOR_SLUG = "GAP-DIFFLBM-EXT-LIT-CYL"

#: A lift record whose amplitude stays below this is not a shedding run.
CL_AMPLITUDE_GATE = 0.05
#: Allowed relative disagreement between the two independent St estimates.
CROSS_CHECK_TOL = 0.05
#: Minimum number of shedding periods inside the statistics window.
MIN_PERIODS = 5.0


def reference_path() -> Path | None:
    """Repo-tracked reference file (``artifacts/case-cylinder-re100/...``)."""
    candidate = (Path(__file__).resolve().parents[3] / "artifacts" /
                 "case-cylinder-re100" / "reference" / "reference.json")
    return candidate if candidate.is_file() else None


def load_reference() -> dict:
    """Reference values + band, preferring the repo-tracked JSON anchor."""
    path = reference_path()
    if path is None:
        return dict(values=dict(REFERENCE), tolerance_band=dict(REFERENCE_BAND),
                    citation=REFERENCE_CITATION, source_url=REFERENCE_URL,
                    file=None)
    data = json.loads(path.read_text())
    return dict(values=data["values"], tolerance_band=data["tolerance_band"],
                citation=data.get("citation", REFERENCE_CITATION),
                source_url=data.get("source_url", REFERENCE_URL), file=str(path))


# --------------------------------------------------------------------- statistics
def _zero_crossings(t: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Linearly interpolated times at which ``signal`` changes sign."""
    s = signal - np.mean(signal)
    sign = np.signbit(s)
    idx = np.nonzero(sign[:-1] != sign[1:])[0]
    if idx.size == 0:
        return np.empty(0)
    t0, t1 = t[idx], t[idx + 1]
    s0, s1 = s[idx], s[idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(s1 != s0, -s0 / (s1 - s0), 0.0)
    return t0 + frac * (t1 - t0)


def spectral_frequency(t: np.ndarray, signal: np.ndarray, *, pad: int = 16
                       ) -> tuple[float, float]:
    """Peak frequency of ``signal`` by a zero-padded, Hann-windowed FFT.

    Returns ``(frequency, peak_magnitude)``.  The discrete peak is refined with a
    three-point parabolic fit on the log magnitude, so the resolution is not
    limited by ``1/T`` (a six-period record has ``1/T = 0.027 Hz``, i.e. 16 % of
    the St = 0.164 peak; the interpolated peak lands within a few 1e-3).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size < 8:
        return 0.0, 0.0
    dt = float(np.median(np.diff(t)))
    y = (y - np.mean(y)) * np.hanning(y.size)
    n = 1 << int(math.ceil(math.log2(y.size * pad)))
    spec = np.abs(np.fft.rfft(y, n))
    freqs = np.fft.rfftfreq(n, d=dt)
    lo = np.searchsorted(freqs, 1.0 / (t.size * dt) * 0.5)  # skip the trend bin
    if lo >= spec.size - 2:
        lo = 1
    k = int(lo + np.argmax(spec[lo:]))
    if 0 < k < spec.size - 1:
        a, b, c = (math.log(max(spec[k - 1], 1e-300)),
                   math.log(max(spec[k], 1e-300)),
                   math.log(max(spec[k + 1], 1e-300)))
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    return float((k + delta) * (freqs[1] - freqs[0])), float(spec[k])


def bandpass(t: np.ndarray, signal: np.ndarray, f0: float, *, rel_band: float = 0.4
             ) -> np.ndarray:
    """Zero-phase FFT band-pass of ``signal`` around ``f0`` (+-``rel_band``).

    Used only to make the *crossing* estimate of the shedding period robust
    against force noise; the spectral peak ``f0`` is reported separately from the
    raw (unfiltered) record.
    """
    y = np.asarray(signal, dtype=float)
    y = y - np.mean(y)
    if y.size < 8 or f0 <= 0.0:
        return y
    dt = float(np.median(np.diff(t)))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    spec = np.fft.rfft(y)
    keep = (freqs >= f0 * (1.0 - rel_band)) & (freqs <= f0 * (1.0 + rel_band))
    return np.fft.irfft(spec * keep, y.size)


def shedding_statistics(times, cd, cl, *, diameter: float = 1.0,
                        velocity: float = 1.0, max_periods: float = 10.0,
                        min_periods: float = MIN_PERIODS,
                        cl_gate: float = CL_AMPLITUDE_GATE) -> dict:
    """Cd_mean / Cl amplitude / St from a force history, with honest gates.

    Steps
    -----
    1. Shedding gate: half peak-to-peak of the *second half* of the record must
       exceed ``cl_gate``.  Otherwise ``shedding_detected=False`` and no
       frequency is reported (``st=None``).
    2. First period estimate from zero crossings of the second half.
    3. Statistics window: the last ``max_periods`` periods of the record, but at
       least ``min_periods``; the startup transient is therefore excluded by
       construction rather than by a hand-picked warmup index.  Fewer than
       ``min_periods`` periods available => ``window_ok=False``.
    4. Primary St = zero-crossing period (``T = 2 * mean(dt_cross)``); the FFT
       estimate is an independent cross-check that must agree to
       ``CROSS_CHECK_TOL``.
    """
    t = np.asarray(times, dtype=float)
    cd = np.asarray(cd, dtype=float)
    cl = np.asarray(cl, dtype=float)
    out: dict = dict(n_samples=int(t.size), t_start=float(t[0]), t_end=float(t[-1]),
                     shedding_detected=False, st=None, st_fft=None,
                     st_zerocross=None, cl_amplitude=None, cd_mean=None,
                     gate_reasons=[], freq_cross_check=None)
    if t.size < 32:
        out["gate_reasons"].append(f"record too short ({t.size} samples)")
        out.update(cd_mean=float(np.mean(cd)), cd_min=float(np.min(cd)),
                   cd_max=float(np.max(cd)), cl_amplitude=0.5 * float(np.ptp(cl)),
                   cl_amplitude_second_half=0.5 * float(np.ptp(cl)),
                   statistics_window=[float(t[0]), float(t[-1])],
                   statistics_periods=0.0, window_ok=False)
        return out
    half = t.size // 2
    cl_half = cl[half:]
    amp_half = 0.5 * float(np.ptp(cl_half))
    out["cl_amplitude_second_half"] = amp_half
    if not np.isfinite(amp_half) or amp_half < cl_gate:
        # No periodic regime: the mean drag is still defined over the tail of the
        # record, but no frequency may be reported (the m20955 module read St out
        # of 1e-7 rounding noise here).
        out["gate_reasons"].append(
            f"no vortex shedding: lift half-amplitude {amp_half:.3e} < gate {cl_gate:.3g}")
        tail = t >= (t[-1] - 0.25 * (t[-1] - t[0]))
        out.update(cd_mean=float(np.mean(cd[tail])), cd_min=float(np.min(cd[tail])),
                   cd_max=float(np.max(cd[tail])),
                   cl_amplitude=0.5 * float(np.ptp(cl[tail])),
                   statistics_window=[float(t[tail][0]), float(t[-1])],
                   statistics_periods=0.0, window_ok=False,
                   cl_amplitude_second_half=amp_half)
        return out

    # First period estimate from the *spectral* peak: raw zero crossings are not
    # usable here because force noise multiplies the crossing count, which would
    # shrink the statistics window by an arbitrary factor.
    freq0, _ = spectral_frequency(t[half:], cl_half)
    if freq0 <= 0.0:
        out["gate_reasons"].append("no spectral peak found in the lift record")
        return out
    period0 = 1.0 / freq0
    available = (t[-1] - t[0]) / period0
    n_periods = min(max_periods, available)
    window_ok = n_periods >= min_periods
    t_stat0 = float(t[-1] - n_periods * period0)
    mask = t >= t_stat0
    if int(mask.sum()) < 32:
        out["gate_reasons"].append("statistics window has fewer than 32 samples")
        return out

    tw, cdw, clw = t[mask], cd[mask], cl[mask]
    out.update(shedding_detected=True, statistics_window=[float(tw[0]), float(tw[-1])],
               statistics_periods=float(n_periods), window_ok=bool(window_ok),
               available_periods=float(available),
               cd_mean=float(np.mean(cdw)), cd_min=float(np.min(cdw)),
               cd_max=float(np.max(cdw)),
               cl_amplitude=0.5 * float(np.ptp(clw)),
               cl_mean=float(np.mean(clw)))
    if not window_ok:
        out["gate_reasons"].append(
            f"only {available:.2f} shedding periods in the record (< {min_periods})")

    freq, _ = spectral_frequency(tw, clw)
    st_fft = freq * diameter / velocity
    out["st_fft"] = float(st_fft)
    # Crossing estimate on the band-passed signal: raw force noise puts extra
    # crossings inside one cycle (a 6 % noise-to-signal lift record doubles the
    # crossing count without this filter).
    filtered = bandpass(tw, clw, freq) if freq > 0.0 else clw - np.mean(clw)
    cross = _zero_crossings(tw, filtered)
    if cross.size >= 4:
        periods = np.diff(cross)
        st_zc = (diameter / velocity) / (2.0 * float(np.mean(periods)))
        out.update(st_zerocross=float(st_zc), periods=len(periods),
                   period_std_over_mean=float(np.std(periods) / np.mean(periods)))
    if out["st_zerocross"] is None:
        out["st"] = float(st_fft)
        out["gate_reasons"].append("zero-crossing estimate unavailable; FFT used")
    else:
        out["st"] = float(out["st_zerocross"])
        rel = abs(st_fft - out["st_zerocross"]) / out["st_zerocross"]
        out["freq_cross_check"] = float(rel)
        if rel > CROSS_CHECK_TOL:
            out["gate_reasons"].append(
                f"St estimates disagree: FFT {st_fft:.4f} vs zero-crossing "
                f"{out['st_zerocross']:.4f} ({100 * rel:.1f} % > "
                f"{100 * CROSS_CHECK_TOL:.0f} %)")
    return out


# ----------------------------------------------------------------- error judge
def compute_errors(result: dict, reference: dict | None = None) -> dict:
    """Judge every reported quantity against the literature anchor."""
    ref = reference or load_reference()
    values, bands = ref["values"], ref["tolerance_band"]
    errors: dict = {}
    for key in ("cd_mean", "st"):
        computed = result.get(key)
        band = bands[key]
        entry = dict(reference=values[key], tolerance_band=band,
                     tolerance_relative=band / values[key], computed=computed,
                     passed=False)
        if computed is None:
            entry["reason"] = "not reported (see gate_reasons)"
            entry["relative_error"] = None
        else:
            rel = abs(computed - values[key]) / values[key]
            entry["relative_error"] = float(rel)
            entry["passed"] = bool(rel <= band / values[key])
        errors[key] = entry
    # Informational: the independent FFT estimate and the lift amplitude.
    if result.get("st_fft") is not None:
        rel = abs(result["st_fft"] - values["st"]) / values["st"]
        errors["st_fft"] = dict(reference=values["st"], computed=result["st_fft"],
                                relative_error=float(rel), role="cross-check")
    if result.get("cl_amplitude") is not None and "cl_amplitude" in values:
        ref_cl = values["cl_amplitude"]
        errors["cl_amplitude"] = dict(
            reference=ref_cl, computed=result["cl_amplitude"],
            relative_error=float(abs(result["cl_amplitude"] - ref_cl) / ref_cl),
            role="informational (Braza 1986 reports Cl' ~ 0.3 at Re = 100)")
    errors["gates"] = dict(reasons=list(result.get("gate_reasons") or []),
                           shedding_detected=bool(result.get("shedding_detected")),
                           passed=not (result.get("gate_reasons") or []))
    return errors


def verdict(errors: dict) -> tuple[bool, str]:
    gates = errors.get("gates", {})
    if not gates.get("passed", False):
        return False, "GATED: " + "; ".join(gates.get("reasons", []))
    failed = [k for k in ("cd_mean", "st") if not errors[k]["passed"]]
    if failed:
        return False, "FAIL: " + ", ".join(
            f"{k} = {errors[k]['computed']:.4f} vs {errors[k]['reference']:.4f} "
            f"({100 * errors[k]['relative_error']:.2f} %)" for k in failed)
    return True, "PASS"


# ---------------------------------------------------------------- immersed backend
@dataclass
class ImmersedSpec:
    """Numerical setting of one immersed production run (all lengths in D)."""

    cells_per_diameter: int = 20
    diameter: float = 1.0
    velocity: float = 1.0
    reynolds: float = 100.0
    dt: float = 0.02
    t_end: float = 100.0
    perturbation: float = 0.1
    upstream_lengths: float = 10.0
    downstream_lengths: float = 20.0
    half_height: float = 10.0
    box: tuple = (-2.0, 3.0, -1.5, 1.5)     # x_min, x_max, y_min, y_max (in D)
    poisson_tol: float = 1e-10
    max_steps: int = 10 ** 7


def build_immersed(spec: ImmersedSpec):
    """Grid + solver + force recorder for one immersed run."""
    from ..immersed.forces import ForceRecorder, control_box
    from ..immersed.geometry import circular_cylinder
    from ..immersed.solver import ImmersedConfig, ImmersedFlow
    import torch

    d = spec.diameter
    grid = circular_cylinder(spec.cells_per_diameter, diameter=d,
                             upstream_lengths=spec.upstream_lengths,
                             downstream_lengths=spec.downstream_lengths,
                             half_height=spec.half_height)
    cfg = ImmersedConfig(
        grid=grid, viscosity=spec.velocity * d / spec.reynolds, density=1.0,
        projection="masked", inlet_mean_velocity=spec.velocity,
        inlet_bounds=(-spec.half_height * d, spec.half_height * d),
        reference_length=d, dt=spec.dt, t_end=spec.t_end,
        max_steps=spec.max_steps, poisson_tol=spec.poisson_tol)
    flow = ImmersedFlow(cfg)
    # Uniform freestream inlet (the geometry module's default is parabolic).
    flow.inlet_profile = torch.ones_like(flow.inlet_profile) * spec.velocity
    if spec.perturbation:
        xc = grid.x0 + (torch.arange(grid.nx, dtype=torch.float64) + 0.5) * grid.dx
        yf = grid.y0 + torch.arange(grid.ny + 1, dtype=torch.float64) * grid.dy
        X, Y = xc[None, :], yf[:, None]
        flow.v = flow.v + spec.perturbation * torch.exp(
            -(((X - 0.8 * d) / (0.4 * d)) ** 2 + ((Y - 0.25 * d) / (0.25 * d)) ** 2))
    x0, x1, y0, y1 = spec.box
    box = control_box(grid, x_min=x0 * d, x_max=x1 * d, y_min=y0 * d, y_max=y1 * d)
    recorder = ForceRecorder(flow, box)
    return cfg, flow, recorder, box


def run_immersed(spec: ImmersedSpec, *, log_every: float = 10.0,
                 progress=print) -> tuple[dict, object, dict]:
    """Run one immersed case, returning ``(result, flow, snapshot)``."""
    cfg, flow, recorder, box = build_immersed(spec)
    d, u = spec.diameter, spec.velocity
    q = 0.5 * cfg.density * u * u * d
    start = time.perf_counter()
    next_log = 0.0
    while flow.ledger.time < spec.t_end - 1e-12:
        flow.step()
        recorder.sample(flow.ledger.time)
        if flow.ledger.time >= next_log:
            sample = recorder.history[-1]
            progress(f"t={flow.ledger.time:7.2f} Cd={sample.fx / q:+.4f} "
                     f"Cl={sample.fy / q:+.4f} steps={flow.ledger.steps} "
                     f"umax={flow.snapshot()['u_max']:.3f}")
            next_log += log_every
    elapsed = time.perf_counter() - start
    times = np.array([s.time for s in recorder.history])
    cd = np.array([s.fx / q for s in recorder.history])
    cl = np.array([s.fy / q for s in recorder.history])
    stats = shedding_statistics(times, cd, cl, diameter=d, velocity=u)
    result = dict(backend="immersed", anchor_slug=ANCHOR_SLUG, spec=asdict(spec),
                  grid=dict(nx=cfg.grid.nx, ny=cfg.grid.ny, dx=cfg.grid.dx,
                            dy=cfg.grid.dy, cells_per_diameter=spec.cells_per_diameter),
                  control_box={**box.as_dict(),
                               "x": [spec.box[0] * d, spec.box[1] * d],
                               "y": [spec.box[2] * d, spec.box[3] * d]},
                  elapsed_s=elapsed, steps=int(flow.ledger.steps),
                  s_per_step=elapsed / max(flow.ledger.steps, 1),
                  force_history=[dict(step=None, time=float(t), drag=float(c),
                                      lift=float(l)) for t, c, l in zip(times, cd, cl)])
    result.update({k: v for k, v in stats.items() if k not in ("gate_reasons",)})
    result["gate_reasons"] = stats["gate_reasons"]
    return result, flow, field_snapshot(flow)


def field_snapshot(flow) -> dict:
    """Cell-centred vorticity/velocity snapshot for the cloud figures."""
    from ..immersed.forces import cell_velocity  # same stagger convention

    grid = flow.config.grid
    uc, vc = cell_velocity(flow)
    uc, vc = uc.numpy(), vc.numpy()
    dvdx = np.gradient(vc, grid.dx, axis=1)
    dudy = np.gradient(uc, grid.dy, axis=0)
    omega = dvdx - dudy
    solid = grid.solid.numpy()
    omega = np.where(solid, np.nan, omega)
    xc = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    yc = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dy
    speed = np.where(solid, np.nan, np.hypot(uc, vc))
    return dict(x=xc, y=yc, vorticity=omega, speed=speed)


# ------------------------------------------------------------- body-fitted backend
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
    """Legacy body-fitted driven run (kept for the m20955 entry point)."""
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
    diameter = 2.0 * config.cylinder_radius
    stats = shedding_statistics(times, cd_vals, cl_vals,
                                diameter=diameter, velocity=config.inlet_velocity)
    result = dict(backend="body-fitted", anchor_slug=ANCHOR_SLUG,
                  warmup_steps=int(warmup_steps), elapsed_s=elapsed,
                  config=asdict(config), steps=total_steps)
    result.update({k: v for k, v in stats.items() if k != "gate_reasons"})
    result["gate_reasons"] = stats["gate_reasons"]
    result["force_history"] = [{"step": f["step"], "time": f["time"],
                                "drag": f["drag"], "lift": f["lift"]}
                               for f in force_hist]
    return result, solver


# -------------------------------------------------------------------- publishing
def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float))


def publish(output: Path, result: dict, errors: dict, *, snapshot: dict | None = None,
            extra_figures: dict | None = None) -> bool:
    """Write artifacts: tables, figures, report, and the referee payload."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    hist = result["force_history"]
    times = np.array([h["time"] for h in hist])
    cd = np.array([h["drag"] for h in hist])
    cl = np.array([h["lift"] for h in hist])

    slim = {k: v for k, v in result.items() if k != "force_history"}
    slim["force_history_rows"] = len(hist)
    _write_json(output / "result.json", slim)
    _write_json(output / "errors.json", errors)
    with (output / "force_history.csv").open("w") as fh:
        fh.write("time,drag,lift\n")
        for t, c, l in zip(times, cd, cl):
            fh.write(f"{t:.6f},{c:.8f},{l:.8f}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = result.get("statistics_window")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.plot(times, cd, "b-", lw=0.7, label="Cd(t)")
    ax1.axhline(ref["values"]["cd_mean"], color="r", ls="--",
                label=f"literature Cd = {ref['values']['cd_mean']}")
    if result.get("cd_mean") is not None:
        ax1.axhline(result["cd_mean"], color="g", ls=":",
                    label=f"computed mean = {result['cd_mean']:.4f}")
    ax2.plot(times, cl, "r-", lw=0.7, label="Cl(t)")
    for ax in (ax1, ax2):
        if window:
            ax.axvspan(*window, color="0.85", alpha=0.6, lw=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("Cd")
    ax2.set_ylabel("Cl")
    ax2.set_xlabel("t U / D")
    fig.suptitle(f"Cylinder Re=100 ({result['backend']} backend): "
                 f"grey band = statistics window")
    fig.tight_layout()
    fig.savefig(output / "force_time_series.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if window:
        mask = (times >= window[0]) & (times <= window[-1])
    else:
        mask = np.ones_like(times, dtype=bool)
    sig = cl[mask]
    if sig.size >= 8:
        dt = float(np.median(np.diff(times[mask])))
        n = 1 << int(math.ceil(math.log2(sig.size * 16)))
        spec = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(sig.size), n))
        freqs = np.fft.rfftfreq(n, d=dt)
        ax.plot(freqs, spec / max(spec.max(), 1e-30), "b-", lw=0.9)
        ref_f = ref["values"]["st"] * result.get("spec", {}).get("velocity", 1.0) / (
            result.get("spec", {}).get("diameter", 1.0))
        ax.axvline(ref_f, color="r", ls="--", label=f"literature St = {ref['values']['st']}")
        if result.get("st") is not None:
            ax.axvline(result["st"], color="g", ls=":", label=f"computed St = {result['st']:.4f}")
        ax.set_xlim(0, max(4 * ref_f, 0.6))
        ax.legend(fontsize=8)
    ax.set(xlabel="frequency f D / U  (= St)", ylabel="normalised |FFT(Cl)|",
           title="Lift spectrum over the statistics window")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "lift_spectrum.png", dpi=150)
    plt.close(fig)

    keys = [k for k in ("cd_mean", "st") if errors.get(k, {}).get("computed") is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if keys:
        labels = {"cd_mean": "Cd_mean", "st": "St"}
        rel = [100 * errors[k]["relative_error"] for k in keys]
        tol = [100 * errors[k]["tolerance_relative"] for k in keys]
        x = np.arange(len(keys))
        ax.bar(x - 0.2, rel, 0.4, label="relative error (%)", color="tab:blue")
        ax.bar(x + 0.2, tol, 0.4, label="tolerance band (%)", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in keys])
        for xi, value in zip(x - 0.2, rel):
            ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(15.0, color="k", ls=":", lw=0.8, label="platform 15 % floor")
    ax.set_ylabel("percentage (%)")
    ax.set_title("Re=100 cylinder: computed vs literature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output / "error_bars.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(cl, cd, "b-", lw=0.5, alpha=0.7)
    ax.set(xlabel="Cl", ylabel="Cd", title="Cd-Cl phase portrait")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "phase_plot.png", dpi=150)
    plt.close(fig)

    if snapshot is not None:
        x, y = snapshot["x"], snapshot["y"]
        w = snapshot["vorticity"]
        lim = np.nanpercentile(np.abs(w), 99.0) if np.isfinite(w).any() else 1.0
        fig, ax = plt.subplots(figsize=(11, 5.2))
        im = ax.pcolormesh(x, y, w, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set_aspect("equal")
        ax.set(xlabel="x / D", ylabel="y / D",
               title="Vorticity at the end of the run (masked cylinder, staircase body)")
        theta = np.linspace(0, 2 * np.pi, 200)
        radius = result.get("spec", {}).get("diameter", 1.0) / 2.0
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", lw=1.0)
        fig.colorbar(im, ax=ax, label="omega D / U")
        fig.tight_layout()
        fig.savefig(output / "vorticity_cloud.png", dpi=150)
        plt.close(fig)
        sp = snapshot["speed"]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        im = ax.pcolormesh(x, y, sp, cmap="viridis", shading="auto")
        ax.set_aspect("equal")
        ax.set(xlabel="x / D", ylabel="y / D", title="Velocity magnitude |u| / U")
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "w-", lw=1.0)
        fig.colorbar(im, ax=ax, label="|u| / U")
        fig.tight_layout()
        fig.savefig(output / "speed_cloud.png", dpi=150)
        plt.close(fig)

    for name, path in (extra_figures or {}).items():
        if Path(path).is_file():
            import shutil
            shutil.copyfile(path, output / name)

    passed, message = verdict(errors)
    referee = dict(
        anchor_slug=ANCHOR_SLUG, case="cylinder-re100-unbounded",
        backend=result.get("backend"), repo="TensorFVM",
        reference=dict(values=ref["values"], tolerance_band=ref["tolerance_band"],
                       citation=ref["citation"], source_url=ref["source_url"],
                       file=ref["file"]),
        got={k: errors[k]["computed"] for k in ("cd_mean", "st")},
        errors={k: dict(reference=errors[k]["reference"],
                        computed=errors[k]["computed"],
                        relative_error=errors[k]["relative_error"],
                        tolerance_relative=errors[k]["tolerance_relative"],
                        passed=errors[k]["passed"]) for k in ("cd_mean", "st")},
        cross_check={k: errors[k] for k in ("st_fft", "cl_amplitude") if k in errors},
        gates=errors["gates"], passed=passed, verdict=message,
        run=dict(steps=result.get("steps"), elapsed_s=result.get("elapsed_s"),
                 grid=result.get("grid"), spec=result.get("spec"),
                 statistics_window=result.get("statistics_window")))
    _write_json(output / "benchmark.json", referee)

    lines = [
        "# Cylinder Re=100 (unbounded): vortex-shedding verification",
        "",
        f"Backend: **{result.get('backend')}**   Verdict: **{message}**",
        "",
        "## 1. Physical problem",
        "",
        "Two-dimensional incompressible flow past a circular cylinder, "
        "Re = U D / nu = 100: laminar vortex shedding. Domain "
        f"{result.get('spec', {}).get('upstream_lengths', 10)}D upstream, "
        f"{result.get('spec', {}).get('downstream_lengths', 20)}D downstream, "
        f"+-{result.get('spec', {}).get('half_height', 10)}D laterally "
        f"(blockage {100 / (2 * result.get('spec', {}).get('half_height', 10)):.1f} %).",
        "",
        "## 2. Literature anchor",
        "",
        "| Quantity | Reference | Band | Source |",
        "|---|---|---|---|",
        f"| Cd_mean | {ref['values']['cd_mean']:.3f} | +/-{ref['tolerance_band']['cd_mean']:.3f} "
        f"({100 * ref['tolerance_band']['cd_mean'] / ref['values']['cd_mean']:.1f} %) | Braza 1986; Schaefer-Turek 1996 |",
        f"| St | {ref['values']['st']:.3f} | +/-{ref['tolerance_band']['st']:.3f} "
        f"({100 * ref['tolerance_band']['st'] / ref['values']['st']:.1f} %) | Braza 1986; Schaefer-Turek 1996 |",
        "",
        f"Citation: {ref['citation']}",
        "",
        "## 3. Results",
        "",
        "| Quantity | Computed | Reference | Rel. error | Status |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("cd_mean", "Cd_mean"), ("st", "St")):
        e = errors[key]
        if e["computed"] is None:
            lines.append(f"| {label} | not reported | {e['reference']:.4f} | - | FAIL |")
        else:
            lines.append(f"| {label} | {e['computed']:.4f} | {e['reference']:.4f} | "
                         f"{100 * e['relative_error']:.2f} % | "
                         f"{'PASS' if e['passed'] else 'FAIL'} |")
    if "st_fft" in errors:
        e = errors["st_fft"]
        lines.append(f"| St (FFT cross-check) | {e['computed']:.4f} | {e['reference']:.4f} | "
                     f"{100 * e['relative_error']:.2f} % | cross-check |")
    if "cl_amplitude" in errors:
        e = errors["cl_amplitude"]
        lines.append(f"| Cl amplitude | {e['computed']:.4f} | {e['reference']:.4f} | "
                     f"{100 * e['relative_error']:.2f} % | informational |")
    lines += [
        "",
        f"Record: {result.get('n_samples')} force samples, "
        f"t in [{result.get('t_start', 0):.2f}, {result.get('t_end', 0):.2f}] tU/D; "
        f"statistics window {result.get('statistics_window')} "
        f"({result.get('statistics_periods', 0):.2f} periods).",
        "",
        "## 4. Gates",
        "",
    ]
    lines += [f"- {reason}" for reason in (errors["gates"]["reasons"] or ["all gates passed"])]
    lines += ["", "## 5. Figures", ""]
    for name in ("force_time_series.png", "lift_spectrum.png", "error_bars.png",
                 "phase_plot.png", "vorticity_cloud.png", "speed_cloud.png"):
        if (output / name).is_file():
            lines.append(f"![{name}]({name})")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="immersed",
                        choices=("immersed", "body-fitted"))
    parser.add_argument("--cells-per-diameter", type=int, default=20)
    parser.add_argument("--grid", default="128x48", help="body-fitted O-grid")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=None,
                        help="body-fitted step count (overrides --t-end)")
    parser.add_argument("--warmup", type=int, default=200,
                        help="body-fitted only; the immersed backend derives the "
                             "statistics window from the shedding signal")
    parser.add_argument("--box", nargs=4, type=float, default=[-2.0, 3.0, -1.5, 1.5],
                        help="control-volume box in diameters: x_min x_max y_min y_max")
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--log-every", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.backend == "immersed":
        spec = ImmersedSpec(cells_per_diameter=args.cells_per_diameter, dt=args.dt,
                            t_end=args.t_end, box=tuple(args.box),
                            perturbation=args.perturbation)
        result, _flow, snapshot = run_immersed(spec, log_every=args.log_every)
    else:
        steps = args.steps if args.steps is not None else int(round(args.t_end / args.dt))
        config = _build_config(args.grid, args.dt, steps)
        result, _solver = run_transient(config, warmup_steps=args.warmup)
        snapshot = None
    errors = compute_errors(result)
    passed = publish(args.output, result, errors, snapshot=snapshot)
    print("=" * 70)
    for key, label in (("cd_mean", "Cd_mean"), ("st", "St")):
        e = errors[key]
        if e["computed"] is None:
            print(f"  {label}: not reported ({e.get('reason')})")
        else:
            print(f"  {label}: {e['computed']:.4f} (ref {e['reference']:.4f}, "
                  f"err {100 * e['relative_error']:.2f} %, "
                  f"{'PASS' if e['passed'] else 'FAIL'})")
    print(f"  verdict: {verdict(errors)[1]}")
    print("=" * 70)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())