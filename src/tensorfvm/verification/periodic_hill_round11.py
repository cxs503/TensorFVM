"""Round 11: closure-parametric *time-averaged* periodic-hill gate.

Anchor: ``CASE-FVM-PERIODIC-HILLS`` (TensorFVM), acceptance ``x_r/H`` and
``u/U_b`` at ``y/H = 0.5`` vs Breuer et al. (2009) within 10 %.

Why this round exists
---------------------
Round 9 (:mod:`tensorfvm.verification.periodic_hill_round9`) time-averaged the
**SA** run at 96x48 and settled that branch in the negative: the mean wall shear
never returns positive downstream of the crest (``x_r/H`` undefined), the
``y/H = 0.5`` profile is 70-112 % low and ``max nu_t/nu = 65`` is 2.5x the
turbulent-channel level of round 7 -- i.e. SA is over-diffusive at
``Re_b = 2800``.

Round 10 (:mod:`tensorfvm.verification.periodic_hill_round10`) added the SST
``k-omega`` closure and verified its *implementation* (algebraic identity,
positivity, held bulk velocity) on a coarse 48x24 grid over 300 steps, where it
showed ``x_r/H = 2.22`` and ``nu_t/nu = 7.5`` -- the right direction, but a
short transient, and explicitly *not* the acceptance gate.

The open item (round 10, section 6, item 1) is therefore a *compute* question,
not a model-class one: **time-average an SST run at full ``Re_b = 2800`` on the
same meshes and the same budget round 9 used for SA, and read both acceptance
metrics off the time mean.**  This module is the driver for that, written so the
SA branch runs through the *identical* loop, so the two closures are compared
apples to apples.

What is new here (and what is deliberately reused)
--------------------------------------------------
New: closure selection for the long-run driver (:func:`build_solver`), the
time-averaged run loop with a divergence guard (:func:`run_time_averaged_case`),
the machine-readable cross-closure table (:func:`closure_comparison`) and the
explicit gate reasons (:func:`gate_summary`).

Reused, **not** re-implemented: every *metric* is round 9's
(``time_averaged_reattachment``, ``horizontal_profile``, ``compare_profile``,
``stationarity_report``, ``averaging_window``) and the reference numbers are
round 6's (``REFERENCE``, ``REFERENCE_TOLERANCE``, ``BREUER_PROFILE_Y05``).  This
is what makes the SST verdict comparable with ``round9_96x48.json``: no
tolerance convention, no interpolation rule, no reference edit was touched by
this round.

The payload schema is round 9's, plus a ``"closure"`` block, so
:func:`closure_comparison` can read a round-9 SA artefact and a round-11 SST
artefact without translation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics
from ..periodic_hill_sst import SSTKOmega, SSTKOmegaConfig
from ..periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig
from .periodic_hill_round6 import (
    BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
)
from .periodic_hill_round9 import (
    averaging_window,
    compare_profile,
    horizontal_profile,
    stationarity_report,
    time_averaged_reattachment,
)

__all__ = [
    "CLOSURES",
    "build_solver",
    "closure_comparison",
    "gate_summary",
    "load_payload",
    "run_time_averaged_case",
    "strict_finite",
]

#: closures this driver can time-average (``sa`` = round 8/9 baseline)
CLOSURES = ("sa", "sst")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strict_finite(obj):
    """Replace non-finite floats by ``None`` so the JSON is strict (no NaN).

    Same convention as the round-9 driver: artefacts must be loadable by
    ``json.loads`` in strict mode (a ``nan`` in the JSON silently poisons every
    downstream comparison, and ``NaN != NaN`` would make a gate look "changed"
    when nothing changed).
    """
    if isinstance(obj, dict):
        return {k: strict_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_finite(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def build_solver(closure: str = "sst", *, nx: int = 96, ny: int = 48,
                 reynolds: float = 2800.0, dt: float = 0.02,
                 hill_amplitude: float = 1.0, length_m: float = 9.0,
                 channel_top_m: float = 3.036, bulk_velocity: float = 1.0,
                 convection_blend: float | None = None,
                 transport_convection_blend: float | None = None,
                 production_limit: float | None = None,
                 freestream_intensity: float | None = None,
                 body_force_relaxation: float | None = None,
                 turbulence_momentum_interval: int = 1,
                 ) -> PeriodicHillSolver:
    """A :class:`PeriodicHillSolver` with the requested eddy-viscosity closure.

    ``closure="sst"`` wires :class:`SSTKOmega` (round 10), ``closure="sa"`` wires
    :class:`SpalartAllmaras` (rounds 5-9).  Everything else -- mesh, convection
    blend, pressure Poisson, bulk-velocity controller -- is the parent solver and
    is therefore identical between the two branches.

    ``convection_blend`` is the **momentum** convection blend of the parent
    solver; the ``k``/``omega`` transport has its own blend.  Rounds 10--14
    conflated the two: the driver's single ``--convection-blend`` flag set the
    momentum blend only, while ``SSTKOmegaConfig.convection_blend`` stayed at its
    first-order default, and every artefact recorded both numbers without anyone
    comparing them (round 11: momentum 0.75, transport 0.0 -- so round 14's
    "second-order transport" next step had in fact never been run).  Round 15
    splits the knob: ``transport_convection_blend`` is passed to the closure and
    recorded, so the artefact cannot disagree with itself.  An unknown closure name is
    an error, not a silent fallback to a default: a typo must not silently
    produce an SA artefact labelled SST.
    """
    name = str(closure).strip().lower()
    if name not in CLOSURES:
        raise ValueError(f"unknown closure {closure!r}; expected one of {CLOSURES}")
    kwargs = {}
    if convection_blend is not None:
        kwargs["convection_blend"] = float(convection_blend)
    if body_force_relaxation is not None:
        kwargs["body_force_relaxation"] = float(body_force_relaxation)
    config = PeriodicHillConfig(nx=int(nx), ny=int(ny), reynolds=float(reynolds),
                                hill_amplitude=float(hill_amplitude),
                                length_m=float(length_m),
                                channel_top_m=float(channel_top_m),
                                bulk_velocity=float(bulk_velocity),
                                time_step_s=float(dt), **kwargs)
    solver = PeriodicHillSolver(config, turbulence_momentum_interval=turbulence_momentum_interval)
    if name == "sst":
        sst_cfg = SSTKOmegaConfig(production_limit=production_limit)
        if freestream_intensity is not None:
            sst_cfg.freestream_intensity = float(freestream_intensity)
        if transport_convection_blend is not None:
            # round 15: the transport blend is a closure knob, not a momentum knob
            object.__setattr__(sst_cfg, "convection_blend",
                               float(transport_convection_blend))
            SSTKOmegaConfig.__post_init__(sst_cfg)
        solver.turbulence = SSTKOmega(solver, sst_cfg)
    else:
        solver.turbulence = SpalartAllmaras(
            solver, SpalartAllmarasConfig(production_limit=production_limit))
    solver.set_time_step(float(dt))
    return solver


def _closure_block(closure, solver):
    """Machine-readable record of which closure produced an artefact."""
    turb = solver.turbulence
    block = {"name": str(closure).lower(), "class": type(turb).__name__}
    if isinstance(turb, SSTKOmega):
        cfg = turb.config
        block.update({
            "freestream_intensity": float(cfg.freestream_intensity),
            "freestream_nut_ratio": float(cfg.freestream_nut_ratio),
            "max_viscosity_ratio": float(cfg.max_viscosity_ratio),
            "transport_convection_blend": float(cfg.convection_blend),
            "momentum_convection_blend": float(solver.config.convection_blend),
            "production_limit": cfg.production_limit,
            "a1": float(cfg.a1),
        })
    elif isinstance(turb, SpalartAllmaras):
        block.update({"production_limit": getattr(turb.config, "production_limit", None)})
    return block


# --------------------------------------------------------------------------- #
# gate bookkeeping
# --------------------------------------------------------------------------- #
def gate_summary(payload: dict) -> dict:
    """Re-derive the acceptance verdict from a payload's own numbers.

    ``pass`` requires *all three* of

    1. ``x_r/H`` inside ``5.34 +- 10 %`` (and finite -- "undefined" is a fail,
       not a skip),
    2. every admissible ``y/H = 0.5`` station inside tolerance,
    3. an averaging window that is stationary within the same tolerance.

    The reasons are spelled out so a failing verdict can be quoted without
    re-reading the numbers, and so the same function can score a round-9 SA
    artefact (schema-compatible).
    """
    rea = payload.get("time_averaged") or {}
    x_r = rea.get("x_r_over_H")
    ref = REFERENCE["x_r_over_H"]
    x_r_ok = bool(x_r is not None and math.isfinite(float(x_r))
                  and abs(float(x_r) - ref) / ref <= REFERENCE_TOLERANCE)
    cmp = payload.get("profile_comparison") or {}
    n_stations = int(cmp.get("n_stations") or 0)
    n_pass = int(cmp.get("n_pass") or 0)
    profile_ok = bool(n_stations > 0 and n_pass == n_stations)
    stat = payload.get("stationarity") or {}
    spread = stat.get("x_r_over_H_block_spread_rel")
    cover = stat.get("reattachment_coverage")
    stationary = bool(spread is not None and math.isfinite(float(spread))
                      and float(spread) <= REFERENCE_TOLERANCE)

    reasons = []
    if x_r is None or not math.isfinite(float(x_r)):
        cover_txt = "undefined" if cover is None else f"{float(cover):.1%}"
        reasons.append(
            f"x_r/H undefined (reattachment crossing found in {cover_txt} of samples)")
    elif not x_r_ok:
        reasons.append(f"x_r/H = {float(x_r):.3f} vs {ref} = "
                       f"{abs(float(x_r) - ref) / ref:.1%} error > "
                       f"{REFERENCE_TOLERANCE:.0%}")
    if not profile_ok:
        reasons.append(f"y/H=0.5 profile: {n_pass}/{n_stations} stations "
                       f"inside {REFERENCE_TOLERANCE:.0%}")
    if not stationary:
        reasons.append("averaging window not stationary "
                       f"(block spread {spread!r})")
    return {
        "closure": (payload.get("closure") or {}).get("name"),
        "x_r_over_H": x_r,
        "x_r_over_H_error": (abs(float(x_r) - ref) / ref
                             if x_r is not None and math.isfinite(float(x_r)) else None),
        "profile_stations": n_stations,
        "profile_stations_pass": n_pass,
        "stationary": stationary,
        "reattachment_coverage": cover,
        "max_nu_t_ratio": rea.get("max_nu_t_ratio"),
        "pass": bool(x_r_ok and profile_ok and stationary),
        "reasons": reasons,
    }


def closure_comparison(payloads: dict) -> dict:
    """Cross-closure table from ``{name: payload}`` (round-9 SA, round-11 SST)."""
    rows = []
    for name, payload in payloads.items():
        summary = gate_summary(payload)
        cfg = payload.get("config") or {}
        rows.append({
            "closure": name,
            "mesh": [cfg.get("nx"), cfg.get("ny")],
            "end_time": cfg.get("end_time"),
            "samples": payload.get("samples"),
            "x_r_over_H": summary["x_r_over_H"],
            "x_r_over_H_error": summary["x_r_over_H_error"],
            "profile_stations_pass": f"{summary['profile_stations_pass']}/"
                                     f"{summary['profile_stations']}",
            "max_nu_t_ratio": summary["max_nu_t_ratio"],
            "reattachment_coverage": summary["reattachment_coverage"],
            "stationary": summary["stationary"],
            "gate_pass": summary["pass"],
            "reasons": summary["reasons"],
        })
    return strict_finite({
        "reference": {"x_r_over_H": REFERENCE["x_r_over_H"],
                      "tolerance": REFERENCE_TOLERANCE,
                      "profile_y_H_0.5": dict(BREUER_PROFILE_Y05)},
        "rows": rows,
        "n_pass": sum(1 for r in rows if r["gate_pass"]),
    })


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_time_averaged_case(tag, closure="sst", *, nx=96, ny=48, steps=20000,
                           sample_every=20, burn_in_steps=8000, dt=0.02,
                           reynolds=2800.0, hill_amplitude=1.0, n_blocks=4,
                           production_limit=None, convection_blend=None,
                           transport_convection_blend=None,
                           freestream_intensity=None, solver=None,
                           verbose=False, progress_every=5000,
                           out_dir=None, save_fields=False):
    """Time-average one periodic-hill run and return the round-9-schema payload.

    The loop is the round-9 loop (advance, sample every ``sample_every`` steps
    after ``burn_in_steps``, accumulate ``Statistics`` + per-block wall shear)
    with two additions that make it safe to launch on a cluster:

    * the closure is selectable (:func:`build_solver`), and the choice is
      recorded in ``payload["closure"]``;
    * a non-finite field raises inside SST transport (round 10's guard); here
      that is caught, recorded as ``diverged_at_step`` and the partial time mean
      is still analysed and written -- a diverged run is an honest FAIL with
      numbers, not a missing artefact.
    """
    if solver is None:
        solver = build_solver(closure, nx=nx, ny=ny, reynolds=reynolds, dt=dt,
                              hill_amplitude=hill_amplitude,
                              production_limit=production_limit,
                              convection_blend=convection_blend,
                              transport_convection_blend=transport_convection_blend,
                              freestream_intensity=freestream_intensity)
    config = solver.config
    stats = Statistics(solver)
    series = {"t": [], "step": [], "x_s_over_H": [], "x_r_over_H": [],
              "body_force": [], "bulk_velocity": [], "max_nu_t_ratio": [],
              "mean_nu_t_ratio": []}
    block_tau = [np.zeros(solver.mesh.tau_bottom.shape[0]) for _ in range(n_blocks)]
    block_count = np.zeros(n_blocks, dtype=int)
    block_width = max((steps - burn_in_steps) / n_blocks, 1.0)

    diverged_at = None
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            solver.advance(1)
        except FloatingPointError:
            diverged_at = step
            break
        if step >= burn_in_steps and (step - burn_in_steps) % sample_every == 0:
            sep, rea = solver.reattachment()
            nu_t_ratio = solver.turbulence.viscosity() / solver.nu
            series["t"].append(step * dt)
            series["step"].append(step)
            series["x_s_over_H"].append(float(sep))
            series["x_r_over_H"].append(float(rea))
            series["body_force"].append(float(solver.body_force))
            series["bulk_velocity"].append(float(solver.bulk_velocity()))
            series["max_nu_t_ratio"].append(float(nu_t_ratio.max()))
            series["mean_nu_t_ratio"].append(float(nu_t_ratio.mean()))
            stats.accumulate(solver)
            k = min(n_blocks - 1, int((step - burn_in_steps) / block_width))
            block_tau[k] += solver.wall_shear()[0]
            block_count[k] += 1
        if verbose and progress_every and step % progress_every == 0:
            x_r = series["x_r_over_H"][-1] if series["x_r_over_H"] else float("nan")
            print(f"[{tag}/{closure}] step {step}/{steps} t={step * dt:g} "
                  f"elapsed {time.time() - t0:.0f}s x_r={x_r}", flush=True)
    wall = time.time() - t0

    mean_u, mean_v, mean_G = stats.mean()
    mean_tau_b, mean_tau_t = stats.mean_wall_shear()
    x_over_h = solver.mesh.xc[0] / config.height_m

    rea = time_averaged_reattachment(mean_tau_b, x_over_h)
    hp = horizontal_profile(mean_u, solver.mesh.yc, x_over_h, 0.5, config.height_m)
    cmp = compare_profile(hp["x_over_h"], hp["u"], bulk_velocity=config.bulk_velocity)

    t = np.asarray(series["t"])
    xr = np.asarray(series["x_r_over_H"])
    finite = np.isfinite(xr)
    stationarity = {
        "x_r_over_H": stationarity_report(t[finite], xr[finite], n_blocks=n_blocks),
        "block_window_x_r_over_H": averaging_window(t[finite], xr[finite],
                                                    n_blocks=min(8, max(2, n_blocks * 2))),
        "max_nu_t_ratio": stationarity_report(
            t, np.asarray(series["max_nu_t_ratio"]), n_blocks=n_blocks) if t.size else {},
        "body_force": stationarity_report(
            t, np.asarray(series["body_force"]), n_blocks=n_blocks) if t.size else {},
        "bulk_velocity": stationarity_report(
            t, np.asarray(series["bulk_velocity"]), n_blocks=n_blocks) if t.size else {},
        "snapshots_without_reattachment": int(np.count_nonzero(~finite)),
    }

    coverage = float(finite.mean()) if finite.size else 0.0
    block_xr = []
    for k in range(n_blocks):
        if block_count[k] == 0:
            block_xr.append(None)
            continue
        block_xr.append(time_averaged_reattachment(
            block_tau[k] / block_count[k], x_over_h)["x_r_over_H"])
    finite_blocks = [v for v in block_xr if v is not None and np.isfinite(v)]
    block_spread = (float(max(finite_blocks) - min(finite_blocks))
                    / abs(float(np.mean(finite_blocks)))
                    if len(finite_blocks) == n_blocks and float(np.mean(finite_blocks)) != 0.0
                    else float("inf"))
    stationarity["x_r_over_H_by_block"] = block_xr
    stationarity["x_r_over_H_block_spread_rel"] = block_spread
    stationarity["x_r_over_H_block_samples"] = [int(v) for v in block_count]
    stationarity["reattachment_coverage"] = coverage

    payload = {
        "case": "CASE-FVM-PERIODIC-HILLS round 11 (closure-parametric time mean)",
        "tag": tag,
        "closure": _closure_block(closure, solver),
        "config": {
            "nx": config.nx, "ny": config.ny, "steps": steps,
            "sample_every": sample_every, "burn_in_steps": burn_in_steps,
            "time_step_s": dt, "end_time": steps * dt,
            "averaging_window": [burn_in_steps * dt, steps * dt],
            "reynolds": reynolds, "hill_amplitude": hill_amplitude,
            "production_limit": production_limit,
            "convection_blend": config.convection_blend,
            "bulk_velocity": config.bulk_velocity,
            "nu": solver.nu,
            "diverged_at_step": diverged_at,
        },
        "wall_clock_s": wall,
        "samples": int(stats.samples),
        "series": {k: [float(x) for x in v] for k, v in series.items()},
        "time_averaged": {
            "x_s_over_H": rea["x_s_over_H"],
            "x_r_over_H": rea["x_r_over_H"],
            "crossings": rea["crossings"],
            "body_force": mean_G,
            "max_nu_t_ratio": (float(series["max_nu_t_ratio"][-1])
                               if series["max_nu_t_ratio"] else None),
            "mean_nu_t_ratio_window": (
                float(stationarity["max_nu_t_ratio"]["mean"])
                if stationarity["max_nu_t_ratio"] else None),
        },
        "horizontal_profile_y_H_0.5": {
            "x_over_H": [float(v) for v in hp["x_over_h"]],
            "u_over_Ub": [float(v) for v in hp["u"]],
            "inadmissible_columns": int(len(hp["inadmissible"])),
            "reference": {str(k): v for k, v in BREUER_PROFILE_Y05.items()},
        },
        "profile_comparison": cmp,
        "stationarity": stationarity,
    }
    # gate is derived by the same function that scores round-9 artefacts, so the
    # two closures cannot drift apart in how they are judged
    payload["gate"] = strict_finite(gate_summary(payload))

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"round11_{closure}_{tag}"
        (out / f"{stem}.json").write_text(
            json.dumps(strict_finite(payload), indent=1, allow_nan=False))
        if save_fields:
            np.savez_compressed(
                out / f"{stem}.npz",
                mean_u=mean_u, mean_v=mean_v, mean_tau_b=mean_tau_b,
                mean_tau_t=mean_tau_t, yc=solver.mesh.yc, xc=solver.mesh.xc,
                x_over_h=x_over_h, u=solver.u, v=solver.v)
    return payload


def load_payload(path) -> dict:
    """Load a round-9 / round-11 JSON artefact (``gate`` is re-derived)."""
    payload = json.loads(Path(path).read_text())
    return payload