"""Verification of the shallow-water bottom-friction / slope source terms.

Why this module exists
----------------------
``shallow_water.swe1d`` and ``shallow_water.bed`` integrate the inviscid
depth-averaged equations.  Every channel question -- normal depth, flood-wave
routing, harbour seiching damping, ice-channel discharge -- needs a bed
friction closure, and a friction term is trivially *plausible*: any monotone
momentum sink makes a transient decay look "physical".  A sink that is off by
a factor ``h`` or that forgets ``|u|`` still looks like damping.

This driver therefore does not ask "does the flow slow down".  It binds the
implemented source to **closed-form solutions of the ODE it is supposed to
integrate**:

* friction-only decay (uniform state, flat bed, no forcing).  The uniform state
  has zero flux divergence, so the semi-discrete system *is* the scalar ODE
  ``du/dt = -C u|u|``, ``C = g n^2 / h^(4/3)`` (Manning, wide channel
  ``R ~ h``), whose exact solution is ``u(t) = u0 / (1 + C u0 t)``;
* normal-flow relaxation (uniform slope ``S_0``, friction).  The same reduction
  gives ``du/dt = g S_0 (1 - (u/u_n)^2)`` with the Manning normal velocity
  ``u_n = h^(2/3) sqrt(S_0) / n``, whose exact solution is
  ``u(t) = u_n tanh(g S_0 t / u_n)``.  This is the anchor that fixes the
  *magnitude* of the drag, because ``u_n`` inherits ``n``, ``h`` and ``S_0``
  with the textbook exponents;
* exact rest: a lake at rest stays at rest (``S_f = 0`` at ``u = 0``), i.e. the
  well-balanced property of the Audusse bed solver survives the new source;
* negative controls: with ``manning_n = 0, slope = 0`` the residual must be
  bit-identical to the flux-only inviscid residual, and the decay run must not
  decay at all -- a gate that cannot fail proves nothing.

Everything is 1-D and runs in a few seconds; the criteria are pre-declared in
``docs/verification-swe-friction/criteria.md`` and executed by ``run()``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..shallow_water.bed import ShallowWaterBed1D
from ..shallow_water.swe1d import ShallowWater1D

G = 9.81
H0 = 1.0
U0 = 1.5
MANNING = 0.03
SLOPE = 1.0e-3


# --------------------------------------------------------------------- closed form
def decay_coefficient(manning_n: float = MANNING, h: float = H0, g: float = G) -> float:
    """``C`` in ``du/dt = -C u|u|``: the wide-channel Manning law ``g n^2 / h^(4/3)``."""
    return g * manning_n ** 2 / h ** (4.0 / 3.0)


def decay_velocity(t, u0: float = U0, manning_n: float = MANNING, h: float = H0,
                   g: float = G):
    """Exact solution of ``du/dt = -C u|u|`` from ``u(0) = u0``: ``u0/(1+C u0 t)``."""
    c = decay_coefficient(manning_n, h, g)
    return u0 / (1.0 + c * u0 * np.asarray(t, dtype=float))


def normal_velocity(h: float = H0, manning_n: float = MANNING, slope: float = SLOPE,
                    g: float = G) -> float:
    """Manning normal-flow velocity ``u_n = h^(2/3) sqrt(S_0) / n``."""
    return h ** (2.0 / 3.0) * math.sqrt(slope) / manning_n


def relaxation_velocity(t, h: float = H0, manning_n: float = MANNING,
                        slope: float = SLOPE, g: float = G):
    """Exact solution of ``du/dt = g S_0 (1 - (u/u_n)^2)`` from rest."""
    un = normal_velocity(h, manning_n, slope, g)
    return un * np.tanh(g * slope * np.asarray(t, dtype=float) / un)


# ----------------------------------------------------------------------- helpers
def _criterion(cid, name, value, threshold, mode="lt", unit="", note=""):
    if mode == "lt":
        ok = value < threshold
    elif mode == "gt":
        ok = value > threshold
    elif mode == "le":          # exactness gates: value <= threshold
        ok = value <= threshold
    else:                       # "abs" -> |value| < threshold
        ok = abs(value) < threshold
    return dict(id=cid, name=name, value=float(value), threshold=float(threshold),
                mode=mode, unit=unit, verdict="PASS" if ok else "FAIL", note=note)


def uniform_solver(n: int = 64, length: float = 200.0, h: float = H0, u: float = U0,
                   manning_n: float = 0.0, slope: float = 0.0, cfl: float = 0.4,
                   order: int = 2) -> ShallowWater1D:
    """Straight, flat, uniform channel.

    The ends are transmissive with a zero-gradient ghost, so a *uniform* state
    has identical fluxes on both end faces and the flux divergence is exactly
    zero: the semi-discrete system collapses onto the local ODE.  That is what
    makes these runs closed-form verifiable.
    """
    solver = ShallowWater1D(n, 0.0, length, gravity=G, cfl=cfl, order=order,
                            manning_n=manning_n, slope=slope)
    solver.set_state(np.full(n, h), np.full(n, u))
    return solver


def march(solver: ShallowWater1D, t_end: float, dt: float | None = None,
          samples=()):
    """March to ``t_end`` (fixed ``dt`` or adaptive CFL); return the time series.

    Returns ``(times, velocities, depths)`` sampled at the requested times plus
    the final time.
    """
    samples = sorted(float(s) for s in samples if s <= t_end)
    times, speeds, depths = [], [], []
    targets = list(samples) + [float(t_end)]
    for target in targets:
        if dt is None:
            solver.advance(target)
        else:
            while solver.time < target - 1e-12:
                step = min(dt, target - solver.time)
                solver.step(step)
        _, u = solver.primitive()
        times.append(solver.time)
        speeds.append(float(u[0]))
        depths.append(float(solver.h[0]))
    return np.array(times), np.array(speeds), np.array(depths)


def observed_order(errors, dts):
    """Least-squares slope of ``log(error)`` against ``log(dt)``."""
    e = np.asarray(errors, dtype=float)
    d = np.asarray(dts, dtype=float)
    good = e > 0.0
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[good]), np.log(e[good]), 1)[0])


# --------------------------------------------------------------------- the checks
def _check_c1_c2():
    """C1 production-setup decay error; C2 fixed-dt convergence order."""
    t_end = 60.0
    solver = uniform_solver(manning_n=MANNING)
    times, speeds, depths = march(solver, t_end, samples=(10.0, 30.0))
    exact = decay_velocity(times)
    c1 = float(np.max(np.abs(speeds / exact - 1.0)))
    c1_info = dict(times=times, numerical=speeds, exact=exact,
                   depth_drift=float(np.max(np.abs(depths - H0))))

    dts = (0.4, 0.2, 0.1)
    errors = []
    for dt in dts:
        run = uniform_solver(manning_n=MANNING)
        march(run, 20.0, dt=dt)
        _, u = run.primitive()
        errors.append(abs(float(u[0]) / float(decay_velocity(20.0)) - 1.0))
    c2 = observed_order(errors, dts)
    return c1, c1_info, c2, dict(dts=list(dts), errors=errors)


def _check_c3_c4_c5():
    """Normal-flow relaxation: C3 tanh agreement, C4 normal velocity, C5 depth."""
    t_end = 1200.0                              # tanh saturates: 1 - tanh(9.3) < 1e-8
    solver = uniform_solver(u=0.0, manning_n=MANNING, slope=SLOPE)
    times, speeds, depths = march(solver, t_end, samples=(20.0, 60.0, 120.0))
    exact = relaxation_velocity(times)
    c3 = float(np.max(np.abs(speeds[:3] / exact[:3] - 1.0)))
    un = normal_velocity()
    c4 = float(abs(speeds[-1] / un - 1.0))
    c5 = float(np.max(np.abs(depths - H0)))
    info = dict(times=times, numerical=speeds, exact=exact, u_normal=un,
                tanh_error_end=float(abs(speeds[-1] / exact[-1] - 1.0)),
                friction_slope_end=float(solver.friction_slope()[0]),
                slope_forced=SLOPE)
    return c3, c4, c5, info


def _check_c6_negative_control():
    """No source -> no decay, and the residual is bit-identical to flux-only."""
    solver = uniform_solver()                   # manning_n = 0, slope = 0
    march(solver, 60.0)
    _, u = solver.primitive()
    drift = float(np.max(np.abs(u / U0 - 1.0)))

    # independent re-derivation of the inviscid residual
    h = solver.h
    hu = solver.hu
    hL, uL, hR, uR = solver._face_states()
    fh, fhu = solver._hll_flux(hL, uL, hR, uR)
    ref_dh = -(fh[1:] - fh[:-1]) / solver.dx
    ref_dhu = -(fhu[1:] - fhu[:-1]) / solver.dx
    got_dh, got_dhu = solver.rhs(h, hu)
    bitwise = float(max(np.max(np.abs(got_dh - ref_dh)),
                        np.max(np.abs(got_dhu - ref_dhu))))
    return drift, bitwise


def _check_c7_lake_at_rest():
    """Friction must not disturb a lake at rest over a sloped bed."""
    n = 200
    length = 200.0
    s0 = SLOPE
    x = np.linspace(0.0, length, n) + 0.5 * (length / n)
    bed = -s0 * x                                  # constant longitudinal slope
    surface = 5.0
    solver = ShallowWaterBed1D(n, 0.0, length, bed=bed, gravity=G, manning_n=MANNING)
    solver.set_state(np.maximum(surface - bed, 0.0))
    solver.advance(60.0)
    return (float(np.max(np.abs(solver.primitive()[1]))),
            float(solver.surface_drift()))


class _Unguarded(ShallowWater1D):
    """Negative control: the same drag with the reversal guard disabled.

    A guard that is never exercised is untested code; this subclass shows what
    the guard protects against (the drag drives the velocity through zero and
    leaves a spurious counter-flow).
    """

    def _source_momentum(self, h, hu, dt=None):
        return super()._source_momentum(h, hu, None)


def _check_c8_stiff_drag():
    """Stiff drag at a large Courant number: no reversal with the guard."""
    kwargs = dict(n=16, u=3.0, manning_n=0.15, cfl=0.95)
    solver = uniform_solver(**kwargs)
    march(solver, 30.0)
    _, u = solver.primitive()
    exact = float(decay_velocity(30.0, u0=3.0, manning_n=0.15))

    guarded_min, _ = _fixed_step_channel(guarded=True)
    unguarded_min, _ = _fixed_step_channel(guarded=False)

    return dict(u_min=float(np.min(u)), finite=bool(np.all(np.isfinite(solver.h))),
                decay_error=abs(float(u[0]) / exact - 1.0), exact=exact,
                guard_fired=int(solver.friction_clipped),
                fixed_step_guarded_min=guarded_min,
                fixed_step_unguarded_min=unguarded_min)


def _fixed_step_channel(guarded: bool, dt: float = 10.0, steps: int = 3):
    """Uniform channel stepped at ``dt`` far beyond the CFL limit.

    The flux divergence of a uniform state is exactly zero, so the only
    dynamics left is the drag: this isolates the guard.  ``dt = 10 s`` gives
    ``C u dt ~ 6.6``, i.e. the drag of a single stage is several times the
    momentum the cell holds -- an unguarded point-implicit-free update must
    therefore drive ``u`` through zero.
    """
    cls = ShallowWater1D if guarded else _Unguarded
    solver = cls(16, 0.0, 200.0, gravity=G, manning_n=0.15)
    solver.set_state(np.full(16, H0), np.full(16, 3.0))
    history = [float(np.min(solver.primitive()[1]))]
    with np.errstate(all="ignore"):             # the unguarded control is meant to blow up
        for _ in range(steps):
            solver.step(dt)
            history.append(float(np.min(solver.primitive()[1])))
    return float(min(history)), history


def _check_c9_energy():
    """Energy is dissipated by friction and by nothing else."""
    solver = uniform_solver(manning_n=MANNING)
    energies = [solver.kinetic_energy()]
    for _ in range(400):
        solver.step(0.1)
        energies.append(solver.kinetic_energy())
    energies = np.array(energies)
    length = solver.x_max - solver.x_min
    monotone = float(np.max(np.diff(energies)))            # must be <= 0
    u_exact = float(decay_velocity(40.0))
    exact = length * (0.5 * H0 * u_exact ** 2 + 0.5 * G * H0 ** 2)
    final_rel = abs(energies[-1] / exact - 1.0)
    return monotone, final_rel, energies


def _check_c10_wet_dry():
    """Dam break over a dry bed with/without friction: friction must slow the front."""
    def run(manning_n):
        solver = ShallowWater1D(400, -4.0, 4.0, gravity=G, cfl=0.4, order=2,
                                manning_n=manning_n)
        h = np.where(solver.x < 0.0, 1.0, 0.0)
        solver.set_state(h)
        v0 = solver.total_volume()
        solver.advance(0.6385)                 # t* = 2 for h0 = 1 m
        _, u = solver.primitive()
        wet = solver.h > 1e-6
        front = float(solver.x[wet].max()) if np.any(wet) else float("nan")
        return dict(front=front, volume_drift=abs(solver.total_volume() / v0 - 1.0),
                    finite=bool(np.all(np.isfinite(solver.h))),
                    u_max=float(np.max(np.abs(u))))
    wet = run(MANNING)
    dry = run(0.0)
    return wet, dry


# --------------------------------------------------------------------------- run
def run() -> dict:
    """Execute every pre-declared criterion and return the payload."""
    start = time.time()
    criteria = []

    c1, c1_info, c2, c2_info = _check_c1_c2()
    criteria.append(_criterion(
        "C1", "friction-only decay u(t) vs u0/(1+C u0 t) (production CFL)", c1, 0.01,
        unit="rel. max",
        note=f"u(60 s)={c1_info['numerical'][-1]:.6f} vs {c1_info['exact'][-1]:.6f} m/s"))
    criteria.append(_criterion(
        "C2", "observed time order of the friction source (fixed-dt ladder)", c2, 2.7,
        mode="gt", unit="p",
        note="dt=" + ",".join(f"{d:g}" for d in c2_info["dts"]) + " err="
             + ",".join(f"{e:.2e}" for e in c2_info["errors"])))

    c3, c4, c5, c3_info = _check_c3_c4_c5()
    criteria.append(_criterion(
        "C3", "normal-flow relaxation u(t) vs u_n tanh(g S0 t/u_n)", c3, 0.03,
        unit="rel. max",
        note="t=" + ",".join(f"{t:g}" for t in c3_info["times"][:3])))
    criteria.append(_criterion(
        "C4", "steady velocity vs Manning u_n = h^(2/3) sqrt(S0)/n", c4, 0.03,
        unit="rel.",
        note=f"u_num={c3_info['numerical'][-1]:.6f} vs u_n={c3_info['u_normal']:.6f} m/s "
             f"(tanh residual {c3_info['tanh_error_end']:.2e})"))
    criteria.append(_criterion(
        "C5", "uniform depth preserved while friction acts (mass consistency)", c5,
        1e-12, unit="m",
        note="a non-uniform answer would mean the source leaked into the mass flux"))

    drift, bitwise = _check_c6_negative_control()
    criteria.append(_criterion(
        "C6", "negative control: no source -> no decay (drift of u)", drift, 1e-12,
        unit="rel."))
    criteria.append(_criterion(
        "C6b", "bit-identical inviscid residual when sources are off", bitwise, 0.0,
        mode="le", unit="abs.", note="max |rhs_with_sources_off - rhs_flux_only|"))

    rest_u, rest_eta = _check_c7_lake_at_rest()
    criteria.append(_criterion(
        "C7", "lake at rest over a sloped bed with friction: max|u|", rest_u, 1e-12,
        unit="m/s", note=f"surface drift {rest_eta:.3e} m"))

    stiff = _check_c8_stiff_drag()
    criteria.append(_criterion(
        "C8", "stiff drag at CFL 0.95: velocity never reverses", -stiff["u_min"], 0.0,
        unit="-u_min",
        note=f"finite={stiff['finite']}, min u={stiff['u_min']:.6f} m/s, "
             f"decay error {stiff['decay_error']:.2%} (n=0.15: accuracy limited by dt)"))
    criteria.append(_criterion(
        "C8b", "reversal guard actually fires in the stiff-drag run",
        float(stiff["guard_fired"]), 1.0, mode="gt", unit="count",
        note="a guard that never binds is untested code"))
    criteria.append(_criterion(
        "C8c", "guarded drag at dt far beyond CFL (dt=10 s): no reversal",
        -stiff["fixed_step_guarded_min"], 0.0, unit="-u_min",
        note=f"min u over 3 steps = {stiff['fixed_step_guarded_min']:.6f} m/s"))
    criteria.append(_criterion(
        "C8d", "negative control: the same run unguarded does reverse the flow",
        -stiff["fixed_step_unguarded_min"], 0.0, mode="gt", unit="-u_min_unguarded",
        note=f"min u = {stiff['fixed_step_unguarded_min']:.6g} m/s "
             f"(diverged) -> the guard is not vacuous"))

    monotone, final_rel, energies = _check_c9_energy()
    criteria.append(_criterion(
        "C9", "mechanical energy non-increasing in a friction-only run", monotone,
        0.0, unit="J/m step", note="max forward difference of E(t) (must be <= 0)"))
    criteria.append(_criterion(
        "C9b", "energy left after 40 s vs exact decay", final_rel, 0.01, unit="rel."))

    wet, dry = _check_c10_wet_dry()
    criteria.append(_criterion(
        "C10", "wet/dry dam break: friction slows the front", wet["front"],
        dry["front"], mode="lt", unit="m",
        note=f"front no-friction {dry['front']:.4f} m, with n={MANNING} {wet['front']:.4f} m"))
    criteria.append(_criterion(
        "C11", "wet/dry dam break with friction: volume conservation", wet["volume_drift"],
        1e-10, unit="rel.", note=f"finite={wet['finite']}, max|u|={wet['u_max']:.3f} m/s"))

    n_pass = sum(c["verdict"] == "PASS" for c in criteria)
    return dict(
        module="tensorfvm.verification.shallow_water_friction",
        criteria=criteria,
        n_pass=n_pass, n_total=len(criteria),
        all_pass=bool(n_pass == len(criteria)),
        runtime_s=time.time() - start,
        setup=dict(gravity=G, h0=H0, u0=U0, manning_n=MANNING, slope=SLOPE),
        decay_information=c1_info, relaxation_information=c3_info,
        energy_information=dict(energies=energies.tolist()),
        wet_dry=dict(friction=wet, inviscid=dry), stiff_drag=stiff,
    )


def figure(payload: dict, path: Path | str) -> Path:
    """Three-panel figure: decay, normal-flow relaxation, energy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    info = payload["decay_information"]
    t = np.linspace(0.0, 60.0, 301)
    axes[0].plot(t, decay_velocity(t), "-", label="closed form $u_0/(1+Cu_0t)$")
    axes[0].plot(info["times"], info["numerical"], "o", label="solver (n=0.03)")
    axes[0].set_xlabel("t / s")
    axes[0].set_ylabel("u / (m/s)")
    axes[0].set_title("friction-only decay, flat bed")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    rinfo = payload["relaxation_information"]
    tt = np.linspace(0.0, 240.0, 401)
    axes[1].plot(tt, relaxation_velocity(tt), "-", label=r"$u_n\tanh(gS_0t/u_n)$")
    axes[1].plot(rinfo["times"], rinfo["numerical"], "o", label="solver (S0=1e-3)")
    axes[1].axhline(rinfo["u_normal"], ls=":", color="k", lw=1.0,
                    label=f"$u_n$={rinfo['u_normal']:.4f} m/s")
    axes[1].set_xlabel("t / s")
    axes[1].set_ylabel("u / (m/s)")
    axes[1].set_title("relaxation to Manning normal flow")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    e = np.array(payload["energy_information"]["energies"])
    axes[2].plot(np.arange(e.size) * 0.1, e, "-", label="solver E(t)")
    axes[2].set_xlabel("t / s")
    axes[2].set_ylabel(r"$\int(\frac{1}{2}hu^2+\frac{g}{2}h^2)\,dx$  (J/m)")
    axes[2].set_title("energy dissipated by friction only")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def results_markdown(payload: dict) -> str:
    """Machine-generated results tables (no number in the report is hand-copied)."""
    lines = ["# Verification results (generated from ``summary.json``)", "",
             f"Module: `{payload['module']}`  ",
             f"Verdict: **{payload['n_pass']}/{payload['n_total']} PASS** "
             f"(all_pass = {payload['all_pass']}, runtime {payload['runtime_s']:.1f} s)", "",
             "## Criteria", "",
             "| id | criterion | value | gate | unit | verdict | note |",
             "|---|---|---|---|---|---|---|"]
    for c in payload["criteria"]:
        op = {"lt": "<", "gt": ">", "le": "<=", "abs": "|.|&lt;"}[c["mode"]]
        lines.append(f"| {c['id']} | {c['name']} | {c['value']:.6e} | {op} "
                     f"{c['threshold']:.3e} | {c['unit']} | **{c['verdict']}** | {c['note']} |")

    dec = payload["decay_information"]
    lines += ["", "## Friction-only decay in a uniform channel (n = 0.03, h = 1 m, "
                  "u0 = 1.5 m/s)", "",
              "| t / s | u solver / (m/s) | u closed form / (m/s) | rel. difference |",
              "|---|---|---|---|"]
    for t, num, ex in zip(dec["times"], dec["numerical"], dec["exact"]):
        lines.append(f"| {t:g} | {num:.12f} | {ex:.12f} | {abs(num / ex - 1.0):.3e} |")
    lines.append("")
    lines.append(f"depth drift of the uniform state over the whole run: "
                 f"{dec['depth_drift']:.3e} m")

    rel = payload["relaxation_information"]
    lines += ["", "## Relaxation to Manning normal flow (n = 0.03, S0 = 1e-3, from rest)", "",
              "| t / s | u solver / (m/s) | u_n tanh(g S0 t/u_n) / (m/s) | rel. difference |",
              "|---|---|---|---|"]
    for t, num, ex in zip(rel["times"], rel["numerical"], rel["exact"]):
        lines.append(f"| {t:g} | {num:.12f} | {ex:.12f} | {abs(num / ex - 1.0):.3e} |")
    lines += ["",
              f"analytic normal velocity u_n = {rel['u_normal']:.12f} m/s; "
              f"friction slope at the end of the run S_f = {rel['friction_slope_end']:.12e} "
              f"vs the imposed S0 = {rel['slope_forced']:.3e} "
              f"(rel. {abs(rel['friction_slope_end'] / rel['slope_forced'] - 1.0):.2e})"]

    wd = payload["wet_dry"]
    lines += ["", "## Wet/dry dam break (Ritter geometry, t* = 2) with and without friction", "",
              "| quantity | inviscid | with n = 0.03 |", "|---|---|---|",
              f"| front position / m | {wd['inviscid']['front']:.4f} | "
              f"{wd['friction']['front']:.4f} |",
              f"| volume drift (rel.) | {wd['inviscid']['volume_drift']:.3e} | "
              f"{wd['friction']['volume_drift']:.3e} |",
              f"| max abs u / (m/s) | {wd['inviscid']['u_max']:.4f} | "
              f"{wd['friction']['u_max']:.4f} |"]

    st = payload["stiff_drag"]
    lines += ["", "## Stiff-drag robustness", "",
              "| quantity | value |", "|---|---|",
              f"| min u at CFL 0.95, n = 0.15 / (m/s) | {st['u_min']:.6f} |",
              f"| decay error vs closed form (dt-limited) | {st['decay_error']:.2%} |",
              f"| reversal-guard activations | {st['guard_fired']} |",
              f"| min u, dt = 10 s, guarded / (m/s) | {st['fixed_step_guarded_min']:.6f} |",
              f"| min u, dt = 10 s, unguarded / (m/s) | {st['fixed_step_unguarded_min']:.6g} "
              f"(diverged) |"]
    return "\n".join(lines) + "\n"


def _json_default(obj):
    """Serialise numpy scalars/arrays (``default=float`` cannot do arrays)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return float(obj)


def write_artifacts(outdir: Path | str = "docs/verification-swe-friction") -> dict:
    """Run every criterion, then write ``summary.json`` and the figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = run()
    public = {k: v for k, v in payload.items() if not k.startswith("_")}
    (outdir / "summary.json").write_text(json.dumps(public, indent=2, default=_json_default))
    (outdir / "results.md").write_text(results_markdown(public))
    figure(payload, outdir / "fig_swe_friction.png")
    return public


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default="docs/verification-swe-friction")
    args = parser.parse_args(argv)
    payload = write_artifacts(args.outdir)
    for crit in payload["criteria"]:
        op = {"lt": "<", "gt": ">", "le": "<=", "abs": "|.|<"}[crit["mode"]]
        print(f"{crit['id']:5s} {crit['verdict']:4s} {crit['value']:.6e} "
              f"{op} {crit['threshold']:.3e} {crit['unit']:9s} {crit['name']}")
    print(f"{payload['n_pass']}/{payload['n_total']} PASS   "
          f"all_pass={payload['all_pass']}   {payload['runtime_s']:.1f} s")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())