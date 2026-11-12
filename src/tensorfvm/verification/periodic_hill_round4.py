"""Round-4 verification of the periodic-hill finite-volume solver.

Mission 16765 (selfstart, TensorFVM).  Rounds 1-3 (``docs/case-periodic-hills``)
left the case at an honest FAIL: the Re = 2800 run over the hill went NaN within
1000 steps while the plane channel (``hill_amplitude = 0``) was clean, and round 3
blamed it on the pressure-gradient operator without ever *isolating* it.  This
driver isolates the blow-up with three negative controls and then measures the
baseline metrics on the now-stable run.

Defects fixed in round 4 (both in the transport path, neither in the geometry):

* **(viii) non-conservative east-face convection.**  ``_convection`` divided the
  outflow *and* the inflow of one face by the **donor** cell's volume.  On this
  terrain-following mesh the column thickness -- hence the cell volume -- varies
  by a factor 1.49 between crest and trough, so the two sides of a face were
  weighted differently and the operator no longer annihilated a constant field on
  a divergence-free flux (measured ``max|C(1)| = 4.13`` against ``max|div| =
  2.8e-09``): a volume source of kinetic energy.
* **(ix) inverted eta-face convection sign.**  ``out`` is the *net outflow* per
  unit volume, so the cell a face flux *leaves* must gain the contribution.  The
  eta branch had it the other way round, i.e. the wall-normal direction was
  discretised as an **anti-upwind** (negative-diffusion) operator: convection
  pumped variance into the field instead of smoothing it.  This is why only runs
  with a curved wall diverged -- on the plane channel ``flux_n`` vanishes
  identically, so the sign error is invisible there.
* **(x) bulk-velocity control was a marginally unstable oscillator.**  The law was
  ``G += relaxation (u_b^* - u_b) / dt``, a proportional controller on a plant
  that responds to ``G`` with inertia.  Linearising (plant ``du_b/dt = G - lam u_b``,
  explicit update of ``G``) gives the characteristic roots of
  ``G_{n+2} - 2 G_{n+1} + (1 + relaxation) G_n = 0``, i.e. ``|lambda|^2 =
  1 + relaxation > 1`` for every ``relaxation > 0``: ``G`` grew without bound and
  the flow was hammered until it overflowed (measured ``G = -22`` where the
  physical force is ``~5e-03``).  The body force is now obtained from the
  *affine* impulse predictor (``us(G) = us(0) + G M^{-1} 1``), which gives the
  exact force in closed form -- a constraint instead of a feedback loop --
  with ``body_force_relaxation`` demoted to an under-relaxation whose bulk error
  decays by ``(1 - relaxation)`` per step.

Criteria (pre-declared, machine-evaluated; the numbers in brackets are the round-4
values, produced by ``python -m tensorfvm.verification.periodic_hill_round4``):

C1  a conservative convective operator annihilates constants on a divergence-free
    flux: ``max|C(1)| < 1e-6`` (round 4: 2.7e-09).
C2  the negative control is *not* vacuous: the round-3 assembly, re-evaluated on
    the same flux, is O(1) wrong (round 4: 4.13).
C3  the fixed operator matches an independently assembled textbook upwind stencil
    on both face families to round-off (round 4: <=1e-12).
C4  the bulk velocity is imposed, not chased: ``|u_b - 1| < 1e-8`` at Re = 2800,
    and with ``relaxation = 1/2`` the bulk error decays with ratio ``1/2``.
C5  the plane-channel gate survives: ``u_max/u_b`` within 3 % of the analytic
    band value, ``max|div u| < 1e-10``.
C6  Re = 2800, 48x24, ``dt = 0.01`` stays finite for 30000 steps (``t = 300``),
    which rounds 2 and 3 could not reach (NaN inside 1000 steps).
C7  accuracy against Breuer et al. 2009 (as tabulated by Volpiani et al. 2021,
    §III): mean ``x_r/H`` and ``x_s/H`` within 10 % of 5.34 and 0.227, from the
    time-averaged wall shear of the stable run.
C8  grid sensitivity of ``x_r/H`` between 48x24 and 96x48 is reported (not gated
    at 10 %: two grids and an unsteady 2D flow cannot settle convergence).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillConfig, PeriodicHillSolver

__all__ = [
    "constant_field_error",
    "legacy_constant_field_error",
    "upwind_stencil_mismatch",
    "bulk_control_checks",
    "channel_gate",
    "reach_trace",
    "mean_field_metrics",
    "run",
]

#: Breuer et al. (2009) as tabulated by Volpiani et al. (2021) §III, Re = 2800.
X_R_REF = 5.34
X_S_REF = 0.227


def _hill_solver(nx=48, ny=24, reynolds=2800.0, dt=0.01, **kwargs):
    return PeriodicHillSolver(PeriodicHillConfig(
        nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt,
        body_force_relaxation=kwargs.pop("relaxation", 1.0), **kwargs))


def _projected_flux(solver, steps=3):
    solver.advance(steps)
    return solver._flux_e, solver._flux_n


def constant_field_error(solver):
    """``max|C(1)|`` for the shipped convective operator on a div-free flux."""
    flux_e, flux_n = _projected_flux(solver)
    out = solver._convection(np.ones_like(solver.u), flux_e, flux_n)
    div = np.abs(solver.flux_divergence()).max()
    return float(np.abs(out).max()), float(div)


def legacy_constant_field_error(solver):
    """Round-3 assembly (donor volume on both sides + inverted eta sign).

    Kept as the negative control: it shows that C1 measures a real property of
    the operator instead of a tautology.  ``face`` uses the same blended stencil
    as the shipped code, so the only difference is the volume weighting and the
    sign of the eta family.
    """
    m = solver.mesh
    flux_e, flux_n = _projected_flux(solver)
    phi = np.ones_like(solver.u)
    beta = solver.config.convection_blend
    face_e = beta * 0.5 * (phi + phi[:, m.ip]) + (1.0 - beta) * np.where(
        flux_e > 0.0, phi, phi[:, m.ip])
    out = np.zeros_like(phi)
    contrib = flux_e * face_e / m.volume
    out += contrib
    out[:, m.ip] -= contrib
    f = flux_n[1:-1]
    lower, upper = phi[:-1], phi[1:]
    face_n = beta * 0.5 * (lower + upper) + (1.0 - beta) * np.where(f > 0.0, lower, upper)
    contrib_n = f * face_n
    out[:-1] -= contrib_n / m.volume[:-1]
    out[1:] += contrib_n / m.volume[1:]
    return float(np.abs(out).max())


def upwind_stencil_mismatch(solver):
    """``max|_convection - independent upwind assembly|`` (should be round-off)."""
    m = solver.mesh
    phi = np.sin(2.0 * np.pi * solver.mesh.xc / solver.config.length_m) * (
        1.0 + 0.5 * np.cos(np.pi * solver.mesh.yc / 3.0))
    flux_e, flux_n = _projected_flux(solver)
    uf_e = flux_e > 0.0
    # east faces: out of cell i (upwind = cell i) or into it (upwind = i+1)
    face_e = np.where(uf_e, phi, phi[:, m.ip])
    ref = np.zeros_like(phi)
    contrib = flux_e * face_e
    ref += contrib / m.volume
    ref[:, m.ip] -= contrib / m.volume[:, m.ip]
    f = flux_n[1:-1]
    up = f > 0.0
    lower, upper = phi[:-1], phi[1:]
    face_n = np.where(up, lower, upper)
    contrib_n = f * face_n
    ref[:-1] += contrib_n / m.volume[:-1]
    ref[1:] -= contrib_n / m.volume[1:]
    got = solver._convection(phi, flux_e, flux_n, blend=0.0)
    return float(np.abs(got - ref).max())


def bulk_control_checks():
    """Imposed bulk velocity at Re = 2800, and the (1-relaxation) filter.

    The filter is measured as a *response*: run to stationarity, step the target
    from 1.0 to 1.1, and look at the ratio of successive bulk errors.  Measured in
    stationarity instead, the ratio would be ~1 simply because both errors are
    already at round-off -- a trap worth recording.
    """
    exact = _hill_solver(nx=48, ny=24)
    trace = []
    for _ in range(4):
        exact.advance(250)
        trace.append(float(abs(exact.bulk_velocity() - 1.0)))
    lagged = _hill_solver(nx=48, ny=24, relaxation=0.5)
    lagged.advance(400)
    lagged.config = lagged.config.with_options(bulk_velocity=1.1)
    prev = abs(lagged.bulk_velocity() - 1.1)
    ratios = []
    for _ in range(3):
        lagged.advance(1)
        cur = abs(lagged.bulk_velocity() - 1.1)
        if cur > 0.0 and prev > 0.0:
            ratios.append(cur / prev)
        prev = cur
    return {
        "max_abs_bulk_error_exact": max(trace),
        "trace_exact": trace,
        "body_force": float(exact.body_force),
        "observed_ratio_relaxation_half": ratios,
        "legacy_growth_factor_sqrt_1_plus_relaxation": {
            str(r): float(np.sqrt(1.0 + r)) for r in (0.1, 0.2, 0.5, 1.0)},
    }


def channel_gate():
    solver = _hill_solver(nx=8, ny=32, reynolds=100.0, dt=0.05, hill_amplitude=0.0,
                          channel_top_m=3.036)
    solver.advance(2000)
    profile = solver.mesh.yc * (solver.config.channel_top_m - solver.mesh.yc)
    analytic = float(profile.max() / solver.bulk_velocity(profile))
    measured = float(solver.u.max() / solver.bulk_velocity())
    return {"u_max_over_ub": measured, "analytic": analytic,
            "rel_error": abs(measured - analytic) / analytic,
            "max_cell_divergence": float(np.abs(solver.divergence()).max())}


def reach_trace(nx=48, ny=24, steps=30000, chunk=3000, dt=0.01):
    """Long stable run; returns the per-chunk trace (used for criterion C6)."""
    solver = _hill_solver(nx=nx, ny=ny, dt=dt)
    trace = []
    for _ in range(steps // chunk):
        solver.advance(chunk)
        trace.append({"step": solver.steps, "t": solver.steps * dt,
                      "max_abs_velocity": float(np.abs(solver.u).max()),
                      "bulk_velocity": solver.bulk_velocity(),
                      "max_cell_divergence": float(np.abs(solver.divergence()).max())})
    return trace


def _crossings(tau, x, length=9.0):
    """All zero crossings of the wall shear, with their sign direction.

    The hill surface is slanted, so the streamwise wall-shear component changes
    sign at *both* ends of the separation bubble; the ``+ -> -`` crossings are
    candidate separations and the ``- -> +`` ones candidate reattachments.
    """
    out = []
    for i in range(len(x)):
        a, b = tau[i], tau[(i + 1) % len(x)]
        if a == 0.0 or not ((a < 0.0 <= b) or (a > 0.0 >= b)):
            continue
        t = -a / (b - a)
        span = (x[(i + 1) % len(x)] - x[i]) % length
        out.append({"x_over_H": float(x[i] + t * span),
                    "direction": "-to+" if a < 0.0 else "+to-"})
    return out


def mean_field_metrics(path, units_per_hill=9.0):
    """Separation / reattachment from a time-averaged wall shear artefact.

    ``x_s`` is the first ``+ -> -`` crossing downstream of the crest (the
    ERCOFTAC bubble starts just behind it) and ``x_r`` the **last** ``- -> +``
    crossing inside one hill period -- the mean bubble in this solver extends to
    ``x/H ~ 8``, i.e. past the reference value, so taking the first crossing found
    would flatter the comparison.
    """
    data = np.load(path)
    tau = data["tau_b"]
    nx = int(data["nx"])
    x = (np.arange(nx) + 0.5) * units_per_hill / nx
    crossings = _crossings(tau, x, units_per_hill)
    sep = [c["x_over_H"] for c in crossings
           if c["direction"] == "+to-" and c["x_over_H"] <= 2.5]
    rea = [c["x_over_H"] for c in crossings
           if c["direction"] == "-to+" and 1.9 <= c["x_over_H"] <= 8.9]
    return {
        "file": str(path),
        "samples": int(data["n"]),
        "x_s_over_H": float(sep[0]) if sep else float("nan"),
        "x_r_over_H": float(rea[-1]) if rea else float("nan"),
        "all_crossings": crossings,
        "wall_shear_min": float(tau.min()),
        "wall_shear_max": float(tau.max()),
    }


def run(out_dir=None, reach_steps=30000, mean_artifacts=None):
    """Evaluate all round-4 criteria and (optionally) write the summary."""
    solver = _hill_solver()
    c1, div_flux = constant_field_error(solver)
    c2 = legacy_constant_field_error(solver)
    c3 = upwind_stencil_mismatch(_hill_solver())
    c4 = bulk_control_checks()
    c5 = channel_gate()
    c6 = reach_trace(steps=reach_steps)

    metrics = {}
    if mean_artifacts:
        for grid, path in mean_artifacts.items():
            p = Path(path)
            if p.exists():
                metrics[grid] = mean_field_metrics(p)
    finest = metrics.get("48x24")
    c7 = None
    if finest is not None and np.isfinite(finest["x_r_over_H"]):
        c7 = {
            "x_r_over_H": finest["x_r_over_H"], "ref": X_R_REF,
            "rel_error": abs(finest["x_r_over_H"] - X_R_REF) / X_R_REF,
            "x_s_over_H": finest["x_s_over_H"], "x_s_ref": X_S_REF,
            "x_s_rel_error": (abs(finest["x_s_over_H"] - X_S_REF) / X_S_REF
                              if np.isfinite(finest["x_s_over_H"]) else float("nan")),
        }
        c7["pass"] = bool(c7["rel_error"] < 0.10)
    c8 = None
    if "96x48" in metrics and "48x24" in metrics:
        a, b = metrics["48x24"]["x_r_over_H"], metrics["96x48"]["x_r_over_H"]
        c8 = {"x_r_48x24": a, "x_r_96x48": b,
              "relative_change": float(abs(b - a) / a) if np.isfinite(a) and a else float("nan")}

    stable = all(np.isfinite(r["max_abs_velocity"]) and r["max_abs_velocity"] < 1e3
                 for r in c6)
    criteria = {
        "C1_conservative_convection": {
            "max_abs_conv_of_constant": c1,
            "max_abs_flux_divergence": div_flux,
            "pass": bool(c1 < 1.0e-6),
        },
        "C2_round3_assembly_negative_control": {
            "max_abs_conv_of_constant_legacy": c2,
            "pass": bool(c2 > 1.0e-1),
        },
        "C3_upwind_stencil_identity": {"max_abs_mismatch": c3,
                                       "pass": bool(c3 < 1.0e-10)},
        "C4_bulk_velocity_constraint": {
            **c4,
            "pass": bool(c4["max_abs_bulk_error_exact"] < 1.0e-8
                         and c4["observed_ratio_relaxation_half"]
                         and abs(c4["observed_ratio_relaxation_half"][0] - 0.5) < 0.05),
        },
        "C5_plane_channel_gate": {**c5,
                                  "pass": bool(c5["rel_error"] < 0.03
                                               and c5["max_cell_divergence"] < 1e-10)},
        "C6_re2800_reach_stable": {
            "steps": reach_steps,
            "trace": c6,
            "pass": bool(stable),
        },
        "C7_baseline_accuracy": c7 if c7 is not None else {"pass": False,
                                                          "note": "no mean artefact"},
        "C8_grid_sensitivity": c8 if c8 is not None else {"note": "single grid only"},
    }
    summary = {
        "node": "CASE-FVM-PERIODIC-HILLS",
        "round": 4,
        "defects": ["viii east-face donor volume", "ix eta-face sign inversion",
                    "x marginally unstable bulk-velocity controller"],
        "criteria": criteria,
        "all_pass": all(item.get("pass", True) for item in criteria.values()),
        "note": ("C1-C6 are the round-4 defect gates; C7 is the acceptance metric "
                 "of the anchor (10 % on the Breuer mean reattachment); C8 is "
                 "reported, not gated."),
    }
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "round4_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "docs/case-periodic-hills"
    root = Path("runs/periodic-hill-round4")
    result = run(target, mean_artifacts={
        "48x24": root / "a48_mean.npz", "96x48": root / "a96_mean.npz"})
    for name, item in result["criteria"].items():
        print(f"{name:38s} {'PASS' if item.get('pass') else 'FAIL'}")
    print("all_pass:", result["all_pass"])