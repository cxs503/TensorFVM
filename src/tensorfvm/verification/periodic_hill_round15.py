"""Round 15 of ``CASE-FVM-PERIODIC-HILLS``: is the over-diffusivity the closure's?

Why this round exists
---------------------
Rounds 11--14 measured the SST ``k-omega`` branch of the Re = 2800 periodic hill
at ``96x48`` and found ``x_r/H = 2.05`` (round 11, defective transport) rising to
``3.14`` (round 12, fixed transport) against the DNS ``5.41`` -- with
``max nu_t/nu = 342`` where a shear layer of thickness ``~H`` and ``u' ~ 0.1 U_b``
supports O(10--30).  Rounds 12 and 13 then fixed three transport defects and the
eta-face viscosity pairing in the momentum / SA operators; all of those fixes are
in ``HEAD``, but the *physics* number for the healed operator was never measured
(round 13's ``cj3108`` was still queued when round 14 froze the record).

Round 14 listed five candidate next steps and was honest that none was
discriminating: a hill re-run alone cannot separate "the closure over-produces
``nu_t``" from "the terrain-following scheme over-diffuses".  This round adds the
discriminating experiment and then re-scores the hill.

The discriminator: plane channel, same solver, same discretisation
------------------------------------------------------------------
With ``hill_amplitude = 0`` the very same :class:`PeriodicHillSolver` integrates
a fully developed plane channel driven by the bulk-velocity controller, and a
plane channel has *three closed-form anchors* that a terrain-following-mesh
scheme cannot fake:

* **Linear total stress.**  For a fully developed channel the time-mean momentum
  balance gives ``(nu + nu_t) du/dy = u_tau^2 (1 - y/h)`` with
  ``u_tau^2 = tau_w``; this is an identity, so the residual measures the closure /
  operator pairing, not a correlation (gate C2, < 5 % over ``0.1 <= y/h <= 0.9``).
* **Log law.**  ``u+ = ln(y+)/kappa + B`` with ``kappa = 0.41``, ``B = 5.2``
  (gate C1: ``kappa`` within 10 %, ``B`` within 1.0).
* **Log-layer eddy-viscosity identity.**  Differentiating the log law gives
  ``nu_t+ = kappa y+ - 1`` *exactly*.  This is the number that decides the
  question: the periodic hill needs ``nu_t+ ~ 10--30`` in the shear layer and the
  solver produces 342.  If the channel also produces a multiple of
  ``kappa y+ - 1``, the defect is in the closure, not in the hill geometry
  (gate C3: median ratio inside ``[0.7, 1.4]``).

Gate C4 adds the friction check ``Re_tau`` against Dean (1978)
``C_f = 0.073 Re_b^{-0.25}`` (``Re_b`` on the **full** channel height; then
``Re_tau = (Re_b/2) sqrt(C_f/2)``), 10 % band; it is a correlation, so it is a
consistency check, not a verification anchor, and it is reported as such.

Second finding this round: a knob that never reached the closure
----------------------------------------------------------------
Round 14's next step 2 was "run the SST transport with second-order convection
(``convection_blend = 1.0``)".  The round-11 driver's ``--convection-blend``
flag only set :class:`~tensorfvm.periodic_hill.PeriodicHillConfig.convection_blend`
(the **momentum** convection) while ``SSTKOmegaConfig.convection_blend`` kept its
first-order default; the JSON recorded both numbers and nobody compared them
(round 11: momentum 0.75, transport 0.0).  :func:`build_solver` now takes an
explicit ``transport_convection_blend`` and the artefact records it; the round-15
hill runs are the first ones where the transport blend is what the config says.

Pre-declared gates (evaluated mechanically by :func:`channel_verdict`)
----------------------------------------------------------------------
C1  log law: ``|kappa_eff/0.41 - 1| <= 0.10`` and ``|B - 5.2| <= 1.0``
C2  linear total stress: max relative residual <= 5 % (``0.1 <= y/h <= 0.9``)
C3  eddy-viscosity identity: median ``nu_t+/(kappa y+ - 1)`` in ``[0.7, 1.4]``
    over ``30 <= y+ <= 0.2 h+``
C4  friction: ``Re_tau`` within 10 % of Dean (1978)

C3 is the round's scientific question and is allowed to fail loudly: a factor of
5--10 is the expected signature of the hill's over-diffusivity and is reported
as measured, never tuned away.

Usage
-----
    PYTHONPATH=src python scripts/round15_channel_and_rescore.py channel \
        --ny 64 --nx 8 --steps 30000 --burn-in 10000
    PYTHONPATH=src python scripts/round15_channel_and_rescore.py hill --tag r15_96x48
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillSolver
from .periodic_hill_round5 import LOG_B, LOG_KAPPA, log_law_fit

__all__ = [
    "LOG_KAPPA",
    "LOG_B",
    "DEAN_CF_COEFF",
    "DEAN_CF_EXPONENT",
    "dean_friction",
    "channel_solver",
    "channel_postprocess",
    "channel_verdict",
    "run_channel",
]

#: Dean (1978), *J. Fluids Eng.* 100:215, friction correlation
#: ``C_f = 0.073 Re_b^{-0.25}`` with ``Re_b = U_b * (2h) / nu``
DEAN_CF_COEFF = 0.073
DEAN_CF_EXPONENT = -0.25

#: log-layer window for the eddy-viscosity identity
NUT_PLUS_YPLUS_LO = 30.0
NUT_PLUS_YPLUS_HI_FRACTION = 0.2

#: total-stress window (fraction of the half height)
STRESS_WINDOW = (0.1, 0.9)

#: pre-declared tolerances
TOL_KAPPA = 0.10
TOL_B = 1.0
TOL_STRESS = 0.05
NUT_PLUS_RATIO_BAND = (0.7, 1.4)
TOL_RE_TAU = 0.10


def dean_friction(re_bulk: float) -> dict:
    """Dean (1978) plane-channel friction for ``Re_b`` on the *full* height.

    ``C_f = 0.073 Re_b^{-0.25}``; with ``C_f = 2 (u_tau/U_b)^2`` and
    ``Re_tau = u_tau h / nu = (Re_b/2) (u_tau/U_b)`` this gives
    ``Re_tau = (Re_b/2) sqrt(C_f/2)``.  The basis (full height vs half height)
    is the classical source of factor-2 confusion, so it is stated here and
    recorded in every artefact.
    """
    re_bulk = float(re_bulk)
    if not (math.isfinite(re_bulk) and re_bulk > 0.0):
        raise ValueError(f"Re_b must be finite and positive, got {re_bulk!r}")
    cf = DEAN_CF_COEFF * re_bulk ** DEAN_CF_EXPONENT
    u_tau_over_ub = math.sqrt(cf / 2.0)
    return {
        "correlation": "Dean 1978 J.Fluids Eng. 100:215, Cf = 0.073 Re_b^-0.25",
        "re_bulk_full_height": re_bulk,
        "cf": cf,
        "u_tau_over_ub": u_tau_over_ub,
        "re_tau": 0.5 * re_bulk * u_tau_over_ub,
    }


def channel_solver(closure: str = "sst", *, ny: int = 64, nx: int = 8,
                   reynolds: float = 2800.0, dt: float = 0.01,
                   transport_convection_blend: float | None = None,
                   production_limit: float | None = None,
                   body_force_relaxation: float = 1.0) -> PeriodicHillSolver:
    """The periodic-hill solver with ``hill_amplitude = 0``: a plane channel.

    Same mesh generator, same operators, same closures as the hill case; only
    the wall shape is removed, so any over-diffusivity that survives here is a
    property of the closure / discretisation and not of the hill geometry.
    """
    from .periodic_hill_round11 import build_solver

    return build_solver(
        closure, nx=nx, ny=ny, reynolds=reynolds, dt=dt,
        hill_amplitude=0.0,
        transport_convection_blend=transport_convection_blend,
        production_limit=production_limit,
        body_force_relaxation=body_force_relaxation,
    )


def _central_dudy(y: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Central-difference ``du/dy`` on a (uniform-in-eta) channel column."""
    dudy = np.gradient(u, y, edge_order=2)
    return dudy


def channel_postprocess(y, u_mean, nu_t_mean, *, nu: float, u_tau: float,
                        half_height: float) -> dict:
    """All four gate quantities from one time-averaged channel column.

    Pure function of the profiles so it can be tested against an analytic
    log-law / linear-stress profile without running the solver.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u_mean, dtype=float)
    nut = np.asarray(nu_t_mean, dtype=float)
    if not (y.shape == u.shape == nut.shape):
        raise ValueError("y, u_mean and nu_t_mean must have the same shape")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(u)) and np.all(np.isfinite(nut))):
        raise ValueError("profiles must be finite")
    nu = float(nu)
    u_tau = float(u_tau)
    h = float(half_height)
    if not (math.isfinite(u_tau) and u_tau > 0.0):
        raise ValueError(f"u_tau must be finite and positive, got {u_tau!r}")
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError(f"half_height must be finite and positive, got {h!r}")

    y_plus = u_tau * y / nu
    u_plus = u / u_tau
    nut_plus = nut / nu
    fit = log_law_fit(y_plus, u_plus)

    # C2: linear total shear stress identity
    dudy = _central_dudy(y, u)
    tau = (nu + nut) * dudy
    tau_ref = u_tau ** 2 * (1.0 - y / h)
    lo, hi = STRESS_WINDOW
    mask_s = (y / h >= lo) & (y / h <= hi) & (y / h <= 1.0)
    rel = np.abs(tau[mask_s] - tau_ref[mask_s]) / max(u_tau ** 2, 1e-30)
    stress_residual = float(np.max(rel)) if np.count_nonzero(mask_s) else float("nan")

    # C3: nu_t+ = kappa y+ - 1 in the log layer
    h_plus = float(y_plus.max())
    mask_n = (y_plus >= NUT_PLUS_YPLUS_LO) & (y_plus <= NUT_PLUS_YPLUS_HI_FRACTION * h_plus)
    model = LOG_KAPPA * y_plus - 1.0
    ratio = np.full(y_plus.shape, np.nan)
    valid = mask_n & (model > 1.0)
    ratio[valid] = nut_plus[valid] / model[valid]
    if np.count_nonzero(valid):
        nut_ratio_median = float(np.median(ratio[valid]))
        nut_ratio_min = float(np.min(ratio[valid]))
        nut_ratio_max = float(np.max(ratio[valid]))
    else:
        nut_ratio_median = nut_ratio_min = nut_ratio_max = float("nan")

    return {
        "y": [float(v) for v in y],
        "y_plus": [float(v) for v in y_plus],
        "u_plus": [float(v) for v in u_plus],
        "nu_t_plus": [float(v) for v in nut_plus],
        "log_law_model_nu_t_plus": [float(v) for v in model],
        "nu_t_plus_ratio": [float(v) if np.isfinite(v) else None for v in ratio],
        "log_law_fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                            else v) for k, v in fit.items()},
        "stress_residual_max_rel": stress_residual,
        "nu_t_plus_ratio_median": nut_ratio_median,
        "nu_t_plus_ratio_min": nut_ratio_min,
        "nu_t_plus_ratio_max": nut_ratio_max,
        "nu_t_plus_max": float(np.max(nut_plus)),
        "nu_t_plus_mean": float(np.mean(nut_plus)),
        "max_nu_t_ratio": float(np.max(nut)),
        "h_plus": h_plus,
        "nut_plus_window_cells": int(np.count_nonzero(valid)),
    }


def channel_verdict(post: dict, *, re_tau: float, reference: dict) -> dict:
    """Mechanical evaluation of the four pre-declared gates."""
    fit = post["log_law_fit"]
    kappa_eff = float(fit.get("kappa_eff", float("nan")))
    b_eff = float(fit.get("B", float("nan")))
    kappa_dev = abs(kappa_eff / LOG_KAPPA - 1.0) if math.isfinite(kappa_eff) else float("inf")
    gates = {
        "C1_log_law": bool(math.isfinite(kappa_eff) and kappa_dev <= TOL_KAPPA
                           and math.isfinite(b_eff) and abs(b_eff - LOG_B) <= TOL_B),
        "C2_linear_stress": bool(math.isfinite(post["stress_residual_max_rel"])
                                 and post["stress_residual_max_rel"] <= TOL_STRESS),
        "C3_eddy_viscosity_identity": bool(
            math.isfinite(post["nu_t_plus_ratio_median"])
            and NUT_PLUS_RATIO_BAND[0] <= post["nu_t_plus_ratio_median"]
            <= NUT_PLUS_RATIO_BAND[1]),
        "C4_friction": bool(math.isfinite(re_tau) and math.isfinite(reference["re_tau"])
                            and reference["re_tau"] > 0.0
                            and abs(re_tau / reference["re_tau"] - 1.0) <= TOL_RE_TAU),
    }
    over = (post["nu_t_plus_ratio_median"]
            if math.isfinite(post["nu_t_plus_ratio_median"]) else float("nan"))
    return {
        "gates": gates,
        "kappa_eff": kappa_eff,
        "kappa_relative_deviation": kappa_dev if math.isfinite(kappa_dev) else None,
        "B": b_eff,
        "re_tau": float(re_tau),
        "re_tau_dean": float(reference["re_tau"]),
        "re_tau_relative_deviation": (abs(re_tau / reference["re_tau"] - 1.0)
                                      if reference["re_tau"] > 0 else None),
        "nu_t_plus_over_log_law": over if math.isfinite(over) else None,
        "all_gates_pass": all(gates.values()),
        "interpretation": _interpretation(gates, over),
    }


def _interpretation(gates: dict, over: float) -> str:
    if not math.isfinite(over):
        return "no log-layer cells with kappa*y+ - 1 > 1; eddy viscosity unresolved"
    if gates["C3_eddy_viscosity_identity"] and gates["C2_linear_stress"]:
        return ("closure reproduces the log-layer eddy viscosity on a flat channel: "
                "the hill gap is a geometry / scheme effect, not a closure defect")
    if over > NUT_PLUS_RATIO_BAND[1] and gates["C2_linear_stress"]:
        return (f"closure over-produces nu_t by {over:.1f}x the log-law identity on a "
                "flat channel: closure-level defect (production/destruction balance), "
                "not the hill geometry")
    return "gates disagree; see the per-gate table before drawing a conclusion"


def run_channel(closure: str = "sst", *, ny: int = 64, nx: int = 8,
                dt: float = 0.01, reynolds: float = 2800.0,
                steps: int = 30000, burn_in: int = 10000, block_steps: int = 5000,
                chunk: int = 250, transport_convection_blend: float | None = None,
                production_limit: float | None = None,
                body_force_relaxation: float = 1.0, verbose: bool = True) -> dict:
    """Integrate the plane channel with the selected closure and score the gates.

    One statistics block is ``block_steps`` steps; inside a block the fields are
    sampled every ``chunk`` steps, so ``nu_t`` is a *time mean* like ``u`` (the
    closure's instantaneous field is not an equilibrium quantity on a coarse
    grid).  Blocks are reported individually so a non-equilibrated record is
    visible instead of being averaged into a single number.
    """
    solver = channel_solver(
        closure, ny=ny, nx=nx, reynolds=reynolds, dt=dt,
        transport_convection_blend=transport_convection_blend,
        production_limit=production_limit,
        body_force_relaxation=body_force_relaxation)
    turb = solver.turbulence
    nu = float(solver.nu)
    half_height = 0.5 * float(solver.config.channel_top_m)
    re_bulk = float(solver.config.bulk_velocity) * float(solver.config.channel_top_m) / nu
    reference = dean_friction(re_bulk)

    t0 = time.time()
    if burn_in > 0:
        solver.advance(int(burn_in))
    if verbose:
        print(f"[round15/channel/{closure}] spin-up {burn_in} steps "
              f"({time.time() - t0:.0f}s)", flush=True)

    blocks = []
    remaining = int(steps) - int(burn_in)
    while remaining > 0:
        this_block = int(min(block_steps, remaining))
        acc_u = np.zeros_like(solver.u)
        acc_v = np.zeros_like(solver.u)
        acc_nut = np.zeros_like(solver.u)
        samples = 0
        done = 0
        while done < this_block:
            c = min(chunk, this_block - done)
            solver.advance(c)
            acc_u += solver.u
            acc_v += solver.v
            acc_nut += turb.viscosity()
            samples += 1
            done += c
        mean_u = acc_u / samples
        mean_v = acc_v / samples
        mean_nut = acc_nut / samples
        tau_b, tau_t = solver.wall_shear(mean_u, mean_v)
        tau_w = 0.5 * (abs(float(np.mean(tau_b))) + abs(float(np.mean(tau_t))))
        u_tau = math.sqrt(max(tau_w, 0.0))
        re_tau = u_tau * half_height / nu
        j = nx // 2
        post = channel_postprocess(solver.mesh.yc[:, j], mean_u[:, j], mean_nut[:, j],
                                   nu=nu, u_tau=u_tau, half_height=half_height)
        verdict = channel_verdict(post, re_tau=re_tau, reference=reference)
        block = {
            "t": float(solver.steps * dt),
            "samples": samples,
            "u_tau": u_tau,
            "re_tau": float(re_tau),
            "bulk_velocity": float(solver.bulk_velocity(mean_u)),
            "body_force": float(solver.body_force),
            "profile": post,
            "verdict": verdict,
        }
        blocks.append(block)
        remaining -= this_block
        if verbose:
            print(f"[round15/channel/{closure}] t={block['t']:7.1f} "
                  f"Re_tau={re_tau:7.2f} (Dean {reference['re_tau']:.2f}) "
                  f"kappa_eff={verdict['kappa_eff']:.3f} "
                  f"nu_t+/loglaw={verdict['nu_t_plus_over_log_law']} "
                  f"max nu_t/nu={post['max_nu_t_ratio']:.1f} "
                  f"gates={verdict['gates']}", flush=True)
    wall = time.time() - t0

    last = blocks[-1]["verdict"]
    return {
        "case": "CASE-FVM-PERIODIC-HILLS",
        "round": 15,
        "experiment": "plane channel closure audit (hill_amplitude = 0)",
        "closure": closure,
        "config": {
            "nx": int(nx), "ny": int(ny), "dt": float(dt),
            "reynolds": float(reynolds), "nu": nu,
            "full_height": float(solver.config.channel_top_m),
            "half_height": half_height,
            "re_bulk_full_height": re_bulk,
            "steps": int(steps), "burn_in": int(burn_in),
            "block_steps": int(block_steps), "chunk": int(chunk),
            "body_force_relaxation": float(body_force_relaxation),
            "transport_convection_blend": (None if transport_convection_blend is None
                                           else float(transport_convection_blend)),
            "production_limit": production_limit,
            "momentum_convection_blend": float(solver.config.convection_blend),
        },
        "reference": reference,
        "tolerances": {
            "C1_kappa": TOL_KAPPA, "C1_B": TOL_B, "C2_stress": TOL_STRESS,
            "C3_ratio_band": list(NUT_PLUS_RATIO_BAND), "C4_re_tau": TOL_RE_TAU,
        },
        "blocks": blocks,
        "final": last,
        "equilibrated": bool(len(blocks) >= 2 and all(
            abs(b["re_tau"] / blocks[-1]["re_tau"] - 1.0) <= 0.02 for b in blocks[-2:])),
        "wall_clock_s": wall,
    }


def write_json(payload: dict, path) -> Path:
    """Write an artefact without NaN/Infinity (which JSON readers reject)."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=1, allow_nan=False))
    return path

# --------------------------------------------------------------------------- #
# hill re-score against the single-sourced reference (round 14's DNS audit)
# --------------------------------------------------------------------------- #
def score_hill(payload: dict) -> dict:
    """Score a round-11-schema hill payload against the **current** reference.

    Two reasons not to reuse the payload's own ``gate`` block:

    * the payload's ``profile_comparison`` is built from
      :data:`~tensorfvm.verification.periodic_hill_round6.REFERENCE` /
      ``BREUER_PROFILE_Y05``, i.e. the legacy digitised table that round 10
      measured as up to 3.5x too large downstream and round 14 retired;
    * the ledger reference moved to ``x_r/H = 5.41`` (the ERCOFTAC UFR 3-30
      value read from the distributed DNS files) while the round-11 gate still
      scores ``5.34``.

    So this function is the round-15 scoring path: ``x_r/H`` against
    :data:`~tensorfvm.verification.periodic_hill_reference.X_R_OVER_H`, and the
    ``y/H = 0.5`` profile against :func:`...periodic_hill_reference.y05_table`
    with a 10 % relative band, interpolating the computed profile at the
    reference stations.  Both the legacy (5.34) and current (5.41) errors are
    reported so the artefact cannot hide which number it is quoting.
    """
    from .periodic_hill_reference import (
        STATIONS, X_R_OVER_H, X_R_TOLERANCE, y05_table)

    ta = payload.get("time_averaged") or {}
    x_r = ta.get("x_r_over_H")
    x_r = float(x_r) if isinstance(x_r, (int, float)) and math.isfinite(float(x_r)) else None
    ref = float(X_R_OVER_H)
    x_r_err = abs(x_r - ref) / ref if x_r is not None else None

    hp = payload.get("horizontal_profile_y_H_0.5") or {}
    xs = np.asarray(hp.get("x_over_H") or [], dtype=float)
    us = np.asarray(hp.get("u_over_Ub") or [], dtype=float)
    table = y05_table()
    stations = []
    if xs.size >= 2:
        order = np.argsort(xs)
        xs, us = xs[order], us[order]
        for x in STATIONS:
            if float(x) not in table:
                continue
            ref_u = float(table[float(x)])
            comp = float(np.interp(float(x), xs, us))
            delta = abs(comp - ref_u)
            # round-9/11 tolerance convention (one gate, not one per round):
            # relative where |ref| >= 0.1, absolute 0.10 U_b otherwise -- with
            # u/U_b = -0.046 a purely relative band would call a 0.005 miss 10 %
            if abs(ref_u) >= 0.1:
                band, tol = "relative", X_R_TOLERANCE * abs(ref_u)
            else:
                band, tol = "absolute", X_R_TOLERANCE
            stations.append({
                "x_over_H": float(x),
                "u_over_Ub_computed": comp,
                "u_over_Ub_reference": ref_u,
                "delta": delta,
                "error_rel": (delta / abs(ref_u) if abs(ref_u) > 0 else None),
                "band": band,
                "tolerance": tol,
                "pass": bool(delta <= tol),
            })
    n_admissible = len(stations)
    n_pass = sum(1 for s in stations if s["pass"])
    return {
        "closure": (payload.get("closure") or {}),
        "config": (payload.get("config") or {}),
        "samples": payload.get("samples"),
        "x_r_over_H": x_r,
        "x_r_over_H_reference": ref,
        "x_r_over_H_error": x_r_err,
        "x_r_over_H_legacy_reference": 5.34,
        "x_r_over_H_error_legacy": (abs(x_r - 5.34) / 5.34 if x_r is not None else None),
        "x_r_over_H_pass": bool(x_r_err is not None and x_r_err <= X_R_TOLERANCE),
        "profile_stations": stations,
        "profile_stations_pass": int(n_pass),
        "profile_stations_admissible": int(n_admissible),
        "profile_pass": bool(n_admissible > 0 and n_pass == n_admissible),
        "stationary": bool(((payload.get("stationarity") or {}).get("x_r_over_H") or {})
                           .get("ok", False)),
        "max_nu_t_ratio": ta.get("max_nu_t_ratio"),
        "acceptance_pass": bool(x_r_err is not None and x_r_err <= X_R_TOLERANCE
                                and n_admissible > 0 and n_pass == n_admissible),
        "reference_source": "tensorfvm.verification.periodic_hill_reference "
                            "(ERCOFTAC UFR 3-30 distributed DNS profiles)",
    }
