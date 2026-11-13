"""Reproduce the CASE-FVM-DAMBREAK-STOKER headline numbers from code.

The point of this script is that **no number in the report is typed by hand**:
every table entry of ``docs/case-fvm-dambreak-stoker/REPORT.md`` is produced
here and written to ``artifacts/reproduction.json``.  It also records which
quantities converge monotonically under grid refinement and which do not, so
that the report's convergence statement can be checked instead of asserted.

Usage::

    PYTHONPATH=src python docs/case-fvm-dambreak-stoker/verify_reference.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.shallow_water import (
    RITTER_DRY,
    STOKER_WET,
    DamBreakCase,
    convergence_study,
    exactness_checks,
    run_case,
)
from tensorfvm.shallow_water.crosscheck import (
    jump_trajectory_speed,
    level_trajectory_speed,
    riemann_solution,
    riemann_star_state,
)
from tensorfvm.shallow_water.exact import (
    ritter_state,
    stoker_constant_state,
    stoker_state,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
G = 9.81
H0 = 1.0
#: h1/h0 grid used to cross-check the closed form against the generic solver
RATIOS = (0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 1e-3)


def crosscheck_analytic() -> dict:
    """Closed-form Stoker/Ritter vs an independent generic Riemann solver."""
    rows = []
    for ratio in RATIOS:
        h1 = ratio * H0
        closed = stoker_constant_state(H0, h1, G)
        generic = riemann_star_state(H0, 0.0, h1, 0.0, G)
        rows.append({
            "h1_over_h0": ratio,
            "h2_closed": closed["h2"],
            "h2_generic": generic["h_star"],
            "h2_relative": abs(closed["h2"] - generic["h_star"]) / closed["h2"],
            "u2_closed": closed["u2"],
            "u2_generic": generic["u_star"],
            "u2_relative": abs(closed["u2"] - generic["u_star"]) / closed["u2"],
            "shock_speed_closed": closed["shock_speed"],
            "shock_speed_generic": generic["s_shock_right"],
            "shock_speed_relative": abs(closed["shock_speed"]
                                        - generic["s_shock_right"])
            / closed["shock_speed"],
            "wave_families": f"{generic['left_family']}/{generic['right_family']}",
            "bisection_iterations": generic["iterations"],
        })
    # dry limit: Stoker profile must approach Ritter as h1 -> 0
    t = 2.0 * math.sqrt(H0 / G)
    x = np.linspace(RITTER_DRY.x_min, RITTER_DRY.x_max, 40_001)
    h_ritter, _ = ritter_state(x, t, H0, G)
    dry_limit = []
    for h1 in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        h_wet, _ = stoker_state(x, t, H0, h1, G)
        sol = stoker_constant_state(H0, h1, G)
        dry_limit.append({
            "h1": h1,
            "l1_profile_relative": float(np.abs(h_wet - h_ritter).sum()
                                         / h_ritter.sum()),
            "front_speed": sol["shock_speed"],
            "front_speed_relative": abs(sol["shock_speed"]
                                        - 2.0 * math.sqrt(G * H0))
            / (2.0 * math.sqrt(G * H0)),
        })
    return {"ratio_rows": rows, "dry_limit": dry_limit}


def measure_wave_speeds() -> dict:
    """Wave speeds measured from the sampled reference, not from its formulas."""
    t0, t1 = 1.0, 2.0
    c0 = math.sqrt(G * H0)
    out = {"stoker": [], "ritter": []}
    for h1 in (0.5, 0.25, 0.05):
        closed = stoker_constant_state(H0, h1, G)
        traj = jump_trajectory_speed(
            lambda xi: stoker_state(xi, 1.0, H0, h1, G),
            0.5 * closed["shock_speed"], 1.5 * closed["shock_speed"],
            t0, t1)
        out["stoker"].append({
            "h1": h1,
            "shock_speed_formula": closed["shock_speed"],
            "shock_speed_measured": traj["speed"],
            "relative": abs(traj["speed"] - closed["shock_speed"])
            / closed["shock_speed"],
            "jump_height": traj["jump_height"],
            "jump_height_exact": closed["h2"] - closed["h1"],
            "fan_tail_speed_formula": closed["tail_speed"],
            "fan_tail_speed_from_invariant": 2.0 * closed["c0"] - 3.0 * closed["c2"],
            "fan_levels": [
                {
                    "level": level,
                    "measured_speed": level_trajectory_speed(
                        lambda xi: stoker_state(xi, 1.0, H0, h1, G), level,
                        closed["head_speed"], closed["tail_speed"],
                        t0, t1)["speed"],
                    "fan_law_speed": 2.0 * closed["c0"]
                    - 3.0 * math.sqrt(G * level),
                }
                for level in (closed["h2"] + 0.1 * (closed["h0"] - closed["h2"]),
                              closed["h2"] + 0.5 * (closed["h0"] - closed["h2"]),
                              closed["h2"] + 0.9 * (closed["h0"] - closed["h2"]))
            ],
        })
    for level in (0.01, 0.1, 0.3, 0.6):
        traj = level_trajectory_speed(
            lambda xi: ritter_state(xi, 1.0, H0, G), level,
            -c0, 2.5 * c0, t0, t1)
        exact = 2.0 * c0 - 3.0 * math.sqrt(G * level)
        out["ritter"].append({
            "level": level,
            "level_speed_measured": traj["speed"],
            "level_speed_exact": exact,
            "relative": abs(traj["speed"] - exact) / (2.0 * c0),
        })
    # the dry-bed front speed from the generic dry Riemann construction
    generic_dry = riemann_star_state(H0, 0.0, 0.0, 0.0, G)
    out["ritter_front"] = {
        "front_speed_closed_form": 2.0 * c0,
        "front_speed_generic_riemann": generic_dry["s_shock_right"],
        "relative": abs(generic_dry["s_shock_right"] - 2.0 * c0) / (2.0 * c0),
    }
    return out


def weak_bore_conditioning() -> list[dict]:
    """Bore speed near ``h1 -> h0``: subtractive vs cancellation-free form.

    The subtractive Rankine-Hugoniot form ``U = h2 u2 / (h2 - h1)`` is compared
    with the cancellation-free identity actually used by the package and with a
    60-digit ``Decimal`` reference solved by this script.
    """
    from decimal import Decimal, localcontext

    rows = []
    for gap in (1e-4, 1e-6, 1e-9, 1e-12):
        h1 = H0 * (1.0 - gap)
        sol = stoker_constant_state(H0, h1, G)
        with localcontext() as ctx:
            ctx.prec = 60
            H0d, H1d, Gd = Decimal(repr(H0)), Decimal(repr(h1)), Decimal(repr(G))
            c0 = (Gd * H0d).sqrt()
            lo, hi = (Gd * H1d).sqrt(), c0

            def residual(c2):
                h2 = c2 * c2 / Gd
                return 2 * (c0 - c2) - (h2 - H1d) * (Gd * (h2 + H1d)
                                                    / (2 * h2 * H1d)).sqrt()

            for _ in range(400):
                mid = (lo + hi) / 2
                if residual(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            c2 = (lo + hi) / 2
            h2d = c2 * c2 / Gd
            u_ref = float((Gd * h2d * (h2d + H1d) / (2 * H1d)).sqrt())
        subtractive = sol["h2"] * sol["u2"] / (sol["h2"] - h1)
        rows.append({
            "depth_gap": gap,
            "reference_speed_decimal60": u_ref,
            "package_speed": sol["shock_speed"],
            "package_relative_error": abs(sol["shock_speed"] - u_ref) / u_ref,
            "subtractive_speed": subtractive,
            "subtractive_relative_error": abs(subtractive - u_ref) / u_ref,
            "weak_bore_coefficient_package": (math.sqrt(G * H0)
                                              - sol["shock_speed"])
            / (math.sqrt(G * H0) * gap),
            "weak_bore_coefficient_subtractive": (math.sqrt(G * H0)
                                                  - subtractive)
            / (math.sqrt(G * H0) * gap),
        })
    return rows


def profile_l1(case: DamBreakCase, n: int = 800, stride: int = 4) -> dict:
    """Analytic vs generic-Riemann L1 error + subsampled profiles for figures."""
    run = run_case(case, n=n)
    x = np.asarray(run["profile"]["x"])
    t = run["discretisation"]["t_end"]
    h_ref, u_ref = riemann_solution(x, t, case.h0, 0.0, case.h1, 0.0,
                                    case.gravity, x0=case.x_dam)
    h_ex = np.asarray(run["profile"]["h_exact"])
    h_num = np.asarray(run["profile"]["h_numeric"])
    sel = slice(None, None, stride)
    return {
        "case": case.key,
        "n": n,
        "t_end": t,
        "reference_vs_generic_riemann_l1_depth": float(
            np.abs(h_ex - h_ref).sum() / max(h_ref.sum(), 1e-300)),
        "reference_vs_generic_riemann_linf_velocity": float(
            np.abs(np.asarray(run["profile"]["u_exact"]) - u_ref).max()),
        "profile": {
            "x": x[sel].tolist(),
            "h_numeric": h_num[sel].tolist(),
            "h_exact": h_ex[sel].tolist(),
            "h_generic_riemann": h_ref[sel].tolist(),
            "exact_front": case.exact_front(t),
        },
    }


def main() -> dict:
    payload = {
        "case": "CASE-FVM-DAMBREAK-STOKER",
        "anchor": "CASE-FVM-DAMBREAK-STOKER",
        "tolerance": 0.05,
        "parameters": {
            "gravity": G,
            "h0": H0,
            "t_star_end": 2.0,
            "ritter_dry": {"h1": 0.0, "domain": [RITTER_DRY.x_min,
                                                 RITTER_DRY.x_max]},
            "stoker_wet": {"h1": 0.5, "domain": [STOKER_WET.x_min,
                                                 STOKER_WET.x_max]},
            "stoker_strong": {"h0_over_h1": 4.0},
        },
        "exactness": exactness_checks(G),
        "crosscheck": crosscheck_analytic(),
        "wave_speeds": measure_wave_speeds(),
        "weak_bore_conditioning": weak_bore_conditioning(),
        "profiles": [profile_l1(RITTER_DRY), profile_l1(STOKER_WET)],
        "convergence": {},
        "headline": {},
    }
    for case in (RITTER_DRY, STOKER_WET):
        study = convergence_study(case, resolutions=(100, 200, 400, 800))
        payload["convergence"][case.key] = study
        run = run_case(case, n=800)
        m = {k: v for k, v in run["metrics"].items()
             if not isinstance(v, list)}
        payload["headline"][case.key] = {
            "metrics": m,
            "verdict": run["verdict"],
            "discretisation": {k: v for k, v in run["discretisation"].items()},
        }
    strong = DamBreakCase("stoker-wet-r4", "wet", H0, 0.25, G, 0.0, -4.0, 6.0,
                          2.0, "Stoker (1957) ch. 10")
    run = run_case(strong, n=800)
    payload["headline"]["stoker-wet-r4"] = {
        "metrics": {k: v for k, v in run["metrics"].items()
                    if not isinstance(v, list)},
        "verdict": run["verdict"],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "reproduction.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return payload


if __name__ == "__main__":
    main()