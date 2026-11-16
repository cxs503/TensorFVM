"""Does the CASE-FVM-DAMBREAK-STOKER gate actually have detection power?

Motivation
----------
The anchor's PASS rests on graded criteria (``shock_position<5%``,
``step_ratio<5%``, ``plateau_h2<5%``, ``depth_l1<5%``, ``monotone(no_extrema)``)
whose values are all 100x-10000x *inside* the 5 % tolerance.  A number that sits
that deep inside a gate is only evidence if the gate can be shown to *move* when
the solution is wrong -- otherwise "PASS with 100x margin" is a statement about
the estimator, not about the solver.

This driver is a verifier audit, not a solver change.  It answers three
questions with measurements:

E0 **Does the gate ever fail on this solver?**  The grid ladder is re-run and
   every graded criterion is recorded per grid, so the audit never cites a gate
   that has only ever been observed to pass.  (The coarse grids fail on
   ``front_series`` -- an honest, pre-existing negative result.)

E1 **Does the committed artifact reproduce?**  The ladder is re-integrated here
   and every headline number is differenced against
   ``src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json``
   (the file the case-pool entry is pinned to).  A reproduction delta is
   reported for each quantity; the pool entry is only as good as this file.

E2 **How much injected error does each graded criterion need before it fires?**
   The accepted-grid (``n = 1600``) profiles are perturbed with *declared*
   transforms and re-graded through the **same** private code path the gate
   uses (``_wet_metrics`` / ``_dry_metrics`` -> ``_verdict``), so nothing is
   re-implemented and nothing is re-tuned:

   * ``P1 translate``  ``h(x - delta)``: displaces every wave coherently --
     the "solver put the bore/front in the wrong place" error class;
   * ``P2 amplitude``  scale the downstream state by ``1 + eps``: the
     "right position, wrong still-water step" error class;
   * ``P3 oscillation`` one-cell peak of declared amplitude at the analytic
     fan/plateau kink: the "limiter let a ripple through" error class;
   * ``P4 boundary`` the same peak moved onto a boundary cell, where the
     detector mask structurally cannot see it (scope limit, quantified).

   For each row the *responsiveness* ``graded_error / injected_relative`` is
   reported.  A gate with detection power has responsiveness ``O(1)`` and a
   detection threshold at or below its declared tolerance; a near-tautological
   metric shows up as responsiveness ``<< 1`` and a threshold far above 5 %.

Declared before any run
-----------------------
``TRANSLATIONS``, ``AMPLITUDES``, ``EXTREMUM_MULTIPLES``, ``GRIDS`` and the
perturbation *positions* (analytic kinks) are fixed here; nothing in the sweep
is chosen from the results.  The audit grades the depth-based criteria only --
momentum/velocity surfaces are covered by ``audit_dambreak_momentum.py`` and
are out of scope by declaration, not by omission.

Outputs
-------
``src/tensorfvm/verification/artifacts/shallow-water-dam-break-gate-power/``
(``detection_power.json`` + ``detection_power.png``), written programmatically;
the report quotes the JSON, never re-types a number.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.shallow_water.dam_break import (  # noqa: E402
    EXTREMUM_FRACTION,
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    _dry_metrics,
    _verdict,
    _wet_metrics,
    run_case,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                            "artifacts", "shallow-water-dam-break-gate-power")
COMMITTED = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                         "artifacts", "shallow-water-dam-break", "metrics.json")

#: grids of the negative-control ladder (E0).  1600 is the accepted grid the
#: case pool pins its gate values to; 100/200 are expected to fail.
GRIDS = (100, 200, 400, 800, 1600)

#: accepted grid for the perturbation battery (E2)
GRID = 1600

#: declared relative displacements for P1, in units of the analytic wave travel
TRANSLATIONS = (0.005, 0.01, 0.02, 0.05, 0.08)

#: declared relative amplitude perturbations for P2, applied downstream only
AMPLITUDES = (0.005, 0.01, 0.02, 0.05, 0.08)

#: declared oscillation amplitudes for P3, in multiples of the gate floor
#: ``EXTREMUM_FRACTION * h0``.  The detector fires on a one-cell peak of height
#: ``A`` only when ``A`` also *reverses* the local trend: on a flat cell the
#: condition is ``A > floor``, on a sloped cell it is ``A > floor + |slope|``.
#: The sweep therefore runs three decades so the *effective* threshold can be
#: located on both a flat and a sloped probe position instead of assumed.
EXTREMUM_MULTIPLES = (0.3, 0.6, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

#: amplitude (in floor multiples) used for the P4 boundary probe
BOUNDARY_MULTIPLE = 10.0

#: which graded criterion each perturbation is *for*.  A criterion only has
#: detection power if its own error class makes it fail, so each perturbation is
#: graded against its target criterion and *all* criteria are recorded.
TARGET = {
    "translate": {"wet": "shock_position<5%", "dry": "front_fan<5%"},
    "amplitude": {"wet": "step_ratio<5%", "dry": None},
    "oscillation": {"wet": "monotone(no_extrema)", "dry": "monotone(no_extrema)"},
}

#: reproduction tolerance: same code, same platform -> differences come only
#: from summation order.  Loose enough not to be flaky, tight enough that any
#: real drift (a re-tuned constant, a changed estimator) fails.
REPRO_REL_TOL = 1e-6


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"


def _load_committed() -> dict:
    with open(COMMITTED, encoding="utf-8") as handle:
        return json.load(handle)


def _ladder(cases) -> dict:
    """E0: per-grid verdicts -- proof the gate is capable of failing."""
    out = {}
    for case in cases:
        rows = []
        for n in GRIDS:
            run = run_case(case, n=n)
            verdict = {k: {"value": v["value"], "pass": v["pass"]}
                       for k, v in run["verdict"].items() if k != "all_pass"}
            rows.append({
                "n": n,
                "dx": run["discretisation"]["dx"],
                "wall_time_s": run["discretisation"]["wall_time_s"],
                "graded": verdict,
                "all_pass": run["verdict"]["all_pass"],
                "failing": sorted(k for k, v in verdict.items() if not v["pass"]),
            })
        out[case.key] = rows
    return out


def _reproduction(ladder: dict, committed: dict) -> dict:
    """E1: headline quantities vs the committed artifact."""
    rows = []
    for key, rows_ladder in ladder.items():
        ref = committed["cases"][key]["metrics"]
        live = next(r for r in rows_ladder if r["n"] == GRID)
        pairs = []
        # graded quantities, mapped to their artifact names
        if key == "stoker-wet":
            pairs = [("shock_error", "shock_position<5%"),
                     ("step_ratio_error", "step_ratio<5%"),
                     ("h2_error", "plateau_h2<5%"),
                     ("depth_l1", "depth_l1<5%")]
        else:
            pairs = [("front_fan_error", "front_fan<5%"),
                     ("depth_l1", "depth_l1<5%")]
        for art_key, crit in pairs:
            a = float(ref[art_key])
            b = float(live["graded"][crit]["value"])
            denom = max(abs(a), 1e-300)
            rows.append({
                "case": key,
                "quantity": art_key,
                "artifact": a,
                "rerun": b,
                "abs_delta": abs(a - b),
                "rel_delta": abs(a - b) / denom,
                "reproduced": bool(abs(a - b) / denom <= REPRO_REL_TOL),
            })
        rows.append({
            "case": key,
            "quantity": "local_extrema",
            "artifact": float(ref["local_extrema"]),
            "rerun": float(live["graded"]["monotone(no_extrema)"]["value"]),
            "abs_delta": abs(float(ref["local_extrema"])
                             - float(live["graded"]["monotone(no_extrema)"]["value"])),
            "rel_delta": 0.0 if int(ref["local_extrema"]) == int(
                live["graded"]["monotone(no_extrema)"]["value"]) else 1.0,
            "reproduced": bool(int(ref["local_extrema"]) == int(
                live["graded"]["monotone(no_extrema)"]["value"])),
        })
    return {
        "rows": rows,
        "all_reproduced": all(r["reproduced"] for r in rows),
        "max_rel_delta": max(r["rel_delta"] for r in rows),
        "tolerance": REPRO_REL_TOL,
    }


def _regrade(case, h: np.ndarray, n: int, baseline: dict) -> dict:
    """Re-grade a perturbed depth profile through the gate's own code path.

    ``set_state`` resets the clock, so the end time is restored afterwards; the
    ``x_f(t)`` series is *not* re-graded here (an end-of-run depth perturbation
    cannot change it), its accepted-grid value is carried over verbatim and
    flagged in the output.
    """
    solver = ShallowWater1D(n, case.x_min, case.x_max,
                            gravity=case.gravity, cfl=0.4, order=2)
    solver.set_state(h)
    solver.time = baseline["discretisation"]["t_end"]
    metrics = (_dry_metrics if case.kind == "dry" else _wet_metrics)(
        case, solver, baseline["discretisation"]["volume_final"])
    metrics["front_series"] = baseline["metrics"]["front_series"]
    verdict = _verdict(case, metrics)
    return {
        "metrics": metrics,
        "graded": {k: {"value": v["value"], "pass": v["pass"]}
                   for k, v in verdict.items() if k != "all_pass"},
        "all_pass": verdict["all_pass"],
        "failing": sorted(k for k, v in verdict.items()
                          if k != "all_pass" and not v["pass"]),
    }


def _translate(x: np.ndarray, h: np.ndarray, delta: float) -> np.ndarray:
    """``h(x - delta)`` with edge clamping (mass is not conserved by design)."""
    return np.interp(x - delta, x, h)


def _case_reference(case, baseline: dict) -> dict:
    m = baseline["metrics"]
    if case.kind == "wet":
        fan_end = float(m["fan_end_exact"])
        shock = float(m["shock_exact"])
        return {"wave_x": shock,
                "crit": "shock_position<5%", "graded": "shock_error",
                "kink_x": fan_end,
                # mid-plateau: the exact solution is *constant* here, so any
                # extremum is numerical and the detector works at its floor
                "flat_x": 0.5 * (fan_end + shock)}
    # dry bed: the constant state meets the rarefaction at x = -c0 t, i.e. half
    # of the analytic front travel (-1/2 * 2 c0 t)
    kink = -0.5 * float(m["front_exact"])
    return {"wave_x": float(m["front_exact"]),
            "crit": "front_fan<5%", "graded": "front_fan_error",
            "kink_x": kink,
            "flat_x": kink - 1.0}


def _battery(cases, baselines: dict) -> dict:
    """E2: injected-error sweeps, graded by the production gate."""
    out = {}
    floor = EXTREMUM_FRACTION  # * h0, h0 = 1 for both pool cases
    for case in cases:
        base = baselines[case.key]
        x = np.asarray(base["profile"]["x"], dtype=float)
        h = np.asarray(base["profile"]["h_numeric"], dtype=float)
        ref = _case_reference(case, base)
        rows = []

        # ---- P1: coherent displacement of every wave --------------------
        for rel in TRANSLATIONS:
            delta = rel * abs(ref["wave_x"])
            graded = _regrade(case, _translate(x, h, delta), GRID, base)
            value = float(graded["graded"][ref["crit"]]["value"])
            rows.append({
                "probe": "P1_translate",
                "target_criterion": ref["crit"],
                "injected": {"rel_displacement": rel, "delta_x": delta},
                "graded_value": value,
                "pass": graded["graded"][ref["crit"]]["pass"],
                "responsiveness": value / rel if rel else None,
                "all_failing": graded["failing"],
            })

        # ---- P2: downstream amplitude only (wet bed has a still-water step) ----
        if case.kind == "wet":
            shock = float(base["metrics"]["shock_position"])
            for eps in AMPLITUDES:
                h_p = h.copy()
                h_p[x > shock] *= (1.0 + eps)
                graded = _regrade(case, h_p, GRID, base)
                value = float(graded["graded"]["step_ratio<5%"]["value"])
                rows.append({
                    "probe": "P2_amplitude",
                    "target_criterion": "step_ratio<5%",
                    "injected": {"rel_amplitude": eps},
                    "graded_value": value,
                    "pass": graded["graded"]["step_ratio<5%"]["pass"],
                    "responsiveness": value / eps if eps else None,
                    "all_failing": graded["failing"],
                })

        # ---- P3: one-cell spike, flat position and sloped (kink) position ----
        crit = "monotone(no_extrema)"
        for where, x_probe in (("flat", ref["flat_x"]), ("kink", ref["kink_x"])):
            k = int(np.argmin(np.abs(x - x_probe)))
            background = float(h[k + 1] - h[k - 1]) / 2.0  # local trend per cell
            for mult in EXTREMUM_MULTIPLES:
                amp = mult * floor * case.h0
                h_p = h.copy()
                h_p[k] += amp
                graded = _regrade(case, h_p, GRID, base)
                rows.append({
                    "probe": f"P3_oscillation_{where}",
                    "target_criterion": crit,
                    "injected": {"floor_multiple": mult,
                                 "amplitude_over_h0": amp / case.h0,
                                 "cell": k, "x": float(x[k]),
                                 "local_trend_per_cell": background},
                    "graded_value": float(graded["graded"][crit]["value"]),
                    "extrema": int(graded["metrics"]["local_extrema"]),
                    "pass": graded["graded"][crit]["pass"],
                    "responsiveness": None,
                    "all_failing": graded["failing"],
                })

        # ---- P4: the same peak on a boundary cell (detector scope) ------
        bound = []
        for idx in (0, x.size - 1):
            h_p = h.copy()
            h_p[idx] += BOUNDARY_MULTIPLE * floor * case.h0
            graded = _regrade(case, h_p, GRID, base)
            bound.append({
                "cell": int(idx),
                "extrema": int(graded["metrics"]["local_extrema"]),
                "monotone_pass": graded["graded"]["monotone(no_extrema)"]["pass"],
                "depth_l1": float(graded["graded"]["depth_l1<5%"]["value"]),
            })
        out[case.key] = {"reference": ref, "rows": rows,
                         "boundary_probe": {
                             "floor_multiple": BOUNDARY_MULTIPLE,
                             "probes": bound,
                             "detected": any(p["extrema"] > 0 for p in bound),
                         }}
    return out


def _thresholds(battery: dict) -> dict:
    """Smallest injected magnitude at which each target criterion fires."""
    out = {}
    for key, block in battery.items():
        per = {}
        for row in block["rows"]:
            probe = row["probe"]
            if probe.startswith("P3_oscillation"):
                mag = row["injected"]["floor_multiple"]
                crit = row["target_criterion"] + "@" + probe.rsplit("_", 1)[-1]
                unit = "floor_multiple"
            elif probe == "P2_amplitude":
                mag = row["injected"]["rel_amplitude"]
                crit, unit = row["target_criterion"], "relative"
            else:
                mag = row["injected"]["rel_displacement"]
                crit, unit = row["target_criterion"], "relative"
            if row["pass"]:
                continue
            cur = per.get(crit)
            if cur is None or mag < cur["injected"]:
                per[crit] = {"injected": mag, "unit": unit,
                             "graded_value": row["graded_value"]}
        out[key] = per
    return out


def _figure(payload: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    for key, style in (("stoker-wet", "o-"), ("ritter-dry", "s-")):
        rows = [r for r in payload["battery"][key]["rows"] if r["probe"] == "P1_translate"]
        xv = [r["injected"]["rel_displacement"] * 100 for r in rows]
        yv = [r["graded_value"] * 100 for r in rows]
        ax.plot(xv, yv, style, label=f"{key} ({payload['battery'][key]['reference']['crit']})")
    lim = np.array([0.4, 10.0])
    ax.plot(lim, lim, "k--", lw=1.0, label="responsiveness = 1")
    ax.axhline(TOLERANCE * 100, color="r", ls=":", label="5 % gate")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("injected displacement [% of analytic wave travel]")
    ax.set_ylabel("graded wave-position error [%]")
    ax.set_title("P1: does the gate move 1:1 with a displaced wave?")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for where, color in (("flat", "C0"), ("kink", "C2")):
        rows = [r for r in payload["battery"]["stoker-wet"]["rows"]
                if r["probe"] == f"P3_oscillation_{where}"]
        if not rows:
            continue
        ax.step([r["injected"]["floor_multiple"] for r in rows],
                [r["extrema"] for r in rows], where="mid", color=color,
                label=f"stoker-wet, {where} cell")
    bnd = payload["battery"]["stoker-wet"]["boundary_probe"]
    ax.axhline(0.0, color="C3", ls=":", label="boundary cells (mask-excluded)")
    ax.axvline(1.0, color="k", ls="--", lw=1.0, label="declared floor 1e-4 h0")
    ax.annotate(f"boundary probe ({bnd['floor_multiple']}x floor):\n"
                f"extrema = {[p['extrema'] for p in bnd['probes']]}",
                xy=(0.03, 0.72), xycoords="axes fraction", fontsize=7, color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("one-cell oscillation amplitude [/ gate floor]")
    ax.set_ylabel("counted local extrema")
    ax.set_title("P3/P4: monotonicity gate effective threshold and blind spot")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(n: int = GRID, fig: bool = True) -> dict:
    cases = (STOKER_WET, RITTER_DRY)
    committed = _load_committed()
    wall = time.perf_counter()
    ladder = _ladder(cases)
    reproduction = _reproduction(ladder, committed)

    baselines = {c.key: run_case(c, n=n) for c in cases}
    battery = _battery(cases, baselines)
    thresholds = _thresholds(battery)
    wall = time.perf_counter() - wall

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
            "command": ("PYTHONPATH=src python scripts/audit_dambreak_gate_detection_power.py"
                        f" --n {n}"),
            "grid": n,
            "grids": list(GRIDS),
            "tolerance": TOLERANCE,
            "extremum_floor_over_h0": EXTREMUM_FRACTION,
            "translations_rel": list(TRANSLATIONS),
            "amplitudes_rel": list(AMPLITUDES),
            "extremum_multiples": list(EXTREMUM_MULTIPLES),
            "boundary_multiple": BOUNDARY_MULTIPLE,
            "reproduction_rel_tol": REPRO_REL_TOL,
            "wall_time_s": wall,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scope": ("depth-based graded criteria only; the x_f(t) series is carried "
                      "from the accepted-grid baseline because an end-of-run depth "
                      "perturbation cannot change it"),
        },
        "negative_control_ladder": ladder,
        "reproduction": reproduction,
        "battery": battery,
        "detection_thresholds": thresholds,
        "verdict": {
            "gate_can_fail": all(any(not item["pass"] for item in rows[0]["graded"].values())
                                 for rows in ladder.values()),
            "artifact_reproduced": reproduction["all_reproduced"],
            "translation_detected_at_5pct": {
                k: bool(thresholds[k].get(_TARGET_CRIT[k], {}).get("injected", 1e9) <= 0.05)
                for k in thresholds},
            "monotone_effective_threshold_floor_multiples": {
                f"{case_key}@{crit.split('@')[-1]}": float(thr[crit]["injected"])
                for case_key, thr in thresholds.items()
                for crit in thr if crit.startswith("monotone(no_extrema)")},
            "boundary_blind_spot": {k: battery[k]["boundary_probe"]["detected"]
                                    for k in battery},
        },
    }
    payload["verdict"]["all_sound"] = bool(
        payload["verdict"]["gate_can_fail"]
        and payload["verdict"]["artifact_reproduced"]
        and all(payload["verdict"]["translation_detected_at_5pct"].values())
        and all(payload["verdict"]["monotone_effective_threshold_floor_multiples"].values()))
    return payload


_TARGET_CRIT = {
    "stoker-wet": "shock_position<5%",
    "ritter-dry": "front_fan<5%",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=GRID,
                        help="accepted grid for the perturbation battery (default 1600)")
    parser.add_argument("--out-dir", default=ARTIFACT_DIR)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    payload = build(n=args.n, fig=not args.no_figure)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "detection_power.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    if not args.no_figure:
        _figure(payload, os.path.join(args.out_dir, "detection_power.png"))
    print(json.dumps(payload["verdict"], indent=1))
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())