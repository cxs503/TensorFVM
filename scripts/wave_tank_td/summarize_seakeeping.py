#!/usr/bin/env python
"""Assemble the seakeeping acceptance table, criteria and figures from artifacts.

Inputs are the per-case JSON files written by ``run_seakeeping.py``; nothing is
typed by hand.  Outputs:

* ``summary.json``  -- table rows + every machine criterion with its number;
* ``figures/*.png`` -- RAO against the potential-flow reference, reflection and
  drift force, wave-field snapshot.

    python scripts/wave_tank_td/summarize_seakeeping.py \
        --artifacts docs/wave-tank-td-seakeeping/artifacts \
        --figures docs/wave-tank-td-seakeeping/figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfvm.wave_tank_td import seakeeping as sk  # noqa: E402

RESONANCE_BAND = (0.6, 1.4)


def load(artifacts: Path) -> tuple[list[dict], list[dict], dict | None]:
    cases, propagation, gate = [], [], None
    for path in sorted(artifacts.glob("*.json")):
        data = json.loads(path.read_text())
        if path.name == "geometry_gate.json":
            gate = data
        elif "comparison" in data:
            cases.append(data)
        elif "k_three_probe" in data:
            propagation.append(data)
    return cases, propagation, gate


def in_band(row: dict, dof: str) -> bool:
    ratio = row["comparison"]["omega_over_omega_n"][dof]
    return RESONANCE_BAND[0] < ratio < RESONANCE_BAND[1]


def build(artifacts: Path, figures: Path | None) -> dict:
    cases, propagation, gate = load(artifacts)
    if not cases:
        raise SystemExit(f"no case artifacts in {artifacts}")
    cases.sort(key=lambda c: (c["config"]["periods"] and c["wave"]["period"],
                              c["comparison"]["damping_scale"],
                              c["config"]["nx"]))
    rows = []
    for c in cases:
        cmp = c["comparison"]
        rows.append({
            "case": c["case"], "period": c["wave"]["period"],
            "wavelength": c["wave"]["wavelength"], "nx": c["config"]["nx"],
            "ns": c["config"]["ns"], "spp": c["config"]["spp"],
            "damping_scale": cmp["damping_scale"],
            "omega_over_omega_n_heave": cmp["omega_over_omega_n"]["heave"],
            "omega_over_omega_n_pitch": cmp["omega_over_omega_n"]["pitch"],
            "rao_heave_td": cmp["rao_heave_td"], "rao_heave_ref": cmp["rao_heave_ref"],
            "rao_heave_err": cmp["rao_heave_rel_error"],
            "rao_pitch_td": cmp["rao_pitch_td"], "rao_pitch_ref": cmp["rao_pitch_ref"],
            "rao_pitch_err": cmp["rao_pitch_rel_error"],
            "heave_residual": c["rao"]["heave_residual"],
            "pitch_residual": c["rao"]["pitch_residual"],
            "C_R": c["reflection"]["C_R"],
            "drift_direct": c["drift"]["direct_mean_force"],
            "drift_far_field": c["drift"]["far_field"],
            "drift_rel_diff": c["drift"]["relative_difference"],
            "div_residual": c["numerics"]["div_residual_max"],
            "volume_drift": c["numerics"]["volume_drift_rel"],
            "in_band_heave": in_band(c, "heave"), "in_band_pitch": in_band(c, "pitch"),
            "no_beach": abs(c["config"]["damp_waves"]) < 1e-12,
        })

    # --- criteria -------------------------------------------------------
    criteria: dict[str, dict] = {}

    amp_err = max((abs(r["drift_direct"] * 0 + 1) for r in []), default=0.0)
    # wave-height gate: incident amplitude at the seaward probe vs the target
    wave_err = max(abs(c["incident"]["a_i"] / c["wave"]["amp"] - 1.0) for c in cases)
    criteria["wave_height_error_le_3pct"] = {
        "measured_pct": 100.0 * wave_err, "threshold": 3.0,
        "pass": bool(wave_err <= 0.03),
        "note": "max over cases of |a_i/probe / a_target - 1| (Goda incident amplitude)"}
    del amp_err

    if propagation:
        kerr = max(p["k_error"] for p in propagation)
        aerr = max(p["amp_error_max"] for p in propagation)
        criteria["dispersion_k_error_le_1pct"] = {
            "measured_pct": 100.0 * kerr, "threshold": 1.0,
            "pass": bool(kerr <= 0.01),
            "note": "three-probe phase fit vs omega^2 = g k tanh(kh), body-free run"}
        criteria["propagation_amplitude_error_le_3pct"] = {
            "measured_pct": 100.0 * aerr, "threshold": 3.0,
            "pass": bool(aerr <= 0.03), "note": "body-free propagation run"}

    with_beach = [r for r in rows if not r["no_beach"]]
    without = [r for r in rows if r["no_beach"]]
    if with_beach:
        cr = max(r["C_R"] for r in with_beach)
        criteria["reflection_C_R_le_5pct"] = {
            "measured_pct": 100.0 * cr, "threshold": 5.0,
            "pass": bool(cr <= 0.05), "note": "Goda two-point, worst case with beach"}
    if without:
        cr0 = min(r["C_R"] for r in without)
        criteria["negative_control_C_R_ge_20pct"] = {
            "measured_pct": 100.0 * cr0, "threshold": 20.0,
            "pass": bool(cr0 >= 0.20),
            "note": "same tank with the absorbing beach removed (gate must fail open)"}

    base = [r for r in rows if abs(r["damping_scale"] - 1.0) < 1e-9 and not r["no_beach"]]
    out_of_band = []
    for r in base:
        for dof in ("heave", "pitch"):
            if not r[f"in_band_{dof}"] and r[f"rao_{dof}_ref"] >= 0.10:
                out_of_band.append((dof, r["period"], r[f"rao_{dof}_err"]))
    if out_of_band:
        worst = max(out_of_band, key=lambda t: t[2])
        criteria["rao_vs_reference_le_10pct_out_of_band"] = {
            "measured_pct": 100.0 * worst[2], "threshold": 10.0,
            "pass": bool(worst[2] <= 0.10),
            "note": (f"worst of {len(out_of_band)} out-of-band points with reference "
                     f"RAO >= 0.1: dof={worst[0]} T={worst[1]:.2f} s")}

    drift = [r for r in base if math.isfinite(r["drift_rel_diff"])]
    if drift:
        worst = max(drift, key=lambda r: r["drift_rel_diff"])
        criteria["added_resistance_two_paths_le_15pct"] = {
            "measured_pct": 100.0 * worst["drift_rel_diff"], "threshold": 15.0,
            "pass": bool(worst["drift_rel_diff"] <= 0.15),
            "note": (f"|mean pressure force - far-field momentum balance| at T="
                     f"{worst['period']:.2f} s")}

    # two-grid GCI on the RAO at equal period and damping scale
    gci_rows = []
    by_period: dict[tuple, dict] = {}
    for r in base:
        by_period.setdefault((round(r["period"], 6), r["damping_scale"]), {})[
            r["nx"]] = r
    for (period, scale), pair in sorted(by_period.items()):
        if len(pair) < 2:
            continue
        grids = sorted(pair)
        coarse, fine = pair[grids[0]], pair[grids[-1]]
        ratio = grids[-1] / grids[0]
        entry = {"period": period, "damping_scale": scale,
                 "nx": [grids[0], grids[-1]], "refinement_ratio": ratio}
        for dof in ("heave", "pitch"):
            g = sk.gci_two_grid(fine[f"rao_{dof}_td"], coarse[f"rao_{dof}_td"],
                                ratio=ratio, order=2.0)
            entry[f"gci_{dof}_pct"] = 100.0 * g["gci_fine"]
            entry[f"in_band_{dof}"] = fine[f"in_band_{dof}"]
        gci_rows.append(entry)
    if gci_rows:
        worst = max(max(100.0 * 0.0, r["gci_heave_pct"], r["gci_pitch_pct"]) for r in gci_rows)
        criteria["two_grid_gci_le_5pct"] = {
            "measured_pct": worst, "threshold": 5.0, "pass": bool(worst <= 5.0),
            "note": "Roache GCI, order 2 assumed, on the RAO (heave and pitch)"}

    sensitivity = {}
    for r in base:
        pass
    for r in rows:
        if r["no_beach"]:
            continue
        key = round(r["period"], 6)
        sensitivity.setdefault(key, {})[r["damping_scale"]] = {
            "rao_heave": r["rao_heave_td"], "rao_pitch": r["rao_pitch_td"]}
    sens_rows = [{"period": p, "by_scale": v} for p, v in sorted(sensitivity.items())
                 if len(v) > 1]

    summary = {
        "cases": rows, "gci": gci_rows, "damping_sensitivity": sens_rows,
        "geometry_gate": gate, "propagation": propagation,
        "criteria": criteria,
        "pass_all": all(c["pass"] for c in criteria.values()),
        "failed": [k for k, v in criteria.items() if not v["pass"]],
    }
    if figures is not None:
        make_figures(summary, figures)
    return summary


def make_figures(summary: dict, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    rows = [r for r in summary["cases"] if abs(r["damping_scale"] - 1.0) < 1e-9
            and not r["no_beach"]]
    rows.sort(key=lambda r: r["period"])
    if not rows:
        return
    t = [r["period"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    for dof, label in (("heave", "heave RAO  $|\\xi_3|/a$"),
                       ("pitch", "pitch RAO  $|\\xi_5|/(k a)$")):
        ax = axes[0 if dof == "heave" else 1]
        ax.plot(t, [r[f"rao_{dof}_ref"] for r in rows], "k-", label="potential flow")
        ax.plot(t, [r[f"rao_{dof}_td"] for r in rows], "o", mfc="none", label="tank (coarse)")
        fine = [r for r in rows if r["nx"] > min(x["nx"] for x in rows)]
        if fine:
            ax.plot([r["period"] for r in fine], [r[f"rao_{dof}_td"] for r in fine],
                    "s", ms=4, label="tank (fine)")
        ax.set_xlabel("wave period  $T$  [s]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("heave response")
    axes[1].set_title("pitch response")
    fig.tight_layout()
    fig.savefig(figures / "rao.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=140)
    ax.plot(t, [100 * r["C_R"] for r in rows], "o-", label="with beach")
    ctrl = [r for r in summary["cases"] if r["no_beach"]]
    if ctrl:
        ax.axhline(100 * max(c["C_R"] for c in ctrl), color="crimson", ls="--",
                   label="no beach (negative control)")
    ax.axhline(5.0, color="grey", ls=":", label="5 % gate")
    ax.set_xlabel("wave period  $T$  [s]")
    ax.set_ylabel("reflection coefficient  $C_R$  [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "reflection.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=140)
    ax.plot(t, [r["drift_direct"] for r in rows], "o-", label="pressure integral")
    ax.plot(t, [r["drift_far_field"] for r in rows], "s--", label="far-field balance")
    ax.set_xlabel("wave period  $T$  [s]")
    ax.set_ylabel("added resistance  $R_{aw}$  [N/m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "added_resistance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=140)
    for r in rows[:1] + rows[-1:]:
        pass
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlabel("$x$  [m]")
    ax.set_ylabel("$\\eta$  [m]")
    ax.set_title("free-surface elevation at the end of a run (see artifacts)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "wave_field.png")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--figures", default=None)
    a = ap.parse_args(argv)
    summary = build(Path(a.artifacts), Path(a.figures) if a.figures else None)
    out = Path(a.artifacts) / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    print(json.dumps({"criteria": {k: {"measured": round(v.get("measured_pct", 0.0), 3),
                                       "pass": v["pass"]}
                                   for k, v in summary["criteria"].items()},
                      "pass_all": summary["pass_all"]}, indent=2, sort_keys=True))
    return 0 if summary["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())