"""Reduce the NWT evidence JSON into one acceptance table (no hand-copied digits)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ART = Path("docs/wave-tank-td/artifacts")


def load(name):
    p = ART / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def main() -> int:
    wave = load("wave_main")
    ctrl = load("control_no_beach")
    pc = load("prop_coarse")
    pf = load("prop_fine")
    geo = load("geometry_gate")
    rao = load("rao_coarse")

    rows = []
    crit = {}

    def add(name, measured, threshold, ok, note=""):
        rows.append((name, measured, threshold, "PASS" if ok else "FAIL", note))
        crit[name] = {"measured": measured, "threshold": threshold,
                      "pass": bool(ok), "note": note}

    if geo:
        add("geometry_gate Cb (Wigley)", geo["wigley"]["Cb_abs_error"], 5e-3,
            geo["passed"], "Cb / L/B / B/T / nabla pre-gate")
    if wave:
        add("wave-height error (max over 7 probes)",
            round(100 * wave["amp_error_max"], 3), 3.0,
            wave["criteria"]["wave_height_error_le_3pct"],
            f"{wave['config']['nx']}x{wave['config']['ns']}, "
            f"{wave['steps']} steps, {wave['config']['length']} m tank")
        add("dispersion k error (7-probe fit vs w^2=gk tanh(kh))",
            round(100 * wave["k_error"], 3), 1.0,
            wave["criteria"]["dispersion_k_error_le_1pct"],
            f"k_fit={wave['k_three_probe']:.5f} vs {wave['k_reference']:.5f}")
        add("reflection C_R (Goda two-point)",
            round(100 * wave["goda"]["C_R"], 3), 5.0,
            wave["criteria"]["reflection_C_R_le_5pct"],
            "beach = 1.5 lambda")
    if ctrl:
        add("negative control C_R (beach removed)",
            round(100 * ctrl["goda"]["C_R"], 2), ">= 20",
            ctrl["criteria"]["C_R_ge_20pct_required_for_control"],
            "closed tank, total reflection")
    if pc and pf:
        ec = 100 * pc["amp_error_max"]
        ef = 100 * pf["amp_error_max"]
        p = math.log(ec / ef) / math.log(2.0) if ef > 0 else float("nan")
        gci = 1.25 * ef / (2.0 ** p - 1.0) if p == p and p > 0 else float("nan")
        add("propagation amplitude error, fine grid", round(ef, 3), 3.0,
            pf["criteria"]["wave_height_error_le_3pct"],
            f"coarse {ec:.2f} % -> apparent order {p:.2f}, GCI(fine) {gci:.2f} %")
        add("propagation dispersion k error, fine grid",
            round(100 * pf["k_error"], 3), 1.0,
            pf["criteria"]["dispersion_k_error_le_1pct"], "no wavemaker")
        crit["gci_propagation_amplitude"] = {
            "coarse_error_pct": ec, "fine_error_pct": ef,
            "apparent_order": p, "gci_fine_pct": gci,
            "note": "two grids only; time step refined together with the grid"}
    if rao:
        crit["rao"] = {
            "rao_heave": rao["rao_heave"], "rao_pitch": rao["rao_pitch"],
            "heave_harmonic_residual": rao["heave_residual"],
            "pitch_harmonic_residual": rao["pitch_residual"],
            "pass": False,
            "note": "motion not converged to a harmonic response; T=0.8 s sits "
                    "next to the pitch resonance of the 0.6x0.2 m section "
                    "(T_n5 ~ 0.97 s) where the potential damping is tiny, so the "
                    "response saturates against the waterplane-loss nonlinearity",
        }
        crit["added_resistance"] = {
            "near_field": rao["added_resistance"],
            "control_surface": rao["added_resistance_control_surface"],
            "pass": False,
            "note": "control-surface balance is not steady in the same run "
                    "(residual-dominated), so no cross-check is claimed",
        }
    out = {"criteria": crit, "table": rows,
           "wave": wave["wave"] if wave else None,
           "config": wave["config"] if wave else None,
           "div_residual_max": wave["div_residual_max"] if wave else None,
           "volume_drift_rel": wave["volume_drift_rel"] if wave else None}
    with open(ART / "acceptance.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("| criterion | measured | threshold | verdict | note |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    print()
    for k in ("gci_propagation_amplitude", "rao", "added_resistance"):
        if k in crit:
            print(k, json.dumps(crit[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())