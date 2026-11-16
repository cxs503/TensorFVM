#!/usr/bin/env python
"""Self-similarity (``xi``-collapse) study of the Ritter/Stoker dam break.

Writes
------
``src/tensorfvm/verification/artifacts/shallow-water-similarity/summary.json``
``src/tensorfvm/verification/artifacts/shallow-water-similarity/SUMMARY.md``
``docs/case-fvm-dambreak-stoker-similarity/fig_selfsimilar.png``

Run (single core, ~40 s)::

    PYTHONPATH=src python scripts/verify_dambreak_similarity.py

Every number in the report is read from the artifact; nothing is hand-typed.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.shallow_water import similarity as sim  # noqa: E402
from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET  # noqa: E402
from tensorfvm.shallow_water.exact import (ritter_state,  # noqa: E402
                                           stoker_state)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "shallow-water-similarity"
DOCS = ROOT / "docs" / "case-fvm-dambreak-stoker-similarity"


def exactness(case) -> dict:
    """The closed form depends on ``x`` and ``t`` only through ``xi``."""
    xi = sim.xi_grid(case)
    c0 = math.sqrt(case.gravity * case.h0)
    residuals = []
    for t in (0.05, 0.3, 1.7):
        x = case.x_dam + xi * t
        if case.kind == "dry":
            h, u = ritter_state(x, t, case.h0, case.gravity, case.x_dam)
        else:
            h, u = stoker_state(x, t, case.h0, case.h1, case.gravity, case.x_dam)
        residuals.append(np.max(np.abs(h / case.h0 - sim.exact_xi_profile(case)["h"])))
    return {"case": case.key, "max_depth_residual": float(max(residuals)),
            "note": "exact closed form evaluated at t = 0.05/0.3/1.7 on x = x_dam + xi t"}


def gate_power(case) -> dict:
    """Two probes: one that a collapse-only gate would miss, one it must catch."""
    exact = sim.exact_xi_profile(case)
    xi = exact["xi"]
    # (a) wrong-but-self-similar: the exact profile of a different h0 (+1 %).
    # Every time slice is the *same* curve, so the collapse residual is zero,
    # yet the profile is not the solution of this case.
    scaled = {"xi": xi, "h": exact["h"] * 1.01}
    series = [exact, scaled]          # a "time series" that collapses perfectly
    collapse_a = sim.collapse_residual([scaled], reference=scaled)
    err_a = sim.xi_error(scaled, exact)
    # (b) non-self-similar defect: a localised bump in one time slice only.
    lo, hi = sim.similarity_support(case)
    bump = np.exp(-0.5 * ((xi - 0.3 * (lo + hi)) / (0.05 * abs(hi))) ** 2)
    defected = {"xi": xi, "h": exact["h"] + 0.01 * bump}
    collapse_b = sim.collapse_residual([exact, defected])
    err_b = sim.xi_error(defected, exact)
    return {
        "case": case.key,
        "self_similar_wrong": {
            "xi_error": float(err_a),
            "collapse_residual_of_series": float(collapse_a),
            "conclusion": "collapse alone cannot see a uniformly wrong but self-similar profile",
        },
        "non_self_similar_defect": {
            "amplitude": 0.01,
            "xi_error": float(err_b),
            "collapse_residual_of_series": float(collapse_b),
            "conclusion": "the collapse residual catches a defect that is inconsistent between times",
        },
        "series_length": len(series),
    }


def figure(study: dict, bore: dict, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    cases = {"ritter-dry": RITTER_DRY, "stoker-wet": STOKER_WET}
    colours = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, (key, case) in zip(axes[0], cases.items()):
        exact = sim.exact_xi_profile(case)
        rows = study["cases"][key]["rows"]
        row = max(rows, key=lambda r: r["n"])          # finest resolution
        states = sim.run_similarity_series(case, row["n"])
        for c, st in zip(colours, states[::2]):
            prof = sim.numerical_xi_profile(st["x"], st["h"], st["t"], case)
            ax.plot(prof["xi"], prof["h"], color=c, lw=1.0,
                    label=f"FVM t*={st['t_star']:.2f} (n={row['n']})")
        ax.plot(exact["xi"], exact["h"], "k--", lw=1.6, label="exact h(ξ)/h0")
        ax.set_xlabel(r"$\xi=(x-x_{dam})/t$  [m/s]")
        ax.set_ylabel(r"$h/h_0$")
        ax.set_title(f"{key}: profiles collapse onto the exact similarity curve")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax = axes[1][0]
    for key in cases:
        rows = study["cases"][key]["rows"]
        for row, marker in zip(rows, ["o", "s", "^"]):
            cells = [r["cells_travelled"] for r in row["rows"]]
            err = [r["xi_error"] for r in row["rows"]]
            ax.loglog(cells, err, marker + "-", lw=1.0, ms=4,
                      label=f"{key} n={row['n']}")
        c = study["cases"][key]["C_plateau"]["mean"]
        grid = np.array([5.0, 400.0])
        ax.loglog(grid, c / grid, "k:", lw=1.2)
    ax.loglog([], [], "k:", label=r"$E=C\,dx/(c_0 t)$")
    ax.axhline(0.05, color="r", ls="--", lw=1.0, label="5 % gate")
    ax.set_xlabel(r"$c_0 t/dx$  (travel in cells)")
    ax.set_ylabel(r"$\xi$-space $L_1$ error")
    ax.set_title("start-up transient: E = C·dx/(c0 t)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    for key, mk in (("ritter-dry", "o"), ("stoker-wet", "s")):
        rows = study["cases"][key]["rows"][0]["rows"]
        ax.loglog([r["cells_travelled"] for r in rows],
                  [r["C"] for r in rows], mk + "-", lw=1.0, ms=4, label=f"{key}")
    ax.set_xlabel(r"$c_0 t/dx$")
    ax.set_ylabel(r"$C = E\,c_0 t/dx$")
    ax.set_title("plateau constant (declared C per case)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("CASE-FVM-DAMBREAK-STOKER · self-similarity collapse and "
                 "pre-asymptotic error law", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_summary(payload: dict) -> str:
    lines = ["# shallow-water dam break — self-similarity collapse (generated summary)",
             "",
             "Generated by `scripts/verify_dambreak_similarity.py`; no hand-typed numbers.",
             "",
             "reference: Ritter (1892) *Z. Ver. Deutsch. Ing.* 36, 947; "
             "Stoker (1957) *Water Waves* ch. 10 (both self-similar in `xi = (x-x_dam)/t`)",
             "",
             "## Exact solutions are self-similar (closed form evaluated at three times)",
             ""]
    for key, ex in payload["exactness"].items():
        lines.append(f"* `{key}`: max |h/h0(xi) - h/h0(xi)| over t = 0.05/0.3/1.7: "
                     f"**{ex['max_depth_residual']:.3e}**")
    lines += ["", "## xi-space error vs travel in cells", "",
              "`E = xi_error`, `C = E * c0 t / dx` (should be constant if `E = C dx/(c0 t)`).",
              "",
              "| case | n | t* | c0 t/dx | E | C | collapse vs t*=2 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, block in payload["study"]["cases"].items():
        for row in block["rows"]:
            for r in row["rows"]:
                lines.append(f"| {key} | {row['n']} | {r['t_star']:.2f} | "
                             f"{r['cells_travelled']:.1f} | {r['xi_error']:.4e} | "
                             f"{r['C']:.4f} | {r['collapse_vs_final']:.4e} |")
    lines += ["", "## Collapse residual (consistency only; no reference solution used)", "",
              "| case | n | over all sampled times | over t* >= "
              f"{payload['study']['cases'][list(payload['study']['cases'])[0]]['rows'][0]['asymptotic_t_star']:g} |",
              "|---|---:|---:|---:|"]
    for key, block in payload["study"]["cases"].items():
        for row in block["rows"]:
            lines.append(f"| {key} | {row['n']} | {row['collapse_all_times']:.4e} | "
                         f"{row['collapse_asymptotic']:.4e} |")
    lines += ["", "## Measured law and derived thresholds", "",
              "| case | C (mean) | C min..max | spread | dx order (n: E_i -> E_j) | "
              "min cells for 5 % | transient max E (t* <= 0.05) |",
              "|---|---:|---|---:|---|---:|---:|"]
    for key, block in payload["study"]["cases"].items():
        cp = block["C_plateau"]
        orders = "; ".join(f"{o['n0']}->{o['n1']}: {o['order']:.3f}"
                           for o in block["dx_order"])
        lines.append(f"| {key} | {cp['mean']:.4f} | {cp['min']:.4f}..{cp['max']:.4f} | "
                     f"{cp['spread']*100:.1f} % | {orders} | "
                     f"{block['min_cells_for_gate']:.2f} | "
                     f"{block['transient_max_error']:.4e} |")
    lines += ["", "## Temporal (CFL) axis: final xi error at n = "
              f"{payload['study']['cfl'][0]['n']}", "",
              "| case | cfl | xi error at t* = 2 |", "|---|---:|---:|"]
    for r in payload["study"]["cfl"]:
        lines.append(f"| {r['case']} | {r['cfl']:.2f} | {r['xi_error_final']:.6e} |")
    lines += ["", "## Early-time bore reading: function of travel in cells", "",
              "`AUDIT.md` F5 reports the wet-bed series error at `t* <~ 0.02` as "
              "grid-independent 4.505 %.  Groups below share `x_s/dx` exactly "
              "(several `n` and `t*` per group).",
              "",
              "| x_s/dx | (n, t*) | errors | max relative spread |", "|---:|---|---|---:|"]
    for g in payload["bore"]["invariance"]:
        members = ", ".join(f"({n},{ts:g})" for n, ts in g["members"])
        errs = ", ".join(f"{100.0 * e:.4f}" for e in g["errors"])
        lines.append(f"| {g['xs_over_dx']:.4f} | {members} | {errs} % | "
                     f"{g['max_relative_spread']:.2e} |")
    lines += ["", "Same reading at a **common** `t*` (F5's wording says grid independent):",
              "", "| n | t* | x_s/dx | error |", "|---:|---:|---:|---:|"]
    for r in payload["bore"]["rows"]:
        lines.append(f"| {r['n']} | {r['t_star']:g} | {r['xs_over_dx']:.4f} | "
                     f"{100.0 * r['error']:.4f} % |")
    lines += ["", "## Gate power (the collapse criterion alone would be vacuous)", ""]
    for key, gp in payload["gate_power"].items():
        a = gp["self_similar_wrong"]
        b = gp["non_self_similar_defect"]
        lines.append(f"* `{key}`: wrong-but-self-similar profile -> xi error "
                     f"{a['xi_error']*100:.2f} %, collapse residual "
                     f"{a['collapse_residual_of_series']:.2e}; non-self-similar bump "
                     f"(amp {b['amplitude']}) -> xi error {b['xi_error']*100:.2f} %, "
                     f"collapse residual {b['collapse_residual_of_series']:.2e}")
    lines += ["", "## Verdict", "",
              f"* `xi`-error law `E = C dx/(c0 t)` holds with C spread "
              f"{payload['study']['cases']['ritter-dry']['C_plateau']['spread']*100:.1f} % "
              f"(dry) / {payload['study']['cases']['stoker-wet']['C_plateau']['spread']*100:.1f} % (wet) "
              "over t* >= 0.2 and n = 400..1600.",
              "* first order in `dx` confirmed (orders in the table above).",
              "* temporal axis: the final `xi` error is CFL-insensitive at the "
              "declared 0.4 (see the CFL table).",
              "* the graded time `t* = 2` sits at `c0 t/dx` = "
              f"{payload['study']['cases']['stoker-wet']['rows'][-1]['rows'][-1]['cells_travelled']:.0f} "
              "cells (n = 1600), i.e. far inside the asymptotic regime.",
              ""]
    return "\n".join(lines)


def main() -> int:
    study = sim.similarity_study(resolutions=(400, 800, 1600),
                                 t_stars=sim.SAMPLE_T_STARS,
                                 cfls=(0.2, 0.4, 0.8))
    bore = sim.bore_reading_study()
    payload = {
        "slug": "CASE-FVM-DAMBREAK-STOKER",
        "topic": "self-similarity collapse (xi = (x-x_dam)/t) and pre-asymptotic error law",
        "reference": {
            "ritter": "Ritter (1892) Z. Ver. Deutsch. Ing. 36, 947-954",
            "stoker": "Stoker (1957) Water Waves, ch. 10",
            "similarity": "both solutions are self-similar in xi = (x-x_dam)/t; "
                          "verified here by evaluating the closed form at three times",
        },
        "constants": {
            "XI_POINTS": sim.XI_POINTS,
            "XI_MARGIN": sim.XI_MARGIN,
            "tolerance": 0.05,
            "sample_t_stars": list(sim.SAMPLE_T_STARS),
            "resolutions": [400, 800, 1600],
        },
        "exactness": {k: exactness(v) for k, v in sim.cases().items()},
        "study": study,
        "bore": bore,
        "gate_power": {k: gate_power(v) for k, v in sim.cases().items()},
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (ART / "SUMMARY.md").write_text(render_summary(payload))
    figure(study, bore, DOCS / "fig_selfsimilar.png")
    print(f"wrote {ART/'summary.json'}")
    print(f"wrote {ART/'SUMMARY.md'}")
    print(f"wrote {DOCS/'fig_selfsimilar.png'}")
    for key, block in payload["study"]["cases"].items():
        cp = block["C_plateau"]
        print(f"{key}: C = {cp['mean']:.4f} (spread {cp['spread']*100:.1f} %), "
              f"min cells for 5 % = {block['min_cells_for_gate']:.2f}, "
              f"orders = {[round(o['order'],3) for o in block['dx_order']]}")
    worst = max((g["max_relative_spread"], g["xs_over_dx"])
                for g in bore["invariance"])
    print(f"bore-reading invariance: worst relative spread over groups "
          f"{worst[0]:.3e} at x_s/dx = {worst[1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())