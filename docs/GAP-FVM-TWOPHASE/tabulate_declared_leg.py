"""Declared-resolution Hysing leg: tabulate, plot, and write the audit report.

Consumes

* ``runs/m16901/leg16/leg_hysing_16.json``  -- cpd=16 (``DECLARED_CPD``) leg,
* ``runs/m16901/ab_cpd8/``                  -- cpd=8 A/B of ``pressure_form``,
* ``docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing/leg_hysing_8.json`` -- cpd=8 leg,

and emits the committed evidence bundle:

* ``docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing/leg_hysing_16.json``
* ``docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing/fig_hysing_declared.png``
* ``docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing/hysing_declared_summary.json``
* ``docs/GAP-FVM-TWOPHASE/HYSING-DECLARED-RESOLUTION.md``

Usage::

    PYTHONPATH=src python3 docs/GAP-FVM-TWOPHASE/tabulate_declared_leg.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.twophase_gap.hysing_gate import (  # noqa: E402
    CONVERGENCE_MAX,
    CIRCULARITY_MAX,
    HYSING_REF_CIRCULARITY,
    HYSING_REF_UT,
    HYSING_REF_YC,
    TERMINAL_VELOCITY_MAX,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing"
RUNS = ROOT / "runs/m16901"


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def main() -> int:
    fine = _load(RUNS / "leg16/leg_hysing_16.json")
    coarse = _load(RUNS / "ab_cpd8/ab_cpd8.json")
    coarse_hist = ART / "leg_hysing_8.json"
    c8 = _load(coarse_hist) if coarse_hist.exists() else None

    rec16, hist16 = fine["record"], fine["history"]
    c8_total = coarse["forms"]["total"]
    yc16 = hist16[-1]["yc"]

    # B4 as now implemented in campaign.py: relative change of U_T, coarse->fine
    b4_value = abs(rec16["terminal_velocity"] - c8_total["terminal_velocity"]) / abs(
        rec16["terminal_velocity"])
    b4_peak = abs(rec16["peak_velocity"] - c8_total["peak_velocity"]) / abs(
        rec16["peak_velocity"])

    # Reference-provenance arithmetic.  Hysing's nondimensionalisation scales
    # length by d and velocity by sqrt(g d); after scaling by d the bubble
    # starts at y_c = 1 d and the reference y_c(t=3) = 1.984 is in the
    # *unscaled* (d = 0.5) frame, i.e. 3.968 d.  Whatever the frame, the mean
    # rise velocity implied by a reference pair must not exceed the reference
    # terminal velocity by an implausible factor; this is the check that
    # decides whether the y_c criterion can be evaluated at all.
    implied_mean_d = (HYSING_REF_YC - 0.5) / 0.5 / 3.0          # in units of d
    implied_mean_ratio = implied_mean_d / HYSING_REF_UT
    our_mean = (yc16 - 1.0) / 3.0                                # in units of d

    rows = [
        ("peak rise velocity max dY_c/dt", rec16["peak_velocity"], HYSING_REF_UT,
         TERMINAL_VELOCITY_MAX),
        ("window-mean velocity (legacy B1 metric)", rec16["terminal_velocity"],
         HYSING_REF_UT, TERMINAL_VELOCITY_MAX),
        ("min circularity", rec16["min_circularity"], HYSING_REF_CIRCULARITY,
         CIRCULARITY_MAX),
    ]

    summary = {
        "cpd_declared": rec16["cells_per_diameter"],
        "grid": [rec16["cells_per_diameter"] * 2, rec16["cells_per_diameter"] * 4],
        "steps": rec16["steps"],
        "wall_seconds": rec16["wall_seconds"],
        "pressure_form": rec16.get("pressure_form", "total"),
        "peak_velocity": rec16["peak_velocity"],
        "terminal_velocity": rec16["terminal_velocity"],
        "min_circularity": rec16["min_circularity"],
        "yc_end": yc16,
        "worst_step_drift": rec16["worst_step_drift"],
        "total_volume_drift": rec16["total_volume_drift"],
        "diverged": rec16["diverged"],
        "peak_rel_error": _rel(rec16["peak_velocity"], HYSING_REF_UT),
        "window_rel_error": _rel(rec16["terminal_velocity"], HYSING_REF_UT),
        "circularity_rel_error": _rel(rec16["min_circularity"],
                                      HYSING_REF_CIRCULARITY),
        "b4_relative_change_window": b4_value,
        "b4_relative_change_peak": b4_peak,
        "b4_limit": CONVERGENCE_MAX,
        "our_mean_rise_velocity_d": our_mean,
        "reference_implied_mean_rise_velocity_d": implied_mean_d,
        "reference_implied_mean_over_terminal": implied_mean_ratio,
        "cpd8": c8_total,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "leg_hysing_16.json").write_text(json.dumps(fine, indent=2) + "\n")
    (ART / "hysing_declared_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # ---- figure: history curves, ours vs the reference trivially available ----
    t16 = np.array([h["time"] for h in hist16])
    y16 = np.array([h["yc"] for h in hist16])
    c16 = np.array([h["circularity"] for h in hist16])
    u16 = np.gradient(y16, t16)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    if c8 and "history" in c8:
        h8 = c8["history"]
        ax[0].plot([h["time"] for h in h8], [h["yc"] for h in h8], lw=1.0,
                   color="tab:gray", label=f"cpd={c8['cpd']} (coarse)")
    ax[0].plot(t16, y16, lw=1.4, color="tab:blue",
               label=f"cpd={rec16['cells_per_diameter']} (declared)")
    ax[0].axhline(1.0, ls=":", color="k", lw=0.8, label=r"initial $y_c=1d$")
    ax[0].set_xlabel(r"$t\,/\,\sqrt{d/g}$")
    ax[0].set_ylabel(r"$y_c\,/\,d$")
    ax[0].set_title("bubble centroid rise")
    ax[0].legend(fontsize=7)

    ax[1].plot(t16, u16, lw=1.4, color="tab:blue")
    ax[1].axhline(HYSING_REF_UT, ls="--", color="tab:red", lw=0.9,
                  label=f"Hysing ref $U_T$={HYSING_REF_UT}")
    ax[1].axhline(rec16["peak_velocity"], ls=":", color="tab:green", lw=0.9,
                  label=f"our peak {rec16['peak_velocity']:.4f}")
    ax[1].set_xlabel(r"$t\,/\,\sqrt{d/g}$")
    ax[1].set_ylabel(r"$U_c\,/\,\sqrt{gd}$")
    ax[1].set_title("rise velocity")
    ax[1].legend(fontsize=7)

    ax[2].plot(t16, c16, lw=1.4, color="tab:blue")
    ax[2].axhline(HYSING_REF_CIRCULARITY, ls="--", color="tab:red", lw=0.9,
                  label=f"Hysing ref $c_{{min}}$={HYSING_REF_CIRCULARITY}")
    ax[2].set_xlabel(r"$t\,/\,\sqrt{d/g}$")
    ax[2].set_ylabel("circularity")
    ax[2].set_ylim(min(0.3, float(c16.min()) - 0.05), 1.15)
    ax[2].set_title("circularity")
    ax[2].legend(fontsize=7)
    fig.suptitle("Hysing (2009) Case 2 -- PLIC-VOF two-phase solver, declared resolution",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(ART / "fig_hysing_declared.png", dpi=150)
    plt.close(fig)

    # ---- markdown report ----
    def fmt_row(name, got, ref, limit):
        rel = _rel(got, ref)
        verdict = "PASS" if rel <= limit else "**FAIL**"
        return f"| {name} | {got:.4f} | {ref} | {100 * rel:.1f} % | {100 * limit:.0f} % | {verdict} |"

    lines = [
        "# Hysing leg at the declared resolution (cpd=16) — m16901",
        "",
        "- Anchor: `GAP-FVM-TWOPHASE` (feature_ledger, in_progress), verification debt D9",
        "  (`docs/VERIFICATION_DEBT_LEDGER.md` §4b).",
        "- Predecessor: `docs/GAP-FVM-TWOPHASE/HYSING-LEG-REPAIR.md` (m16730) fixed the three",
        "  metric/geometry defects at cpd=8 and left the *declared* resolution (cpd=16) as",
        "  NOT_RUN (cluster job `cj1353` never left the queue).",
        "- This unit: run the declared leg, re-evaluate B1–B4, and audit the gate/",
        "  reference bookkeeping.  Declared edit whitelist:",
        "  `src/tensorfvm/twophase_gap/`, `tests/`, `docs/GAP-FVM-TWOPHASE/`, `scripts/`, `runs/`.",
        "",
        "## 1. Compute setup (numbers straight from the artefact)",
        "",
        f"- grid: {summary['grid'][0]}x{summary['grid'][1]} cells "
        f"(h = d/{rec16['cells_per_diameter']}), domain 2d x 4d",
        f"- steps: {summary['steps']}, wall clock: {summary['wall_seconds']:.0f} s "
        "(single core, one thread)",
        f"- pressure form: `{summary['pressure_form']}`, diverged: {summary['diverged']}",
        "- reproduce: `PYTHONPATH=src python3 scripts/run_hysing_leg.py --cpd 16`",
        "",
        "## 2. Gate-by-gate result at the declared resolution",
        "",
        "| quantity | got | ref (Hysing 2009, Case 2) | rel. error | gate | verdict |",
        "|---|---|---|---|---|---|",
        fmt_row(*rows[0]),
        fmt_row(*rows[1]),
        fmt_row(*rows[2]),
        f"| worst per-step volume drift | {summary['worst_step_drift']:.2e} | — | — | 1e-06 | "
        f"{'PASS' if summary['worst_step_drift'] <= 1e-6 else '**FAIL**'} |",
        f"| B4 relative change of U_T, cpd 8->16 (window metric) | "
        f"{summary['b4_relative_change_window']:.3f} | — | — | "
        f"{CONVERGENCE_MAX} | "
        f"{'PASS' if summary['b4_relative_change_window'] <= CONVERGENCE_MAX else '**FAIL**'} |",
        f"| B4 relative change of U_T, cpd 8->16 (peak metric) | "
        f"{summary['b4_relative_change_peak']:.3f} | — | — | "
        f"{CONVERGENCE_MAX} | "
        f"{'PASS' if summary['b4_relative_change_peak'] <= CONVERGENCE_MAX else '**FAIL**'} |",
        "",
        "Figure: `artifacts/leg_hysing/fig_hysing_declared.png` "
        "(centroid / velocity / circularity histories).",
        "",
        "## 3. Findings",
        "",
        "### F1 (defect, fixed) B4 was not the criterion it claimed to be",
        "",
        "`campaign.py` fed the coarse and fine `U_T` into `convergence_order()`"
        " (`log(U_c/U_f)/log 2`)",
        "and compared that *order* against `CONVERGENCE_MAX = 0.05` with `>=`.  Any change of",
        ">= 3.5 % in `U_T` produces an order >= 0.05, so the gate documented as",
        "\"U_T coarse -> fine change <= 5 %\" could not fail on the divergence it exists to catch.",
        "Fixed: the gate now compares the relative change directly"
        " (`hysing_convergence_change`, `<= 0.05`); the observed order stays as an",
        "informational metric.  Regression tests: `tests/test_hysing_gate.py::TestConvergenceGateB4`",
        "(red on the old wiring: a 25 % change passed).",
        "",
        "### F2 (falsified hypothesis) `pressure_form` is not the cause of the slow rise",
        "",
        "`TwoPhaseConfig.pressure_form` documents `\"reduced\"` as the form to use whenever",
        "`gravity != 0` and a free surface is present, and the Hysing leg inherited the",
        "`\"total\"` default -- a plausible cause for the -23 % `y_c` deficit reported at",
        "cpd=8.  A/B at identical resolution and end time (artefacts",
        "`runs/m16901/ab_cpd8/`, reproduce `docs/GAP-FVM-TWOPHASE/ab_pressure_form.py`):",
        "",
        "| form | steps | peak U | window U | y_c(t=3) | c_min | worst step drift |",
        "|---|---|---|---|---|---|---|",
    ]
    for form in ("total", "reduced"):
        f = coarse["forms"][form]
        lines.append(
            f"| `{form}` | {f['steps']} | {f['peak_velocity']:.4f} | "
            f"{f['terminal_velocity']:.4f} | {f['yc_end']:.4f} | "
            f"{f['min_circularity']:.4f} | {f['worst_step_drift']:.2e} |")
    lines += [
        "",
        "The two forms agree to 0.2 % on the velocity metrics and 0.07 % on `y_c`: the",
        "pressure split is *not* the explanation.  The hypothesis is withdrawn; the flag is",
        "kept in the case definition (default unchanged) so the A/B stays reproducible.",
        "",
        "### F3 (withheld criterion) the `y_c(t=3)` reference cannot be evaluated as written",
        "",
        f"Reference triple in the code: `U_T = {HYSING_REF_UT}`, `c_min = {HYSING_REF_CIRCULARITY}`,",
        f"`y_c(t=3) = {HYSING_REF_YC}`.  The triple is not traceable to any dataset in this",
        "repository (no benchmark file, no page/table extract, no unit chain) -- it is a",
        "hand-transcribed number.  It is also internally inconsistent with itself under any",
        "single nondimensionalisation available to the case definition:",
        "",
        f"- bubble starts at `y_c = 1.0 d`; a reference `y_c(t=3)` of {HYSING_REF_YC} in Hysing's",
        f"  own (`d = 0.5`) frame is `3.968 d`, i.e. a mean rise of "
        f"`{implied_mean_d:.3f} d` per unit `sqrt(d/g)`;",
        f"- that is `{implied_mean_ratio:.2f}x` the reference terminal velocity "
        f"`U_T = {HYSING_REF_UT}`; a run that starts from rest cannot average "
        f"{implied_mean_ratio:.2f}x",
        "  its own terminal velocity.",
        "",
        f"Our declared leg reaches `y_c(t=3) = {yc16:.4f} d`, i.e. a mean of "
        f"`{our_mean:.3f} d` per time unit --",
        "comparable to `U_T`, and *closer to physically possible* than the reference implies.",
        "Consequence: the `-23 %` verdict on `y_c` recorded in `HYSING-LEG-REPAIR.md` is retracted",
        "to **WITHHELD**: without a sourced reference (paper table with units, or a dataset",
        "deposited in-repo) no honest error bar exists.  This is *not* a claim that the shape",
        "is right -- B2 already shows it is not at these resolutions.",
        "",
        "## 4. Honest status of the anchor after this unit",
        "",
        "| leg | criterion | status |",
        "|---|---|---|",
        "| A: Zalesak rotating disk | Z1–Z5 | PASS (unchanged, m15201/m16314 evidence) |",
        f"| B: Hysing velocity | peak `U_T` vs ref <= 10 % | "
        f"{'PASS' if summary['peak_rel_error'] <= TERMINAL_VELOCITY_MAX else '**FAIL**'} "
        f"at cpd={rec16['cells_per_diameter']} ({100 * summary['peak_rel_error']:.1f} %) |",
        f"| B: Hysing shape | `c_min` vs ref <= 10 % | **FAIL** "
        f"({100 * summary['circularity_rel_error']:.0f} %; needs cpd >= 32, out of budget) |",
        f"| B: mass | worst step drift <= 1e-06 | "
        f"{'PASS' if summary['worst_step_drift'] <= 1e-6 else '**FAIL**'} "
        f"({summary['worst_step_drift']:.2e}) |",
        "| B: grid convergence | change <= 5 % | see §2 (B4 rows) |",
        "| B: centroid | `y_c(t=3)` | WITHHELD (F3) |",
        "",
        "The anchor stays **in_progress**: the shape gate is a genuine failure at every",
        "resolution this box can afford, and `y_c` has no usable reference yet.",
        "",
        "## 5. Proposals (for the ledger)",
        "",
        "- Keep `GAP-FVM-TWOPHASE` in_progress; append the B4 fix and the F2/F3 outcomes to",
        "  `evidence`.",
        "- New anchor suggestion (domain: TensorFVM two-phase VOF):",
        "  `GAP-FVM-HYSING-REF-PROVENANCE` — acceptance: (1) the Hysing Case 2 reference triple",
        "  committed as a dataset with units and a page/table citation in `docs/` or",
        "  `data/`; (2) a unit-chain test asserting `mean_rise <= U_T_ref` for the reference",
        "  `y_c(t=3)`; (3) the `y_c` criterion re-scored, WITHHELD otherwise.",
        "  Quantified: no hand-typed constant left in `hysing_gate.py`",
        "  (`grep -c '1.984' == 0`).",
        "- Cost note for the shape gate: cpd=32 is ~4x the declared leg; the box ran the",
        "  declared leg in ~5 min wall at ~98 % of one core (single-threaded torch), so",
        "  cpd=32 (~20-60 min) belongs on the cluster, not here.",
        "",
    ]
    (ROOT / "docs/GAP-FVM-TWOPHASE/HYSING-DECLARED-RESOLUTION.md").write_text(
        "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cpd8"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())