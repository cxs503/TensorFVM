#!/usr/bin/env python
"""Grade the ``CASE-FVM-DAMBREAK-WET`` ladder and render the report artefacts.

    PYTHONPATH=src python docs/CASE-FVM-DAMBREAK-WET/render_report.py

Reads ``artifacts/leg_*.json`` (written by :mod:`run_ladder`), grades every leg
against **two independent digitisations** of Martin & Moyce (1952) Fig. 3 on the
paper's own axes, writes ``artifacts/report_data.json`` (every number in the
report is injected from there), ``artifacts/figures/*.png`` and ``report.md``.

Criteria (declared before the ladder was run, ``run_ladder.GATE``):

* **C1** |X_sim - X_ref| / X_ref <= 8 % at *every* predeclared checkpoint
  ``T in {1.0, 1.5, 2.0, 2.5, 2.96}`` on the primary digitisation
  (``anchors_mm``, axes documented verbatim against the reference
  implementation);
* **C2** the same against the *second, independent* digitisation (the legacy
  TensorFVM table re-read on the paper's axes), which bounds the digitisation
  uncertainty of a 1952 figure;
* **C3** the checkpoint error must not grow with refinement within a scheme;
* **C4** global volume drift <= 1e-6;
* **C5** every graded sample finite (no divergent leg).
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.vof_phase0 import anchors_mm as A  # noqa: E402
from tensorfvm.vof_phase0.anchors import (  # noqa: E402
    MARTIN_MOYCE_FRONT_T as LEG_T,
    MARTIN_MOYCE_FRONT_X as LEG_X,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(ART, "figures")
A_W = 0.05715
G = 9.81
TOL = 0.08

#: second digitisation: the legacy table re-read as (T, X) -- see
#: ``tests/test_dambreak_mm_anchor.py::test_legacy_axis_is_a_defect_and_not_a_relabelling``
LEGACY_T = np.asarray(LEG_T, dtype=float)  # read as the paper time axis T
LEGACY_X = np.asarray(LEG_X, dtype=float)  # read as the paper ordinate X


def secondary_front(T: float) -> float:
    # only T >= 1 is a meaningful ordinate reading: the legacy table starts at
    # 0 (displacement), the paper ordinate starts at 1, so the two readings have
    # no common point below T = 1.
    ok = LEGACY_T >= 1.0
    if T < 1.0 or T > LEGACY_T[ok][-1]:
        return float("nan")
    return float(np.interp(T, LEGACY_T[ok], LEGACY_X[ok]))


def load_legs() -> list[dict]:
    legs = []
    for path in sorted(glob.glob(os.path.join(ART, "leg_*.json"))):
        with open(path, encoding="utf-8") as fh:
            legs.append(json.load(fh))
    return legs


def grade(legs: list[dict]) -> dict:
    rows = []
    for leg in legs:
        samples = []
        for s in leg["samples"]:
            mm = s["mm"]
            T = mm["T"]
            sec = secondary_front(T)
            samples.append(
                {
                    "T": T,
                    "t": mm["t"],
                    "X_sim": mm["X"],
                    "X_primary": mm["X_reference"],
                    "err_primary": mm["relative_error"],
                    "X_secondary": sec,
                    "err_secondary": (mm["X"] - sec) / sec if sec == sec else float("nan"),
                    "x_front": mm["x_front"],
                })
        rows.append(
            {
                "key": leg["key"],
                "scheme": leg["scheme"],
                "nx": leg["nx"],
                "ny": leg["ny"],
                "dx": leg["dx"],
                "cells_across_column": leg["cells_across_column"],
                "steps": leg["steps"],
                "wall_clock_s": leg["wall_clock_s"],
                "mass_drift_max": leg["mass_drift_max"],
                "samples": samples,
                "max_abs_err_primary": max(abs(s["err_primary"]) for s in samples),
                "max_abs_err_secondary": max(abs(s["err_secondary"]) for s in samples),
                "finite": all(s["X_sim"] == s["X_sim"] for s in samples),
            })

    def worst(sel, field):
        vals = [r[field] for r in rows if sel(r)]
        return max(vals) if vals else float("nan")

    up = lambda r: r["scheme"] == "upwind"  # noqa: E731
    l2 = lambda r: r["scheme"] == "limited2"  # noqa: E731
    finest = max((r["nx"] for r in rows), default=0)
    prod = [r for r in rows if r["nx"] == 100]

    def monotone(sel):
        seq = sorted([r for r in rows if sel(r)], key=lambda r: r["nx"])
        if len(seq) < 2:
            return None
        return bool(seq[-1]["max_abs_err_primary"] <= seq[0]["max_abs_err_primary"])

    crit = {
        "C1_primary_le_8pct": worst(lambda r: r["nx"] == finest, "max_abs_err_primary") <= TOL,
        "C2_secondary_le_8pct": worst(lambda r: r["nx"] == finest, "max_abs_err_secondary") <= TOL,
        "C3_converges_upwind": monotone(up),
        "C3_converges_limited2": monotone(l2),
        "C4_mass_drift_le_1e-6": worst(lambda r: True, "mass_drift_max") <= 1e-6,
        "C5_all_finite": all(r["finite"] for r in rows),
    }
    return {
        "reference": {
            "primary": "Martin & Moyce (1952) Fig. 3, h0/a = 2, 14-point digitisation "
                       "(anchors_mm; axes T=t sqrt(2g/a), X=x_f/a, X(0)=1)",
            "secondary": "legacy TensorFVM digitisation re-read on the paper's axes "
                         "(T = t_old, X = x_old): an independent reading of the same "
                         "figure, 1 <= T <= 2.5",
            "digitisation_spread_max": float(max(
                abs(secondary_front(T) - A.mm_front(T)) / A.mm_front(T)
                for T in A.MM_CHECKPOINTS_T if secondary_front(T) == secondary_front(T))),
            "digitisation_pairs": [
                [T, A.mm_front(T), secondary_front(T)] for T in A.MM_CHECKPOINTS_T],
        },
        "gate": {"tolerance": TOL, "checkpoints_T": list(A.MM_CHECKPOINTS_T),
                 "production_nx": 100},
        "legs": rows,
        "criteria": crit,
        "verdict": "PASS" if all(crit.values()) else "FAIL",
    }


def figure_front(rows: list[dict]) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    Td = np.linspace(0.0, A.MM_T[-1], 200)
    ax[0].plot(Td, [A.mm_front(t) for t in Td], "k-", lw=2.2,
               label="Martin & Moyce 1952 (primary digitisation)")
    ax[0].plot(A.MM_T, A.MM_X, "ks", ms=4, alpha=0.6)
    ax[0].plot(LEGACY_T, LEGACY_X, "o", mfc="none", mec="tab:gray", ms=6,
               label="same figure, 2nd digitisation")
    for r in rows:
        xs = [s["X_sim"] for s in r["samples"]]
        ts = [s["T"] for s in r["samples"]]
        ax[0].plot(ts, xs, "-o", ms=3.2,
                   label=f"{r['key']} ({r['cells_across_column']:.0f} cells/column)")
    ax[0].set_xlabel(r"$T = t\sqrt{2g/a}$")
    ax[0].set_ylabel(r"$X = x_f/a$")
    ax[0].set_title("surge front vs experiment")
    ax[0].legend(fontsize=6.4)
    ax[0].grid(alpha=0.3)

    cps = [s["T"] for s in rows[0]["samples"]] if rows else []
    width = 0.8 / max(len(rows), 1)
    for k, r in enumerate(rows):
        ax[1].bar(np.arange(len(cps)) + k * width,
                  [100 * s["err_primary"] for s in r["samples"]], width,
                  label=r["key"])
    ax[1].axhline(8.0, color="tab:red", ls="--", lw=1.2)
    ax[1].axhline(-8.0, color="tab:red", ls="--", lw=1.2, label="±8 % gate")
    ax[1].set_xticks(np.arange(len(cps)) + 0.4)
    ax[1].set_xticklabels([f"T={t:g}" for t in cps])
    ax[1].set_ylabel("(X_sim - X_ref)/X_ref  [%]")
    ax[1].set_title("front error vs the primary digitisation")
    ax[1].legend(fontsize=6.0)
    ax[1].grid(alpha=0.3, axis="y")
    path = os.path.join(FIG, "fig1_front_and_error.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_fields() -> list[str]:
    out = []
    files = sorted(glob.glob(os.path.join(ART, "fields", "*.npz")))
    if not files:
        return out
    best = files[-1]
    with np.load(best) as z:
        alpha = z["alpha"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(alpha, origin="lower", extent=[0, 0.4, 0, 0.2], cmap="Blues",
                   aspect="equal", vmin=0, vmax=1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"alpha at t = 0.172 s ({os.path.basename(best)})")
    fig.colorbar(im, ax=ax, label=r"$\alpha$")
    path = os.path.join(FIG, "fig2_alpha_field.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    out.append(path)
    # front shape (interface height) of the same snapshot
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    col = alpha.max(axis=0)
    ax.fill_between(np.arange(alpha.shape[1]) * (0.4 / alpha.shape[1]), 0, col,
                    step="mid", color="tab:blue", alpha=0.6)
    ax.axvline(A_W, color="k", ls=":", lw=1.0, label="initial column edge a")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"column $\alpha$")
    ax.set_title("free-surface footprint at t = 0.172 s")
    ax.legend(fontsize=7)
    path = os.path.join(FIG, "fig3_surface_profile.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    out.append(path)
    return out


def md_table(rows: list[dict], crit: dict, ref: dict) -> str:
    out = []
    cps = list(A.MM_CHECKPOINTS_T)
    out.append("| leg | grid | cells/col | " +
               " | ".join(f"T={t:g}" for t in cps) +
               " | max\\|err\\| (primary) | max\\|err\\| (2nd digit.) | drift | wall [s] |")
    out.append("|---|---|---:|" + "---|" * len(cps) + "---:|---:|---:|---:|")
    for r in rows:
        cells = " | ".join(
            f"{s['X_sim']:.3f} vs {s['X_primary']:.3f} ({100*s['err_primary']:+.1f} %)"
            for s in r["samples"])
        out.append(
            f"| `{r['key']}` | {r['nx']}x{r['ny']} | {r['cells_across_column']:.1f} | "
            f"{cells} | **{100*r['max_abs_err_primary']:.2f} %** | "
            f"{100*r['max_abs_err_secondary']:.2f} % | "
            f"{r['mass_drift_max']:.1e} | {r['wall_clock_s']:.0f} |")
    out.append("")
    out.append("| criterion | verdict |")
    out.append("|---|---|")
    for k, v in crit.items():
        out.append(f"| `{k}` | **{'PASS' if v else 'FAIL'}** |")
    return "\n".join(out)


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    legs = load_legs()
    if not legs:
        print("no artifacts/leg_*.json -- run run_ladder.py first")
        return 1
    data = grade(legs)
    f1 = figure_front(data["legs"])
    fields = figure_fields()
    data["figures"] = [os.path.relpath(f1, HERE)] + [os.path.relpath(f, HERE) for f in fields]
    with open(os.path.join(ART, "report_data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    table = md_table(data["legs"], data["criteria"], data["reference"])
    print(table)
    print("\nverdict:", data["verdict"])
    with open(os.path.join(ART, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())