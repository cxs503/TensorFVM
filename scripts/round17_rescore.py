#!/usr/bin/env python3
"""Round 17: re-score every stored wall-shear profile with the certified operator.

Round 16 left ``x_r/H`` as ``null`` (UNDEFINED) for two variants and proposed a
grid-refinement ladder to find out whether the null would disappear with
refinement.  It does not: the crossing was measured all along and the round-9
operator's admissibility window ``[1.9, 7.1]`` discarded it.  This driver

* loads every stored mean bottom-wall shear profile (round 9 SA, round 11 SST,
  round 12 SST) and re-scores it with
  :func:`tensorfvm.verification.wall_shear_bubbles.separation_reattachment`,
  printing the legacy value next to it;
* recovers the round-16 patankar / newton2 reattachment from the *crossing list*
  that those artefacts did store (their mean profile was not retained -- an
  artefact-contract defect recorded in ROUND17.md);
* writes ``docs/case-periodic-hills/round17_rescore.json`` and the figures
  ``fig_round17_shear_profiles.png`` / ``fig_round17_xr_comparison.png``.

    PYTHONPATH=src python3 scripts/round17_rescore.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_reference import (
    X_R_OVER_H,
    X_S_OVER_H,  # noqa: F401  (None at Re=2800: WITHHELD, not 0.227)
)
from tensorfvm.verification.periodic_hill_round9 import zero_crossings_periodic
from tensorfvm.verification.wall_shear_bubbles import (
    separation_reattachment,
    signed_crossings,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
WINDOW = (1.9, 7.1)
REF_XR = float(X_R_OVER_H)

NPZ_CASES = [
    ("round9_96x48", "SA (round 9, 96x48)"),
    ("round11_sst_96x48", "SST r11 closure (96x48)"),
    ("round11_sst_r12-96x48", "SST r12 transport fix (96x48)"),
]
JSON_CASES = [
    ("r16_patankar_96x48.json", "SST patankar sink (round 16, 96x48)"),
    ("r16_newton2_96x48.json", "SST Newton sink (rounds 10-15, 96x48)"),
]


def _f(value):
    return None if value is None or not np.isfinite(value) else float(value)


def score_npz(stem, label):
    data = np.load(DOCS / f"{stem}.npz")
    tau = np.asarray(data["mean_tau_b"], dtype=float)
    x = np.asarray(data["x_over_h"], dtype=float)
    legacy = zero_crossings_periodic(tau, x, *WINDOW)
    new = separation_reattachment(tau, x, window=WINDOW)
    return {
        "case": stem,
        "label": label,
        "provenance": "stored mean_tau_b profile (npz)",
        "legacy": {"x_s_over_H": _f(legacy["separation"]),
                   "x_r_over_H": _f(legacy["reattachment"]),
                   "crossings": [float(c) for c in legacy["crossings"]]},
        "certified": {
            "status": new["status"],
            "x_s_over_H": _f(new["x_s_over_H"]),
            "x_r_over_H": _f(new["x_r_over_H"]),
            "out_of_window": bool(new["out_of_window"]),
            "n_bubbles": new["n_bubbles"],
            "bubbles": new["bubbles"],
            "all_crossings": new["all_crossings"],
        },
        "x_r_error_vs_ref": (abs(new["x_r_over_H"] - REF_XR) / REF_XR
                             if np.isfinite(new["x_r_over_H"]) else None),
        "tau": tau.tolist(),
        "x_over_H": x.tolist(),
    }


def score_json(name, label):
    payload = json.loads((DOCS / name).read_text())
    ta = payload["time_averaged"]
    crossings = [float(c) for c in ta.get("crossings") or []]
    # legacy artefact: only rising crossings were stored, and a crossing outside
    # the window was dropped into x_r_over_H = null even though it was measured.
    recovered = max(crossings) if crossings else float("nan")
    return {
        "case": name,
        "label": label,
        "provenance": ("legacy rising-crossing list; mean profile NOT retained, "
                       "so x_s/H cannot be recovered for this run"),
        "legacy": {"x_s_over_H": ta.get("x_s_over_H"),
                   "x_r_over_H": ta.get("x_r_over_H") if "x_r_over_H" in ta
                   else payload.get("x_r_over_H"),
                   "crossings": crossings},
        "certified": {
            "status": "recovered_from_crossing_list",
            "x_s_over_H": None,
            "x_r_over_H": _f(recovered),
            "out_of_window": bool(np.isfinite(recovered) and not
                                  (WINDOW[0] <= recovered <= WINDOW[1])),
            "n_bubbles": 1 if crossings else 0,
            "bubbles": [],
            "all_crossings": [],
        },
        "x_r_error_vs_ref": (abs(recovered - REF_XR) / REF_XR
                             if np.isfinite(recovered) else None),
    }


def _figures(rows, outdir=DOCS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = [r for r in rows if "tau" in r]
    if prof:
        fig, axes = plt.subplots(1, len(prof), figsize=(4.6 * len(prof), 3.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, prof):
            x = np.asarray(row["x_over_H"])
            tau = np.asarray(row["tau"])
            ax.plot(x, tau, color="C0", lw=1.4, label=r"mean $\tau_w/\rho$")
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.axvspan(WINDOW[0], WINDOW[1], color="C1", alpha=0.12,
                       label="round-5 window")
            cert = row["certified"]
            legacy = row["legacy"]
            if cert["x_s_over_H"] is not None:
                ax.axvline(cert["x_s_over_H"], color="C3", ls="--", lw=1.2,
                           label=r"certified $x_s/H$")
            ax.axvline(cert["x_r_over_H"], color="C2", ls="-", lw=1.4,
                       label=r"certified $x_r/H$=%.3f" % cert["x_r_over_H"])
            if legacy["x_r_over_H"] is not None:
                ax.axvline(legacy["x_r_over_H"], color="k", ls=":", lw=1.0,
                           label=r"legacy $x_r/H$=%.3f" % legacy["x_r_over_H"])
            else:
                ax.set_title(f"{row['case']}\nlegacy: UNDEFINED (was $>7.1$)",
                             fontsize=8)
            if legacy["x_r_over_H"] is not None:
                ax.set_title(row["case"], fontsize=8)
            ax.set_xlabel(r"$x/H$")
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel(r"$\tau_w/\rho$  [m$^2$/s$^2$]")
        axes[0].legend(fontsize=6, loc="upper left")
        fig.suptitle("Round 17: certified reattachment from the stored mean wall shear",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "fig_round17_shear_profiles.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels, values, colors = [], [], []
    for row in rows:
        xr = row["certified"]["x_r_over_H"]
        labels.append(row["label"])
        values.append(np.nan if xr is None else xr)
        colors.append("C2" if row["certified"]["out_of_window"] else "C0")
    idx = np.arange(len(labels))
    ax.axhspan(REF_XR * 0.9, REF_XR * 1.1, color="0.85", label="Breuer 2009 +/-10%")
    ax.axhline(REF_XR, color="C3", lw=1.2, label=r"Breuer 2009 $x_r/H$=%.2f" % REF_XR)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.scatter(values, idx, c=colors, s=46, zorder=3)
    for j, (value, row) in enumerate(zip(values, rows)):
        tag = row["certified"]["status"]
        ax.annotate("" if not np.isfinite(value) else "%.3f (%.0f%%)"
                    % (value, 100 * row["x_r_error_vs_ref"]),
                    (value, j), textcoords="offset points", xytext=(6, 6),
                    fontsize=7)
        ax.annotate(tag, (0.02, j), xycoords=("axes fraction", "data"),
                    fontsize=6, color="0.35", va="center")
    ax.set_yticks(idx)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 9)
    ax.set_xlabel(r"$x_r/H$")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Round 17: reattachment re-adjudicated (green = outside the round-5 window)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_round17_xr_comparison.png", dpi=160)
    plt.close(fig)


def main() -> int:
    rows = [score_npz(stem, label) for stem, label in NPZ_CASES]
    rows += [score_json(name, label) for name, label in JSON_CASES]

    print(f"{'case':32s} {'legacy x_r':>11s} {'certified x_r':>14s} "
          f"{'err%':>7s}  status")
    for row in rows:
        legacy = row["legacy"]["x_r_over_H"]
        cert = row["certified"]["x_r_over_H"]
        err = row["x_r_error_vs_ref"]
        print(f"{row['case']:32s} "
              f"{'nan' if legacy is None else f'{legacy:.4f}':>11s} "
              f"{'nan' if cert is None else f'{cert:.4f}':>14s} "
              f"{'--' if err is None else f'{100 * err:.1f}':>7s}  "
              f"{row['certified']['status']}")
        if row["certified"].get("x_s_over_H") is not None:
            legacy_s = row["legacy"]["x_s_over_H"]
            print(f"{'':32s} x_s/H legacy="
                  f"{'nan' if legacy_s is None or not np.isfinite(legacy_s) else f'{legacy_s:.3f}'}"
                  f"  certified={row['certified']['x_s_over_H']:.3f}")

    summary = {
        "round": 17,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "reference": {"x_r_over_H": REF_XR, "source": "Breuer et al. 2009"},
        "window": list(WINDOW),
        "finding": ("round-9 operator discarded a measured reattachment outside "
                    "the round-5 window and never detected a falling crossing; "
                    "no grid refinement was needed to obtain these numbers"),
        "rows": rows,
    }
    (DOCS / "round17_rescore.json").write_text(json.dumps(summary, indent=1))
    _figures(rows)
    print(f"\nwrote {DOCS / 'round17_rescore.json'}")
    print(f"wrote {DOCS / 'fig_round17_shear_profiles.png'} "
          f"{DOCS / 'fig_round17_xr_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())