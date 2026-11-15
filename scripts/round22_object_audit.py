#!/usr/bin/env python3
"""Round 22 object audit: does the published ``x_r/H`` measure the reference bubble?

Round 21 published ``x_r/H = 8.26`` for the closure-free ``Re = 2800`` run and
compared it with Breuer et al. (2009) ``5.41``.  The certified operator is run
here with **both** selection rules on the same stored mean wall shear, and on
every closure-free artefact of the anchor, to find out whether the two numbers
are the same physical object.

Outputs
-------
* ``docs/case-periodic-hills/round22_object_audit.json`` -- one record per run
* ``docs/case-periodic-hills/ROUND22-OBJECT-TABLE.md``  -- the markdown table
  quoted by ``ROUND22.md`` (generated, never hand-typed)
* ``docs/case-periodic-hills/fig_round22_object.png``   -- shear + bubbles, and
  the ``y/H = 0.5`` profile against the reference
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_field_metrics import bubble_object_report
from tensorfvm.verification.periodic_hill_round11 import _REATTACHMENT_WINDOW

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
REFERENCE_X_R = 5.41

RUNS = [
    ("Re=2800 closure-free (round 20, published +53 %)", "round20_laminar_Re2800_96x48"),
    ("Re=700 closure-free (round 20, published +54 %)", "round20_laminar_Re700_96x48"),
    ("Re=2800 SST r18 (round 18, published +42 %)", "round11_sst_r18_96x48"),
    ("Re=2800 SST r12 (round 11)", "round11_sst_r12-96x48"),
    ("Re=2800 SST r15 (round 11)", "round11_sst_r15_96x48"),
]


def main() -> int:
    records = []
    for label, stem in RUNS:
        npz = CASEDIR / f"{stem}.npz"
        if not npz.exists():
            print(f"[skip] {stem}")
            continue
        with np.load(npz) as z:
            tau = np.asarray(z["mean_tau_b"], dtype=float)
            x = np.asarray(z["x_over_h"], dtype=float)
            mean_u = np.asarray(z["mean_u"], dtype=float)
            yc = np.asarray(z["yc"], dtype=float)
        rep = bubble_object_report(tau, x, window=_REATTACHMENT_WINDOW)
        rep["label"] = label
        rep["stem"] = stem
        rep["error_largest_vs_ref"] = (rep["x_r_over_H"]["largest_area"] - REFERENCE_X_R)
        if rep["x_r_over_H"]["first"] is not None:
            rep["error_crest_vs_ref"] = rep["x_r_over_H"]["first"] - REFERENCE_X_R
        records.append(rep)

    (CASEDIR / "round22_object_audit.json").write_text(
        json.dumps({"reference_x_r_over_H": REFERENCE_X_R, "runs": records},
                   indent=1, default=float))

    lines = ["| run | n bubbles | selected (largest area) | crest bubble `x_r/H` |"
             " published value vs 5.41 | flag |",
             "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r['label']} | {r['n_bubbles']} | "
            f"{r['x_r_over_H']['largest_area']:.3f} | "
            f"{r['x_r_over_H']['first']:.3f} | "
            f"{(r['x_r_over_H']['largest_area'] - REFERENCE_X_R) / REFERENCE_X_R * 100:+.1f} % | "
            f"`{r['flag']}` |")
    (CASEDIR / "ROUND22-OBJECT-TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for r in records:
        with np.load(CASEDIR / f"{r['stem']}.npz") as z:
            tau = np.asarray(z["mean_tau_b"], dtype=float)
            x = np.asarray(z["x_over_h"], dtype=float)
        tag = r["stem"].replace("round20_laminar_", "").replace("round11_", "")
        ax[0].plot(x, tau, lw=1.1, label=tag)
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xlabel("x/H")
    ax[0].set_ylabel(r"$\langle\tau_w\rangle$ (arb. norm.)")
    ax[0].set_title("time-mean wall shear: sign changes = bubble boundaries")
    for r in records:
        if r["n_bubbles"] > 1:
            for b in (r["selected"], r["crest"]):
                ax[0].axvline(b["separation"], color="0.7", lw=0.4, ls=":")
                ax[0].axvline(b["reattachment"], color="0.7", lw=0.4, ls=":")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)

    path = CASEDIR / "round20_laminar_Re2800_96x48.json"
    payload = json.loads(path.read_text())
    hp = payload["horizontal_profile_y_H_0.5"]
    ref = {float(k): v for k, v in hp["reference"].items()}
    ax[1].plot(list(ref), list(ref.values()), "k-o", ms=3, label="Breuer 2009 (y/H=0.5)")
    ax[1].plot(hp["x_over_H"], hp["u_over_Ub"], "C3.-", ms=4,
               label="closure-free Re=2800 (mean)")
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].set_xlabel("x/H")
    ax[1].set_ylabel(r"$u/U_b$ at $y/H=0.5$")
    ax[1].set_title("the anchor's second metric: 0/7 stations within 10 %")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CASEDIR / "fig_round22_object.png", dpi=140)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())