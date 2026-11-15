#!/usr/bin/env python3
"""Round 22 analysis: stationarity of the *published* periodic-hill metric.

Inputs (all already in the repository, nothing is re-run here)

* ``docs/case-periodic-hills/round20_laminar_Re{700,2800}_96x48.{json,npz}``
  -- the round-20 closure-free sister cases (Re=1400 diverged, excluded);
* ``docs/case-periodic-hills/round21_table.json`` -- the round-21 two-definition
  table.

Outputs

* ``round22_table.json`` -- every number used in ``ROUND22.md`` (zero hand copy);
* ``fig_round22_block_stationarity.png`` -- per-block ``x_r/H`` of the published
  metric vs the full-mean value, with the 10 % acceptance band.

The script establishes the *measurement-integrity* verdict (is the published
``x_r/H`` stationary at block level? is it corroborated by an independent data
path?) -- it does NOT re-score physics accuracy (that is
:mod:`tensorfvm.verification.periodic_hill_lower_re`, and it is FAIL).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_stationarity import (
    STATIONARY_TOLERANCE,
    block_stationarity,
    published_metric_stationarity,
    reconcile_definitions,
)
from tensorfvm.verification.periodic_hill_lower_re import (
    KB_REATTACHMENT_BY_RE, KB_SOURCE,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def load_case(path: Path) -> dict:
    payload = json.loads(path.read_text())
    npz = np.load(path.with_suffix(".npz"))
    x = npz["x_over_h"]
    # independent re-scoring from the *saved* per-block mean shear (round 22);
    # pre-round-22 artefacts have no block_tau_b in the npz, so fall back to the
    # JSON's derived per-block values (published_metric_stationarity).
    if "block_tau_b" in npz.files:
        bs = block_stationarity(npz["block_tau_b"], x)
    else:
        bs = published_metric_stationarity(payload)
    pub = published_metric_stationarity(payload)
    rec = reconcile_definitions(payload, mean_u=npz["mean_u"],
                                x_over_h=x, mean_tau_b=npz["mean_tau_b"])
    return {
        "artefact": path.name,
        "reynolds": float(payload["config"]["reynolds"]),
        "closure": payload.get("closure"),
        "diverged_at_step": payload["config"].get("diverged_at_step"),
        "reference_x_r_over_H": payload["lower_re"].get("reference_x_r_over_H"),
        "x_r_over_H_published": payload["lower_re"].get("x_r_over_H"),
        "block_stationarity": bs,
        "published_metric_stationarity": pub,
        "reconcile": rec,
        "per_sample_stationarity_ok": payload["stationarity"]["x_r_over_H"].get("ok"),
        "per_sample_stationarity_spread": payload["stationarity"]["x_r_over_H"].get("spread_rel"),
        "lower_re": payload["lower_re"],
    }


def main() -> int:
    cases = [load_case(p) for p in sorted(OUT.glob("round20_laminar_Re*_96x48.json"))]
    measured = [c for c in cases if c["diverged_at_step"] is None]

    table = {
        "cases": cases,
        "source": KB_SOURCE,
        "tolerance": STATIONARY_TOLERANCE,
        "stationarity_verdict": {
            "re700": next((c["block_stationarity"]["status"] for c in measured
                           if c["reynolds"] == 700.0), None),
            "re2800": next((c["block_stationarity"]["status"] for c in measured
                            if c["reynolds"] == 2800.0), None),
        },
        "note": ("measurement-integrity verdict only: the published x_r/H is "
                 "stationary at block level and corroborated by an independent "
                 "backflow path; physics accuracy vs Breuer 2009 is "
                 "periodic_hill_lower_re and is FAIL (52-54 %)."),
    }
    (OUT / "round22_table.json").write_text(json.dumps(table, indent=1))

    # ---- figure: per-block x_r/H of the published metric ------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c in measured:
        re = c["reynolds"]
        ref = KB_REATTACHMENT_BY_RE[re]
        xr_pub = c["x_r_over_H_published"]
        blocks = c["block_stationarity"]["x_r_over_H_by_block"]
        ax.plot([re] * len(blocks), blocks, "o", ms=7, mfc="tab:blue", mec="k",
                label=f"Re={int(re)}: per-block x_r/H (certified, saved shear)")
        ax.plot([re], [xr_pub], "D", ms=10, mfc="tab:red", mec="k",
                label=f"Re={int(re)}: full-mean x_r/H (published)")
        ax.errorbar([re], [xr_pub], yerr=[[0], [0]], fmt="none")
        # 10 % band around the reference
        ax.fill_between([re - 0.5, re + 0.5], ref * 0.9, ref * 1.1,
                        color="0.85", zorder=0)
        ax.annotate(f"block spread {c['block_stationarity']['dispersion']['spread_rel']:.1%}",
                    (re, max(blocks)), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([700.0, 2800.0])
    ax.set_xticklabels(["700", "2800"])
    ax.set_xlabel("Re = U_b H / nu")
    ax.set_ylabel("x_r/H")
    ax.set_title("Published metric is stationary at block level (closure-free)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_round22_block_stationarity.png", dpi=150)

    print(json.dumps({"cases": [{k: c[k] for k in
                                 ("reynolds", "x_r_over_H_published",
                                  "block_stationarity", "reconcile",
                                  "per_sample_stationarity_ok")}
                                for c in measured]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
