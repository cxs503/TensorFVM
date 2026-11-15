#!/usr/bin/env python3
"""Round 20 analysis: lower-Re sister cases vs the KB reattachment ladder.

Builds ``docs/case-periodic-hills/round20_table.json`` and
``fig_round20_ladder.png`` from whatever round-20 artefacts exist, and states
per case whether it is measured, still queued, or withheld.  A queued case is
drawn on the figure as an open symbol labelled "queued" -- it is never smoothed
into a trend line.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_lower_re import (
    KB_REATTACHMENT_BY_RE, KB_SOURCE, sister_table,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
#: jobs submitted in this round, in submission order (provenance of a pending row)
PENDING = {"laminar:700:96x48": "cj3663", "laminar:1400:96x48": "cj3664",
           "laminar:2800:96x48": "cj3665"}


def load_round20():
    rows = []
    for path in sorted(OUT.glob("round20_*_Re*_*.json")):
        payload = json.loads(path.read_text())
        row = dict(payload["lower_re"])
        row["artefact"] = path.name
        rows.append(row)
    return rows


def load_re2800_closure_runs():
    """The Re=2800 closure runs already published by rounds 17-19."""
    rows = []
    for path in (OUT / "round18_ladder.json",):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        levels = data.get("levels") or data.get("rows") or []
        for lvl in levels:
            if lvl.get("diverged_at_step") is not None:
                continue
            x_r = lvl.get("x_r_over_H")
            if x_r is None:
                continue
            rows.append({"reynolds": 2800.0, "closure": "sst",
                         "x_r_over_H": float(x_r), "tag": lvl.get("tag")})
    return rows


def main() -> int:
    rows = load_round20()
    table = sister_table({f"r20_{r['reynolds']:g}_{r['closure']}": r for r in rows}
                         ) if rows else {"rows": [], "n_pass": 0, "n_fail": 0,
                                         "n_withheld": 0, "n_undefined": 0, "n_diverged": 0}
    closure_rows = load_re2800_closure_runs()
    table["pending"] = PENDING
    table["re2800_closure_runs"] = closure_rows
    table["source"] = KB_SOURCE
    (OUT / "round20_table.json").write_text(json.dumps(table, indent=1))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    res = np.array(sorted(KB_REATTACHMENT_BY_RE))
    xr = np.array([KB_REATTACHMENT_BY_RE[r] for r in res])
    ax.plot(res, xr, "k-o", lw=1.6, ms=6, label="Breuer 2009 / ERCOFTAC KB (reference)")
    ax.fill_between(res, xr * 0.9, xr * 1.1, color="0.85", zorder=0,
                    label="+/-10 % acceptance band")
    if closure_rows:
        ax.plot([r["reynolds"] for r in closure_rows], [r["x_r_over_H"] for r in closure_rows],
                "s", ms=7, mfc="tab:red", mec="k", label="this repo, Re=2800, SST k-w (r18/r19)")
    for key, jid in PENDING.items():
        closure, re_s, tag = key.split(":")
        got = [r for r in rows if abs(r["reynolds"] - float(re_s)) < 1e-9]
        if got:
            ax.plot(float(re_s), got[0]["x_r_over_H"], "D", ms=8, mfc="tab:blue",
                    mec="k", label="this round, closure-free (measured)")
            ax.annotate(f"{got[0]['x_r_over_H']:.2f}\n({got[0]['verdict']})",
                        (float(re_s), got[0]["x_r_over_H"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=8)
        else:
            ax.plot(float(re_s), KB_REATTACHMENT_BY_RE[float(re_s)], "D", ms=9,
                    mfc="none", mec="tab:blue", mew=1.6,
                    label="this round, closure-free (queued)")
            ax.annotate("queued " + jid, (float(re_s), KB_REATTACHMENT_BY_RE[float(re_s)]),
                        textcoords="offset points", xytext=(8, -12), fontsize=8,
                        color="tab:blue")
    ax.set_xscale("log")
    ax.set_xticks(res)
    ax.set_xticklabels([f"{int(r)}" for r in res])
    ax.set_xlabel("Re = U_b H / nu")
    ax.set_ylabel("x_r/H")
    ax.set_title("Periodic hills: reattachment ladder (measured vs reference)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_round20_ladder.png", dpi=150)
    print(json.dumps({"n_round20_rows": len(rows), "pending": PENDING,
                      "table_counts": {k: table.get(k) for k in
                                       ("n_pass", "n_fail", "n_withheld",
                                        "n_undefined", "n_diverged")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
