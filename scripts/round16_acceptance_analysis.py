#!/usr/bin/env python3
"""Round-16 analysis: does the SST ``omega`` sink linearisation move the bubble?

Reads the acceptance artefacts written by ``scripts/round16_acceptance.py``
(``r16_patankar_96x48.json`` / ``r16_newton2_96x48.json``) plus the round-15
baseline (``round11_sst_r15_96x48.json``, legacy sink) and writes

    docs/case-periodic-hills/round16_acceptance.json   (table, zero hand-copy)
    docs/case-periodic-hills/fig_round16_acceptance.png

The verdict is the one the round-15 scoring path gives (``score_hill``: 5.41 and
the DNS ``y/H = 0.5`` table), never the payload's own legacy 5.34 gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_reference import (
    STATIONS, X_R_OVER_H, X_R_TOLERANCE, load_profile, y05_table)

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
SOURCES = [OUT, Path("/tmp/r16")]


def load(name):
    for root in SOURCES:
        path = root / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def row(name):
    payload = load(name)
    if payload is None:
        return None
    score = payload.get("score_hill") or {}
    xr = payload["time_averaged"].get("x_r_over_H")
    return {
        "artefact": name,
        "variant": payload.get("variant", "legacy (round 15, newton2)"),
        "sink_coefficient_ratio": (payload.get("sink_coefficient_ratio") or {}).get("ratio"),
        "mesh": f"{payload['config']['nx']}x{payload['config']['ny']}",
        "steps": payload["config"]["steps"],
        "window": payload["config"]["averaging_window"],
        "x_r_over_H": xr,
        "error_vs_5.41": (abs(xr - X_R_OVER_H) / X_R_OVER_H) if xr else None,
        "stationarity_spread_rel": (payload.get("stationarity") or {}).get(
            "x_r_over_H_block_spread_rel"),
        "profile_pass": (f"{score.get('profile_stations_pass')}"
                         f"/{score.get('profile_stations_admissible')}"),
        "diverged_at_step": payload["config"].get("diverged_at_step"),
        "wall_clock_s": payload.get("wall_clock_s"),
        "profile": (payload.get("horizontal_profile_y_H_0.5") or {}),
        "max_nu_t_ratio": (payload["time_averaged"] or {}).get("max_nu_t_ratio"),
        "body_force": (payload["time_averaged"] or {}).get("body_force"),
    }


def figure(rows, path):
    """y/H = 0.5 mean profile vs the distributed DNS, and the x_r/H ladder."""
    band = (X_R_OVER_H * (1 - X_R_TOLERANCE), X_R_OVER_H * (1 + X_R_TOLERANCE))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) the y/H = 0.5 profile
    ts = {s: load_profile(s) for s in STATIONS}
    y05 = y05_table()
    ax[0].plot([s for s in y05], [y05[s] for s in y05], "ko-", ms=5,
               label="DNS Breuer/UFR 3-30 (KB, y/H=0.5)")
    styles = {"patankar": ("C3", "o"), "newton2": ("C0", "s"),
              "legacy (round 15, newton2)": ("C0", "s")}
    for r in rows:
        hp = r["profile"] or {}
        xs, us = hp.get("x_over_H"), hp.get("u_over_Ub")
        if not xs:
            continue
        color, marker = styles.get(r["variant"], ("C2", "^"))
        label = (f"{r['variant']} {r['mesh']} "
                 f"(S_P/[beta*omega]={r['sink_coefficient_ratio']})")
        ax[0].plot(xs, us, marker, color=color, ms=3, ls="-", lw=1.0, label=label)
    ax[0].set_xlabel("x/H")
    ax[0].set_ylabel("U/U_b at y/H = 0.5")
    ax[0].set_xlim(0, 9)
    ax[0].set_ylim(-0.15, 1.25)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # (b) reattachment length vs the 10 % band
    labels, vals, colors = [], [], []
    for r in rows:
        if r["x_r_over_H"] is None:
            continue
        labels.append(f"{r['variant']}\n{r['mesh']}")
        vals.append(r["x_r_over_H"])
        colors.append(styles.get(r["variant"], ("C2", "^"))[0])
    idx = np.arange(len(vals))
    ax[1].axhspan(*band, color="0.85", label="5.41 +- 10 %")
    ax[1].axhline(X_R_OVER_H, color="k", ls="--", lw=1,
                  label=f"DNS x_r/H = {X_R_OVER_H}")
    ax[1].bar(idx, vals, color=colors, width=0.55)
    for i, v in zip(idx, vals):
        ax[1].text(i, v + 0.08, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("x_r/H (time mean)")
    ax[1].set_ylim(0, 6.5)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("CASE-FVM-PERIODIC-HILLS round 16: omega sink linearisation A/B "
                 "(Re = 2800, SST, 96x48)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main():
    rows = [r for r in (row("r16_patankar_96x48.json"),
                        row("r16_newton2_96x48.json"),
                        row("round11_sst_r15_96x48.json")) if r]
    fig = figure(rows, OUT / "fig_round16_acceptance.png")
    payload = {
        "round": 16,
        "kind": "acceptance A/B of the SST omega sink linearisation",
        "reference": {"x_r_over_H": X_R_OVER_H, "tolerance": X_R_TOLERANCE},
        "rows": rows,
        "figure": str(fig.name),
    }
    (OUT / "round16_acceptance.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()