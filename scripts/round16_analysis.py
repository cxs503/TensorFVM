#!/usr/bin/env python3
"""Round-16 analysis: does the SST ``omega`` sink fix move the periodic-hill bubble?

Reads the round-11-schema artefacts produced by
``scripts/round15_channel_and_rescore.py hill`` for round 16 (and the round-15
baseline), writes

    docs/case-periodic-hills/round16_analysis.json
    docs/case-periodic-hills/fig_round16_hill.png
    docs/case-periodic-hills/fig_round16_nut.png

Nothing is hand-copied: every number in ``ROUND16.md`` comes from these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_reference import X_R_OVER_H, y05_table

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
SOURCES = [OUT, Path("/tmp/r16")]


def load(name):
    for root in SOURCES:
        path = root / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def score(payload):
    """10 % bands, round-15 convention (relative for |ref| >= 0.1, else 0.10 U_b)."""
    if payload is None:
        return None
    ta = payload["time_averaged"]
    xr = ta["x_r_over_H"]
    row = {
        "tag": payload["tag"],
        "mesh": f"{payload['config']['nx']}x{payload['config']['ny']}",
        "steps": payload["config"]["steps"],
        "x_r_over_H": xr,
        "error_vs_5.41": (abs(xr - X_R_OVER_H) / X_R_OVER_H) if xr else None,
        "error_vs_5.34": (abs(xr - 5.34) / 5.34) if xr else None,
        "max_nu_t_ratio": (max(payload["series"]["max_nu_t_ratio"])
                           if payload["series"]["max_nu_t_ratio"] else None),
        "mean_nu_t_ratio": (float(np.mean(payload["series"]["mean_nu_t_ratio"]))
                            if payload["series"]["mean_nu_t_ratio"] else None),
        "stationary": payload["stationarity"]["x_r_over_H"].get("stationary"),
        "gate_pass": payload["gate"]["pass"],
    }
    ref = y05_table()
    prof = payload["horizontal_profile_y_H_0.5"]
    xs = np.asarray(prof["x_over_H"], dtype=float)
    us = np.asarray(prof["u_over_Ub"], dtype=float)
    stations, passing = [], 0
    for station, u_ref in sorted(ref.items()):
        # linear interpolation in x on the computed vertical line profile
        u = float(np.interp(station, xs, us))
        band = 0.10 * abs(u_ref) if abs(u_ref) >= 0.1 else 0.10
        ok = abs(u - u_ref) <= band
        passing += int(ok)
        stations.append({"x_over_H": station, "u_computed": u,
                         "u_reference": u_ref, "delta": u - u_ref,
                         "rel_error": (u - u_ref) / abs(u_ref) if u_ref else None,
                         "pass": ok})
    row["profile_stations"] = stations
    row["profile_passing"] = f"{passing}/{len(stations)}"
    return row


def main():
    arts = {tag: load(f"round11_sst_{tag}.json")
            for tag in ("r15_96x48", "r16_fix_b0", "r16_fix_b1")}
    rows = {tag: score(p) for tag, p in arts.items()}
    local = load("round16_local_ab.json")

    summary = {"round": 16, "reference_x_r_over_H": X_R_OVER_H,
               "reference_x_r_over_H_ledger": 5.34, "runs": rows,
               "local_ab": local}
    (OUT / "round16_analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (None if v is None else {
        "x_r_over_H": v["x_r_over_H"], "error_vs_5.41": v["error_vs_5.41"],
        "profile": v["profile_passing"], "max_nu_t_ratio": v["max_nu_t_ratio"],
        "stationary": v["stationary"], "pass": v["gate_pass"]})
        for k, v in rows.items()}, indent=1))

    # ---------------------------------------------------------------- figure 1
    ref = y05_table()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot([k for k in sorted(ref)], [ref[k] for k in sorted(ref)], "k^-",
            label="Breuer 2009 DNS (Re=2800)", ms=5)
    style = {"r15_96x48": ("C0", "s", "round 15 (2*beta*omega sink)"),
             "r16_fix_b0": ("C3", "o", "round 16, transport blend 0"),
             "r16_fix_b1": ("C2", "D", "round 16, transport blend 1")}
    for tag, (colour, marker, label) in style.items():
        row = rows.get(tag)
        if row is None:
            continue
        xs = [s["x_over_H"] for s in row["profile_stations"]]
        us = [s["u_computed"] for s in row["profile_stations"]]
        ax.plot(xs, us, colour + marker + "-", ms=4, lw=1.2,
                label=f"{label} (x_r/H={row['x_r_over_H']:.2f})")
    ax.set_xlabel("x/H"); ax.set_ylabel(r"$\bar u/U_b$ at $y/H=0.5$")
    ax.set_title("mean velocity profile above the crest level")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    tags = [t for t in ("r15_96x48", "r16_fix_b0", "r16_fix_b1")
            if rows.get(t) and rows[t]["x_r_over_H"]]
    vals = [rows[t]["x_r_over_H"] for t in tags]
    ax.bar(range(len(tags)), vals,
           color=["C0", "C3", "C2"][:len(tags)])
    ax.axhline(X_R_OVER_H, color="k", ls="--",
               label=f"DNS $x_r/H$={X_R_OVER_H}")
    ax.axhspan(0.9 * X_R_OVER_H, 1.1 * X_R_OVER_H, color="k", alpha=0.08,
               label="+-10 % acceptance band")
    ax.axhline(5.34, color="grey", ls=":", label="ledger 5.34")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([t.replace("r16_fix_", "r16 b=") for t in tags], fontsize=8)
    ax.set_ylabel(r"$x_r/H$"); ax.set_ylim(0, 6.4)
    ax.set_title("reattachment length"); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_round16_hill.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    if local:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        variants = [r["variant"] for r in local["rows"]]
        maxr = [r["max_nu_t_ratio"] for r in local["rows"]]
        meanr = [r["mean_nu_t_ratio"] for r in local["rows"]]
        idx = np.arange(len(variants))
        ax.bar(idx - 0.2, maxr, width=0.4, label="max")
        ax.bar(idx + 0.2, meanr, width=0.4, label="time-mean")
        ax.set_xticks(idx)
        ax.set_xticklabels(["Patankar\n(fix)", "2 beta omega\n(old)"])
        ax.set_ylabel(r"$\nu_t/\nu$")
        ax.set_title(f"coarse {local['mesh']} A/B, {local['steps']} steps")
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_round16_nut.png", dpi=140)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())