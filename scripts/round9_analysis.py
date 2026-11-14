#!/usr/bin/env python3
"""Round 9 analysis: figures + machine-readable summary for the time-averaged gate.

Reads the ``docs/case-periodic-hills/round9_<tag>.json`` / ``.npz`` pairs written
by :mod:`scripts.round9_long_time_average` and produces

    docs/case-periodic-hills/round9_summary.json
    docs/case-periodic-hills/fig_round9_<name>.png

Every number in the summary is read from the run artefacts (no hand-copied
values), so the report's tables can be regenerated from the committed files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "case-periodic-hills"
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.periodic_hill import hill_profile  # noqa: E402
from tensorfvm.verification.periodic_hill_round6 import (  # noqa: E402
    BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
)

COLORS = {"48x24": "#1f77b4", "96x48": "#d62728", "144x72": "#2ca02c", "smoke": "#777777"}


def load(tags=None):
    out = {}
    for path in sorted(DOCS.glob("round9_*.json")):
        tag = path.stem.replace("round9_", "")
        if tag in ("summary", "smoke"):
            continue
        if tags and tag not in tags:
            continue
        npz = path.with_suffix(".npz")
        if not npz.exists():
            continue
        out[tag] = (json.loads(path.read_text()), np.load(npz))
    return out


def fig_xr_timeseries(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    ref = REFERENCE["x_r_over_H"]
    for tag, (payload, _) in runs.items():
        t = np.array(payload["series"]["t"])
        xr = np.array([np.nan if v is None else v for v in payload["series"]["x_r_over_H"]])
        axes[0].plot(t, xr, lw=0.9, color=COLORS.get(tag), label=f"{tag}")
        avg = payload["time_averaged"]["x_r_over_H"]
        if avg is not None:
            axes[0].axhline(avg, ls=":", lw=1.0, color=COLORS.get(tag))
        axes[1].plot(t, [np.nan if v is None else v for v in payload["series"]["max_nu_t_ratio"]],
                     lw=0.9, color=COLORS.get(tag), label=f"{tag}")
    axes[0].axhspan(0.9 * ref, 1.1 * ref, color="0.85", zorder=0,
                    label=f"Breuer 2009 $x_r/H$={ref} ±10%")
    axes[0].axhline(ref, color="k", lw=1.0)
    axes[0].set_xlabel("$t\\,U_b/H$")
    axes[0].set_ylabel("instantaneous $x_r/H$")
    axes[0].set_ylim(0, 10)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("(a) reattachment length (dotted = time mean)")
    axes[1].set_xlabel("$t\\,U_b/H$")
    axes[1].set_ylabel("$\\max\\,\\nu_t/\\nu$")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    axes[1].set_title("(b) closure level over the same window")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_field_mean(payload, npz, out, tag):
    u = npz["mean_u"] / payload["config"]["bulk_velocity"]
    yc = npz["yc"]
    xc = npz["x_over_h"]
    H = 1.0
    fig, ax = plt.subplots(figsize=(9, 3.2))
    dx = float(xc[1] - xc[0])
    x_edges = np.append(xc - 0.5 * dx, xc[-1] + 0.5 * dx)
    y_edges = np.append(yc[:, 0] - 0.5 * (yc[1, 0] - yc[0, 0]),
                        yc[-1, 0] + 0.5 * (yc[-1, 0] - yc[-2, 0]))
    mesh = ax.pcolormesh(x_edges, y_edges, u, shading="flat",
                         cmap="RdBu_r", vmin=-0.5, vmax=1.5)
    hx = np.linspace(0.0, 9.0, 901)
    hy = hill_profile(hx)[0]
    ax.fill_between(hx, 0.0, hy, color="0.35", zorder=3)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.036)
    ax.set_xlabel("$x/H$")
    ax.set_ylabel("$y/H$")
    ax.set_title(f"time-averaged $u/U_b$ ({tag}, gray = hill body)")
    fig.colorbar(mesh, ax=ax, label="$\\bar u/U_b$")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_profile_y05(runs, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ref_x = np.array(sorted(BREUER_PROFILE_Y05))
    ref_u = np.array([BREUER_PROFILE_Y05[k] for k in ref_x])
    ax.plot(ref_x, ref_u, "k-o", ms=5, lw=1.2, label="Breuer et al. 2009 (LES, UFR 3-30)")
    ax.fill_between(ref_x, ref_u * (1 - REFERENCE_TOLERANCE), ref_u * (1 + REFERENCE_TOLERANCE),
                    color="0.85", alpha=0.6, zorder=0, label="±10 % band")
    for tag, (payload, _) in runs.items():
        prof = payload["horizontal_profile_y_H_0.5"]
        ax.plot(prof["x_over_H"], prof["u_over_Ub"], "-", lw=1.3,
                color=COLORS.get(tag), label=f"time mean, {tag}")
    ax.set_xlabel("$x/H$")
    ax.set_ylabel("$u/U_b$ at $y/H=0.5$")
    ax.set_title("mean velocity on the line $y/H=0.5$ vs reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_vertical_profiles(npz, payload, out, stations=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0)):
    mean_u = npz["mean_u"] / payload["config"]["bulk_velocity"]
    yc = npz["yc"]
    xc = npz["x_over_h"]
    fig, axes = plt.subplots(1, len(stations), figsize=(2.3 * len(stations), 3.6), sharey=True)
    for ax, xs in zip(np.atleast_1d(axes), stations):
        i = int(np.argmin(np.abs(xc - xs)))
        ax.plot(mean_u[:, i], yc[:, i], "o-", ms=2, lw=1.0, color="#1f77b4")
        ax.axvline(0.0, color="0.6", lw=0.6)
        ax.set_title(f"$x/H={xc[i]:.2f}$", fontsize=9)
        ax.set_xlabel("$u/U_b$")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("$y/H$")
    fig.suptitle("time-averaged vertical profiles (finest mesh)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_stationarity(runs, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    tags = list(runs)
    for k, tag in enumerate(tags):
        blocks = runs[tag][0]["stationarity"]["x_r_over_H"]["blocks"]
        ax.bar(np.arange(len(blocks)) + k * 0.25, blocks, width=0.22,
               color=COLORS.get(tag), label=tag)
    ax.axhline(REFERENCE["x_r_over_H"], color="k", lw=1.0)
    ax.axhspan(0.9 * REFERENCE["x_r_over_H"], 1.1 * REFERENCE["x_r_over_H"], color="0.88", zorder=0)
    ax.set_xticks(range(4))
    ax.set_xlabel("contiguous block of the averaging window")
    ax.set_ylabel("block-mean $x_r/H$ (instantaneous crossings)")
    ax.set_title("stationarity of the sampling window")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    runs = load()
    if not runs:
        print("no round9_*.json/.npz artefacts yet")
        return 1
    print("tags:", list(runs))
    fig_xr_timeseries(runs, DOCS / "fig_round9_xr_timeseries.png")
    fig_profile_y05(runs, DOCS / "fig_round9_profile_y05.png")
    fig_stationarity(runs, DOCS / "fig_round9_stationarity.png")
    finest = max(runs, key=lambda t: runs[t][0]["config"]["nx"] * runs[t][0]["config"]["ny"])
    payload, npz = runs[finest]
    fig_field_mean(payload, npz, DOCS / "fig_round9_field_mean.png", finest)
    fig_vertical_profiles(npz, payload, DOCS / "fig_round9_vertical_profiles.png")

    summary = {"reference": REFERENCE, "tolerance": REFERENCE_TOLERANCE,
               "runs": {}, "finest": finest}
    for tag, (payload, _) in runs.items():
        gate = payload["gate"]
        summary["runs"][tag] = {
            "config": payload["config"],
            "samples": payload["samples"],
            "wall_clock_s": payload["wall_clock_s"],
            "x_r_over_H": payload["time_averaged"]["x_r_over_H"],
            "x_s_over_H": payload["time_averaged"]["x_s_over_H"],
            "x_r_rel_error": gate["x_r_over_H"]["rel_error"],
            "x_r_pass": gate["x_r_over_H"]["pass"],
            "profile_stations": gate["y_H_0.5_profile"]["stations"],
            "profile_stations_pass": gate["y_H_0.5_profile"]["stations_pass"],
            "profile_worst": gate["y_H_0.5_profile"]["worst"],
            "profile_ok": gate["y_H_0.5_profile"]["ok"],
            "stationary": gate["stationary_window"],
            "reattachment_coverage": gate["reattachment_coverage"],
            "x_r_stationarity": payload["stationarity"]["x_r_over_H"],
            "block_window": payload["stationarity"]["block_window_x_r_over_H"],
            "max_nu_t_ratio_last": payload["series"]["max_nu_t_ratio"][-1],
            "gate_pass": gate["pass"],
        }
    summary["all_pass"] = all(r["gate_pass"] for r in summary["runs"].values())
    (DOCS / "round9_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("x_r_over_H", "x_r_rel_error", "profile_ok",
                                               "stationary", "gate_pass")}
                      for k, v in summary["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())