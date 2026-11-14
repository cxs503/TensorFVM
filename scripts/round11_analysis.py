#!/usr/bin/env python3
"""Round 11 analysis: cross-closure figures + machine summary.

Reads the ``round11_<closure>_<tag>.json`` / ``.npz`` artefacts written by
:mod:`scripts.round11_sst_long_average` (SST) and the frozen
``round9_<tag>.json`` / ``.npz`` artefacts (SA baseline), and produces

    docs/case-periodic-hills/round11_summary.json
    docs/case-periodic-hills/fig_round11_<name>.png

Every number is read from the artefacts -- nothing is hand-copied -- so the
report tables can be regenerated from the committed files.
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
from tensorfvm.verification.periodic_hill_round11 import (  # noqa: E402
    closure_comparison,
    gate_summary,
    strict_finite,
)

COLORS = {"sst": "#d62728", "sa": "#1f77b4"}


def load():
    """``{"sst:96x48": payload, "sa:96x48": round-9 payload, ...}``."""
    runs = {}
    for path in sorted(DOCS.glob("round11_*.json")):
        stem = path.stem.replace("round11_", "")
        if stem.startswith(("summary", "comparison")):
            continue
        closure, _, tag = stem.partition("_")
        runs[f"{closure}:{tag}"] = json.loads(path.read_text())
    for path in sorted(DOCS.glob("round9_*.json")):
        tag = path.stem.replace("round9_", "")
        if tag in ("summary", "smoke"):
            continue
        payload = json.loads(path.read_text())
        payload.setdefault("closure", {"name": "sa", "class": "SpalartAllmaras"})
        runs.setdefault(f"sa:{tag}", payload)
    return runs


def npz_path(closure, tag):
    for name in (f"round11_{closure}_{tag}.npz", f"round9_{tag}.npz"):
        path = DOCS / name
        if path.exists():
            return path
    return None


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_profile_y05(runs, out):
    """Time-averaged ``u/U_b`` on ``y/H = 0.5``: SST vs SA vs Breuer 2009."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.array(sorted(BREUER_PROFILE_Y05))
    ys = np.array([BREUER_PROFILE_Y05[x] for x in xs])
    ax.plot(xs, ys, "ko-", ms=5, lw=1.4, label="Breuer et al. 2009 (Re=2800)")
    ax.fill_between(xs, ys * (1 - REFERENCE_TOLERANCE), ys * (1 + REFERENCE_TOLERANCE),
                    color="k", alpha=0.10, label="±10 % band")
    for key, payload in sorted(runs.items()):
        closure, _, tag = key.partition(":")
        if tag != "96x48":
            continue
        profile = payload["horizontal_profile_y_H_0.5"]
        ax.plot(profile["x_over_H"], profile["u_over_Ub"], "--",
                color=COLORS.get(closure, "#777777"), lw=1.8,
                label=f"{closure.upper()} time mean ({tag})")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("x/H")
    ax.set_ylabel("u / U_b")
    ax.set_title("Periodic hills, y/H = 0.5, time-averaged (round 11)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_xr_timeseries(runs, out):
    """Instantaneous ``x_r/H(t)`` and ``max nu_t/nu(t)`` for the 96x48 runs."""
    keys = [k for k in sorted(runs) if k.endswith(":96x48")]
    if not keys:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    ref = REFERENCE["x_r_over_H"]
    axes[0].axhspan(ref * (1 - REFERENCE_TOLERANCE), ref * (1 + REFERENCE_TOLERANCE),
                    color="k", alpha=0.10, label="5.34 ± 10 %")
    axes[0].axhline(ref, color="k", lw=1.0, ls=":")
    for key in keys:
        closure, _, tag = key.partition(":")
        series = runs[key]["series"]
        color = COLORS.get(closure, "#777777")
        axes[0].plot(series["t"], series["x_r_over_H"], color=color, lw=0.7,
                     alpha=0.75, label=f"{closure.upper()} instantaneous")
        axes[1].plot(series["t"], series["max_nu_t_ratio"], color=color, lw=0.9,
                     label=f"{closure.upper()} max $\\nu_t/\\nu$")
    axes[0].set_ylabel("$x_r/H$ (instantaneous)")
    axes[1].set_ylabel("max $\\nu_t/\\nu$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("t  (unit: $H/U_b$)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Periodic-hill time histories at Re_b = 2800 (96x48)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_nu_levels(runs, out, channel_reference=26.1):
    """Closure over-diffusion: window-mean ``max nu_t/nu`` vs the channel level."""
    rows = []
    for key, payload in sorted(runs.items()):
        closure, _, tag = key.partition(":")
        value = (payload.get("time_averaged") or {}).get("max_nu_t_ratio")
        if value is None:
            continue
        rows.append((f"{closure.upper()}\n{tag}", float(value), COLORS.get(closure, "#777")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    ax.axhline(channel_reference, color="k", ls="--", lw=1.0,
               label=f"turbulent channel level {channel_reference:g} (round 7)")
    ax.set_ylabel("window-mean max $\\nu_t/\\nu$")
    ax.set_title("Closure over-diffusion on the periodic-hill case")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_field_mean(key, payload, out):
    """Time-averaged ``u/U_b`` field with the hill body (gray)."""
    closure, _, tag = key.partition(":")
    path = npz_path(closure, tag)
    if path is None:
        return
    data = np.load(path)
    x_over_h = data["x_over_h"]
    yc = data["yc"]
    u = data["mean_u"] / float(payload["config"].get("bulk_velocity", 1.0))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    dx = float(x_over_h[1] - x_over_h[0])
    x_edges = np.append(x_over_h - 0.5 * dx, x_over_h[-1] + 0.5 * dx)
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
    ax.set_title(f"time-averaged $u/U_b$ ({closure.upper()} {tag}, gray = hill body)")
    fig.colorbar(mesh, ax=ax, label="$\\bar u/U_b$")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    runs = load()
    if not runs:
        print("no round-11 / round-9 artefacts found", file=sys.stderr)
        return 1
    table = closure_comparison(runs)
    summary = {"comparison": table, "gate_by_run": {k: gate_summary(v) for k, v in runs.items()}}
    (DOCS / "round11_summary.json").write_text(
        json.dumps(strict_finite(summary), indent=1, allow_nan=False))

    fig_profile_y05(runs, DOCS / "fig_round11_profile_y05.png")
    fig_xr_timeseries(runs, DOCS / "fig_round11_xr_timeseries.png")
    fig_nu_levels(runs, DOCS / "fig_round11_nu_levels.png")
    for key in sorted(runs):
        if key.endswith(":96x48"):
            fig_field_mean(key, runs[key], DOCS / f"fig_round11_field_{key.replace(':', '_')}.png")

    print(json.dumps(table, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())