#!/usr/bin/env python3
"""Draw ``crosscheck.png`` from ``artifacts/reproduction.json``.

Every number in the figure comes from the JSON written by
``verify_reference.py`` -- the figure script does not re-run the solver and does
not contain any transcribed value, so the picture and the tables cannot drift
apart.

Panels
------
(a) Stoker wet-bed depth at ``t* = 2``: numerical profile vs closed form, with
    the four similarity regions shaded and the fixed-fraction probes marked.
(b) Same for the velocity: the plateau ``u2`` and the bore are the two features
    a shallow-water scheme must reproduce here.
(c) Grid convergence in log-log: L1 / L2 / fan-mid probe error vs ``dx`` for both
    bed conditions, with the fitted slopes printed in the legend.
(d) Where the error lives: ``h_num - h_exact`` in similarity coordinates, so the
    kink at the fan head and the captured bore are both visible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "artifacts/reproduction.json"
DEFAULT_OUTPUT = HERE / "crosscheck.png"

REGION_COLOURS = {"upstream": "#eef3fb", "fan": "#e8f4ea",
                  "plateau": "#fdf2e3", "downstream": "#f6ecf4",
                  "dry_bed": "#f4f4f4"}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _panel_profiles(ax_h, ax_u, payload: dict) -> None:
    case = payload["profiles"]["stoker-wet"]
    t_star = case["t_star"]
    b = case["bounds"]
    x = np.asarray(case["x"], dtype=float)
    t = case["t_end"]
    bounds = [("upstream", -np.inf, b["head"]), ("fan", b["head"], b["tail"]),
              ("plateau", b["tail"], b["shock"]), ("downstream", b["shock"], np.inf)]
    for ax in (ax_h, ax_u):
        for name, xi_lo, xi_hi in bounds:
            lo = max(x[0], b.get("x_dam", 0.0) + xi_lo * t) if np.isfinite(xi_lo) else x[0]
            hi = min(x[-1], b.get("x_dam", 0.0) + xi_hi * t) if np.isfinite(xi_hi) else x[-1]
            if hi > lo:
                ax.axvspan(lo, hi, color=REGION_COLOURS[name], zorder=0)
                ax.text(0.5 * (lo + hi), 0.94, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="#555555")
    ax_h.plot(x, case["h_exact"], "k--", lw=1.2, label="exact (Stoker 1957)")
    ax_h.plot(x, case["h_numeric"], color="#1f77b4", lw=1.4,
              label=f"HLL+MUSCL, n={case['n']}")
    run = payload["runs"]["stoker-wet"][str(case["n"])]
    for name, probe in run["metrics"]["probes"].items():
        ax_h.axvline(probe["x_measured"], color="#d62728", lw=0.8, ls=":")
        ax_h.plot([probe["x_measured"]], [probe["level_depth"]], "o", ms=4,
                  color="#d62728", mfc="white", zorder=5)
        ax_h.annotate(name, (probe["x_measured"], probe["level_depth"]),
                      textcoords="offset points", xytext=(3, 6), fontsize=7,
                      color="#d62728")
    ax_h.set_xlabel("x [m]")
    ax_h.set_ylabel("h [m]")
    ax_h.set_title(f"(a) Stoker wet bed, depth at t* = {t_star:.2f} "
                   f"(h0 = 1.0 m, h1 = 0.5 m)", fontsize=9)
    ax_h.legend(fontsize=7, loc="lower right")
    ax_h.set_xlim(-3.0, 3.0)

    ax_u.plot(x, case["u_exact"], "k--", lw=1.2, label="exact")
    ax_u.plot(x, case["u_numeric"], color="#2ca02c", lw=1.4, label="numeric")
    plateau = run["metrics"].get("plateau")
    if plateau:
        ax_u.axhline(plateau["u_exact"], color="#9467bd", lw=0.8, ls="--")
        ax_u.annotate(f"u2 = {plateau['u_exact']:.4f} m/s",
                      (0.02, 0.12), xycoords="axes fraction", fontsize=7,
                      color="#9467bd")
    ax_u.set_xlabel("x [m]")
    ax_u.set_ylabel("u [m/s]")
    ax_u.set_title("(b) Velocity: plateau u2 and the captured bore", fontsize=9)
    ax_u.legend(fontsize=7, loc="upper left")
    ax_u.set_xlim(-3.0, 3.0)


def _panel_convergence(ax, payload: dict) -> None:
    for key, style in (("stoker-wet", {"color": "#1f77b4", "marker": "o"}),
                       ("ritter-dry", {"color": "#d62728", "marker": "s"})):
        series = payload["convergence"][key]
        rows = series["rows"]
        dx = np.array([row["dx"] for row in rows], dtype=float)
        l1 = np.array([row["depth_l1"] for row in rows], dtype=float)
        fan = np.array([row["fan_mid"] for row in rows], dtype=float)
        ax.loglog(dx, l1, "--", lw=1.0, **style,
                  label=f"{key}: L1 (order {series['observed_order_l1']:.2f})")
        ax.loglog(dx, fan, ":", lw=1.0, mfc="white", **style,
                  label=f"{key}: fan-mid (order {series['observed_order_fan_mid']:.2f})")
    first = payload["convergence"]["stoker-wet"]["rows"][0]
    ax.axhline(payload["tolerance"], color="#7f7f7f", lw=0.8, ls="-.")
    ax.annotate(f"tolerance {100 * payload['tolerance']:.0f} %",
                (first["dx"], payload["tolerance"]), textcoords="offset points",
                xytext=(2, 3), fontsize=7, color="#7f7f7f")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error [-]")
    ax.set_title("(c) Grid convergence of the cross-check metrics", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)


def _panel_error_field(ax, payload: dict) -> None:
    for key, colour in (("stoker-wet", "#1f77b4"), ("ritter-dry", "#d62728")):
        case = payload["profiles"][key]
        xi = np.asarray(case["x"], dtype=float) / case["t_end"]
        err = (np.asarray(case["h_numeric"], dtype=float)
               - np.asarray(case["h_exact"], dtype=float))
        ax.plot(xi, err, lw=1.0, color=colour,
                label=f"{key} (n={case['n']})")
        bounds = case["bounds"]
        for boundary in ("head", "tail", "shock"):
            ax.axvline(bounds[boundary], color=colour, lw=0.5, ls=":", alpha=0.7)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(r"similarity coordinate $\xi = (x - x_{dam})/t$ [m/s]")
    ax.set_ylabel(r"$h_{num} - h_{exact}$ [m]")
    ax.set_title("(d) Error field: fan-head kink and captured bore", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    payload = _load(Path(args.input))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    _panel_profiles(axes[0][0], axes[0][1], payload)
    _panel_convergence(axes[1][0], payload)
    _panel_error_field(axes[1][1], payload)

    reference = payload["reference"]
    worst = reference["worst_relative"]
    # Judge on the finest grid drawn in the panes (the same record as panel (a)).
    check = max(payload["runs"]["stoker-wet"].values(),
                key=lambda record: record["discretisation"]["n"])
    verdict = check["verdict"]
    fig.suptitle(
        "CASE-FVM-DAMBREAK-STOKER — analytic cross-check "
        f"(reference worst residual {worst:.1e}, gate {100 * payload['tolerance']:.0f} %: "
        f"{'PASS' if verdict['pass'] else 'FAIL'} at n={check['discretisation']['n']})",
        fontsize=10)
    fig.savefig(args.output, dpi=140)
    size_kb = Path(args.output).stat().st_size / 1024.0
    print(f"wrote {args.output} ({size_kb:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())