"""Audit figure for m15310 (REPRO-FVM-SOD1978).

Reads the *committed* round-1 artifact JSON (the only versioned run artifact of
this anchor) and draws the measured convergence orders against the benchmark
criterion that the mission has to satisfy:

* panel A: whole-field L1 order of the Sod solution (order 1 / order 2 schemes)
  versus the benchmark floor ``C3 >= 1.0 (literal, unchanged)``;
* panel B: design order on the smooth periodic entropy wave versus the declared
  bands (round-1 one-sided 2 % / round-2 one-sided 3 %), which is where the
  round-1 and round-2 verdicts disagree.

No number is typed by hand: every value comes from ``--artifact``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pairs(orders):
    return [f"{a}-{b}" for a, b in zip((100, 200, 400), (200, 400, 800))][: len(orders)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    art = Path(args.artifact)
    raw = art.read_bytes()
    data = json.loads(raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sod = data["l1_orders_sod"]
    smooth = data["smooth_wave_orders"]
    c3_floor = data["thresholds"]["sod_l1_order_min"]
    band_r1 = data["thresholds"]["smooth_order_design_band"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = range(len(sod["order1"]))
    ax.plot(list(x), sod["order1"], "o-", label="order-1 scheme, Sod L1 rate")
    ax.plot(list(x), sod["order2"], "s-", label="order-2 scheme, Sod L1 rate")
    ax.axhline(c3_floor, color="crimson", ls="--",
               label=f"C3 benchmark floor = {c3_floor}")
    for series, name in ((sod["order1"], "o1"), (sod["order2"], "o2")):
        for i, v in enumerate(series):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_pairs(sod["order1"]))
    ax.set_xlabel("grid pair nx_i -> nx_{i+1}")
    ax.set_ylabel("observed L1 order")
    ax.set_title("A. Sod whole-field L1 order vs C3 floor (FAIL)")
    ax.set_ylim(0.4, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    x = range(len(smooth["order2"]))
    ax.plot(list(x), smooth["order1"], "o-", label="order-1 scheme, entropy wave")
    ax.plot(list(x), smooth["order2"], "s-", label="order-2 scheme, entropy wave")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.axhline(2.0, color="grey", ls=":", lw=1)
    ax.axhline(2.0 * (1 - band_r1), color="darkorange", ls="--", lw=1,
               label=f"design band floor 2*(1-{band_r1})")
    for i, v in enumerate(smooth["order2"]):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_pairs(smooth["order1"]))
    ax.set_xlabel("grid pair nx_i -> nx_{i+1}")
    ax.set_ylabel("observed design order")
    ax.set_title("B. smooth entropy-wave design order (window-dependent verdict)")
    ax.set_ylim(0.8, 2.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("m15310 audit: Sod(1978) t=0.14 reproduction, numbers from "
                 "docs/repro-fvm-sod1978/artifacts/sod1978_t0.14.json", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out / "m15310_audit_orders_vs_criterion.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    summary = {
        "artifact": str(art),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "figure": str(png),
        "figure_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "c3_floor": c3_floor,
        "sod_l1_orders": sod,
        "smooth_wave_orders": smooth,
        "verdicts": data["verdicts"],
    }
    (out / "figure_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("artifact_sha256", "figure", "figure_sha256")},
                     indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())