#!/usr/bin/env python3
"""Robustness sweep of the ``CASE-FVM-DAMBREAK-STOKER`` graded numbers.

Runs :func:`tensorfvm.shallow_water.robustness.run_sweep` and writes

* ``artifacts/robustness.json`` -- every graded number, every threshold and the
  verdict, so that the report quotes no hand-typed value, and
* ``robustness.png`` -- the figure of the report,

plus the markdown tables between the ``<!-- BEGIN SWEEP-TABLES -->`` and
``<!-- END SWEEP-TABLES -->`` markers of ``REPORT.md``.

Usage::

    PYTHONPATH=src python docs/case-fvm-dambreak-stoker-robustness/run_robustness_sweep.py
    PYTHONPATH=src python docs/case-fvm-dambreak-stoker-robustness/run_robustness_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.shallow_water.robustness import (  # noqa: E402
    DEPTH_RATIO_MAX_ERROR,
    VARIANTS,
    run_sweep,
    sweep_verdict,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

_LABEL = {
    "baseline": "baseline",
    "translated_quarter": "translated dx/4",
    "translated_half": "translated dx/2",
    "tight_downstream": "boundary at 8 cells",
    "tight_dx_rounding": "control: dx rounded",
    "wide_domain": "domain 2x (same dx)",
    "similarity_x2": "similarity x2",
    "depth_ratio_0.2": "h1/h0 = 0.2",
    "depth_ratio_0.8": "h1/h0 = 0.8",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover - only outside a checkout
        return "unknown"


def _figure(sweep: dict, verdict: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, base in zip(axes[:2], ("ritter-dry", "stoker-wet")):
        for i, variant in enumerate(VARIANTS):
            rows = sorted((r for r in sweep["runs"]
                           if r["base"] == base and r["variant"] == variant),
                          key=lambda r: r["n"])
            if not rows:
                continue
            xs = [r["graded"]["dx"] for r in rows]
            ys = [100.0 * r["graded"]["headline"] for r in rows]
            marker = "o" if variant == "baseline" else "s"
            ls = "-" if variant == "baseline" else "--"
            ax.loglog(xs, ys, ls, marker=marker, ms=4, lw=1.3,
                      color=colors[i % 10], label=_LABEL[variant])
        ax.axhline(5.0, color="k", ls=":", lw=1.0)
        ax.text(0.02, 0.93, "acceptance 5 %", transform=ax.transAxes,
                fontsize=8, color="k")
        ax.set_xlabel("dx [m]")
        ax.set_ylabel("graded headline error [%]")
        ax.set_title(f"{base}: headline error under the sweep")
        ax.grid(alpha=0.3, which="both")
        if ax.get_legend() is None:
            ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    keys = [k for k in verdict["checks"] if k.endswith(("_quarter", "_half",
                                                        "tight_downstream",
                                                        "wide_domain"))]
    labels, values, limits = [], [], []
    for key in keys:
        check = verdict["checks"][key]
        labels.append(key.replace("/", "\n"))
        values.append(abs(check["worst_delta_pp"]))
        limits.append(check["limit_pp"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, values, 0.4, label="worst |delta| [pp]", color="#4c78a8")
    ax.bar(x + 0.2, limits, 0.4, label="predeclared limit [pp]",
           color="#b0b0b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("percentage points of the 5 % budget")
    ax.set_title("perturbation debt vs predeclared budgets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("CASE-FVM-DAMBREAK-STOKER robustness envelope "
                 "(Ritter/Stoker, t* = 2)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _tables(sweep: dict, verdict: dict) -> str:
    lines: list[str] = []
    for base in ("ritter-dry", "stoker-wet"):
        lines.append(f"### {base}\n")
        lines.append("| variant | dx [m] | n | graded headline | vs baseline [pp] "
                     "| L1 depth | extrema (# / amp·h0⁻¹) | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for variant in VARIANTS:
            rows = sorted((r for r in sweep["runs"]
                           if r["base"] == base and r["variant"] == variant),
                          key=lambda r: r["n"])
            for r in rows:
                g = r["graded"]
                delta = ""
                if r["ref_n"] is not None:
                    ref = next((q for q in sweep["runs"]
                                if q["base"] == base and q["n"] == r["ref_n"]
                                and q["variant"] == "baseline"), None)
                    if ref is not None:
                        delta = (f"{100.0 * (g['headline'] - ref['graded']['headline']):+.4f}")
                lines.append(
                    f"| {_LABEL[variant]} | {g['dx']:.5f} | {r['n']} "
                    f"| {100.0 * g['headline']:.4f} % | {delta} "
                    f"| {100.0 * g['depth_l1']:.4f} % "
                    f"| {g['local_extrema']} / {g['extrema_amplitude_over_h0']:.2e} "
                    f"| {'PASS' if g['all_pass'] else 'FAIL'} |")
        lines.append("")

    lines.append("### Predeclared thresholds (applied to the headline error)\n")
    lines.append("| check | worst measured | limit | verdict |")
    lines.append("|---|---|---|---|")
    for key, check in verdict["checks"].items():
        if "worst_delta_pp" in check:
            measured = f"{check['worst_delta_pp']:+.4f} pp (n = {check['worst_n']})"
            limit = f"{check['limit_pp']:g} pp"
        elif "max_abs_delta_pp" in check:
            measured = (f"{check['max_abs_delta_pp']:.4f} pp @ dx off "
                        f"{100.0 * check['max_dx_rel']:.3f} % (control, ungated)")
            limit = "—"
        elif "worst_rel" in check:
            measured = f"{check['worst_rel']:.3e} rel (n = {check['worst_n']})"
            limit = f"{check['limit_rel']:g} rel"
        else:
            measured = f"{100.0 * check['worst_error']:.4f} % ({check['worst_variant']})"
            limit = f"{100.0 * check['limit']:.0f} %"
        lines.append(f"| {key} | {measured} | {limit} "
                     f"| {'PASS' if check['pass'] else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="only n = 200, 400 (smoke run)")
    parser.add_argument("--no-inject", action="store_true",
                        help="do not rewrite the report tables")
    args = parser.parse_args(argv)

    ns = (200, 400) if args.quick else (400, 800, 1600)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    sweep = run_sweep(ns=ns)
    verdict = sweep_verdict(sweep)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "tolerance": DEPTH_RATIO_MAX_ERROR,
        },
        "sweep": sweep,
        "verdict": verdict,
    }
    out = ARTIFACTS / "robustness.json"
    out.write_text(json.dumps(payload, indent=1))
    _figure(sweep, verdict, HERE / "robustness.png")

    if not args.no_inject:
        report = HERE / "REPORT.md"
        text = report.read_text()
        begin, end = ("<!-- BEGIN SWEEP-TABLES -->", "<!-- END SWEEP-TABLES -->")
        if begin in text and end in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            report.write_text(head + begin + "\n\n" + _tables(sweep, verdict)
                              + "\n" + end + tail)

    print(json.dumps({"artifact": str(out),
                      "all_pass": verdict["all_pass"],
                      "checks": {k: v["pass"] for k, v in verdict["checks"].items()}},
                     indent=1))
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())