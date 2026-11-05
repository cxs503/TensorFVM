"""Audit the transcribed Ghia et al. (1982) tables against the solver ladder.

The analysis lives in :mod:`tensorfvm.verification.ghia_reference_audit`; this
script reads the committed three-grid field bundle, runs it, and writes the
report bundle (numbers + two figures):

  * ``results.json``      - every number quoted in the report, machine generated;
  * ``profile.png``       - Re = 400 v-centreline: table vs the three grids;
  * ``convergence.png``   - per-point residual vs n for all three Reynolds.

Usage::

    PYTHONPATH=src python3 scripts/audit_ghia_reference_consistency.py \
        --bundle docs/verification-lid-cavity-ladder/summary.json \
        --out docs/verification-ghia-reference-consistency
"""

from __future__ import annotations

import argparse
import json
import os

from tensorfvm.verification.ghia_reference_audit import (
    ACCEPTANCE_BAND,
    audit_bundle,
)


def make_figures(report: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    # ---- per-point residual vs grid -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, which in zip(axes, ("u", "v")):
        for re_key, block in report["reynolds"].items():
            ns = block["grids"]
            rows = block["which"][which]["points"]
            suspect = {
                (f["reynolds"], f["which"], f["x"])
                for f in report["verdict"]["reference_defects"]
            }
            for row in rows:
                y = [abs(row["delta"][str(n)]) for n in ns]
                hot = (float(re_key), which, row["x"]) in suspect
                ax.plot(
                    ns,
                    y,
                    marker="o",
                    ms=3,
                    lw=2.2 if hot else 0.8,
                    color="crimson" if hot else "0.65",
                    zorder=5 if hot else 1,
                    alpha=1.0 if hot else 0.8,
                )
                if hot:
                    ratio = row["ratio_prev_over_finest"]
                    ax.annotate(
                        f"Re={float(re_key):g}, x={row['x']:.4f}\n"
                        f"$|\\delta|$={y[-1]:.3f}, ratio="
                        + ("n/a" if ratio is None else f"{ratio:.2f}"),
                        (ns[-1], y[-1]),
                        textcoords="offset points",
                        xytext=(-96, 6),
                        fontsize=8,
                        color="crimson",
                    )
        ax.axhline(
            ACCEPTANCE_BAND, color="k", ls="--", lw=1, label="acceptance band 0.03"
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(
            sorted({n for b in report["reynolds"].values() for n in b["grids"]})
        )
        ax.set_xticklabels([str(t) for t in ax.get_xticks()])
        ax.set_xlabel("grid n")
        ax.set_title(f"{which}-centreline residual vs the published table")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("$|f_{computed} - f_{table}|$  (lid speed)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Consistent points converge to the table; a reference defect saturates",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    # ---- Re=400 v profile ---------------------------------------------------
    block = report["reynolds"]["400.0"]
    rows = block["which"]["v"]["points"]
    xr = [r["x"] for r in rows]
    ref = [r["reference"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(xr, ref, "ks-", ms=5, lw=1.0, label="Ghia Table II as digitised")
    for n, style in zip(block["grids"], ("-", "--", "-.")):
        ax.plot(
            xr,
            [r["computed"][str(n)] for r in rows],
            style,
            marker="+",
            ms=6,
            lw=1.0,
            label=f"solver n={n}",
        )
    bad = report["verdict"]["reference_defects"]
    for f in bad:
        if f["reynolds"] == 400.0 and f["which"] == "v":
            ratio = f["ratio_prev_over_finest"]
            ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
            ax.annotate(
                "table point off by "
                f"{abs(f['delta_finest']):.3f}\n(ratio {ratio_text}"
                " -> not refinement)",
                (f["x"], f["reference"]),
                textcoords="offset points",
                xytext=(10, 34),
                fontsize=8,
                color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
            )
            ax.plot(
                [f["x"]],
                [f["reference"]],
                "o",
                ms=9,
                mfc="none",
                mec="crimson",
                mew=1.6,
            )
    ax.set_xlabel("x/L  (y/L = 0.5)")
    ax.set_ylabel("v/U")
    ax.set_title("Re = 400 v-centreline: all grids fall on one smooth curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "profile.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle", default="docs/verification-lid-cavity-ladder/summary.json"
    )
    ap.add_argument("--out", default="docs/verification-ghia-reference-consistency")
    args = ap.parse_args()

    with open(args.bundle) as handle:
        bundle = json.load(handle)
    report = audit_bundle(bundle)
    os.makedirs(args.out, exist_ok=True)
    report["figures"] = [os.path.basename(p) for p in make_figures(report, args.out)]
    with open(os.path.join(args.out, "results.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)

    defects = report["verdict"]["reference_defects"]
    print(
        f"findings: {report['verdict']['n_findings']}, reference defects: {len(defects)}"
    )
    for f in defects:
        print(
            f"  Re={f['reynolds']:g} {f['which']} x={f['x']:.4f}: table {f['reference']:.5f} "
            f"computed {f['computed_finest']:.5f} delta {f['delta_finest']:+.5f} "
            f"ratio {f['ratio_prev_over_finest']:.2f} "
            f"residual/uncertainty {f['excess_factor']:.1f}"
        )
    print(f"wrote {args.out}/results.json and {report['figures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
