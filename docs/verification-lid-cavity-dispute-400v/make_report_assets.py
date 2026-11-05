#!/usr/bin/env python3
"""Regenerate the evidence assets for the Re=400 v-centreline dispute note.

Reads the committed ladder bundle (``docs/verification-lid-cavity-ladder/summary.json``)
through the same helpers the attribution audit uses, so the numbers quoted in
``report.md`` are never hand-copied:

    python3 docs/verification-lid-cavity-dispute-400v/make_report_assets.py

Outputs, written next to this script:

* ``numbers.json`` -- sampled computed values per grid, the published value, the
  observed order and Richardson limit of the disputed point;
* ``re400-v-profile.png`` -- published profile, solver ladder, disputed entry.
"""

from __future__ import annotations

import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from tensorfvm.lid_cavity_reference import DISPUTES, table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
AUDIT = os.path.join(REPO, "scripts", "audit_ghia_re400_v_attribution.py")
BUNDLE = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")

REYNOLDS, WHICH = 400.0, "v"
DISPUTED_X = 0.9063
NEIGHBOURS = (0.8047, 0.8594, 0.9453)


def _audit_module():
    spec = importlib.util.spec_from_file_location("ghia_attribution_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build():
    audit = _audit_module()
    bundle = audit.load_bundle(BUNDLE)
    rows = audit.pointwise(audit.cases_by_grid(bundle), REYNOLDS, WHICH)
    grids = sorted(rows[DISPUTED_X]["grid"])
    converged = audit.observed_order(rows[DISPUTED_X]["grid"])
    record = DISPUTES[(REYNOLDS, WHICH, DISPUTED_X)]
    published = {x: value for x, value in table(REYNOLDS, WHICH)}
    numbers = {
        "reynolds": REYNOLDS,
        "component": WHICH,
        "grids": grids,
        "published": published,
        "computed": {f"{x:.4f}": {str(n): rows[x]["grid"][n] for n in grids}
                     for x in sorted(rows) if x in (DISPUTED_X, *NEIGHBOURS)},
        "gaps": {f"{x:.4f}": {str(n): rows[x]["grid"][n] - published[x] for n in grids}
                 for x in sorted(rows) if x in (DISPUTED_X, *NEIGHBOURS)},
        "observed_order": converged,
        "register": {k: v for k, v in record.items() if k != "computed_n"},
        "register_computed_n": record["computed_n"],
        "fabricated_value": -0.38760,
        "fabricated_minus_limit": -0.38760 - float(record["richardson_limit"]),
        "fabricated_minus_n128": -0.38760 - float(record["computed_n"][max(grids)]),
    }
    return numbers, rows, published


def figure(numbers, rows, published):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    xs = [x for x, _ in sorted(published.items())]
    ys = [y for _, y in sorted(published.items())]
    ax.plot(xs, ys, "k-o", ms=3.5, lw=1.2,
            label="Ghia (1982) Table II, as printed")
    for n in numbers["grids"]:
        ax.plot([x for x in sorted(rows)], [rows[x]["grid"][n] for x in sorted(rows)],
                "--", lw=1.0, label=f"TensorFVM n={n}")
    ax.plot([DISPUTED_X], [published[DISPUTED_X]], "r*", ms=15,
            label="disputed entry -0.23827")
    ax.plot([DISPUTED_X], [numbers["fabricated_value"]], "x", ms=10, color="darkorange",
            label="PR #127 fabricated -0.38760")
    ax.set_xlabel("y / L")
    ax.set_ylabel("v / U_lid")
    ax.set_title("Re = 400, v-centreline (x = 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    window = sorted([DISPUTED_X, *NEIGHBOURS])
    for x in window:
        ax.semilogx(numbers["grids"], [rows[x]["grid"][n] for n in numbers["grids"]],
                    "-o", ms=3.5, label=f"x={x:.4f} computed")
        ax.axhline(published[x], lw=0.8, ls=":", color=ax.lines[-1].get_color())
    ax.set_xlabel("grid n")
    ax.set_ylabel("v / U_lid")
    ax.set_title("ladder values (dotted = printed reference)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(HERE, "re400-v-profile.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    numbers, rows, published = build()
    with open(os.path.join(HERE, "numbers.json"), "w") as handle:
        json.dump(numbers, handle, indent=2, sort_keys=True)
        handle.write("\n")
    figure(numbers, rows, published)
    print(json.dumps({k: numbers[k] for k in
                      ("observed_order", "fabricated_minus_limit", "fabricated_minus_n128")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())