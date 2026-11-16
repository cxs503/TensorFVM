"""Build the closure dossier + figures for ``CASE-FVM-DAMBREAK-STOKER``.

Writes, under ``src/tensorfvm/verification/artifacts/dambreak-anchor-closure/``:

``dossier.json``        the audit dossier (static part + live re-run)
``reproducibility.json`` the solver re-run at ``n = 800`` vs the artifact
``closure.png``         (a) graded margins on a log scale, (b) the
                        ``front_wetted_diagnostic`` ladder

Every number in ``docs/audit-dambreak-anchor-closure/AUDIT.md`` is read from
these files; nothing is transcribed by hand.

    PYTHONPATH=src python scripts/run_dambreak_anchor_closure.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.verification import dambreak_anchor_closure as closure  # noqa: E402

DEFAULT_OUT = ("src/tensorfvm/verification/artifacts/"
               "dambreak-anchor-closure")


def _graded_rows(dossier: dict) -> list[tuple[str, str, float, bool]]:
    """``(case, gate, value, pass)`` for every graded key with a numeric value."""
    rows = []
    for case, block in dossier["cases"].items():
        verdict = closure.load_json(closure.METRICS_ARTIFACT)["cases"][case]["verdict"]
        flagged = {finding["gate"] for finding in block["diagnostic_graded"]}
        for gate in block["graded_keys"]:
            value = verdict[gate].get("value")
            if not isinstance(value, (int, float)):
                continue
            rows.append((case, gate, float(value), gate not in flagged))
    return rows


def figure(dossier: dict, path: str) -> None:
    tol = float(dossier["tolerance"])
    rows = _graded_rows(dossier)
    demoted_all = [finding for block in dossier["cases"].values()
                   for finding in block.get("demoted_diagnostics", [])]
    floor = 1e-9
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    labels, values, colors = [], [], []
    for case, gate, value, clean in rows:
        labels.append(f"{case}\n{gate}")
        values.append(max(value, floor))
        colors.append("#2b7bba" if clean else "#c0392b")
    order = np.argsort(values)
    ax.bar(range(len(values)),
           [values[i] for i in order], color=[colors[i] for i in order])
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.axhline(tol, color="k", lw=1.6, ls="--", label=f"5 % gate")
    ax.axhline(floor, color="0.7", lw=0.8)
    ax.set_ylabel("graded relative error (log)")
    ax.set_title("(a) what feeds `all_pass`: margins per graded key\n"
                 f"(0 graded diagnostics; {len(demoted_all)} demoted)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    demoted = [finding for block in dossier["cases"].values()
               for finding in block.get("demoted_diagnostics", [])
               if finding.get("ladder")]
    for finding in demoted:
        ns = [n for n, _ in finding["ladder"]]
        vs = [v for _, v in finding["ladder"]]
        ax.plot(ns, vs, "o-", color="#c0392b",
                label=f"{finding['gate']} (= {finding['diagnostic_name']})")
    ax.axhline(tol, color="k", ls="--", lw=1.6, label="5 % gate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("grid n")
    ax.set_ylabel("relative error")
    ax.set_title("(b) published diagnostic, ungraded: the ladder does not\n"
                 "converge (8.75 % -> 1.56 % -> 2.89 %, non-monotone)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    dossier = closure.dossier(reproduce=True)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "dossier.json"), "w", encoding="utf-8") as fh:
        json.dump(dossier, fh, indent=1, sort_keys=False)
    with open(os.path.join(args.out, "reproducibility.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dossier["reproducibility"], fh, indent=1)
    figure(dossier, os.path.join(args.out, "closure.png"))

    print(f"disposition: {dossier['disposition']}")
    print(f"acceptance fully covered: {dossier['acceptance_fully_covered']}")
    for finding in dossier["findings"]:
        print(f"  finding: {finding}")
    print(f"written to {args.out}: dossier.json, reproducibility.json, closure.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())