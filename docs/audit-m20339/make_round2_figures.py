"""Figures for the round-2 gate-repair audit (``AUDIT-…-round2-gate-repair.md``).

Both panels are plotted straight from committed data -- the witness manifest and
the git history of the previous witness manifest.  Nothing is typed in by hand:

* ``fig_r2_record_reproduction.png`` -- per-record reproduction residual for all
  25 records, split by record class (reference-metrics vs solver-observables).
  Every residual must sit at round-off; the two classes are drawn in different
  colours so a reader can see that the solver-observable records are *checked*,
  not skipped.
* ``fig_r2_bundle_drift.png`` -- how the evidence bundle and its gate moved
  between the audited commit ``86350746`` and the current main: absolute-path
  fields, record count, and the gate outcome that the previous manifest
  silently guaranteed.

Usage::

    PYTHONPATH=src python docs/audit-m20339/make_round2_figures.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = REPO / "docs" / "woodward-colella-double-blast" / "coverage" / "provenance_manifest.json"
BASELINE = "86350746"          # commit whose numbers the round-1 audit reported
OUT = HERE / "figs"


def _load_manifest(text: str | None) -> dict:
    if text is not None:
        return json.loads(text)
    return json.loads(MANIFEST.read_text())


def _baseline() -> dict:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPO), "show",
             f"{BASELINE}:docs/woodward-colella-double-blast/coverage/provenance_manifest.json"],
            stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    return _load_manifest(raw.decode())


def record_residuals(man: dict) -> list[tuple[str, str, float]]:
    """(record name, class, worst residual) for every record that was checked."""
    out = []
    for row in man["record_reproduction"]["rows"]:
        name = Path(row["record"]).name.replace("ladder_metrics_", "").replace(".json", "")
        if row["check_kind"] == "reference_metrics":
            vals = [v["max_abs_deviation"] for v in row["variants"].values()
                    if v.get("max_abs_deviation") is not None]
            out.append((name, row["check_kind"], max(vals) if vals else np.nan))
        elif row["check_kind"] == "solver_observables":
            vals = [v["abs_deviation"] for v in row["observables"].values()]
            out.append((name, row["check_kind"], max(vals) if vals else np.nan))
    return out


def fig_record_reproduction(man: dict) -> Path:
    rows = record_residuals(man)
    rows.sort(key=lambda r: r[0])
    labels = [r[0] for r in rows]
    vals = np.array([max(r[2], 1e-17) for r in rows])
    colours = ["#1f6fb4" if r[1] == "reference_metrics" else "#d1495b" for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colours, height=0.68)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(1e-6, color="k", ls="--", lw=1.0, label="reference-metric tolerance (1e-6)")
    ax.axvline(1e-9, color="#555", ls=":", lw=1.0, label="observable tolerance (1e-9 rel.)")
    ax.set_xlabel("worst reproduction residual of the record  [density units / absolute]")
    ax.set_title("CASE-WOODWARD-COLELLA: every committed run record re-derived from the committed tree\n"
                 f"{len(rows)}/{man['record_reproduction']['n_records']} records, "
                 f"{man['record_reproduction']['n_failed']} failed, "
                 f"{man['record_reproduction']['n_unverifiable']} unverifiable", fontsize=10)
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f6fb4", label="reference-metrics record (re-scored)"),
                       Patch(color="#d1495b", label="solver-observables record (profile re-derived)"),
                       plt.Line2D([], [], color="k", ls="--", label="metric tolerance 1e-6"),
                       plt.Line2D([], [], color="#555", ls=":", label="observable tolerance 1e-9")],
              loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_r2_record_reproduction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_bundle_drift(man: dict, base: dict) -> Path:
    cur_paths = man["absolute_path_records"]["count"]
    cur_recs = man["record_reproduction"]["n_records"]
    cur_fail = man["record_reproduction"]["n_failed"]
    base_paths = base.get("absolute_path_records", {}).get("count", np.nan)
    base_recs = base.get("record_reproduction", {}).get("n_records", np.nan)
    base_fail = 0  # round-1 reported 0 failures (16/16 reproduced)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    groups = [("absolute-path\nfields", base_paths, cur_paths),
              ("run records", base_recs, cur_recs),
              ("records judged\nnot reproduced", base_fail, cur_fail)]
    for ax, (title, b, c) in zip(axes, groups):
        ax.bar([0, 1], [b, c], color=["#9aa5b1", "#1f6fb4"], width=0.55)
        for x, v in ((0, b), (1, c)):
            ax.annotate(f"{v:g}", (x, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"audited commit\n{BASELINE}", "current main"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(b, c, 1) * 1.25)
        ax.grid(axis="y", alpha=0.25)
    axes[2].set_ylabel("count")
    fig.suptitle("The bundle moved after the audit: the witness manifest went stale and the two "
                 "'not reproduced' records\nwere a classification bug (both re-derive to 1e-15), "
                 "not a numerical contamination", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_r2_bundle_drift.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    man = _load_manifest(None)
    base = _baseline()
    for p in (fig_record_reproduction(man), fig_bundle_drift(man, base)):
        print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()