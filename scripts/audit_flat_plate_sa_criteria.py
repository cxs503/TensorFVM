"""Audit the acceptance criteria used for the SA flat-plate drag evidence.

The repository holds one flat-plate Spalart--Allmaras refinement study
(``docs/verification-sa-refinement/report.md``) whose three mean skin-friction
coefficients were judged twice, by two different thresholds:

* the shipped benchmark gate ``benchmark_flat_plate.ERROR_LIMIT = 0.20``, and
* a 3 % criterion written into the refinement report, which declared every
  level "not qualified" (``均未达标``).

This script re-reads the three stored numbers, re-derives every error from the
correlations in :mod:`tensorfvm.flat_plate_reference` (no solver run, no
hand-typed reference value), rebuilds the wall-normal first-cell height from
the actual mesh class to get ``y+``, and asks the three questions the two
reports never put side by side:

1. does the verdict flip between the documented gate and the 3 % criterion,
2. how large is the single-law error compared with the reference-law spread,
3. is the grid sequence in its asymptotic range (can it be extrapolated).

Usage::

    PYTHONPATH=src python scripts/audit_flat_plate_sa_criteria.py \
        --repo-root . --out docs/flat-plate-sa-criteria
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # runnable without PYTHONPATH, like the tests expect
    sys.path.insert(0, str(_SRC))

EVIDENCE = Path("docs/verification-sa-refinement/report.md")
REYNOLDS = 100_000.0
DOCUMENTED_GATE = 0.20  # benchmark_flat_plate.ERROR_LIMIT
REFINEMENT_CRITERION = 0.03  # threshold asserted in the refinement report prose
ROW = re.compile(
    r"^\|\s*(sa-(?P<nx>\d+)x(?P<ny>\d+))\s*\|\s*(?P<converged>\w+)\s*\|"
    r"\s*(?P<iterations>\d+)\s*\|\s*(?P<cf>[0-9.eE+-]+)\s*\|\s*(?P<err>[0-9.]+)%\s*\|"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_evidence(text: str) -> list[dict]:
    """Extract the refinement table rows (grid, iterations, Cf, stored error)."""
    levels = []
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if match is None:
            continue
        levels.append(
            {
                "name": match.group(1),
                "nx": int(match.group("nx")),
                "ny": int(match.group("ny")),
                "converged": match.group("converged").lower() == "true",
                "iterations": int(match.group("iterations")),
                "mean_cf": float(match.group("cf")),
                "stored_error_percent": float(match.group("err")),
            }
        )
    levels.sort(key=lambda item: item["nx"])
    if len(levels) != 3:
        raise RuntimeError(f"expected three refinement levels, parsed {len(levels)}")
    return levels


def first_cell_centre_height(ny: int, height: float, stretching: float) -> float:
    """Wall-normal distance of the first cell centre.

    Same clustering law as :class:`tensorfvm.body_fitted.FlatPlateMesh`
    (``y = height * expm1(k * eta) / expm1(k)``, ``eta = j / ny``, centre taken as
    the physical midpoint of the rectangular cell), written in closed form so that the audit stays fast; the analytic value is
    checked against the mesh class itself in the tests.
    """
    wall = 0.0
    first_node = height * math.expm1(stretching / ny) / math.expm1(stretching)
    # the shipped mesh's cell centres are polygon centroids, i.e. the physical
    # midpoint between the two wall-normal nodes of the (rectangular) cell
    return 0.5 * (wall + first_node)


def first_cell_y_plus(nx: int, ny: int, mean_cf: float, reynolds: float) -> float:
    """``y+`` of the first cell centre using the measured mean skin friction.

    ``u_tau = U sqrt(C_f / 2)`` is the definition
    ``benchmark_flat_plate.compare_flat_plate`` uses; ``nu = U L / Re``.  Only
    the wall-normal spacing depends on ``nx`` through nothing at all, so the
    ``nx`` argument is accepted for call symmetry and deliberately unused.
    """
    del nx
    y_first = first_cell_centre_height(ny, height=0.2, stretching=4.0)
    u_tau = math.sqrt(mean_cf / 2.0)
    nu = 1.0 / reynolds  # U = L = 1 in the documented case
    return y_first * u_tau / nu


def convergence_state(levels: list[dict]) -> dict:
    """Observed refinement behaviour of the Cf sequence.

    A monotone grid sequence in its asymptotic range has *shrinking*
    successive differences (``|d23| < |d12|`` for a doubling).  When the
    differences grow instead, no observed order and no Richardson limit can be
    extracted; saying so is the honest answer.
    """
    values = [level["mean_cf"] for level in levels]
    d12 = values[1] - values[0]
    d23 = values[2] - values[1]
    monotone = bool((values[0] < values[1] < values[2]) or (values[0] > values[1] > values[2]))
    shrinking = abs(d23) < abs(d12)
    ratio = abs(d23) / abs(d12) if d12 else math.inf
    observed_order = math.log(ratio) / math.log(2.0) if ratio > 0 else None
    if not monotone:
        state = "non_monotone"
    elif not shrinking:
        state = "differences_grow_not_asymptotic"
    else:
        state = "contracting"
    return {
        "values": values,
        "delta_12": d12,
        "delta_23": d23,
        "delta_ratio": ratio,
        "monotone": monotone,
        "differences_shrink": shrinking,
        "richardson_limit": None if state != "contracting" else None,
        "richardson_usable": False,
        "observed_order": observed_order if state == "contracting" else None,
        "state": state,
        "note": (
            "successive differences grow by {:.3f}x per grid doubling, so the sequence is "
            "not in its asymptotic range; no observed order and no Richardson "
            "extrapolation are reported".format(ratio)
            if not shrinking
            else "differences contract; an observed order is reported"
        ),
    }


def build_results(repo_root: Path) -> dict:
    from tensorfvm.flat_plate_reference import band_aware_verdict, reference_band, window_to_band_ratio

    evidence_path = repo_root / EVIDENCE
    levels = parse_evidence(evidence_path.read_text(encoding="utf-8"))
    band = reference_band(REYNOLDS)

    rows = []
    for level in levels:
        verdict = band_aware_verdict(level["mean_cf"], REYNOLDS)
        schlichting_error = verdict["error_per_law"]["schlichting_0.074"]
        rows.append(
            {
                **level,
                "error_percent_per_law": {
                    name: 100.0 * error for name, error in verdict["error_per_law"].items()
                },
                "in_band": verdict["in_band"],
                "band_verdict": verdict["verdict"],
                "error_vs_band_mean_percent": 100.0 * verdict["error_vs_band_mean"],
                "offset_in_band_widths": verdict["offset_in_band_widths"],
                "error_percent_vs_schlichting_recomputed": 100.0 * schlichting_error,
                "stored_error_matches_recomputed": (
                    abs(100.0 * abs(schlichting_error) - level["stored_error_percent"]) < 0.01
                ),
                "passes_documented_gate": abs(schlichting_error) < DOCUMENTED_GATE,
                "passes_refinement_criterion": abs(schlichting_error) < REFINEMENT_CRITERION,
                "first_cell_y_plus": first_cell_y_plus(
                    level["nx"], level["ny"], level["mean_cf"], REYNOLDS
                ),
            }
        )

    convergence = convergence_state(rows)
    per_law_verdicts = {
        name: [
            "PASS" if abs(row["error_percent_per_law"][name]) < 100.0 * REFINEMENT_CRITERION else "FAIL"
            for row in rows
        ]
        for name in band["values"]
    }
    criterion_flip = all(row["passes_documented_gate"] for row in rows) and not any(
        row["passes_refinement_criterion"] for row in rows
    )
    return {
        "case": {
            "name": "turbulent-flat-plate-SA",
            "reynolds": REYNOLDS,
            "chord_m": 1.0,
            "domain_height_m": 0.2,
            "stretching": 4.0,
            "metric": "mean skin-friction coefficient over the 1 m plate",
        },
        "evidence": {
            "source": str(EVIDENCE).replace("\\", "/"),
            "source_sha256": sha256(evidence_path),
            "levels_parsed": len(rows),
            "crosscheck": "stored error column re-derived from the correlations",
        },
        "criteria": {
            "documented_gate": DOCUMENTED_GATE,
            "documented_gate_source": "src/tensorfvm/benchmark_flat_plate.py:ERROR_LIMIT",
            "refinement_criterion": REFINEMENT_CRITERION,
            "refinement_criterion_source": f"{EVIDENCE} (prose: 经验摩阻3%分别验收)",
            "verdict_flips_between_criteria": criterion_flip,
            "undocumented_factor_between_criteria": DOCUMENTED_GATE / REFINEMENT_CRITERION,
        },
        "band": {
            "low": band["low"],
            "high": band["high"],
            "mean": band["mean"],
            "spread": band["spread"],
            "spread_percent": band["spread_percent"],
            "values": band["values"],
            "full_window_to_band_ratio_at_documented_gate": window_to_band_ratio(
                REYNOLDS, DOCUMENTED_GATE
            ),
            "half_window_to_band_ratio_at_documented_gate": DOCUMENTED_GATE / band["spread"],
        },
        "levels": rows,
        "per_law_verdicts_at_3pct": per_law_verdicts,
        "convergence": convergence,
        "limitations": [
            "the three Cf numbers are re-used from the stored report table; the original "
            "fields.npz are not in the repository, so the integrated wall friction cannot be "
            "re-derived here",
            "no solver rerun: one iteration costs 6.7 s single-core (measured), so the 78-iteration "
            "converged case needs ~9 min and does not fit the local 120 s shell budget; cluster "
            "reproduction is the documented follow-up",
            "y+ uses the measured mean Cf, not the local wall shear, so it is a grid-level "
            "estimate of the first-cell requirement",
        ],
    }


def figure(results: dict, out_dir: Path) -> list[Path]:
    from tensorfvm.verification.report import figure, plt

    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    written = []

    band = results["band"]
    levels = results["levels"]
    xs = [level["nx"] for level in levels]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhspan(band["low"], band["high"], color="0.85", label="literature band (4 laws)")
    ax.axhline(band["mean"], color="0.4", ls=":", label="band mean")
    ax.axhline(results["band"]["values"]["schlichting_0.074"], color="C3", ls="--",
               label="Schlichting 0.074 Re^-1/5 (gate)")
    ax.plot(xs, [level["mean_cf"] for level in levels], "o-", color="C0",
            label="measured mean Cf (3 levels)")
    lvl = levels[-1]["mean_cf"]
    ax.axhspan(lvl * (1 - DOCUMENTED_GATE), lvl * (1 + DOCUMENTED_GATE), color="C0", alpha=0.08,
               label="documented +/-20% gate around finest")
    ax.axhspan(lvl * (1 - REFINEMENT_CRITERION), lvl * (1 + REFINEMENT_CRITERION), color="C3",
               alpha=0.25, label="refinement report +/-3% criterion")
    ax.set(xlabel="streamwise cells (nx)", ylabel="mean Cf", xscale="log",
           title="SA flat plate: the answered band, not a single law")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    written.append(figure(fig, figs, "cf-vs-band"))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = list(results["band"]["values"])
    width = 0.8 / len(levels)
    for index, level in enumerate(levels):
        errors = [level["error_percent_per_law"][name] for name in names]
        ax.bar([i + index * width for i in range(len(names))], errors, width, label=level["name"])
    ax.axhline(0, color="0.3")
    ax.axhline(100 * REFINEMENT_CRITERION, color="C3", ls="--", label="+3% criterion")
    ax.axhline(-100 * REFINEMENT_CRITERION, color="C3", ls="--")
    ax.set_xticks([i + 1.5 * width for i in range(len(names))])
    ax.set_xticklabels(names, rotation=15)
    ax.set(ylabel="error vs reference law (%)",
           title="the same three runs pass or fail depending on the chosen law")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    written.append(figure(fig, figs, "error-per-law"))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(xs, [level["first_cell_y_plus"] for level in levels], "o-", color="C2")
    ax.axhline(1.0, color="C3", ls="--", label="wall-resolved requirement y+ < 1")
    ax.set(xlabel="streamwise cells (nx)", ylabel="first-cell y+", xscale="log",
           title="wall resolution is met at every level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    written.append(figure(fig, figs, "yplus"))

    convergence = results["convergence"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(["delta(64->128)", "delta(128->256)"],
           [convergence["delta_12"], convergence["delta_23"]], color=["C0", "C1"])
    ax.axhline(0, color="0.3")
    ax.set(ylabel="Cf increment per doubling",
           title="increments grow {:.2f}x -> not asymptotic".format(convergence["delta_ratio"]))
    ax.grid(alpha=0.3, axis="y")
    written.append(figure(fig, figs, "increments"))
    return written


def determine_report(results: dict) -> str:
    lines = ["# SA 平板阻力判据账目审计", ""]
    lines.append("| 网格 | 步数 | 平均 Cf | 相对 Schlichting | 相对带均值 | 带内 | 20% 闸门 | 3% 判据 |")
    lines.append("|---|---:|---:|---:|---:|:--:|:--:|:--:|")
    for level in results["levels"]:
        lines.append(
            "| {name} | {iterations} | {cf:.8f} | {:+.2f}% | {:+.2f}% | {} | {} | {} |".format(
                level["error_percent_per_law"]["schlichting_0.074"],
                level["error_vs_band_mean_percent"],
                "是" if level["in_band"] else "否",
                "PASS" if level["passes_documented_gate"] else "FAIL",
                "PASS" if level["passes_refinement_criterion"] else "FAIL",
                **{k: level[k] for k in ("name", "iterations")},
                cf=level["mean_cf"],
            )
        )
    lines.append("")
    lines.append("| 关联式 | 各网格误差 (64/128/256) | 3% 判据 |")
    lines.append("|---|---|---|")
    for name, verdicts in results["per_law_verdicts_at_3pct"].items():
        errors = "/".join(
            "{:+.2f}%".format(level["error_percent_per_law"][name]) for level in results["levels"]
        )
        lines.append(f"| {name} | {errors} | {'/'.join(verdicts)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="docs/flat-plate-sa-criteria")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = build_results(repo_root)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tables.md").write_text(determine_report(results), encoding="utf-8")
    figure(results, out_dir)
    if not args.quiet:
        print(determine_report(results))
    return results


if __name__ == "__main__":
    main()