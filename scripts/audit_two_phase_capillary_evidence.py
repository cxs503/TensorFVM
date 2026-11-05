#!/usr/bin/env python3
"""Regenerate the two-phase CSF capillary evidence and test the published record.

Why this exists (verification-debt refresh)
-------------------------------------------
``docs/two-phase-surface-tension-audit/report.md`` records a Young--Laplace run
whose numbers are physically impossible: ``dP = -1.18714e+22 Pa`` against an
exact ``sigma/R = 0.35 Pa``, ``|u|max = 4.47e+20 m/s`` and ``div_final =
4.26e+22 1/s``.  That record is the only evidence for the capillary chain's
verdict ("all criteria passed: False"), so a reader cannot tell whether the
per-face criterion failed because the error is 10 % or because the run diverged
by 22 orders of magnitude.

This script answers that question mechanically:

1. re-runs the *same* study (:mod:`audit_two_phase_surface_tension` functions)
   on the current source at the grids the published record used (24, 32) and at
   the grids the current script defaults to (32, 48, 64);
2. classifies each published number as reproducible / not reproducible with an
   explicit tolerance, instead of silently overwriting the old record;
3. writes the fresh evidence, the figure and the machine record.

Nothing is deleted: the historical FAIL record stays in the repository and the
new report links to it.  Usage::

    PYTHONPATH=src python scripts/audit_two_phase_capillary_evidence.py \
        --output docs/two-phase-capillary-evidence-refresh
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "audit_two_phase_surface_tension.py"

#: Numbers copied verbatim from docs/two-phase-surface-tension-audit/summary.json
PUBLISHED = {
    "grids": [24, 32],
    "n24": dict(
        pressure_jump_pa=-1.18714e22,
        relative_error=-3.391822194870826e22,
        umax_m_s=4.47e20,
        capillary_number=6.39e19,
        final_divergence=4.26e22,
        kappa_relative_error=0.0032,
    ),
    "n32": dict(
        pressure_jump_pa=8.49545e23,
        relative_error=2.4272702330011291e24,
        umax_m_s=1.45e23,
        capillary_number=2.07e22,
        final_divergence=1.85e25,
        kappa_relative_error=0.1005,
    ),
}

RADIUS = 0.2
SIGMA = 0.07
MU = 0.01
DT = 1e-5
STEPS = 10
EXACT_JUMP = SIGMA / RADIUS
TOLERANCE = 3e-2  # 3 %: the platform evidence bar


def load_audit():
    spec = importlib.util.spec_from_file_location("_st_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reproduce(module, n):
    curvature = module.curvature_study(n, RADIUS)
    force = module.force_balance(n, RADIUS, SIGMA)
    laplace = module.laplace_study(n, RADIUS, SIGMA, MU, STEPS, DT)
    return dict(curvature=curvature, force=force, laplace=laplace)


def classify(published, fresh):
    keys = dict(
        pressure_jump_pa="pressure_jump_pa",
        relative_error="relative_error",
        umax_m_s="umax_m_s",
        capillary_number="capillary_number",
        final_divergence="final_divergence",
    )
    rows = {}
    for label, key in keys.items():
        old = float(published[key])
        new = fresh["laplace"][key]
        scale = max(abs(old), abs(new), 1e-300)
        rows[label] = dict(
            published=old,
            regenerated=new,
            magnitude_ratio=abs(old) / max(abs(new), 1e-300),
            agrees_within_tolerance=bool(abs(old - new) <= TOLERANCE * scale),
        )
    old_kappa = float(published["kappa_relative_error"])
    new_kappa = fresh["curvature"]["relative_error"]
    rows["kappa_relative_error"] = dict(
        published=old_kappa,
        regenerated=new_kappa,
        magnitude_ratio=abs(old_kappa) / max(abs(new_kappa), 1e-300),
        agrees_within_tolerance=bool(
            abs(old_kappa - new_kappa)
            <= TOLERANCE * max(abs(old_kappa), abs(new_kappa), 1e-300)
        ),
    )
    return rows


def figures(directory, runs, comparisons):
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ns = [r["n"] for r in runs]
    constraints = [r["laplace"]["capillary_number"] for r in runs]

    ax = axes[0]
    dp = [r["laplace"]["pressure_jump_pa"] for r in runs]
    ax.plot(ns, dp, "o-", label="regenerated (current source)")
    ax.axhline(EXACT_JUMP, color="k", ls="--", lw=1, label="exact $\\sigma/R$")
    ax.fill_between(
        [min(ns) - 4, max(ns) + 4],
        EXACT_JUMP * 0.97,
        EXACT_JUMP * 1.03,
        color="tab:green",
        alpha=0.15,
        label="$\\pm3\\%$ band",
    )
    published_n = [PUBLISHED["grids"][0], PUBLISHED["grids"][1]]
    published_dp = [
        PUBLISHED["n24"]["pressure_jump_pa"],
        PUBLISHED["n32"]["pressure_jump_pa"],
    ]
    ax.plot(published_n, published_dp, "rx", ms=9, label="published record (stale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("n (cells per side)")
    ax.set_ylabel("Young–Laplace jump $\\Delta p$ [Pa]")
    ax.set_title("(a) pressure jump: record vs regeneration")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    kappa = [100 * r["curvature"]["relative_error"] for r in runs]
    ax.plot(ns, kappa, "o-", color="tab:purple", label="band-mean $\\kappa$ error")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-3, 3, color="tab:green", alpha=0.15, label="$\\pm3\\%$")
    ax.set_xlabel("n")
    ax.set_ylabel("curvature error [%]")
    ax.set_title("(b) discrete curvature of the exact circle ($\\kappa=1/R$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.semilogy(ns, constraints, "o-", color="tab:red", label="capillary number Ca")
    ax.axhline(1e-2, color="k", ls="--", lw=1, label="Ca gate $10^{-2}$")
    ax.set_xlabel("n")
    ax.set_ylabel("Ca $=\\mu |u|_{max}/\\sigma$")
    ax.set_title("(c) spurious-current magnitude stays quiet")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    figure.suptitle(
        "Two-phase CSF capillary chain: published record vs regenerated evidence (m3646)"
    )
    figure.tight_layout()
    figure.savefig(directory / "figure.png", dpi=140)
    plt.close(figure)

    # second figure: the magnitude gap that made the published record unreadable
    figure, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = ["$\\Delta p$", "$|u|_{max}$", "$Ca$", "$div$"]
    keys = ["pressure_jump_pa", "umax_m_s", "capillary_number", "final_divergence"]
    fresh32 = comparisons[str(32)]
    published = [abs(float(PUBLISHED["n32"][k])) for k in keys]
    regenerated = [abs(float(fresh32[k]["regenerated"])) for k in keys]
    x = range(len(labels))
    ax.bar(
        [i - 0.2 for i in x],
        published,
        width=0.4,
        label="published record (n=32)",
        color="tab:red",
    )
    ax.bar(
        [i + 0.2 for i in x],
        regenerated,
        width=0.4,
        label="regenerated (n=32)",
        color="tab:blue",
    )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title("Same study, same grid: 20+ orders of magnitude apart")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for i, (old, new) in enumerate(zip(published, regenerated)):
        ax.text(
            i - 0.2,
            old,
            f"{old:.1e}",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
        )
        ax.text(
            i + 0.2,
            new,
            f"{new:.1e}",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
        )
    figure.tight_layout()
    figure.savefig(directory / "magnitude-gap.png", dpi=140)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "two-phase-capillary-evidence-refresh",
    )
    parser.add_argument("--grids", type=int, nargs="+", default=[24, 32, 48, 64])
    options = parser.parse_args(argv)
    torch.set_num_threads(1)
    module = load_audit()
    options.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    runs = []
    for n in options.grids:
        fresh = reproduce(module, n)
        runs.append(dict(n=n, **fresh))
        print(
            f"[n={n}] dP={fresh['laplace']['pressure_jump_pa']:.6f} Pa "
            f"(rel {100 * fresh['laplace']['relative_error']:+.2f}%) "
            f"kappa rel {100 * fresh['curvature']['relative_error']:+.2f}% "
            f"Ca={fresh['laplace']['capillary_number']:.3e} "
            f"finite={fresh['laplace']['finite']}"
        )
    elapsed = time.time() - started

    published_cases = {24: PUBLISHED["n24"], 32: PUBLISHED["n32"]}
    comparisons = {}
    for run in runs:
        published = published_cases.get(run["n"])
        if published is None:
            continue
        comparisons[str(run["n"])] = classify(published, run)

    summary = dict(
        mission=3646,
        parameters=dict(
            radius_m=RADIUS,
            sigma_n_m=SIGMA,
            mu_pa_s=MU,
            dt_s=DT,
            steps=STEPS,
            exact_jump_pa=EXACT_JUMP,
        ),
        published_record=dict(
            path="docs/two-phase-surface-tension-audit/summary.json",
            grids=PUBLISHED["grids"],
            values=PUBLISHED,
        ),
        regenerated=[
            dict(
                n=r["n"],
                curvature=r["curvature"],
                force=r["force"],
                laplace={k: v for k, v in r["laplace"].items() if k != "history"},
            )
            for r in runs
        ],
        comparisons=comparisons,
        toolchain=dict(torch=torch.__version__, seconds=round(elapsed, 2)),
    )
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2))
    figures(options.output, runs, comparisons)

    reproducible = [
        k
        for k, v in comparisons.items()
        if all(row["agrees_within_tolerance"] for row in v.values())
    ]
    print(f"\npublished numbers reproducible at grids: {reproducible or 'none'}")
    print(
        f"wrote {options.output}/summary.json, figure.png, magnitude-gap.png "
        f"({elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
