#!/usr/bin/env python3
"""Is the two-grid fall-back band of ``richardson_gci`` actually conservative?

Why this exists
---------------
``richardson_gci`` reports an uncertainty band for every "+-x %" quoted by the
benchmark reports.  When the three-grid sequence carries *no order information*
the tool demotes the study to the two-grid fall-back: formal order + safety
factor 3, which the docstring advertises as "a broken refinement study can never
report a confident band".  The coverage audit of the *monotone* path
(``scripts/audit_gci_band_coverage.py``) does not touch this path, and neither
does ``power_law_series`` (one monotone error term).  This audit asks three
questions with closed-form truths:

* **F-fallback (non-monotone)**: with a genuine limit ``E`` and a sign-alternating
  or phase-locked error ``e_i = A h_i^p s_i``, does the fall-back band contain
  ``E``, and does the efficiency index stay ``E_eff = |error|/GCI <= 1``?
* **F-cancel (two-term cancellation)**: two truncation terms of opposite sign can
  make the two finest metrics nearly equal while both are far from the limit.
  The fall-back ``eps`` used to be the fine/medium difference alone, so the band
  collapsed exactly there.
* **F-implausible (vanishing first difference)**: the same cancellation through
  the *monotone* path: ``(f3-f2)/(f2-f1)`` stays positive but huge, the order
  solver (bracket up to 40) returns ``p >> p_theory``, and ``GCI ~ 1/(r^p-1)``
  collapses.  A triplet like that says nothing about convergence, so it is not a
  two-grid study, it is no study at all.

Two rules are compared per case: the shipped estimator, and ``_pre_fix`` -- a
faithful reimplementation of the previous rule (no plausibility window,
``eps = |f2-f1|/|f1|`` in the fall-back).  Every number below is produced by this
script; nothing is transcribed.

Families without a limit (``F-bias``: a non-decaying constant offset in the
metric) are reported too, as an honest scope limit: no band built from three
grids can see a constant bias, and this audit does not claim otherwise.

Usage
-----
    PYTHONPATH=src python scripts/audit_gci_fallback_band.py out=gci-fallback-band
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.verification.richardson import (
    SAFETY_FALLBACK,  # noqa: E402
    SAFETY_OBSERVED,
    richardson_gci,
)

EXACT = 1.0
H0 = 0.1
RATIOS = (1.5, 2.0, 3.0)
ORDERS = (1.0, 2.0, 3.0)
AMPLITUDES = (0.02, 0.1, 0.4)


def _spacings(r21, r32):
    return [H0, H0 * r21, H0 * r21 * r32]


def _pre_fix(values, spacings, theory_order):
    """The rule this repository shipped before the plausibility window.

    Kept verbatim (no order cap, ``eps`` from the fine/medium difference only) so
    that "before" and "after" are computed by the same code, not by memory.
    """
    h = np.asarray(spacings, float)
    v = np.asarray(values, float)
    r21 = h[1] / h[0]
    d1, d2 = v[1] - v[0], v[2] - v[1]
    p = float("nan")
    if d1 != 0.0:
        q = d2 / d1
        if q > 0.0:  # same bracket bisection as observed_order()
            lo, hi = 1e-8, 40.0

            def ratio(x):
                return math.expm1(x * math.log(h[2] / h[1])) / (
                    -math.expm1(-x * math.log(r21))
                )

            if ratio(lo) <= q <= ratio(hi):
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ratio(mid) < q else (lo, mid)
                p = 0.5 * (lo + hi)
    fallback = p != p
    p_used = theory_order if fallback else p
    eps = abs(d1 / v[0])
    gci = (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED) * eps / (r21**p_used - 1.0)
    extrapolated = float(v[0] - (v[1] - v[0]) / (r21**p_used - 1.0))
    half = gci * abs(extrapolated)
    return dict(
        order=p,
        convergence="inconsistent" if fallback else "monotone",
        safety_factor=SAFETY_FALLBACK if fallback else SAFETY_OBSERVED,
        gci=gci,
        extrapolated=extrapolated,
        band=(extrapolated - half, extrapolated + half),
    )


def _score(study_band, gci, extrapolated, exact):
    lo, hi = study_band
    error = abs(extrapolated - exact) / abs(exact)
    covered = bool(lo <= exact <= hi)
    return dict(
        covered=covered,
        error=float(error),
        gci=float(gci),
        efficiency=float("inf") if gci == 0.0 else float(error / gci),
        half_width=float(0.5 * (hi - lo)),
        extrapolated=float(extrapolated),
    )


def cases():
    """Deterministic case list: (family, label, spacings, values, exact)."""
    out = []
    for r21 in RATIOS:
        for r32 in RATIOS:
            h = _spacings(r21, r32)
            for p in ORDERS:
                for amp in AMPLITUDES:
                    # F-control: pure monotone C h^p, the path the fix must not touch
                    out.append(
                        (
                            "F-control",
                            f"monotone p={p} A={amp} r=({r21},{r32})",
                            h,
                            [EXACT + amp * x**p for x in h],
                            EXACT,
                        )
                    )
                    # F-signed: decaying sign-alternating error, limit exists
                    out.append(
                        (
                            "F-signed",
                            f"alternating p={p} A={amp} r=({r21},{r32})",
                            h,
                            [
                                EXACT + amp * (x**p) * s
                                for x, s in zip(h, (1.0, -1.0, 1.0))
                            ],
                            EXACT,
                        )
                    )
                    for phase in (0.3, 0.48, 0.62):
                        # F-phaselock: oscillation whose two finest samples nearly cancel
                        out.append(
                            (
                                "F-phaselock",
                                f"phase-locked p={p} A={amp} phi={phase} r=({r21},{r32})",
                                h,
                                [
                                    EXACT
                                    + amp
                                    * (x**p)
                                    * math.sin(2 * math.pi * index * phase)
                                    for index, x in enumerate(h)
                                ],
                                EXACT,
                            )
                        )
                    # F-cancel: opposing first/second-order terms; |e0| ~ |e1| << |e2|
                    out.append(
                        (
                            "F-cancel",
                            f"two-term cancel A={amp} r=({r21},{r32})",
                            h,
                            [EXACT + amp * (2.0 * x - 9.0 * x**2) for x in h],
                            EXACT,
                        )
                    )
                    # F-implausible: fine ~ medium, coarse far away -> p >> p_theory
                    out.append(
                        (
                            "F-implausible",
                            f"vanishing d1 p={p} r=({r21},{r32})",
                            h,
                            [
                                EXACT + amp,
                                EXACT + amp * (1.0 + 1e-3),
                                EXACT + 10.0 * amp,
                            ],
                            EXACT,
                        )
                    )
                    # F-bias: no limit (constant offset) -- documented scope limit
                    out.append(
                        (
                            "F-bias",
                            f"biased p={p} A={amp} r=({r21},{r32})",
                            h,
                            [EXACT + 0.5 + amp * x**p for x in h],
                            EXACT,
                        )
                    )
    return out


def evaluate(theory_order=2.0):
    records = []
    for family, label, h, values, exact in cases():
        study = richardson_gci(*values, h, theory_order=theory_order)
        after = _score(study.band, study.gci, study.extrapolated, exact)
        before_study = _pre_fix(values, h, theory_order)
        before = _score(
            before_study["band"],
            before_study["gci"],
            before_study["extrapolated"],
            exact,
        )
        records.append(
            dict(
                family=family,
                label=label,
                spacings=list(map(float, h)),
                values=list(map(float, values)),
                exact=exact,
                convergence=study.convergence,
                fallback_reason=study.fallback_reason,
                observed_order=study.order,
                order_used=study.order_used,
                safety_factor=study.safety_factor,
                before=dict(before, convergence=before_study["convergence"]),
                after=dict(after, convergence=study.convergence),
            )
        )
    return records


def summarise(records):
    rows = []
    for family in dict.fromkeys(r["family"] for r in records):
        group = [r for r in records if r["family"] == family]
        fallback = [r for r in group if r["after"]["convergence"] != "monotone"]
        rows.append(
            dict(
                family=family,
                cases=len(group),
                fallback_cases=len(fallback),
                coverage_before=float(np.mean([r["before"]["covered"] for r in group])),
                coverage_after=float(np.mean([r["after"]["covered"] for r in group])),
                worst_efficiency_before=float(
                    max(r["before"]["efficiency"] for r in group)
                ),
                worst_efficiency_after=float(
                    max(r["after"]["efficiency"] for r in group)
                ),
                median_band_ratio=float(
                    np.median(
                        [
                            r["after"]["half_width"] / r["before"]["half_width"]
                            for r in group
                            if r["before"]["half_width"] > 0.0
                        ]
                    )
                ),
                unchanged_gci=int(
                    sum(
                        abs(r["after"]["gci"] - r["before"]["gci"])
                        <= 1e-12 * max(1.0, abs(r["before"]["gci"]))
                        for r in group
                    )
                ),
            )
        )
    return rows


def _figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["family"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(
        x - 0.2, [row["coverage_before"] for row in rows], 0.4, label="pre-fix rule"
    )
    axes[0].bar(
        x + 0.2, [row["coverage_after"] for row in rows], 0.4, label="shipped rule"
    )
    axes[0].set_ylabel("band contains the exact limit")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    worst_old = [min(max(row["worst_efficiency_before"], 1e-3), 1e6) for row in rows]
    worst_new = [min(max(row["worst_efficiency_after"], 1e-3), 1e6) for row in rows]
    axes[1].semilogy(x - 0.2, worst_old, "o", label="pre-fix rule")
    axes[1].semilogy(x + 0.2, worst_new, "s", label="shipped rule")
    axes[1].axhline(1.0, color="r", lw=0.9, ls="--")
    axes[1].set_ylabel("worst efficiency index  E = |error|/GCI")
    axes[1].text(
        len(labels) - 1.2, 1.4, "E > 1: band not a bound", color="r", fontsize=8
    )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [lab.replace("F-", "") for lab in labels], rotation=30, ha="right"
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(
        "GCI two-grid fall-back: band coverage and efficiency (three-grid studies, p_theory=2)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _case_figure(path):
    """Concrete case: pre-fix band collapsed, shipped band contains the limit."""
    h = _spacings(2.0, 2.0)
    values = [
        EXACT + 0.1,
        EXACT + 0.1 * (1.0 + 1e-3),
        EXACT + 1.0,
    ]  # fine ~ medium, coarse far
    from tensorfvm.verification.richardson import richardson_gci as shipped

    after = shipped(*values, h, theory_order=2.0)
    before = _pre_fix(values, h, 2.0)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    error = [v - EXACT for v in values]
    axis.plot([1, 2, 3], error, "ko-", label="metric error  f(h_i) - f_exact")
    axis.axhline(0.0, color="k", lw=0.7)

    def parts(study):
        if isinstance(study, dict):
            return study["gci"], study["extrapolated"], study["band"]
        return study.gci, study.extrapolated, study.band

    for label, study, colour in (
        ("pre-fix band (on f_ext)", before, "tab:red"),
        ("shipped band (on f_ext)", after, "tab:blue"),
    ):
        gci, ext, band = parts(study)
        axis.axhline(ext - EXACT, color=colour, lw=0.8, ls="--")
        axis.fill_between(
            [2.5, 3.2],
            band[0] - EXACT,
            band[1] - EXACT,
            color=colour,
            alpha=0.35,
            label=f"{label}: GCI={gci:.2e}",
        )
    axis.set_xlabel("refinement level (fine -> coarse)")
    axis.set_ylabel("error of the quoted quantity")
    axis.set_title(
        "vanishing fine/medium difference:\npre-fix band is 1e-7 wide, shipped band covers"
    )
    axis.legend(fontsize=7, loc="lower right")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _results_md(rows, records):
    worst = max(records, key=lambda r: r["after"]["efficiency"])
    lines = [
        "# GCI two-grid fall-back audit -- aggregate tables",
        "",
        "Generated by `scripts/audit_gci_fallback_band.py` (zero transcription).",
        "",
        "| family | cases | fall-back cases | coverage (pre-fix) | coverage (shipped) "
        "| worst E (pre-fix) | worst E (shipped) | median band ratio | gci unchanged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {}/{} |".format(
                row["family"],
                row["cases"],
                row["fallback_cases"],
                row["coverage_before"],
                row["coverage_after"],
                row["worst_efficiency_before"],
                row["worst_efficiency_after"],
                row["median_band_ratio"],
                row["unchanged_gci"],
                row["cases"],
            )
        )
    lines += [
        "",
        "Worst remaining case (shipped rule): `{}` -- family `{}`, convergence `{}`, "
        "E={:.3f}, error={:.3e}, GCI={:.3e}.".format(
            worst["label"],
            worst["family"],
            worst["convergence"],
            worst["after"]["efficiency"],
            worst["after"]["error"],
            worst["after"]["gci"],
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path("outputs/gci-fallback-band")
    for arg in argv:
        if arg.startswith("out="):
            out = Path("outputs") / arg.split("=", 1)[1]
    records = evaluate()
    rows = summarise(records)
    out.mkdir(parents=True, exist_ok=True)
    # Excerpt only: the full 648-case dump is regenerable in seconds and the
    # repository caps artifacts at ~200 KB.
    excerpt = [r for r in records if r["family"] != "F-control"][:48]
    payload = dict(
        audit="gci-fallback-band",
        exact=EXACT,
        spacings_h0=H0,
        ratios=list(RATIOS),
        orders=list(ORDERS),
        amplitudes=list(AMPLITUDES),
        theory_order=2.0,
        case_count=len(records),
        families=rows,
        excerpt=excerpt,
        scope=(
            "Deterministic closed-form families; the monotone path is a control "
            "(gci must be unchanged). F-bias has no limit by construction and is "
            "reported as a scope limit, not as a defect."
        ),
    )
    (out / "audit.json").write_text(json.dumps(payload, indent=1, allow_nan=True))
    (out / "RESULTS.md").write_text(_results_md(rows, records))
    _figure(rows, out / "fallback-band.png")
    _case_figure(out / "collapsed-band-case.png")
    print(_results_md(rows, records))
    print(
        f"wrote {out}/audit.json ({len(records)} cases), RESULTS.md, fallback-band.png, "
        "collapsed-band-case.png"
    )


if __name__ == "__main__":
    main()
