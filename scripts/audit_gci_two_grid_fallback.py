#!/usr/bin/env python3
"""Is the two-grid ``Fs = 3`` fall-back band of ``richardson_gci`` actually a bound?

Why this exists
---------------
``docs/gci-band-coverage/report.md`` closed two questions (T1 exactness, T2
truncation) for a **monotone** refinement series, and left one open:

    > 未审计两点回退（``Fs = 3``）的保守度是否足够.

That fall-back is what the docstring promises for a *broken* study: when the
three-grid sequence is not a monotone ``C h^p`` error decay, the observed order
is unavailable and the band is rebuilt from the assumed order with the two-grid
safety factor ``Fs = 3``.  The promise is

    "a broken refinement study can never report a confident band".

Before this audit the fall-back computed its whole width from the **fine/medium
pair only**:

    GCI_fb = Fs |(f2 - f1)/f1| / (r21^p - 1),   band = f_ext (1 +- GCI_fb).

A non-monotone series has, by definition, the two differences ``f2-f1`` and
``f3-f2`` of *opposite sign* -- and the interesting way to get there is not
noise but **two error terms of opposite sign that cancel between the fine and
the medium grid** (false convergence: the fine pair looks converged, the coarse
grid is far from the limit).  In that regime ``|f2-f1|`` is the *small*
difference and the band collapses exactly when the discretisation error is
largest; scaling it by ``|f_ext|`` can collapse it further, because in an
inconsistent study ``f_ext`` itself is meaningless and may pass through zero.

What is audited here
--------------------
An **analytic** family with an interior extremum, which is the cleanest way to
produce a genuine non-monotone three-grid series:

    f(h) = E + A h^p - (A p / q) h*^(p-q) h^q        (extremum at h = h*)

For ``h_fine < h* < h_coarse`` the series rises-then-falls (or vice versa), the
observed order is undefined, the fall-back triggers, and ``E`` is the *exact*
truth the band must cover.  Every quantity is closed form: no solver, no
tolerance, no sampling error in the truth itself.

Three conventions are compared on the identical case set:

* ``legacy`` -- the pre-fix formula above (fine pair only, ``|f_ext|`` scaling);
* ``implementation`` -- ``richardson_gci`` after the fix (worst of the two
  consecutive pairs, scaled by the largest metric, and never narrower than the
  spread of the three values).

Judged by coverage (does the band contain ``E``?) and by the Roache efficiency
index ``E_eff = |f_ext - E| / half_width`` (``<= 1`` means the band is a bound).
Coverage is reported on two denominators, because they answer different
questions:

* ``all`` -- every fall-back case, including those where ``E`` lies *outside*
  the span ``[min f, max f]``.  In that regime the three numbers carry no
  information about the limit at all (a false-convergence series can be
  arbitrarily far from it), so no data-driven band can cover it and the honest
  verdict is "disqualified", not "covered";
* ``informative`` -- the subset where ``E`` lies inside the span of the three
  computed values, i.e. where a bound is achievable in principle.  This is the
  denominator the implementation must be judged on.

A third family -- *consistent* series with a second error term -- must be
bit-identical under both conventions, because the fall-back does not trigger
there: that is the no-regression control.

Usage
-----
    PYTHONPATH=src python scripts/audit_gci_two_grid_fallback.py out=gci-two-grid-fallback
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from tensorfvm.verification.richardson import (SAFETY_FALLBACK, observed_order,
                                               richardson_gci)

EXACTS = (1.0, 10.0, 100.0)
P_SETS = ((2.0, 1.0), (1.5, 1.0), (3.0, 1.0), (2.0, 0.5))
RATIOS = ((1.5, 1.5), (2.0, 2.0), (2.0, 1.5), (1.5, 2.0))
H_STARS = (1.3, 1.6, 2.0, 2.6, 3.2)
THEORY_ORDER = 2.0
AMPLITUDE = 1.0
RANDOM_CASES = 300
RANDOM_SEED = 20261105


def extremum_series(exact, p, q, h_star, spacings, amplitude=AMPLITUDE):
    """``E + A h^p - (A p/q) h*^(p-q) h^q``: stationary point exactly at ``h_star``."""
    h = np.asarray(spacings, dtype=float)
    coefficient = amplitude * p / q * h_star ** (p - q)
    return exact + amplitude * h ** p - coefficient * h ** q


def legacy_band(fine, medium, coarse, spacings, theory_order=THEORY_ORDER):
    """Pre-fix fall-back width: the fine/medium pair alone, ``Fs = 3``, ``|f_ext|`` scaling.

    Replicated here (not imported) so that the audit keeps its control arm even
    after the library changed; ``test_gci_two_grid_fallback`` pins this replica
    to the library on consistent series, where the two must agree exactly.
    """
    h = tuple(float(x) for x in spacings)
    p = observed_order(h, (fine, medium, coarse))
    fallback = p != p
    p_used = float(theory_order) if fallback else p
    eps = abs((medium - fine) / fine)
    r21 = h[1] / h[0]
    gci = (SAFETY_FALLBACK if fallback else 1.25) * eps / (r21 ** p_used - 1.0)
    extrapolated = float(fine - (medium - fine) / (r21 ** p_used - 1.0))
    half = gci * abs(extrapolated)
    return dict(fallback=bool(fallback), gci=float(gci), extrapolated=extrapolated,
                half=float(half), band=(extrapolated - half, extrapolated + half))


def case_row(exact, spacings, values, theory_order=THEORY_ORDER):
    study = richardson_gci(*values, spacings, theory_order=theory_order)
    legacy = legacy_band(*values, spacings, theory_order=theory_order)
    truth = float(exact)
    fixed_half = 0.5 * (study.band[1] - study.band[0])
    fixed_error = abs(study.extrapolated - truth)
    legacy_error = abs(legacy["extrapolated"] - truth)
    return dict(
        exact=truth,
        spacings=[float(x) for x in spacings],
        values=[float(x) for x in values],
        informative=bool(min(values) <= truth <= max(values)),
        convergence=study.convergence,
        asymptotic=bool(study.asymptotic),
        order=None if study.order != study.order else float(study.order),
        order_used=float(study.order_used),
        extrapolated=float(study.extrapolated),
        gci=float(study.gci),
        half=float(fixed_half),
        covers=bool(study.band[0] <= truth <= study.band[1]),
        efficiency=float(fixed_error / fixed_half) if fixed_half else float("inf"),
        legacy_gci=float(legacy["gci"]),
        legacy_half=float(legacy["half"]),
        legacy_covers=bool(legacy["band"][0] <= truth <= legacy["band"][1]),
        legacy_efficiency=(float(legacy_error / legacy["half"])
                           if legacy["half"] else float("inf")),
        width_ratio=float(fixed_half / legacy["half"]) if legacy["half"] else float("inf"),
        fallback=bool(legacy["fallback"]),
    )


def extremum_rows():
    rows = []
    for exact in EXACTS:
        for p, q in P_SETS:
            for r21, r32 in RATIOS:
                spacings = (1.0, r21, r21 * r32)
                for h_star in H_STARS:
                    if not spacings[0] < h_star < spacings[2]:
                        continue
                    values = extremum_series(exact, p, q, h_star, spacings)
                    row = case_row(exact, spacings, values)
                    row.update(family="extremum", p=p, q=q, r21=r21, r32=r32,
                               h_star=h_star)
                    rows.append(row)
    return rows


def consistent_rows():
    """Control arm: monotone two-term series -- the fall-back must not trigger."""
    rows = []
    for exact in EXACTS:
        for p, q in P_SETS:
            for r21, r32 in RATIOS:
                spacings = (1.0, r21, r21 * r32)
                h = np.asarray(spacings)
                for alpha in (0.01, 0.1, 0.3):
                    values = exact + h ** p + alpha * h ** q
                    row = case_row(exact, spacings, values)
                    row.update(family="consistent", p=p, q=q, r21=r21, r32=r32,
                               alpha=alpha)
                    rows.append(row)
    return rows


def random_rows():
    """Randomised analytic series, kept only where the fall-back triggers."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    while len(rows) < RANDOM_CASES:
        exact = float(10.0 ** rng.uniform(-1.0, 2.0))
        p = float(rng.uniform(0.5, 3.0))
        q = float(rng.uniform(0.5, 3.0))
        r21 = float(rng.uniform(1.2, 3.0))
        r32 = float(rng.uniform(1.2, 3.0))
        spacings = (1.0, r21, r21 * r32)
        amplitude = float(exact * 10.0 ** rng.uniform(-3.0, -0.5))
        sign = -1.0 if rng.uniform() < 0.5 else 1.0
        h_star = float(rng.uniform(spacings[0] * 1.05, spacings[2] * 0.95))
        values = (exact + amplitude * np.asarray(spacings) ** p
                  + sign * amplitude * (p / q) * h_star ** (p - q)
                  * np.asarray(spacings) ** q)
        if min(values) <= 0.0 or values[0] == values[1]:
            continue
        if observed_order(spacings, tuple(values)) == observed_order(spacings, tuple(values)):
            continue  # monotone: not a fall-back case
        row = case_row(exact, spacings, values)
        row.update(family="random", p=p, q=q, r21=r21, r32=r32, h_star=h_star,
                   amplitude=amplitude)
        rows.append(row)
    return rows


def _rate(sub, key):
    return sum(1 for r in sub if r[key]) / len(sub) if sub else 1.0


def _worst(sub, key):
    return max((r[key] for r in sub), default=0.0)


def aggregate(rows):
    fallback = [r for r in rows if r["fallback"]]
    consistent = [r for r in rows if not r["fallback"]]
    informative = [r for r in fallback if r["informative"]]
    return dict(
        cases=len(rows),
        fallback_cases=len(fallback),
        fallback_informative_cases=len(informative),
        fallback_informative_fraction=len(informative) / max(1, len(fallback)),
        consistent_cases=len(consistent),
        coverage_all=_rate(fallback, "covers"),
        legacy_coverage_all=_rate(fallback, "legacy_covers"),
        coverage_informative=_rate(informative, "covers"),
        legacy_coverage_informative=_rate(informative, "legacy_covers"),
        worst_efficiency_all=_worst(fallback, "efficiency"),
        legacy_worst_efficiency_all=_worst(fallback, "legacy_efficiency"),
        worst_efficiency_informative=_worst(informative, "efficiency"),
        legacy_worst_efficiency_informative=_worst(informative, "legacy_efficiency"),
        consistent_unchanged=all(
            abs(r["half"] - r["legacy_half"]) <= 1e-12 * max(1.0, r["half"])
            for r in consistent),
        median_width_ratio=float(np.median([r["width_ratio"] for r in fallback])) if fallback else 1.0,
        max_width_ratio=_worst(fallback, "width_ratio"),
        legacy_misses=[{k: r[k] for k in ("exact", "p", "q", "r21", "r32", "h_star",
                                          "values", "legacy_efficiency", "efficiency")}
                       for r in informative if not r["legacy_covers"]][:8],
    )


def by_hstar(rows):
    table = []
    for h_star in H_STARS:
        sub = [r for r in rows if r["family"] == "extremum" and r["h_star"] == h_star
               and r["fallback"] and r["informative"]]
        if not sub:
            continue
        table.append(dict(
            h_star=h_star, cases=len(sub),
            coverage=_rate(sub, "covers"),
            legacy_coverage=_rate(sub, "legacy_covers"),
            worst_efficiency=_worst(sub, "efficiency"),
            legacy_worst_efficiency=_worst(sub, "legacy_efficiency"),
            median_width_ratio=float(np.median([r["width_ratio"] for r in sub])),
        ))
    return table


def by_ratio(rows):
    table = []
    for ratio in RATIOS:
        sub = [r for r in rows if r["family"] == "extremum"
               and (r["r21"], r["r32"]) == ratio and r["fallback"] and r["informative"]]
        if not sub:
            continue
        table.append(dict(r21=ratio[0], r32=ratio[1], cases=len(sub),
                          coverage=_rate(sub, "covers"),
                          legacy_coverage=_rate(sub, "legacy_covers"),
                          worst_efficiency=_worst(sub, "efficiency"),
                          legacy_worst_efficiency=_worst(sub, "legacy_efficiency")))
    return table


def figures(rows, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) the mechanism: one non-monotone series, both bands around their own centre.
    exact, p, q, h_star = 10.0, 2.0, 1.0, 2.0
    spacings = (1.0, 2.0, 4.0)
    grid = np.linspace(0.6, 5.0, 400)
    axes[0].plot(grid, extremum_series(exact, p, q, h_star, grid), color="0.3", lw=1.6,
                 label=r"$f(h)=E+h^2-4h$, $E=10$")
    values = extremum_series(exact, p, q, h_star, spacings)
    axes[0].plot(spacings, values, "o", color="C0", ms=7, label="three grids")
    axes[0].axhline(exact, color="k", ls="--", lw=1.2, label="exact $E$")
    study = richardson_gci(*values, spacings, theory_order=THEORY_ORDER)
    legacy = legacy_band(*values, spacings)
    axes[0].errorbar([0.85], [legacy["extrapolated"]],
                     yerr=[[legacy["half"]], [legacy["half"]]], fmt="s", color="C3",
                     capsize=6, label="legacy Fs=3 band (fine pair)")
    half = 0.5 * (study.band[1] - study.band[0])
    axes[0].errorbar([0.65], [study.extrapolated], yerr=[[half], [half]], fmt="D",
                     color="C2", capsize=6, label="fixed fall-back band")
    axes[0].annotate("fine pair cancels", xy=(1.5, -6.0), fontsize=8,
                     ha="center", color="C0")
    axes[0].set_xlabel("grid spacing $h$")
    axes[0].set_ylabel("metric value")
    axes[0].set_title("(a) false convergence: the fine pair cancels,\nthe coarse grid carries the error")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.3)

    # (b) coverage of the exact value on the informative subset, per refinement ratio.
    table = by_ratio(rows)
    labels = [f"r21={e['r21']:g}\nr32={e['r32']:g}" for e in table]
    x = np.arange(len(table))
    axes[1].bar(x - 0.2, [e["legacy_coverage"] for e in table], width=0.4,
                color="C3", label="legacy (fine pair)")
    axes[1].bar(x + 0.2, [e["coverage"] for e in table], width=0.4,
                color="C2", label="fixed fall-back")
    axes[1].axhline(1.0, color="k", ls=":", lw=1.0)
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("coverage of the exact value")
    axes[1].set_title("(b) non-monotone cases, informative subset")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    out = sys.argv[1].split("=", 1)[1] if len(sys.argv) > 1 and "=" in sys.argv[1] \
        else "gci-two-grid-fallback"
    target = os.path.join("outputs", out)
    os.makedirs(target, exist_ok=True)

    rows = extremum_rows() + consistent_rows() + random_rows()
    summary = aggregate(rows)
    hstar_table = by_hstar(rows)
    ratio_table = by_ratio(rows)
    full = json.dumps(dict(summary=summary, by_h_star=hstar_table,
                           by_ratio=ratio_table, rows=rows), indent=1)
    # The audit is regenerable bit-for-bit, so the committed record keeps the
    # summary, the tables and an excerpt of the per-case rows (the informative
    # rows in full, the rest strided) and archives the full dump by md5.
    excerpt = [r for r in rows if r["informative"]] + rows[::10]
    payload = dict(summary=summary, by_h_star=hstar_table, by_ratio=ratio_table,
                   row_count=len(rows), excerpt_count=len(excerpt),
                   full_dump_md5=hashlib.md5(full.encode()).hexdigest(),
                   rows=excerpt)
    with open(os.path.join(target, "audit.json"), "w") as handle:
        json.dump(payload, handle, indent=1)
    figures(rows, os.path.join(target, "gci_two_grid_fallback.png"))

    lines = ["# GCI two-grid fall-back audit", "",
             f"cases={summary['cases']} (fall-back={summary['fallback_cases']}, "
             f"informative={summary['fallback_informative_cases']}, "
             f"consistent controls={summary['consistent_cases']})", "",
             "Informative subset (exact value inside the span of the three computed values):", "",
             "| convention | coverage | worst efficiency |",
             "|---|---|---|",
             f"| legacy (fine pair, \\|f_ext\\| scaling) | "
             f"{summary['legacy_coverage_informative']:.4f} | "
             f"{summary['legacy_worst_efficiency_informative']:.3f} |",
             f"| fixed (worst pair, data scaling) | "
             f"{summary['coverage_informative']:.4f} | "
             f"{summary['worst_efficiency_informative']:.3f} |",
             "", "| h*/h_fine | cases | legacy coverage | fixed coverage | legacy worst E | fixed worst E | median width ratio |",
             "|---|---|---|---|---|---|---|"]
    for entry in hstar_table:
        lines.append(f"| {entry['h_star']:g} | {entry['cases']} | "
                     f"{entry['legacy_coverage']:.3f} | {entry['coverage']:.3f} | "
                     f"{entry['legacy_worst_efficiency']:.3f} | "
                     f"{entry['worst_efficiency']:.3f} | "
                     f"{entry['median_width_ratio']:.3f} |")
    lines += ["", "| r21 | r32 | cases | legacy coverage | fixed coverage | legacy worst E | fixed worst E |",
              "|---|---|---|---|---|---|---|"]
    for entry in ratio_table:
        lines.append(f"| {entry['r21']:g} | {entry['r32']:g} | {entry['cases']} | "
                     f"{entry['legacy_coverage']:.3f} | {entry['coverage']:.3f} | "
                     f"{entry['legacy_worst_efficiency']:.3f} | "
                     f"{entry['worst_efficiency']:.3f} |")
    with open(os.path.join(target, "RESULTS.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "legacy_misses"}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()