"""Does the GCI band that every report quotes actually contain the true value?

``richardson_gci`` was previously verified only against ``power_law_series`` -- a
one-term family built by the module itself.  These gates pin the audit in
``scripts/audit_gci_band_coverage.py`` (independently constructed two-term
truncation family + additive noise family), so a change to the estimator, the
safety factor or the band convention cannot silently shrink every quoted error
bar in the repository.

Conventions compared: this repository bands the *extrapolated* value
(``RefinementStudy.band``, used by ``reference_agreement(...)['band_covered']``);
Celik et al. (2008) / ASME V&V 20 band the *fine-grid* value with the same GCI
width and ``Fs``=1.25, so a coverage difference is a convention effect only.
"""

from __future__ import annotations

import numpy as np

from tensorfvm.verification.richardson import richardson_gci

EXACT, H0 = 1.0, 0.5
P_TRUE = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
RATIOS = (1.2, 1.5, 2.0, 3.0)
ALPHA = (0.0, 0.01, 0.05, 0.1, 0.3)


def _series(p, r21, r32, alpha=0.0, s=2.0, sign=1.0):
    h = np.array([H0, H0 * r21, H0 * r21 * r32], dtype=float)
    leading = sign * h**p
    second = alpha * sign * h**s * (h[0] ** (p - s))
    return [float(x) for x in h], [float(EXACT + leading[i] + second[i]) for i in range(3)]


def _cover(study):
    return study.band[0] <= EXACT <= study.band[1]


def _fine_cover(study, fine):
    half = study.gci * abs(fine)
    return fine - half <= EXACT <= fine + half


def _sweep(alphas=ALPHA):
    rows = []
    for p in P_TRUE:
        for r21 in RATIOS:
            for r32 in RATIOS:
                for alpha in alphas:
                    for sign in (1.0, -1.0):
                        h, v = _series(p, r21, r32, alpha, 2.0, sign)
                        study = richardson_gci(v[0], v[1], v[2], tuple(h), theory_order=p)
                        rows.append(dict(
                            p=p, alpha=alpha, sign=sign, study=study,
                            implementation=_cover(study),
                            published=_fine_cover(study, v[0]),
                            efficiency=abs(study.extrapolated - EXACT) / study.gci))
    return rows


def test_pure_one_term_band_covers_exact_and_is_an_upper_bound():
    pure = _sweep(alphas=(0.0,))
    assert len(pure) == 192
    assert all(row["implementation"] for row in pure)
    assert all(row["published"] for row in pure)
    assert max(row["efficiency"] for row in pure) <= 1.0


def test_two_term_band_never_misses_the_exact_value():
    truncation = [row for row in _sweep() if row["alpha"] > 0.]
    assert len(truncation) == 768
    assert all(row["implementation"] for row in truncation)
    assert max(row["efficiency"] for row in truncation) <= 1.0


def test_fine_grid_convention_misses_where_this_repository_does_not():
    # The published convention reports the uncertainty on the fine-grid value;
    # once a second error term is present it under-covers while the banded
    # extrapolated value does not.  Pin the direction, not just the magnitude.
    truncation = [row for row in _sweep() if row["alpha"] > 0.]
    published_rate = sum(1 for row in truncation if row["published"]) / len(truncation)
    assert published_rate < 0.99
    worse = [row for row in truncation if row["implementation"] and not row["published"]]
    assert worse, "the convention difference must be visible, not hypothetical"
    # The misses concentrate at low observed order and coarse refinement ratios.
    assert all(row["p"] <= 1.5 or row["alpha"] >= 0.05 for row in worse)


def test_additive_noise_breaks_the_bound_not_the_safety_factor():
    # 10 % relative grid noise is what an unconverged solve looks like. The
    # measured verdict (docs/gci-band-coverage/report.md) is that the band keeps
    # covering the exact value in most cases but its efficiency index exceeds
    # one, i.e. it stops being a bound.  Noise alone does not make the series
    # inconsistent, so the Fs=3 fallback is *not* what rescues the user.
    rng = np.random.default_rng(20261104)
    rows = []
    for _ in range(200):
        h, v = _series(2.0, 2.0, 2.0, 0.05)
        noisy = [value * (1.0 + 0.10 * rng.uniform(-1.0, 1.0)) for value in v]
        study = richardson_gci(noisy[0], noisy[1], noisy[2], tuple(h), theory_order=2.0)
        rows.append((
            abs(study.extrapolated - EXACT) / study.gci,
            study.band[0] <= EXACT <= study.band[1],
            study.convergence,
        ))
    assert max(row[0] for row in rows) > 1.0, "noise must be able to defeat the bound"
    assert sum(1 for row in rows if row[1]) / len(rows) < 1.0
    assert all(row[2] == "monotone" for row in rows), (
        "noise is symmetric per grid, so it does not by itself break monotone decay")


def test_unresolved_second_term_stays_inside_the_band():
    # The realistic non-noisy case: a second error term never defeats the band,
    # it only widens it (efficiency stays below one at every alpha).
    for alpha in (0.01, 0.05, 0.1, 0.3):
        h, v = _series(1.0, 1.5, 1.5, alpha)
        study = richardson_gci(v[0], v[1], v[2], tuple(h), theory_order=1.0)
        assert study.band[0] <= EXACT <= study.band[1]
        assert abs(study.extrapolated - EXACT) / study.gci <= 1.0
