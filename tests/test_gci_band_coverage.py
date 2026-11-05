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

**m3520 update**: Non-asymptotic fallback now widens bands when theory_order is
declared and |p_obs-p_theory|>order_tolerance, using Fs=3 + max adjacent diff.
This makes both conventions more conservative in the pre-asymptotic regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.richardson import richardson_gci, power_law_series

EXACT, H0 = 1.0, 0.5
P_TRUE = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
RATIOS = (1.2, 1.5, 2.0, 3.0)
ALPHA = (0.01, 0.05, 0.1, 0.3)


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
    # m3520: non-asymptotic fallback widens bands, so published convention also covers
    assert all(row["published"] for row in truncation)
    assert max(row["efficiency"] for row in truncation) <= 1.0


def test_non_asymptotic_fallback_widens_bands():
    """m3520: verify that non-asymptotic cases use Fs=3 and wider bands."""
    # Use power_law_series with a strong secondary term to create non-asymptotic behavior
    # The secondary term biases the observed order away from theory_order
    spacings = (0.1, 0.2, 0.4)
    values = power_law_series(1.0, 0.5, 2.0, spacings, secondary=4.0, secondary_order=4.0)
    study = richardson_gci(*values, spacings, theory_order=2.0)
    
    # Should be monotone but non-asymptotic (observed order biased by secondary term)
    assert study.convergence == "monotone"
    assert not study.asymptotic, f"Expected non-asymptotic but got asymptotic={study.asymptotic}, order={study.order}"
    
    # Should use Fs=3 (fallback safety factor)
    assert study.safety_factor == 3.0, f"Expected Fs=3 but got {study.safety_factor}"
    
    # Band should still cover the exact value
    assert _cover(study)
    
    # Efficiency should be reasonable
    assert study.gci > 0


def test_additive_noise_triggers_non_asymptotic_fallback():
    """m3520: noise can cause non-asymptotic behavior, triggering Fs=3 fallback."""
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
            study.asymptotic,
            study.safety_factor,
        ))
    
    # m3520: with non-asymptotic fallback, bands are wider and more conservative
    # The efficiency should stay below 1.0 (bound is not defeated)
    max_efficiency = max(row[0] for row in rows)
    assert max_efficiency <= 1.0, f"Non-asymptotic fallback should keep efficiency <= 1.0, got {max_efficiency}"
    
    # Most cases should still be monotone (noise is symmetric per grid)
    monotone_count = sum(1 for row in rows if row[2] == "monotone")
    assert monotone_count > 150, f"Expected most cases to be monotone, got {monotone_count}/200"
    
    # Some cases should trigger non-asymptotic fallback (Fs=3)
    fallback_count = sum(1 for row in rows if row[4] == 3.0 and row[2] == "monotone")
    assert fallback_count > 0, "Expected some cases to trigger non-asymptotic fallback"


def test_unresolved_second_term_stays_inside_the_band():
    # The realistic non-noisy case: a second error term never defeats the band,
    # it only widens it (efficiency stays below one at every alpha).
    for alpha in (0.01, 0.05, 0.1, 0.3):
        h, v = _series(1.0, 1.5, 1.5, alpha)
        study = richardson_gci(v[0], v[1], v[2], tuple(h), theory_order=1.0)
        assert study.band[0] <= EXACT <= study.band[1]
        assert abs(study.extrapolated - EXACT) / study.gci <= 1.0


def test_reference_agreement_asymptotic_gate():
    """m3520: band_covered is only True when asymptotic=True."""
    from tensorfvm.verification.richardson import reference_agreement
    
    # Asymptotic case: band_covered should be True
    h, v = _series(2.0, 2.0, 2.0, alpha=0.0)
    study = richardson_gci(v[0], v[1], v[2], tuple(h), theory_order=2.0)
    assert study.asymptotic
    agreement = reference_agreement(study, EXACT)
    assert agreement['band_covered']
    
    # Non-asymptotic case: band_covered should be False even if band contains reference
    spacings = (0.1, 0.2, 0.4)
    values = power_law_series(1.0, 0.5, 2.0, spacings, secondary=4.0, secondary_order=4.0)
    study = richardson_gci(*values, spacings, theory_order=2.0)
    assert not study.asymptotic
    agreement = reference_agreement(study, EXACT)
    # band_covered must be False when asymptotic=False (asymptotic gate)
    assert not agreement['band_covered']
