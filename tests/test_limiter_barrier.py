"""Self-verification of the TVD-limiter accuracy-barrier study.

Five groups of assertions, in the order they must be believed:

``mirror``
    the NumPy limiter arithmetic used to *inspect* the field agrees with the
    solver's own ``torch`` path to round-off, so the mechanism numbers are read
    off the same limiter the solver ran.
``structure``
    the TVD properties the mechanism argument rests on are exact, not measured:
    ``minmod``/``vanleer``/``superbee`` return an exactly zero slope at a cell
    whose data carries a local extremum, no limiter ever amplifies the centred
    slope, and the extremum-cell count is a property of the solution (two per
    period) rather than of the grid.
``measurement``
    on the merged solver: the unlimited control recovers order 2 under both
    norms, every TVD limiter stays inside the 15 % band on L1 and leaves it
    downwards on L-inf, and the price (pointwise amplification) grows under
    refinement.
``criteria``
    the predeclared criteria fire, and they are *not* vacuous: a fabricated study
    in which a limited scheme is second order in L-inf must fail B4/B5.
``doc drift``
    the reconstruction's own docstring must carry the norm qualification this
    module measured, so the unqualified "second order for smooth flow" claim
    cannot come back.
"""
import copy
from functools import lru_cache

import numpy as np
import pytest

from tensorfvm.verification import limiter_barrier as lb

GRIDS = (40, 80, 160)


# ------------------------------------------------------------------ fixtures
@lru_cache(maxsize=1)
def _study():
    return lb.study(GRIDS)


@lru_cache(maxsize=1)
def _criteria():
    return lb.criteria(_study())


def _by_name(checks):
    return {check["name"]: check for check in checks}


def _synthetic(**overrides):
    """Deep copy of the real study with selected readout orders overwritten."""
    out = copy.deepcopy(_study())
    for limiter, norms in overrides.items():
        for norm, value in norms.items():
            out["readouts"][limiter][norm]["p_loglog"] = float(value)
    return out


# -------------------------------------------------------------------- mirror
def test_numpy_mirror_reproduces_the_solver_limiter():
    errors = lb.mirror_error()
    assert set(errors) == set(lb.PROBES)
    for limiter, err in errors.items():
        assert err <= 1e-12, (limiter, err)


def test_centred_slope_is_the_unlimited_baseline():
    q = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    assert lb.slopes(q, "none") == pytest.approx(lb.centred_slope(q))
    # the pure ramp has a constant centred slope, one second difference apart
    ramp = np.linspace(0.0, 1.0, 33)
    # interior cells only: the two cells next to the periodic seam see the jump
    assert np.allclose(lb.slopes(ramp, "none")[1:-1], 1.0 / 32.0)


def test_unknown_limiter_is_rejected():
    with pytest.raises(ValueError):
        lb.slopes(np.zeros(8), "koren")


# ---------------------------------------------------------------- structure
def test_tvd_limiters_return_exactly_zero_slope_at_extrema():
    x = (np.arange(64) + 0.5) / 64
    q = 1.0 + 0.2 * np.cos(2.0 * np.pi * x)
    extrema = lb.extremum_cells(q)
    # symmetric grid: each extremum sits between two cells, whose vanishing
    # neighbour difference makes both of them extremum cells
    assert int(extrema.sum()) == 4
    for limiter in ("minmod", "vanleer", "superbee"):
        assert np.all(lb.slopes(q, limiter)[extrema] == 0.0), limiter


def test_diffusive_limiters_never_amplify_the_centred_slope():
    """minmod/vanleer/mc are bounded by the centred slope; superbee may amplify."""
    rng = np.random.default_rng(7)
    for q in (rng.standard_normal(101), 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(80) / 80)):
        centred = lb.slopes(q, "none")
        for limiter in ("minmod", "vanleer", "mc"):
            assert np.all(np.abs(lb.slopes(q, limiter)) <= np.abs(centred) + 1e-12)
        # superbee is compressive: it is allowed to exceed, and it is observed to
        assert np.any(np.abs(lb.slopes(q, "superbee")) > np.abs(centred))


def test_extremum_cell_count_is_a_property_of_the_solution_not_the_grid():
    counts = []
    for nx in (32, 64, 128, 256):
        x = (np.arange(nx) + 0.5) / nx
        counts.append(int(lb.extremum_cells(1.0 + 0.2 * np.sin(2 * np.pi * x)).sum()))
    assert len(set(counts)) == 1 and counts[0] <= 4


def test_clipping_reports_the_defective_set_on_a_real_field():
    clip = lb.clipping(1.0 + 0.2 * np.cos(2 * np.pi * (np.arange(40) + 0.5) / 40), "minmod")
    assert 2 <= clip["extremum_cells"] <= 4
    assert clip["extremum_slope_zero"] is True
    assert clip["never_amplifies"] is True
    assert 0.0 <= clip["min_clip_factor"] <= 1.0


# -------------------------------------------------------------- measurement
def test_unlimited_control_recovers_order_two_in_both_norms():
    study = _study()
    orders = {norm: study["readouts"]["none"][norm]["p_loglog"] for norm in lb.NORMS}
    for norm, value in orders.items():
        assert abs(value - 2.0) / 2.0 <= lb.THRESHOLD, (norm, value)


def test_l1_masks_the_limiter_barrier():
    study = _study()
    for limiter in lb.TVD_LIMITERS:
        p_l1 = study["readouts"][limiter]["l1"]["p_loglog"]
        assert abs(p_l1 - 2.0) / 2.0 <= lb.THRESHOLD, (limiter, p_l1)


def test_linf_exposes_the_limiter_barrier():
    study = _study()
    for limiter in lb.TVD_LIMITERS:
        p_linf = study["readouts"][limiter]["linf"]["p_loglog"]
        assert (2.0 - p_linf) / 2.0 >= lb.THRESHOLD, (limiter, p_linf)
        assert p_linf > 1.0, (limiter, p_linf)          # not first order either


def test_norm_gap_is_what_decides_the_verdict():
    study = _study()
    for limiter in lb.TVD_LIMITERS:
        gap = (study["readouts"][limiter]["l1"]["p_loglog"]
               - study["readouts"][limiter]["linf"]["p_loglog"])
        assert gap >= lb.MIN_NORM_GAP, (limiter, gap)


def test_pointwise_price_and_its_growth():
    study = _study()
    for limiter in lb.TVD_LIMITERS:
        series = lb.amplification(study, limiter)
        assert series[-1] >= lb.MIN_AMPLIFICATION, (limiter, series)
        assert series[-1] / series[0] >= lb.MIN_AMPLIFICATION_GROWTH, (limiter, series)


def test_verdict_never_promotes_a_tvd_limiter_to_uniform_second_order():
    labels = lb.verdict(_study())["per_probe"]
    assert labels["none"]["label"] == "second-order-in-both-norms"
    for limiter in lb.TVD_LIMITERS:
        assert labels[limiter]["label"] == "second-order-in-L1-only"


# ---------------------------------------------------------------- criteria
def test_criteria_pass_except_the_long_ladder_which_is_not_run():
    checks = _by_name(_criteria())
    failed = [name for name, check in checks.items() if not check["passed"]]
    assert failed == ["B10"], failed
    assert "not_run" in str(checks["B10"]["observed"])


def test_criteria_are_not_vacuous_on_a_fabricated_second_order_limiter():
    fake = _synthetic(minmod={"l1": 2.0, "linf": 2.0})
    checks = _by_name(lb.criteria(fake))
    assert checks["B4"]["passed"] is False
    assert checks["B5"]["passed"] is False
    assert lb.verdict(fake)["per_probe"]["minmod"]["label"] in (
        "second-order-in-both-norms",)  # the probe itself looks fine ...
    # ... which is exactly why B4 (the barrier) has to be a separate criterion.


def test_criteria_detect_a_scheme_that_is_not_second_order_anywhere():
    fake = _synthetic(mc={"l1": 1.0, "linf": 1.0})
    checks = _by_name(lb.criteria(fake))
    assert checks["B3"]["passed"] is False
    assert lb.verdict(fake)["per_probe"]["mc"]["label"] != "second-order-in-L1-only"


def test_idealised_first_order_is_declared_not_confirmed():
    check = _by_name(_criteria())["B11"]
    assert check["passed"] is True
    assert "NOT_CONFIRMED" in check["threshold"]


# --------------------------------------------------------------- doc drift
def test_reconstruction_docstring_carries_the_norm_qualification():
    audit = lb.declared_claim_audit()
    assert audit["mentions_L1"] and audit["mentions_Linf"]
    assert audit["documented"] is True


def test_the_merged_cellcentre_study_is_the_L1_readout_this_module_resolves():
    """The barrier study must reuse the merged ladder, not re-derive its own."""
    from tensorfvm.verification import cellcentre_loglog as cc

    study = _study()
    merged = cc.run_case(GRIDS[0], order=2, cfl=0.4, t_end=0.2, limiter="mc")
    assert study["rows"]["mc"][0]["cell_average"]["l1"] == pytest.approx(
        merged["cell_average"]["l1"], rel=1e-12)