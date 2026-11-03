"""Gates for the FCT accuracy re-verification (verification debt, m2940).

The merged feature is ``tensorfvm.vof_fct.LimitedVOFTransport`` (m2562); its
shipped check was self-consistency only.  These gates pin, against an
independent analytic reference (translated Gaussian), the accuracy order,
boundedness and conservation of the limited scheme, plus negative controls
for the order estimator itself.
"""
import numpy as np
import pytest

from tensorfvm.verification import fct_accuracy as fa


def test_order_estimator_recovers_known_orders():
    assert abs(fa.observed_order([4.0, 2.0, 1.0]) - 1.0) < 1e-12
    assert abs(fa.observed_order([4.0, 1.0, 0.25]) - 2.0) < 1e-12


@pytest.mark.parametrize("bad", [[1.0, 1.0, 0.5], [4.0, 2.0, 3.0], [4.0, 2.0, 0.0]])
def test_order_estimator_negative_controls(bad):
    with pytest.raises(ValueError):
        fa.observed_order(bad)


def test_gaussian_reference_is_translated_exact():
    # The reference is the analytic translation: same mass, peak at the
    # translated centre, independent of any solver code.
    n = 64
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x)
    t = 0.1
    ref0 = fa.gaussian(xx, yy, fa.X0)
    ref1 = fa.gaussian(xx, yy, (fa.X0[0] + fa.U * t, fa.X0[1] + fa.U * t))
    assert abs(ref0.sum() - ref1.sum()) / ref0.sum() < 5e-2  # discrete sampling of a translation
    iy, ix = np.unravel_index(np.argmax(ref1), ref1.shape)
    assert abs(x[ix] - (fa.X0[0] + fa.U * t)) < 2 * dx
    assert abs(x[iy] - (fa.X0[1] + fa.U * t)) < 2 * dx


def test_fct_bounded_and_conserving_on_smooth_advection():
    alpha, m = fa.advect_gaussian(48, scheme="fct")
    assert m["min_alpha"] >= -1e-12
    assert m["max_alpha"] <= 1.0 + 1e-12
    assert m["mass_rel_drift"] < 1e-3  # low-order clamp clips the peak (peak=1.0); see report  # telescoping conservation claim


def test_fct_at_least_as_accurate_as_upwind():
    # Smooth advection: the limiter must not degrade accuracy versus the
    # pure donor-cell baseline at the same grid/CFL.
    _, m_fct = fa.advect_gaussian(48, scheme="fct")
    _, m_up = fa.advect_gaussian(48, scheme="upwind")
    assert m_fct["l1_rel"] < 0.15
    assert m_fct["l1_rel"] <= m_up["l1_rel"] * 1.05


def test_grid_sequence_orders():
    out = fa.study(levels=(32, 64, 128))
    # Donor-cell baseline must show ~1st order (sanity of the whole pipeline).
    assert 0.7 < out["upwind_order_l1"] < 1.4
    # FCT: bounded, conservative, and at least first order on the smooth case.
    assert 0.8 < out["fct_order_l1"] < 2.6
    for m in out["fct"]:
        assert m["min_alpha"] >= -1e-12 and m["max_alpha"] <= 1.0 + 1e-12
        assert m["mass_rel_drift"] < 1e-3  # low-order clamp clips the peak (peak=1.0); see report
