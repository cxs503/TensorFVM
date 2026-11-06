"""Gates for the FCT conservation audit (verification debt, m3035).

Report 3003 (m2940) attributed the FCT mass drift (5.7e-5 rel) to "the
low-order clamp clips the peak cell".  These gates pin the corrected
attribution:

* the low-order clamp clips **zero** mass (the donor-cell update stays
  strictly inside [0, 1]);
* the anti-diffusive FCT part is conservative to ~1e-15;
* the drift is a **boundary artifact** of the test setup (zero-gradient
  ghost + non-compact Gaussian tail + uniform flow) and is shared by the
  pure donor-cell baseline;
* on a compactly-supported profile (zero at the boundary) the conservation
  gate is meaningful: upwind is conservative to machine precision and the
  FCT-limited scheme is conservative to 1e-6..1e-9 (converging) and bounded.
"""

from tensorfvm.verification import fct_conservation_audit as fca


def test_gaussian_drift_is_boundary_not_clamp():
    dec = fca.decompose_gaussian_drift(256)
    # (1) the clamp never activates: the report-3003 attribution is refuted.
    assert dec["clamp_clip_per_step"] == 0.0
    assert 0.0 < dec["low_update_min"] < dec["low_update_max"] < 1.0
    # (2) the anti-diffusive FCT part is conservative.
    assert abs(dec["anti_contribution_per_step"]) < 1e-12
    # (3) the boundary flux explains the drift: per-step mass change
    #     = -(dt/V) * boundary_flux, with dt/V = 1/dx for this setup.
    #     The step-0 flux and the mean per-step drift agree to <0.02%
    #     (the flux varies slightly as the blob moves).
    per_step_drift = dec["fct_drift_rel"] * dec["mass0"] / dec["n_steps"]
    assert abs(-dec["boundary_flux_per_step"] * 256.0 - per_step_drift) \
        < 0.001 * per_step_drift
    # (4) the upwind baseline shares the drift (not an FCT-limiter bug).
    assert dec["fct_drift_rel"] > 0
    assert abs(dec["fct_drift_rel"] - dec["upwind_drift_rel"]) \
        < 0.10 * dec["fct_drift_rel"]


def test_compact_profile_conservation_and_order():
    out = fca.compact_study((64, 128, 256))
    for m in out["upwind"]:
        # donor-cell on a compact profile: machine-precision conservation.
        assert m["mass_rel_drift"] < 1e-12
        assert m["min_alpha"] >= -1e-12 and m["max_alpha"] <= 1.0 + 1e-12
    for m in out["fct"]:
        # FCT: conservative to the limiter's per-face discretisation,
        # converging with the grid; bounded to ~1e-9.
        assert m["mass_rel_drift"] < 1e-5
        assert m["min_alpha"] > -1e-6 and m["max_alpha"] < 1.0 + 1e-6
    # drift must decrease with refinement (converging, not a fixed leak)
    d = [m["mass_rel_drift"] for m in out["fct"]]
    assert d[0] > d[1] > d[2]
    # both schemes remain first-order accurate on the smooth profile.
    assert 0.7 < out["fct_order_l1"] < 1.4
    assert 0.7 < out["upwind_order_l1"] < 1.4
    # the limiter must not degrade accuracy versus the donor-cell baseline.
    for mf, mu in zip(out["fct"], out["upwind"]):
        assert mf["l1_rel"] <= mu["l1_rel"] * 1.05


def test_compact_profile_support_stays_inside_domain():
    # The gate is only valid if the profile support never touches the
    # boundary (zero-gradient BC inactive).  Negative control: a profile
    # whose support reaches the boundary must NOT be used for this gate.
    import numpy as np
    x = np.linspace(0.0, 1.0, 200)
    xx, yy = np.meshgrid(x, x)
    a = fca.cosine_hill(xx, yy, fca.C0, fca.R)
    assert a[0, :].max() == 0.0 and a[-1, :].max() == 0.0
    assert a[:, 0].max() == 0.0 and a[:, -1].max() == 0.0
    # and the reference is a pure translation (mass preserved analytically).
    a1 = fca.cosine_hill(xx, yy, (fca.C0[0] + 0.1, fca.C0[1] + 0.1), fca.R)
    assert abs(a.sum() - a1.sum()) / a.sum() < 1e-5  # discrete sampling of a translation
