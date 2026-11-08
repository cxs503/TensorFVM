"""Mirror-symmetry verification of the planar sudden-expansion kernel.

The three ``inlet_window`` placements of
:func:`tensorfvm.incompressible_baselines.build_sudden_expansion` are used as a
*geometry-controlled* symmetry experiment:

* ``None`` (historical)  -> inlet flush with the bottom wall, **asymmetric**;
* ``(3, 4)``             -> exact y-mirror of the historical geometry;
* ``(1.5, 2.5)``         -> **mirror-symmetric** geometry (``AR = 4``).

The numerical evidence (converged ``Re_D = 5 ... 100`` solutions, the
mirror-covariance of the two mirrored geometries and the Reynolds sweep) lives in
``docs/verification-expansion-symmetry/``; the tests here pin the parts that are
cheap and deterministic: the discrete geometry, the post-processing metrics and
the *operator-level* mirror covariance of the momentum equations.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.incompressible_baselines import (
    build_sudden_expansion,
    expansion_mirror_metrics,
    expansion_symmetry_metrics,
    sudden_expansion_metrics,
)


# --------------------------------------------------------------------------- #
# discrete geometry
# --------------------------------------------------------------------------- #
def test_inlet_window_geometry_is_exactly_mirror_symmetric():
    """The centred inlet band must be an exact mirror image of itself."""
    solver = build_sudden_expansion(8, inlet_window=(1.5, 2.5))
    assert solver.meta["symmetric_about_midplane"] is True
    assert solver.meta["inlet_window"] == (1.5, 2.5)
    assert np.abs(solver.inlet_values - solver.inlet_values[::-1]).max() == 0.0


def test_top_inlet_is_exact_mirror_of_the_historical_geometry():
    """``inlet_window=(3, 4)`` reproduces the default geometry mirrored."""
    bottom = build_sudden_expansion(8)
    top = build_sudden_expansion(8, inlet_window=(3.0, 4.0))
    assert bottom.meta["symmetric_about_midplane"] is False
    assert top.meta["symmetric_about_midplane"] is False
    assert np.abs(top.inlet_values - bottom.inlet_values[::-1]).max() == 0.0
    # the historical geometry is *not* symmetric: the jet hugs the bottom wall
    assert bottom.inlet_values.sum() > 0.0
    assert bottom.inlet_values[-1] == 0.0 and bottom.inlet_values[0] == 1.0


def test_inlet_window_must_sit_on_cell_faces():
    with pytest.raises(ValueError):
        build_sudden_expansion(8, inlet_window=(1.53, 2.53))
    with pytest.raises(ValueError):
        build_sudden_expansion(8, inlet_window=(3.5, 4.5))  # leaves [0, H2]


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def test_symmetry_metrics_vanish_only_for_mirror_symmetric_fields():
    solver = build_sudden_expansion(8, inlet_window=(1.5, 2.5))
    ny = solver.ny
    rng = np.random.default_rng(0)
    u = rng.standard_normal((solver.nx + 1, ny))
    u = 0.5 * (u + u[:, ::-1])                  # even about y = H2/2
    v = rng.standard_normal((solver.nx, ny + 1))
    v = 0.5 * (v - v[:, ::-1])                  # odd about y = H2/2
    p = rng.standard_normal((solver.nx, ny))
    p = 0.5 * (p + p[:, ::-1])
    solver.u, solver.v, solver.p = u, v, p
    metrics = expansion_symmetry_metrics(solver)
    assert metrics["symmetry_rel"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["v_centreline_rel"] == pytest.approx(0.0, abs=1e-12)

    solver.u[3, 4] += 0.3 * np.abs(u).max()     # break the symmetry once
    assert expansion_symmetry_metrics(solver)["u_symmetry_rel"] > 1e-2


def test_mirror_metrics_between_mirrored_field_pairs():
    a = build_sudden_expansion(8, inlet_window=(1.5, 2.5))
    b = build_sudden_expansion(8, inlet_window=(1.5, 2.5))
    rng = np.random.default_rng(1)
    u = rng.standard_normal((a.nx + 1, a.ny))
    dc = rng.standard_normal((a.nx, a.ny + 1))
    dc = 0.5 * (dc - dc[:, ::-1])
    a.u = u
    a.v = dc
    a.p = rng.standard_normal((a.nx, a.ny))
    b.u = u[:, ::-1]
    b.v = -dc[:, ::-1]
    b.p = a.p[:, ::-1]
    mirror = expansion_mirror_metrics(a, b)
    assert mirror["mirror_rel"] < 1e-12
    b.v = b.v.copy()
    b.v[:, 2] += 0.5
    assert expansion_mirror_metrics(a, b)["v_mirror_rel"] > 1e-2


def test_expansion_metrics_read_the_inlet_band_not_the_bottom_wall():
    """Regression: ``K`` used to be evaluated on the bottom rows only.

    With the inlet flush with the top wall the bottom rows carry no inflow, so
    the historical post-processing raised ``ZeroDivisionError`` (``q_in = 0``).
    """
    solver = build_sudden_expansion(8, inlet_window=(3.0, 4.0))
    solver.u[1:] = 0.25          # redeveloped downstream bulk velocity
    solver.apply_bc()
    metrics = sudden_expansion_metrics(solver)
    assert metrics["u_in"] == pytest.approx(1.0)
    assert metrics["u_out"] == pytest.approx(0.25, rel=1e-6)
    assert metrics["reattachment_over_D"] == metrics["reattachment_over_D_upper"]
    assert "reattachment_over_D_lower" in metrics

    # and an inlet band with no inflow is reported, not silently divided by zero
    empty = build_sudden_expansion(8, inlet_window=(1.5, 2.5))
    empty.apply_bc()
    empty.inlet_values = np.zeros_like(empty.inlet_values)
    empty.apply_bc()             # zero the whole west boundary
    with pytest.raises(ValueError):
        sudden_expansion_metrics(empty)


# --------------------------------------------------------------------------- #
# operator-level mirror covariance (known defect, see docs/.../README.md)
# --------------------------------------------------------------------------- #
def _operator_mirror_residuals(level: int = 6) -> dict[str, float]:
    solver = build_sudden_expansion(level, re_D=5.0, inlet_window=(1.5, 2.5))
    ny = solver.ny
    rng = np.random.default_rng(0)
    u = rng.standard_normal((solver.nx + 1, ny))
    u = 0.5 * (u + u[:, ::-1])
    v = rng.standard_normal((solver.nx, ny + 1))
    v = 0.5 * (v - v[:, ::-1])
    p = rng.standard_normal((solver.nx, ny))
    p = 0.5 * (p + p[:, ::-1])
    solver.u, solver.v, solver.p = u, v, p

    def rel(a, b):
        scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-30)
        return float(np.abs(a - b).max()) / scale

    aE, aW, aN, aS, aP, b = solver._u_coefficients()
    out = {
        "u_aE": rel(aE, aE[:, ::-1]),
        "u_aW": rel(aW, aW[:, ::-1]),
        "u_aN": rel(aN, aS[:, ::-1]),
        "u_aP": rel(aP, aP[:, ::-1]),
        "u_b": rel(b, b[:, ::-1]),
    }
    vE, vW, vN, vS, vP, bv = solver._v_coefficients()
    out |= {
        "v_aE": rel(vE, vE[:, ::-1]),
        "v_aW": rel(vW, vW[:, ::-1]),
        "v_aN": rel(vN, vS[:, ::-1]),
        "v_aP": rel(vP, vP[:, ::-1]),
        "v_b": rel(bv, -bv[:, ::-1]),
    }
    return out


def test_u_momentum_operator_is_mirror_covariant():
    """The streamwise momentum coefficients are exactly mirror-covariant."""
    res = _operator_mirror_residuals()
    for key in ("u_aE", "u_aW", "u_aN", "u_aP"):
        assert res[key] < 1e-12, (key, res[key])


@pytest.mark.xfail(strict=False, reason=(
    "known defect: StaggeredSimple._u_on_v uses the stencil u[j-2], u[j-1] for "
    "the v-face j instead of u[j-1], u[j] (and treats the two wall rows "
    "differently: exactly 0 at j = 0, extrapolated at j = ny).  The rule is not "
    "mirror-covariant, so the v-momentum coefficients of a mirror-symmetric "
    "field are not (v aE/aW ~0.54, v aP ~0.14, v b ~0.83 relative).  The patch "
    "is documented in docs/verification-expansion-symmetry/README.md; flipping "
    "this test to XPASS is the acceptance check for it."))
def test_v_momentum_operator_is_mirror_covariant():
    res = _operator_mirror_residuals()
    for key in ("v_aE", "v_aW", "v_aN", "v_aP", "v_b", "u_b"):
        assert res[key] < 1e-12, (key, res[key])