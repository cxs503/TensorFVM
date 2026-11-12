"""Verification tests for the checkerboard (odd-even) pressure-decoupling index.

The collocated-grid pressure-velocity decoupling pathology is the classic
"velocity right, pressure wrong" silent failure: a checkerboard pressure mode
``p = (-1)^(i+j)`` is invisible to a momentum residual that only sees the
gradient, so a solver can converge to a pressure field that is spatially
decoupled.  The catalogue's existing N-S verification faces score *results*
against analytic/experimental anchors but never measure the *spatial quality*
of the pressure field -- this module pins the metric that does.

The key machine-checkable facts:

* the low-pass stencil ``(1, 2, 1)/4`` annihilates the Nyquist checkerboard mode
  (``S(p) = 0``, so ``CI = 1``) -- the *correct* detector;
* the second-difference stencil ``(-1, 2, -1)/4`` is the **identity** on the
  checkerboard mode (``S(p) = p``, so ``CI = 0``) -- the *wrong* detector, kept
  to make the failure explicit (this is the falsification of the naive
  "second-difference = checkerboard detector" hypothesis);
* a smooth (affine / analytic) pressure field has ``CI ~ 0`` under the low-pass
  detector (positive control);
* a checkerboard-perturbed field has ``CI`` strictly larger than the smooth
  field under the low-pass detector (negative control, must be red).
"""

import numpy as np
import pytest

from tensorfvm.collocated_diagnostics import checkerboard_index


def _checkerboard(nx, ny, nz):
    """Pure Nyquist checkerboard mode ``p = (-1)^(i+j+k)`` on a (nz,ny,nx) grid."""
    i = np.arange(nx)[None, None, :]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[:, None, None]
    return ((-1.0) ** (i + j + k)).astype(float)


def _smooth(nx, ny, nz):
    """Smooth affine field ``p = 1 + 0.3 x + 0.2 y - 0.1 z`` (no checkerboard)."""
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    z = (np.arange(nz) + 0.5) / nz
    return 1.0 + 0.3 * x[None, None, :] + 0.2 * y[None, :, None] - 0.1 * z[:, None, None]


def test_lowpass_annihilates_pure_checkerboard():
    """The correct detector: ``(1,2,1)/4`` gives CI = 1 on a pure checkerboard."""
    for n in (8, 16, 32):
        p = _checkerboard(n, n, n)
        r = checkerboard_index(p, stencil="lowpass")
        assert r["checkerboard_index"] == pytest.approx(1.0, abs=1e-6), r


def test_second_difference_is_identity_on_checkerboard():
    """Falsification: ``(-1,2,-1)/4`` is the identity on the checkerboard mode.

    ``S(p) = p`` for ``p = (-1)^(i+j)``, so the naive second-difference detector
    reports CI = 0 -- it *cannot* see the very mode it is meant to flag.
    """
    p = _checkerboard(16, 16, 16)
    r = checkerboard_index(p, stencil="second")
    assert r["checkerboard_index"] == pytest.approx(0.0, abs=1e-6), r


def test_lowpass_smooth_field_is_small():
    """Positive control: a smooth affine field has CI ~ 0 under the low-pass."""
    p = _smooth(16, 16, 16)
    r = checkerboard_index(p, stencil="lowpass")
    assert r["checkerboard_index"] < 1e-3, r


def test_negative_control_checkerboard_perturbation_is_red():
    """A checkerboard-perturbed field must be strictly more decoupled than smooth."""
    base = _smooth(16, 16, 16)
    for amp in (0.05, 0.2, 1.0):
        perturbed = base + amp * _checkerboard(16, 16, 16)
        ci_smooth = checkerboard_index(base, stencil="lowpass")["checkerboard_index"]
        ci_pert = checkerboard_index(perturbed, stencil="lowpass")["checkerboard_index"]
        assert ci_pert > ci_smooth + 0.05, (amp, ci_smooth, ci_pert)
        # the perturbation is monotonically detected
        assert ci_pert > 0.05 * amp, (amp, ci_pert)


def test_flat_pressure_requires_shape():
    with pytest.raises(ValueError):
        checkerboard_index(np.zeros(8 * 8 * 8))


def test_flat_pressure_reshape_matches_3d():
    p = _checkerboard(8, 8, 8)
    flat = p.ravel()
    r3 = checkerboard_index(p, stencil="lowpass")
    r1 = checkerboard_index(flat, nx=8, ny=8, nz=8, stencil="lowpass")
    assert r1["checkerboard_index"] == pytest.approx(r3["checkerboard_index"], abs=1e-12)


def test_zero_field_returns_zero():
    r = checkerboard_index(np.zeros((8, 8, 8)), stencil="lowpass")
    assert r["checkerboard_index"] == 0.0
    assert r["denominator"] == 0.0
