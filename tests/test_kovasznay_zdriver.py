"""Pin the D10 z-momentum driver attribution (m16684).

The ``CASE-FVM-KOVASZNAY-ORDER`` attribution (m16396) left open *which* term
of the z-momentum equation produces the O(h) spanwise ``w``.  This driver
answers it with the exact-state residual probe: every term of the momentum
equation evaluated at the exact Kovasznay state (``w == 0``).

These tests pin the finding so it cannot silently rot:

* every *cell-w* row has residual 0.0 for every term -- the z-momentum cell
  equation is satisfied identically by ``w == 0`` (so the O(h) ``w`` is NOT a
  defect of the z-momentum cell discretisation);
* the only non-zero spanwise residuals live in the *face-w* rows (the
  shared-face equilibrium), dominated by the pressure term -- the coupling
  channel;
* the finding is stable across the symmetry-extrusion and the
  dirichlet-tangential controls (the cell-w equation is satisfied in both).
"""

from tensorfvm.verification.kovasznay_zdriver import run

# physical terms only; 'total' is the sum and trivially dominates
PHYSICAL = ('viscous', 'convection', 'pressure', 'load')


def test_cell_w_residual_is_zero_for_every_term():
    """The z-momentum cell equation is satisfied identically by w == 0."""
    r = run(n=8)
    assert r['cell_w_all_zero'] is True
    for term in PHYSICAL:
        assert r['cell_w_residual'][term] == 0.0, \
            f'cell-w {term} residual should be 0, got {r["cell_w_residual"][term]}'


def test_face_w_residual_is_nonzero_and_pressure_dominated():
    """The only non-zero spanwise residuals live in the face-w (coupling) rows."""
    r = run(n=8)
    # among the physical terms, pressure is the dominant face-w coupling channel
    dominant = max(PHYSICAL, key=lambda t: abs(r['face_w_residual'][t]))
    assert dominant == 'pressure', f'expected pressure, got {dominant}'
    assert r['face_w_residual']['pressure'] > 0.0
    assert r['face_w_residual']['pressure'] > r['face_w_residual']['viscous']
    assert r['face_w_residual']['pressure'] > r['face_w_residual']['convection']


def test_cell_w_zero_holds_across_tangential_controls():
    """The cell-w equation is satisfied in both symmetry and dirichlet modes."""
    for tangential in ('symmetry', 'dirichlet'):
        r = run(n=8, tangential=tangential)
        assert r['cell_w_all_zero'] is True, tangential


def test_cell_w_zero_holds_at_finer_grid():
    """The cell-w equation is satisfied at n=16 too (not a coarse-grid fluke)."""
    r = run(n=16)
    assert r['cell_w_all_zero'] is True
