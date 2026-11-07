"""Channel-solver checks: convergence, grid independence, wall treatments.

Two of these tests are deliberately external rather than self-referential:

* ``test_matches_repository_independent_sst_reference`` compares this solver with
  the *independent* one-dimensional SST reference already in the repository
  (``docs/verification-native-sst-model-limit``, Cf = 0.0086491333 at
  ``Re_tau = 182.088``, n = 8192), which was written from a different source and
  does not share a line of code with this package;
* ``test_Cf_deviation_from_DNS_reproduces_the_documented_value`` compares the
  deviation from the original DNS with the 6.4737 per cent recorded in the same
  repository document.
"""
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel, stretch_for_first_center
from tensorfvm.rans_validation.reference import load_dns_profile

DNS_ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'
REPOSITORY_INDEPENDENT_SST_CF = .0086491333


def test_sst_channel_converges_and_satisfies_the_steady_equations():
    result = solve_channel('sst', 128, re_tau=180.)
    assert result['converged']
    assert result['relative_local_residual'] < 1e-10
    assert result['iterations'] > 100
    assert result['Cf'] > 0 and result['bulk_u_plus'] > 0


@pytest.mark.parametrize('model', ['sst', 'komega'])
def test_grid_refinement_leaves_the_friction_coefficient_unchanged(model):
    coarse = solve_channel(model, 128, re_tau=180.)
    fine = solve_channel(model, 256, re_tau=180.)
    assert coarse['converged'] and fine['converged']
    assert abs(fine['Cf'] / coarse['Cf'] - 1) < .02


def test_wall_function_domain_starts_at_the_first_wall_node():
    result = solve_channel('kepsilon', 32, re_tau=590., first_cell_y_plus=30.)
    assert result['inner_boundary_y_plus'] == pytest.approx(30.)
    assert result['y_plus'][0] > 30.
    assert result['wall_k'] == pytest.approx(1 / np.sqrt(.09))
    assert result['wall_variable'] == pytest.approx(1 / (.41 * 30. / 590.))
    assert result['converged']


def test_wall_function_bulk_velocity_includes_the_unresolved_sublayer():
    result = solve_channel('kepsilon', 32, re_tau=590., first_cell_y_plus=30.)
    resolved_only = float(result['volume'] @ result['u_plus'])
    assert result['bulk_u_plus'] > resolved_only
    assert 1.005 < result['bulk_u_plus'] / resolved_only < 1.05


@pytest.mark.parametrize('options', [dict(n=4), dict(re_tau=-1.), dict(dt=0.), dict(relaxation=2.),
                                     dict(tolerance=0.), dict(max_iterations=0)])
def test_invalid_controls_are_rejected(options):
    with pytest.raises(ValueError):
        solve_channel('sst', options.pop('n', 32), **options)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        solve_channel('k-omega-sst', 32)


def test_unreachable_first_cell_target_is_rejected():
    with pytest.raises(ValueError):
        stretch_for_first_center(16, .4)


@pytest.mark.parametrize('model', ['sst', 'komega', 'kepsilon'])
def test_profiles_are_positive_and_monotone(model):
    options = {'first_cell_y_plus': 30.} if model == 'kepsilon' else {}
    result = solve_channel(model, 64, re_tau=590., **options)
    assert np.all(result['u_plus'] > 0)
    assert np.all(np.diff(result['u_plus']) > 0)
    assert np.all(result['k'] >= 0)
    assert np.all(result['variable_values'] > 0)
    assert np.all(result['nut'] > 0)


@pytest.mark.parametrize('model', ['sst', 'komega', 'kepsilon'])
def test_friction_decreases_with_reynolds_number(model):
    options = {'first_cell_y_plus': 30.} if model == 'kepsilon' else {}
    low = solve_channel(model, 64, re_tau=180., **options)
    high = solve_channel(model, 64, re_tau=590., **options)
    assert high['Cf'] < low['Cf']


def test_implied_slope_is_not_one_of_the_two_published_options():
    """The three models must be distinguishable by the slope they imply."""
    slopes = {model: solve_channel(model, 32, re_tau=590.,
                                   **({'first_cell_y_plus': 30.} if model == 'kepsilon' else {}))['implied_kappa']
              for model in ('sst', 'komega', 'kepsilon')}
    assert slopes['komega'] > slopes['sst'] + .01
    assert slopes['kepsilon'] > slopes['sst'] + .01


def test_matches_repository_independent_sst_reference():
    """External cross-check against the repository's independent 1-D SST solve."""
    result = solve_channel('sst', 2048, re_tau=182.088)
    assert result['Cf'] == pytest.approx(REPOSITORY_INDEPENDENT_SST_CF, rel=.01)


@pytest.mark.skipif(not (DNS_ROOT / '0180' / 'mean.csv').is_file(), reason='DNS case 0180 not imported')
def test_Cf_deviation_from_DNS_reproduces_the_documented_value():
    dns = load_dns_profile(DNS_ROOT / '0180')
    result = solve_channel('sst', 512, re_tau=dns['re_tau'])
    deviation = result['Cf'] / dns['Cf'] - 1
    assert deviation == pytest.approx(.064737, abs=.01)