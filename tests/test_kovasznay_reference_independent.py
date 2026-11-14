"""Non-circular check of the Kovasznay 1948 reference behind CASE-FVM-KOVASZNAY-ORDER.

Why this file exists
--------------------
``tests/test_kovasznay_32_64_128_acceptance.py`` recomputes every relative L2
error from the committed ``n<N>-fields.npz`` dumps, which rules out hand
transcription -- but it recomputes them with the *same* helper the producer used
(``tensorfvm.native_manufactured3d.volume_weighted_error``) and it takes the
``exact_velocity`` / ``exact_pressure`` arrays inside those dumps on trust.  If
the reference field itself were wrong (wrong Re, wrong sign of ``lam``, a field
fitted to the solution instead of the published one), every downstream number
would still be perfectly reproducible.

This gate re-derives the reference from the published formulae alone
(``scripts/verify_kovasznay_reference_independent.py``, zero TensorFVM imports),
proves it is a steady Navier-Stokes solution, compares it against the ``exact_*``
arrays actually stored in the dumps, and redoes the norms/slopes with its own
numpy code.

Reference: Kovasznay (1948), Proc. Cambridge Philos. Soc. 44:58-62,
doi:10.1017/s0305004100023999.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'verify_kovasznay_reference_independent.py'
RUNS = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128-zero_gradient'
GRIDS = (32, 64, 128)
THRESHOLD = 1.8


def _module():
    spec = importlib.util.spec_from_file_location('kovasznay_reference_independent', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def ref():
    return _module()


@pytest.fixture(scope='module')
def report(ref):
    for n in GRIDS:
        if not (RUNS / f'n{n}' / f'n{n}-fields.npz').is_file():
            pytest.skip(f'n={n} field dump not committed')
    for n in GRIDS:
        if not (RUNS / f'n{n}' / 'cases.json').is_file():
            pytest.skip(f'n={n} cases.json not committed')
    return ref.main(['--runs', str(RUNS), '--grids', ','.join(str(n) for n in GRIDS)])


def test_decay_rate_solves_the_published_dispersion_relation(ref):
    lam = ref.decay_rate()
    assert lam < 0.
    # lam^2 - Re*lam = (2*pi)^2, checked independently of the closed form
    assert abs(lam ** 2 - ref.REYNOLDS * lam - ref.TWO_PI ** 2) <= 1e-12
    # published value for Re = 40
    assert lam == pytest.approx(-0.9637405441957689, rel=1e-12)


def test_reconstructed_field_is_a_steady_navier_stokes_solution(report):
    """div u == 0 and the momentum residual vanishes to finite-difference accuracy."""
    check = report['residual_check']
    assert check['max_abs_divergence'] <= 1e-9
    assert check['max_abs_momentum_residual'] <= 1e-5


def test_dumped_exact_fields_are_the_published_solution(report, ref):
    """The reference arrays in every dump must equal the paper's field at round-off."""
    for entry in report['exact_field_check']:
        assert entry['max_abs_velocity_diff'] <= 1e-12, entry
        assert entry['max_abs_pressure_diff_after_gauge'] <= 1e-12, entry
        # only the additive pressure constant may differ between conventions
        assert abs(entry['pressure_gauge_offset']) <= 1e-12, entry


def test_reference_is_not_degenerate(ref):
    """A zero or spatially uniform reference would make the L2 metric meaningless."""
    grid = np.linspace(0.05, 0.95, 19)
    x, y = np.meshgrid(grid, grid, indexing='ij')  # avoid the v == 0 centreline
    u, v, _w, p = ref.reference(x, y)
    assert np.std(u) > 1e-3 and np.std(v) > 1e-3 and np.std(p) > 1e-3
    assert np.max(np.abs(p)) > 1e-3


def test_independent_slopes_reproduce_the_published_summary(report):
    """Own numpy quadrature + own least squares vs the recorded summary."""
    published = report['cross_check_vs_summary']
    assert published is not None, 'summary.json missing for the committed triple'
    for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
        assert published[key]['slope_diff'] <= 1e-9, (key, published[key])
    assert published['verdict_slope_diff'] <= 1e-9


def test_ledger_criterion_holds_under_independent_estimator(report):
    """The >= 1.8 acceptance, re-derived without any project code path."""
    estimates = report['independent_estimates']
    assert estimates['velocity_full']['slope'] >= THRESHOLD
    assert estimates['velocity_inplane']['slope'] >= THRESHOLD
    assert estimates['pressure_meanfree']['slope'] >= THRESHOLD
    # monotone approach to second order, i.e. the triple really sits in the
    # asymptotic range rather than fitting a line by accident
    pairwise = estimates['velocity_full']['pairwise']
    assert pairwise[0] < pairwise[1] <= 2.0 + 0.05


def test_spanwise_share_is_round_off_under_the_symmetry_closure(report, ref):
    """zero_gradient must leave |w| at round-off; that is what licenses the 3D norm."""
    for level in report['levels']:
        npz = np.load(RUNS / f'n{level["n"]}' / f'n{level["n"]}-fields.npz')
        assert np.max(np.abs(npz['velocity'][:, 2])) <= 1e-12
    assert report['levels'][-1]['velocity_spanwise'] <= 1e-15

def test_centre_sampling_bias_is_second_order(report):
    """The finer the grid, the smaller the point-sampling truncation, at O(h^2).

    If the bias decayed only at first order it could bend the measured slope, so
    this must be measured rather than assumed.
    """
    sampling = report['sampling_bias']
    assert sampling is not None, 'summary.json missing'
    bias = sampling['per_level']
    assert all(bias[i] > bias[i + 1] > 0. for i in range(len(bias) - 1))
    assert sampling['order'] >= THRESHOLD


def test_richardson_estimates_are_internally_consistent(report):
    """Reported extrapolation must be a real O(h^p) extrapolation, not a rescale."""
    for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
        block = report['richardson'][key]
        assert block['order_gain_factor'] > 1.
        assert block['remaining_error_estimate_finest'] < report['independent_estimates'][key]['per_level'][-1]
        assert block['gci_finest'] < 2. * block['remaining_error_estimate_finest'] * 1.5
        # the two refinement segments must not disagree by more than 10% of p
        assert block['pairwise_spread'] <= 0.1 * block['observed_order']
