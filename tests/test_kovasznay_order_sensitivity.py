"""The Kovasznay order verdict must be able to say *no*.

``tests/test_kovasznay_32_64_128_acceptance.py`` checks that the committed
32/64/128 evidence reproduces the ledger claim (log-log velocity slope >= 1.8).
A slope estimator that returned ~2 for any scheme would satisfy that test just as
well, so the claim is only meaningful together with a run in which the expected
order is *different* and the estimator follows it.

The control series under ``runs-32-64-128-firstorder`` is the same case, mesh
family and extrusion closure as the acceptance series, with upwind
(``convection_order=1``) convection.  Classical first-order upwind has an O(h)
velocity error, so the estimator must return a slope near 1 and the >= 1.8 gate
must reject it.

Every number here is recomputed from the saved cell-centred fields
(``n<N>-fields.npz``); ``summary.json`` / ``cases.json`` are only consulted for
the run configuration, so a hand-edited summary cannot make this test pass.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_manufactured3d import volume_weighted_error

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
REFERENCE = ROOT / 'runs-32-64-128-zero_gradient'
CONTROL = ROOT / 'runs-32-64-128-firstorder'
GRIDS = (32, 64, 128)
THRESHOLD = 1.8
#: A control slope above this would not be rejected by the ledger gate, i.e. a
#: genuine first-order defect would pass unnoticed.
CONTROL_CEILING = 1.5
#: Separation between the two series that a real order change must show.
MIN_MARGIN = .5


def _series(directory):
    if not all((directory / f'n{n}' / 'cases.json').is_file() for n in GRIDS):
        pytest.skip(f'control artifacts absent: {directory}')
    levels = []
    for n in GRIDS:
        level_dir = directory / f'n{n}'
        row = json.loads((level_dir / 'cases.json').read_text())[0]
        fields = np.load(level_dir / f'n{n}-fields.npz')
        volumes = fields['cell_volumes']
        levels.append(dict(
            n=n, h=float(row['h']), cells=int(row['cells']),
            converged=bool(row['numerically_converged']),
            reynolds=row['reynolds'], convection_order=row['convection_order'],
            symmetry_closure=row['symmetry_closure'],
            velocity_full=volume_weighted_error(fields['velocity'], fields['exact_velocity'],
                                                volumes),
            pressure_meanfree=volume_weighted_error(fields['pressure'],
                                                    fields['exact_pressure'], volumes,
                                                    remove_mean=True)))
    return levels


def _slope(levels, key):
    slope, _ = np.polyfit(np.log([level['h'] for level in levels]),
                          np.log([level[key] for level in levels]), 1)
    return float(slope)


def test_control_differs_from_reference_only_in_convection_order():
    reference, control = _series(REFERENCE), _series(CONTROL)
    for fine, coarse in zip(reference, control):
        assert fine['reynolds'] == coarse['reynolds'] == pytest.approx(40.)
        assert fine['symmetry_closure'] == coarse['symmetry_closure']
        assert fine['h'] == coarse['h']
        assert fine['cells'] == coarse['cells']
        assert fine['convection_order'] == 2
        assert coarse['convection_order'] == 1, (
            'the control must be the upwind scheme, otherwise the slope comparison '
            'is between two second-order runs and proves nothing')


def test_control_series_converged_and_second_order_retained():
    reference, control = _series(REFERENCE), _series(CONTROL)
    for level in reference + control:
        assert level['converged'] is True, f'n={level["n"]} did not converge'
    reference_slope = _slope(reference, 'velocity_full')
    assert reference_slope >= THRESHOLD, (
        f'reference velocity slope {reference_slope:.4f} below ledger threshold {THRESHOLD}')


def test_upwind_control_is_rejected_by_the_ledger_gate():
    control = _series(CONTROL)
    slope = _slope(control, 'velocity_full')
    assert slope <= CONTROL_CEILING, (
        f'upwind control slope {slope:.4f} is not below {CONTROL_CEILING}: the >= {THRESHOLD} '
        'gate would not detect a first-order convection defect')
    assert slope < THRESHOLD


def test_slope_margin_separates_the_two_schemes():
    reference_slope = _slope(_series(REFERENCE), 'velocity_full')
    control_slope = _slope(_series(CONTROL), 'velocity_full')
    assert reference_slope - control_slope >= MIN_MARGIN, (
        f'margin {reference_slope - control_slope:.4f} below {MIN_MARGIN}: the fitted slope does '
        'not discriminate the scheme order, so the acceptance number is not evidence of order')


def test_control_error_decays_monotonically():
    control = _series(CONTROL)
    errors = [level['velocity_full'] for level in control]
    assert errors[0] > errors[1] > errors[2], errors