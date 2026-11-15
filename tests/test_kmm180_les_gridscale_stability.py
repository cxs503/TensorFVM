"""Regression gate for the Re_tau = 180 LES grid-scale dissipation diagnosis.

The claims under test are the frozen criteria of
:mod:`tensorfvm.verification.kmm180_les_gridscale_stability`:

* the amplification is **broadband** (no single mode), so odd-even decoupling of
  the central-difference convection is not the cause;
* the wall-normal operator identity and the channel mirror symmetry hold, so the
  divergence is not a boundary/operator defect;
* the divergence is removed by grid-scale dissipation (``Cs >= 0.2``) and is made
  *worse* by refinement at fixed ``Cs``;
* the default laminar initial condition confounds the historical step counts.

Every test asserts an invariant or a falsifiable inequality.  A run that reaches
the step cap is reported as "did not diverge", never as "stable" -- the semantic
test :func:`test_capped_run_is_not_a_stability_claim` pins that down.
"""
import json

import numpy as np
import pytest

from tensorfvm.verification import kmm180_les_gridscale_stability as mod
from tensorfvm.verification.kmm180_les_gridscale_stability import (
    PREDECLARED, REFINED, SMALL, evaluate, growth_spectrum, mirror_symmetry_defect,
    steps_to_divergence, wall_normal_operator_identity,
)

FAST_CAP = 150


@pytest.fixture(scope='module')
def record():
    """One evaluation shared by the critique tests (the sweeps cost ~12 s)."""
    return evaluate(cap=FAST_CAP)


# --------------------------------------------------------------------------- #
# Frozen criteria
# --------------------------------------------------------------------------- #
def test_frozen_criteria_are_complete():
    ids = [c['id'] for c in PREDECLARED]
    assert ids == ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    assert len(set(ids)) == len(ids)
    for spec in PREDECLARED:
        assert spec['statement'] and spec['falsified_if'] and spec['metric']


def test_every_criterion_is_graded_and_can_fail(record):
    verdicts = {c['id']: c['verdict'] for c in record['criteria']}
    assert set(verdicts) == {c['id'] for c in PREDECLARED}
    assert all(v in ('PASS', 'FAIL', 'NOT_GRADED') for v in verdicts.values())
    # the record must state how many criteria it actually graded -- a record that
    # grades nothing must not look like a record that passed
    assert record['graded'] == sum(1 for v in verdicts.values() if v != 'NOT_GRADED')
    assert record['passed'] == sum(1 for v in verdicts.values() if v == 'PASS')
    for entry in record['criteria']:
        assert 'measured' in entry


def test_grading_is_not_vacuous(record):
    """The frozen criteria must pass *because* of the measurements, not by default."""
    for entry in record['criteria']:
        if entry['verdict'] == 'PASS':
            assert entry['measured'] is not None
            assert any(v is not None for v in entry['measured'].values())


# --------------------------------------------------------------------------- #
# Invariants: operator + symmetry
# --------------------------------------------------------------------------- #
def test_wall_normal_operator_reproduces_explicit_stencil():
    """The IMEX split must not hide an operator: identity to round-off."""
    result = wall_normal_operator_identity(SMALL)
    assert result['max_relative_defect'] < 1e-12


def test_operator_identity_needs_a_wall_normal_field():
    """Guard the methodology: a 3-D field makes the check meaningless.

    ``lap - _y_apply`` is the *transverse* operator on a general field, so the
    same measurement on a three-dimensional field is O(1) and says nothing about
    the wall-normal operator.  This test pins the distinction so the check cannot
    be "fixed" by loosening the threshold.
    """
    from tensorfvm.channel_les3d import ChannelLES3D, ChannelLESConfig

    nx, ny, nz = SMALL
    solver = ChannelLES3D(ChannelLESConfig(nx=nx, ny=ny, nz=nz))
    solver.initialise(noise=0.1, profile='reichardt')
    nut = solver.eddy_viscosity()
    cell, _ = solver._y_coefficients(solver.nu + nut)
    lap_u, _, _ = solver.diffusion(nut)
    defect_3d = np.abs(lap_u - solver._y_apply(cell, solver.u)).max()
    assert defect_3d > 1e-6  # i.e. the 3-D measurement is *not* a defect measure


def test_channel_mirror_symmetry_holds():
    """Top and bottom walls are treated identically by the discrete scheme."""
    result = mirror_symmetry_defect(SMALL, steps=60)
    assert result['defect'] is not None
    assert result['defect'] < 0.05
    assert result['bottom_row_share'] > 0
    assert result['top_row_share'] > 0


# --------------------------------------------------------------------------- #
# The falsifiable claims
# --------------------------------------------------------------------------- #
def test_default_smagorinsky_diverges_but_higher_cs_survives():
    """Cs = 0.1 (the shipping default) is below the dissipation threshold."""
    default = steps_to_divergence(grid=SMALL, cs=0.1, cap=FAST_CAP)
    assert default['diverged'] is True
    assert 40 < default['steps'] < FAST_CAP

    strong = steps_to_divergence(grid=SMALL, cs=0.2, cap=FAST_CAP)
    assert strong['capped'] is True and strong['diverged'] is False


def test_refinement_at_fixed_cs_does_not_cure_the_divergence():
    coarse = steps_to_divergence(grid=SMALL, cs=0.1, cap=FAST_CAP)
    refined = steps_to_divergence(grid=REFINED, cs=0.1, cap=FAST_CAP)
    assert refined['diverged'] is True
    assert refined['steps'] < coarse['steps']


def test_default_initial_condition_confounds_the_step_count():
    """The laminar default profile peaks at Re_tau/2 = 90 u_tau at Re_tau = 180."""
    laminar = steps_to_divergence(grid=SMALL, cs=0.1, profile='parabolic', cap=FAST_CAP)
    turbulent = steps_to_divergence(grid=SMALL, cs=0.1, profile='reichardt', cap=FAST_CAP)
    assert laminar['diverged'] and turbulent['diverged']
    assert laminar['steps'] < turbulent['steps']

    from tensorfvm.channel_les3d import ChannelLES3D, ChannelLESConfig
    solver = ChannelLES3D(ChannelLESConfig(nx=16, ny=16, nz=8))
    solver.initialise(noise=0.0, profile='parabolic')
    # the laminar parabola is 5x the turbulent peak (~18 u_tau), not a detail
    assert np.abs(solver.u).max() == pytest.approx(90.0, rel=0.02)


def test_amplification_is_broadband_not_odd_even():
    """No single streamwise mode dominates the growing disturbance."""
    spectrum = growth_spectrum(grid=SMALL, cs=0.1, dt=1.25e-3, steps=240, every=5)
    assert spectrum['samples'] >= 20
    assert spectrum['max_mode_share'] is not None
    assert spectrum['max_mode_share'] < 0.5
    # the growth itself must be positive, otherwise "broadband" is vacuous
    assert max(spectrum['kx_rates']) > 0.0
    assert sum(spectrum['kx_shares']) == pytest.approx(1.0, rel=1e-9)


def test_capped_run_is_not_a_stability_claim():
    """A run that reached the cap reports ``capped``/``diverged`` unambiguously."""
    run = steps_to_divergence(grid=SMALL, cs=0.3, cap=40)
    assert run['capped'] is True
    assert run['diverged'] is False
    assert run['steps'] == 40


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def test_cli_writes_machine_readable_record(tmp_path):
    out = tmp_path / 'les'
    mod.main(['--out', str(out), '--cap', str(FAST_CAP)])
    record = json.loads((out / 'summary.json').read_text())
    assert record['anchor'] == 'CASE-FVM-CHANNEL-KMM180'
    assert len(record['criteria']) == len(PREDECLARED)
    assert (out / 'cs_sweep.csv').exists()
    assert (out / 'modes_diverging.csv').exists()
    header = (out / 'cs_sweep.csv').read_text().splitlines()[0].split(',')
    assert 'cs' in header and 'capped' in header