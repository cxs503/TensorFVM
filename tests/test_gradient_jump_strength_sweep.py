"""Gates for the two-knob admissibility audit of the hybrid viscous operator.

The audited claims (see ``docs/gradient-jump-strength-sweep``) are:

1. on simplex meshes the Dirichlet-eliminated operator is singular for
   ``gradient_jump_stabilization = 0`` *whatever* the face-residual penalty is
   (the defect is structural, not a question of a stronger penalty);
2. on hexahedra the same pair of knobs is non-singular without the jump term,
   which is the negative control for claim 1;
3. both penalties annihilate affine fields exactly, so affine exactness cannot
   be used as evidence against claim 1 -- the singular simplex case solves the
   affine problem while returning a non-affine answer;
4. the jump strength has an interior error optimum on simplices.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'gradient_jump_sweep', ROOT / 'scripts' / 'audit_gradient_jump_strength_sweep.py')
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


@pytest.fixture(scope='module')
def simplex():
    return sweep.tet_mesh(3)


@pytest.fixture(scope='module')
def cartesian():
    return sweep.hex_mesh(4)


def test_kuhn_split_is_conforming_and_volume_exact():
    mesh = sweep.tet_mesh(3)
    assert len(mesh.cells) == 6 * 3 ** 3
    assert mesh.cell_volumes.sum() == pytest.approx(1., abs=1e-14)
    assert mesh.audit()['maximum_relative_closure'] < 1e-14


@pytest.mark.parametrize('stabilization', [1., 100.])
def test_simplex_operator_is_singular_without_the_jump_term(simplex, stabilization):
    row = sweep.case(simplex, 'tet', 3, stabilization, 0., spectrum=True)
    assert not row['solved'], 'jump-free simplex operator must fail the residual gate'
    assert row['spectrum_near_null_tight'] >= 5
    assert abs(row['spectrum_ratio']) < 1e-12


@pytest.mark.parametrize('jump', [.03, .1, 1.])
def test_simplex_operator_is_regular_with_the_jump_term(simplex, jump):
    row = sweep.case(simplex, 'tet', 3, 1., jump)
    assert row['solved']
    assert row['spectrum_near_null_tight'] == 0
    assert row['spectrum_ratio'] > 1e-3
    assert row['energy_balance_error'] < 1e-12
    assert row['linear_relative_residual'] < 1e-9


def test_hexahedron_operator_is_regular_without_the_jump_term(cartesian):
    row = sweep.case(cartesian, 'hex', 4, 1., 0.)
    assert row['solved']
    assert row['spectrum_near_null_tight'] == 0
    assert row['spectrum_ratio'] > 1e-3
    assert row['cell_relative_l2'] < .05


def test_jump_term_raises_the_coercivity_margin_on_hexahedra(cartesian):
    weak = sweep.case(cartesian, 'hex', 4, 1., .1, spectrum=True)
    strong = sweep.case(cartesian, 'hex', 4, 1., 1., spectrum=True)
    assert strong['spectrum_ratio'] > 1.2 * weak['spectrum_ratio']
    assert strong['cell_relative_l2'] < weak['cell_relative_l2']


@pytest.mark.parametrize('kind,size,stabilization,jump', [
    ('tet', 3, 1., 1.), ('tet', 3, 100., 10.), ('hex', 4, 1., 1.), ('hex', 4, 100., 10.)])
def test_affine_fields_are_reproduced_for_every_admissible_knob_pair(
        kind, size, stabilization, jump):
    mesh = sweep.tet_mesh(size) if kind == 'tet' else sweep.hex_mesh(size)
    row = sweep.case(mesh, kind, size, stabilization, jump, affine=True, spectrum=False)
    assert row['solved']
    assert row['cell_max_abs_error'] < 1e-11


def test_affine_exactness_cannot_detect_the_simplex_singularity(simplex):
    """The jump-free simplex operator returns a wrong answer but reports solved."""
    row = sweep.case(simplex, 'tet', 3, 1., 0., affine=True, spectrum=False)
    assert row['solved']
    assert row['cell_max_abs_error'] > 1e-1


def test_jump_strength_has_an_interior_error_optimum_on_simplices(simplex):
    errors = {jump: sweep.case(simplex, 'tet', 3, 1., jump)['cell_relative_l2']
              for jump in (.01, .1, .3, 10.)}
    assert errors[.1] < errors[.01]
    assert errors[.3] < errors[10.]
    assert min(errors.values()) == pytest.approx(errors[.3], rel=.05)


def test_documented_error_levels_are_reproduced(simplex):
    """Regression gate on the published numbers (relative L2 of the analytic field)."""
    tet = sweep.case(simplex, 'tet', 3, 1., 1.)
    hexa = sweep.case(sweep.hex_mesh(4), 'hex', 4, 1., 1.)
    assert tet['cell_relative_l2'] == pytest.approx(.050708, rel=.02)
    assert hexa['cell_relative_l2'] == pytest.approx(.014006, rel=.02)


def test_face_residual_penalty_too_weak_or_too_strong_costs_accuracy(cartesian):
    base = sweep.case(cartesian, 'hex', 4, 1., 1.)
    weak = sweep.case(cartesian, 'hex', 4, .1, 1.)
    strong = sweep.case(cartesian, 'hex', 4, 100., 1.)
    assert base['cell_relative_l2'] < weak['cell_relative_l2']
    assert base['cell_relative_l2'] < strong['cell_relative_l2']
    assert weak['cell_relative_l2'] > 10 * base['cell_relative_l2']