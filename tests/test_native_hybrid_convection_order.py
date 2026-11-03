"""Truncation-order regression tests for NativeHybridConvection3D (verification debt).

The merged operator previously only had algebraic identity tests.  These tests
pin the measured convergence order against an independent analytic reference
(see ``scripts/verify_native_hybrid_convection_order.py``) so that a future
change of the upwind/deferred reconstruction cannot silently lose an order.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).parents[1] / 'scripts/verify_native_hybrid_convection_order.py'
spec = importlib.util.spec_from_file_location('convection_order_evidence', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LEVELS = (2, 4)


def observed(family, scheme):
    order, mode = module.SCHEMES[scheme]
    errors = [module.face_error(module.FAMILIES[family](n), order, mode)['relative_error']
              for n in LEVELS]
    return module.observed_order(errors), errors


@pytest.mark.parametrize('family', ['hex', 'wedge', 'tetra'])
def test_first_order_upwind_is_first_order(family):
    orders, _ = observed(family, 'first_order_upwind')
    assert orders[-1] == pytest.approx(1., abs=.12), orders


@pytest.mark.parametrize('family', ['hex', 'wedge', 'tetra'])
@pytest.mark.parametrize('scheme', ['second_order_exact_gradient',
                                    'second_order_gauss_gradient'])
def test_deferred_linear_upwind_is_second_order(family, scheme):
    orders, _ = observed(family, scheme)
    assert orders[-1] == pytest.approx(2., abs=.12), orders


@pytest.mark.parametrize('family', ['hex', 'wedge', 'tetra'])
def test_second_order_reconstruction_is_affine_exact(family):
    anchor = module.affine_anchor(family)
    assert anchor['second_order_affine_error'] < 1e-14
    # Upwind transports the upstream cell value, so it is *not* affine exact;
    # the residual is the leading O(h) term and must stay clearly nonzero.
    assert anchor['first_order_affine_error'] > 1e-4


@pytest.mark.parametrize('family', ['hex', 'wedge', 'tetra'])
def test_energy_ledger_and_momentum_balance_close_to_roundoff(family):
    mesh = module.FAMILIES[family](LEVELS[-1])
    for order, mode in module.SCHEMES.values():
        row = module.face_error(mesh, order, mode)
        assert row['energy_identity_error'] < 1e-12
        assert row['momentum_balance_error'] < 1e-12
        assert row['assembly_error'] < 1e-12


@pytest.mark.parametrize('family', ['hex', 'wedge', 'tetra'])
def test_refinement_families_are_conforming_and_volume_preserving(family):
    for n in LEVELS:
        mesh = module.FAMILIES[family](n)
        assert np.sum(mesh.cell_volumes) == pytest.approx(1., abs=1e-13)
        assert np.all(mesh.cell_volumes > 0)
        # every internal face has two distinct owners and no dangling neighbour
        interior = mesh.neighbor >= 0
        assert np.all(mesh.neighbor[interior] != mesh.owner[interior])
        assert len(mesh.owner) == len(mesh.face_vertices)


def test_analytic_reference_agrees_with_divergence_form():
    """Independent cross-check: quadrature flux vs V * div(u x u) at cell centres."""
    mesh = module.hex_box(2)
    face = module.exact_face_flux(mesh)
    integrated = np.zeros((len(mesh.cells), 3))
    np.add.at(integrated, mesh.owner, face)
    interior = mesh.neighbor >= 0
    np.add.at(integrated, mesh.neighbor[interior], -face[interior])
    field = module.abc_fields(mesh.cell_centers)
    divergence_form = np.einsum('cij,cj->ci', field['velocity_gradient'],
                                field['velocity']) * mesh.cell_volumes[:, None]
    scale = np.linalg.norm(divergence_form)
    # The divergence form is a cell-centre value times V, so it differs from
    # the exact integral at O(h^2); h = 1/2 here, hence the 2 % band.
    assert np.linalg.norm(integrated - divergence_form) / scale < 2e-2