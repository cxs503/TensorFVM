"""Thin-layer passivity gates for the implicit native viscous operator.

The open defect in ``docs/verification-native-solver-audit/README.md`` is that
``L = -momentum_diffusion_matrix`` is not dissipative on the hybrid SUBOFF mesh,
and its stated next step is the *thin-layer / tetrahedral interface* regime.
``tensorfvm.verification.native_energy`` covers two-cell tetrahedral pairs only;
these gates pin the envelope measured by
``scripts/audit_thin_layer_viscous_passivity.py``:

* the shipped default (``over-relaxed``) is passive on every sampled thin-layer
  family up to aspect ratio 50 and 87 deg nonorthogonality;
* the ``minimum`` correction and the uncorrected (``none``) flux are not, and no
  damping factor restores passivity for ``minimum`` -- the defect is not
  removable by weakening the correction;
* damping the ``over-relaxed`` correction below ``eta ~ 0.1`` destroys
  passivity, while affine-flux consistency is unaffected by damping to
  round-off.

Numbers are the values recorded in
``docs/verification-thin-layer-viscous-passivity/passivity-envelope.json``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import affine_consistency, passivity_audit

ROOT = Path(__file__).resolve().parents[1]


def _script():
    path = ROOT/'scripts'/'audit_thin_layer_viscous_passivity.py'
    spec = importlib.util.spec_from_file_location('thin_layer_passivity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = pytest.importorskip  # noqa: F841  (documentation only)
SCRIPT = _script()


def _tetra(aspect=20., shear=0.):
    return SCRIPT.tetra_box(2, aspect, shear)


def test_over_relaxed_is_passive_across_thin_layer_families():
    """Default correction: no energy-creating mode anywhere in the sampled grid."""
    for name, builder in SCRIPT.FAMILIES.items():
        for aspect in (2., 5., 20.):
            for shear in (0., 0.4):
                mesh = builder(2, aspect, shear)
                spectrum = passivity_audit(mesh, 'over-relaxed')['vector']
                assert spectrum['passive'], (name, aspect, shear, spectrum['lambda_min'])
                assert spectrum['creation_rate'] == 0.
                assert spectrum['lambda_min'] > 0.


def test_thin_tetrahedra_break_the_minimum_and_uncorrected_corrections():
    """AR=20 tetra box, 86.16 deg: recorded creation fractions are reproduced."""
    mesh = _tetra()
    minimum = passivity_audit(mesh, 'minimum')
    none = passivity_audit(mesh, 'none')
    assert minimum['maximum_nonorthogonality_degrees'] == pytest.approx(86.162232, rel=1e-6)
    assert minimum['vector']['passive'] is False
    assert minimum['vector']['creation_fraction'] == pytest.approx(0.079399, rel=5e-3)
    assert minimum['vector']['lambda_min'] == pytest.approx(-2.07769, rel=5e-3)
    assert none['vector']['passive'] is False
    assert none['vector']['lambda_min'] == pytest.approx(-2.333852, rel=5e-3)


def test_weakened_corrections_cannot_be_repaired_by_less_correction():
    """``none`` fails even on the coarsest orthogonal box: the correction is load bearing."""
    for name, builder in SCRIPT.FAMILIES.items():
        spectrum = passivity_audit(builder(2, 1., 0.), 'none')['vector']
        assert spectrum['passive'] is False, name
        assert spectrum["creation_fraction"] > 1e-4


def test_damping_window_and_free_affine_consistency():
    """Over-relaxed survives damping down to eta ~ 0.1; minimum survives none."""
    mesh = _tetra()
    assert SCRIPT._damped_passivity(mesh, 'over-relaxed', 1.)['passive'] is True
    assert SCRIPT._damped_passivity(mesh, 'over-relaxed', 0.11)['passive'] is True
    assert SCRIPT._damped_passivity(mesh, 'over-relaxed', 0.05)['passive'] is False
    assert SCRIPT._damped_passivity(mesh, 'minimum', 1.)['passive'] is False
    line = SCRIPT.damping_line(mesh, 'over-relaxed', samples=3, iterations=20)
    assert line['minimum_safe_damping'] == pytest.approx(0.1015, abs=5e-3)


def test_affine_traction_is_exact_for_every_damping_factor():
    """The deferred correction vanishes for affine fields, so damping costs nothing."""
    mesh = _tetra()
    gradient = np.outer(np.array([.7, -.3, .2]), [1., 1., 1.])
    stress = gradient+gradient.T-(2/3)*np.trace(gradient)*np.eye(3)
    exact = np.einsum('aj,fj->fa', stress, mesh.area_vectors)
    for damping in (0., .3, 1.):
        viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1., correction='over-relaxed',
                                  correction_damping=damping)
        computed = viscous.momentum_flux(mesh.cell_centers @ gradient.T, 0.,
                                         mesh.face_centers @ gradient.T)
        assert np.abs(computed-exact).max()/np.abs(exact).max() < 1e-12
    reference = affine_consistency(mesh, 'over-relaxed')
    assert reference['maximum_relative_error'] < 1e-12