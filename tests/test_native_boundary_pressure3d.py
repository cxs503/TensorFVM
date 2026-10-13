import numpy as np
import pytest
from tensorfvm.native_boundary_pressure3d import NativeBoundaryPressureTrace3D
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def test_affine_trace_on_actual_curved_suboff_layers_without_wall_pressure_input():
    points,cells,kinds,wall,outer,_=structured_layers(axial=12,azimuthal=8,layers=3,first_height=.0004)
    m=NativeMesh3D.from_cells(points,cells,kinds,{**{tuple(f):'wall' for f in wall},**{tuple(f):'outer' for f in outer}},triangulate_faces=True)
    gradient=np.array([.7,-1.3,2.1]);p=.23+m.cell_centers@gradient
    trace=NativeBoundaryPressureTrace3D(m,m.patch_labels=='wall').reconstruct(p)
    exact=.23+m.face_centers[trace['face_indices']]@gradient
    np.testing.assert_allclose(trace['pressure'],exact,atol=2e-12,rtol=0)
    np.testing.assert_allclose(trace['gradient'],np.tile(gradient,(len(trace['gradient']),1)),atol=1e-10,rtol=0)
    assert not trace['wall_Cp_certified'] and not trace['changes_momentum_equations']


def test_quadratic_trace_improves_with_refinement_and_preserves_pressure_gauge_shift():
    errors=[]
    for n in (3,6,12):
        m=channel_mesh(n,triangulate_faces=False);b=m.neighbor<0
        op=NativeBoundaryPressureTrace3D(m,b)
        p=np.sum(m.cell_centers**2,axis=1);r=op.reconstruct(p)
        exact=np.sum(m.face_centers[b]**2,axis=1)
        errors.append(np.max(abs(r['pressure']-exact)))
        np.testing.assert_allclose(op.reconstruct(p+7)['pressure'],r['pressure']+7,atol=1e-14,rtol=0)
    assert errors[1]<errors[0]/3 and errors[2]<errors[1]/3


def test_unresolved_internal_stencil_rejected_instead_of_assuming_zero_wall_gradient():
    m=channel_mesh((1,4,1),triangulate_faces=False)
    with pytest.raises(ValueError,match='rank-deficient'):
        NativeBoundaryPressureTrace3D(m,m.neighbor<0)


def test_reported_pressure_noise_bound_matches_independent_unit_perturbations():
    m=channel_mesh(3,triangulate_faces=False);op=NativeBoundaryPressureTrace3D(m,m.neighbor<0)
    measured=np.zeros(len(op.face_indices));zero=np.zeros(len(m.cells))
    for cell in range(len(zero)):
        perturbation=zero.copy();perturbation[cell]=1.
        measured+=abs(op.reconstruct(perturbation)['pressure'])
    np.testing.assert_allclose(op.reconstruct(zero)['pressure_noise_amplification'],measured,rtol=1e-13,atol=1e-14)
