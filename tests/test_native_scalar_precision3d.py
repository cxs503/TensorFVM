"""Precision/constant-offset invariance of the actual scalar flux equations."""
import numpy as np
import pytest
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D


@pytest.mark.parametrize('correction',['none','nonlinear_mmp'])
def test_small_resolved_contrasts_survive_large_scalar_offset(correction):
    if np.finfo(np.longdouble).nmant<=np.finfo(float).nmant:pytest.skip('extended floating precision unavailable')
    m=channel_mesh((2,3,1));mask=np.zeros(len(m.owner),bool)
    op=NativeHybridScalar3D(m,.1,dirichlet_mask=mask,correction=correction)
    points=np.vstack((m.cell_centers,m.face_centers)).astype(np.longdouble)
    large=np.longdouble('1e8');phi=large+np.longdouble('1e-9')*(points[:,0]+2*points[:,1]-points[:,2])
    contrast=phi-large
    # These changes exist in the state but would be lost by float64 casting.
    assert np.ptp(phi.astype(float))==0 and np.ptp(phi)>0
    r=op.steady_residual(phi[:op.N],phi[op.N:])
    reference=op.steady_residual(contrast[:op.N],contrast[op.N:])
    assert np.linalg.norm(reference['total_flux'])>1e-10
    np.testing.assert_array_equal(r['total_flux'],reference['total_flux'])
    np.testing.assert_array_equal(r['free_residual'],reference['free_residual'])
    if correction=='nonlinear_mmp':
        frozen,_=op.mmp.operator(phi)
        op.diffusion=frozen
        cells,owner,neighbor=op._local_diffusion_reactions(phi)
        assembled=op.mmp.nonlinear_reaction(phi)
        local=np.r_[cells,owner+neighbor]
        scale=np.linalg.norm(assembled)
        assert np.linalg.norm(local-assembled)<1000*np.finfo(np.longdouble).eps*scale


def test_relative_work_rejects_a_bad_field_even_when_absolute_numbers_are_small():
    m=channel_mesh((2,3,1),triangulate_faces=False)
    absolute=NativeHybridScalar3D(m,.1)
    relative=NativeHybridScalar3D(m,.1,work_balance_policy='relative')
    solved=absolute.solve(boundary_values=m.face_centers[:,0])
    assert absolute.ledger_acceptable(solved) and relative.ledger_acceptable(solved)
    state=np.r_[solved['cell_values'],solved['face_values']];state[0]+=.001
    rows=[]
    for factor in (1.,2.**-20):
        r=absolute._solution_result(state*factor,np.zeros(len(m.owner)),np.zeros(len(m.cells)),
            np.zeros(len(m.cells)),np.zeros(len(m.cells)),0.,np.zeros(len(m.owner)),0.)
        assert not relative.ledger_acceptable(r)
        assert r['relative_work_balance']>1e-10
        rows.append(r)
    assert not absolute.ledger_acceptable(rows[0])
    assert absolute.ledger_acceptable(rows[1])  # demonstrates the unit-sensitive legacy ledger
    np.testing.assert_allclose(rows[0]['relative_work_balance'],rows[1]['relative_work_balance'],rtol=1e-14)


def test_relative_work_does_not_waive_local_flux_gates():
    m=channel_mesh(1,triangulate_faces=False)
    op=NativeHybridScalar3D(m,.1,work_balance_policy='relative')
    r=op.solve(boundary_values=m.face_centers[:,0])
    assert op.ledger_acceptable(r)
    for name in ('cell_balance_max','interface_flux_defect_max','global_balance'):
        bad=dict(r);bad[name]=2e-8
        assert not op.ledger_acceptable(bad)
    with pytest.raises(ValueError,match='work balance policy'):
        NativeHybridScalar3D(m,work_balance_policy='unknown')
