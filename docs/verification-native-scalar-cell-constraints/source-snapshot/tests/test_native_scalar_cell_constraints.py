import numpy as np
import pytest
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


@pytest.mark.parametrize('correction',['none','nonlinear_mmp'])
def test_nonzero_internal_constraint_reaction_is_required_for_conservation(correction):
    m=channel_mesh((2,3,1),triangulate_faces=True);ids=np.array([2]);target=np.array([2.])
    op=NativeHybridScalar3D(m,.3,cell_constraints=(ids,target),correction=correction)
    r=op.solve(boundary_values=1.)
    np.testing.assert_array_equal(r['cell_values'][ids],target)
    assert op.ledger_acceptable(r)
    assert abs(r['global_balance_without_constraint'])>.1
    np.testing.assert_allclose(r['global_balance_without_constraint'],r['cell_constraint_reaction'].sum(),atol=1e-11)
    np.testing.assert_allclose(r['cell_balance_without_constraint']-r['cell_constraint_reaction'],r['cell_balance'],atol=0)
    assert r['cell_constraint_work']>.1
    np.testing.assert_allclose(r['cell_constraint_work'],target@r['cell_constraint_reaction'][ids],atol=1e-12)
    steady=op.steady_residual(r['cell_values'],r['face_values'],boundary_values=1.)
    assert steady['relative_residual']<1e-11 and steady['cell_constraint_error']==0
    assert steady['cell_equation_max']>.1 and steady['unconstrained_cell_equation_max']<1e-11
    assert np.count_nonzero(r['cell_constraint_reaction'])==1


@pytest.mark.parametrize('correction',['none','nonlinear_mmp'])
def test_prescribed_cells_include_storage_sink_and_advection_in_reaction(correction):
    m=channel_mesh((2,2,1));ids=np.arange(len(m.cells))
    op=NativeHybridScalar3D(m,.1,cell_constraints=(ids,2.),correction=correction)
    q=m.area_vectors@np.array([1.,.2,0.])
    r=op.solve(q,boundary_values=2.,inflow_values=2.,old_values=1.,dt=.25,sink=1.)
    np.testing.assert_allclose(r['cell_values'],2.,atol=0)
    np.testing.assert_allclose(r['face_values'],2.,atol=2e-12)
    np.testing.assert_allclose(r['cell_constraint_reaction'],6*m.cell_volumes,atol=1e-12)
    assert op.ledger_acceptable(r)
    assert r['global_balance_without_constraint']>1.
    np.testing.assert_allclose(r['cell_constraint_work'],12*m.cell_volumes.sum(),atol=1e-12)


def test_wrong_prescribed_value_cannot_pass_a_zero_equation_ledger():
    m=channel_mesh((1,1,1));op=NativeHybridScalar3D(m,cell_constraints=(np.array([0]),2.))
    # All nodes at 3 have zero diffusion and zero unconstrained equations,
    # but violate the interior constraint. No value is repaired in the audit.
    r=op._solution_result(np.full(op.N+op.F,3.),np.zeros(op.F),np.zeros(op.N),np.zeros(op.N),np.zeros(op.N),0.,None,0.)
    assert r['cell_balance_max']==0 and r['work_balance']==0
    assert r['cell_constraint_error']==1 and not op.ledger_acceptable(r)
    for correction in ('none','nonlinear_mmp'):
        fixed=NativeHybridScalar3D(m,cell_constraints=(np.array([0]),2.),correction=correction)
        result=fixed.solve(boundary_values=2.)
        assert not len(fixed.free) and fixed.ledger_acceptable(result)


@pytest.mark.parametrize('indices,values',[([0,0],[1.,2.]),([-1],[1.]),([8],[1.]),([.5],[1.]),([0],[np.nan])])
def test_invalid_cell_constraints_rejected(indices,values):
    with pytest.raises(ValueError):NativeHybridScalar3D(channel_mesh((2,2,1)),cell_constraints=(indices,values))


def test_sst_omega_constraints_report_their_reaction_and_current_mismatch():
    from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
    m=channel_mesh((2,3,1));op=NativeHybridSSTTransport3D(m,scalar_correction='nonlinear_mmp')
    ids=np.array([0,2]);values=np.array([3.,4.]);cells=len(m.cells);faces=len(m.owner)
    kwargs=dict(k_boundary=.01,omega_boundary=2.,omega_constraints=(ids,values))
    r=op.step(np.full(cells,.01),np.full(cells,2.),np.full(faces,.01),np.full(faces,2.),
        np.zeros(faces),.1,.1,np.zeros((cells,3,3)),.1,**kwargs)
    np.testing.assert_array_equal(r['omega'][ids],values)
    assert np.max(abs(r['transport']['omega']['cell_constraint_reaction']))>1
    assert r['transport']['omega']['cell_balance_max']<1e-10
    assert abs(r['transport']['omega']['global_balance'])<1e-10
    current=op.steady_residual(r['k'],r['omega'],r['k_faces'],r['omega_faces'],np.zeros(faces),.1,.1,
        np.zeros((cells,3,3)),**kwargs)
    assert current['omega']['cell_constraint_error']==0
    wrong=r['omega'].copy();wrong[ids[0]]+=.1
    bad=op.steady_residual(r['k'],wrong,r['k_faces'],r['omega_faces'],np.zeros(faces),.1,.1,
        np.zeros((cells,3,3)),**kwargs)
    assert bad['omega']['cell_constraint_error']>.09
    with pytest.raises(ValueError,match='constrained omega'):
        op.step(np.full(cells,.01),np.full(cells,2.),np.full(faces,.01),np.full(faces,2.),
            np.zeros(faces),.1,.1,np.zeros((cells,3,3)),.1,k_boundary=.01,omega_boundary=2.,
            omega_constraints=(ids,[0.,4.]))


def test_eliminated_cells_match_independent_lagrange_multiplier_system():
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    m=channel_mesh((2,3,1));ids=np.array([1,4]);values=np.array([1.7,2.3])
    op=NativeHybridScalar3D(m,.2,cell_constraints=(ids,values))
    bd=1.+.1*m.face_centers[:,0];q=m.area_vectors@np.array([.4,-.2,0.])
    r=op.solve(q,boundary_values=bd,inflow_values=bd,source=.2,sink=.4,dt=.3,old_values=.7)
    # Solve the uneliminated equations with independent multiplier unknowns.
    incoming=(m.neighbor<0)&(q<0);fids=np.flatnonzero(~incoming)
    upstream=np.where(q>=0,m.owner,m.neighbor)
    adv=sparse.coo_matrix((q[fids],(fids,upstream[fids])),shape=(op.F,op.N)).tocsr()
    cell=op.D@adv+sparse.diags(m.cell_volumes*(.4+1/.3))
    matrix=op.diffusion.full_matrix+sparse.block_diag((cell,sparse.csr_matrix((op.F,op.F))))
    original_free=op.diffusion.free;prescribed=op.diffusion._prescribed(bd[:,None])
    forcing=np.zeros(op.N+op.F);inflow=np.where(incoming,q*bd,0.)
    forcing[:op.N]=m.cell_volumes*(.2+.7/.3)-op.D@inflow
    a=matrix[original_free][:,original_free]
    c=sparse.coo_matrix((np.ones(len(ids)),(np.arange(len(ids)),ids)),shape=(len(ids),len(original_free)))
    rhs=np.r_[(forcing-matrix@prescribed)[original_free],values]
    answer=spsolve(sparse.bmat([[a,-c.T],[c,None]],format='csc'),rhs)
    state=prescribed.copy();state[original_free]=answer[:-len(ids)]
    np.testing.assert_allclose(np.r_[r['cell_values'],r['face_values']],state,atol=3e-12)
    np.testing.assert_allclose(r['cell_constraint_reaction'][ids],answer[-len(ids):],atol=3e-12)
    assert op.ledger_acceptable(r)
