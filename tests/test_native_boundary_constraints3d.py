import numpy as np
import pytest
from tensorfvm.native_boundary_constraints3d import rigid_motion_constraint_audit,require_rigid_motion_constraints
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def mask(mesh):
    b=mesh.neighbor<0;x,y,z=mesh.face_centers.T
    fixed=np.zeros((len(b),3),bool)
    fixed[b & (y<1e-12)]=True
    fixed[b & (y>1-1e-12),1]=True
    fixed[b & ((z<1e-12)|(z>1-1e-12)),2]=True
    return fixed


def test_single_extrusion_has_actual_unconstrained_zero_energy_rotations():
    m=channel_mesh((1,8,1),triangulate_faces=False);fixed=mask(m)
    report=rigid_motion_constraint_audit(m,fixed)
    assert report['rank']==4 and report['unconstrained_rigid_modes']==2
    x=np.vstack((m.cell_centers,m.face_centers))
    rotation=np.column_stack((-x[:,1],x[:,0]-.5,np.zeros(len(x))))
    np.testing.assert_allclose(rotation[len(m.cells):][fixed],0.,atol=1e-14)
    op=NativeHybridViscous3D(m,dirichlet_mask=fixed)
    np.testing.assert_allclose(op.full_matrix@rotation.ravel(),0.,atol=2e-13)
    with pytest.raises(ValueError,match='rigid motion'):
        require_rigid_motion_constraints(m,fixed)


def test_resolved_half_channel_constrains_all_six_rigid_modes():
    m=channel_mesh((2,8,2),triangulate_faces=False)
    report=require_rigid_motion_constraints(m,mask(m))
    assert report['rank']==6 and report['unconstrained_rigid_modes']==0
