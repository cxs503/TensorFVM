import numpy as np
import pytest
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def channel_constraints(m):
    b=m.neighbor<0;x,y,z=m.face_centers.T
    mask=np.zeros((len(b),3),bool)
    mask[b & (y<1e-12)]=True
    mask[b & (y>1-1e-12),1]=True
    mask[b & ((z<1e-12)|(z>1-1e-12)),2]=True
    return mask


def test_stokes_rejects_hidden_global_rotations_before_factorization():
    m=channel_mesh((1,8,1),triangulate_faces=False)
    with pytest.raises(ValueError,match='2 global rigid motion'):
        NativeHybridStokes3D(m,channel_constraints(m))


def test_stokes_accepts_resolved_symmetry_extrusion():
    m=channel_mesh((2,8,2),triangulate_faces=False)
    s=NativeHybridStokes3D(m,channel_constraints(m))
    assert s.rigid_motion_constraints['rank']==6


def test_pure_traction_stokes_cannot_silently_pick_rigid_motion():
    m=channel_mesh(2)
    with pytest.raises(ValueError,match='6 global rigid motion'):
        NativeHybridStokes3D(m,np.zeros(len(m.owner),bool))
