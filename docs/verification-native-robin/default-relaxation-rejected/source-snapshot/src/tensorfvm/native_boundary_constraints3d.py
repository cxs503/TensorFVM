"""Necessary rigid-motion constraint check for native vector energy solves.

Face constraints act on face means. A wall represented by a single face can
leave a global rotation invisible even when the continuum wall is no-slip.
This six-mode audit is necessary, not a proof of full discrete coercivity.
"""
import numpy as np


def rigid_motion_constraint_audit(mesh,dirichlet_components,robin_coefficient=None):
    mask=np.asarray(dirichlet_components,bool)
    if mask.shape!=(len(mesh.owner),3) or np.any(mask & (mesh.neighbor>=0)[:,None]):
        raise ValueError('boundary component mask required')
    center=np.mean(mesh.vertices,axis=0)
    length=float(np.max(np.ptp(mesh.vertices,axis=0)))
    if not np.isfinite(length) or length<=0:raise ValueError('nondegenerate mesh required')
    x,y,z=((mesh.face_centers-center)/length).T
    modes=np.zeros((len(mask),3,6));modes[:,:,:3]=np.eye(3)
    modes[:,0,4]=z;modes[:,0,5]=-y
    modes[:,1,3]=-z;modes[:,1,5]=x
    modes[:,2,3]=y;modes[:,2,4]=-x
    constrained=modes[mask]
    if robin_coefficient is not None:
        from .native_robin3d import validate_robin
        c=validate_robin(mesh,robin_coefficient,3,mask)
        values,vectors=np.linalg.eigh(c)
        active=values>1e-12*np.max(abs(values),axis=1)[:,None]
        # Normalize constrained directions: this checks rank, not conditioning.
        robin_modes=np.einsum("fij,fik->fjk",vectors,modes)[active]
        constrained=np.vstack((constrained,robin_modes))
    if len(constrained):
        values=np.linalg.svd(constrained,compute_uv=False)
        rank=int(np.count_nonzero(values>1e-12*max(float(values.max(initial=0)),1.)))
    else:values=np.zeros(0);rank=0
    return dict(rank=rank,unconstrained_rigid_modes=6-rank,
                singular_values=values,necessary_uniqueness_condition_passed=rank==6)


def require_rigid_motion_constraints(mesh,dirichlet_components,robin_coefficient=None):
    result=rigid_motion_constraint_audit(mesh,dirichlet_components,robin_coefficient)
    if result['rank']<6:
        raise ValueError(f"velocity face constraints leave {6-result['rank']} global rigid motion(s) undetermined; add resolved boundary constraints")
    return result
