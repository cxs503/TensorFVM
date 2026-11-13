"""Symmetric dissipative Robin data for native shared-face unknowns.

Convention: outward total traction + C u = g, per unit area.
C is positive semidefinite. For a moving frictional wall, g includes C u_wall.
This module does not impose impermeability or implement a turbulence wall law.
"""
import numpy as np
from scipy import sparse


def validate_robin(mesh,coefficient,components,dirichlet_components):
    shape=(len(mesh.owner),components,components)
    if coefficient is None:return np.zeros(shape)
    c=np.asarray(coefficient,float)
    if c.shape!=shape or not np.isfinite(c).all():raise ValueError('finite face Robin tensors with explicit component dimensions required')
    if np.any(c[mesh.neighbor>=0]!=0):raise ValueError('Robin coefficients may act only on boundary faces')
    mask=np.asarray(dirichlet_components,bool)
    if np.any(np.where(mask[:,:,None]|mask[:,None,:],c,0)!=0):
        raise ValueError('Robin components must be free of Dirichlet constraints')
    scale=np.max(abs(c),axis=(1,2))
    if np.any(np.max(abs(c-c.transpose(0,2,1)),axis=(1,2))>1e-12*np.maximum(scale,1e-300)):
        raise ValueError('symmetric Robin coefficients required')
    if np.any(np.linalg.eigvalsh(c)[:,0]<-1e-12*np.maximum(scale,1e-300)):
        raise ValueError('positive semidefinite Robin coefficients required')
    return c.copy()


def robin_matrix(mesh,coefficient):
    n=len(mesh.cells);f,components,_=coefficient.shape
    face,i,j=np.nonzero(coefficient)
    area=np.linalg.norm(mesh.area_vectors,axis=1)
    return sparse.coo_matrix((area[face]*coefficient[face,i,j],
        (components*(n+face)+i,components*(n+face)+j)),
        shape=(components*(n+f),components*(n+f))).tocsr()


def tangential_friction(mesh,faces,coefficient):
    """Return beta (I-n n^T); the caller must separately constrain normal flow."""
    faces=np.asarray(faces,bool)
    if faces.shape!=(len(mesh.owner),) or np.any(faces & (mesh.neighbor>=0)):
        raise ValueError('boundary face selection required')
    beta=np.broadcast_to(np.asarray(coefficient,float),(len(faces),))
    if not np.isfinite(beta).all() or np.any(beta<0):raise ValueError('nonnegative finite friction required')
    normal=mesh.area_vectors/np.linalg.norm(mesh.area_vectors,axis=1)[:,None]
    c=beta[:,None,None]*(np.eye(3)-normal[:,:,None]*normal[:,None,:])
    c[~faces]=0
    return c
