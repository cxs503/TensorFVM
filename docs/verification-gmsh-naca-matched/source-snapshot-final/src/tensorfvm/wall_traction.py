"""Laminar incompressible no-slip wall traction reconstruction.

At a stationary no-slip wall, tangential derivatives of velocity vanish.
Incompressibility then gives zero normal derivative of normal velocity. Thus
only the wall-normal derivative of tangential velocity contributes to viscous
traction. A linear normal-distance estimate is consistent to first order;
cell normal velocity belongs to higher-order terms and is not wall slip.
"""
import torch


def incompressible_wall_gradient(owner_velocity, displacement, area_vectors):
    normal=area_vectors/torch.linalg.vector_norm(area_vectors,dim=-1)[:,None]
    distance=(displacement*normal).sum(-1)
    if not torch.all(distance>0):
        raise ValueError('No-slip wall requires a positive owner-to-wall normal distance')
    tangential=owner_velocity-(owner_velocity*normal).sum(-1)[:,None]*normal
    return -normal[:,:,None]*tangential[:,None,:]/distance[:,None,None]
