"""Unforced Couette flow between moving walls with finite linear slip friction.

The exact shear is tau=1/(1/mu+1/beta_lower+1/beta_upper).
Only physical boundary values/tractions are supplied; interiors start at zero.
This validates a linear Robin boundary, not a turbulence wall function.
"""
import numpy as np
from tensorfvm.native_robin3d import tangential_friction
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from .native_hybrid_flow import channel_mesh


def slip_couette_solver(n=4,mu=.7,method='direct',ns=False):
    if not isinstance(n,int) or n<2 or not np.isfinite(mu) or mu<=0:raise ValueError('n>=2 and positive viscosity required')
    m=channel_mesh((2,n,2),triangulate_faces=True)
    b=m.neighbor<0;x,y,z=m.face_centers.T
    low=b & (y<1e-12);high=b & (y>1-1e-12);walls=low|high
    side=b & ((z<1e-12)|(z>1-1e-12));ends=b & ((x<1e-12)|(x>1-1e-12))
    mask=np.zeros((len(b),3),bool);mask[walls,1]=True;mask[side,2]=True
    beta=np.where(high,5.,2.);c=tangential_friction(m,walls,beta)
    tau=1/(1/mu+1/2+1/5)
    normal=m.area_vectors/np.linalg.norm(m.area_vectors,axis=1)[:,None]
    g=np.zeros((len(b),3));g[high,0]=5.;g[ends,1]=tau*normal[ends,0]
    values=np.zeros_like(g);values[:,0]=tau*(y/mu+1/2)
    cls=NativeHybridNavierStokes3D if ns else NativeHybridStokes3D
    extra=dict(backflow_velocity=values,convection_order=2) if ns else {}
    s=cls(m,mask,values,viscosity=mu,boundary_traction=g,robin_coefficient=c,coupled_linear_method=method,**extra)
    expected=np.zeros((len(m.cells),3));expected[:,0]=tau*(m.cell_centers[:,1]/mu+1/2)
    return m,s,expected,tau,low,high,g

