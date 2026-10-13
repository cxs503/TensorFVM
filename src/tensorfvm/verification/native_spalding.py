"""Unforced plane Couette verification for the cell-sampled Spalding boundary.

Constant bulk viscosity is explicit, not a solved SST eddy viscosity. The
one-dimensional stress balance determines analytical cell velocities and the
incoming profile; interiors start from the native Stokes initialization.
"""
import numpy as np
from scipy.optimize import brentq
from tensorfvm.native_spalding3d import NativeSpaldingWall3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.wall_functions import u_plus_at_y_plus
from .native_hybrid_flow import channel_mesh


def spalding_couette_solver(n=4,mu=.02,nu=.001,wall_speed=0.):
    mesh=channel_mesh((2,n,2),triangulate_faces=True)
    b=mesh.neighbor<0;x,y,z=mesh.face_centers.T
    low=b&(y<1e-12);top=b&(y>1-1e-12)
    side=b&((z<1e-12)|(z>1-1e-12));ends=b&((x<1e-12)|(x>1-1e-12))
    d=.5/n
    # Independent scalar shear balance: top speed = wall-law cell speed +
    # constant-stress bulk increment from the first cell center to top wall.
    ut=brentq(lambda t:t*u_plus_at_y_plus(d*t/nu)+t*t*(1-d)/mu-1.,0.,np.sqrt(mu/(1-d)),xtol=1e-15)
    tau=ut*ut;uc=ut*u_plus_at_y_plus(d*ut/nu)
    profile=lambda y:wall_speed+uc+(y-d)*tau/mu
    mask=np.zeros((len(b),3),bool);mask[top]=True;mask[low,1]=True
    mask[side,2]=True;mask[ends,1:]=True
    values=np.zeros((len(b),3));values[:,0]=profile(y);values[top,0]=wall_speed+1.
    wall=NativeSpaldingWall3D(mesh,low,d,molecular_viscosity=nu,wall_velocity=[wall_speed,0,0])
    s=NativeHybridNavierStokes3D(mesh,mask,values,viscosity=mu,backflow_velocity=values,
        convection_order=2,wall_model=wall,momentum_relaxation_basis='absolute_row_sum')
    expected=np.zeros((s.N,3));expected[:,0]=profile(mesh.cell_centers[:,1])
    return mesh,s,expected,tau,low,top

