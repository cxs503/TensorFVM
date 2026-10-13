"""Unforced finite-domain shear flow with solved SST and mixed wall treatment.

Lower wall: native Spalding + blended omega/G. Upper moving wall: resolved
no-slip, k=0 and low-Re omega. Scalar inlet data are explicit assumptions;
outlet and spanwise faces carry zero diffusive flux. No physical validation
or analytical full-RANS solution is supplied by this diagnostic.
"""
import numpy as np
from .native_spalding import spalding_couette_solver
from ..native_sst3d import wall_omega
from ..native_hybrid_rans3d import NativeHybridRans3D


def sst_wall_shear_solver(n=4,nu=.001):
    mesh,flow,*_=spalding_couette_solver(n,mu=.02,nu=nu)
    boundary=mesh.neighbor<0;x,y,z=mesh.face_centers.T
    top=boundary&(y>1-1e-12);inlet=boundary&(x<1e-12)
    mask=top|inlet
    distance=np.minimum(mesh.cell_centers[:,1],1-mesh.cell_centers[:,1])
    kb=np.full(flow.F,.01);wb=np.full(flow.F,2.)
    kb[top]=0.;wb[top]=wall_omega(nu,distance[mesh.owner[top]])
    s=NativeHybridRans3D(flow,distance,nu,k_dirichlet_mask=mask,omega_dirichlet_mask=mask,
        k_boundary=kb,omega_boundary=wb,k_inflow=.01,omega_inflow=2.,
        scalar_correction='nonlinear_mmp',scalar_work_balance_policy='relative',
        scalar_warm_start=True,scalar_precision='longdouble',wall_treatment='spalding_blended')
    flow.initialize_stokes();s.initialize(.01,2.,face_initialization='cell_average')
    return s
