"""Small SST coupling controls, not an experimentally qualified turbulent flow."""
import numpy as np
from .native_hybrid_flow import channel_mesh
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_sst3d import wall_omega


def coupled_shear_solver(n=4,kind='reservoir',*,density=1.,triangulate_faces=None):
    if kind not in ('reservoir','low_re','laminar_limit'):raise ValueError('unknown coupling control')
    if triangulate_faces is None:triangulate_faces=kind!='low_re'
    mesh=channel_mesh((2,n,1),triangulate_faces=triangulate_faces);b=mesh.neighbor<0;nu=.1;shear=.3
    boundary=np.zeros((len(mesh.owner),3));boundary[:,0]=shear*mesh.face_centers[:,1]
    flow=NativeHybridNavierStokes3D(mesh,b,boundary,density=density,
                                  viscosity=density*nu,convection_order=2)
    flow.initialize_stokes()
    k0=0. if kind=='laminar_limit' else .01
    w0=np.sqrt(.44/.0828)*shear if kind=='laminar_limit' else 2.
    kb=np.full(len(mesh.owner),k0);wb=np.full(len(mesh.owner),w0)
    distance=np.minimum(mesh.cell_centers[:,1],1-mesh.cell_centers[:,1])
    if kind=='low_re':
        wall=b & ((mesh.face_centers[:,1]<1e-10)|(mesh.face_centers[:,1]>1-1e-10))
        kb[wall]=0.;wb[wall]=wall_omega(nu,distance[mesh.owner[wall]])
        # Boundary-driven diffusion initial guess, not reference interior data.
        from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
        w0=NativeHybridScalar3D(mesh,nu).solve(boundary_values=wb)['cell_values']
    rans=NativeHybridRans3D(flow,distance,nu,k_boundary=kb,omega_boundary=wb,
                          k_inflow=kb,omega_inflow=wb)
    rans.initialize(k0,w0)
    return mesh,rans
