#!/usr/bin/env python3
"""Reproduce native diffusion-energy, relaxation and interrupted-wall-force audits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
from tests.test_native_collocated3d import skew_mesh
from tests.test_native_rans3d import channel
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.native_simple3d import NativeSimpleSolver3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeSimpleSSTRansSolver3D,NativeRansSettings
from tensorfvm.wall_functions import wall_traction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--energy-witness',type=Path,help='Replay archived velocity; otherwise compute a smallest symmetric eigenmode')
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    write=lambda name,value:(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    m=NativeMesh3D.from_gmsh(args.mesh,triangulate_faces=True)
    v=NativeViscous3D(m,m.neighbor<0,1.);A=-v.momentum_diffusion_matrix
    coefficient=np.sum(m.area_vectors**2,axis=1)/np.sum(m.area_vectors*v.op.d,axis=1)
    diagonal=(-v.divergence_matrix@sparse.diags(coefficient)@v.delta_matrix).diagonal()
    R=sparse.diags(np.repeat(1/np.sqrt(diagonal),3));H=R@((A+A.T)*.5)@R
    if args.energy_witness is None:
        values,vectors=eigsh(H,k=1,which='SA',tol=1e-8,maxiter=500,v0=np.random.default_rng(19).normal(size=A.shape[0]))
        velocity=(R@vectors[:,0]).reshape(-1,3)
    else:velocity=np.load(args.energy_witness,allow_pickle=False)['velocity']
    if velocity.shape!=(len(m.cells),3):raise ValueError('witness shape differs from mesh')
    z=velocity.ravel()*np.repeat(np.sqrt(diagonal),3)
    energy=float(velocity.ravel()@(A@velocity.ravel()))
    rayleigh=float(z@(H@z)/(z@z))
    # Independent face traction -> acceleration -> discrete kinetic-energy rate.
    acceleration=v.op.integrate(v.momentum_flux(velocity))/m.cell_volumes[:,None]
    rate=float(np.sum(m.cell_volumes[:,None]*velocity*acceleration))
    np.testing.assert_allclose(rate,-energy,rtol=1e-12,atol=1e-12)
    write('viscous-energy.json',dict(cells=len(m.cells),mesh_sha256=sha(args.mesh),
        boundary='all zero Dirichlet; constant viscosity 1; no convection, pressure or external force',
        scaled_symmetric_rayleigh=rayleigh,diffusion_energy_form=energy,kinetic_energy_rate=rate,
        symmetric_eigen_residual=float(np.linalg.norm(H@z-rayleigh*z)/np.linalg.norm(z)),
        dissipative_for_this_witness=bool(energy>=0),
        qualification='Energy passivity diagnostic, not an eigenvalue or stability proof of the complete incompressible solver'))
    np.savez_compressed(out/'viscous-energy-witness.npz',velocity=velocity)
    print('ENERGY_AUDIT_COMPLETE',flush=True)

    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z));cases=[];fields=[]
    for relax in (.3,.8):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=relax,max_iterations=1500,nonlinear_tolerance=1e-9))
        result=s.solve();fields.append((s.u.copy(),s.p.copy()))
        cases.append(dict(relaxation=relax,converged=result['converged'],iterations=len(result['history']),last=result['history'][-1]))
    if not all(r['converged'] for r in cases):raise RuntimeError('Relaxation comparison requires both cases converged')
    write('relaxation-dependence.json',dict(cases=cases,
        relative_velocity_difference=float(np.linalg.norm(fields[0][0]-fields[1][0])/np.linalg.norm(fields[1][0])),
        relative_pressure_difference=float(np.linalg.norm(fields[0][1]-fields[1][1])/np.linalg.norm(fields[1][1]))))

    m=channel();wall=m.patch_labels=='wall';bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[wall]=0.
    s=NativeSimpleSSTRansSolver3D(m,(m.neighbor<0)&(m.patch_labels!='outlet'),bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.001,max_iterations=1,convection_order=1,relaxation=.3),
        rans_settings=NativeRansSettings(wall_treatment='spalding'))
    s.u[:,0]=1.
    def stop(row):raise StopIteration('imitate interruption at momentum checkpoint')
    try:s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.,momentum_callback=stop)
    except StopIteration:pass
    f=s.forces();area=np.linalg.norm(m.area_vectors[wall],axis=1);n=m.area_vectors[wall]/area[:,None]
    u=s.u[m.owner[wall]];tangent=u-n*np.sum(u*n,axis=1)[:,None]
    expected=wall_traction(tangent,s.wall_distance[m.owner[wall]],1.,.001)['traction_pa']*area[:,None]
    actual=f['anisotropic_stress_traction'][wall];actual=actual-n*np.sum(actual*n,axis=1)[:,None]
    write('interrupted-wall-force.json',dict(relative_shear_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
        reported_viscous_force_x=float(-actual[:,0].sum()),current_model_viscous_force_x=float(-expected[:,0].sum())))
    sources=[Path(__file__).resolve()]+[Path(m.__file__).resolve() for name,m in sys.modules.items()
        if name.startswith(('tensorfvm.','tests.test_native_')) and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py']
    bindings={}
    for source in sorted(set(sources)):
        relative=source.relative_to(repo);target=out/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());bindings[str(relative)]=sha(target)
    write('sources.json',bindings)
    print('AUDIT_COMPLETE: diffusion passivity remains unqualified',flush=True)


if __name__=='__main__':main()
