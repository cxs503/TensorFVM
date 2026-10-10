"""Independently reconstruct laminar FV equations from saved external-flow fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tensorfvm.solver import SolverConfig
from tensorfvm.body_fitted import BodyFittedSolver
from tensorfvm import gmsh_external  # Register optional external mesh readers.
from tensorfvm.verification.core import write_json,sha,artifact_manifest


def process(output):
    summary=json.loads((output/'summary.json').read_text());records=[]
    torch.set_num_threads(1)
    for row in summary['runs']:
        directory=output/row['directory'];f=np.load(directory/'fields.npz');config=dict(row['config']);config['device']='cpu'
        if config.get('mesh_file'):
            config['mesh_file']=str((Path(__file__).resolve().parents[1]/config['mesh_file']).resolve())
        c=SolverConfig(**config);s=BodyFittedSolver(c)
        # Only geometry and prescribed boundary data are borrowed; gradients and
        # equations below are NumPy reconstructions, never solver residual calls.
        o=f['face_owner'];n=f['face_neighbor'];internal=f['internal_face_mask'];oi=o[internal];ni=n[internal];S=f['face_area_vectors_m'];centers=f['cell_centers_m'].reshape(-1,2);count=len(centers);d=s.mesh.face_centers.numpy()-centers[o];d[internal]=centers[ni]-centers[oi];volume=f['cell_volumes_m2'];out=s.mesh.masks['outlet'].numpy();fixed=~internal&~out;bc=s.boundary_velocity.numpy();u=f['cell_velocity_m_s'];p=f['cell_pressure_pa'];flux=f['face_mass_flux'];k=(S*S).sum(1)/(S*d).sum(1);T=S-k[:,None]*d
        np.testing.assert_allclose(centers,s.mesh.centers.numpy().reshape(-1,2),rtol=0,atol=1e-14)
        np.testing.assert_allclose(S,s.S.numpy(),rtol=0,atol=1e-14)
        def divergence(q):
            result=np.zeros((count,)+q.shape[1:]);np.add.at(result,o,q);np.add.at(result,ni,-q[internal]);return result
        scheme=row.get('discretization',{})
        fraction=np.full(np.count_nonzero(internal),.5)
        if scheme.get('face_interpolation')=='projected-linear':
            fraction=np.einsum('ij,ij->i',S[internal],s.mesh.face_centers.numpy()[internal]-centers[oi])/np.einsum('ij,ij->i',S[internal],d[internal])
        def interpolate(q):
            result=q[o].copy();shape=(-1,)+(1,)*(q.ndim-1);weight=fraction.reshape(shape)
            result[internal]=(1-weight)*q[oi]+weight*q[ni];return result
        def gradient(q,pressure=False):
            if pressure and scheme.get('pressure_gradient')=='gauss-linear':
                face=interpolate(q);face[out]=0
                return divergence(face[:,None]*S)/volume[:,None]
            scalar=q.ndim==1;qq=q[:,None] if scalar else q;delta=np.zeros((len(o),qq.shape[1]));delta[internal]=qq[ni]-qq[oi];active=internal|out if pressure else internal|fixed
            if pressure:delta[out]=-qq[o[out]]
            else:delta[fixed]=bc[fixed]-qq[o[fixed]]
            weight=active/(d*d).sum(1);matrix=np.zeros((count,2,2));face=weight[:,None,None]*d[:,:,None]*d[:,None,:];np.add.at(matrix,o,face);np.add.at(matrix,ni,face[internal]);rhs=np.zeros((count,2,qq.shape[1]));face_rhs=weight[:,None,None]*d[:,:,None]*delta[:,None,:];np.add.at(rhs,o,face_rhs);np.add.at(rhs,ni,face_rhs[internal]);g=np.linalg.solve(matrix,rhs);return g[:,:,0] if scalar else g
        gp=gradient(p,True);gu=gradient(u);diff=c.viscosity*k;diff[out]=0;diagonal=np.zeros(count);np.add.at(diagonal,o,diff+np.maximum(flux,0));np.add.at(diagonal,ni,diff[internal]+np.maximum(-flux[internal],0));np.add.at(diagonal,o[out],np.minimum(flux[out],0));ao=diff[internal]+np.maximum(-flux[internal],0);an=diff[internal]+np.maximum(flux[internal],0)
        correction=c.viscosity*np.einsum('fi,fij->fj',T,interpolate(gu));correction[out]=0;source=divergence(np.where(fixed[:,None],(diff+np.maximum(-flux,0))[:,None]*bc,0))+divergence(correction)-volume[:,None]*gp
        if scheme.get('momentum_convection')=='linear-upwind':
            up=np.where(flux[internal]>0,oi,ni)
            delta=s.mesh.face_centers.numpy()[internal]-centers[up]
            correction=np.zeros_like(S)
            correction[internal]=np.einsum('fi,fij->fj',delta,gu[up])
            source-=divergence(flux[:,None]*correction)
        residual=diagonal[:,None]*u-source;np.add.at(residual,oi,-ao[:,None]*u[ni]);np.add.at(residual,ni,-an[:,None]*u[oi]);scale=max(c.density*c.inlet_velocity**2*c.residual_length,c.viscosity*c.inlet_velocity);momentum=float(np.max(np.abs(residual).sum(0))/scale)
        history=json.loads((directory/'history.json').read_text());last=history[-1]
        if abs(momentum-last['momentum'])>max(1e-10,1e-7*momentum):raise ValueError('Independent momentum reconstruction mismatch')
        D=volume/diagonal;df=interpolate(D);dp=-p[o];dp[internal]=p[ni]-p[oi];normal=k*dp+(T*interpolate(gp)).sum(1);physical_flux=c.density*((interpolate(u+D[:,None]*gp)*S).sum(1)-df*normal);physical_flux[fixed]=c.density*(bc[fixed]*S[fixed]).sum(1);defect=float(np.linalg.norm(physical_flux-flux)/np.linalg.norm(flux));mass=divergence(flux);continuity=float(np.max(abs(mass))/(c.density*c.inlet_velocity*c.height))
        mask=f['surface_mask'];own=o[mask];wall_d=d[mask];wallg=gu[own].copy();error=-u[own]-np.einsum('fi,fij->fj',wall_d,wallg);wallg+=wall_d[:,:,None]*error[:,None,:]/(wall_d*wall_d).sum(1)[:,None,None];wp=p[own]+(gp[own]*wall_d).sum(1)
        np.testing.assert_allclose(wallg,f['surface_gradient'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(wp,f['surface_wall_pressure_pa'],rtol=1e-10,atol=1e-10)
        audit=dict(resolution=row['resolution'],independent_momentum=momentum,independent_continuity=continuity,physical_rhie_chow_relative_l2_defect=defect,physical_fixedpoint_passed=bool(momentum<c.tolerance and continuity<c.tolerance and defect<1e-6),discretization=scheme,scope='Independent NumPy configured gradients/interpolation, upwind or linear-upwind full laminar momentum, mass and physical inertia-free Rhie-Chow flux; production geometry and prescribed boundary data reconstructed from saved config')
        write_json(directory/'equations-audit.json',audit);records.append(audit);f.close()
    write_json(output/'equations-audit.json',dict(runs=records,passed=all(r['physical_fixedpoint_passed'] for r in records)))
    with (output/'report.md').open('a') as f:f.write('\n## 独立方程复核\n\n`equations-audit.json` 从保存场独立用 NumPy 重建最小二乘梯度、完整层流动量、质量守恒和无伪时间的物理 Rhie–Chow 通量，另外重建壁面无滑移梯度与压力。只复用原几何生成及给定边界数据；审计没有调用原求解器的梯度/动量/通量计算。固定点通量相对缺陷要求 <10⁻⁶，独立方程状态见JSON，不能以内部残差取代此审查。\n')
    m=json.loads((output/'manifest.json').read_text());m['source_sha256']['scripts/audit_external_equations.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path,nargs='+');a=p.parse_args()
    for output in a.outputs:process(output)
