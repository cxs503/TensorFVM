#!/usr/bin/env python3
"""Decompose a frozen quadratic MMS operator's exact-state truncation residual.

Analytic interior samples are used only in this read-only diagnostic. No
Navier--Stokes solve, fitted forcing or modification of production data occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss


def face_integrals(mesh, field, nu, rho, order):
    t,w=leggauss(order);t=(t+1)/2;w=w/2
    bary=np.array([[1-r,r*(1-s),r*s] for r in t for s in t])
    weights=np.array([2*r*wr*ws for r,wr in zip(t,w) for s,ws in zip(t,w)])
    F=len(mesh.owner)
    q=np.zeros(F);pressure=np.zeros(F);convection=np.zeros((F,3));stress=np.zeros((F,3))
    for start in range(0,F,256):
        ids=np.arange(start,min(start+256,F));tri=mesh.vertices[np.asarray([mesh.face_vertices[f] for f in ids])]
        nodes=np.einsum('qj,fjk->fqk',bary,tri)
        data=field(nodes.reshape(-1,3));u=data['velocity'].reshape(len(ids),len(weights),3)
        g=data['velocity_gradient'].reshape(len(ids),len(weights),3,3)
        tau=rho*nu*(g+g.swapaxes(-1,-2)-(2/3)*np.trace(g,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(3))
        normal_flux=np.einsum('fqk,fk->fq',u,mesh.area_vectors[ids])
        q[ids]=normal_flux@weights
        pressure[ids]=data['pressure'].reshape(len(ids),len(weights))@weights
        convection[ids]=rho*np.einsum('q,fqk,fq->fk',weights,u,normal_flux)
        stress[ids]=np.einsum('q,fqij,fj->fi',weights,tau,mesh.area_vectors[ids])
    return q,pressure,convection,stress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve();out=a.output
    if out.exists():parser.error('Output must be new')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config=json.loads((run/'config.json').read_text())
    if config['mode']!='manufactured' or config['reconstruction']!='quadratic' or config.get('merge_coplanar_faces',False):
        raise ValueError('Requires triangular-face quadratic manufactured run')
    if sha(run/'input.msh')!=config['mesh_sha256']:raise ValueError('Mesh binding mismatch')
    sys.path.insert(0,str(run/'source-snapshot/src'))
    from tensorfvm.native_mesh3d import NativeMesh3D
    from tensorfvm.native_collocated3d import NativeSteadySettings
    from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
    from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means
    bindings={}
    for name,module in list(sys.modules.items()):
        if name=='tensorfvm' or name.startswith('tensorfvm.'):
            path=Path(module.__file__).resolve();relative=str(path.relative_to(run/'source-snapshot'))
            if sha(path)!=config['source_sha256'].get(relative):raise ValueError('Frozen source binding mismatch: '+relative)
            bindings[relative]=sha(path)
    m=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True);c=NativeSteadySettings(**config['settings'])
    field=lambda x:abc_fields(x,nu=c.viscosity,rho=c.density)
    e=field(m.cell_centers);bc=field(m.face_centers)
    solver=NativeQuadraticSteadySolver3D(m,m.neighbor<0,bc['velocity'],settings=c,
        body_force=e['acceleration'],boundary_flux=exact_boundary_flux(m,field),gauge_value=e['pressure'][0],
        boundary_velocity_means=exact_boundary_velocity_means(m,field),boundary_velocity_gradients=bc['velocity_gradient'],quadratic_linear_solver='assembled')
    hi=face_integrals(m,field,c.viscosity,c.density,8);lo=face_integrals(m,field,c.viscosity,c.density,6)
    q,pf,cf,sf=hi;V=m.cell_volumes;integrate=solver.viscous.op.integrate
    solver.u=e['velocity'].copy();solver.p=e['pressure'].copy();solver._linear_flux=q
    block,rhs,_,correction,source=solver._system(q)
    ac,advection_source=solver._convection(q)
    conv=(ac@solver.u.ravel()+advection_source+solver._quadratic_convection_difference(q)).reshape(-1,3)
    visc=-integrate(solver.quadratic_traction());pressure=integrate(solver.quadratic_pressure_faces()[:,None]*m.area_vectors)
    exact_conv=integrate(cf);exact_visc=-integrate(sf);exact_pressure=integrate(pf[:,None]*m.area_vectors)
    surface_force=exact_conv+exact_visc+exact_pressure
    terms=dict(pressure_reconstruction=pressure-exact_pressure,viscous_reconstruction=visc-exact_visc,
        convection_reconstruction=conv-exact_conv,centroid_force_quadrature=surface_force-solver.force_integrated.reshape(-1,3))
    total=sum(terms.values());algebraic=(block@np.r_[solver.u.ravel(),solver.p]-rhs)[:3*solver.N].reshape(-1,3)
    identity=float(abs(total-algebraic).max())
    if identity>1e-10:raise ValueError('Term sum does not match current original momentum residual')
    terms['total_exact_state']=total
    qv=np.einsum('fi,fi->f',solver.quadratic_velocity_faces(),m.area_vectors)
    qc=solver.velocity_flux_matrix@solver.u.ravel()+correction@solver.p+source
    divergences=dict(exact_surface=integrate(q)/V,velocity_reconstruction=integrate(qv)/V,
                     rhie_chow_correction=integrate(qc-qv)/V,current_coupled=integrate(qc)/V)
    with np.load(run/'fields.npz') as f:
        error=f['pressure']-e['pressure'];error-=np.dot(error,V)/V.sum()
    hotspot=int(np.argmax(abs(error)))
    wall=np.zeros(solver.N,bool);wall[np.unique(m.owner[m.patch_labels=='wall'])]=True
    tail=(m.cell_centers[:,0]>=.9)&(m.cell_centers[:,0]<=1.025)&(np.linalg.norm(m.cell_centers[:,1:],axis=1)<=.025)
    groups=dict(all=np.ones(solver.N,bool),wall=wall,tail=tail,wall_tail=wall&tail,wall_tail_wedge=wall&tail&(np.asarray(m.cell_types)=='wedge'))
    summary={}
    for name,mask in groups.items():
        summary[name]=dict(cells=int(mask.sum()),terms={k:dict(acceleration_RMS=float(np.sqrt(np.sum(z[mask]**2/V[mask,None])/V[mask].sum())),acceleration_max=float(np.linalg.norm(z[mask]/V[mask,None],axis=1).max())) for k,z in terms.items()},continuity={k:dict(maximum=float(abs(z[mask]).max()),RMS=float(np.sqrt(np.dot(z[mask]**2,V[mask])/V[mask].sum()))) for k,z in divergences.items()})
    quad_force_delta=surface_force-(integrate(lo[2])-integrate(lo[3])+integrate(lo[1][:,None]*m.area_vectors))
    out.mkdir(parents=True);shutil.copy2(__file__,out/'executed-audit.py')
    np.savez_compressed(out/'residual-terms.npz',**terms,**{'divergence_'+k:v for k,v in divergences.items()},cell_centers=m.cell_centers,cell_volumes=V,pressure_error=error)
    result=dict(scope='read-only exact-state truncation decomposition; not a coupled solution error budget or new solve',cells=solver.N,source_sha256=bindings,input_sha256={n:sha(run/n) for n in ['config.json','input.msh','fields.npz']},algebraic_term_identity_max=identity,face_quadrature_6_vs_8_force_acceleration_max=float(np.linalg.norm(quad_force_delta/V[:,None],axis=1).max()),regions=summary,hotspot=dict(cell=hotspot,center=m.cell_centers[hotspot].tolist(),type=m.cell_types[hotspot],pressure_error=float(error[hotspot]),terms_acceleration={k:(v[hotspot]/V[hotspot]).tolist() for k,v in terms.items()},divergence={k:float(v[hotspot]) for k,v in divergences.items()}))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(hotspot=result['hotspot'],wall_tail_wedge=summary['wall_tail_wedge'],quadrature_check=result['face_quadrature_6_vs_8_force_acceleration_max']),indent=2))


if __name__=='__main__':main()
