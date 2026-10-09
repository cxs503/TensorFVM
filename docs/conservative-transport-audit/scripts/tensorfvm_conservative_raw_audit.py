import json,gzip,hashlib,math
from pathlib import Path
import numpy as np
from tensorfvm_conservative_review import stress,upwind_faces,independent_wall,div_vector_flux,global_boundary_flux,pair,face_fluid
from tensorfvm_projection_independent_review import np_div,np_grad
ROOT=Path('/home/jsyc/tensor-suite-development/TensorFVM');OUT=ROOT/'docs/conservative-transport'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):
    err=np.max(abs(np.asarray(a)-np.asarray(b)));assert err<tol,err;return float(err)
s=json.loads((OUT/'study.json').read_text());assert s['physical_accuracy_qualified'] is False and s['full_mac_qualified'] is False
for name,digest in s['source_sha256'].items():assert sha(ROOT/name)==digest
for name,digest in s['artifacts_sha256'].items():assert sha(OUT/name)==digest
cases=[]
for device in ('cpu','cuda'):
    r=json.loads(gzip.decompress((OUT/device/'transport-raw.json.gz').read_bytes()));c=r['config'];sp=r['spacing_m'];fluid=np.array(r['fluid']);t=r['transport'];u=np.array(t['old_velocity']);mu=np.array(t['dynamic_viscosity']);q=[np.array(a) for a in t['advecting_face_velocity']];conv=[np.array(a) for a in t['convective_flux']];visc=[np.array(a) for a in t['viscous_flux']];pf=[np.array(a) for a in t['pressure_flux']];inlet=np.array([c['inlet_velocity'],0,0]);molecular=c['density']*c['inlet_velocity']*2*c['cylinder_radius']/c['reynolds'];vol=np.prod(sp)
    p=np.asarray(r['pressure']);pfaces=[]
    for axis in range(3):
        left,right=pair(p[...,None],axis);a,b=face_fluid(fluid,axis);value=np.where(a[...,None],left,right);value=np.where((a&b)[...,None],.5*(left+right),value)
        if axis==0:value[...,-1,:]=0
        pv=np.zeros((*value.shape[:-1],3));pv[...,axis]=value[...,0];pfaces.append(pv)
    pressure_flux_error=max(close(a,b) for a,b in zip(pf,pfaces))
    gradient=np_grad(p,fluid,*sp);projection_error=max(close(np.array(after),np.array(before)-c['time_step']/c['density']*g,1e-10) for after,before,g in zip(r['face_velocity'],r['tentative_face_velocity'],gradient))
    conv_err=max(close(a,c['density']*b) for a,b in zip(conv,upwind_faces(u,q,fluid,inlet)));stress_err=max(close(a,b) for a,b in zip(visc,stress(u,mu,fluid,inlet,molecular,sp)))
    force,moment,_=independent_wall(pf,visc,fluid,sp);close(force,t['body_force']);close(moment,t['body_moment'],1e-8)
    wall=json.loads((OUT/device/'wall-facets.json').read_text());fp=np.zeros(3);fv=np.zeros(3);wm=np.zeros(3)
    for facet in wall['facets']:
        points=np.asarray(facet['points_m']).reshape(-1,3);p=np.asarray(facet['pressure_force_on_body_N']).reshape(-1,3);v=np.asarray(facet['viscous_force_on_body_N']).reshape(-1,3);fp+=p.sum(0);fv+=v.sum(0);wm+=np.cross(points,p+v).sum(0)
    close(fp+fv,force);close(wm,moment,1e-8);close(wall['viscous_evaluation_time_s'],r['time_s']-c['time_step']);close(wall['pressure_evaluation_time_s'],r['time_s'])
    residuals=[];boundaryerrors=[];previous=None
    for i,l in enumerate(r['ledgers']):
        before=np.array(l['momentum_before']);after=np.array(l['momentum_after']);forces=sum(np.asarray(l[k]) for k in ('convective_force_on_fluid','viscous_force_on_fluid','pressure_force_on_fluid'));ext=sum(np.asarray(l[k]) for k in ('exterior_convective_force_on_fluid','exterior_viscous_force_on_fluid','exterior_pressure_force_on_fluid'))-np.asarray(l['body_force']);res=after-before-c['time_step']*forces;bres=after-before-c['time_step']*ext;residuals.append(close(res,0));boundaryerrors.append(close(bres,0));close(res,l['momentum_residual']);close(bres,l['boundary_momentum_residual']);close(sum(np.asarray(l[k]) for k in ('wall_pressure_force_on_body','wall_viscous_force_on_body')),l['body_force'])
        if previous is not None:close(before,previous)
        previous=after
    final=np.asarray(r['velocity']);close(final,u+c['time_step']/c['density']*np.where(fluid[...,None],-div_vector_flux(conv,sp)+div_vector_flux(visc,sp)-div_vector_flux(pf,sp),0),1e-10);close(final.sum(axis=(0,1,2))*c['density']*vol,t['momentum_after'])
    div=np_div([np.array(f) for f in r['face_velocity']],fluid,*sp);continuity=np.max(abs(div))*2*c['cylinder_radius']/c['inlet_velocity'];close(continuity,r['history'][-1]['continuity']);assert continuity<1e-6
    cases.append(dict(case=device,steps=len(r['ledgers']),conv_error=conv_err,stress_error=stress_err,pressure_flux_error=pressure_flux_error,projection_error=projection_error,max_momentum_residual_Ns=max(residuals),max_external_boundary_residual_Ns=max(boundaryerrors),last_face_continuity=float(continuity),max_center_divergence=max(h['center_velocity_divergence'] for h in r['history']),wall_force_N=force.tolist(),wall_moment_Nm=moment.tolist()))
mms=[]
for r in json.loads((OUT/'manufactured.json').read_text()):
    x=np.asarray(r['x']);y=np.asarray(r['y']);a=2*np.pi/20;b=2*np.pi/12;cc=2*a;u=np.sin(a*x)*np.sin(b*y);v=np.sin(cc*x)*np.cos(b*y);mu=1+.2*np.sin(a*x);mux=.2*a*np.cos(a*x)
    xx=2*mux*a*np.cos(a*x)*np.sin(b*y)+mu*(-(2*a*a+b*b)*u-cc*b*np.cos(cc*x)*np.sin(b*y));yy=mux*(b*np.sin(a*x)*np.cos(b*y)+cc*np.cos(cc*x)*np.cos(b*y))+mu*(a*b*np.cos(a*x)*np.cos(b*y)-(cc*cc+2*b*b)*v);exact=np.stack((xx,yy,np.zeros_like(xx)),axis=-1);close(exact,r['analytic'],1e-12);computed=np.asarray(r['computed']);error=np.linalg.norm(computed-exact)/np.linalg.norm(exact);close(error,r['relative_l2_error'],1e-12);mms.append(dict(n=r['n'],relative_l2=float(error)))
report=dict(raw_audit_passed=True,cases=cases,MMS=mms,source_sha256=s['source_sha256'],study_sha256=sha(OUT/'study.json'),auditor_sha256=sha(Path(__file__)),physical_accuracy_qualified=False,full_mac_qualified=False);Path('/tmp/tensorfvm-conservative-raw-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
