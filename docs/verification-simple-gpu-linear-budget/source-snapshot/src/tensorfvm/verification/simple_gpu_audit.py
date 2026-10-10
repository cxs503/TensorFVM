"""Independent NumPy reconstruction of curved SIMPLE geometry, forces and residuals."""
import json,math,statistics
from pathlib import Path
import numpy as np
from .core import sha,write_json,artifact_manifest
from .audit import close


def continuous(xy,c):
    x,y=xy.T;H=c['height'];U=c['inlet_velocity'];rho=c['density'];mu=rho*U*H/c['reynolds'];k=np.pi/c['length'];a=.2*H;s=np.sin(k*x);co=np.cos(k*x);g=a*s**4;g1=4*a*k*s**3*co;g2=4*a*k*k*(3*s*s*co*co-s**4);g3=4*a*k**3*(6*s*co**3-10*s**3*co);eta=(y-g)/H;w=6*U*eta*(1-eta);wy=6*U/H*(1-2*eta);wyy=-12*U/H**2;G=12*mu*U/H**2
    u=np.stack((w,g1*w),1);p=G*(c['length']-x);force=np.stack((-G-mu*((1+g1*g1)*wyy-g2*wy),rho*g2*w*w-mu*(g3*w-3*g1*g2*wy+g1*(1+g1*g1)*wyy)),1)
    return u,p,force


def reconstruction(f,c,kind):
    center=f['cell_centers_m'].reshape(-1,2);vol=f['cell_volumes_m2'];S=f['face_area_vectors_m'];o=f['face_owner'];n=f['face_neighbor'];internal=f['interior_face_mask'];oi=o[internal];ni=n[internal];u=f['velocity_m_s'];p=f['pressure_pa'];flux=f['face_mass_flux_kg_s_m'];out=f['outlet_face_mask'];fixed=(~internal)&(~out);disp=f['face_centers_m']-center[o];disp[internal]=center[ni]-center[oi];k=np.sum(S*S,1)/np.sum(S*disp,1);T=S-k[:,None]*disp;close(disp,f['owner_neighbor_displacement_m']);close(T,f['nonorthogonal_area_vectors_m'])
    closure=np.zeros_like(center);np.add.at(closure,o,S);np.add.at(closure,ni,-S[internal]);close(closure,0.,1e-11)
    def div(face):
        result=np.zeros((len(vol),)+face.shape[1:]);np.add.at(result,o,face);np.add.at(result,ni,-face[internal]);return result
    def interpolate(cell):
        result=cell[o].copy();result[internal]=(cell[oi]+cell[ni])/2;return result
    def gradient(q,pressure=False):
        scalar=q.ndim==1
        if scalar:q=q[:,None]
        delta=np.zeros((len(o),q.shape[1]));delta[internal]=q[ni]-q[oi];active=internal|(out if pressure else fixed)
        if pressure:delta[out]=-q[o[out]]
        else:delta[fixed]=f['boundary_velocity_m_s'][fixed]-q[o[fixed]]
        weight=active/np.sum(disp*disp,1);mf=weight[:,None,None]*disp[:,:,None]*disp[:,None,:];mat=np.zeros((len(vol),2,2));np.add.at(mat,o,mf);np.add.at(mat,ni,mf[internal]);inverse=np.linalg.inv(mat);wd=weight[:,None]*disp;go=np.einsum('fij,fj->fi',inverse[o],wd);gn=np.einsum('fij,fj->fi',inverse[ni],wd[internal]);result=np.zeros((len(vol),2,q.shape[1]));np.add.at(result,o,go[:,:,None]*delta[:,None,:]);np.add.at(result,ni,gn[:,:,None]*delta[internal,None,:]);return result[:,:,0] if scalar else result
    gu,gp=gradient(u),gradient(p,True);close(gu,f['velocity_gradient_s_inv'],1e-9);close(gp,f['pressure_gradient_pa_m'],1e-9)
    rho,U,H=c['density'],c['inlet_velocity'],c['height'];reference_length=c['length'] if kind=='sa' else H;mu=rho*U*reference_length/c['reynolds'];visc=interpolate(mu+rho*f['eddy_viscosity_m2_s']);visc[f['wall_face_mask']]=mu;diff=visc*k;diff[out]=0.;diag=np.zeros(len(vol));np.add.at(diag,o,diff+np.maximum(flux,0));np.add.at(diag,ni,diff[internal]+np.maximum(-flux[internal],0));np.add.at(diag,o[out],np.minimum(flux[out],0));ao=diff[internal]+np.maximum(-flux[internal],0);an=diff[internal]+np.maximum(flux[internal],0)
    source=div(np.where(fixed[:,None],(diff+np.maximum(-flux,0))[:,None]*f['boundary_velocity_m_s'],0.));correction=visc[:,None]*np.einsum('fi,fij->fj',T,interpolate(gu));correction[out]=0;source+=div(correction)-vol[:,None]*gp
    if kind=='curved':
        eu,ep,force=continuous(center,c);close(eu,f['reference_velocity_m_s']);close(ep,f['reference_pressure_pa']);close(force,f['body_force_n_m3']);source+=vol[:,None]*force
    net=diag[:,None]*u-source;np.add.at(net,oi,-ao[:,None]*u[ni]);np.add.at(net,ni,-an[:,None]*u[oi]);scale=max(rho*U*U*reference_length,mu*U);momentum=np.max(np.sum(abs(net),0))/scale;mass=div(flux);continuity=np.max(abs(mass))/(rho*U*H);imbalance=abs(flux[~internal].sum())/(rho*U*H)
    return dict(momentum=float(momentum),continuity=float(continuity),mass_imbalance=float(imbalance),maximum_nonorthogonal_angle_deg=float(np.max(np.arccos(np.clip(np.sum(S*disp,1)/np.linalg.norm(S,axis=1)/np.linalg.norm(disp,axis=1),-1,1)))*180/np.pi))


def audit(directory):
    directory=Path(directory);root=Path(__file__).resolve().parents[3];s=json.loads((directory/'summary.json').read_text());records=[]
    for name,digest in s['provenance']['source_sha256'].items():assert sha(root/name)==digest,name
    for case in s['cases']:
        fields={}
        for device in ['cpu','cuda']:
            row=case[device];path=directory/case['name']/device;f=np.load(path/'fields.npz',allow_pickle=False);fields[device]=f;values=reconstruction(f,row['config'],row['case']);hist=json.loads((path/'history.json').read_text());assert len(hist)==row['iterations'];assert row['actual_solver_device'].startswith(device)
            for key in ['momentum','continuity','mass_imbalance']:close(values[key],hist[-1][key],2e-9)
            for q in row['linear_history']:
                assert q['true_residual']<=q['target']*1.05 and q['actual_device'].startswith(device)
                if q['pressure']:assert q['assembled_operator_relative_defect']<2e-12
            if row.get('coarse_history'):
                for q in row['coarse_history']:assert q['operator_relative_defect']<2e-11 and q['linear_true_residual']<1e-9 and q['actual_device'].startswith(device)
                J,q,b=f['coupled_coarse_matrix'],f['coupled_coarse_last_solution'],f['coupled_coarse_last_rhs'];value=np.linalg.norm(J@q-b)/max(np.linalg.norm(b),1e-30);close(value,row['coarse_history'][-1]['linear_true_residual']);assert value<1e-9
            if row['case']=='curved':
                vol=f['cell_volumes_m2'];u=f['velocity_m_s'];ur=f['reference_velocity_m_s'];p=f['pressure_pa'];pr=f['reference_pressure_pa'];errors=dict(velocity_volume_l2=np.sqrt(np.sum(vol[:,None]*(u-ur)**2)/np.sum(vol[:,None]*ur**2)),velocity_linf=np.max(abs(u-ur))/np.max(abs(ur)),pressure_volume_l2=np.sqrt(np.sum(vol*(p-pr)**2)/np.sum(vol*pr**2)),pressure_linf=np.max(abs(p-pr))/np.max(abs(pr)))
                for key,value in errors.items():
                    q=next(q for q in row['metrics'] if q['name']==key);close(value,q['value']);assert q['limit']==.03 and q['passed']==(value<.03)
            assert row['passed']==all(q['passed'] for q in row['metrics'])
            timing=json.loads((directory/case['name']/(device+'-window.json')).read_text());assert len(timing['samples_s'])==3 and all(t>0 for t in timing['samples_s']);close(statistics.median(timing['samples_s']),timing['median_s'])
        for key,field in [('velocity','velocity_m_s'),('pressure','pressure_pa'),('nu_tilde','nu_tilde_m2_s')]:
            if key not in case['steady_field_differences']:continue
            a,b=fields['cpu'][field],fields['cuda'][field];value=np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-14);close(value,case['steady_field_differences'][key]);assert case['steady_equivalent']==all(v<=1e-6 for v in case['steady_field_differences'].values())
        records.append(dict(name=case['name'],cells=int(fields['cpu']['cell_volumes_m2'].size),steady_equivalent=case['steady_equivalent'],both_converged=case['cpu']['converged'] and case['cuda']['converged'],physical_passed=case['cpu']['passed'] and case['cuda']['passed'],maximum_nonorthogonal_angle_deg=values['maximum_nonorthogonal_angle_deg']))
    result=dict(raw_reconstruction_passed=True,cases=records,summary_sha256=sha(directory/'summary.json'),auditor_sha256=sha(Path(__file__)),scope='independent NumPy geometry/LS gradients/complete final momentum and mass residuals, continuous manufactured forcing, physical and CPU/CUDA field gates; SA history not independently replayed')
    write_json(directory/'audit.json',result);m=json.loads((directory/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(directory);write_json(directory/'manifest.json',m);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();print(audit(args.directory))
