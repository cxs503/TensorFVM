"""Reconstruct inertia-free Rhie--Chow face flux at each saved steady state."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();s=json.loads((a.directory/'summary.json').read_text());results=[]
for case in s['cases']:
 for device in ['cpu','cuda']:
    row=case[device];c=row['config'];f=np.load(a.directory/case['name']/device/'fields.npz');u=f['velocity_m_s'];pressure=f['pressure_pa'];volume=f['cell_volumes_m2'];o=f['face_owner'];internal=f['interior_face_mask'];ni=f['face_neighbor'][internal];out=f['outlet_face_mask'];fixed=(~internal)&(~out);S=f['face_area_vectors_m'];disp=f['owner_neighbor_displacement_m'];T=f['nonorthogonal_area_vectors_m'];k=np.sum(S*S,1)/np.sum(S*disp,1);flux=f['face_mass_flux_kg_s_m'];rho=c['density'];mu=rho*c['inlet_velocity']*(c['length'] if row['case']=='sa' else c['height'])/c['reynolds']
    def face(v):
        result=v[o].copy();result[internal]=(v[o[internal]]+v[ni])/2;return result
    visc=face(mu+rho*f['eddy_viscosity_m2_s']);visc[f['wall_face_mask']]=mu;diff=visc*k;diff[out]=0.;diag=np.zeros(len(volume));np.add.at(diag,o,diff+np.maximum(flux,0));np.add.at(diag,ni,diff[internal]+np.maximum(-flux[internal],0));np.add.at(diag,o[out],np.minimum(flux[out],0));D=volume/diag;gp=f['pressure_gradient_pa_m'];delta=-pressure[o].copy();delta[internal]=pressure[ni]-pressure[o[internal]];normal=k*delta+np.sum(T*face(gp),1);pred=rho*(np.sum(face(u+D[:,None]*gp)*S,1)-face(D)*normal);pred[fixed]=rho*np.sum(f['boundary_velocity_m_s'][fixed]*S[fixed],1);value=np.linalg.norm(pred-flux)/max(np.linalg.norm(flux),1e-14)
    results.append(dict(case=case['name'],device=device,physical_steady_flux_relative_defect=float(value),passed=bool(value<1e-6)))
write_json(a.directory/'rhie-chow-fixedpoint-audit.json',dict(raw_evidence_verified=True,results=results,all_passed=all(r['passed'] for r in results),limit=1e-6,auditor_sha256=sha(Path(__file__)),summary_sha256=sha(a.directory/'summary.json'),scope='actual final steady inertia-free Rhie-Chow flux; gradients independently verified by main audit'))
m=json.loads((a.directory/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(a.directory);write_json(a.directory/'manifest.json',m);print(results)
