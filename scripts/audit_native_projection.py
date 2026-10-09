"""Independent NumPy reconstruction of native exported final face fields.

Checks all-step acceptance flags from stored histories; only final raw fields
are independently reconstructed. Does not rerun the production fluid solver.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from validate_compatible_projection import np_grad,np_div
ROOT=Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(folder):
    manifest=json.loads((folder/'study.json').read_text())
    if manifest['physical_accuracy_qualified'] is not False:raise ValueError('false physical qualification')
    for name,expected in manifest['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('native source hash mismatch: '+name)
    for name,expected in manifest['raw_sha256'].items():
        if digest(folder/name)!=expected:raise ValueError('native raw hash mismatch: '+name)
    rows=[]
    for case in manifest['cases']:
        name=case['case'];raw=json.loads((folder/name/'face-fields.json').read_text())
        history=json.loads((folder/name/'history.json').read_text());c=raw['config']
        if raw['schema']!='tensorfvm.face-projection-field/1' or raw['physical_accuracy_qualified'] is not False or raw['partition']['world_size']!=1:
            raise ValueError('unsupported or falsely qualified field schema')
        if len(history)!=case['steps'] or raw['final']!=history[-1]:raise ValueError('field/history clock mismatch')
        passed=all(all(math.isfinite(v) for v in h.values()) and h['cfl']<1 and h['pressure_converged']==1
                   and h['pressure_residual']<=h['pressure_target_residual'] and h['continuity']<1e-6
                   and h['boundary_mass_imbalance']<1e-6 for h in history)
        if bool(case['passed'])!=passed:raise ValueError('all-step acceptance disagreement')
        fluid=np.asarray(raw['fluid'],dtype=bool);p=np.asarray(raw['pressure']);spacing=raw['spacing']
        faces=tuple(np.asarray(f) for f in raw['face_velocity']);pred=tuple(np.asarray(f) for f in raw['tentative_face_velocity'])
        if any(not np.isfinite(a).all() for a in (*faces,*pred,p)):raise ValueError('nonfinite raw field')
        grad=np_grad(p,fluid,*spacing);div=np_div(faces,fluid,*spacing)
        correction=max(float(np.max(abs(a-b+c['time_step']/c['density']*g))) for a,b,g in zip(faces,pred,grad))
        continuity=float(np.max(abs(div)))*2*c['cylinder_radius']/c['inlet_velocity']
        x,y,z=faces;dx,dy,dz=spacing
        flux=(x[...,-1].sum()-x[...,0].sum())*dy*dz+(y[:,-1].sum()-y[:,0].sum())*dx*dz
        mass=abs(flux)/(c['inlet_velocity']*c['height']*c['span'])
        masks=(fluid[...,:-1]&fluid[...,1:],fluid[:,:-1]&fluid[:,1:],fluid&np.roll(fluid,-1,axis=0))
        if any(not np.array_equal(mask,np.asarray(saved,dtype=bool)) for mask,saved in zip(masks,raw['internal_face_masks'])):
            raise ValueError('invalid solid face mask')
        wall=max(float(np.max(abs(a[~mask]),initial=0)) for a,mask in zip((x[...,1:-1],y[:,1:-1],z),masks))
        if correction>1e-12 or abs(continuity-history[-1]['continuity'])>1e-12 or abs(mass-history[-1]['boundary_mass_imbalance'])>1e-12 or wall!=0 or abs(flux-div.sum()*dx*dy*dz)>1e-12:
            raise ValueError('independent face field reconstruction failed')
        rows.append({'case':name,'all_step_passed':passed,'steps':len(history),'independent_final_continuity':continuity,
                     'independent_final_mass_imbalance':mass,'solid_face_flux_max':wall,'face_correction_error':correction,
                     'failed_pressure_steps':sum(not h['pressure_converged'] for h in history)})
    byname={row['case']:row for row in rows}
    if byname['cpu-150-failed']['all_step_passed'] or not byname['cpu-250']['all_step_passed']:
        raise ValueError('expected budget control outcome differs')
    if 'cuda-250' in byname and not byname['cuda-250']['all_step_passed']:raise ValueError('CUDA native validation failed')
    return {'native_manifest_sha256':digest(folder/'study.json'),'cases':rows,'physical_accuracy_qualified':False,
            'source_sha256':{name:digest(ROOT/name) for name in ['scripts/audit_native_projection.py','scripts/validate_compatible_projection.py']},
            'scope':'independent final raw face reconstruction plus all-step stored-history acceptance, not independent fluid time integration'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,default=ROOT/'docs/compatible-projection/native')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/compatible-projection/independent/native-audit.json')
    args=parser.parse_args();result=audit(args.native);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
