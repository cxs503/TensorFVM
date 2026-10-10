"""Independently reconstruct DFG pressure gradients, wall sampling and mass gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args();directory=args.directory
    s=json.loads((directory/'summary.json').read_text());checks=[]
    for row in s['runs']:
        base=directory/row['directory'];f=np.load(base/'fields.npz');c=row['config'];p=f['cell_pressure_pa'];o=f['face_owner'];ne=f['face_neighbor'];internal=f['internal_face_mask'];outlet=f['outlet_face_mask'];center=f['cell_centers_m'].reshape(-1,2)
        delta=np.zeros(len(o));delta[internal]=p[ne[internal]]-p[o[internal]];delta[outlet]=-p[o[outlet]]
        displacement=f['face_centers_m']-center[o];displacement[internal]=center[ne[internal]]-center[o[internal]]
        active=internal|outlet;weight=active.astype(float)/np.sum(displacement**2,axis=1);matrixface=weight[:,None,None]*displacement[:,:,None]*displacement[:,None,:]
        matrix=np.zeros((len(p),2,2));np.add.at(matrix,o,matrixface);np.add.at(matrix,ne[internal],matrixface[internal]);inv=np.linalg.inv(matrix);wd=weight[:,None]*displacement
        go=np.einsum('fij,fj->fi',inv[o],wd);gn=np.einsum('fij,fj->fi',inv[ne[internal]],wd[internal]);grad=np.zeros((len(p),2));np.add.at(grad,o,go*delta[:,None]);np.add.at(grad,ne[internal],gn*delta[internal,None])
        mask=f['cylinder_face_mask'];wall=p[o[mask]]+np.sum(grad[o[mask]]*displacement[mask],axis=1)
        if not np.allclose(wall,f['cylinder_wall_pressure_pa'],rtol=0,atol=1e-11):raise ValueError('independent wall-pressure reconstruction failed')
        angle=f['cylinder_angle_rad'];order=np.argsort(angle);aa=angle[order];wp=wall[order];a=np.r_[aa[-1]-2*np.pi,aa,aa[0]+2*np.pi];v=np.r_[wp[-1],wp,wp[0]]
        dp=float(np.interp(np.pi,a,v)-np.interp(0,a,v))
        if abs(dp-row['computed']['pressure_difference_pa'])>1e-11:raise ValueError('pressure difference mismatch')
        flux=f['face_mass_flux_kg_s_m'];net=np.zeros(len(p));np.add.at(net,o,flux);np.add.at(net,ne[internal],-flux[internal]);volume=f['cell_volumes_m2'];D=2*c['cylinder_radius']
        values=dict(scaled_face_divergence=float(np.max(abs(net/volume)))*D/(c['density']*c['inlet_velocity']),relative_boundary_mass_imbalance=float(abs(flux[f['boundary_face_mask']].sum())/abs(flux[f['inlet_face_mask']].sum())))
        for key,value in values.items():
            q=next(q for q in row['metrics'] if q['name']==key)
            if abs(value-q['value'])>1e-12 or q['limit']!=1e-6 or q['passed']!=(value<1e-6):raise ValueError('false mass gate')
        hist=json.loads((base/'history.json').read_text());last=hist[-1]
        values['solver_residual']=max(last[k] for k in ('continuity','momentum','mass_imbalance'));values['convergence_failure']=0. if row['converged'] else 1.
        for key,value in values.items():
            q=next(q for q in row['metrics'] if q['name']==key)
            if abs(value-q['value'])>1e-12 or q['passed']!=(value<q['limit']):raise ValueError('false numerical gate')
        if row['converged']!=(all(last[k]<c['tolerance'] for k in ('continuity','momentum','mass_imbalance'))):raise ValueError('false convergence badge')
        checks.append(dict(directory=row['directory'],pressure_difference_pa=dp,raw_reconstruction_passed=True,physical_passed=row['passed'],fields_sha256=sha(base/'fields.npz')))
    write_json(directory/'independent-pressure-audit.json',dict(cases=checks,source_sha256=sha(Path(__file__)),summary_sha256=sha(directory/'summary.json'),scope='independent NumPy least-squares pressure gradient, polygon-wall extrapolation/front-rear interpolation and actual mass/residual gates; does not certify accuracy when reference gates fail'))


if __name__=='__main__':main()
