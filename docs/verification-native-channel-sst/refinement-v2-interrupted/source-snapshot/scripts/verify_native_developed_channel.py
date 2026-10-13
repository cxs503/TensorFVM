#!/usr/bin/env python3
"""Native pressure-driven channel: laminar audit then SST versus original DNS."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from tensorfvm.verification.native_developed_channel import developed_channel_flow,developed_channel_rans,solve_developed_rans
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[16,32,64])
    parser.add_argument('--zero-initial-pressure',action='store_true',help='Discard the Stokes initial pressure before SST iteration')
    parser.add_argument('--momentum',choices=['coupled','simplec'],default='coupled')
    args=parser.parse_args()
    if any(n<16 or n%8 for n in args.levels):parser.error('levels must be multiples of 8 and >=16')
    repo=Path(__file__).resolve().parents[1];out=args.output;evidence=NativeRunEvidence(out,__file__)
    reference=repo/'docs/verification-native-channel-sst/reference'
    meta=json.loads((reference/'provenance.json').read_text())
    data=np.loadtxt(reference/'mean.csv',delimiter=',',skiprows=1)
    for entry in meta['files'].values():
        p=reference/entry['csv'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['csv_sha256']
    interpolate=PchipInterpolator(data[:,0],data[:,2],extrapolate=False)
    write_json(out/'settings.json',dict(levels=args.levels,Re_tau=meta['Re_tau_reported'],density=1.,half_width=1.,
        length=.1,span=.1,axial='n/8',spanwise=2,wall_normal_stretch=3.,body_force=[0,0,0],
        pressure_drop=.1,pressure_variable='p*=p+2*k/3',total_shear='tau_xy=1-y on inlet/outlet',
        wall='no-slip, k=0, omega=60*nu/(beta1*first_cell_distance^2)',wall_function=False,
        advection='zero developed-flow predictor; actual nonzero face flux in FULL current equation acceptance',
        inflow='current adjacent cell velocity/k/omega extrapolation, no DNS interior or inlet seeding',
        momentum_predictor=args.momentum,zero_initial_pressure=args.zero_initial_pressure,
        laminar_algorithm='native Stokes initialization plus NS Newton residual check',
        momentum_predictor_settings={'relaxation':1.,'pressure_relaxation':1.,'basis':'diagonal'} if args.momentum!='coupled' else {},scalar_correction='none',dt=1.,relaxation=.5,
        current_residual_and_change_gate=1e-8,scalar_absolute_ledger_gate=1e-8,accuracy_gate=.03,
        reference_provenance=meta,physical_suboff_benchmark_passed=False))
    rows=[]
    for kind in ('laminar','sst'):
        for n in args.levels:
            tick=time.perf_counter();config=dict(axial=n//8,length=.1,span=.1)
            if kind=='laminar':
                m,s,bc=developed_channel_flow(n,nu=.1,**config)
                r=s.solve_newton(tolerance=1e-9)
                y=m.cell_centers[:,1];exact=(y-.5*y*y)/.1
                current=r['newton_history'][-1]
                converged=r['numerically_converged']
            else:
                m,s,bc=developed_channel_rans(n,re_tau=meta['Re_tau_reported'],**config)
                if args.zero_initial_pressure:s.flow.p.fill(0.)
                def progress(row):
                    if row['iteration']%50==0:print(dict(case=f'{kind}-n{n}',iteration=row['iteration'],momentum=row['momentum'],k=row['k'],omega=row['omega'],ledgers=row['scalar_absolute_ledger']),flush=True)
                r=solve_developed_rans(s,max_iterations=400,callback=progress,momentum_method=args.momentum)
                y=m.cell_centers[:,1];exact=interpolate(y)
                if not np.isfinite(exact).all():raise ValueError('DNS interpolation would extrapolate; reject comparison')
                current={k:v for k,v in r['current_rans_residual'].items() if k!='transport'}
                converged=r['numerically_converged']
            area=np.linalg.norm(m.area_vectors,axis=1)
            outlet=(m.neighbor<0)&(m.face_centers[:,0]>.1-1e-12)
            wall=bc['wall'];bulk=float(r['flux'][outlet].sum()/area[outlet].sum())
            bulk_cell=float(m.cell_volumes@r['velocity'][:,0]/m.cell_volumes.sum())
            tau=-float(r['total_traction_integrated'][wall,0].sum()/area[wall].sum())
            cf=2*tau/bulk**2;cf_ref=meta['Cf'] if kind=='sst' else 2/(1/(3*.1))**2
            error=float(np.sqrt(np.sum(m.cell_volumes*(r['velocity'][:,0]-exact)**2)/np.sum(m.cell_volumes*exact**2)))
            cf_error=abs(cf-cf_ref)/cf_ref
            row=dict(case=f'{kind}-n{n}-{args.momentum if kind=="sst" else "newton"}',kind=kind,n=n,cells=len(m.cells),faces=len(m.owner),
                numerical_convergence=bool(converged),velocity_relative_l2=error,Cf=cf,Cf_reference=cf_ref,Cf_relative_error=cf_error,
                bulk_velocity_from_face_flux=bulk,bulk_velocity_from_cells=bulk_cell,wall_shear=tau,
                wall_shear_balance_error=abs(tau-1),first_cell_y_plus=float(y.min()*np.sqrt(abs(tau))/bc['nu']),
                current=current,iterations=len(r['rans_history']) if kind=='sst' else len(r['newton_history']),
                scalar_absolute_ledger=r.get('scalar_absolute_ledger'),energy_balance_error=r['energy_balance_error'],
                global_momentum_balance_max=r['global_momentum_balance_max'],runtime_seconds=time.perf_counter()-tick,
                accuracy_passed=bool(converged and max(error,cf_error)<.03),physical_suboff_benchmark_passed=False)
            fields={k:r[k] for k in ('velocity','face_velocity','flux','total_traction_integrated','momentum_flux_integrated')}
            if kind=='sst':
                fields.update({k:r[k] for k in ('physical_pressure','modified_pressure','k','omega','k_faces','omega_faces','nu_t')})
                for name in ('k','omega'):
                    t=r['current_rans_residual']['transport'][name]
                    fields[name+'_free_residual']=t['free_residual'];fields[name+'_residual_scale']=t['residual_scale']
                    fields[name+'_total_flux']=t['total_flux'];fields[name+'_source']=r['current_rans_residual']['transport']['closure'][name+'_source']
                history=r['rans_history']
            else:fields['physical_pressure']=r['pressure'];history=r['newton_history']
            file=out/(row['case']+'-fields.npz')
            np.savez_compressed(file,vertices=m.vertices,cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                face_vertices=np.asarray(m.face_vertices),cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,
                face_centers=m.face_centers,area_vectors=m.area_vectors,owner=m.owner,neighbor=m.neighbor,wall=wall,outlet=outlet,reference_velocity=exact,**fields)
            write_json(out/(row['case']+'-history.json'),history)
            row['fields_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row)
            write_json(out/'cases.json',rows);print(row,flush=True)
    evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,all_numerically_converged=all(r['numerical_convergence'] for r in rows),
        all_accuracy_gates_passed=all(r['accuracy_passed'] for r in rows),physical_suboff_benchmark_passed=False,
        all_cases_wall_resolved=True,wall_functions_verified=False))
    raise SystemExit(0 if all(r['numerical_convergence'] for r in rows) else 2)


if __name__=='__main__':main()
