#!/usr/bin/env python3
"""Native SST coupling controls; no physical SUBOFF or turbulent accuracy claim."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.native_sections3d import cell_section_polygons


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[2,4,8]);args=p.parse_args();out=args.output
    evidence=NativeRunEvidence(out,__file__);rows=[];comparison={}
    write_json(out/'settings.json',dict(levels=args.levels,dt=1.,relaxation=.7,tolerance=1e-8,
        density=1.,nu=.1,shear=.3,velocity_boundary='U=(0.3*y,0,0) on all boundary faces',
        reservoir='k=.01,omega=2 on all boundaries; this is not a no-slip turbulence wall condition',
        low_re='reservoir data except y=0/1: k=0,omega=60*nu/(beta1*d1^2)',
        pressure='solve p*=p+2*rho*k/3, output physical p; gauge p*[0]=0',
        physical_benchmark_passed=False,wall_function_verified=False))
    for kind in ('reservoir','low_re','laminar_limit'):
        for n in (args.levels if kind!='laminar_limit' else [args.levels[-1]]):
            methods=('newton','simplec') if kind=='reservoir' and n==args.levels[0] else ('newton',)
            for method in methods:
                name=f'{kind}-n{n}-{method}';m,s=coupled_shear_solver(n,kind)
                r=s.solve(max_iterations=120,dt=1.,relaxation=.7,tolerance=1e-8,momentum_method=method)
                current=r['current_rans_residual'];last=r['rans_history'][-1]
                row=dict(case=name,kind=kind,n=n,cells=len(m.cells),faces=len(m.owner),iterations=len(r['rans_history']),
                    converged=r['numerically_converged'],current={k:v for k,v in current.items() if k!='transport'},
                    final_changes=last['changes'],minimum_k=last['minimum_k'],minimum_omega=last['minimum_omega'],
                    energy_balance_error=r['energy_balance_error'],global_momentum_balance_max=r['global_momentum_balance_max'],
                    physical_benchmark_passed=False)
                if kind=='laminar_limit':
                    row['velocity_max_absolute_error']=float(np.max(np.abs(r['velocity']-np.column_stack((.3*m.cell_centers[:,1],np.zeros((len(m.cells),2)))))))
                    upper=(m.neighbor<0)&(m.face_centers[:,1]>1-1e-10)
                    row['upper_wall_shear_relative_error']=abs(float(r['total_traction_integrated'][upper,0].sum())-.03)/.03
                path=out/(name+'-fields.npz')
                fields={key:r[key] for key in ('velocity','face_velocity','physical_pressure','modified_pressure',
                    'k','omega','k_faces','omega_faces','nu_t','flux','total_traction_integrated','momentum_flux_integrated',
                    'pressure_traction_integrated','isotropic_turbulent_traction_integrated',
                    'deviatoric_traction_integrated','numerical_traction_correction')}
                for scalar in ('k','omega'):
                    fields[scalar+'_steady_free_residual']=current['transport'][scalar]['free_residual']
                    fields[scalar+'_residual_scale']=current['transport'][scalar]['residual_scale']
                    fields[scalar+'_total_flux']=current['transport'][scalar]['total_flux']
                    fields[scalar+'_source']=current['transport']['closure'][scalar+'_source']
                np.savez_compressed(path,vertices=m.vertices,cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                    cell_vertices=np.concatenate(m.cells),face_vertices=np.asarray(m.face_vertices),
                    cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,face_centers=m.face_centers,
                    area_vectors=m.area_vectors,owner=m.owner,neighbor=m.neighbor,wall_distance=s.wall_distance,
                    k_boundary=s.boundary['k_boundary'],omega_boundary=s.boundary['omega_boundary'],**fields)
                row['fields_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                write_json(out/(name+'-history.json'),r['rans_history']);rows.append(row)
                print(row,flush=True)
                if kind=='reservoir' and n==args.levels[0]:comparison[method]=fields
                if kind=='low_re' and n==args.levels[-1]:plot(m,r,out)
    difference={k:float(np.max(np.abs(comparison['newton'][k]-comparison['simplec'][k]))) for k in ('velocity','modified_pressure','k','omega','nu_t')}
    write_json(out/'cases.json',rows)
    fig,ax=plt.subplots()
    for row in rows:
        import json
        h=json.loads((out/(row['case']+'-history.json')).read_text())
        ax.semilogy([v['iteration'] for v in h],[max(v[k] for k in ('momentum','divergence','k','omega')) for v in h],label=row['case'])
    ax.set(xlabel='Outer iteration',ylabel='Maximum current equation residual');ax.legend(fontsize=7);ax.grid(True)
    fig.savefig(out/'convergence.png',dpi=170);fig.savefig(out/'convergence.pdf');plt.close(fig)
    evidence.verify_sources()
    passed=all(row['converged'] and max(row['energy_balance_error'],row['global_momentum_balance_max'])<1e-7 for row in rows) and max(difference.values())<1e-7
    write_json(out/'summary.json',dict(source_integrity_verified=True,coupling_controls_passed=bool(passed),
        simplec_newton_max_absolute_differences=difference,physical_suboff_benchmark_passed=False,
        turbulent_accuracy_verified=False,wall_function_verified=False))
    raise SystemExit(0 if passed else 2)


def plot(mesh,r,out):
    polygons,ids=cell_section_polygons(mesh,value=.47);bounds=np.concatenate(polygons)
    fig,axes=plt.subplots(1,4,figsize=(15,3.6),constrained_layout=True)
    for ax in axes:ax.set(xlim=(bounds[:,0].min(),bounds[:,0].max()),ylim=(bounds[:,1].min(),bounds[:,1].max()),aspect='equal',xlabel='x',ylabel='y')
    axes[0].add_collection(PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.5));axes[0].set_title('Actual section z=0.47')
    for ax,key,title in zip(axes[1:],['k','omega','nu_t'],['k','omega','Turbulent viscosity']):
        pc=PolyCollection(polygons,array=r[key][ids],cmap='viridis',edgecolors='none');ax.add_collection(pc);ax.set_title(title);fig.colorbar(pc,ax=ax)
    fig.savefig(out/'low-re-scalar-fields.png',dpi=170);fig.savefig(out/'low-re-scalar-fields.pdf');plt.close(fig)


if __name__=='__main__':main()
