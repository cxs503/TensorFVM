#!/usr/bin/env python3
"""Freeze unforced variable-viscosity velocity and wall-traction checks."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.verification.native_hybrid_shear import shear_solver,shear_fields,shear_metrics
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;evidence=NativeRunEvidence(out,__file__);rows=[]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for kind in ('smooth','jump'):
        for n in (8,16,32):
            m,s,exact=shear_solver(n,kind);r=s.solve(algorithm='coupled',tolerance=1e-10)
            row=dict(case=f'{kind}-n{n}',kind=kind,n=n,algorithm='coupled',**shear_metrics(m,r,exact))
            rows.append(row);print(row,flush=True)
            np.savez_compressed(out/f'{row["case"]}.npz',vertices=m.vertices,cells=np.asarray(m.cells),
                face_vertices=np.asarray(m.face_vertices),owner=m.owner,neighbor=m.neighbor,
                area_vectors=m.area_vectors,cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,
                viscosity=exact['viscosity'],exact_velocity=exact['velocity'],
                velocity_dirichlet_components=s.viscous.dirichlet_components,
                **{k:r[k] for k in ('velocity','pressure','face_velocity','flux','total_traction_integrated')})
            write_json(out/f'{row["case"]}-history.json',r['history'])
            if n==32:
                y=np.linspace(0,1,300);e=shear_fields(np.column_stack((y*0+.25,y,y*0+.5)),kind)
                ax=axes[0 if kind=='smooth' else 1]
                ax.plot(y,e['velocity'][:,0],label='analytic')
                ax.plot(m.cell_centers[::2,1],r['velocity'][::2,0],'o',markersize=3,label='native Stokes')
                ax.set(xlabel='y',ylabel='u_x',title=f'{kind} viscosity');ax.legend()
    selected=[r for r in rows if r['kind']=='smooth']
    for key,label in [('velocity_relative_l2','Velocity'),('lower_force_relative_error','Lower wall'),('upper_force_relative_error','Upper wall')]:
        axes[2].loglog([r['n'] for r in selected],[100*r[key] for r in selected],'o-',label=label)
    axes[2].legend();axes[2].set(xlabel='Wall-normal subdivisions',ylabel='Relative error (%)',title='Smooth viscosity refinement')
    fig.savefig(out/'profiles-wall-forces.png',dpi=180);fig.savefig(out/'profiles-wall-forces.pdf');plt.close(fig)
    write_json(out/'cases.json',rows)
    passed=all(r['velocity_and_wall_force_3_percent_passed'] for r in rows)
    evidence.verify_sources()
    write_json(out/'summary.json',dict(variable_viscosity_stress_verification_passed=passed,
        sst_integrated=False,wall_functions_integrated=False,physical_suboff_benchmark_passed=False,
        source_integrity_verified=True))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
