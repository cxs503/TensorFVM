#!/usr/bin/env python3
"""Independent analytic scalar verification with original mesh/field evidence."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means,volume_weighted_error
from tensorfvm.native_sections3d import cell_section_polygons
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[2,4,8])
    args=parser.parse_args();out=args.output
    if len(args.levels)<3 or sorted(set(args.levels))!=args.levels or min(args.levels)<1:
        parser.error('at least three increasing positive grid levels required')
    evidence=NativeRunEvidence(out,__file__);rows=[]
    U=np.array([.6,-.3,.2]);sink=.7
    write_json(out/'settings.json',dict(levels=args.levels,velocity=U.tolist(),sink=sink,
        scalar='1+x^2+y^2+z^2',diffusivity='1+x',source='2*U.dot(x,y,z) - (6+8*x) + sink*phi',
        advection='first-order cell upwind',diffusion='native hybrid scalar energy',
        boundary='independent polygon area means, order-4 Gauss integration',
        accuracy_gate=.03,ledger_absolute_gate=1e-8,physical_suboff_benchmark_passed=False))
    for n in args.levels:
        mesh=pressure_mesh(n);x=mesh.cell_centers;exact=1+np.sum(x*x,axis=1)
        boundary=exact_boundary_velocity_means(mesh,lambda p:dict(velocity=
            np.column_stack((1+np.sum(p*p,axis=1),np.zeros((len(p),2))))),order=4)[:,0]
        source=2*x @ U-(6+8*x[:,0])+sink*exact
        op=NativeHybridScalar3D(mesh,1+x[:,0])
        result=op.solve(mesh.area_vectors @ U,source=source,sink=sink,
                        boundary_values=boundary,inflow_values=boundary)
        error=volume_weighted_error(result['cell_values'],exact,mesh.cell_volumes)
        ledger={k:float(result[k]) for k in ('linear_relative_residual','cell_balance_max',
            'interface_flux_defect_max','global_balance','work_balance','diffusion_energy')}
        passed=bool(error<.03 and max(abs(ledger[k]) for k in ledger if k!='diffusion_energy')<1e-8)
        path=out/f'scalar-n{n}-fields.npz'
        np.savez_compressed(path,vertices=mesh.vertices,cell_offsets=np.r_[0,np.cumsum([len(c) for c in mesh.cells])],
            cell_vertices=np.concatenate(mesh.cells),face_vertices=np.asarray(mesh.face_vertices),
            cell_centers=x,cell_volumes=mesh.cell_volumes,face_centers=mesh.face_centers,
            area_vectors=mesh.area_vectors,owner=mesh.owner,neighbor=mesh.neighbor,
            exact=exact,source=source,sink=sink,boundary_values=boundary,volume_flux=mesh.area_vectors @ U,
            diffusivity=1+x[:,0],**result)
        row=dict(n=n,cells=len(mesh.cells),faces=len(mesh.owner),relative_l2=error,
                 accuracy_and_ledger_passed=passed,fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**ledger)
        rows.append(row);print(row,flush=True)
    write_json(out/'cases.json',rows)
    polygons,ids=cell_section_polygons(mesh,value=.47)
    fig,axes=plt.subplots(1,4,figsize=(15,3.7),constrained_layout=True)
    bounds=np.concatenate(polygons)
    for ax in axes:ax.set(xlim=(bounds[:,0].min(),bounds[:,0].max()),ylim=(bounds[:,1].min(),bounds[:,1].max()),aspect='equal',xlabel='x',ylabel='y')
    axes[0].add_collection(PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.4));axes[0].set_title('Actual section z=0.47')
    for ax,values,title in zip(axes[1:],[exact,result['cell_values'],result['cell_values']-exact],['Analytic scalar','Computed scalar','Signed error']):
        collection=PolyCollection(polygons,array=values[ids],edgecolors='none',cmap='viridis')
        if ax is not axes[-1]:collection.set_clim(min(exact[ids].min(),result['cell_values'][ids].min()),max(exact[ids].max(),result['cell_values'][ids].max()))
        ax.add_collection(collection);ax.set_title(title);fig.colorbar(collection,ax=ax)
    fig.savefig(out/'mesh-scalar.png',dpi=170);fig.savefig(out/'mesh-scalar.pdf');plt.close(fig)
    fig,ax=plt.subplots();ax.loglog([r['n'] for r in rows],[100*r['relative_l2'] for r in rows],'o-')
    ax.set(xlabel='Grid level n',ylabel='Volume-weighted relative L2 error (%)');ax.grid(True,which='both')
    fig.savefig(out/'refinement.png',dpi=170);fig.savefig(out/'refinement.pdf');plt.close(fig)
    evidence.verify_sources()
    passed=bool(rows[-1]['accuracy_and_ledger_passed'] and all(a['relative_l2']>b['relative_l2'] for a,b in zip(rows[:-1],rows[1:])))
    write_json(out/'summary.json',dict(source_integrity_verified=True,finest_and_refinement_passed=passed,
        all_grids_passed=all(r['accuracy_and_ledger_passed'] for r in rows),
        coupled_rans_verified=False,wall_functions_verified=False,physical_suboff_benchmark_passed=False))
    raise SystemExit(0 if passed else 2)


if __name__=='__main__':main()
