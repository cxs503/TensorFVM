#!/usr/bin/env python3
"""Audit native SUBOFF startup fields and plot actual mesh/flow; no Cp validation."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_sections3d import cell_section_polygons
from tensorfvm.native_boundary_pressure3d import NativeBoundaryPressureTrace3D


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);root=p.parse_args().directory
    summary=json.loads((root/'summary.json').read_text());settings=json.loads((root/'settings.json').read_text())
    repo=Path(__file__).resolve().parents[1];post_sources={}
    for name in ('scripts/report_native_hybrid_suboff.py','src/tensorfvm/native_boundary_pressure3d.py','src/tensorfvm/native_sections3d.py'):
        target=root/'postprocessing-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
        post_sources[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'postprocessing-sources.json').write_text(json.dumps(post_sources,indent=2)+'\n')
    assert summary['source_integrity_verified']
    if settings.get('initial_run'):
        for name,digest in settings['initial_run']['files'].items():
            assert hashlib.sha256((root/('initial-'+name)).read_bytes()).hexdigest()==digest
        previous=json.loads((root/'initial-settings.json').read_text())
        assert previous['mesh_sha256']==settings['mesh_sha256']
    for path,digest in json.loads((root/'sources.json').read_text()).items():
        assert hashlib.sha256((root/'source-snapshot'/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'input.msh').read_bytes()).hexdigest()==settings['mesh_sha256']
    assert hashlib.sha256((root/'fields.npz').read_bytes()).hexdigest()==summary['fields_sha256']
    rejected_audit=None
    if (root/'rejected-linear-candidate.json').exists():
        record=json.loads((root/'rejected-linear-candidate.json').read_text())
        assert not record['accepted'] and not record['changes_accepted_state']
        assert hashlib.sha256((root/'rejected-linear-candidate.npz').read_bytes()).hexdigest()==record['fields_sha256']
        with np.load(root/'rejected-linear-candidate.npz',allow_pickle=False) as rejected:
            nv=len(rejected['free_velocity_dofs']);defect=rejected['equation_defect']
            assert rejected['candidate'].shape==defect.shape==(nv+len(rejected['cell_volumes']),)
            constraint=defect[nv:]/rejected['cell_volumes']
            constraint_cells=np.arange(len(rejected['cell_volumes']))
            if bool(rejected['gauge']):constraint=constraint[1:];constraint_cells=constraint_cells[1:]
            if record['candidate_finite']:
                assert np.isfinite(rejected['candidate']).all() and np.isfinite(defect).all()
                np.testing.assert_allclose(float(np.max(np.abs(constraint),initial=0.)),record['algebraic_continuity_defect_per_volume_max'],rtol=1e-12)
                np.testing.assert_allclose(float(np.max(np.abs(defect[:nv]))),record['momentum_equation_defect_max'],rtol=1e-12)
            rejected_audit=dict(candidate_hash_verified=True,saved_equation_defect_metrics_replayed=True,
                coupled_matrix_reassembled=False,candidate_is_physical_flow=False,record={k:v for k,v in record.items() if k!='preconditioned_residual_history'})
    with np.load(root/'fields.npz',allow_pickle=False) as f:
        v=f['cell_volumes'];owner=f['owner'];neighbor=f['neighbor'];inside=neighbor>=0;wall=f['patch_labels']=='wall'
        area=np.linalg.norm(f['area_vectors'],axis=1);normal=f['area_vectors']/area[:,None]
        def div(q):
            value=np.zeros((len(v),)+q.shape[1:]);np.add.at(value,owner,q);np.add.at(value,neighbor[inside],-q[inside]);return value
        mass=float(np.max(np.abs(div(f['flux'])/v)))
        momentum=float(np.max(np.abs(div(f['momentum_flux_integrated']))))
        force=-f['total_traction_integrated'][wall].sum(0)
        remote=f['momentum_flux_integrated'][~inside & ~wall].sum(0)
        balance=float(np.max(np.abs(force+remote)))
        np.testing.assert_allclose(f['flux'],np.sum(f['face_velocity']*f['area_vectors'],axis=1),rtol=0,atol=1e-14)
        np.testing.assert_allclose(f['face_velocity'][wall],0.,atol=1e-14)
        np.testing.assert_allclose(force,summary['wall_force'],rtol=1e-12)
        np.testing.assert_allclose([mass,momentum,balance],[summary['mass_divergence_max'],summary['local_integrated_momentum_max'],summary['force_balance_max']],rtol=1e-12)
        np.testing.assert_allclose(f['wall_cp_owner'],2*f['pressure'][owner[wall]],atol=1e-14)
        np.testing.assert_allclose(f['wall_cp_normal_traction_proxy'],-2*np.sum(f['total_traction_integrated'][wall]*normal[wall],axis=1)/area[wall],atol=1e-14)
        history=json.loads((root/'history.json').read_text())
        if summary['numerically_converged']:
            assert max(mass,momentum,balance)<1e-8
            assert max(history[-1][k] for k in ('current_momentum_residual','divergence_max','velocity_change','pressure_change'))<settings['numerical_tolerance']
        if rejected_audit is not None and rejected_audit['record']['candidate_finite']:
            locations=f['cell_centers'][constraint_cells];magnitudes=np.abs(constraint)
            top=np.argsort(magnitudes)[-20:][::-1]
            rejected_audit['largest_local_constraints']=[dict(cell=int(constraint_cells[i]),
                center=locations[i].tolist(),volume=float(v[constraint_cells[i]]),
                vertex_count=int(f['cell_offsets'][constraint_cells[i]+1]-f['cell_offsets'][constraint_cells[i]]),
                absolute_defect_per_volume=float(magnitudes[i])) for i in top]
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            colors=np.log10(np.maximum(magnitudes,1e-20))
            for ax in axes:
                points=ax.scatter(locations[:,0],np.linalg.norm(locations[:,1:],axis=1),c=colors,s=6,cmap='magma')
                ax.set(xlabel='x/L',ylabel='Cell-center radius / L');ax.grid(True)
            axes[0].set_title('Rejected algebraic candidate: local continuity defect')
            axes[1].set(xlim=(-.05,1.05),ylim=(0,.1),title='Near-body cells; NOT a converged flow field')
            fig.colorbar(points,ax=axes,label='log10(abs(equation defect / cell volume))')
            fig.savefig(root/'rejected-linear-local-defects.png',dpi=180)
            fig.savefig(root/'rejected-linear-local-defects.pdf');plt.close(fig)
        mesh=SimpleNamespace(vertices=f['vertices'],cells=list(range(len(v))),cell_centers=f['cell_centers'],face_centers=f['face_centers'],face_vertices=f['face_vertices'],owner=owner,neighbor=neighbor)
        trace=NativeBoundaryPressureTrace3D(mesh,wall).reconstruct(f['pressure'])
        np.testing.assert_array_equal(trace['face_indices'],np.flatnonzero(wall))
        np.savez_compressed(root/'wall-pressure-traces.npz',**trace,input_fields_sha256=summary['fields_sha256'])
        trace_audit=dict(maximum_pressure_noise_amplification=float(trace['pressure_noise_amplification'].max()),maximum_rings=int(trace['stencil_rings'].max()),minimum_singular_ratio=float(trace['singular_ratio'].min()),
            owner_to_linear_trace_Cp_absolute_rms=float(np.sqrt(np.average((f['wall_cp_owner']-2*trace['pressure'])**2,weights=area[wall]))),
            wall_Cp_certified=False,fields_sha256=hashlib.sha256((root/'wall-pressure-traces.npz').read_bytes()).hexdigest())
        polygons,ids=cell_section_polygons(mesh,value=.001)
        fig,axes=plt.subplots(4,1,figsize=(12,12),constrained_layout=True)
        if not summary['numerically_converged']:fig.suptitle('REJECTED RUN — saved solver state is not converged flow')
        for ax,key,title in zip(axes,[None,None,'velocity','pressure'],['Actual domain mesh, z/L=0.001','Body-fitted mesh detail','Native streamwise velocity U/U_inf','Native cell pressure p/(rho U_inf^2)']):
            if key is None:pc=PolyCollection(polygons,facecolors='none',edgecolors='.35',linewidths=.3)
            else:
                values=f[key][ids];values=values[:,0] if values.ndim==2 else values
                pc=PolyCollection(polygons,array=values,cmap='viridis',edgecolors='none');fig.colorbar(pc,ax=ax)
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x/L',ylabel='y/L',title=title)
            if ax is not axes[0]:ax.set_xlim(-.15,1.25);ax.set_ylim(-.2,.2)
        fig.savefig(root/'mesh-and-flow.png',dpi=180);fig.savefig(root/'mesh-and-flow.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        x=f['face_centers'][wall,0]
        axes[0].scatter(x,f['wall_cp_owner'],s=9,label='Adjacent-cell pressure proxy')
        axes[0].scatter(x,f['wall_cp_normal_traction_proxy'],s=9,marker='x',label='Negative normal traction proxy')
        axes[0].scatter(x,2*trace['pressure'],s=9,marker='+',label='Connected-cell linear pressure trace')
        axes[0].set(xlabel='x/L',ylabel='Pressure coefficient proxy',title='Wall Cp NOT certified; all actual wall facets');axes[0].legend(fontsize=8);axes[0].grid(True)
        if history:
            axes[1].semilogy([h['iteration'] for h in history],[h['current_momentum_residual'] for h in history],'o-',label='Current momentum')
            axes[1].semilogy([h['iteration'] for h in history],[h['divergence_max'] for h in history],'s-',label='Current divergence')
            axes[1].axhline(1e-8,color='k',ls='--',label='Gate');axes[1].legend(fontsize=8)
        axes[1].set(xlabel='NS iteration',ylabel='Residual');axes[1].grid(True)
        fig.savefig(root/'wall-proxies-convergence.png',dpi=180);fig.savefig(root/'wall-proxies-convergence.pdf');plt.close(fig)
    audit=dict(source_and_fields_verified=True,mass_divergence_max=mass,local_integrated_momentum_max=momentum,
        pressure_trace=trace_audit,rejected_linear_candidate=rejected_audit,wall_force=force.tolist(),force_balance_max=balance,numerically_converged=summary['numerically_converged'],
        wall_Cp_certified=False,grid_convergence_certified=False,physical_suboff_benchmark_passed=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit)


if __name__=='__main__':main()
