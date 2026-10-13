#!/usr/bin/env python3
"""Audit actual SUBOFF SST startup artifacts without promoting failed states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_sections3d import cell_section_polygons


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);root=p.parse_args().directory
    summary=json.loads((root/'summary.json').read_text());settings=json.loads((root/'settings.json').read_text())
    assert summary['source_integrity_verified'] and digest(root/'fields.npz')==summary['fields_sha256']
    for name,sha in json.loads((root/'sources.json').read_text()).items():assert digest(root/'source-snapshot'/name)==sha
    for name,sha in settings['initial_bindings'].items():assert digest(root/('initial-'+name))==sha
    repo=Path(__file__).resolve().parents[1];sources={}
    for name in ('scripts/report_native_suboff_sst_startup.py','src/tensorfvm/native_sections3d.py'):
        dst=root/'postprocessing-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,dst);sources[name]=digest(dst)
    (root/'postprocessing-sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    with np.load(root/'fields.npz',allow_pickle=False) as f:
        owner=f['owner'];neighbor=f['neighbor'];inside=neighbor>=0;volume=f['cell_volumes']
        def div(q):
            d=np.zeros(len(volume),dtype=q.dtype);np.add.at(d,owner,q);np.add.at(d,neighbor[inside],-q[inside]);return d
        flux=np.sum(f['face_velocity']*f['area_vectors'],axis=1)
        audit=dict(sources_and_fields_verified=True,initialized=summary['initialized'],
            completed_steps=summary['completed_steps'],mass_divergence_max=float(np.max(abs(div(flux)/volume))),
            numerically_converged=summary['numerically_converged'],physical_suboff_benchmark_passed=False,
            coupled_force_and_work_not_independently_audited=True)
        if summary['initialized']:
            audit['scalar_minima']={key:float(f[key].min()) for key in ('k','omega','k_faces','omega_faces')}
            with np.load(root/'initial-turbulence.npz',allow_pickle=False) as initial:
                unchanged=all(np.array_equal(f[key],initial[key]) for key in ('k','omega','k_faces','omega_faces'))
            audit['accepted_turbulence_unchanged_from_initial_guess']=unchanged
            if not summary['startup_steps_completed'] and summary['completed_steps']==0:assert unchanged
            wall=f['patch_labels']=='wall'
            np.testing.assert_array_equal(f['k_faces'][wall],0.)
            np.testing.assert_array_equal(f['omega_faces'][wall],f['omega_boundary'][wall])
        if (root/'rejected-scalar-candidate.npz').exists():
            meta=json.loads((root/'rejected-scalar-candidate.json').read_text())
            assert digest(root/'rejected-scalar-candidate.npz')==meta['fields_sha256']
            with np.load(root/'rejected-scalar-candidate.npz',allow_pickle=False) as c:
                balance=c['storage']+div(c['total_flux'])+c['destruction']-c['production']
                np.testing.assert_allclose(balance,c['cell_balance'],rtol=0,atol=1e-10)
                interface=c['total_flux'][inside]+c['neighbor_total_flux'][inside]
                audit['rejected_scalar']=dict(name=meta['scalar'],fields_verified=True,
                    cell_balance_max=float(np.max(abs(balance))),interface_defect_max=float(np.max(abs(interface))),
                    global_balance=float(c['storage'].sum()+c['total_flux'][~inside].sum()+c['destruction'].sum()-c['production'].sum()),
                    sum_saved_work_terms=float(np.sum(c['work_balance_cell_terms'],dtype=np.longdouble)+np.sum(c['work_balance_interface_terms'],dtype=np.longdouble)),
                    cell_min=float(c['cell_values'].min()),face_min=float(c['face_values'].min()),
                    candidate_is_accepted_flow=False)
        mesh=SimpleNamespace(vertices=f['vertices'],face_vertices=f['face_vertices'],owner=owner,neighbor=neighbor,cells=list(range(len(volume))))
        polys,ids=cell_section_polygons(mesh,value=.001)
        keys=[('velocity','Streamwise velocity'),('modified_pressure','Modified pressure p*')]
        if summary['initialized']:keys += [('k','Accepted initial k; SST step failed' if not summary['startup_steps_completed'] else 'Current k'),('omega','Current omega')]
        fig,axes=plt.subplots(len(keys)+1,1,figsize=(12,3*(len(keys)+1)),constrained_layout=True)
        fig.suptitle('SST STARTUP DIAGNOSTIC — no physical benchmark qualification')
        for ax in axes:ax.set(xlim=(-.15,1.25),ylim=(-.2,.2),aspect='equal',xlabel='x/L',ylabel='y/L')
        axes[0].add_collection(PolyCollection(polys,facecolors='none',edgecolors='.3',linewidths=.3));axes[0].set_title('Actual body-fitted section z/L=0.001')
        for ax,(key,title) in zip(axes[1:],keys):
            values=f[key][ids];values=values[:,0] if values.ndim==2 else values
            collection=PolyCollection(polys,array=values.astype(float),cmap='viridis',edgecolors='none')
            ax.add_collection(collection);ax.set_title(title);fig.colorbar(collection,ax=ax)
        fig.savefig(root/'startup-fields.png',dpi=160);fig.savefig(root/'startup-fields.pdf');plt.close(fig)
    (root/'artifact-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');print(audit)


if __name__=='__main__':main()
