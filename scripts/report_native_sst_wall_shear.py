#!/usr/bin/env python3
"""Plot saved coupled wall-SST diagnostic fields and actual acceptance history."""
import argparse,hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
from tensorfvm.native_sections3d import cell_section_polygons


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['source_integrity_verified'] or sha(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('changed/unverified fields')
    sources=json.loads((root/'sources.json').read_text())
    if 'src/tensorfvm/native_sst_wall3d.py' not in sources:raise ValueError('wall formula snapshot missing')
    for name,value in sources.items():
        if sha(root/'source-snapshot'/name)!=value:raise ValueError('producer source changed')
    repo=Path(__file__).resolve().parents[1];manifest={}
    for name in ('scripts/report_native_sst_wall_shear.py','src/tensorfvm/native_sections3d.py'):
        dest=root/'postprocessing-source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/name,dest);manifest[name]=sha(dest)
    (root/'postprocessing-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with np.load(root/'fields.npz') as f:
        mesh=SimpleNamespace(vertices=f['vertices'],face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'],cells=list(range(len(f['k']))))
        polygons,ids=cell_section_polygons(mesh,axis=2,value=.37)
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True);axes=axes.ravel()
        for ax in axes:ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x',ylabel='y')
        axes[0].add_collection(PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.5));axes[0].set_title('Actual mesh section z=0.37')
        for ax,key,title in zip(axes[1:],('velocity','physical_pressure','k','omega','nu_t'),('Streamwise velocity','Physical pressure p = p* - 2k/3','Turbulent kinetic energy k','Specific dissipation omega','Solved SST eddy viscosity')):
            value=f[key][ids];value=value[:,0] if value.ndim==2 else value
            collection=PolyCollection(polygons,array=np.asarray(value,float),edgecolors='none',cmap='viridis')
            ax.add_collection(collection);ax.set_title(title);fig.colorbar(collection,ax=ax,shrink=.7)
        fig.suptitle('Coupled native SST / Spalding diagnostic — no experimental qualification')
        fig.savefig(root/'coupled-fields.png',dpi=160);fig.savefig(root/'coupled-fields.pdf');plt.close(fig)
    history=json.loads((root/'history.json').read_text())
    if history:
        steps=[r['iteration'] for r in history]
        fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True,constrained_layout=True)
        for name in ('momentum','k','omega'):
            axes[0].semilogy(steps,[max(r[name],1e-30) for r in history],label=name)
        axes[0].axhline(cfg['tolerance'],ls='--',color='.4',label='equation / change tolerance')
        axes[0].set(ylabel='Current equation residual')
        for name in ('velocity','modified_pressure','k','omega','k_faces','omega_faces'):
            axes[1].semilogy(steps,[max(r['changes'][name],1e-30) for r in history],label=name)
        axes[1].axhline(cfg['tolerance'],ls='--',color='.4');axes[1].set(ylabel='Complete coupled field change')
        axes[2].semilogy(steps,[max(r['omega_constraint'],1e-30) for r in history],label='current wall omega constraint')
        axes[2].axhline(1e-12,ls='--',color='.4',label='strict cell constraint gate')
        checked=[r for r in history if r.get('steady_ledger_acceptance') is not None]
        for name in ('k','omega'):
            axes[2].semilogy([r['iteration'] for r in checked],
                [max(r['steady_ledger_acceptance']['scalar'][name]['relative_work_balance'],1e-30) for r in checked],
                marker='o',ms=3,label=name+' relative work (checked candidates)')
        axes[2].set(ylabel='Wall constraint / relative work',xlabel='Coupled iteration')
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in axes:ax.grid(True,alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Actual current constraints and steady ledgers; no storage in acceptance')
        fig.savefig(root/'coupled-history.png',dpi=160);fig.savefig(root/'coupled-history.pdf');plt.close(fig)
    print(dict(sources_and_fields_verified=True,completed_steps=summary['completed_steps'],solver_converged=summary['solver_converged'],physical_suboff_benchmark_passed=False))


if __name__=='__main__':main()
