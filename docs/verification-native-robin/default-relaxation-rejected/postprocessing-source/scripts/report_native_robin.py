#!/usr/bin/env python3
"""Independent saved-field force, Robin-sign and power audit for slip Couette."""
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


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);root=p.parse_args().run
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert summary['source_integrity_verified']
    for name,digest in json.loads((root/'sources.json').read_text()).items():assert sha(root/'source-snapshot'/name)==digest
    mu=cfg['mu'];lower,upper=cfg['friction'];u0,u1=cfg['wall_speeds'];h=cfg['height']
    tau=(u1-u0)/(h/mu+1/lower+1/upper);rows=[];profiles=[]
    for case in json.loads((root/'cases.json').read_text()):
        path=root/(case['case']+'-fields.npz');assert sha(path)==case['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            area=np.linalg.norm(f['area_vectors'],axis=1);b=f['neighbor']<0;inside=~b;vol=f['cell_volumes']
            def div(q):
                result=np.zeros((len(vol),)+q.shape[1:]);np.add.at(result,f['owner'],q)
                np.add.at(result,f['neighbor'][inside],-q[inside]);return result
            actual_flux=np.sum(f['face_velocity']*f['area_vectors'],axis=1)
            np.testing.assert_allclose(actual_flux,f['flux'],atol=1e-14,rtol=1e-12)
            mass=float(np.max(abs(div(actual_flux)/vol)))
            c=f['robin_coefficient'];robin=area[:,None]*np.einsum('fij,fj->fi',c,f['face_velocity'])
            np.testing.assert_allclose(robin,f['robin_reaction_integrated'],rtol=1e-12,atol=1e-13)
            defect=f['total_traction_integrated']+robin-area[:,None]*f['boundary_load']
            free=b[:,None]&~f['dirichlet_components'];boundary=float(np.max(abs(defect[free])))
            local_force=float(np.max(abs(div(f['total_traction_integrated']))))
            y=f['cell_centers'][:,1];exact=u0+tau*(y/mu+1/lower)
            expected=np.zeros_like(f['velocity']);expected[:,0]=exact
            error=float(np.sqrt(np.sum(vol[:,None]*(f['velocity']-expected)**2)/np.sum(vol[:,None]*expected**2)))
            lo=f['lower_wall'];hi=f['upper_wall'];slip=f['face_velocity'].copy();slip[lo,0]-=u0;slip[hi,0]-=u1
            slip_power=float(np.einsum('fi,fij,fj,f->',slip,c,slip,area))
            power=float(u0*f['total_traction_integrated'][lo,0].sum()+u1*f['total_traction_integrated'][hi,0].sum())
            gradient=np.stack([div(f['face_velocity']*f['area_vectors'][:,j,None])/vol[:,None] for j in range(3)],axis=2)
            stress=mu*(gradient+gradient.transpose(0,2,1)-2/3*np.trace(gradient,axis1=1,axis2=2)[:,None,None]*np.eye(3))
            bulk=float(np.einsum('fij,fij,f->',stress,gradient,vol))
            ledger=dict(divergence=mass,robin_boundary=boundary,local_force=local_force,mechanical_power=abs(bulk+slip_power-power),
                bulk_dissipation_replay=abs(bulk-case['bulk_dissipation']),slip_dissipation_replay=abs(slip_power-case['relative_slip_dissipation']))
            np.testing.assert_allclose(error,case['velocity_relative_l2'],rtol=1e-4,atol=1e-13)
            if case['control_passed']:assert max(ledger.values())<1e-8 and error<.03
            rows.append(dict(case=case['case'],control_passed=case['control_passed'],ledger=ledger,velocity_relative_l2=error,bulk_dissipation=bulk,slip_dissipation=slip_power,wall_power=power))
            if case['algorithm']=='newton':profiles.append((case,y.copy(),f['velocity'][:,0].copy()))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    grid=np.linspace(0,h,200);axes[0].plot(u0+tau*(grid/mu+1/lower),grid,'k-',label='Independent analytic slip Couette')
    for case,y,u in profiles:axes[0].plot(u,y,'.',label=f"Native NS n={case['n']}")
    axes[0].set(xlabel='u',ylabel='y/h',title='Finite wall slip, no body force');axes[0].legend();axes[0].grid(True)
    labels=[r['case']+(' (failed)' if not r['control_passed'] else '') for r in rows];axes[1].bar(labels,[r['bulk_dissipation'] for r in rows],label='Bulk viscous dissipation')
    axes[1].bar(labels,[r['slip_dissipation'] for r in rows],bottom=[r['bulk_dissipation'] for r in rows],label='Relative-slip dissipation')
    axes[1].plot(labels,[r['wall_power'] for r in rows],'kx',label='Moving-wall input power');axes[1].tick_params(axis='x',rotation=65)
    axes[1].set(ylabel='Power',title='Independent mechanical energy balance');axes[1].legend(fontsize=8)
    fig.savefig(root/'slip-velocity-work.png',dpi=180);fig.savefig(root/'slip-velocity-work.pdf');plt.close(fig)
    fine=profiles[-1][0]
    with np.load(root/(fine['case']+'-fields.npz'),allow_pickle=False) as f:
        mesh=SimpleNamespace(vertices=f['vertices'],cells=list(range(len(f['cell_volumes']))),face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'])
        polygons,ids=cell_section_polygons(mesh,value=.37)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,key,title in zip(axes,[None,'velocity','pressure'],['Actual mesh section','Streamwise velocity','Pressure, fixed scale ±1e-8']):
            if key is None:pc=PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.4)
            else:
                value=f[key][ids];value=value[:,0] if value.ndim==2 else value
                pc=PolyCollection(polygons,array=value,cmap='viridis',edgecolors='none')
                if key=='pressure':pc.set_clim(-1e-8,1e-8)
                fig.colorbar(pc,ax=ax)
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x',ylabel='y',title=title)
        fig.savefig(root/'mesh-velocity-pressure.png',dpi=180);fig.savefig(root/'mesh-velocity-pressure.pdf');plt.close(fig)
    repo=Path(__file__).resolve().parents[1];sources={}
    for name in ('scripts/report_native_robin.py','src/tensorfvm/native_sections3d.py'):
        dst=root/'postprocessing-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,dst);sources[name]=sha(dst)
    audit=dict(source_and_fields_verified=True,postprocessing_sources=sources,cases=rows,physical_suboff_benchmark_passed=False,
        scope='Affine linear-slip boundary control only; bulk power reconstructed directly from saved face velocity gradients')
    (root/'artifact-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit)


if __name__=='__main__':main()
