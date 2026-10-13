#!/usr/bin/env python3
"""Recompute channel accuracy and conservation from saved native fields."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_sections3d import cell_section_polygons


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);root=p.parse_args().directory
    repo=Path(__file__).resolve().parents[1];ref=repo/'docs/verification-native-channel-sst/reference'
    settings=json.loads((root/'settings.json').read_text());meta=settings['reference_provenance']
    for r in meta['files'].values():assert hashlib.sha256((ref/r['csv']).read_bytes()).hexdigest()==r['csv_sha256']
    dns=np.loadtxt(ref/'mean.csv',delimiter=',',skiprows=1);unc=np.loadtxt(ref/'mean_standard_deviation.csv',delimiter=',',skiprows=1)
    interp=PchipInterpolator(dns[:,0],dns[:,2],extrapolate=False)
    source_hashes=json.loads((root/'sources.json').read_text())
    required=['native_hybrid_rans3d','native_hybrid_sst3d','native_hybrid_scalar3d','native_sst3d','native_boundary_constraints3d']
    missing=[name for name in required if 'src/tensorfvm/'+name+'.py' not in source_hashes]
    for file,digest in source_hashes.items():assert hashlib.sha256((root/'source-snapshot'/file).read_bytes()).hexdigest()==digest
    assert json.loads((root/'summary.json').read_text())['source_integrity_verified']
    rows=json.loads((root/'cases.json').read_text());audit=[];profiles=[]
    for row in rows:
        path=root/(row['case']+'-fields.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==row['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            v=f['cell_volumes'];inside=f['neighbor']>=0;area=np.linalg.norm(f['area_vectors'],axis=1)
            def div(q):
                a=np.zeros((len(v),)+q.shape[1:]);np.add.at(a,f['owner'],q);np.add.at(a,f['neighbor'][inside],-q[inside]);return a
            mass=float(np.max(np.abs(div(f['flux'])/v)))
            momentum=float(np.max(np.abs(f['momentum_flux_integrated'][~inside].sum(0))))
            cell_momentum=div(f['momentum_flux_integrated'])
            cell_momentum_integrated=float(np.max(np.abs(cell_momentum)))
            cell_momentum_density=float(np.max(np.abs(cell_momentum)/v[:,None]))
            y=f['cell_centers'][:,1];expected=(y-.5*y*y)/.1 if row['kind']=='laminar' else interp(y)
            assert np.isfinite(expected).all()
            error=float(np.sqrt(np.sum(v*(f['velocity'][:,0]-expected)**2)/np.sum(v*expected**2)))
            qbulk=float(f['flux'][f['outlet']].sum()/area[f['outlet']].sum())
            tau=-float(f['total_traction_integrated'][f['wall'],0].sum()/area[f['wall']].sum())
            cf=2*tau/qbulk**2
            np.testing.assert_allclose([error,cf,qbulk],[row['velocity_relative_l2'],row['Cf'],row['bulk_velocity_from_face_flux']],rtol=1e-12)
            scalar={}
            if row['kind']=='sst':
                history=json.loads((root/(row['case']+'-history.json')).read_text());last=history[-1]
                if row['numerical_convergence']:
                    assert max(last[k] for k in ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary'))<1e-8
                    assert max(last['changes'].values())<1e-8 and max(last['inflow_changes'].values())<1e-8
                    assert max(last['scalar_absolute_ledger'].values())<1e-8
                np.testing.assert_allclose(f['physical_pressure'],f['modified_pressure']-2*f['k']/3,atol=1e-14)
                for name in ('k','omega'):
                    val=float(np.linalg.norm(f[name+'_free_residual'])/f[name+'_residual_scale'])
                    np.testing.assert_allclose(val,row['current'][name],rtol=1e-12)
                    balance=div(f[name+'_total_flux'])-v*f[name+'_source']
                    assert float(np.max(np.abs(balance)))<1e-8
                    assert min(f[name].min(),f[name+'_faces'].min())>=0
                    if name=='omega':assert min(f[name].min(),f[name+'_faces'].min())>0
                    scalar[name]=val
                unique=np.unique(y);u=np.array([np.average(f['velocity'][y==yy,0],weights=v[y==yy]) for yy in unique])
                profiles.append((row,unique,u))
            if row['numerical_convergence']:assert mass<1e-8 and momentum<1e-7 and cell_momentum_integrated<1e-8 and all(x<1e-8 for x in scalar.values())
            audit.append(dict(case=row['case'],divergence=mass,global_momentum=momentum,cell_momentum_integrated=cell_momentum_integrated,cell_momentum_per_volume=cell_momentum_density,current_scalar=scalar,velocity_relative_l2=error,Cf=cf))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].plot(dns[:,1],dns[:,2],'k-',label='Lee–Moser DNS')
    axes[0].fill_between(dns[:,1],dns[:,2]-2*unc[:,2],dns[:,2]+2*unc[:,2],color='k',alpha=.15,label='DNS ±2 sample SD')
    for row,y,u in profiles:axes[0].plot(y*meta['Re_tau_reported'],u,'o-',ms=3,label=f"SST n={row['n']}")
    axes[0].set(xscale='log',xlabel='y+',ylabel='U+');axes[0].legend(fontsize=8);axes[0].grid(True)
    axes[1].plot([r['n'] for r,y,u in profiles],[r['Cf'] for r,y,u in profiles],'o-',label='Native SST, face-flux bulk velocity')
    axes[1].axhline(meta['Cf'],color='k',label='DNS');axes[1].axhspan(meta['Cf']*.97,meta['Cf']*1.03,color='g',alpha=.1,label='±3% nominal band')
    axes[1].set(xlabel='Wall-normal cells',ylabel='Cf');axes[1].legend(fontsize=8);axes[1].grid(True)
    fig.savefig(root/'dns-profiles-friction.png',dpi=180);fig.savefig(root/'dns-profiles-friction.pdf');plt.close(fig)
    if profiles:
        fine=profiles[-1][0]
        with np.load(root/(fine['case']+'-fields.npz')) as f:
            mesh=SimpleNamespace(vertices=f['vertices'],cells=list(range(len(f['cell_volumes']))),face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'])
            polygons,ids=cell_section_polygons(mesh,value=.047)
            polygons=[a[:,::-1] for a in polygons]
            fig,axes=plt.subplots(4,1,figsize=(11,7),constrained_layout=True)
            for ax,key,label in zip(axes,[None,'velocity','modified_pressure','physical_pressure'],['Actual section mesh z/h=0.047','Streamwise velocity U+','Modified pressure p*','Physical model pressure p=p*−2k/3']):
                if key is None:pc=PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.3)
                else:
                    values=f[key][ids];values=values[:,0] if values.ndim==2 else values
                    pc=PolyCollection(polygons,array=values,cmap='viridis',edgecolors='none');fig.colorbar(pc,ax=ax)
                ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='y/h',ylabel='x/h',title=label)
            fig.savefig(root/'mesh-velocity-pressure.png',dpi=180);fig.savefig(root/'mesh-velocity-pressure.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for row,y,u in profiles:
        history=json.loads((root/(row['case']+'-history.json')).read_text())
        x=[h['iteration'] for h in history]
        axes[0].semilogy(x,[max(h[k] for k in ('momentum','divergence','k','omega')) for h in history],label=f"n={row['n']} current equations")
        axes[1].semilogy(x,[max(h['changes'].values()) for h in history],label=f"n={row['n']} state change")
    for ax in axes:
        ax.axhline(1e-8,color='k',ls='--',label='gate');ax.set(xlabel='Outer iteration');ax.grid(True);ax.legend(fontsize=8)
    axes[0].set_ylabel('Maximum current residual');axes[1].set_ylabel('Maximum relative state change')
    fig.savefig(root/'convergence.png',dpi=180);fig.savefig(root/'convergence.pdf');plt.close(fig)
    write=dict(source_and_field_integrity_verified=True,source_snapshot_complete=not missing,missing_source_modules=missing,cases=audit,physical_suboff_benchmark_passed=False,
        grid_convergence_certified=False,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(write,indent=2)+'\n');print(write)


if __name__=='__main__':main()
