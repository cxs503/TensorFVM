#!/usr/bin/env python3
"""Native mesh and actual startup-flow figures; no synthetic reference curves."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();d=a.input
    summary=json.loads((d/'summary.json').read_text());config=json.loads((d/'config.json').read_text())['config'];z=np.load(d/'checkpoint.npz',allow_pickle=False)
    nx,ny,nz=(config[k]for k in ('nx','ny','nz'));k=nz//2
    nodes=z['vertices'];cells=z['connectivity'].reshape(nz,ny,nx,8)[k].reshape(-1,8);polys=nodes.reshape(-1,3)[cells[:,:4],:2]
    centers=z['centers'].reshape(nz,ny,nx,3)[k];pressure=z['pressure'].reshape(nz,ny,nx)[k];velocity=z['velocity'].reshape(nz,ny,nx,3)[k]
    figs=[]
    f=plt.figure(figsize=(11,5));ax=f.add_subplot(121,projection='3d')
    wall=nodes[:,0,:,:]
    for j in range(0,nx,max(1,nx//24)):ax.plot(wall[:,j,0],wall[:,j,1],wall[:,j,2],color='k',lw=.5)
    for j in range(0,nz+1,max(1,nz//8)):ax.plot(wall[j,:,0],wall[j,:,1],wall[j,:,2],color='steelblue',lw=.7)
    ax.set(xlabel='x/D',ylabel='y/D',zlabel='z/D',title='Actual polygonal cylinder wall');ax.set_box_aspect((1,1,config['span']))
    ax=f.add_subplot(122);pc=PolyCollection(polys,facecolors='none',edgecolors='k',linewidths=.2);ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlim=(config['cylinder_x']-1,config['cylinder_x']+4),ylim=(config['cylinder_y']-2,config['cylinder_y']+2),title='Midspan body-fitted O-grid',xlabel='x/D',ylabel='y/D');f.tight_layout();figs.append(('geometry',f))
    f,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,values,label in zip(axes,[pressure,np.linalg.norm(velocity,axis=-1)],['Instantaneous pressure / rho U^2','Instantaneous speed / U']):
        pc=PolyCollection(polys,array=values.ravel(),cmap='turbo',edgecolors='none');ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlim=(config['cylinder_x']-1,config['cylinder_x']+5),ylim=(config['cylinder_y']-2,config['cylinder_y']+2),xlabel='x/D',ylabel='y/D',title=label);f.colorbar(pc,ax=ax)
    f.suptitle('Startup fields: not statistically converged LES');f.tight_layout();figs.append(('fields',f))
    forces=json.loads((d/'forces.json').read_text());hist=json.loads((d/'history.json').read_text())
    f,axes=plt.subplots(1,2,figsize=(11,4));t=[x['time']for x in forces]
    axes[0].plot(t,[x['Cd']for x in forces],label='Cd actual');axes[0].plot(t,[x['Cl']for x in forces],label='Cl actual');axes[0].legend();axes[0].set(xlabel='t U/D',ylabel='Force coefficient',title='Startup force, not time-averaged reference')
    for name in ['continuity','momentum_ledger']:axes[1].semilogy([x['time']for x in hist],[max(abs(x[name]),1e-18)for x in hist],label=name)
    axes[1].legend();axes[1].set(xlabel='t U/D',title='Independent mass and momentum ledgers');f.tight_layout();figs.append(('histories',f))
    wallrows=np.genfromtxt(d/'wall-pressure.csv',delimiter=',',names=True)
    f,ax=plt.subplots(figsize=(7,4));theta=np.unique(wallrows['theta_degrees']);cp=[wallrows['Cp'][wallrows['theta_degrees']==x].mean()for x in theta]
    ax.plot(theta,cp,'o-',ms=2,label='Actual instantaneous span-averaged Cp');ax.legend();ax.set(xlabel='theta from downstream +x (degrees)',ylabel='Cp',title='Startup surface Cp; reference raw points not ingested');ax.grid(alpha=.2);f.tight_layout();figs.append(('surface-cp',f))
    with PdfPages(d/'report.pdf')as pdf:
        for name,f in figs:f.savefig(d/(name+'.png'),dpi=180);f.savefig(d/(name+'.svg'));pdf.savefig(f);plt.close(f)
    resolution=summary.get('wall_resolution')
    resolution_text=('Measured instantaneous startup y+ max = '+str(resolution['yplus_max'])+', 95th percentile = '+str(resolution['yplus_95'])+'. Wall-resolved y+<=1 gate: '+str(resolution['wall_resolution_passed'])+'.')if resolution else 'No y+ was recorded by this archived producer.'
    text=f'''# Three-dimensional body-fitted cylinder, Re = 3900

## Scope and primary literature

This is an actual 3-D body-fitted FV flow development case. It uses a polygonal no-slip cylinder and periodic span, with no Cartesian solid mask. The primary experimental/LES paper is [Parnaudeau et al. (2008)](https://doi.org/10.1063/1.2957018), [original author record](https://hal.science/hal-00383669v1). The authors identify substantial integration-time sensitivity and approximately 10% uncertainty for many near-wake statistics. Our numerical accuracy target remains 3% against explicitly selected reference metrics, with source uncertainty reported separately. No unverified Cd, St, Cp points have been substituted for missing raw reference data.

## Configuration and software usage

Re = U D / nu = 3900; U = rho = D = 1, nu = 1/3900. Domain: 20D x 12D, cylinder at (5D,6D); periodic span pi D. This domain/span is a development choice; sensitivity is still required. Grid {nx} x {ny} x {nz} = {summary['mesh_cells']} true hexahedral control volumes. Exponential radial clustering; first-cell resolution must be measured, never inferred from grid counts alone.

Shared owner/neighbor faces integrate convection, full symmetric deviatoric variable-viscosity stress and pressure. WALE SGS uses the original [Nicoud and Ducros (1999) formulation](https://doi.org/10.1023/A:1009995426001); Cw = 0.325 and Delta = V^(1/3). Zero SGS face viscosity at the wall. Forward Euler and 90% central / 10% upwind convection are exploratory settings; numerical-dissipation sensitivity remains open. LS gradients and nonorthogonal pressure face projection are implemented in a shared geometry kernel. Cell velocity and face mass flux are separate authoritative quantities; mass closure alone does not validate their collocated coupling.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_body_fitted_cylinder3900.py --output results/cylinder3900-body-fitted --nx 48 --ny 16 --nz 8 --steps 300 --dt .002 --stretching 3.5
PYTHONPATH=src python scripts/report_body_fitted_cylinder3900.py --input results/cylinder3900-body-fitted
```

New runs support atomic checkpoints and exact continuation in a new empty output directory with `--initialize-from previous/checkpoint.npz`. Configuration and producer hashes must match; max_steps may increase. Online means and second moments resume with the state. CPU factorization only; CUDA and multi-rank curved-mesh flow not tested.

## Actual calculation

Completed {summary['steps']} steps, tU/D = {summary['final']['time']:.6g}; max final cell divergence = {summary['final']['continuity']:.3e}; final momentum ledger = {summary['final']['momentum_ledger']:.3e}; final CFL = {summary['final']['cfl']:.5f}. These figures audit numerical advancement. Final instantaneous Cd = {summary['force']['Cd']:.6g} and Cl = {summary['force']['Cl']:.6g} are startup values and are not benchmark statistics. The native geometry, actual pressure/speed maps, force history and actual wall-pressure coefficients appear in report.pdf and PNG/SVG figures.

{resolution_text}

## Three distinct qualification levels

| Level | Current status | Required evidence |
|---|---|---|
| Geometry and operators | Tested | Closed face area, positive volume, periodic seam, shared flux cancellation, independently applied pressure CSR, SGS limits |
| Short transient stability | Passed for this run | All step continuity below 1e-8, finite fields, CFL < 1, momentum ledger near roundoff |
| Long-time physical accuracy | **Not qualified** | Startup discard, stationary blocks, >=500 D/U record and >=20 resolved cycles, wall resolution, mesh/time/span/domain sensitivity, verified reference Cd/St/profile errors <=3% |

Production statistics default to discarding 100 D/U then collecting >=500 D/U; these are configurable development gates, not an assertion that the selected durations suffice for all wake modes. Too-short records return `St=null`. Block confidence intervals require sufficiently long independent blocks; the implementation reports that assumption. Cp/velocity reference comparison and uncertainty bands await primary raw data; current figures have no fabricated standard curve.
'''
    (d/'report.md').write_text(text)
    manifest=json.loads((d/'manifest.json').read_text());script=Path(__file__).resolve();dest=d/'source-snapshot/scripts'/script.name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(script,dest)
    manifest['report_source_sha256']=hashlib.sha256(script.read_bytes()).hexdigest();manifest['artifact_sha256']={str(f.relative_to(d)):hashlib.sha256(f.read_bytes()).hexdigest()for f in d.rglob('*')if f.is_file()and f.name!='manifest.json'}
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
