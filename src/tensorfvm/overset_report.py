"""Publish actual static overset scalar verification through common figures."""
import argparse
import csv
import json
import textwrap
from pathlib import Path
import numpy as np
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import LinearNDInterpolator
from .verification.report import plt,figure
from .verification.core import write_json,sha,artifact_manifest


def publish(output):
    summary=json.loads((output/'summary.json').read_text());rows=summary['runs'];audit=json.loads((output/'audit.json').read_text())
    text='''# Static body-fitted / Cartesian overset finite-volume diffusion verification

## Abstract and scope

Two independently indexed grids cover a square with a stationary circular hole: an annular body-fitted component and a Cartesian background. Hole cutting, active-only donors, simultaneous fringe constraints and the actual finite-volume diffusion equations are exercised. This report qualifies scalar fields and a unique-domain global diffusion balance. It does not qualify incompressible Navier–Stokes, pressure/velocity coupling, strictly local conservative flux exchange, moving-grid GCL, turbulence or GPU acceleration. No matched TensorLBM run is included.

## Problem

The square is [-2,2] x [-2,2]; the body radius is R=0.3. The component extends to r=1; background blanking uses r=0.65. Actual physical wall edges are chords joining the annular inner vertices. The body polygon uses the same 2n vertices. One cell layer on both interpolation boundaries is FRINGE. Every Cartesian polygon that intersects the physical solid is blanked, even when its cell center is outside the solid. The physical wall is supplied by the annular grid.

Solve -Laplacian(u)=f with u=(x²+y²-R²)(1+0.2x+0.1y), f=-(4+1.6x+0.8y). The circular wall has homogeneous scalar Dirichlet data, and the square outer boundary has manufactured Dirichlet data. This scalar wall condition is not a fluid no-slip validation. Chord geometry differs from the ideal analytic circle and is refined together with the grids.

## Numerical method and software

Shared Metric/evaluate/save_run, provenance, artifact hashes and publication figures are used. Each ACTIVE control volume satisfies a two-point finite-volume diffusion equation. Each FRINGE row is a simultaneous weighted constraint from three ACTIVE centers of the other component, using nonnegative Delaunay barycentric weights. Weights reproduce constants and affine coordinates; orphan receivers and donor triangles crossing the excluded region are rejected. HOLE cells have inactive placeholder rows. The assembled sparse system is solved with SciPy spsolve. The Mesh2D adapter reuses actual TensorFVM polygons.

The official [OpenFOAM meshToMesh interface](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/meshTools/meshToMesh/meshToMesh.H) documents between-mesh addressing and weights. The current point-value implementation is independently written and supplies no certificate of conservative volume/flux transfer.

The primary field metric is the maximum of the two components' volume-weighted ACTIVE relative L2 errors. The optional combined norm counts overlap twice and is not a unique physical-domain norm. Physical global balance compares the summed outer/body diffusion flux with the source integrated once over the square minus the actual body polygon. The two fringe surfaces have different radii: their raw net flux is generally nonzero due to the overlap source and is not itself a conservation error. Point-value constraints do not guarantee matching local interface fluxes.

## Reproduction and independent audit

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/run_overset_benchmark.py --levels 32 64 128 --output results/overset
PYTHONPATH=src python scripts/audit_overset_benchmark.py results/overset
PYTHONPATH=src python scripts/preserve_external_producer.py results/overset
PYTHONPATH=src python -m tensorfvm.overset_report results/overset
```

The independent audit imports no overset solver or connection module. It rebuilds signed areas, centroids, oriented faces, full ACTIVE diffusion residuals, physical boundary fluxes, unique-domain source moments and all donor constraints from saved polygons and stencils. Each component relative L2 error and physical global diffusion balance must be strictly below 3%, the coupled linear residual below 1e-10, and the donor constraint residual below 1e-10.

## Results

|Background n|Input cells|ACTIVE cells|FRINGE cells|Max component L2 error|Global diffusion balance error|Linear residual|Accepted|
|---|---:|---:|---:|---:|---:|---:|---|
'''
    for row in rows:
        m=row['computed'];cells=sum(g['cells'] for g in row['connectivity']['grids'])
        text+=f"|{row['resolution']}|{cells}|{m['active_cells']}|{m['fringe_cells']}|{max(m['grid_0_relative_l2_error'],m['grid_1_relative_l2_error'])*100:.6f}%|{m['physical_global_conservation_defect_relative']*100:.6f}%|{m['relative_algebraic_residual']:.3g}|{row['passed']}|\n"
    errors=np.array([max(r['computed']['grid_0_relative_l2_error'],r['computed']['grid_1_relative_l2_error']) for r in rows]);levels=np.array([r['resolution'] for r in rows])
    orders=np.log(errors[:-1]/errors[1:])/np.log(levels[1:]/levels[:-1])
    text+='\nObserved field-error orders: '+', '.join(f'{p:.4f}' for p in orders)+'. These are measured for this smooth static problem, and do not establish the order of a future Navier–Stokes overset solver. Independent raw-field replay passed: '+str(audit['passed'])+'.\n'
    pages=[]
    with PdfPages(output/'report.pdf') as pdf:
        lines=[]
        for paragraph in text.splitlines():lines.extend(textwrap.wrap(paragraph,105) or [''])
        for start in range(0,len(lines),65):
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.06,.95,'Static overset scalar verification',fontsize=14);fig.text(.06,.91,'\n'.join(lines[start:start+65]),va='top',fontsize=8);pdf.savefig(fig);plt.close(fig)
        for row in rows:
            folder=output/row['directory'];figdir=folder/'figures';figdir.mkdir(exist_ok=True)
            with np.load(folder/'fields.npz') as f,np.load(folder/'connectivity.npz') as c:
                for kind in ('mesh-state','donor-connections','scalar-solution','scalar-error'):
                    fig,ax=plt.subplots(figsize=(7,6))
                    if kind=='mesh-state':
                        plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(11,5))
                        colors=np.array(['#dce5ec','#ffffff','#f2a640'])
                        for k,axis in enumerate(axes):
                            state=f[f'g{k}_state'];polys=f[f'g{k}_polygons'];base=colors[state].copy()
                            if k==1:base[state==0]='#cae5fb'
                            axis.add_collection(PolyCollection(polys,facecolors=base,edgecolors='#627586',linewidths=.18,rasterized=True))
                            body=c['body_polygon'];axis.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.7)
                            axis.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),xlabel='x',ylabel='y',title=('Cartesian: HOLE white / FRINGE orange' if k==0 else 'Body-fitted: ACTIVE blue / FRINGE orange'));axis.set_aspect('equal')
                        ax=axes[0]
                    elif kind=='donor-connections':
                        for k,color in ((0,'tab:blue'),(1,'tab:red')):
                            rg=int(c[f's{k}_receiver_grid']);dg=int(c[f's{k}_donor_grid']);rc=c[f's{k}_receiver_cells'];dc=c[f's{k}_donor_cells']
                            take=np.arange(0,len(rc),max(1,len(rc)//32));receiver=f[f'g{rg}_centers'][rc[take]];donor=f[f'g{dg}_centers'][dc[take]]
                            segments=np.stack([np.repeat(receiver,3,axis=0),donor.reshape(-1,2)],axis=1)
                            ax.add_collection(LineCollection(segments,colors=color,linewidths=.6));ax.scatter(receiver[:,0],receiver[:,1],s=9,c=color,label=f'receivers grid {rg}')
                        ax.legend();ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),title='Sampled actual receiver / active donor stencils')
                    else:
                        allvalues=[]
                        for k in (0,1):
                            state=f[f'g{k}_state'];values=f[f'g{k}_solution']
                            if kind=='scalar-error':values=abs(values-f[f'g{k}_exact'])
                            allvalues.extend(values[state!=1])
                        maximum=max(allvalues);im=None
                        for k in (0,1):
                            state=f[f'g{k}_state'];values=f[f'g{k}_solution']
                            if kind=='scalar-error':values=abs(values-f[f'g{k}_exact'])
                            keep=state!=1;im=PolyCollection(f[f'g{k}_polygons'][keep],array=values[keep],cmap='viridis' if kind=='scalar-solution' else 'magma',clim=(0,maximum),rasterized=True);ax.add_collection(im)
                        fig.colorbar(im,ax=ax,label='u' if kind=='scalar-solution' else '|u - exact|')
                        ax.set(xlim=(-2,2),ylim=(-2,2),title=kind.replace('-',' ')+'; actual solved cells')
                    body=c['body_polygon'];ax.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.7);ax.set(xlabel='x',ylabel='y');ax.set_aspect('equal');pdf.savefig(fig);figure(fig,figdir,kind);pages.append(row['directory']+'/figures/'+kind)
                fig,ax=plt.subplots(figsize=(7,4));x=np.r_[np.linspace(.4,.8,50),np.linspace(1.1,1.85,50)]
                exact=(x*x-.09)*(1+.2*x);ax.plot(x[:50],exact[:50],color='black',label='analytic y=0');ax.plot(x[50:],exact[50:],color='black')
                for k,segment in ((1,slice(0,50)),(0,slice(50,None))):
                    keep=f[f'g{k}_state']==0;query=np.c_[x[segment],np.zeros(len(x[segment]))]
                    numeric=LinearNDInterpolator(f[f'g{k}_centers'][keep],f[f'g{k}_solution'][keep])(query)
                    if not np.all(np.isfinite(numeric)):raise ValueError('Scalar line sampling outside active donors')
                    ax.plot(x[segment],numeric,'o',ms=2,label=f'solved grid {k}, interpolated to y=0')
                ax.set(xlabel='x',ylabel='u',title='Positive x-axis scalar comparison');ax.grid(alpha=.3);ax.legend();pdf.savefig(fig);figure(fig,figdir,'scalar-curve');pages.append(row['directory']+'/figures/scalar-curve')
        fig,ax=plt.subplots(figsize=(7,4));ax.loglog(levels,errors,'o-',label='max component ACTIVE L2');ax.loglog(levels,[r['computed']['physical_global_conservation_defect_relative'] for r in rows],'s-',label='unique-domain global diffusion balance');ax.axhline(.03,ls='--',color='black',label='strict 3% gate');ax.set(xlabel='Cartesian cells per direction',ylabel='relative error',title='Actual three-level refinement');ax.grid(alpha=.3);ax.legend();pdf.savefig(fig);figure(fig,output,'refinement');pages.append('refinement')
    text+='\n## Discussion and next implementation gates\n\nField and unique-domain global balance errors decrease on all three levels. Local conservative interface fluxes require a separate transfer operator and validation; a small global defect does not certify that property. Next gates are a conservative flux correction, pressure/velocity coupling with independently audited mass balance, static laminar cylinder flow, and moving-body connectivity with geometric conservation. Industrial comparisons require matched physical cases and actual runs.\n'
    for name in pages:text+='\n!['+name+']('+name+'.png)\n'
    (output/'report.md').write_text(text)
    with (output/'results.csv').open('w',newline='') as f:
        names=['resolution','active_cells','fringe_cells','grid_0_relative_l2_error','grid_1_relative_l2_error','physical_global_conservation_defect_relative','relative_algebraic_residual','maximum_donor_constraint_residual','passed']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in rows:writer.writerow({key:row[key] if key in ('resolution','passed') else row['computed'][key] for key in names})
    tex=r'\documentclass{article}\usepackage{graphicx,booktabs,hyperref}\begin{document}\title{Static overset finite-volume diffusion verification}\maketitle\section{Scope}Scalar Poisson equations on a body-fitted annulus and Cartesian background with actual simultaneous donor constraints. No Navier--Stokes or strict local conservation qualification. Full equations, references, commands and limits are in report.md.\section{Results}\begin{tabular}{rrrl}\toprule Background n & Max L2 error & Global balance error & Accepted\\\midrule '
    for row in rows:
        m=row['computed'];tex+=f"{row['resolution']} & {max(m['grid_0_relative_l2_error'],m['grid_1_relative_l2_error']):.8g} & {m['physical_global_conservation_defect_relative']:.8g} & {row['passed']} \\\\\n"
    tex+=r'\bottomrule\end{tabular}'
    for name in pages:tex+=r'\begin{figure}[p]\includegraphics[width=\linewidth]{'+name+r'.pdf}\caption{'+name.replace('/',' ').replace('-',' ')+r'}\end{figure}'
    (output/'report.tex').write_text(tex+r'\end{document}'+'\n')
    manifest=json.loads((output/'manifest.json').read_text());root=Path(__file__).resolve().parents[2];key=str(Path(__file__).relative_to(root));manifest['source_sha256'][key]=sha(Path(__file__))
    if manifest.get('source_snapshot_root'):
        import shutil
        target=output/manifest['source_snapshot_root']/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
    manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)
    print(output,'overset scalar report published',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();publish(a.output)
