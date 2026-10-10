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
    text='''# NACA0012 body-fitted / Cartesian overset scalar verification

## Abstract and scope

Actual NACA0012 meshes with ordered wall quadrilaterals and outer triangles overlap a Cartesian background. The airfoil is rotated clockwise by 4 degrees about its quarter-chord point. Three coupled finite-volume diffusion systems are solved with active-only two-way donor constraints and independently replayed from raw arrays. This report qualifies static scalar diffusion and mesh connection. Incompressible Navier–Stokes, lift/drag/Cp, local conservative flux transfer, moving-grid GCL, turbulence and GPU performance remain unverified.

## Geometry and problem

Unit chord NACA0012 has unrotated leading edge (1.25,1.5) and quarter chord (1.5,1.5). Gmsh interpolating splines through 161 cosine-spaced profile points define the closed body. Actual wall edges are linear chords. The component occupies [0,4] x [0,3]; the Cartesian background occupies [-2,6] x [-2,5]. The blanking contour is [0.5,3.5] x [0.5,2.5]. One cell layer is FRINGE on each transfer boundary. All Cartesian cells touching the physical solid are excluded.

Solve -Laplacian(u)=-4 with analytic u=1+x²+y²+0.2xy+0.1x+0.15y. Analytic scalar Dirichlet data apply on the physical outer rectangle boundary and actual polygonal airfoil wall. The arbitrary scalar wall values are manufactured boundary data, and do not represent fluid no-slip. No analytic data are supplied to either artificial interpolation boundary.

## Numerical method

The shared Mesh2D adapter, Metric/evaluate/save_run, SHA256 provenance and figure modules are used. The [Gmsh BoundaryLayer/Distance/Threshold fields](https://gmsh.info/doc/texinfo/gmsh.html) generate the component mesh. First wall-layer thickness is 0.0005 times scale, growth 1.18, cap 0.025. Outer component size is limited to 0.06 times scale to preserve a resolved overlap. Tail fan and wake refinement are retained.

The original unlimited outer size produced 33 orphan background receivers at the coarse level and was rejected before a field solve. The actual failed geometry, source snapshot and error are preserved in ../verification-overset-naca-connectivity-failed. The corrected hierarchy has no orphans. This repair changes overlap resolution and leaves the physical geometry and manufactured equation fixed.

ACTIVE rows satisfy finite-volume diffusion with the full nonorthogonal flux correction. Least-squares cell gradients use adjacent available cells and physical Dirichlet boundaries. The face-area vector is decomposed into a centroid-connector component plus a tangential correction; both contributions are assembled into the same sparse matrix. FRINGE rows simultaneously constrain values to three ACTIVE donor centers on the other grid using nonnegative barycentric interpolation. The matrix is solved by SciPy spsolve; fields are actual numerical solutions, not analytic replacements.

The primary scalar error is the maximum of the two volume-weighted ACTIVE component L2 errors. Physical global balance uses the source integral once over the Cartesian rectangle minus the actual airfoil polygon. Point-value interpolation does not enforce strictly local interface flux equality, even when the global balance is accurate. No matched TensorLBM or commercial-software run is included.

## Software and independent verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/run_naca_overset.py --levels 32 64 128 --output results/naca-overset
PYTHONPATH=src python scripts/audit_naca_overset.py results/naca-overset
PYTHONPATH=src python scripts/preserve_external_producer.py results/naca-overset
PYTHONPATH=src python -m tensorfvm.overset_naca_report results/naca-overset
```

The NumPy audit imports no production overset or scalar solver module. It reconstructs polygon areas, centroids, face incidence, LS gradients, complete nonorthogonal diffusion, physical boundary flux and all donor rows. A separately sampled 4097-point analytic NACA profile checks body vertices after inverse geometry rotation. Source/artifact snapshots preserve the actual numerical producer. Each component L2 error and unique-domain global diffusion balance must be strictly below 3%, while linear and donor residual limits are 1e-10.

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
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.06,.95,'NACA0012 static overset scalar verification',fontsize=14);fig.text(.06,.91,'\n'.join(lines[start:start+65]),va='top',fontsize=8);pdf.savefig(fig);plt.close(fig)
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
                            axis.set(xlim=(-.2,4.2),ylim=(-.2,3.2),xlabel='x',ylabel='y',title=('Cartesian: HOLE white / FRINGE orange' if k==0 else 'Body-fitted: ACTIVE blue / FRINGE orange'));axis.set_aspect('equal')
                        ax=axes[0]
                    elif kind=='donor-connections':
                        for k,color in ((0,'tab:blue'),(1,'tab:red')):
                            rg=int(c[f's{k}_receiver_grid']);dg=int(c[f's{k}_donor_grid']);rc=c[f's{k}_receiver_cells'];dc=c[f's{k}_donor_cells']
                            take=np.arange(0,len(rc),max(1,len(rc)//32));receiver=f[f'g{rg}_centers'][rc[take]];donor=f[f'g{dg}_centers'][dc[take]]
                            segments=np.stack([np.repeat(receiver,3,axis=0),donor.reshape(-1,2)],axis=1)
                            ax.add_collection(LineCollection(segments,colors=color,linewidths=.6));ax.scatter(receiver[:,0],receiver[:,1],s=9,c=color,label=f'receivers grid {rg}')
                        ax.legend();ax.set(xlim=(.2,3.8),ylim=(.2,2.8),title='Sampled actual receiver / active donor stencils')
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
                        ax.set(xlim=(-2,6),ylim=(-2,5),title=kind.replace('-',' ')+'; actual solved cells')
                    body=c['body_polygon'];ax.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.7);ax.set(xlabel='x',ylabel='y');ax.set_aspect('equal');pdf.savefig(fig);figure(fig,figdir,kind);pages.append(row['directory']+'/figures/'+kind)
                fig,ax=plt.subplots(figsize=(7,4));x=np.linspace(.7,3.3,100);y=np.full_like(x,1.2)
                analytic=1+x*x+y*y+.2*x*y+.1*x+.15*y;ax.plot(x,analytic,color='black',label='analytic y=1.2')
                keep=f['g1_state']==0;numeric=LinearNDInterpolator(f['g1_centers'][keep],f['g1_solution'][keep])(np.c_[x,y])
                if not np.isfinite(numeric).all():raise ValueError('Sampling outside active component')
                ax.plot(x,numeric,'o',ms=2,label='actual component field, interpolated to line');ax.set(xlabel='x',ylabel='u',title='Scalar profile below the airfoil');ax.grid(alpha=.3);ax.legend();pdf.savefig(fig);figure(fig,figdir,'scalar-curve');pages.append(row['directory']+'/figures/scalar-curve')
        fig,ax=plt.subplots(figsize=(7,4));ax.loglog(levels,errors,'o-',label='max component ACTIVE L2');ax.loglog(levels,[r['computed']['physical_global_conservation_defect_relative'] for r in rows],'s-',label='unique-domain global diffusion balance');ax.axhline(.03,ls='--',color='black',label='strict 3% gate');ax.set(xlabel='Cartesian cells per direction',ylabel='relative error',title='Actual three-level refinement');ax.grid(alpha=.3);ax.legend();pdf.savefig(fig);figure(fig,output,'refinement');pages.append('refinement')
    text+='\n## Discussion and next implementation gates\n\nField and unique-domain global balance errors decrease on all three levels. Local conservative interface fluxes require a separate transfer operator and validation; a small global defect does not certify that property. Next gates are a conservative flux correction, pressure/velocity coupling with independently audited mass balance, static laminar cylinder flow, and moving-body connectivity with geometric conservation. Industrial comparisons require matched physical cases and actual runs.\n'
    for name in pages:text+='\n!['+name+']('+name+'.png)\n'
    (output/'report.md').write_text(text)
    with (output/'results.csv').open('w',newline='') as f:
        names=['resolution','active_cells','fringe_cells','grid_0_relative_l2_error','grid_1_relative_l2_error','physical_global_conservation_defect_relative','relative_algebraic_residual','maximum_donor_constraint_residual','passed']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in rows:writer.writerow({key:row[key] if key in ('resolution','passed') else row['computed'][key] for key in names})
    tex=r'\documentclass{article}\usepackage{graphicx,booktabs,hyperref}\begin{document}\title{NACA0012 static overset scalar verification}\maketitle\section{Scope}Scalar Poisson equations on a NACA0012 component and Cartesian background with actual simultaneous donor constraints. No Navier--Stokes or strict local conservation qualification. Full equations, references, commands and limits are in report.md.\section{Results}\begin{tabular}{rrrl}\toprule Background n & Max L2 error & Global balance error & Accepted\\\midrule '
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
    print(output,'NACA overset scalar report published',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();publish(a.output)
