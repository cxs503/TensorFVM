"""Publish saved three-element scalar evidence and figures without rerunning."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.verification.core import write_json,artifact_manifest,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output
    summary=json.loads((out/'summary.json').read_text());audit=json.loads((out/'audit.json').read_text());rows=[]
    with PdfPages(out/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.3,11.7));fig.text(.08,.93,'30P30N static overset verification',fontsize=19)
        text='''Problem: three distinct solid profiles (slat, main, flap), normalized chord.
One Gmsh fitted component contains all three physical walls and fluid slots;
a Cartesian background supplies the far field. This is a two-grid case,
not three independently moving fitted grids.

Equation: -Laplacian(u) = -4.
Exact field: u = 1 + x^2 + y^2 + 0.2xy + 0.1x + 0.15y.
Exact Dirichlet data apply only at physical walls and far-field boundaries.
Fringe values are unknowns constrained by ACTIVE donor interpolation.
Nonorthogonal FV fluxes and least-squares gradients are solved together.

Acceptance: both component volume-weighted L2 errors < 3%;
unique physical-domain integrated flux error < 3%; algebraic residual
and actual donor constraints < 1e-10. Independent NumPy replay checks
raw polygons, source profiles, gradients, fluxes and donor constraints.

Scope: scalar equation verification. No overset pressure, velocity, Cp,
lift/drag or turbulence validation is claimed. Point interpolation does
not provide strict local conservative interface flux coupling.

Geometry source: linuxguy123/30P-30N-Validation-Case (GPLv3).
Packaged profile files and exact numerical producer hashes are archived.

Reproduce from the repository:
PYTHONPATH=src python scripts/run_three_element_overset.py --output NEW_DIR
PYTHONPATH=src python scripts/audit_three_element_overset.py NEW_DIR
PYTHONPATH=src python scripts/report_three_element_overset.py NEW_DIR
'''
        fig.text(.08,.87,text,va='top',fontsize=10,linespacing=1.6);pdf.savefig(fig);plt.close(fig)
        for row,ar in zip(summary['runs'],audit['runs']):
            d=out/row['directory'];m=row['computed'];di=row['connectivity'];f=np.load(d/'fields.npz');c=np.load(d/'connectivity.npz');offset=c['body_offsets'];bodies=[c['body_vertices'][offset[i]:offset[i+1]] for i in range(3)]
            rows.append([row['resolution'],sum(g['cells'] for g in di['grids']),max(m[f'grid_{k}_relative_l2_error'] for k in (0,1)),m['physical_global_conservation_defect_relative'],ar['independent_equation_max_residual'],row['passed']])
            for kind in ('mesh','solution','error'):
                fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
                for k,ax in enumerate(axes):
                    state=f[f'g{k}_state'];mask=state!=1;poly=f[f'g{k}_polygons'][mask]
                    if kind=='mesh':values=state[mask];collection=PolyCollection(poly,array=values,cmap='viridis',edgecolors='gray',linewidths=.08);collection.set_clim(0,2)
                    else:
                        values=f[f'g{k}_solution'][mask]
                        if kind=='error':values=abs(values-f[f'g{k}_exact'][mask])/np.abs(f[f'g{k}_exact'][mask])
                        collection=PolyCollection(poly,array=values,cmap='magma' if kind=='error' else 'viridis',edgecolors='none');fig.colorbar(collection,ax=ax,label='Relative absolute scalar error' if kind=='error' else 'Scalar u')
                    ax.add_collection(collection)
                    for body in bodies:ax.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.7)
                    ax.set_xlim((-1,3) if k==0 else (-.03,1.04));ax.set_ylim((-1.5,1.5) if k==0 else (-.15,.08));ax.set_aspect('equal');ax.set_title(('Cartesian background','Fitted three-element component')[k]);ax.set_xlabel('x / C');ax.set_ylabel('y / C')
                fig.suptitle(f"Background {row['resolution']}: {kind} (physical-slot close-up)")
                fig.savefig(d/f'{kind}.png',dpi=180);fig.savefig(d/f'{kind}.svg');pdf.savefig(fig);plt.close(fig)
            from scipy.interpolate import LinearNDInterpolator
            fig,ax=plt.subplots(figsize=(8,4));x=np.linspace(-.2,1.2,300);points=np.c_[x,np.full_like(x,-.15)];exact=1+x*x+.15**2-.03*x+.1*x-.0225
            ax.plot(x,exact,'k--',label='Analytic scalar')
            for k in (0,1):
                mask=f[f'g{k}_state']==0;sample=LinearNDInterpolator(f[f'g{k}_centers'][mask],f[f'g{k}_solution'][mask])(points);ax.plot(x,sample,label=f'Grid {k}')
            ax.set_xlabel('x / C');ax.set_ylabel('Scalar u at y/C=-0.15');ax.legend();ax.grid();fig.savefig(d/'curve.png',dpi=180);pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(8.3,11.7));fig.text(.08,.92,f"Grid {row['resolution']}: measured results",fontsize=18);fig.text(.08,.85,json.dumps(dict(cells=di['grids'],metrics=m,audit=ar),indent=2),va='top',family='monospace',fontsize=8);pdf.savefig(fig);plt.close(fig);f.close();c.close()
        fig,ax=plt.subplots(figsize=(7,5));ax.loglog([r[0] for r in rows],[r[2] for r in rows],'o-',label='Maximum component L2 error');ax.loglog([r[0] for r in rows],[r[3] for r in rows],'s-',label='Global diffusive balance');ax.axhline(.03,color='red',ls='--',label='3% threshold');ax.set_xlabel('Background x cells');ax.set_ylabel('Relative error');ax.legend();ax.grid(True,which='both');pdf.savefig(fig);fig.savefig(out/'refinement.png',dpi=180);plt.close(fig)
    header=['background_n','input_cells','maximum_relative_l2_error','global_balance_relative_error','independent_equation_max_residual','passed']
    with (out/'comparison.csv').open('w') as stream:w=csv.writer(stream);w.writerow(header);w.writerows(rows)
    table='\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)
    (out/'report.md').write_text('# 30P30N 三段翼 overset 验证\n\n'+text+'\n\n| '+ ' | '.join(header)+' |\n|'+ '|'.join(['---']*len(header))+'|\n'+table+'\n\n[PDF](report.pdf)\n')
    (out/'report.tex').write_text('\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n\\section{30P30N static overset scalar verification}\n\\begin{verbatim}\n'+text+'\\end{verbatim}\n'+''.join('\\includegraphics[width=\\linewidth]{'+r['directory']+'/solution.png}\n' for r in summary['runs'])+'\\end{document}\n')
    manifest=json.loads((out/'manifest.json').read_text());key='scripts/report_three_element_overset.py';manifest['source_sha256'][key]=sha(Path(__file__));target=out/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes());manifest['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()
