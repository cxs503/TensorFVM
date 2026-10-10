"""Paper-style supplementary figures and editable manuscript from saved fields."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.verification.report import figure,plt
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def process(output):
    s=json.loads((output/'summary.json').read_text());plots=[];pages=output/'supplement.pdf'
    with PdfPages(pages) as pdf:
        for row in s['runs']:
            d=output/row['directory'];f=np.load(d/'fields.npz');v=f['vertices_m'];cells=np.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),axis=2).reshape(-1,4,2);c=row['config'];figs=d/'figures'
            for name,value,label in [('u-component',f['cell_velocity_m_s'][:,0],'u (m/s)'),('v-component',f['cell_velocity_m_s'][:,1],'v (m/s)')]:
                fig,ax=plt.subplots(figsize=(8,4));p=PolyCollection(cells,array=value,cmap='coolwarm',rasterized=True);ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title=row['case']+' '+row['resolution']);fig.colorbar(p,ax=ax,label=label)
                if row['case']=='naca':ax.set_xlim(c['airfoil_x']-.3,c['airfoil_x']+2);ax.set_ylim(c['airfoil_y']-.5,c['airfoil_y']+.5)
                else:ax.set_xlim(0,.7)
                pdf.savefig(fig);figure(fig,figs,name);plots.append(row['directory']+'/figures/'+name)
            fig,ax=plt.subplots(figsize=(8,4));p=PolyCollection(cells,array=np.linalg.norm(f['cell_velocity_m_s'],axis=1),cmap='viridis',rasterized=True);ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='Wall and wake detail: '+row['resolution']);fig.colorbar(p,ax=ax,label='Speed (m/s)')
            if row['case']=='naca':ax.set_xlim(c['airfoil_x']-.3,c['airfoil_x']+2);ax.set_ylim(c['airfoil_y']-.5,c['airfoil_y']+.5)
            else:ax.set_xlim(0,.7)
            pdf.savefig(fig);figure(fig,figs,'wake-detail');plots.append(row['directory']+'/figures/wake-detail')
            fig,ax=plt.subplots(figsize=(8,4));ax.add_collection(PolyCollection(cells,facecolors='none',edgecolors='gray',linewidths=.15,rasterized=True));ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='Actual polygon mesh '+row['resolution'])
            if row['case']=='naca':ax.set_xlim(c['airfoil_x']-.1,c['airfoil_x']+1.2);ax.set_ylim(c['airfoil_y']-.3,c['airfoil_y']+.3)
            else:ax.set_xlim(.1,.4);ax.set_ylim(.1,.3)
            pdf.savefig(fig);figure(fig,figs,'mesh-detail');plots.append(row['directory']+'/figures/mesh-detail')
            f.close()
        for path in ['refinement']+[r['directory']+'/figures/residuals' for r in s['runs']]:
            fig,ax=plt.subplots(figsize=(8,5));ax.imshow(plt.imread(output/(path+'.png')));ax.axis('off');pdf.savefig(fig);plt.close(fig)
    with tempfile.TemporaryDirectory() as td:
        merged=Path(td)/'merged.pdf';subprocess.run(['pdfunite',str(output/'report.pdf'),str(pages),str(merged)],check=True);(output/'report.pdf').write_bytes(merged.read_bytes())
    tex=r'''\documentclass[11pt]{article}
\usepackage{graphicx,booktabs,geometry,hyperref}
\geometry{margin=22mm}
\title{Body-Fitted Finite-Volume External-Flow Verification}
\author{TensorFVM benchmark development}
\begin{document}\maketitle
\begin{abstract}
Actual steady laminar cylinder and NACA0012 calculations assess pressure, velocity and integrated traction on three polygon grids. Independent physical error gates are strictly below 3 percent. Numerical convergence, physical qualification and mesh convergence are reported separately; failures are retained.
\end{abstract}
\section{Physical problem and references}
The DFG 2D-1 cylinder has $Re=20$, diameter 0.1 m and channel size $2.2\times0.41$ m. Parabolic inlet mean velocity is 0.2 m/s. The FeatFlow references are $C_D=5.57953523384$, $C_L=0.010618948146$ and $\Delta p=0.11752016697$ Pa.
The NACA0012 has $Re=1000$, incidence 4 degrees, unit chord, and a $20\times16$ domain with leading edge at $(6,8)$.
Di Ilio et al. (2020), arXiv:2006.10487, figures 10 and 11 provide vector-digitized present-study markers $C_L=27.41/190.95\times1.4$ and $C_D=23.87/190.95$. Each carries a conservative graph-reading interval $\pm0.0001$, excluding author numerical uncertainty. Worst endpoint relative errors are used. The historical 0.205/0.120 reference and pass claim are withdrawn. No matching 4-degree tabulated $C_p$ reference is claimed.
\section{Numerical method and reproducibility}
Production collocated SIMPLE uses Rhie--Chow mass flux, first-order upwind momentum transport and nonorthogonal diffusion. CPU SuperLU solves identical sparse equations; CUDA optionally uses Torch Krylov. True linear residuals and final steady momentum, continuity and mass imbalance are checked. Circle and airfoil walls are polygon chords. The airfoil grid is periodic radial topology despite its c-grid API name. Owner pressure traction and a no-slip reconstructed molecular velocity gradient give body forces. Surface pressure is separately least-squares extrapolated for plots and cylinder pressure difference. No analytic solution initializes these external-flow fields.
Run the common module \texttt{tensorfvm.verification.external\_flow}; exact commands are in report.md. Raw NPZ fields, complete histories, metrics, per-solve residuals, source/artifact SHA256 and independent NumPy force audits are supplied. Five-step CPU/CUDA agreement is an auxiliary check only.
\section{Results}
\begin{tabular}{lrrrrl}\toprule Grid & Cells & Iterations & $C_D$ & $C_L$ & Qualified\\\midrule
'''
    for r in s['runs']:tex+=f"{r['resolution']} & {r['config']['nx']*r['config']['ny']} & {r['iterations']} & {r['computed']['drag']:.8g} & {r['computed']['lift']:.8g} & {r['passed']} \\\\\n"
    tex+='\\bottomrule\\end{tabular}\n\\section{Fields, mesh and refinement}\n'
    for name in plots+['refinement']:tex+='\\begin{figure}[p]\\centering\\includegraphics[width=.94\\linewidth]{'+name+'.pdf}\\caption{'+name.replace('/',' ').replace('-',' ')+'}\\end{figure}\n'
    tex+='\\section{Discussion and limits} Three resolutions do not automatically establish an asymptotic regime. First-order upwind, polygon geometry, wall traction and finite-domain effects remain candidates for error separation. No matched TensorLBM runs or performance ranking are claimed. This is steady two-dimensional laminar verification, not high-Reynolds-number shedding or industrial airfoil certification.\\end{document}\n';(output/'report.tex').write_text(tex)
    with (output/'report.md').open('a') as f:
        f.write('\n## 局部场与实际网格\n\n可编辑论文源见 `report.tex`，完整PDF含局部速度分量、尾流和网格。\n')
        for name in plots:f.write(f'\n![{name}]({name}.png)\n')
    m=json.loads((output/'manifest.json').read_text());m['source_sha256']['scripts/publish_external_manuscript.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',nargs='+',type=Path);a=p.parse_args()
    for output in a.outputs:process(output)
