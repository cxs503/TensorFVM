"""Replace the machine summary page by manuscript problem/method/results pages."""
import argparse,json,re,subprocess,tempfile,textwrap
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.verification.report import plt
from tensorfvm.verification.core import write_json,sha,artifact_manifest


def process(output):
    s=json.loads((output/'summary.json').read_text());rows=s['runs'];name=rows[0]['case'];equations=json.loads((output/'equations-audit.json').read_text());audits={r['resolution']:r for r in equations['runs']}
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);front=td/'front.pdf'
        with PdfPages(front) as pdf:
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,'Body-fitted '+('DFG cylinder' if name=='cylinder' else 'NACA0012')+' verification',fontsize=17)
            sections=[('Abstract','Three actual polygon-grid simulations assess steady laminar external flow. Numerical convergence and strict physical error below 3% are separate gates. Complete raw fields, force components, histories, independent NumPy equation audits and source hashes are supplied. Failed metrics remain failed.'),('1. Physical problem', 'DFG 2D-1: Re=20, rho=1 kg/m3, mu=0.001 Pa s, channel 2.2 x 0.41 m, cylinder center (0.2,0.2) m and diameter 0.1 m. Parabolic inlet has mean speed 0.2 m/s. Walls are no-slip; outlet pressure is zero.' if name=='cylinder' else 'NACA0012: Re=1000, angle of attack 4 degrees, unit chord, domain 20 x 16, leading edge (6,8). Laminar, incompressible, steady, two-dimensional flow. The public reference is Di Ilio et al. (2020), arXiv:2006.10487, figures 10 and 11.'),('2. Method and mesh','Collocated SIMPLE uses first-order upwind momentum transport, nonorthogonal diffusion and Rhie-Chow face flux. Walls are polygon chords. CPU SuperLU solves the same finite-volume linear equations with true residual checks. '+('A full-grid coupled Picard residual correction retains periodic connectivity and exact assembled operator probes. Final residuals are recomputed after correction; physical flux defect must be below 1e-7.' if name=='cylinder' else 'The original SIMPLE equations are retained; the finest grid uses current-matrix GMRES with cached LU preconditioning and existing Anderson mixing. Post-mixing residuals and physical flux are explicitly checked. The radial periodic airfoil grid keeps its c-grid API name. Coupled Picard acceleration is not certified for airfoils and is rejected by the production interface.')),('3. Acceptance and reproducibility','Each integrated physical reference error must be strictly below 3%; steady momentum, continuity and boundary imbalance must be below 1e-6. Independent NumPy gradients, complete momentum residuals, mass and inertia-free physical Rhie-Chow flux are audited. Grid refinement alone does not prove an asymptotic regime. Exact commands and editable LaTeX source accompany this PDF. Elapsed solve time excludes mesh setup, export and plotting; no matched TensorLBM performance comparison is claimed.'),('4. Reference integrity','FeatFlow DFG 2D-1 high-accuracy tabulated values: Cd=5.57953523384, Cl=0.010618948146, front/rear pressure difference=0.11752016697 Pa.' if name=='cylinder' else 'Vector-digitized present-study markers give Cl=0.2009636030 and Cd=0.1250065462. Each has a conservative graph-reading interval +/-0.0001. The worst relative error at either endpoint is used. This interval excludes author numerical uncertainty. The historical 0.205/0.120 pass claim is withdrawn. No matching four-degree Cp reference is claimed.')]
            y=.88
            for heading,body in sections:
                fig.text(.08,y,heading,fontsize=12,weight='bold',va='top');y-=.03;wrapped='\n'.join(textwrap.wrap(body,100));fig.text(.08,y,wrapped,fontsize=9,va='top',linespacing=1.5);y-=.019*len(wrapped.splitlines())+.035
            pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,'5. Results and independent audits',fontsize=16)
            fig.text(.08,.89,'CPU float64; three actual grids. Physical errors are evaluated separately.',fontsize=10)
            headers=['Grid','Cells','Steps','Steady','Cd','Cl','Max physical\nerror (%)','Qualified']
            data=[]
            for r in rows:
                e=max(q['value'] for q in r['metrics'] if q['name'].endswith('relative_error'));data.append([r['resolution'],r['config']['nx']*r['config']['ny'],r['iterations'],str(r['converged']),f"{r['computed']['drag']:.7g}",f"{r['computed']['lift']:.7g}",f'{100*e:.3f}',str(r['passed'])])
            ax=fig.add_axes([.06,.67,.88,.17]);ax.axis('off');table=ax.table(cellText=data,colLabels=headers,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
            fig.text(.08,.61,'Independent equations reconstructed from saved fields',fontsize=12,weight='bold')
            data=[[r['resolution'],f"{audits[r['resolution']]['independent_momentum']:.3e}",f"{audits[r['resolution']]['independent_continuity']:.3e}",f"{audits[r['resolution']]['physical_rhie_chow_relative_l2_defect']:.3e}",str(audits[r['resolution']]['physical_fixedpoint_passed'])] for r in rows]
            ax=fig.add_axes([.06,.40,.88,.17]);ax.axis('off');table=ax.table(cellText=data,colLabels=['Grid','Momentum','Continuity','Physical flux\ndefect','Fixed point'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
            conclusion='All physical gates must pass independently. Iterative convergence cannot replace physical accuracy. The finest mesh has '+str(rows[-1]['config']['nx']*rows[-1]['config']['ny'])+' control volumes and '+str(3*rows[-1]['config']['nx']*rows[-1]['config']['ny'])+' primary velocity/pressure unknowns. First-order transport, wall traction, polygon geometry and finite-domain effects require further separation before qualification.'
            fig.text(.08,.33,'6. Discussion and limits',fontsize=12,weight='bold');fig.text(.08,.29,'\n'.join(textwrap.wrap(conclusion,100)),fontsize=10,va='top',linespacing=1.6)
            pdf.savefig(fig);plt.close(fig)
        old=output/'report.pdf';info=subprocess.check_output(['pdfinfo',str(old)],text=True);count=int(re.search(r'Pages:\s+(\d+)',info).group(1));subprocess.run(['pdfseparate','-f','2','-l',str(count),str(old),str(td/'page-%d.pdf')],check=True);merged=td/'complete.pdf';subprocess.run(['pdfunite',str(front),*[str(td/f'page-{i}.pdf') for i in range(2,count+1)],str(merged)],check=True);old.write_bytes(merged.read_bytes())
    with (output/'report.md').open('a') as f:f.write('\n## 耗时口径\n\n`elapsed_s` 从求解器构造之后开始，包含实际迭代及线性组装/求解，不包括网格准备、文件导出和作图。各工况单次运行不作统计性能资格或TensorLBM加速比。\n')
    m=json.loads((output/'manifest.json').read_text());m['source_sha256']['scripts/publish_external_frontmatter.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path,nargs='+');a=p.parse_args()
    for output in a.outputs:process(output)
