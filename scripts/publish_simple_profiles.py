"""Add quantitative pressure/velocity curves and reproducible manuscript source."""
import argparse,json,math,subprocess,tempfile
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest
from tensorfvm.verification.report import figure,plt
from matplotlib.backends.backend_pdf import PdfPages

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();directory=args.directory;s=json.loads((directory/'summary.json').read_text());figs=directory/'figures';quant=[];plots=[]
for case in s['cases']:
    row=case['cuda'];c=row['config'];f=np.load(directory/case['name']/'cuda/fields.npz');ny,nx=c['ny'],c['nx'];center=f['cell_centers_m'].reshape(ny,nx,2);u=f['velocity_m_s'].reshape(ny,nx,2);p=f['pressure_pa'].reshape(ny,nx)
    fig,ax=plt.subplots(figsize=(7,4));j=ny//2;ax.plot(center[j,:,0],p[j],label='Computed CUDA pressure')
    if row['case']=='curved':ax.plot(center[j,:,0],f['reference_pressure_pa'].reshape(ny,nx)[j],'--',label='Continuous manufactured reference')
    ax.set(xlabel='x (m)',ylabel='Pressure (Pa)',title=case['name']+' center-row pressure');ax.legend();ax.grid(alpha=.3);name=case['name']+'-pressure-curve';figure(fig,figs,name);plots.append(name)
    fig,ax=plt.subplots(figsize=(7,4));i=nx//2;y=center[:,i,1]
    for component,label in [(0,'u'),(1,'v')]:ax.plot(u[:,i,component],y,label='Computed '+label)
    if row['case']=='curved':
        ref=f['reference_velocity_m_s'].reshape(ny,nx,2)
        for component,label in [(0,'u'),(1,'v')]:ax.plot(ref[:,i,component],y,'--',label='Reference '+label)
    ax.set(xlabel='Velocity (m/s)',ylabel='Physical y (m)',title=case['name']+' mid-channel profile');ax.legend();ax.grid(alpha=.3);name=case['name']+'-velocity-profile';figure(fig,figs,name);plots.append(name)
    fig,ax=plt.subplots(figsize=(7,4));hist=json.loads((directory/case['name']/'cuda/history.json').read_text())
    for key in ['momentum','continuity']+(['turbulence'] if row['case']=='sa' else []):ax.semilogy(np.arange(1,len(hist)+1),[r[key] for r in hist],label=key)
    ax.axhline(c['tolerance'],ls='--',color='k',label='Fixed convergence gate');ax.set(xlabel='Actual SIMPLE outer iteration',ylabel='Residual',title=case['name']+' convergence');ax.legend();ax.grid(alpha=.3);name=case['name']+'-residuals';figure(fig,figs,name);plots.append(name)
    if row['case']=='sa':
        wall=f['wall_face_mask'];owners=f['face_owner'][wall];S=f['face_area_vectors_m'][wall];area=np.linalg.norm(S,axis=1);shear=f['wall_viscous_force_n_m'][:,0]/area;cf=shear/(.5*c['density']*c['inlet_velocity']**2);nu=c['inlet_velocity']*c['length']/c['reynolds'];yp=np.sqrt(abs(shear)/c['density'])*f['cell_centers_m'].reshape(-1,2)[owners,1]/nu;x=f['face_centers_m'][wall,0]
        np.savetxt(directory/case['name']/'wall-cf-yplus.csv',np.column_stack((x,cf,yp)),delimiter=',',header='x_m,local_cf,y_plus',comments='')
        fig,axes=plt.subplots(2,1,figsize=(7,6));axes[0].plot(x,cf);axes[0].set(xlabel='x (m)',ylabel='Local Cf');axes[1].plot(x,yp);axes[1].axhline(1,ls='--',color='k');axes[1].set(xlabel='x (m)',ylabel='Actual local y+');fig.suptitle(case['name']+' wall diagnostics');fig.tight_layout();name=case['name']+'-cf-yplus';figure(fig,figs,name);plots.append(name)
    quant.append(dict(name=case['name'],cells=nx*ny,primary_flow_unknowns=nx*ny*(4 if row['case']=='sa' else 3),cpu_full_solve_s=case['cpu']['full_solve_elapsed_s'],cuda_full_solve_s=row['full_solve_elapsed_s'],metrics={q['name']:q['value'] for q in row['metrics']},strict_physical_passed=row['passed'],steady_equivalent=case['steady_equivalent'],window_speed=case['window_cpu_over_cuda']))
orders=[];curved=[q for q in quant if q['name'].startswith('curved-')]
for a,b in zip(curved[:-1],curved[1:]):
    ratio=math.sqrt(b['cells']/a['cells']);orders.append(dict(coarse=a['name'],fine=b['name'],velocity_order=math.log(a['metrics']['velocity_volume_l2']/b['metrics']['velocity_volume_l2'])/math.log(ratio),pressure_order=math.log(a['metrics']['pressure_volume_l2']/b['metrics']['pressure_volume_l2'])/math.log(ratio)))
write_json(directory/'quantitative-results.json',dict(cases=quant,observed_orders=orders,producer_sha256=sha(Path(__file__)),input_summary_sha256=sha(directory/'summary.json'),scope='single complete solve times; separate three-repetition five-step window; measured orders, no imposed order'))
text='\n## 定量误差与完整求解成本\n\n| 案例 | 控制体/主未知量 | 速度L2 | 压力L2 | CPU完整秒 | CUDA完整秒 |\n|---|---:|---:|---:|---:|---:|\n'
for q in quant:
    text+=f"| {q['name']} | {q['cells']}/{q['primary_flow_unknowns']} | "+(f"{100*q['metrics']['velocity_volume_l2']:.6f}% | {100*q['metrics']['pressure_volume_l2']:.6f}%" if q['name'].startswith('curved-') else '— | —')+f" | {q['cpu_full_solve_s']:.3f} | {q['cuda_full_solve_s']:.3f} |\n"
text+='\n曲壁动量采用生产迎风对流离散，因此按实际结果报告加密阶数，不预设速度必须二阶；体力按连续方程指定，压力全由SIMPLE求解。完整时间各一次；重复窗口性能见前表。\n\n加密阶数：'+str(orders)+'\n'
for name in plots:text+=f'\n![{name}](figures/{name}.png)\n'
(directory/'report.md').write_text((directory/'report.md').read_text()+text)
with PdfPages(directory/'profiles.pdf') as pdf:
    for name in plots:
        fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(figs/(name+'.png')));ax.axis('off');pdf.savefig(fig);plt.close(fig)
with tempfile.TemporaryDirectory() as tmp:
    merged=Path(tmp)/'report.pdf';subprocess.run(['pdfunite',str(directory/'report.pdf'),str(directory/'profiles.pdf'),str(merged)],check=True);(directory/'report.pdf').write_bytes(merged.read_bytes())
tex=r'''\documentclass[11pt]{article}
\usepackage{amsmath,graphicx,booktabs,geometry,hyperref}
\geometry{margin=22mm}
\title{Verification of Full Collocated SIMPLE on Conforming Curved Meshes and CPU/CUDA Execution}
\author{TensorFVM benchmark development}
\begin{document}\maketitle
\begin{abstract}
A continuous manufactured incompressible Navier--Stokes problem tests full momentum, pressure correction and nonorthogonal geometry. A separate fully turbulent Spalart--Allmaras plate tests steady convergence and device equivalence. All failures and independent physical gates are retained.
\end{abstract}
\section{Problem}
Let $g(x)=0.2\sin^4(\pi x/2)$, $0\leq x\leq2$, $g(x)\leq y\leq g(x)+1$.
With $\eta=y-g(x)$, $w=6\eta(1-\eta)$, prescribe
$\mathbf{u}=(w,g'w)$ and $p=12\mu(2-x)$.
The forcing is $\mathbf{f}=\rho(\mathbf{u}\cdot\nabla)\mathbf{u}+\nabla p-\mu\Delta\mathbf{u}$.
Both velocity components and pressure are numerically solved; analytic fields are used only for forcing/boundaries and error evaluation. Refined starts interpolate computed coarse solutions.
\section{Method and acceptance}
Production collocated SIMPLE, Rhie--Chow face flux, upwind momentum convection and nonorthogonal diffusion use conforming polygons. Optional sparse GMRES has true residual checks and line/coarse pressure preconditioning. Host CSR assembly and transfer are explicit; Krylov iterations and preconditioning execute on actual CPU/CUDA. A coupled coarse Picard residual correction accelerates fine-grid SIMPLE. Interpolation and coarse updates use a constant-mobility auxiliary velocity projection while retaining physical pressure. The final physical Rhie--Chow flux defect must be below $10^{-7}$. Boundary faces are polygon chords sampled on the curve. SA has up to eight transport subiterations. Every physical error gate is strictly below 3\%; steady residual and device equivalence are separate. SA empirical mean skin friction is not an exact DNS reference.
\section{Timing}
CPU is single-threaded, float64; CUDA is RTX3090. Complete CPU/CUDA steady solves are performed once each. Three warmed synchronized five-step windows from the same computed intermediate state are reported separately and include assembly and transfer, excluding restore and plotting. Window ratios must not be presented as complete solve speedups.
\section{Results}
\begin{tabular}{lrrrr}\toprule Case & Cells & Velocity L2 (\%) & Pressure L2 (\%) & Window CPU/CUDA\\\midrule
'''
for q in quant:
    uv=f"{100*q['metrics']['velocity_volume_l2']:.6f}" if q['name'].startswith('curved-') else '--';pv=f"{100*q['metrics']['pressure_volume_l2']:.6f}" if q['name'].startswith('curved-') else '--';tex+=f"{q['name']} & {q['cells']} & {uv} & {pv} & {q['window_speed']:.3f} \\\\\n"
tex+='\\bottomrule\\end{tabular}\n\\section{Fields, curves and convergence}\n'
for name in plots:tex+='\\begin{figure}[p]\\centering\\includegraphics[width=.9\\linewidth]{figures/'+name+'.pdf}\\caption{'+name.replace('-',' ')+'}\\end{figure}\n'
tex+='\\section{Scope} This two-dimensional manufactured curved channel is not an unforced bend or an industrial SUBOFF validation. Steady SA, device consistency and empirical skin-friction accuracy remain separate statements. SIMPLEC, PISO and PIMPLE are not implemented in this delivery.\\end{document}\n';(directory/'report.tex').write_text(tex)
m=json.loads((directory/'manifest.json').read_text());m['additional_publication_source_sha256']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(directory);write_json(directory/'manifest.json',m)
print(quant);print('Observed orders',orders)
