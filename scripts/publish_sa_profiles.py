"""Publish actual SA wall profiles, refinement differences and convergence curves."""
import argparse,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.verification.report import figure,plt
from tensorfvm.verification.core import sha,write_json,artifact_manifest
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;s=json.loads((d/'summary.json').read_text());figs=d/'figures';plots=[];fig,axes=plt.subplots(2,1,figsize=(8,7));rows=[]
for q in s['cases']:
 r=q['result'];c=r['config'];f=np.load(d/q['name']/c['device']/'fields.npz');wall=f['wall_face_mask'];o=f['face_owner'][wall];area=np.linalg.norm(f['face_area_vectors_m'][wall],axis=1);tau=f['wall_viscous_force_n_m'][:,0]/area;cf=tau/(.5*c['density']*c['inlet_velocity']**2);yp=np.sqrt(abs(tau)/c['density'])*f['cell_centers_m'].reshape(-1,2)[o,1]/(c['inlet_velocity']*c['length']/c['reynolds']);x=f['face_centers_m'][wall,0];axes[0].plot(x,cf,label=q['name']);axes[1].plot(x,yp,label=q['name']);np.savetxt(d/q['name']/'wall-profiles.csv',np.column_stack((x,cf,yp)),delimiter=',',header='x_m,local_cf,y_plus',comments='');rows.append(dict(name=q['name'],cells=c['nx']*c['ny'],primary_unknowns=4*c['nx']*c['ny'],mean_cf=q['mean_cf'],cf_empirical_error=q['cf_relative_error'],maximum_yplus=float(yp.max()),converged=r['converged']))
 hist=json.loads((d/q['name']/c['device']/'history.json').read_text());f2,ax=plt.subplots(figsize=(7,4))
 for key in ['momentum','continuity','turbulence','rhie_chow_flux_defect']:ax.semilogy(np.arange(1,len(hist)+1),[h[key] for h in hist],label=key)
 ax.axhline(c['tolerance'],ls='--',color='k',label='Momentum/SA gate');ax.axhline(1e-7,ls=':',color='k',label='Physical Rhie-Chow gate');ax.set(xlabel='Actual SIMPLE outer step',ylabel='Residual',title=q['name']+' '+c['device']+' convergence');ax.legend();ax.grid(alpha=.3);name=q['name']+'-residuals';figure(f2,figs,name);plots.append(name)
axes[0].set(xlabel='x (m)',ylabel='Local Cf');axes[1].set(xlabel='x (m)',ylabel='First-cell y+');axes[1].axhline(1,ls='--',color='k');axes[0].legend();axes[1].legend();fig.tight_layout();figure(fig,figs,'wall-profiles-refinement');plots.append('wall-profiles-refinement');diff=[dict(coarse=a['name'],fine=b['name'],relative_cf_change=abs(b['mean_cf']-a['mean_cf'])/abs(b['mean_cf'])) for a,b in zip(rows[:-1],rows[1:])];write_json(d/'refinement-metrics.json',dict(cases=rows,successive_grid_changes=diff,producer_sha256=sha(Path(__file__)),scope='changes in computed mean Cf are grid sensitivity, not independent physical accuracy'))
text='\n## 软件使用与原场审计\n\n```sh\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_sa_mesh_refinement.py\nPYTHONPATH=src python scripts/audit_steady_sa.py docs/verification-sa-refinement\nPYTHONPATH=src python scripts/audit_simple_acceptance.py docs/verification-sa-refinement\n```\n\n网格敏感性按相邻网格实际 Cf 差报告，与经验关联误差分开：\n\n| 相邻网格 | Cf相对变化 |\n|---|---:|\n'
for q in diff:text+=f"| {q['coarse']} → {q['fine']} | {100*q['relative_cf_change']:.6f}% |\n"
text+='\n这些计算使用原首层贴壁积分，没有壁面函数替代；不能证明完整壁面函数 RANS 已达标。两级CPU、一组CUDA，均使用实际设备；最细级没有CPU/CUDA一致性资格。求解单次时钟仅作诊断，计算期间有独立审计和绘图任务，不作为隔离的性能 benchmark。最大网格的变量数计入两个速度分量、压力、SA工作变量。\n'
for name in plots:text+=f'\n![{name}](figures/{name}.png)\n'
(d/'report.md').write_text((d/'report.md').read_text()+text)
with PdfPages(d/'profiles.pdf') as pdf:
 for name in plots:
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(figs/(name+'.png')));ax.axis('off');pdf.savefig(fig);plt.close(fig)
with tempfile.TemporaryDirectory() as td:
 target=Path(td)/'report.pdf';subprocess.run(['pdfunite',str(d/'report.pdf'),str(d/'profiles.pdf'),str(target)],check=True);(d/'report.pdf').write_bytes(target.read_bytes())
m=json.loads((d/'manifest.json').read_text());m['additional_publication_source_sha256']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(d);write_json(d/'manifest.json',m);print(rows,diff)
