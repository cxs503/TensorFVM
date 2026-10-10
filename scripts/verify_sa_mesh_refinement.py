"""Case-first SA grid study; empirical accuracy and solver convergence are separate."""
import json,time,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from tensorfvm.verification.simple_gpu import config,initialize,state,exported
from tensorfvm.sparse_simple import SparseBodyFittedSolver
from tensorfvm.verification.core import provenance,sha,save_run,write_json,artifact_manifest
from tensorfvm.verification.report import plt,figure
from matplotlib.backends.backend_pdf import PdfPages

root=Path(__file__).resolve().parents[1];out=root/'docs/verification-sa-refinement';out.mkdir(exist_ok=False);torch.set_num_threads(1);prov=provenance()
for name in ['src/tensorfvm/sparse_simple.py','src/tensorfvm/coupled_coarse.py','src/tensorfvm/curved_channel.py','scripts/verify_sa_mesh_refinement.py']:prov['source_sha256'][name]=sha(root/name)
base=root/'docs/verification-simple-gpu/sa-64x56/cpu';rows=[dict(name='sa-64x56',directory=str(base.relative_to(root)),result=json.loads((base/'result.json').read_text()),fields_sha256=sha(base/'fields.npz'))];shutil.copytree(base,out/'sa-64x56'/'cpu');rows[0]['directory']=str((out/'sa-64x56'/'cpu').relative_to(root));f=np.load(base/'fields.npz');seed=dict(shape=(56,64),velocity=f['velocity_m_s'],pressure=f['pressure_pa'],nu_tilde=f['nu_tilde_m2_s']);figs=out/'figures';figs.mkdir();plots=[]
base128=root/'docs/verification-sa-refinement-cpu-partial/sa-128x112/cpu';shutil.copytree(base128,out/'sa-128x112'/'cpu');r128=json.loads((base128/'result.json').read_text());rows.append(dict(name='sa-128x112',directory=str((out/'sa-128x112'/'cpu').relative_to(root)),result=r128,fields_sha256=sha(base128/'fields.npz'),reused_actual_source='docs/verification-sa-refinement-cpu-partial'));f128=np.load(base128/'fields.npz');seed=dict(shape=(112,128),velocity=f128['velocity_m_s'],pressure=f128['pressure_pa'],nu_tilde=f128['nu_tilde_m2_s'])
for nx,ny in [(256,224)]:
 s=SparseBodyFittedSolver(config('sa',nx,ny,'cuda',2000));start=time.perf_counter();initialize(s,'sa',seed);s.used_coarse_seed=True;s.coupled_coarse_shape=(32,28)
 cy,cx=seed['shape'];yy,xx=np.meshgrid((np.arange(ny)+.5)/ny,(np.arange(nx)+.5)/nx,indexing='ij');points=np.stack((yy,xx),-1).reshape(-1,2);values=RegularGridInterpolator(((np.arange(cy)+.5)/cy,(np.arange(cx)+.5)/cx),seed['nu_tilde'].reshape(cy,cx),bounds_error=False,fill_value=None)(points);s.nu_tilde.copy_(torch.as_tensor(np.maximum(values,0),device=s.p.device));s.turbulent_kinematic_viscosity=s._sa_eddy_viscosity(s.nu_tilde)
 for i in range(2000):
  s.step()
  if (i+1)%50==0:print(nx,ny,i+1,s.history[-1],flush=True)
  if s.converged:break
 name=f'sa-{nx}x{ny}';torch.cuda.synchronize();raw=exported(s,'sa');raw['full_solve_elapsed_s']=time.perf_counter()-start;row=save_run(out/name/'cuda',raw);rows.append(dict(name=name,directory=str((out/name/'cuda').relative_to(root)),result=row,fields_sha256=sha(out/name/'cuda'/'fields.npz')));seed=dict(shape=(ny,nx),**state(s));write_json(out/'execution-status.json',dict(provenance=prov,cases=rows))
 if not s.converged:break
text='# SA 平板：优先物理验证的网格加密\n\nRe=100000、L=1 m、H=0.2 m，原生产 SA 与首层贴壁条件不变；平均 Cf=0.074 Re^(-1/5) 为经验参考，不能代替匹配实验或 NASA TMR 条件。实际粗场初始化，包含ν̃插值；每级完整求解，未用解析流场。稳态残差、真实 Rhie–Chow 通量固定点和经验摩阻3%分别验收。64×56和128×112为实际CPU结果，256×224为实际CUDA结果，最细网格不声称两端一致性。较细CPU部分运行与原源码保留；采用32×28粗耦合网格加速，不改变原物理参数或方程。GPU一致性独立见64×56报告。\n\n| 网格 | 收敛 | 步数 | Cf | Cf误差 |\n|---|---|---:|---:|---:|\n';table=[]
for q in rows:
 r=q['result'];fields=np.load(root/q['directory']/'fields.npz');cf=float(fields['computed_mean_cf']);error=next(m['value'] for m in r['metrics'] if m['name']=='skin_friction_relative_error');text+=f"| {q['name']} | {r['converged']} | {r['iterations']} | {cf:.8f} | {100*error:.4f}% |\n";table.append([q['name'],str(r['converged']),str(r['iterations']),f'{cf:.8f}',f'{100*error:.4f}%']);q['mean_cf']=cf;q['cf_relative_error']=error
 for label,values in [('pressure',fields['pressure_pa']),('speed',np.linalg.norm(fields['velocity_m_s'],axis=1)),('eddy-viscosity',fields['eddy_viscosity_m2_s'])]:
  c=r['config'];fig,ax=plt.subplots(figsize=(9,3));centers=fields['cell_centers_m'];im=ax.pcolormesh(fields['vertices_m'][:,:,0],fields['vertices_m'][:,:,1],values.reshape(c['ny'],c['nx']),shading='flat');ax.set(xlabel='x (m)',ylabel='y (m)',title=q['name']+' '+label);fig.colorbar(im,ax=ax);name=q['name']+'-'+label;figure(fig,figs,name);plots.append(name);pass
fig,ax=plt.subplots(figsize=(7,4));ax.plot([q['result']['config']['nx'] for q in rows],[q['mean_cf'] for q in rows],'o-',label='Actual steady SA');ax.axhline(.0074,ls='--',label='Fully turbulent empirical reference');ax.set(xlabel='Streamwise cells',ylabel='Mean Cf');ax.legend();ax.grid(alpha=.3);figure(fig,figs,'mean-cf-refinement');plots.append('mean-cf-refinement');text+=''.join(f'\n![{name}](figures/{name}.png)\n' for name in plots);text+='\n![Cf](figures/mean-cf-refinement.png)\n\n经验误差未通过的网格不计入正式合格 benchmark；加密不能自动证明参考工况匹配。首层 y+、局部 Cf 与独立非线性残差另见审计。\n';(out/'report.md').write_text(text)
with PdfPages(out/'report.pdf') as pdf:
 fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off');ax.set_title('SA mesh refinement, with actual CPU/CUDA devices: convergence and empirical accuracy');ax.table(cellText=table,colLabels=['Grid','Converged','Steps','Mean Cf','Empirical error'],loc='upper center');ax.text(0,.4,'Original SA model, Re=100000; actual coarse fields seed refined solves.\nEmpirical mean Cf is not a matched NASA or experimental validation.\nTwo CPU levels and one CUDA level; no finest-grid equivalence claim.\nAll physics and solver gates are reported independently.',fontsize=11);pdf.savefig(fig);plt.close(fig)
 for name in plots:
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(figs/(name+'.png')));ax.axis('off');pdf.savefig(fig);plt.close(fig)

for q in rows:q['devices']={q['result']['config']['device']:q['result']}
summary=dict(provenance=prov,cases=rows,all_converged=all(q['result']['converged'] for q in rows),scope='Case-first SA study: two CPU levels and one CUDA level; original model and empirical reference unchanged; no fine-grid GPU speed claim');write_json(out/'summary.json',summary);write_json(out/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(out)));print([(q['name'],q['mean_cf'],q['cf_relative_error']) for q in rows])
