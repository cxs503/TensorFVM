"""Strict common gates and independent wall-force audit of an actual SA run."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.verification.core import Metric,evaluate,write_json,sha,artifact_manifest
from tensorfvm.verification.report import plt,figure
from matplotlib.backends.backend_pdf import PdfPages


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('empty output required')
    shutil.copytree(a.source,a.output);old=json.loads((a.output/'benchmark.json').read_text());row=old['case'];grid=a.output/f"{row['nx']}x{row['ny']}";summary=json.loads((grid/'summary.json').read_text());c=summary['config'];nx,ny=c['nx'],c['ny'];fields=np.genfromtxt(grid/'fields.csv',delimiter=',',names=True);u=fields['u'].reshape(ny,nx);v=fields['v'].reshape(ny,nx);x=fields['x'].reshape(ny,nx);y=fields['y'].reshape(ny,nx);dx=c['length']/nx;mu=c['density']*c['inlet_velocity']*c['length']/c['reynolds'];first=v[0];dvdx=np.empty(nx);dvdx[1:-1]=(first[2:]-first[:-2])/(2*dx);dvdx[0]=(first[1]+first[0])/(2*dx);dvdx[-1]=(first[-1]-first[-2])/dx
    tau=mu*(u[0]/y[0]+dvdx);force=float(np.sum(tau)*dx);scale=.5*c['density']*c['inlet_velocity']**2*c['length'];cf=force/scale;reference=.074/c['reynolds']**.2
    if abs(cf-row['skin_friction_coefficient'])>1e-10:raise ValueError('independent flat-wall traction differs from actual producer')
    friction=np.sqrt(np.maximum(tau,0)/c['density']);local_yp=y[0]*friction/(mu/c['density']);last=summary['final_residuals'];metrics=[Metric('skin_friction_relative_error',abs(cf-reference)/reference,.03,'strict project gate versus declared fully turbulent empirical correlation; model/transition uncertainty remains'),Metric('solver_residual',max(last[k] for k in ('continuity','momentum','mass_imbalance','turbulence')),c['tolerance'],'all final production residuals including SA'),Metric('convergence_failure',0. if row['converged'] else 1.,.5,'explicit steady convergence required'),Metric('maximum_local_yplus',float(np.max(local_yp)),1.,'actual local first-cell y+; wall-resolved SA')]
    rows,passed=evaluate(metrics);record=dict(config=c,metrics=rows,passed=passed,original_legacy_error_limit=old['error_limit'],independent_cf=cf,reference_cf=reference,wall_force_n_m=force,pressure_force_contribution_x_n_m=0.,scope='actual 1800-iteration SA diagnostic; no current 3% or steady qualification',source_sha256={name:sha(Path(name)) for name in ['src/tensorfvm/benchmark_flat_plate.py','src/tensorfvm/solver.py','src/tensorfvm/body_fitted.py','src/tensorfvm/backend_registry.py','src/tensorfvm/mesh_api.py','scripts/publish_sa_diagnostic.py']})
    write_json(a.output/'strict-result.json',record);write_json(a.output/'independent-wall-audit.json',dict(raw_reconstruction_passed=True,physical_passed=passed,force_n_m=force,cf=cf,producer_cf=row['skin_friction_coefficient'],formula='mu*(u_first/y_first + d_x v_first), exact no-slip normal gradient; NumPy LS tangential gradient on exported rectangular topology',fields_sha256=sha(grid/'fields.csv'),history_sha256=sha(grid/'history.json')))
    turbulence=np.genfromtxt(grid/'turbulence.csv',delimiter=',',names=True);nu=turbulence['nu_t'].reshape(ny,nx);np.savez_compressed(a.output/'fields.npz',x_m=x,y_m=y,u_m_s=u,v_m_s=v,p_pa=fields['p'].reshape(ny,nx),nu_t_m2_s=nu,nu_tilde_m2_s=turbulence['nu_tilde'].reshape(ny,nx),wall_shear_pa=tau,local_first_cell_yplus=local_yp)
    figs=a.output/'figures';figs.mkdir()
    for name,value,label in [('speed',np.hypot(u,v),'Speed (m/s)'),('pressure',fields['p'].reshape(ny,nx),'Pressure (Pa)'),('eddy-viscosity',nu,'Eddy viscosity (m2/s)')]:
        fig,ax=plt.subplots(figsize=(9,3));m=ax.pcolormesh(x,y,value,shading='nearest',cmap='viridis');fig.colorbar(m,ax=ax,label=label);ax.set(xlabel='x (m)',ylabel='y (m)',title='Actual wall-resolved SA; unconverged diagnostic');figure(fig,figs,name)
    local_cf=tau/(.5*c['density']*c['inlet_velocity']**2);ref_local=.0592/(c['reynolds']*x[0]/c['length'])**.2;fig,ax=plt.subplots(figsize=(6,4));ax.plot(x[0],local_cf,label='Actual SA iterate');mask=x[0]>.1*c['length'];ax.plot(x[0,mask],ref_local[mask],'--',label='Fully turbulent correlation; not exact data');ax.set(xlabel='x (m)',ylabel='Local Cf');ax.legend();ax.grid(alpha=.3);figure(fig,figs,'skin-friction')
    np.savetxt(a.output/'wall-data.csv',np.column_stack((x[0],tau,local_cf,local_yp)),delimiter=',',header='x_m,wall_shear_pa,local_cf,local_yplus',comments='')
    text='# SA 高雷诺数贴壁平板：严格门诊断\n\n## 实际问题与结果\n\nRe_L=100000，64×56，拉伸贴壁网格，现有生产SA原模型和SIMPLE，首层贴壁积分，没有壁面函数。实际运行1800次迭代；原20%门保留为历史门，本报告重新按3%物理门验收。\n\n参考平均Cf=0.074 Re_L^(-1/5)为完全湍流经验关联式，不是逐点实验/DNS，低Re与转捩/模型差异会贡献误差。不能换参考或以宽门宣称3%通过。\n\n'
    for q in rows:text+=f"- {q['name']}: {q['value']:.9g}，门 {q['limit']:.9g}，通过 {q['passed']}\n"
    text+=f"\n独立NumPy壁面牵引积分Cf={cf:.9g}，参考{reference:.9g}；总体通过：{passed}。\n\n"
    text+='''## 原场、模型作用与限制

已保存u/v/p、nu_tilde/nu_t云图、局部Cf、每点y+与残差。nu_t非零说明模型实际参与方程，不能据此证明物理预测已准确。
独立壁面审计重建分子壁面剪切及切向梯度；压力对平直底壁的x向力为零。全1800次历史保留，审计不冒充独立重跑所有SA/SIMPLE步骤。
GPU报告另验证128×112网格的五次实际SA迭代及CPU/CUDA一致性；这也不代表稳态湍流已收敛。

下一步先解决原残差与迭代预算，再按[NASA TMR平板验证](https://tmbwg.github.io/turbmodels/flatplate_val.html)匹配入口、Re、边界和剖面数据。
SA贴壁与壁面函数方案需一致的输运边界后才能作同精度成本比较。
'''
    for name in ['speed','pressure','eddy-viscosity','skin-friction']:text+=f"\n![{name}](figures/{name}.png)\n"
    (a.output/'report.md').write_text(text)
    with PdfPages(a.output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.95,'Wall-resolved SA flat-plate diagnostic',fontsize=16,va='top');body='Re_L=100000, production SA/SIMPLE, 64x56 stretched wall-resolved grid.\nActual 1800 iterations; no equilibrium wall function in SA transport.\nOriginal 20% regression limit is not treated as 3% qualification.\n\n'
        body+='\n'.join(f"{q['name']}: {q['value']:.9g}; limit={q['limit']:.9g}; pass={q['passed']}" for q in rows)
        body+=f'\n\nIndependent wall force Cf={cf:.9g}; reference={reference:.9g}.\nPressure has zero x-force on a flat wall.\n\nThe fully turbulent empirical correlation is not exact DNS/experiment.\nCurrent SA and momentum residuals fail the steady gate.\nThe 128x112 CPU/CUDA comparison covers five production iterations only.\nNext: residual convergence, matched public data, grid refinement,\nthen consistent wall-function turbulence boundaries.';fig.text(.08,.89,body,fontsize=9,va='top',linespacing=1.6);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(8.27,8))
        for ax,name in zip(axes.flat,['speed','pressure','eddy-viscosity','skin-friction']):ax.imshow(plt.imread(figs/(name+'.png')));ax.axis('off')
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    write_json(a.output/'manifest.json',dict(source_sha256=record['source_sha256'],artifacts_sha256=artifact_manifest(a.output)));print('SA strict physical qualification',passed,'CF error',abs(cf-reference)/reference)


if __name__=='__main__':main()
