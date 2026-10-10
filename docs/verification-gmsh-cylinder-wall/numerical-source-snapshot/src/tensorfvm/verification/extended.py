"""Run additional analytic benchmarks through shared gates, export and plotting."""
import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
import numpy as np
import torch
from .core import save_run, write_json, provenance, observed_orders, artifact_manifest, metric_value
from .extended_cases import mac_case, shear_lbm


def render(directory,row):
    from .report import figure,plt
    f=np.load(directory/'fields.npz',allow_pickle=False);out=directory/'figures';out.mkdir(exist_ok=True)
    for name,key,label in [('pressure','p_pa','Pressure (Pa)'),('u','u_m_s','u (m/s)'),('v','v_m_s','v (m/s)'),('w','w_m_s','w (m/s)'),('speed',None,'Speed (m/s)')]:
        fig,ax=plt.subplots(figsize=(6,5));value=f[key] if key else np.sqrt(f['u_m_s']**2+f['v_m_s']**2+f['w_m_s']**2)
        im=ax.pcolormesh(f['x_m'],f['y_m'],value,shading='nearest',cmap='coolwarm' if name!='speed' else 'viridis',rasterized=True)
        fig.colorbar(im,ax=ax,label=label);ax.set(xlabel='x (m)',ylabel='y (m)',title=f"{row['case']} {row['resolution']}: first z cell slice");ax.set_aspect('equal');figure(fig,out,name)
    for name,x,a,b,xlabel,ylabel in [('pressure-curve','pressure_x_m','pressure_curve_pa','reference_pressure_curve_pa','x (m)','Pressure (Pa)'),('velocity-profile','profile_y_m','profile_u_m_s','reference_u_profile_m_s','y (m)','u (m/s)')]:
        fig,ax=plt.subplots(figsize=(6,4));ax.plot(f[x],f[a],'o',ms=3,label='LBM' if row['role']=='comparison' else 'FVM');ax.plot(f[x],f[b],'-',label='Analytic reference');ax.set(xlabel=xlabel,ylabel=ylabel);ax.legend();ax.grid(alpha=.3);figure(fig,out,name)
        with (directory/(name+'.csv')).open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow([x,a,b]);w.writerows(zip(f[x],f[a],f[b]))
    hist=json.loads((directory/'history.json').read_text());fig,ax=plt.subplots(figsize=(6,4));key='kinetic_J' if row['role']=='comparison' else 'kinetic_after_J'
    ax.plot([h['time_s'] for h in hist],[h[key] for h in hist],label='Computed KE');ax.set(xlabel='Time (s)',ylabel='Kinetic energy (J; LBM per m depth)');ax.grid(alpha=.3);ax.legend();figure(fig,out,'history')


def publish(output,s):
    from .report import figure,plt
    from matplotlib.backends.backend_pdf import PdfPages
    figs=output/'figures';figs.mkdir(exist_ok=True)
    for kind,metric in [('abc','pressure_l2'),('advected-shear','shear_velocity_l2')]:
        rows=[r for r in s['runs'] if r['case']==kind and r['role']=='spatial'];fig,ax=plt.subplots(figsize=(6,4))
        for m in ['velocity_l2',metric]:ax.loglog([r['spacing_m'] for r in rows],[100*metric_value(r,m) for r in rows],'o-',label=m)
        ax.axhline(3,color='k',ls='--',label='3% limit');ax.set(xlabel='Grid spacing (m)',ylabel='Relative error (%)',title=kind+' spatial refinement');ax.legend();ax.grid(which='both',alpha=.3);figure(fig,figs,kind+'-spatial')
    table=['| Case | Grid | Role | Velocity L2 % | Pressure metric % | Passed |','|---|---|---|---:|---:|---|']
    for r in s['runs']:
        pm='pressure_l2' if r['case']=='abc' else 'pressure_dynamic_scaled_linf'
        table.append(f"| {r['case']} | {r['resolution']} | {r['role']} | {100*metric_value(r,'velocity_l2'):.6f} | {100*metric_value(r,pm):.6f} | {r['passed']} |")
    text='''# TensorFVM 新增解析 benchmark：三维 ABC 涡与对流剪切波

## 1. 问题介绍与解析依据

域为 [0,2π]³，周期边界，ρ=1 kg/m³，μ=0.1 Pa·s，双精度。没有固体、自由液面或外力。

**ABC / Beltrami 涡：** u₀=(sin z+cos y, sin x+cos z, sin y+cos x)，u=u₀ exp(−νt)，p=−ρ|u|²/2+常数。
解析依据可直接代入不可压 Navier–Stokes：div u₀=0，curl u₀=u₀，Δu₀=−u₀；
(u·∇)u=∇(|u|²/2)−u×curl u，因此非线性项由压力梯度抵消，剩余黏性指数衰减。
所有分量非零，场依赖 x/y/z，是真正三维周期流，但不是湍流统计、圆柱或工程绕流。
正式 n=24/32/48，Δt=0.005 s，t=0.1 s；n=16 保留为压力不达标负控制。

**对流剪切波：** u=sin(y−0.7t) exp(−νt)，v=0.7，w=0，p=0。
直接代入得 ∂t u+0.7∂y u=ν∂yy u，验证输运相位、衰减和非零平均动量。
正式 n=16/24/32，z=4 层，Δt=0.01 s，t=0.5 s；该案例是二维场嵌入三维网格。

## 2. 软件使用与离散方法

使用现有 PeriodicMACSolver：面动量共享中心通量、完整对称应力、隐式 midpoint/Picard、相容周期 FFT 压力投影。
速度参考取各分量真实面坐标和末时刻；ABC 压力取 cell 坐标、最后 midpoint 时刻 t−Δt/2，均值规范为零。
所有声明物理误差严格 <3%，另检查全部接受步的连续性、真实非线性残差、动量与能量账本。
三网格观察阶要求 1.8–2.2。每步原始状态保存并由独立 NumPy 算术重建。

复现（仓库根目录）：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.extended --output results/extended --lbm-repo ../TensorLBM
~~~

## 3. TensorLBM 对照与误差定义

剪切波对照使用外部 TensorLBM 原生产 D2Q9/BGK 平衡、碰撞、迁移函数：32²/64²，目标初始 Mach=0.05，整数步精确达到 t=0.5。
使用同一物理 ν、平均速度及周期长度。初始化为解析速度的平衡 populations，没有质量补偿。
LBM 保留原 float32 格式权重常数，populations 为 float64；初始化缺少非平衡应力可能贡献早期误差。
D2Q9 无三维能力，因此没有将 ABC 与二维 LBM 冒充同条件比较。

剪切波解析压力恒零，不能计算相对零参考误差；使用 max|p|/[0.5ρ(1²+0.7²)] 的绝对压力动态尺度误差，要求 <3%。
速度同时检查全部分量的 L2/L∞ 与单独剪切分量 L2，防止平均流掩盖输运误差。
LBM 速度取实际节点，FVM 速度取实际面，各自与解析值比较；压力由 EOS 的均值规范值导出。
单次 CPU 用时含设置、诊断和原场采样，不推论普遍速度优势；四层 FVM 与二维 LBM 的总内存也不能直接比较。

## 4. 计算结果

'''+ '\n'.join(table)+'\n\n观察阶：\n\n'+ '\n'.join(f'- {k}: '+', '.join(f'{v:.6f}' for v in vals) for k,vals in s['observed_orders'].items())+f"\n\nFVM 正式案例资格：{s['passed']}；LBM 对照精度：{s['comparison_passed']}。\n\n"
    shear=next(r for r in s['runs'] if r['case']=='advected-shear' and r['resolution'].startswith('32x'))
    baseline=next((r for r in s['runs'] if r['case']=='advected-shear-lbm' and r['resolution'].startswith('32x')),None)
    if baseline:
        text+=f"同为 n=32 的剪切分量 L2 误差：FVM {100*metric_value(shear,'shear_velocity_l2'):.6f}%，BGK {100*metric_value(baseline,'shear_velocity_l2'):.6f}%。这组实测值反映本工况下的相位与衰减误差，不能推广为所有流动的精度排名。\n\n"
    text+='''## 5. 云图、曲线与证据

'''
    selected=[r for r in s['runs'] if r['role']!='negative-control' and (r['resolution'].startswith('48x') or r['resolution'].startswith('32x'))]
    for r in selected:
        text+=f"### {r['case']}：{r['resolution']}\n\n"
        for name in ['pressure','speed','velocity-profile','pressure-curve']:text+=f"![{name}]({r['directory']}/figures/{name}.png)\n\n"
    text+='''## 6. 讨论、限制与复现材料

本轮扩展验证了三维周期非线性压力平衡以及非零平均流的输运，资格限定于这些参数与网格。
各目录包含完整三维/二维原场 NPZ、逐步 history、压力/速度曲线 CSV 和 300dpi PNG/PDF/SVG。
原始采样时间、面/节点布局、零压力归一化和负控制均明确保存。audit.json 由独立 NumPy 重建每个接受步。
manifest.json 绑定所有原场、报告及当前源码。旧 Poiseuille/Taylor–Green 报告与源文件保持其原始证据版本。
DFG 圆柱和翼型的旧 runner 不因本轮周期案例通过而获得认证；SUBOFF/破冰仍需真实固壁、移动边界与自由液面验证。

本文件与 report.pdf 是可复现研究稿，尚未完成期刊投稿或同行评审。
'''
    (output/'report.md').write_text(text)
    methods=[('Additional TensorFVM analytic benchmarks','Three-dimensional ABC / Beltrami flow and advected shear wave\n\n1. Problem and exact solution\nABC u=(sin z+cos y, sin x+cos z, sin y+cos x) exp(-nu t).\np=-rho |u|^2/2 + mean gauge. Curl u=u; Laplacian u=-u.\nThe nonlinear term is a gradient and balances pressure.\n\nShear: u=sin(y-0.7t) exp(-nu t), v=0.7, w=0, p=0.\nThis checks transport phase with nonzero mean momentum.\n\n2. Methods\nExisting production face-MAC midpoint solver, full symmetric stress,\nshared momentum flux and compatible FFT projection.\nVelocity at end time on real faces; pressure at last midpoint.\nABC: n=24/32/48, dt=0.005, end=0.1 s, true 3D.\nShear: n=16/24/32, four z layers, dt=0.01, end=0.5 s.\nPhysical errors <3%; all-step conservation/nonlinear gates.\n\n3. Reproducibility\nRaw accepted states, CSV curves, source hashes and independent NumPy\nreconstruction accompany each run. No production solver changes.'),
      ('Results and scope','\n'.join(f"{r['case']:20s} {r['resolution']:10s} u L2={100*metric_value(r,'velocity_l2'):.6f}%  pass={r['passed']}" for r in s['runs'])+'\n\nObserved orders:\n'+'\n'.join(f'{k}: {v}' for k,v in s['observed_orders'].items())+'\n\nABC n16 pressure failure retained; excluded from formal qualification.\nD2Q9 cannot compare with three-dimensional ABC.\nShear pressure error is absolute / initial dynamic pressure, since p=0.\nLBM production BGK uses float64 populations, upstream float32 weights.\nNo generalized speed, memory or industrial qualification claim.\nResearch manuscript; not peer-reviewed.')]
    with PdfPages(output/'report.pdf') as pdf:
        for title,body in methods:
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.95,title,fontsize=16,va='top')
            fig.text(.08,.89,body,fontsize=9,va='top',linespacing=1.65);pdf.savefig(fig);plt.close(fig)
        for r in selected:
            fig,axes=plt.subplots(2,2,figsize=(8.27,9));fig.suptitle(r['case']+' '+r['resolution'])
            for ax,name in zip(axes.flat,['pressure','speed','velocity-profile','pressure-curve']):ax.imshow(plt.imread(output/r['directory']/'figures'/(name+'.png')));ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    # Editable paper source uses the same saved numerical figures.
    tex=['\\documentclass{article}','\\usepackage[margin=20mm]{geometry}','\\usepackage{graphicx}','\\usepackage{booktabs}','\\begin{document}','\\title{Additional TensorFVM analytic benchmarks}\\maketitle',
      '\\section{Problems and methods}','ABC: $\\mathbf u=(\\sin z+\\cos y,\\sin x+\\cos z,\\sin y+\\cos x)e^{-\\nu t}$; $p=-\\rho|\\mathbf u|^2/2+C$. Shear: $u=\\sin(y-0.7t)e^{-\\nu t}$, $v=0.7$, $w=0$, $p=0$. Production MAC midpoint, full symmetric stress, shared conservative flux, compatible FFT projection. Velocity at final face coordinates; pressure at last midpoint cell coordinates. Strict physical error below 3 percent.', '\\section{Results}','\\begin{tabular}{lllrr}\\toprule Case & Grid & Role & Velocity error (percent) & Pass\\\\\\midrule']
    for r in s['runs']:tex.append(f"{r['case']} & {r['resolution']} & {r['role']} & {100*metric_value(r,'velocity_l2'):.6f} & {r['passed']} \\\\")
    tex+=['\\bottomrule\\end{tabular}','\\section{Discussion}','The n16 ABC pressure control fails and is preserved. Shear pressure error uses initial dynamic pressure scaling because its exact value is zero. D2Q9 is only compared with two-dimensional shear. Raw states, independent audit and source hashes are included. No cylinder, free-surface or icebreaking qualification is asserted.']
    for r in selected:
        tex.append('\\subsection{'+r['case']+' '+r['resolution']+'}')
        for name in ['pressure','speed','velocity-profile','pressure-curve']:tex.append('\\includegraphics[width=0.48\\linewidth]{'+r['directory']+'/figures/'+name+'.pdf}')
    (output/'report.tex').write_text('\n'.join(tex+['\\end{document}'])+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lbm-repo',type=Path);args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):parser.error('output must be empty; preserve prior evidence')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    summary=dict(schema='tensorfvm.extended-analytic-suite/1',error_limit=.03,provenance=provenance(),runs=[],passed=False,comparison_passed=None,engineering_qualified=False)
    plan=[(f'abc-{n}',lambda n=n:mac_case('abc',n)) for n in (24,32,48)]+[(f'shear-{n}',lambda n=n:mac_case('advected-shear',n)) for n in (16,24,32)]+[('control-abc-16',lambda:mac_case('abc',16,role='negative-control'))]
    if args.lbm_repo:plan +=[(f'lbm-shear-{n}',lambda n=n:shear_lbm(args.lbm_repo,n)) for n in (32,64)]
    for name,fn in plan:
        try:
            raw=fn();raw['directory']=name;row=save_run(args.output/name,raw);summary['runs'].append(row);write_json(args.output/'execution-status.json',dict(completed=name,runs=len(summary['runs'])));render(args.output/name,row);print(name,row['passed'],flush=True)
        except Exception as error:
            write_json(args.output/'execution-failure.json',dict(run=name,type=type(error).__name__,message=str(error)));raise
    summary['observed_orders']={kind+'-'+metric:observed_orders([r for r in summary['runs'] if r['case']==kind and r['role']=='spatial'],metric) for kind,metric in [('abc','velocity_l2'),('abc','pressure_l2'),('advected-shear','shear_velocity_l2')]}
    summary['passed']=all(r['passed'] for r in summary['runs'] if r['role']=='spatial') and all(not r['passed'] for r in summary['runs'] if r['role']=='negative-control') and all(1.8<v<2.2 for values in summary['observed_orders'].values() for v in values)
    comp=[r for r in summary['runs'] if r['role']=='comparison'];summary['comparison_passed']=all(r['passed'] for r in comp) if comp else None
    write_json(args.output/'summary.json',summary)
    from .extended_audit import audit
    write_json(args.output/'audit.json',audit(args.output));publish(args.output,summary)
    write_json(args.output/'manifest.json',dict(source_sha256=summary['provenance']['source_sha256'],artifacts_sha256=artifact_manifest(args.output)))
    print('Formal FVM:',summary['passed'],'LBM comparison:',summary['comparison_passed'],flush=True)
    return 0 if summary['passed'] and summary['comparison_passed'] is not False else 2


if __name__=='__main__':raise SystemExit(main())
