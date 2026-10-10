"""Large conforming-wall annular benchmark and independently checked wall law."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .core import Metric,evaluate,save_run,write_json,provenance,sha,artifact_manifest,metric_value
from .engineering_audit import annular_metrics,audit
from ..annular import AnnularConfig,solve,analytic_velocity,analytic_flow
from ..wall_functions import wall_traction,u_plus_at_y_plus


def case(ntheta,nr):
    c=AnnularConfig(ntheta=ntheta,nr=nr);raw=solve(c);computed=annular_metrics(raw['fields'],raw['config'])
    metrics=[Metric(k,float(v),2e-10 if k=='true_linear_residual' else 1e-9 if k=='momentum_force_balance' else .03,'independent full-face reconstruction; continuous circular annulus reference') for k,v in computed.items()]
    rows,passed=evaluate(metrics)
    raw.update(case='annular-poiseuille',role='spatial',resolution=f'{ntheta}x{nr}',spacing_m=(c.outer_radius_m-c.inner_radius_m)/nr,metrics=rows,passed=passed,reference=dict(pressure_gradient_pa_m=c.pressure_gradient_pa_m,axial_length_m=1.,pressure_policy='imposed p(z)=G(1-z); no transverse pressure Poisson solve',analytic_flow_m3_s=analytic_flow(c),velocity_formula='G/(4 mu) [Ro^2-r^2-(Ro^2-Ri^2) ln(Ro/r)/ln(Ro/Ri)]',scope='exact fully developed laminar axial reduction; not general curved 3D flow'))
    return raw


def wall_case():
    yp=np.logspace(-2,5,141);nu=1e-5;rho=1000.;ut=.05;up=u_plus_at_y_plus(yp);distance=yp*nu/ut;v=np.column_stack((up*ut,np.zeros_like(yp)))
    t=wall_traction(v,distance,rho,rho*nu);expected=rho*ut*ut;err=float(np.max(abs(np.linalg.norm(t['traction_pa'],axis=1)-expected)/expected))
    return dict(case='spalding-wall-law',role='constitutive-verification',maximum_traction_relative_error=err,passed=err<1e-8,physical_validation=False,history=[],fields=dict(y_plus=yp,u_plus_reference=up,reference_friction_velocity_m_s=np.array(ut),density_kg_m3=np.array(rho),kinematic_viscosity_m2_s=np.array(nu),distance_m=distance,relative_tangential_velocity_m_s=v,computed_traction_pa=t['traction_pa'],computed_friction_velocity_m_s=t['friction_velocity_m_s']))


def staircase_geometry(n,ri=.05,ro=.1):
    h=2*ro/n;q=(np.arange(n)+.5)*h-ro;y,x=np.meshgrid(q,q,indexing='ij');m=(x*x+y*y>ri*ri)&(x*x+y*y<ro*ro)
    area=m.sum()*h*h;edges=np.count_nonzero(np.diff(np.pad(m,1).astype(int),axis=0))+np.count_nonzero(np.diff(np.pad(m,1).astype(int),axis=1));perimeter=edges*h
    return dict(cartesian_n=n,total_background_cells=n*n,fluid_cells=int(m.sum()),area_relative_error=abs(area-math.pi*(ro*ro-ri*ri))/(math.pi*(ro*ro-ri*ri)),staircase_perimeter_relative_error=abs(perimeter-2*math.pi*(ro+ri))/(2*math.pi*(ro+ri)),scope='geometry only; not an LBM or Cartesian FV flow computation; curved/interpolated LBM boundary methods are not represented')


def plot_case(directory,row):
    from .report import figure,plt
    from matplotlib.collections import PolyCollection
    f=np.load(directory/'fields.npz');out=directory/'figures';out.mkdir();v=f['vertices_m'];cells=np.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),axis=2).reshape(-1,4,2);w=f['axial_velocity_m_s'].ravel();ref=f['reference_axial_velocity_m_s'].ravel()
    for name,value,label in [('axial-velocity',w,'Axial speed (m/s)'),('velocity-error',w-ref,'Computed - analytic (m/s)')]:
        fig,ax=plt.subplots(figsize=(6,5));p=PolyCollection(cells,array=value,cmap='viridis' if name=='axial-velocity' else 'coolwarm',edgecolors='none',rasterized=True);ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title=f"Conforming annulus: {row['diagnostics']['full_matrix_unknowns']:,} cells");fig.colorbar(p,ax=ax,label=label);figure(fig,out,name)
    radius=np.linalg.norm(f['cell_centers_m'][:,0],axis=1);vel=f['axial_velocity_m_s'][:,0];exact=f['reference_axial_velocity_m_s'][:,0]
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(radius,vel,'o',ms=2,label='Full-matrix FVM');ax.plot(radius,exact,'-',label='Analytic');ax.set(xlabel='Radius (m)',ylabel='Axial speed (m/s)');ax.legend();ax.grid(alpha=.3);figure(fig,out,'velocity-profile')
    np.savetxt(directory/'velocity-profile.csv',np.column_stack((radius,vel,exact)),delimiter=',',header='radius_m,numerical_w_m_s,analytic_w_m_s',comments='')
    z=np.linspace(0,1,101);p=row['config']['pressure_gradient_pa_m']*(1-z);fig,ax=plt.subplots(figsize=(6,3));ax.plot(z,p);ax.set(xlabel='Axial coordinate z (m)',ylabel='Imposed pressure (Pa)',title='Prescribed driving pressure; not a pressure solver result');ax.grid(alpha=.3);figure(fig,out,'imposed-pressure')
    np.savetxt(directory/'imposed-pressure.csv',np.column_stack((z,p)),delimiter=',',header='z_m,imposed_p_pa',comments='')
    faces=f['wall_face_mask'];length=f['face_lengths_m'][faces];traction=f['axial_momentum_flux_n_m'][faces]/length;centers=f['face_centers_m'][faces];angles=np.mod(np.arctan2(centers[:,1],centers[:,0]),2*np.pi);inner=f['inner_wall_face_mask'][faces];fig,ax=plt.subplots(figsize=(6,4))
    for mask,label in [(inner,'Inner wall'),(~inner,'Outer wall')]:order=np.argsort(angles[mask]);ax.plot(angles[mask][order],traction[mask][order],label=label)
    ax.set(xlabel='Angle (rad)',ylabel='Resisting axial traction (Pa)');ax.legend();ax.grid(alpha=.3);figure(fig,out,'wall-traction')
    np.savetxt(directory/'wall-traction.csv',np.column_stack((angles,inner.astype(int),traction)),delimiter=',',header='theta_rad,is_inner,numerical_resisting_traction_pa',comments='')


def publish(directory,s):
    from .report import figure,plt
    from matplotlib.backends.backend_pdf import PdfPages
    figs=directory/'figures';figs.mkdir();rows=s['annular_runs'];fig,ax=plt.subplots(figsize=(6,4))
    for m in ('velocity_volume_l2','flow_rate_relative_error','inner_wall_force_relative_error'):ax.loglog([r['spacing_m'] for r in rows],[100*metric_value(r,m) for r in rows],'o-',label=m)
    ax.set(xlabel='Radial cell spacing (m)',ylabel='Relative error (%)',title='Conforming-wall spatial refinement');ax.legend();ax.grid(which='both',alpha=.3);figure(fig,figs,'spatial-refinement')
    f=np.load(directory/'wall-law/fields.npz');fig,ax=plt.subplots(figsize=(6,4));ax.semilogx(f['y_plus'],f['u_plus_reference'],label='Spalding, kappa=0.41 E=9.8');ax.semilogx(f['y_plus'],f['y_plus'],'--',label='Viscous asymptote u+=y+');mask=f['y_plus']>=30;ax.semilogx(f['y_plus'][mask],np.log(9.8*f['y_plus'][mask])/.41,':',label='Log asymptote');ax.set(xlabel='y+',ylabel='u+',ylim=(0,40),title='Equilibrium wall constitutive law');ax.legend();ax.grid(alpha=.3);figure(fig,figs,'wall-law')
    np.savetxt(directory/'wall-law/profile.csv',np.column_stack((f['y_plus'],f['u_plus_reference'],f['computed_friction_velocity_m_s'])),delimiter=',',header='y_plus,reference_u_plus,recovered_u_tau_m_s',comments='')
    table=['| 网格 | 单元/未知数 | 速度L2误差 | 流量误差 | 内壁力误差 | 真实线性残差 | 通过 |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:table.append(f"| {r['resolution']} | {r['diagnostics']['full_matrix_unknowns']:,} | {100*metric_value(r,'velocity_volume_l2'):.6f}% | {100*metric_value(r,'flow_rate_relative_error'):.6f}% | {100*metric_value(r,'inner_wall_force_relative_error'):.6f}% | {metric_value(r,'true_linear_residual'):.3e} | {r['passed']} |")
    text='''# 大规模贴体环形管与壁面函数基础验证

## 1. 问题、方程与适用范围

同心环形管内的稳态、不可压、充分发展轴向层流。Ri=0.05 m，Ro=0.1 m，μ=0.001 Pa·s，ρ=1 kg/m³，G=−dp/dz=0.01 Pa/m，内外壁无滑移。
由于 w=w(x,y)、横向速度为零，完整三维 NS 精确约化为 −div(μ grad w)=G；不是一般三维入口、弯管、涡脱落或压力投影验证。
解析解 w=G/(4μ)[Ro²−r²−K ln(Ro/r)]，K=(Ro²−Ri²)/ln(Ro/Ri)。可直接对径向 Laplace 算子微分验证，边界为零。
Q=πG/(8μ)[Ro⁴−Ri⁴−(Ro²−Ri²)²/ln(Ro/Ri)]。
内/外壁的阻力（单位轴长）分别是 πG(K−2Ri²)/2 和 πG(2Ro²−K)/2，总和为Gπ(Ro²−Ri²)。

## 2. 软件使用、网格与真实求解

注册新的 AnnularMesh 到现有 Mesh2D/backend 契约，内外圆以贴体多边形表示，周向接缝完全共享面。
标量轴向黏性算子实际调用生产 BodyFittedSolver._momentum/_gradient/_matvec，壁面零速度；没有执行其二维横向对流/压力修正。
全部角向和径向未知数显式组装成完整 CSR 矩阵，SciPy CG/Jacobi 求解，零初值；没有解析初始化、径向平均或事后修正。
解析值只用于误差比较。压力是外加驱动 p(z)=G(1−z)，压力图明确标注为规定值，不冒充数值压力解。

~~~bash
uv pip install --python /path/to/python scipy matplotlib
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.engineering --output results/engineering
~~~

## 3. 误差验收与计算结果

物理指标分别严格 <3%：体积加权速度L2、速度L∞、流量、内外壁力和圆环面积；另要求全局动量力平衡 <1e−9、真实离散残差 <2e−10。
从每个面的几何、实际速度和系数独立重建通量，检查每个CV的几何闭合、整个矩阵未知数数量、圆壁力与流量。
三个正式网格为 256×64、512×128、1024×256，16,384 / 65,536 / 262,144 个单元。

'''+ '\n'.join(table)+f"\n\n速度空间观察阶：{s['velocity_observed_orders']}；正式资格：{s['annular_qualified']}。\n\n"
    text+='''## 4. 贴体几何优势与比较边界

报告另计算相同圆环的笛卡尔阶梯 mask 面积和边界长度。
这是几何比较，没有运行对应 LBM/FVM 流场，也不代表采用曲面插值或 immersed-boundary 的 LBM 能力。
贴体网格直接输出物理壁面法向、长度、剪切与积分力，避免将阶梯面长当圆周长。
不能由该几何测试推论流体误差、GPU速度、峰值内存或一般工业优势。

| 笛卡尔背景网格 | 流体格点数 | 面积误差 | 阶梯周长误差 |
|---|---:|---:|---:|
'''
    for r in s['staircase_geometry']:text+=f"| {r['cartesian_n']}² | {r['fluid_cells']} | {100*r['area_relative_error']:.6f}% | {100*r['staircase_perimeter_relative_error']:.6f}% |\n"
    text+='''
## 5. Spalding 壁面函数

实现 y+=u++[exp(κu+)−1−κu+−(κu+)²/2−(κu+)³/6]/E，κ=0.41、E=9.8。
给定壁面相对切向速度、距离和SI黏度，反解uτ并返回流体侧阻力τ=−ρuτ² Urel/|Urel|。
包含零速、黏性区、小参数的数值稳定处理、缓冲区、对数区及负功检查。
本轮以141个y+样本作构成关系反解验证，独立SciPy标量root重建；不是141个CFD算例。
这验证了可复用壁面本构模块，尚未接入原有SA输运边界；不能把它称作SA壁面函数计算已经通过。
数学测试延伸到大y+只验证反解，不证明外层/分离流中壁面模型有效。

参考公式：[OpenFOAM 原生产 Spalding 壁面函数](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/MomentumTransportModels/momentumTransportModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.C)。
湍流验证资料：[NASA TMR 迁移说明](https://www.nasa.gov/nasa-turbulence-modeling-resource/)、[当前 SA 模型定义](https://tmbwg.github.io/turbmodels/spalart.html)、[平板验证](https://tmbwg.github.io/turbmodels/flatplate_val.html)。

## 6. 由易到难的下一阶段

1. 本轮：大规模曲壁黏性问题与壁面本构模块；保留适用范围。
2. SA贴壁平板：公开匹配工况和剖面/Cf数据，首层y+、残差与三网格共同验收；原20%门不升级成3%资格。
3. 真正壁面函数RANS：SA/其他闭合的壁面边界需一致耦合，比较y+约1与30–100、同误差计算成本。
4. 翼型压力梯度/分离及曲壁湍流：再做NACA/后向台阶/弯管与实验数据比较，最后才是SUBOFF工程流动。

本发布包含完整原场、全矩阵实际迭代数、误差表、曲线CSV、300dpi PNG/PDF/SVG、源码哈希与独立审计。研究稿尚未同行评审。
'''
    for r in [rows[0],rows[-1]]:
        text+=f"\n## {r['resolution']} 原场图版\n\n"
        for name in ['axial-velocity','velocity-error','velocity-profile','wall-traction','imposed-pressure']:text+=f"![{name}]({r['directory']}/figures/{name}.png)\n\n"
    text+='\n![wall law](figures/wall-law.png)\n';(directory/'report.md').write_text(text)
    columns=['Grid','Unknowns','Velocity L2 %','Flow error %','Inner-wall force %','Passed'];cells=[[r['resolution'],f"{r['diagnostics']['full_matrix_unknowns']:,}",f"{100*metric_value(r,'velocity_volume_l2'):.6f}",f"{100*metric_value(r,'flow_rate_relative_error'):.6f}",f"{100*metric_value(r,'inner_wall_force_relative_error'):.6f}",str(r['passed'])] for r in rows]
    with PdfPages(directory/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.95,'Large conforming-wall annular benchmark',fontsize=16,va='top');body='1. Problem\nFully developed incompressible axial annular flow, Ri=0.05 m, Ro=0.1 m,\nmu=0.001 Pa s, rho=1, imposed -dp/dz=0.01 Pa/m. No-slip circular walls.\nExact 3D reduction: -div(mu grad w)=G on a 2D cross-section.\nNo general 3D flow or numerical pressure solution is asserted.\n\n2. Numerical method\nRegistered conforming polar Mesh2D; shared periodic angular faces.\nActual production fitted diffusion/gradient/matvec kernel.\nFull 16,384 / 65,536 / 262,144 unknown CSR matrices, CG/Jacobi, zero initial state.\nNo ring averaging, analytic initialization or numerical force correction.\nIndependent NumPy face geometry/coefficient/flux, flow and wall-force audit.\nSeparate geometry-only staircase comparison does not certify LBM flow.\n\n3. Results\n'
        for r in rows:body+=f"{r['resolution']} ({r['diagnostics']['full_matrix_unknowns']:,} cells):\nvelocity L2 {100*metric_value(r,'velocity_volume_l2'):.6f}%, flow {100*metric_value(r,'flow_rate_relative_error'):.6f}%, pass={r['passed']}\n"
        body+='\nObserved velocity orders: '+', '.join(f'{v:.6f}' for v in s['velocity_observed_orders'])+'\n\n4. Wall-law scope and next steps\nSpalding dimensional traction module independently verified; not yet\ncoupled to wall-resolved SA. Constitutive inversion is not turbulence\nphysical validation. Next: matching public flat-plate data and RANS wall coupling.\nRaw states, references and source SHA included; research manuscript.';fig.text(.08,.89,body,fontsize=9,va='top',linespacing=1.55);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off');ax.set_title('Quantitative full-matrix verification',pad=20);tab=ax.table(cellText=cells,colLabels=columns,loc='upper center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2);ax.text(0,.65,'All physical gates are strict <3%. Each wall force is checked independently.\nPressure is prescribed driving data, not a computed pressure field.\nVelocity is sampled at polygon-cell centers and volume-weighted.\nLinear residual and global force balance are separate numerical gates.\nSpalding law is a constitutive module, not a complete wall-function RANS run.',fontsize=11,linespacing=1.7);pdf.savefig(fig);plt.close(fig)
        for r in [rows[0],rows[-1]]:
            fig,axes=plt.subplots(2,2,figsize=(8.27,9));fig.suptitle(r['resolution']+' full conforming cross-section')
            for ax,name in zip(axes.flat,['axial-velocity','velocity-error','velocity-profile','wall-traction']):ax.imshow(plt.imread(directory/r['directory']/'figures'/(name+'.png')));ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):p.error('empty output required')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);prov=provenance();root=Path(__file__).resolve().parents[3]
    for name in ['src/tensorfvm/annular.py','src/tensorfvm/wall_functions.py']:prov['source_sha256'][name]=sha(root/name)
    import scipy
    prov['scipy']=scipy.__version__;s=dict(provenance=prov,annular_runs=[],annular_qualified=False,wall_law={},staircase_geometry=[staircase_geometry(n) for n in (128,256,512)])
    for n,nr in [(256,64),(512,128),(1024,256)]:
        name=f'annular-{n}x{nr}';raw=case(n,nr);raw['directory']=name;row=save_run(args.output/name,raw);s['annular_runs'].append(row);plot_case(args.output/name,row);print(name,row['passed'],row['diagnostics']['full_matrix_unknowns'],flush=True)
    wall=wall_case();wall['directory']='wall-law';s['wall_law']=save_run(args.output/'wall-law',wall)
    vals=[metric_value(r,'velocity_volume_l2') for r in s['annular_runs']];s['velocity_observed_orders']=[math.log(a/b)/math.log(2) for a,b in zip(vals,vals[1:])];s['annular_qualified']=all(r['passed'] for r in s['annular_runs']) and all(1.8<v<2.2 for v in s['velocity_observed_orders'])
    write_json(args.output/'summary.json',s);write_json(args.output/'audit.json',audit(args.output));publish(args.output,s);write_json(args.output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(args.output)));print('annular qualification',s['annular_qualified'],'wall constitutive inversion',s['wall_law']['passed'],flush=True);return 0 if s['annular_qualified'] and s['wall_law']['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
