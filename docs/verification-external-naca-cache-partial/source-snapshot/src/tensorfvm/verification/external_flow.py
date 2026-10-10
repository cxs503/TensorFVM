"""Shared cylinder/airfoil refinement evidence; failed physical gates stay failed."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from .core import Metric, evaluate, save_run, write_json, provenance, sha, artifact_manifest
from .cases import array
from .dfg import REFERENCE as DFG_REFERENCE, REFERENCE_URL
from ..benchmark_airfoil import REFERENCE as NACA_REFERENCE
from ..solver import SolverConfig
from ..external_linear import ExternalFlowSolver
from ..external_cached import CachedExternalFlowSolver
from ..external_coupled import CoupledExternalFlowSolver


def configuration(case, nx, ny, device, iterations):
    common=dict(nx=nx,ny=ny,device=device,max_iterations=iterations,tolerance=1e-6)
    if case=='cylinder':
        return SolverConfig(**common,mesh_type='body-fitted',inlet_profile='parabolic',length=2.2,height=.41,cylinder_x=.2,cylinder_y=.2,cylinder_radius=.05,inlet_velocity=.2,density=1,reynolds=20)
    return SolverConfig(**common,mesh_type='c-grid',airfoil_code='0012',reynolds=1000,angle_of_attack=4,length=20,height=16,airfoil_x=6,airfoil_y=8,airfoil_chord=1,velocity_relaxation=.5,pressure_relaxation=.3,pseudo_time_step=.2)


def run(case,nx,ny,device,iterations,solver="simple"):
    c=configuration(case,nx,ny,device,iterations);s=({"coupled":CoupledExternalFlowSolver,"cached":CachedExternalFlowSolver}.get(solver,ExternalFlowSolver))(c);start=time.perf_counter()
    for i in range(iterations):
        s.step()
        if (i+1)%100==0:print(case,nx,ny,i+1,s.history[-1]['momentum'],flush=True)
        if s.converged:break
    r=s.solve();mask=s.mesh.masks['cylinder' if case=='cylinder' else 'airfoil'];owners=s.o[mask];d=s.d[mask];grad=s._gradient(s.velocity)[owners].clone()
    defect=-s.velocity[owners]-torch.einsum('fi,fij->fj',d,grad)
    grad+=d[:,:,None]*defect[:,None,:]/d.square().sum(-1)[:,None,None]
    pressure=s.p[owners,None]*s.S[mask];viscous=-torch.einsum('fij,fj->fi',c.viscosity*(grad+grad.transpose(-1,-2)),s.S[mask])
    force=array((pressure+viscous).sum(0));scale=.5*c.density*c.inlet_velocity**2*c.reference_length
    a=np.deg2rad(c.angle_of_attack);rotation=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    coeff=rotation@force/scale;computed=dict(drag=float(coeff[0]),lift=float(coeff[1]))
    wallp=array(s.p[owners]+(s._gradient(s.p,pressure=True)[owners]*d).sum(-1));centers=array(s.mesh.face_centers[mask])
    if case=='cylinder':
        theta=np.mod(np.arctan2(centers[:,1]-c.cylinder_y,centers[:,0]-c.cylinder_x),2*np.pi);order=np.argsort(theta);t=theta[order];wp=wallp[order]
        te=np.r_[t[-1]-2*np.pi,t,t[0]+2*np.pi];pe=np.r_[wp[-1],wp,wp[0]]
        computed['pressure_difference_pa']=float(np.interp(np.pi,te,pe)-np.interp(0,te,pe));reference=dict(DFG_REFERENCE);uncertainty=0.
    else:
        reference=dict(drag=NACA_REFERENCE['drag_coefficient'],lift=NACA_REFERENCE['lift_coefficient']);uncertainty=NACA_REFERENCE['digitization_absolute_uncertainty']
    metrics=[]
    for name,ref in reference.items():
        err=max(abs(computed[name]-end)/abs(end) for end in (ref-uncertainty,ref+uncertainty))
        metrics.append(Metric(name+'_relative_error',err,.03,'Worst endpoint error for figure uncertainty; DFG tabulated reference has zero digitization interval'))
    last=s.history[-1];metrics+=[Metric('solver_residual',max(last[k] for k in ('continuity','momentum','mass_imbalance')),c.tolerance,'Actual steady equations'),Metric('convergence_failure',0 if s.converged else 1,.5,'Explicit convergence required')]
    records,passed=evaluate(metrics)
    fields=dict(vertices_m=array(s.mesh.vertices),cell_centers_m=array(s.mesh.centers),cell_velocity_m_s=array(s.velocity),cell_pressure_pa=array(s.p),cell_volumes_m2=array(s.volume),face_owner=array(s.o),face_neighbor=array(s.n),internal_face_mask=array(s.f),face_mass_flux=array(s.mass_flux),face_area_vectors_m=array(s.S),surface_mask=array(mask),surface_centers_m=centers,surface_displacement_m=array(d),surface_gradient=array(grad),surface_pressure_force=array(pressure),surface_viscous_force=array(viscous),surface_wall_pressure_pa=wallp)
    return dict(case=case,resolution=f'{nx}x{ny}',device=device,config=asdict(c),iterations=len(s.history),elapsed_s=time.perf_counter()-start,converged=s.converged,computed=computed,reference=reference,digitization_absolute_uncertainty=uncertainty,metrics=records,passed=passed,history=s.history,linear_solver_history=s.linear_history,coupled_solver_history=getattr(s,"coupled_history",[]),solver=solver,fields=fields)


def audit(directory,row):
    with np.load(directory/'fields.npz') as f:
        c=row['config'];mask=f['surface_mask'];o=f['face_owner'][mask];S=f['face_area_vectors_m'][mask];g=f['surface_gradient'];mu=c['density']*c['inlet_velocity']*(2*c['cylinder_radius'] if row['case']=='cylinder' else c['airfoil_chord'])/c['reynolds']
        pf=f['cell_pressure_pa'][o,None]*S;vf=-np.einsum('fij,fj->fi',mu*(g+g.transpose(0,2,1)),S)
        np.testing.assert_allclose(pf,f['surface_pressure_force'],rtol=0,atol=1e-12);np.testing.assert_allclose(vf,f['surface_viscous_force'],rtol=0,atol=1e-12)
        wall=f['cell_velocity_m_s'][o]+np.einsum('fi,fij->fj',f['surface_displacement_m'],g)
        if np.max(abs(wall))>1e-10:raise ValueError('wall reconstruction violates no-slip')
        a=np.deg2rad(c['angle_of_attack']);force=(pf+vf).sum(0);scale=.5*c['density']*c['inlet_velocity']**2*(2*c['cylinder_radius'] if row['case']=='cylinder' else c['airfoil_chord']);coeff=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])@force/scale
        np.testing.assert_allclose(coeff,[row['computed']['drag'],row['computed']['lift']],rtol=0,atol=1e-10)
        net=np.zeros(len(f['cell_volumes_m2']));np.add.at(net,f['face_owner'],f['face_mass_flux']);internal=f['internal_face_mask'];np.add.at(net,f['face_neighbor'][internal],-f['face_mass_flux'][internal])
    for k,ref in row['reference'].items():
        u=row['digitization_absolute_uncertainty'];value=max(abs(row['computed'][k]-x)/abs(x) for x in (ref-u,ref+u));metric=next(q for q in row['metrics'] if q['name']==k+'_relative_error')
        if abs(metric['value']-value)>1e-12 or metric['limit']!=.03 or metric['passed']!=(value<.03):raise ValueError('false reference acceptance')
    if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('false case badge')
    return dict(passed=True,physical_passed=row['passed'],maximum_cell_mass_residual=float(np.max(abs(net))),scope='Independent NumPy traction, no-slip, mass-flux and reference gates; not a full independent PDE replay')


def publish(output,rows):
    from .report import plt,figure
    from matplotlib.collections import PolyCollection
    from matplotlib.backends.backend_pdf import PdfPages
    text='# 圆柱与 NACA0012 贴体有限体积 benchmark\n\n## 问题、算法与验收\n\n圆柱采用 DFG 2D-1，Re=20，通道2.2×0.41 m、直径0.1 m、平均入口0.2 m/s、抛物线入口。参考来源：[FeatFlow]('+REFERENCE_URL+')。翼型采用 NACA0012、Re=1000、攻角4°、单位弦长、20×16计算域、翼型前缘(6,8)。参考来源：[Di Ilio 2020](https://arxiv.org/html/2006.10487)，图10/11 present study。\n\n原生贴体 SIMPLE、Rhie–Chow、一次迎风对流、非正交扩散，CPU使用SciPy SuperLU稀疏直接解，翼型最细网格用缓存LU预条件GMRES求解当前矩阵（缓存只作预条件器，不滞后离散方程），CUDA使用Torch Krylov；两者只加速相同线性方程，并检查真实线性残差。三网格主报告另启用CPU全网格耦合Picard残差修正（仅圆柱，翼型采用原SIMPLE），保留真实周期连通、非正交扩散及物理Rhie–Chow，独立检查耦合矩阵响应与真实线性残差；最终记录所有修正后的实际方程残差与<10⁻⁷通量固定点缺陷。名为 c-grid 的翼型网格实际为周期径向拓扑。圆柱 Cd、Cl、前后压力差及翼型 Cd、Cl 分别严格小于3%，同时满足实际稳态残差。保存完整原始场、面通量、分项牵引、历史与源文件哈希。\n\n## 参考值修正\n\n旧翼型0.205/0.120图上估读不能支撑严格3%验收。本轮从矢量路径读出4°处坐标：Cl=27.41/190.95×1.4，Cd=23.87/190.95，各±0.0001图形读取区间。使用两端点最坏相对误差；此区间不包含作者数值方法误差。旧报告保留为历史记录，其通过结论撤回。4°没有匹配的公开Cp表，因此Cp曲线仅作计算结果展示。\n\n## 软件使用\n\n```bash\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.external_flow --output results/external --case cylinder --grids 64x24 128x48 256x96 --solver coupled\n# 翼型: --case naca --grids 64x26 128x52 256x104\n# --device cuda 为相同方程的 GPU 计算，单次耗时不是重复性能测量\n```\n\n## 计算结果\n\n|案例|网格|设备|迭代/收敛|Cd|Cl|最大物理误差|全部通过|\n|---|---|---|---|---|---|---|---|\n'
    for r in rows:
        err=max(q['value'] for q in r['metrics'] if q['name'].endswith('relative_error'))
        text+=f"|{r['case']}|{r['resolution']}|{r['device']}|{r['iterations']}/{r['converged']}|{r['computed']['drag']:.8g}|{r['computed']['lift']:.8g}|{err*100:.4f}%|{r['passed']}|\n"
    text+='\n## 验证范围\n\n独立审计重新积分压力及黏性牵引，检查壁面无滑移和质量通量，重算严格参考验收；未独立重跑全部离散方程。网格增加并不自动证明渐近收敛。未执行匹配 LBM 计算时不作性能或精度优劣排名。本轮为二维稳态层流；高Re尾涡、湍流及三维流动需要额外验证。\n'
    with PdfPages(output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,'Body-fitted external flow validation',fontsize=16);body='SIMPLE/Rhie-Chow; first order upwind; strict independent physical error <3%.\nNACA reference: vector-digitized Di Ilio 2020, uncertainty +/-0.0001.\nHistoric NACA 0.205/0.120 pass claim is withdrawn.\nSingle grid and residual convergence do not establish physical accuracy.\n'
        for r in rows:body+=f"\n{r['case']} {r['resolution']} {r['device']}: converged={r['converged']}, passed={r['passed']}\n"+str(r['computed'])+'\n'+str({q['name']:q['value'] for q in r['metrics']})+'\n'
        fig.text(.08,.87,body,fontsize=9,va='top',wrap=True);pdf.savefig(fig);plt.close(fig)
        for r in rows:
            directory=output/r['directory'];figs=directory/'figures';figs.mkdir(exist_ok=True)
            with np.load(directory/'fields.npz') as f:
                v=f['vertices_m'];cells=np.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),axis=2).reshape(-1,4,2)
                for name,values,label in [('pressure',f['cell_pressure_pa'],'Pressure (Pa)'),('velocity',np.linalg.norm(f['cell_velocity_m_s'],axis=1),'Speed (m/s)')]:
                    fig,ax=plt.subplots(figsize=(10,4));p=PolyCollection(cells,array=values,cmap='viridis',rasterized=True);ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title=r['case']+' '+r['resolution']);fig.colorbar(p,ax=ax,label=label);figure(fig,figs,name);pdf.savefig(fig);plt.close(fig)
                centers=f['surface_centers_m'];c=r['config']
                if r['case']=='cylinder':x=np.mod(np.arctan2(centers[:,1]-c['cylinder_y'],centers[:,0]-c['cylinder_x']),2*np.pi);y=f['surface_wall_pressure_pa'];xlabel='Angle (rad)';ylabel='Wall pressure (Pa)';groups=[np.arange(len(x))]
                else:x=(centers[:,0]-c['airfoil_x'])/c['airfoil_chord'];y=f['surface_wall_pressure_pa']/(.5*c['density']*c['inlet_velocity']**2);xlabel='x/c';ylabel='Cp (outlet reference p=0)';groups=[np.flatnonzero(centers[:,1]>=c['airfoil_y']),np.flatnonzero(centers[:,1]<c['airfoil_y'])]
                fig,ax=plt.subplots(figsize=(6,4))
                for group in groups:
                    idx=group[np.argsort(x[group])];ax.plot(x[idx],y[idx])
                ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.3);figure(fig,figs,'surface-pressure');pdf.savefig(fig);plt.close(fig)
            for name in ('pressure','velocity','surface-pressure'):text+=f"\n![{r['case']} {r['resolution']} {name}]({r['directory']}/figures/{name}.png)\n"
    (output/'report.md').write_text(text)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['cylinder','naca'],required=True);p.add_argument('--grids',nargs='+',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-iterations',type=int,default=3000);p.add_argument('--solver',choices=['simple','coupled','cached'],default='simple');a=p.parse_args(argv)
    if a.case=='naca' and a.solver=='coupled':p.error('Coupled external correction is currently verified for cylinders only')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);prov=provenance();root=Path(__file__).resolve().parents[3]
    for name in ('sparse_simple.py','external_linear.py','external_cached.py','external_coupled.py','coupled_coarse.py','benchmark_airfoil.py'):prov['source_sha256']['src/tensorfvm/'+name]=sha(root/'src/tensorfvm'/name)
    summary=dict(provenance=prov,runs=[],passed=False,grid_convergence_qualified=False)
    for grid in a.grids:
        nx,ny=map(int,grid.split('x'));raw=run(a.case,nx,ny,a.device,a.max_iterations,a.solver);raw['directory']=grid;row=save_run(a.output/grid,raw);write_json(a.output/grid/'audit.json',audit(a.output/grid,row));summary['runs'].append(row);write_json(a.output/'summary.json',summary)
    summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs']);write_json(a.output/'summary.json',summary);publish(a.output,summary['runs']);write_json(a.output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(a.output)));return 0 if summary['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
