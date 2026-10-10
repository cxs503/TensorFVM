"""DFG Re20 cylinder evidence through common metrics and raw exports."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .cases import array
from .core import Metric, evaluate, save_run, write_json, provenance, artifact_manifest
from ..solver import SolverConfig, SimpleSolver
from ..cylinder import export_result

REFERENCE=dict(drag=5.57953523384,lift=.010618948146,pressure_difference_pa=.11752016697)
REFERENCE_URL='https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html'


def run(nx,ny,max_iterations=1500):
    start=time.perf_counter();c=SolverConfig(mesh_type='body-fitted',inlet_profile='parabolic',nx=nx,ny=ny,length=2.2,height=.41,cylinder_x=.2,cylinder_y=.2,cylinder_radius=.05,inlet_velocity=.2,density=1,reynolds=20,tolerance=1e-6,max_iterations=max_iterations)
    s=SimpleSolver(c)
    for i in range(c.max_iterations):
        s.step()
        if (i+1)%100==0:print(f'{nx}x{ny} iteration {i+1}: momentum {s.history[-1]["momentum"]:.6g}',flush=True)
        if s.converged:break
    # solve() simply packages the current converged/iteration-limited state.
    r=s.solve();mask=s.mesh.masks['cylinder'];owners=s.o[mask];grad=s._gradient(s.velocity)[owners].clone();wall_d=s.d[mask]
    error=-s.velocity[owners]-torch.einsum('fi,fij->fj',wall_d,grad)
    grad+=wall_d[:,:,None]*error[:,None,:]/wall_d.square().sum(-1)[:,None,None]
    stress=c.viscosity*(grad+grad.transpose(-1,-2));pressure_force=s.p[owners,None]*s.S[mask];viscous_force=-torch.einsum('fij,fj->fi',stress,s.S[mask]);body_force=pressure_force+viscous_force
    centers=array(s.mesh.face_centers[mask]);angle=np.mod(np.arctan2(centers[:,1]-.2,centers[:,0]-.2),2*np.pi);order=np.argsort(angle)
    wallpressure=array(s.p[owners]+(s._gradient(s.p,pressure=True)[owners]*wall_d).sum(-1))
    angles=angle[order];wp=wallpressure[order];extended_angles=np.r_[angles[-1]-2*np.pi,angles,angles[0]+2*np.pi];extended_p=np.r_[wp[-1],wp,wp[0]]
    dp=float(np.interp(np.pi,extended_angles,extended_p)-np.interp(0.,extended_angles,extended_p))
    coeff=r.aerodynamic_coefficients;values=dict(drag=float(coeff['drag']),lift=float(coeff['lift']),pressure_difference_pa=dp)
    metrics=[Metric(k+'_relative_error',abs(values[k]-ref)/abs(ref),.03,'DFG high-accuracy reference; pressure uses linear wall extrapolation and angle interpolation') for k,ref in REFERENCE.items()]
    last=r.history[-1];flux=array(s.mass_flux);o=array(s.o);nei=array(s.n);internal=array(s.f);volume=array(s.volume)
    net=np.zeros(len(volume));np.add.at(net,o,flux);np.add.at(net,nei[internal],-flux[internal]);boundary=array(s.mesh.boundary);inlet=array(s.mesh.masks['inlet']);outlet=array(s.mesh.masks['outlet'])
    scale=.5*c.density*c.inlet_velocity**2*c.reference_length
    metrics += [Metric('solver_residual',max(float(last[k]) for k in ('continuity','momentum','mass_imbalance')),c.tolerance,'final actual SIMPLE residual'),Metric('convergence_failure',0. if r.converged else 1.,.5,'explicit convergence required'),Metric('relative_boundary_mass_imbalance',abs(flux[boundary].sum())/abs(flux[inlet].sum()),1e-6,'authoritative boundary face mass'),Metric('scaled_face_divergence',float(np.max(abs(net/volume)))*c.reference_length/(c.density*c.inlet_velocity),1e-6,'finite-volume cell continuity from actual mass flux')]
    rows,passed=evaluate(metrics)
    raw=dict(case='dfg-re20',role='spatial',resolution=f'{nx}x{ny}',spacing_m=1/nx,config=asdict(c),elapsed_s=time.perf_counter()-start,iterations=len(r.history),converged=r.converged,computed=values,reference=dict(values=REFERENCE,url=REFERENCE_URL,pressure_sampling='least-squares extrapolate owner pressure to polygon wall-face centers, periodic linear angle interpolation at front/rear; no exact pressure imposed'),metrics=rows,passed=passed,history=r.history,
      fields=dict(cell_centers_m=array(s.mesh.centers),vertices_m=array(s.mesh.vertices),cell_velocity_m_s=array(s.velocity),cell_pressure_pa=array(s.p),cell_volumes_m2=volume,face_owner=o,face_neighbor=nei,internal_face_mask=internal,boundary_face_mask=boundary,inlet_face_mask=inlet,outlet_face_mask=outlet,face_centers_m=array(s.mesh.face_centers),face_area_vectors_m=array(s.S),face_mass_flux_kg_s_m=flux,cylinder_face_mask=array(mask),cylinder_pressure_force_n_m=array(pressure_force),cylinder_viscous_force_n_m=array(viscous_force),cylinder_wall_gradient_s_inv=array(grad),cylinder_owner_displacement_m=array(wall_d),cylinder_angle_rad=angle,cylinder_wall_pressure_pa=wallpressure,force_scale_n_m=np.array(scale),net_cell_mass_flux_kg_s_m=net))
    return raw,r


def audit_case(directory,row):
    f=np.load(directory/'fields.npz',allow_pickle=False);c=row['config'];flux=f['face_mass_flux_kg_s_m'];o=f['face_owner'];nei=f['face_neighbor'];internal=f['internal_face_mask'];mask=f['cylinder_face_mask'];owners=o[mask];S=f['face_area_vectors_m'][mask];grad=f['cylinder_wall_gradient_s_inv'];mu=c['density']*c['inlet_velocity']*(2*c['cylinder_radius'])/c['reynolds']
    pforce=f['cell_pressure_pa'][owners,None]*S;vforce=-np.einsum('fij,fj->fi',mu*(grad+grad.transpose(0,2,1)),S)
    if not np.allclose(pforce,f['cylinder_pressure_force_n_m'],rtol=0,atol=1e-12) or not np.allclose(vforce,f['cylinder_viscous_force_n_m'],rtol=0,atol=1e-12):raise ValueError('traction inconsistency')
    force=(pforce+vforce).sum(0);scale=.5*c['density']*c['inlet_velocity']**2*(2*c['cylinder_radius'])
    if not np.allclose(force/scale,[row['computed']['drag'],row['computed']['lift']],rtol=0,atol=1e-10):raise ValueError('coefficient inconsistency')
    # Audit actual wall no-slip reconstruction independently from saved corrected gradient.
    wallvalue=f['cell_velocity_m_s'][owners]+np.einsum('fi,fij->fj',f['cylinder_owner_displacement_m'],grad)
    if np.max(abs(wallvalue))>1e-10:raise ValueError('wall gradient violates no-slip')
    net=np.zeros(len(f['cell_volumes_m2']));np.add.at(net,o,flux);np.add.at(net,nei[internal],-flux[internal])
    if not np.allclose(net,f['net_cell_mass_flux_kg_s_m'],rtol=0,atol=1e-14):raise ValueError('mass reconstruction mismatch')
    for name,key in [('drag','drag'),('lift','lift'),('pressure_difference_pa','pressure_difference_pa')]:
        q=next(q for q in row['metrics'] if q['name']==name+'_relative_error');value=abs(row['computed'][key]-REFERENCE[key])/abs(REFERENCE[key])
        if q['limit']!=.03 or abs(value-q['value'])>1e-12 or q['passed']!=(value<.03):raise ValueError('false physical metric')
    if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('false case badge')
    return dict(raw_reconstruction_passed=True,physical_passed=row['passed'],pressure_force_n_m=pforce.sum(0).tolist(),viscous_force_n_m=vforce.sum(0).tolist(),maximum_cell_mass_residual_kg_s_m=float(np.max(abs(net))),scope='NumPy integration of stored pressure/gradient traction and authoritative face mass; not an independent entire SIMPLE replay')


def publish(output,rows):
    from .report import figure,plt
    from matplotlib.collections import PolyCollection
    from matplotlib.backends.backend_pdf import PdfPages
    text='# DFG 2D-1 Re20 圆柱绕流诊断报告\n\n## 问题与参考\n\n通道 2.2×0.41 m，圆心 (0.2,0.2) m，直径0.1 m，ρ=1，μ=0.001 Pa·s，入口抛物线平均速度0.2 m/s，Re=20；固壁无滑移、出口定压。\n\n公开高精度参考 Cd=5.57953523384，Cl=0.010618948146，前后压力差0.11752016697 Pa。来源：[FeatFlow DFG 2D-1]('+REFERENCE_URL+')。三个物理指标分别严格 <3%，残差收敛不能代替物理精度。\n\n## 软件与证据\n\n采用现有贴体 O-grid SIMPLE/Rhie–Chow 求解器。压力差按壁面外推和角度插值得到，该近似会贡献误差。共用 Metric/evaluate/save_run 与 publication figure；完整字段、网格、质量面通量、压力/黏性分力和壁面梯度保存于 NPZ。独立 NumPy 审计积分这些保存字段；未声称独立重跑全部 SIMPLE。\n\n复现：\n\n~~~bash\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.dfg --output results/dfg --grids 64x24\n# 后续细化：--grids 64x24 128x48 256x96\n~~~\n\n## 结果\n\n| 网格 | 收敛 | Cd/误差 | Cl/误差 | Δp/误差 | 全部通过 |\n|---|---|---|---|---|---|\n'
    for r in rows:
        vals=[]
        for k in REFERENCE:
            err=next(q['value'] for q in r['metrics'] if q['name']==k+'_relative_error');vals.append(f"{r['computed'][k]:.8g} / {100*err:.4f}%")
        text+=f"| {r['resolution']} | {r['converged']} | "+' | '.join(vals)+f" | {r['passed']} |\n"
        directory=output/r['directory'];f=np.load(directory/'fields.npz');figs=directory/'figures';figs.mkdir(exist_ok=True);v=f['vertices_m'];ny,nx=r['config']['ny'],r['config']['nx'];cells=np.stack((v[:-1,:-1],v[1:,:-1],v[1:,1:],v[:-1,1:]),axis=2).reshape(-1,4,2)
        for name,value,label in [('pressure',f['cell_pressure_pa'],'Pressure (Pa)'),('speed',np.linalg.norm(f['cell_velocity_m_s'],axis=1),'Speed (m/s)')]:
            fig,ax=plt.subplots(figsize=(10,3));p=PolyCollection(cells,array=value,cmap='coolwarm' if name=='pressure' else 'viridis',edgecolors='none',rasterized=True);ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='DFG Re20 '+r['resolution']);fig.colorbar(p,ax=ax,label=label);figure(fig,figs,name)
        angle=f['cylinder_angle_rad'];order=np.argsort(angle);fig,ax=plt.subplots(figsize=(6,4));ax.plot(angle[order],f['cylinder_wall_pressure_pa'][order]);ax.set(xlabel='Cylinder angle (rad)',ylabel='Wall pressure (Pa)',title='Extrapolated cylinder pressure');ax.grid(alpha=.3);figure(fig,figs,'wall-pressure')
        hist=json.loads((directory/'history.json').read_text());fig,ax=plt.subplots(figsize=(6,4))
        for k in ('continuity','momentum','mass_imbalance'):ax.semilogy([q[k] for q in hist],label=k)
        ax.set(xlabel='SIMPLE iteration',ylabel='Residual');ax.legend();ax.grid(alpha=.3);figure(fig,figs,'convergence')
    text+='\n## 结论与推进\n\n这是实际求解并归档的诊断材料。单网格不证明网格收敛；任何未通过指标保留，不调参考或放宽3%门。下一步做网格、壁面牵引及压力重建的误差分解，再执行三网格细化。没有同条件 LBM 圆柱结果，因此不作圆柱 FVM/LBM 优劣排名。\n'
    for r in rows:
        for name in ('pressure','speed','wall-pressure','convergence'):text+=f"\n![{name}]({r['directory']}/figures/{name}.png)\n"
    (output/'report.md').write_text(text)
    with PdfPages(output/'report.pdf') as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.95,'DFG Re20 cylinder: diagnostic evidence',fontsize=16,va='top');body='Physical problem: parabolic inlet, channel 2.2 x 0.41 m, cylinder D=0.1 m,\ncenter (0.2,0.2), mean speed 0.2 m/s, rho=1, mu=0.001, Re=20.\n\nReference: FeatFlow DFG 2D-1; Cd=5.57953523384, Cl=0.010618948146,\nfront/rear pressure difference=0.11752016697 Pa.\nThree separate strict 3% physical gates plus residual/mass gates.\n\nBody-fitted SIMPLE/Rhie-Chow production solver; polygon wall traction.\nPressure difference: least-squares wall extrapolation and angle interpolation.\nRaw field, mesh, pressure/viscous traction, corrected wall gradients and\nauthoritative mass flux exported. Independent NumPy integration audit.\n\n'
        for r in rows:body+=f"{r['resolution']}: converged={r['converged']}, physical gates={r['passed']}\n"+'\n'.join(f"{k}: {r['computed'][k]:.9g}, error {100*next(q['value'] for q in r['metrics'] if q['name']==k+'_relative_error'):.4f}%" for k in REFERENCE)+'\n'
        body+='\nSingle-grid diagnosis does not establish grid convergence.\nNo matched LBM cylinder run; no industrial or icebreaking qualification.\nNext: refine geometry, transport and wall-pressure/traction reconstruction.';fig.text(.08,.89,body,fontsize=9,va='top',linespacing=1.65);pdf.savefig(fig);plt.close(fig)
        for r in rows:
            fig,axes=plt.subplots(2,2,figsize=(8.27,8));fig.suptitle('DFG Re20 '+r['resolution'])
            for ax,name in zip(axes.flat,('pressure','speed','wall-pressure','convergence')):ax.imshow(plt.imread(output/r['directory']/'figures'/(name+'.png')));ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--grids',nargs='+',default=['64x24','128x48','256x96']);p.add_argument('--max-iterations',type=int,default=1500);args=p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):p.error('output must be empty')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);summary=dict(provenance=provenance(),reference=REFERENCE,reference_url=REFERENCE_URL,runs=[],passed=False,grid_convergence_qualified=False)
    for grid in args.grids:
        nx,ny=map(int,grid.split('x'));raw,r=run(nx,ny,args.max_iterations);raw['directory']=grid;row=save_run(args.output/grid,raw);export_result(r,args.output/grid/'portable');write_json(args.output/grid/'audit.json',audit_case(args.output/grid,row));summary['runs'].append(row);print(grid,'physical passed',row['passed'],flush=True)
    summary['passed']=bool(len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs']));write_json(args.output/'summary.json',summary);publish(args.output,summary['runs']);write_json(args.output/'manifest.json',dict(source_sha256=summary['provenance']['source_sha256'],artifacts_sha256=artifact_manifest(args.output)));return 0 if summary['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
