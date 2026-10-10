"""Full collocated SIMPLE: curved manufactured flow, steady SA and CUDA evidence."""
import argparse,json,math,statistics,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from .core import Metric,evaluate,save_run,provenance,sha,write_json,artifact_manifest
from ..solver import SolverConfig
from ..sparse_simple import SparseBodyFittedSolver,host
from ..curved_channel import reference


def config(kind,nx,ny,device,max_iterations):
    if kind=='curved':return SolverConfig(mesh_type='curved-channel',nx=nx,ny=ny,length=2,height=1,cylinder_radius=None,inlet_profile='parabolic',reynolds=1,inlet_velocity=1,max_iterations=max_iterations,tolerance=1e-6,device=device)
    return SolverConfig(mesh_type='flat-plate',turbulence_model='spalart-allmaras',nx=nx,ny=ny,length=1,height=.2,cylinder_radius=None,inlet_velocity=1,reynolds=100000,velocity_relaxation=.5,pressure_relaxation=.3,turbulence_relaxation=.5,pseudo_time_step=None,max_iterations=max_iterations,tolerance=1e-5,device=device)


def state(s):return dict(velocity=host(s.velocity).copy(),pressure=host(s.p).copy(),mass_flux=host(s.mass_flux).copy(),nu_tilde=host(s.nu_tilde).copy())


def restore(s,f):
    for key,name in [('velocity','velocity'),('pressure','p'),('mass_flux','mass_flux'),('nu_tilde','nu_tilde')]:getattr(s,name).copy_(torch.as_tensor(f[key],device=s.p.device))
    s.turbulent_kinematic_viscosity=s._sa_eddy_viscosity(s.nu_tilde);s._anderson_history=[];s.history=[];s.linear_history=[];s.coarse_history=[];s.converged=False
    if hasattr(s,'_coupled_coarse_cache'):del s._coupled_coarse_cache


def initialize(s,kind,seed=None):
    if kind=='curved':
        s.momentum_body_force=reference(s.mesh.centers.reshape(-1,2),s.config)[2];s.anderson_depth=0;s.coupled_coarse_shape=(16,8)
    else:s.sa_inner_iterations=8;s.anderson_depth=0;s.coupled_coarse_shape=(16,14)
    if seed is not None:
        # Interpolate only an actually computed coarse solution, never analytic data.
        ny,nx=seed['shape'];fy,fx=s.field_shape;points=np.stack(np.meshgrid((np.arange(fy)+.5)/fy,(np.arange(fx)+.5)/fx,indexing='ij'),-1).reshape(-1,2)
        axes=((np.arange(ny)+.5)/ny,(np.arange(nx)+.5)/nx)
        for key,field in [('velocity',s.velocity),('pressure',s.p)]:
            a=seed[key].reshape((ny,nx)+( (2,) if key=='velocity' else ()))
            values=RegularGridInterpolator(axes,a,bounds_error=False,fill_value=None)(points);field.copy_(torch.as_tensor(values,device=s.p.device))
        diagonal,_,_,_=s._momentum(s.velocity,s.p,s.mass_flux);D=s.volume/(diagonal/s.config.velocity_relaxation);flux,co=s._rhie_chow(s.velocity,s.p,D);s._correct(flux,co,D)


def exported(s,kind):
    f=dict(vertices_m=host(s.mesh.vertices),cell_centers_m=host(s.mesh.centers),cell_volumes_m2=host(s.volume),velocity_m_s=host(s.velocity),pressure_pa=host(s.p),nu_tilde_m2_s=host(s.nu_tilde),eddy_viscosity_m2_s=host(s.turbulent_kinematic_viscosity),face_owner=host(s.o),face_neighbor=host(s.n),interior_face_mask=host(s.f),face_area_vectors_m=host(s.S),face_centers_m=host(s.mesh.face_centers),face_mass_flux_kg_s_m=host(s.mass_flux),inlet_face_mask=host(s.mesh.masks['inlet']),outlet_face_mask=host(s.mesh.masks['outlet']),wall_face_mask=host(s._no_slip_faces()),boundary_velocity_m_s=host(s.boundary_velocity),pressure_gradient_pa_m=host(s._gradient(s.p,pressure=True)),velocity_gradient_s_inv=host(s._gradient(s.velocity)),nonorthogonal_area_vectors_m=host(s.T),owner_neighbor_displacement_m=host(s.d))
    cache=getattr(s,'_coupled_coarse_cache',None)
    if cache is not None:f.update(coupled_coarse_matrix=host(cache['matrix']),coupled_coarse_last_solution=host(cache['last_q']),coupled_coarse_last_rhs=host(cache['last_rhs']))
    c=s.config;last=s.history[-1];numerical=max(last[k] for k in ('continuity','momentum','mass_imbalance')+(('turbulence',) if kind=='sa' else ()))
    metrics=[Metric('solver_residual',numerical,c.tolerance,'actual production final residual including SA when active'),Metric('convergence_failure',0. if s.converged else 1.,.5,'explicit steady convergence')]
    if kind=='curved':
        eu,ep,force=reference(s.mesh.centers.reshape(-1,2),c);f.update(reference_velocity_m_s=host(eu),reference_pressure_pa=host(ep),body_force_n_m3=host(force));vol=f['cell_volumes_m2'];u=f['velocity_m_s'];p=f['pressure_pa'];ur=f['reference_velocity_m_s'];pr=f['reference_pressure_pa']
        values=dict(velocity_volume_l2=np.sqrt(np.sum(vol[:,None]*(u-ur)**2)/np.sum(vol[:,None]*ur**2)),velocity_linf=np.max(abs(u-ur))/np.max(abs(ur)),pressure_volume_l2=np.sqrt(np.sum(vol*(p-pr)**2)/np.sum(vol*pr**2)),pressure_linf=np.max(abs(p-pr))/np.max(abs(pr)))
        for key,value in values.items():metrics.append(Metric(key,float(value),.03,'manufactured continuous Navier-Stokes solution; strict 3%'))
    else:
        mask=s._no_slip_faces();o=s.o[mask];grad=s._gradient(s.velocity)[o].clone();disp=s.d[mask];error=-s.velocity[o]-torch.einsum('fi,fij->fj',disp,grad);grad+=disp[:,:,None]*error[:,None,:]/disp.square().sum(-1)[:,None,None];tau=-torch.einsum('fij,fj->fi',c.viscosity*(grad+grad.transpose(-1,-2)),s.S[mask]);Cf=float(tau[:,0].sum()/(.5*c.density*c.inlet_velocity**2*c.length));ref=.074*c.reynolds**(-.2)
        f.update(wall_velocity_gradient_s_inv=host(grad),wall_viscous_force_n_m=host(tau),wall_displacement_m=host(disp));metrics.append(Metric('skin_friction_relative_error',abs(Cf-ref)/ref,.03,'fully turbulent empirical mean Cf, not DNS or matched NASA validation'));f['computed_mean_cf']=np.array(Cf);f['reference_mean_cf']=np.array(ref)
    rows,passed=evaluate(metrics)
    return dict(config=asdict(c),case=kind,iterations=len(s.history),converged=s.converged,metrics=rows,passed=passed,actual_solver_device=str(s.p.device),linear_history=s.linear_history,history=s.history,fields=f,initial_state='uniform freestream' if not getattr(s,'used_coarse_seed',False) else 'interpolated actual coarse solution plus production conservative pressure correction',anderson_depth=getattr(s,'anderson_depth',0),sa_inner_iterations=getattr(s,'sa_inner_iterations',1),coupled_coarse_shape=getattr(s,'coupled_coarse_shape',None),coarse_history=getattr(s,'coarse_history',[]),linear_backend='Torch restarted GMRES with double orthogonalization; pressure line/coarse preconditioning; host SciPy CSR assembly included')


def run(kind,nx,ny,device,max_iterations,seed=None):
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();s=SparseBodyFittedSolver(config(kind,nx,ny,device,max_iterations));initialize(s,kind,seed);s.used_coarse_seed=seed is not None;checkpoint=None
    for i in range(max_iterations):
        s.step()
        last=s.history[-1]
        if checkpoint is None and max(last[k] for k in ('momentum','continuity','mass_imbalance'))<1e-3:checkpoint=state(s)
        if (i+1)%50==0:print(kind,nx,ny,device,i+1,'momentum',last['momentum'],'SA',last.get('turbulence'),flush=True)
        if s.converged:break
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start;raw=exported(s,kind);raw['full_solve_elapsed_s']=elapsed;raw['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else None
    return s,raw,checkpoint or state(s)


def window(s,checkpoint,repetitions):
    def once():
        restore(s,checkpoint)
        if s.p.is_cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(5):s.step()
        if s.p.is_cuda:torch.cuda.synchronize()
        return time.perf_counter()-start,state(s),torch.cuda.max_memory_allocated() if s.p.is_cuda else None
    once();times=[]
    for _ in range(repetitions):elapsed,fields,peak=once();times.append(elapsed)
    return dict(samples_s=times,median_s=statistics.median(times),five_actual_iterations=True,actual_solver_device=str(s.p.device),peak_cuda_allocated_bytes=peak),fields


def differences(a,b):
    return {k:float(np.linalg.norm((a[k]-b[k]).ravel())/max(np.linalg.norm(a[k].ravel()),1e-14)) for k in ['velocity','pressure','nu_tilde'] if np.linalg.norm(a[k])>1e-14}


def publish(output,summary):
    from .report import plt,figure
    from matplotlib.collections import PolyCollection
    from matplotlib.backends.backend_pdf import PdfPages
    figs=output/'figures';figs.mkdir(exist_ok=True)
    text='# 完整贴体 SIMPLE：曲壁、非正交、大网格与稳态 SA 的 CPU/CUDA 验证\n\n## 问题与方法\n\n弯曲通道 x∈[0,2]、y∈[g(x),g(x)+1]，g=0.2 sin⁴(πx/2)。入口解析抛物线、曲壁无滑移、出口定压零且速度零法向梯度。制造解 u=w(y−g)、v=g′w、w=6η(1−η)、p=12μ(2−x)，体力由连续 Navier–Stokes 精确导出。曲壁保持不通透；这是含非零横向流和对流的连续制造解，不是无外力自然弯管流动。\n\n复用生产 SIMPLE/Rhie–Chow、非正交修正、动量及 SA 方程；可选稀疏后端逐次核对组装压力矩阵与原矩阵自由算子。压力用行/粗网格预条件的 Torch GMRES，两端同算法，检查真实残差。CSR及粗矩阵在CPU组装并传输，线性迭代、预条件及场更新在实际 CPU/CUDA；不存在CPU线性求解回退。曲壁和 SA 使用可选的粗网格动量—压力耦合残差校正，不使用 Anderson；SA 每次外迭代最多八次真实输运子迭代。细网格以真正计算出的粗网格场插值启动，未使用解析场初始化。\n\nSA仍是 Re=100000 的首层贴壁原模型；平均 Cf 的经验关联误差单独列出。稳态残差通过与 GPU 一致性通过，均不能代替3%物理门。\n\n## 软件使用\n\n```sh\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.simple_gpu --output /tmp/fvm-simple-new\n```\n\n## 完整稳态结果\n\n| 案例/网格 | CPU/CUDA迭代 | 两端收敛 | 物理3% | 最大主场相对差 | 五步窗口 CPU/CUDA |\n|---|---:|---|---|---:|---:|\n'
    tables=[]
    for case in summary['cases']:
        cpu,gpu=case['cpu'],case['cuda'];ratio=case['window_cpu_over_cuda'];err=max(case['steady_field_differences'].values());text+=f"| {case['name']} | {cpu['iterations']}/{gpu['iterations']} | {cpu['converged']}/{gpu['converged']} | {cpu['passed']}/{gpu['passed']} | {err:.3e} | {ratio:.3f} |\n";tables.append([case['name'],f"{cpu['iterations']}/{gpu['iterations']}",str(cpu['converged'] and gpu['converged']),str(cpu['passed'] and gpu['passed']),f'{err:.2e}',f'{ratio:.3f}'])
        path=output/case['name']/'cuda';f=np.load(path/'fields.npz');ny,nx=gpu['config']['ny'],gpu['config']['nx'];v=f['vertices_m'];pol=np.stack((v[:-1,:-1],v[:-1,1:],v[1:,1:],v[1:,:-1]),2).reshape(-1,4,2)
        for label,values in [('pressure',f['pressure_pa']),('speed',np.linalg.norm(f['velocity_m_s'],axis=1)),('eddy-viscosity',f['eddy_viscosity_m2_s'])]:
            if label=='eddy-viscosity' and cpu['case']!='sa':continue
            fig,ax=plt.subplots(figsize=(9,3));collection=PolyCollection(pol,array=values,edgecolors='none',rasterized=True);ax.add_collection(collection);ax.autoscale_view();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title=case['name']+' '+label);fig.colorbar(collection,ax=ax,label='Pa' if label=='pressure' else 'm²/s' if label=='eddy-viscosity' else 'm/s');figure(fig,figs,case['name']+'-'+label)
    text+='\n## GPU计时范围与限制\n\n每个配置都在CPU、CUDA完成真实稳态求解，计时含设置/网格/组装/实际线性求解/主机传输，不含绘图；完整求解时间是一次测量，不能称三次中位数。另以CPU实际中间场（残差第一次低于1e−3）作为双方相同重启，完整预热后各做三次五步SIMPLE窗口，包含CPU组装及传输，排除重启复制和绘图。窗口不是完整稳态求解时间；其速度比只代表该窗口。float64，单线程CPU，RTX3090；小网格可能减速。主场一致性统一≤1e−6。\n\n完整 history、linear_history、原场、粗网格来源与窗口输出均保留。SIMPLEC/PISO/PIMPLE仍无独立生产实现，不能冒用这些名称。\n'
    for case in summary['cases']:
        for label in ['pressure','speed']:text+=f"\n![{label}](figures/{case['name']}-{label}.png)\n"
    (output/'report.md').write_text(text)
    with PdfPages(output/'report.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off');ax.set_title('Full collocated SIMPLE: curved channel and steady SA');t=ax.table(cellText=tables,colLabels=['Case','CPU/CUDA steps','Converged','Physical pass','Field delta','Window CPU/GPU'],colWidths=[.25,.17,.15,.15,.14,.14],loc='upper center');t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.7);ax.text(0,.30,'Curved continuous manufactured Navier-Stokes, with explicitly prescribed body force.\nProduction SIMPLE / Rhie-Chow / nonorthogonal corrections; strict true linear residuals.\nSparse matrices assembled on CPU; GMRES and preconditioning on actual CPU/CUDA.\nRefined initial fields interpolated from computed coarse states, not analytic data.\nFull steady solve on both devices; three repeated five-step timings are separate windows.\nSA empirical Cf gate remains independent of steady convergence and device equivalence.\nCoupled coarse correction accelerates the fine SIMPLE iteration.\nSIMPLEC / PISO / PIMPLE are not implemented or claimed.',fontsize=10,va='top',linespacing=1.6);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        for case in summary['cases']:
            fig,axes=plt.subplots(2,1,figsize=(11.69,8.27))
            for ax,label in zip(axes,['pressure','speed']):ax.imshow(plt.imread(figs/(case['name']+'-'+label+'.png')));ax.axis('off')
            pdf.savefig(fig);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--grids',nargs='+',default=['32x16','64x32','128x64','256x128','512x256']);p.add_argument('--max-iterations',type=int,default=2000);p.add_argument('--sa-grid',default='64x56');p.add_argument('--skip-sa',action='store_true');args=p.parse_args(argv)
    if not torch.cuda.is_available():p.error('real CUDA required')
    if args.output.exists() and any(args.output.iterdir()):p.error('empty output required')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);prov=provenance();root=Path(__file__).resolve().parents[3]
    for name in ['src/tensorfvm/sparse_simple.py','src/tensorfvm/curved_channel.py','src/tensorfvm/coupled_coarse.py']:prov['source_sha256'][name]=sha(root/name)
    summary=dict(provenance=prov,hardware=dict(gpu=torch.cuda.get_device_name(0),cpu_threads=1,dtype='float64'),cases=[]);seed=None
    plan=[('curved',grid) for grid in args.grids]+([] if args.skip_sa else [('sa',args.sa_grid)])
    for kind,grid in plan:
        nx,ny=map(int,grid.split('x'));name=kind+'-'+grid;directory=args.output/name;directory.mkdir();solvers={};states={};checkpoint=None
        for device in ['cpu','cuda']:
            print(name,device,'full solve start',flush=True);solver,raw,cp=run(kind,nx,ny,device,args.max_iterations,seed if kind=='curved' else None);raw['directory']=name+'/'+device;row=save_run(directory/device,raw);solvers[device]=solver;states[device]=state(solver)
            if device=='cpu':checkpoint=cp
            summary.setdefault('progress',{})[name+'-'+device]=dict(converged=row['converged'],iterations=row['iterations']);write_json(args.output/'execution-status.json',summary)
        delta=differences(states['cpu'],states['cuda']);windows={};wf={}
        for device in ['cpu','cuda']:windows[device],wf[device]=window(solvers[device],checkpoint,3);np.savez_compressed(directory/(device+'-window-fields.npz'),**wf[device]);write_json(directory/(device+'-window.json'),windows[device])
        np.savez_compressed(directory/'window-initial-state.npz',**checkpoint)
        cpu=json.loads((directory/'cpu/result.json').read_text());gpu=json.loads((directory/'cuda/result.json').read_text());row=dict(name=name,cpu=cpu,cuda=gpu,steady_field_differences=delta,steady_equivalent=all(v<=1e-6 for v in delta.values()),window_field_differences=differences(wf['cpu'],wf['cuda']),window_cpu_over_cuda=windows['cpu']['median_s']/windows['cuda']['median_s'],coarse_seed=None if seed is None or kind!='curved' else seed['name']);summary['cases'].append(row);write_json(directory/'comparison.json',row);write_json(args.output/'execution-status.json',summary)
        if kind=='curved':seed=dict(shape=(ny,nx),velocity=states['cpu']['velocity'],pressure=states['cpu']['pressure'],name=name)
        print(name,'steady equivalent',row['steady_equivalent'],'window speed',row['window_cpu_over_cuda'],flush=True)
    summary['all_steady_converged']=all(r['cpu']['converged'] and r['cuda']['converged'] for r in summary['cases']);summary['all_steady_equivalent']=all(r['steady_equivalent'] for r in summary['cases']);summary['all_physical_passed']=all(r['cpu']['passed'] and r['cuda']['passed'] for r in summary['cases']);write_json(args.output/'summary.json',summary);publish(args.output,summary);write_json(args.output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(args.output)));return 0 if summary['all_steady_converged'] and summary['all_steady_equivalent'] else 2


if __name__=='__main__':raise SystemExit(main())
