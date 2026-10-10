"""Matched CPU/CUDA correctness and synchronized repeated runtime evidence."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from .core import sha,write_json,provenance,artifact_manifest,evaluate,Metric
from .engineering_audit import annular_metrics
from .extended_cases import exact_velocity,exact_pressure,analytic_metrics
from .lbm import load_lbm
from ..annular import AnnularConfig,solve
from ..periodic_mac import PeriodicMACConfig,PeriodicMACSolver
from ..solver import SimpleSolver,SolverConfig
from ..wall_functions import u_plus_at_y_plus,torch_wall_traction


def arr(x):return x.detach().cpu().numpy()


def annulus(device,n,nr):
    raw=solve(AnnularConfig(ntheta=n,nr=nr),device,'torch-cg');values=annular_metrics(raw['fields'],raw['config'])
    raw['physical_passed']=all(v<(2e-10 if k=='true_linear_residual' else 1e-9 if k=='momentum_force_balance' else .03) for k,v in values.items());raw['metrics']=values;raw['primary_fields']=['axial_velocity_m_s'];raw['actual_solver_device']=raw['diagnostics']['device'];return raw


def periodic(device,kind,n):
    dt,end=(.005,.1) if kind=='abc' else (.0125,.5) if kind=='taylor-green' else (.01,.5);nz=n if kind=='abc' else 4
    c=PeriodicMACConfig(nx=n,ny=n,nz=nz,time_step_s=dt,viscosity_pa_s=.1,device=device);s=PeriodicMACSolver(c)
    def velocity(xyz,t):
        if kind!='taylor-green':return exact_velocity(kind,xyz,t)
        x,y,z=xyz;decay=math.exp(-.2*t);return torch.stack((torch.sin(x)*torch.cos(y),-torch.cos(x)*torch.sin(y),torch.zeros_like(z)))*decay
    for a in range(3):s.velocity[a]=velocity(s.face_coordinates(a),0)[a]
    states=[arr(s.velocity).copy()]
    for _ in range(round(end/dt)):s.step();states.append(arr(s.velocity).copy())
    eu=torch.stack([velocity(s.face_coordinates(a),end)[a] for a in range(3)]);t=end-dt/2
    if kind=='taylor-green':
        x,y,z=s.centers();ep=.25*(torch.cos(2*x)+torch.cos(2*y))*math.exp(-.4*t);ep-=ep.mean()
    else:ep=exact_pressure(kind,s.centers(),t)
    metrics,passed=evaluate(analytic_metrics('abc' if kind=='taylor-green' else kind,arr(s.velocity),arr(eu),arr(s.pressure),arr(ep)))
    numeric=all(h['nonlinear_converged'] and h['projection']['pressure_converged'] and h['max_divergence_s_inv']<1e-10 and abs(h['energy_balance_defect_J'])<1e-10 for h in s.history)
    return dict(config=asdict(c),field_comparison_scales={'pressure_pa':.745} if kind=='advected-shear' else {},physical_passed=passed and numeric,metrics=metrics,actual_solver_device=str(s.velocity.device),history=s.history,primary_fields=['velocity_m_s','pressure_pa'],fields=dict(velocity_m_s=arr(s.velocity),pressure_pa=arr(s.pressure),reference_velocity_m_s=arr(eu),reference_pressure_pa=arr(ep),accepted_faces_m_s=np.stack(states)))


def channel(device):
    c=SolverConfig(nx=72,ny=24,length=.12,height=.02,cylinder_radius=None,inlet_velocity=.0005,density=1000.,reynolds=1.,tolerance=1e-6,max_iterations=1000,device=device);s=SimpleSolver(c);r=s.solve();u,v=r.cell_center_velocity();exact=6*c.inlet_velocity*r.y/c.height*(1-r.y/c.height);i=int(torch.argmin(abs(r.x-5*c.height)));err=float(torch.linalg.vector_norm(u[:,i]-exact)/torch.linalg.vector_norm(exact))
    return dict(config=asdict(c),physical_passed=r.converged and err<.03,metrics=dict(velocity_l2=err),actual_solver_device=str(r.p.device),history=r.history,primary_fields=['u_faces_m_s','v_faces_m_s','pressure_pa'],fields=dict(u_faces_m_s=arr(r.u),v_faces_m_s=arr(r.v),pressure_pa=arr(r.p)))


def sa_iterations(device):
    c=SolverConfig(mesh_type='flat-plate',turbulence_model='spalart-allmaras',nx=128,ny=112,length=1.,height=.2,cylinder_radius=None,inlet_velocity=1.,reynolds=100000.,velocity_relaxation=.3,pressure_relaxation=.2,turbulence_relaxation=.3,pseudo_time_step=.02,sa_freestream_ratio=3.,flat_plate_stretching=4.,tolerance=1e-5,max_iterations=5,device=device);s=SimpleSolver(c)
    for _ in range(5):s.step()
    return dict(config=asdict(c),physical_passed=None,scope='five matched production SA/SIMPLE iterations only; no steady convergence or 3% physical qualification',metrics=dict(iterations=5,all_finite=bool(torch.isfinite(s.velocity).all() and torch.isfinite(s.nu_tilde).all())),actual_solver_device=str(s.p.device),history=s.history,primary_fields=['velocity_m_s','pressure_pa','nu_tilde_m2_s'],fields=dict(velocity_m_s=arr(s.velocity),pressure_pa=arr(s.p),nu_tilde_m2_s=arr(s.nu_tilde),eddy_viscosity_m2_s=arr(s.turbulent_kinematic_viscosity)))


def wall_batch(device):
    yp=np.logspace(-1,3,65536);up=u_plus_at_y_plus(yp);v=torch.from_numpy(np.column_stack((up*.05,np.zeros(len(yp))))).to(device);distance=torch.from_numpy(yp*1e-5/.05).to(device)
    result=torch_wall_traction(v,distance,1000.,.01);err=float((result['friction_velocity_m_s']-.05).abs().max()/.05)
    return dict(config=dict(samples=65536,density_kg_m3=1000.,viscosity_pa_s=.01),physical_passed=None,constitutive_passed=err<1e-8,scope='batched wall constitutive inversion; not a flow solve; includes input construction and transfer',actual_solver_device=str(result['traction_pa'].device),history=[],primary_fields=['traction_pa'],metrics=dict(friction_velocity_relative_linf=err),fields=dict(traction_pa=arr(result['traction_pa']),friction_velocity_m_s=arr(result['friction_velocity_m_s'])))


def lbm_shear(device,repo):
    d,solver,prov=load_lbm(repo);n=64;dx=2*math.pi/n;end=.5;steps=math.ceil(end*math.sqrt(3)*math.sqrt(1+.7**2)/(.05*dx));dt=end/steps;scale=dx/dt;tau=.5+.3*dt/dx**2
    coord=(torch.arange(n,device=device,dtype=torch.float64)+.5)*dx;y,x=torch.meshgrid(coord,coord,indexing='ij');f=d.equilibrium(torch.ones_like(x),torch.sin(y)/scale,torch.full_like(x,.7/scale));history=[];states=[arr(f).copy()]
    for i in range(steps):
        f=solver.stream(solver.collide_bgk(f,tau));states.append(arr(f).copy());history.append(dict(step=i+1,time_s=(i+1)*dt))
    rho,u,v=d.macroscopic(f);vel=torch.stack((u*scale,v*scale,torch.zeros_like(u)));ep=(rho-rho.mean())*scale**2/3;eu=exact_velocity('advected-shear',(x,y,torch.zeros_like(x)),end)
    metrics,passed=evaluate(analytic_metrics('advected-shear',arr(vel),arr(eu),arr(ep),np.zeros((n,n))))
    return dict(config=dict(nx=n,ny=n,time_step_s=dt,tau=tau,physical_speed_scale_m_s=scale),field_comparison_scales={'pressure_pa':.745},provenance=prov,physical_passed=passed,metrics=metrics,actual_solver_device=str(f.device),history=history,primary_fields=['velocity_m_s','pressure_pa'],fields=dict(velocity_m_s=arr(vel),pressure_pa=arr(ep),accepted_populations=np.stack(states)))


def execute(fn,device,repetitions):
    # Warm-up initializes both the actual algorithm and CUDA context; it is excluded.
    raw=fn(device)
    del raw
    elapsed=[];peaks=[]
    for _ in range(repetitions):
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();raw=fn(device)
        if device=='cuda':torch.cuda.synchronize();peaks.append(torch.cuda.max_memory_allocated())
        elapsed.append(time.perf_counter()-start)
    return raw,elapsed,max(peaks) if peaks else None


def publish(output,s):
    from .report import plt,figure
    from matplotlib.backends.backend_pdf import PdfPages
    figures=output/'figures';figures.mkdir();labels=[r['name'] for r in s['cases']];speed=[r['cpu_over_gpu_median'] for r in s['cases']];fig,ax=plt.subplots(figsize=(9,5));ax.barh(labels,speed);ax.axvline(1,color='k',ls='--');ax.set(xlabel='CPU median / synchronized CUDA median',title='Matched algorithm; ratio >1 indicates measured acceleration');ax.grid(axis='x',alpha=.3);figure(fig,figures,'acceleration')
    text='# CPU / CUDA 实际求解与加速验证\n\n## 方法\n\n硬件：'+s['hardware']['gpu_name']+'，float64，CPU单线程。每个案例先完整warm-up，再分别重复'+str(s['repetitions'])+'次，CUDA前后同步，报告中位数。计时含设置、求解、诊断及原场主机导出，不含绘图。两种硬件使用同一算法和参数；环形管统一使用torch CSR/Jacobi CG，另有SciPy CPU正式基准。GPU求解器device实际记录为cuda，不作CPU回退。显存为torch峰值allocated，不是整卡/NVIDIA上下文或总系统RSS。\n\nSA项只验证五次生产迭代，不冒充稳态物理benchmark。壁面函数项为65,536个本构反解，不是CFD流场。小网格/float64/同步与导出开销可能使GPU更慢，全部实测结果保留。\n\n| 案例 | CPU中位s | CUDA中位s | CPU/CUDA | CUDA峰值MiB | 数值一致 | 物理通过 |\n|---|---:|---:|---:|---:|---|---|\n'
    for r in s['cases']:text+=f"| {r['name']} | {r['cpu_median_s']:.6f} | {r['cuda_median_s']:.6f} | {r['cpu_over_gpu_median']:.3f} | {r['cuda_peak_allocated_bytes']/2**20:.2f} | {r['equivalent']} | {r['cuda_physical_passed']} |\n"
    text+='\n## 原场及限制\n\n每个目录包含CPU/CUDA完整末场与MAC/BGK接受状态、实际history、参数、时间样本及来源。CPU/GPU主字段相对L2差要求≤1e−6；剪切波零参考压力以事先指定动态压力0.745 Pa作RMS尺度，其他零场以1e−14尺度定义绝对差。几何/强制边界和原场独立审计同正式案例。实际加速只适用于本硬件、算法、参数与计时范围，不能推广为所有FVM/湍流/多GPU性能。\n\n![acceleration](figures/acceleration.png)\n';(output/'report.md').write_text(text)
    with PdfPages(output/'report.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11.69,8.27));ax.axis('off');ax.set_title('Measured CPU / CUDA correctness and acceleration',pad=20);table=[[r['name'],f"{r['cpu_median_s']:.5f}",f"{r['cuda_median_s']:.5f}",f"{r['cpu_over_gpu_median']:.3f}",str(r['equivalent']),str(r['cuda_physical_passed'])] for r in s['cases']];t=ax.table(cellText=table,colLabels=['Case','CPU s','CUDA s','CPU/CUDA','Equivalent','Physical pass'],loc='upper center');t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.8);ax.text(0,.45,'Float64; one CPU thread; actual algorithm warm-up; three synchronized repeated executions.\nRuntime includes setup/diagnostics/raw host export, excludes plots.\nAnnular CPU and CUDA use identical torch CSR/Jacobi algorithms.\nSA five-iteration comparison is not a converged physical benchmark.\nWall-law throughput is constitutive inversion, not a complete CFD solve.\nAllocated torch GPU peak is not whole-device memory. All slower CUDA results retained.',va='top',fontsize=9,linespacing=1.6);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.69,6));ax.imshow(plt.imread(figures/'acceleration.png'));ax.axis('off');pdf.savefig(fig);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--lbm-repo',type=Path);args=p.parse_args(argv)
    if args.repetitions<3:p.error('at least three repetitions required')
    if not torch.cuda.is_available():p.error('actual CUDA device required; no simulated fallback')
    if args.output.exists() and any(args.output.iterdir()):p.error('empty output required')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);prov=provenance();root=Path(__file__).resolve().parents[3]
    for name in ['src/tensorfvm/annular.py','src/tensorfvm/wall_functions.py','src/tensorfvm/solver3d.py']:prov['source_sha256'][name]=sha(root/name)
    s=dict(provenance=prov,repetitions=args.repetitions,hardware=dict(gpu_name=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,torch_cuda=torch.version.cuda,cpu_threads=torch.get_num_threads()),cases=[])
    plan=[(f'annular-{n}x{nr}',lambda device,n=n,nr=nr:annulus(device,n,nr)) for n,nr in [(256,64),(512,128),(1024,256)]]
    plan += [('poiseuille-72x24',channel),('taylor-green-64',lambda d:periodic(d,'taylor-green',64)),('abc-24-cubed',lambda d:periodic(d,'abc',24)),('advected-shear-32',lambda d:periodic(d,'advected-shear',32)),('wall-law-batch',wall_batch),('SA-128x112-five-iterations',sa_iterations)]
    if args.lbm_repo:plan.append(('TensorLBM-shear-64',lambda d:lbm_shear(d,args.lbm_repo)))
    for name,fn in plan:
        print(name,'CPU start',flush=True);cpu,ct,_=execute(fn,'cpu',args.repetitions);print(name,'CUDA start',flush=True);gpu,gt,peak=execute(fn,'cuda',args.repetitions);d=args.output/name;d.mkdir()
        deltas={}
        for key in cpu['primary_fields']:
            a,b=cpu['fields'][key],gpu['fields'][key]
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite or mismatched device fields')
            scale=cpu.get('field_comparison_scales',{}).get(key)
            denominator=np.sqrt(a.size)*scale if scale is not None else max(np.linalg.norm(a.ravel()),1e-14)
            deltas[key]=float(np.linalg.norm((a-b).ravel())/denominator)
        equivalent=all(v<=1e-6 for v in deltas.values());expected_device=gpu['actual_solver_device']
        if not expected_device.startswith('cuda'):raise ValueError('silent CPU fallback')
        for label,raw in [('cpu',cpu),('cuda',gpu)]:
            fields=raw.pop('fields');np.savez_compressed(d/(label+'-fields.npz'),**fields);write_json(d/(label+'-result.json'),raw)
        row=dict(name=name,cpu_samples_s=ct,cuda_samples_s=gt,cpu_median_s=statistics.median(ct),cuda_median_s=statistics.median(gt),cpu_over_gpu_median=statistics.median(ct)/statistics.median(gt),cuda_peak_allocated_bytes=peak,equivalent=equivalent,relative_field_differences=deltas,cpu_physical_passed=cpu['physical_passed'],cuda_physical_passed=gpu['physical_passed'],cpu_constitutive_passed=cpu.get('constitutive_passed'),cuda_constitutive_passed=gpu.get('constitutive_passed'),field_comparison_scales=cpu.get('field_comparison_scales',{}),actual_cuda_solver_device=expected_device)
        s['cases'].append(row);write_json(d/'comparison.json',row);write_json(args.output/'execution-status.json',s);print(name,'equivalent',equivalent,'CPU/GPU',row['cpu_over_gpu_median'],flush=True)
    s['correctness_passed']=all(r['equivalent'] and r['cpu_physical_passed'] is not False and r['cuda_physical_passed'] is not False and r['cpu_constitutive_passed'] is not False and r['cuda_constitutive_passed'] is not False for r in s['cases']);write_json(args.output/'summary.json',s);publish(args.output,s);write_json(args.output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(args.output)));return 0 if s['correctness_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
