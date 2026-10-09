"""Actual periodic MAC momentum/energy, variable-mu MMS and Taylor-Green refinement."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import torch
from tensorfvm.periodic_mac import PeriodicMACSolver,PeriodicMACConfig


def jsonable(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
    return v


def taylor_green(s):
    x,y,z=s.face_coordinates(0);s.velocity[0]=torch.sin(x)*torch.cos(y)
    x,y,z=s.face_coordinates(1);s.velocity[1]=-torch.cos(x)*torch.sin(y)
    s.velocity[2].zero_()
    return s.velocity.clone()


def random_curl(s,amplitude=.1):
    gen=torch.Generator(device='cpu').manual_seed(741)
    potential=torch.randn((3,*s.shape),generator=gen,dtype=torch.float64).to(s.device)
    def d(f,a):return (f-s.previous(f,a))/s.spacing[a]
    velocity=torch.stack((d(potential[2],1)-d(potential[1],2),d(potential[0],2)-d(potential[2],0),d(potential[1],0)-d(potential[0],1)))
    s.velocity=amplitude*velocity/velocity.abs().max()
    x,y,z=s.centers();s.dynamic_viscosity=.02*(1+.2*torch.sin(x)+.1*torch.cos(z))
    return s.velocity.clone()


def manufactured(n):
    s=PeriodicMACSolver(PeriodicMACConfig(nx=n,ny=n,nz=n))
    x,y,z=s.centers();mu=1+.2*torch.sin(x)+.1*torch.cos(z)
    for a in (0,1):
        x,y,z=s.face_coordinates(a)
        s.velocity[a]=torch.sin(x)*torch.cos(y)*torch.cos(z) if a==0 else -torch.cos(x)*torch.sin(y)*torch.cos(z)
    exact=[]
    for a in range(3):
        x,y,z=s.face_coordinates(a);mf=1+.2*torch.sin(x)+.1*torch.cos(z)
        if a==0:r=-3*mf*torch.sin(x)*torch.cos(y)*torch.cos(z)+.4*torch.cos(x)**2*torch.cos(y)*torch.cos(z)+.1*torch.sin(x)*torch.cos(y)*torch.sin(z)**2
        elif a==1:r=3*mf*torch.cos(x)*torch.sin(y)*torch.cos(z)-.1*torch.cos(x)*torch.sin(y)*torch.sin(z)**2
        else:r=-.2*torch.cos(x)*torch.sin(x)*torch.cos(y)*torch.sin(z)
        exact.append(r)
    exact=torch.stack(exact);stress=s.full_stress(s.velocity,mu);computed=s.stress_divergence(stress)
    error=float(torch.linalg.vector_norm(computed-exact)/torch.linalg.vector_norm(exact))
    return dict(n=n,config=s.config.__dict__,velocity=s.velocity,dynamic_viscosity=mu,normal_stress=stress[0],shear_stress=stress[1],edge_dynamic_viscosity=stress[2],computed_force_density=computed,analytic_force_density=exact,relative_l2_error=error)


def write_gzip(path,value):path.write_bytes(gzip.compress(json.dumps(jsonable(value),separators=(',',':'),allow_nan=False).encode(),mtime=0))


def advance(s,steps):
    s.accepted_face_velocities=[]
    for _ in range(steps):
        s.step();s.accepted_face_velocities.append(s.velocity.clone())


def raw(s,initial):
    return dict(schema='tensorfvm.periodic-mac-case/1',config=s.config.__dict__,spacing_m=s.spacing,
                initial_velocity=initial,final_velocity=s.velocity,history=s.history,last_step=s.last_raw,
                accepted_face_velocities=getattr(s,'accepted_face_velocities',[]),
                distributed_supported=False,physical_accuracy_qualified=False,
                scope='closed periodic MAC numerical momentum/energy; not cylinder, LES or FSI')


def numerical_qualified(history):
    """Fixed numerical gates for this closed periodic MAC prototype only."""
    return bool(history and all(
        h['nonlinear_converged'] and h['nonlinear_residual_m_s'] <= 1e-12
        and h['max_divergence_s_inv'] < 1e-10
        and max(abs(x) for x in h['momentum_change_Ns']) < 1e-10
        and abs(h['energy_balance_defect_J']) < 1e-10*max(1.,h['kinetic_before_J'])
        and h['kinetic_after_J'] <= h['kinetic_before_J']+1e-12*max(1.,h['kinetic_before_J'])
        and abs(h['convective_power_W']) < 1e-10 and h['viscous_power_W'] <= 1e-12
        and h['projection']['pressure_converged']
        and h['projection']['projected_energy_J'] <= h['projection']['tentative_energy_J']+1e-12*max(1.,h['projection']['tentative_energy_J'])
        for h in history))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('docs/periodic-mac'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cases=[]
    for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
        s=PeriodicMACSolver(PeriodicMACConfig(nx=12,ny=12,nz=12,device=device,time_step_s=.002))
        initial=random_curl(s)
        advance(s,10)
        name='random-'+device.replace(':','-');write_gzip(args.output/(name+'.json.gz'),raw(s,initial))
        r=dict(case=name,max_energy_defect_J=max(abs(h['energy_balance_defect_J']) for h in s.history),
               max_divergence_s_inv=max(h['max_divergence_s_inv'] for h in s.history),
               max_momentum_change_Ns=max(max(abs(v) for v in h['momentum_change_Ns']) for h in s.history),
               max_convective_power_W=max(abs(h['convective_power_W']) for h in s.history),
               total_energy_change_J=s.history[-1]['kinetic_after_J']-s.history[0]['kinetic_before_J'],
               nonlinear_all_converged=all(h['nonlinear_converged'] for h in s.history),
               numerical_passed=numerical_qualified(s.history))
        cases.append(r);print(json.dumps(r),flush=True)
    grids=[]
    for n in (8,16,32):
        s=PeriodicMACSolver(PeriodicMACConfig(nx=n,ny=n,nz=4,time_step_s=.0125,viscosity_pa_s=.1))
        initial=taylor_green(s)
        advance(s,40)
        analytic=initial*math.exp(-2*.1*.5)
        error=float(torch.linalg.vector_norm(s.velocity-analytic)/torch.linalg.vector_norm(analytic))
        name=f'tg-grid-{n}';write_gzip(args.output/(name+'.json.gz'),dict(raw(s,initial),analytic_final_velocity=analytic,relative_l2_error=error))
        grids.append(dict(n=n,relative_l2_error=error,error_3_percent_passed=error<.03,numerical_passed=numerical_qualified(s.history)))
    times=[]
    for dt in (.1,.05,.025):
        s=PeriodicMACSolver(PeriodicMACConfig(nx=16,ny=16,nz=4,time_step_s=dt,viscosity_pa_s=.1))
        initial=taylor_green(s)
        advance(s,round(1/dt))
        eigenvalue=sum(4*math.sin(s.spacing[a]/2)**2/s.spacing[a]**2 for a in (0,1))
        reference=initial*math.exp(-.1*eigenvalue)
        error=float(torch.linalg.vector_norm(s.velocity-reference)/torch.linalg.vector_norm(reference))
        name=f'tg-time-{dt}';write_gzip(args.output/(name+'.json.gz'),dict(raw(s,initial),semi_discrete_reference=reference,relative_l2_error=error,
            reference_note='fixed-grid Taylor-Green discrete Laplacian eigenmode; isolates midpoint time error'))
        times.append(dict(dt_s=dt,relative_l2_error=error,error_3_percent_passed=error<.03,numerical_passed=numerical_qualified(s.history)))
    mms=[]
    for n in (8,16,32):
        case=manufactured(n);write_gzip(args.output/f'mms-{n}.json.gz',case);mms.append(dict(n=n,relative_l2_error=case['relative_l2_error'],error_3_percent_passed=case['relative_l2_error']<.03))
    # Deliberately inadequate nonlinear budget: rejected candidate never advances state.
    s=PeriodicMACSolver(PeriodicMACConfig(nx=12,ny=12,nz=12,time_step_s=.01,nonlinear_iterations=4))
    initial=random_curl(s)
    try:s.step()
    except RuntimeError:pass
    else:raise AssertionError('negative-control Picard budget unexpectedly converged')
    write_gzip(args.output/'nonlinear-rejected.json.gz',raw(s,initial))
    if not torch.equal(initial,s.velocity) or s.time!=0 or s.history:raise AssertionError('failed step changed state')
    # A second rejection has a genuinely excessive true nonlinear residual.
    hard=PeriodicMACSolver(PeriodicMACConfig(nx=12,ny=12,nz=12,time_step_s=.5,nonlinear_iterations=4))
    hard_initial=random_curl(hard)
    try:hard.step()
    except RuntimeError:pass
    else:raise AssertionError('hard nonlinear negative control unexpectedly converged')
    if hard.last_raw['record']['nonlinear_residual_m_s']<=hard.config.nonlinear_tolerance:raise AssertionError('hard case true residual was not excessive')
    write_gzip(args.output/'nonlinear-hard-rejected.json.gz',raw(hard,hard_initial))
    if not torch.equal(hard_initial,hard.velocity) or hard.time!=0 or hard.history:raise AssertionError('hard failed step changed state')
    spatial_order=[math.log(grids[i]['relative_l2_error']/grids[i+1]['relative_l2_error'],2) for i in range(2)]
    temporal_order=[math.log(times[i]['relative_l2_error']/times[i+1]['relative_l2_error'],2) for i in range(2)]
    mms_order=[math.log(mms[i]['relative_l2_error']/mms[i+1]['relative_l2_error'],2) for i in range(2)]
    root=Path(__file__).resolve().parents[1];sources=['src/tensorfvm/periodic_mac.py','src/tensorfvm/runtime.py','scripts/run_periodic_mac.py']
    summary=dict(schema='tensorfvm.periodic-mac-study/1',cases=cases,spatial_refinement=grids,temporal_refinement=times,variable_mu_mms=mms,
        nonlinear_failure_rejected=True,
        rejection_true_residuals_m_s=[s.last_raw['record']['nonlinear_residual_m_s'],hard.last_raw['record']['nonlinear_residual_m_s']],
        spatial_order=spatial_order,temporal_order=temporal_order,mms_order=mms_order,
        refinement_order_passed=all(1.8<o<2.2 for o in spatial_order+temporal_order+mms_order),
        mms_all_grids_3_percent_passed=all(c['error_3_percent_passed'] for c in mms),
        mms_finest_grid_3_percent_passed=mms[-1]['error_3_percent_passed'],
        distributed_supported=False,physical_accuracy_qualified=False,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        artifacts_sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('*.json.gz'))})
    (args.output/'study.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
