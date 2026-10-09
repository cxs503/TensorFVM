"""Real opt-in conservative momentum cases plus full-stress manufactured refinement."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import torch
from tensorfvm.conservative3d import ConservativeCylinder3DSolver
from tensorfvm.solver3d import Cylinder3DConfig
from tensorfvm.benchmark_cylinder3d import export_result,smoke_qualified


def convert(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [convert(v) for v in value]
    return value


def manufactured(n):
    s=ConservativeCylinder3DSolver(Cylinder3DConfig(nx=n,ny=n,nz=4,max_steps=1))
    x,y,z=s.mesh.centers();a=2*math.pi/s.config.length;b=2*math.pi/s.config.height;cc=2*a
    u=torch.sin(a*x)*torch.sin(b*y);v=torch.sin(cc*x)*torch.cos(b*y)
    velocity=torch.stack((u,v,torch.zeros_like(u)),-1);mu=1+.2*torch.sin(a*x);mux=.2*a*torch.cos(a*x)
    exact=torch.stack((2*mux*a*torch.cos(a*x)*torch.sin(b*y)+mu*(-(2*a*a+b*b)*u-cc*b*torch.cos(cc*x)*torch.sin(b*y)),
                      mux*(b*torch.sin(a*x)*torch.cos(b*y)+cc*torch.cos(cc*x)*torch.cos(b*y))
                      +mu*(a*b*torch.cos(a*x)*torch.cos(b*y)-(cc*cc+2*b*b)*v),torch.zeros_like(u)),-1)
    actual=s.vector_divergence(s.stress_fluxes(velocity,mu))
    mask=(x>10)&(x<15)&(y>3)&(y<8)
    error=float(torch.linalg.vector_norm((actual-exact)[mask])/torch.linalg.vector_norm(exact[mask]))
    return dict(n=n,assessment_domain_m=[10,15,3,8],relative_l2_error=error,
                x=x[mask],y=y[mask],computed=actual[mask],analytic=exact[mask])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('docs/conservative-transport'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cases=[]
    configs=[('cpu','cpu',.005,50)]
    if torch.cuda.is_available():configs.append(('cuda','cuda:0',.005,50))
    for name,device,dt,steps in configs:
        s=ConservativeCylinder3DSolver(Cylinder3DConfig(nx=48,ny=32,nz=12,max_steps=steps,time_step=dt,device=device))
        ledgers=[]
        for _ in range(steps):
            s.step();ledgers.append(convert({k:v for k,v in s.last_transport.items() if not isinstance(v,(tuple,list)) and k not in ['old_velocity','tentative_velocity','dynamic_viscosity']}))
        result=s.solve();out=args.output/name;export_result(result,out)
        raw=dict(schema='tensorfvm.conservative-cell-transport/1',config=s.config.__dict__,units=dict(length='m',time='s',mass='kg',viscosity_note='synthetic Re3900 viscosity; not calibrated water'),
                 time_s=s.time,spacing_m=[s.mesh.dx,s.mesh.dy,s.mesh.dz],fluid=s.mesh.fluid,
                 velocity=s.velocity,pressure=s.pressure,face_velocity=s.face_velocity,
                 tentative_face_velocity=s.last_tentative_faces,face_dynamic_viscosity=s.last_face_dynamic_viscosity,
                 face_velocity_gradients=s.last_face_gradient,transport=s.last_transport,ledgers=ledgers,history=s.history,
                 physical_accuracy_qualified=False,full_mac_qualified=False)
        data=json.dumps(convert(raw),separators=(',',':'),allow_nan=False).encode()
        (out/'transport-raw.json.gz').write_bytes(gzip.compress(data,mtime=0))
        wall=dict(schema='tensorfvm.actual-wall-facets/1',time_s=s.time,viscous_evaluation_time_s=s.time-dt,
                  pressure_evaluation_time_s=s.time,config=s.config.__dict__,units=raw['units'],
                  force_owner='fluid-only; actual traction from conservative step',facets=s.last_transport['wall_facets'],
                  force_on_body_N=s.last_transport['body_force'],moment_on_body_about_origin_Nm=s.last_transport['body_moment'],physical_accuracy_qualified=False)
        (out/'wall-facets.json').write_text(json.dumps(convert(wall),indent=2,allow_nan=False)+'\n')
        report=dict(case=name,discrete_projection_passed=smoke_qualified(s.history),
            max_momentum_residual_Ns=max(h['momentum_ledger_residual_Ns'] for h in s.history),
            max_boundary_momentum_residual_Ns=max(h['boundary_momentum_residual_Ns'] for h in s.history),
            max_continuity=max(h['continuity'] for h in s.history),
            max_center_divergence=max(h['center_velocity_divergence'] for h in s.history),
            max_hypothetical_reconstruction_defect_Ns=max(h['hypothetical_reconstruction_defect_Ns'] for h in s.history),
            cell_incompressibility_qualified=False,physical_accuracy_qualified=False)
        cases.append(report);print(json.dumps(report),flush=True)
    mms=[manufactured(n) for n in (32,64,128)]
    (args.output/'manufactured.json').write_text(json.dumps(convert(mms),indent=2)+'\n')
    root=Path(__file__).resolve().parents[1]
    sources=['src/tensorfvm/conservative3d.py','src/tensorfvm/solver3d.py','src/tensorfvm/benchmark_cylinder3d.py','src/tensorfvm/runtime.py','src/tensorfvm/mesh3d.py','scripts/run_conservative_transport.py']
    summary=dict(schema='tensorfvm.conservative-transport-study/1',cases=cases,mms_relative_l2_errors=[r['relative_l2_error'] for r in mms],
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        artifacts_sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob('*')) if p.is_file() and p.name not in ['study.json','README.md']},
        physical_accuracy_qualified=False,full_mac_qualified=False)
    (args.output/'study.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
