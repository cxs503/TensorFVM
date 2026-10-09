import json,tempfile,hashlib
from pathlib import Path
import torch
from tensorfvm.conservative3d import ConservativeCylinder3DSolver
from tensorfvm.solver3d import Cylinder3DConfig
from tensorfvm.runtime import DistributedRuntime

def perturb(s):
    m=s.mesh;c=s.config
    z=(torch.arange(s.partition.start,s.partition.stop,dtype=torch.float64)+.5)*m.dz
    phase=2*torch.pi*z/c.span
    s.velocity[...,0]+=.02*torch.sin(phase)[:,None,None]
    s.velocity[...,1]+=.01*torch.cos(phase)[:,None,None]
    s.velocity.masked_fill_(~m.fluid[...,None],0)
    s.face_velocity=s._predict_faces(s.velocity)

def worker(rank,init,out):
    torch.distributed.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    try:
        c=Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,pressure_iterations=1000,pressure_relative_tolerance=1e-12,pressure_absolute_tolerance=1e-13,time_step=.001);s=ConservativeCylinder3DSolver(c,DistributedRuntime.discover('cpu'));perturb(s);s.solve();raw={'velocity':s.velocity,'pressure':s.pressure,'conv':s.last_transport['convective_flux'],'visc':s.last_transport['viscous_flux'],'faces':s.face_velocity};collected={}
        for name,value in raw.items():
            values=(value,) if isinstance(value,torch.Tensor) else value;parts=[]
            for t in values:
                recv=[torch.empty_like(t) for _ in range(2)];torch.distributed.all_gather(recv,t);parts.append(torch.cat(recv,axis=0))
            collected[name]=parts
        collected['history']=s.history;collected['body_force']=s.last_transport['body_force'];collected['body_moment']=s.last_transport['body_moment']
        if rank==0:torch.save(collected,out)
    finally:torch.distributed.destroy_process_group()
if __name__=='__main__':
    c=Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,pressure_iterations=1000,pressure_relative_tolerance=1e-12,pressure_absolute_tolerance=1e-13,time_step=.001);s=ConservativeCylinder3DSolver(c);perturb(s);s.solve();reference={'velocity':(s.velocity,),'pressure':(s.pressure,),'conv':s.last_transport['convective_flux'],'visc':s.last_transport['viscous_flux'],'faces':s.face_velocity};errors={}
    with tempfile.TemporaryDirectory(prefix='fvm-independent-distributed-') as tmp:
        torch.multiprocessing.spawn(worker,args=(str(Path(tmp)/'init'),str(Path(tmp)/'result.pt')),nprocs=2,join=True);r=torch.load(Path(tmp)/'result.pt',weights_only=True)
        for key,values in reference.items():
            error=max(float((a-b).abs().max()) for a,b in zip(values,r[key]));errors[key]=error
        for key in ('body_force','body_moment'):
            error=float((s.last_transport[key]-r[key]).abs().max());errors[key]=error
    report={'serial_vs_two_rank_errors':errors,'passed':all(v<1e-9 for v in errors.values()),'initial_state':'spanwise sine/cosine perturbations excite real z halos','physical_accuracy_qualified':False,'source_sha256':hashlib.sha256(Path('src/tensorfvm/conservative3d.py').read_bytes()).hexdigest()};Path('/tmp/tensorfvm-conservative-distributed-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
