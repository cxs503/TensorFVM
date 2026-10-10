#!/usr/bin/env python3
"""Run the real body-fitted 3-D Re3900 case and archive replayable evidence."""
import argparse,csv,hashlib,json,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tensorfvm.solver3d import Cylinder3DConfig
from tensorfvm.body_fitted_transient3d import BodyFittedTransient3D,TransientSettings
from tensorfvm.cylinder3900_statistics import statistics
from tensorfvm.cylinder3900_checkpoint import FieldMoments,save_checkpoint,restore_checkpoint

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialize-from',type=Path);p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--output',type=Path,required=True);p.add_argument('--nx',type=int,default=48);p.add_argument('--ny',type=int,default=24);p.add_argument('--nz',type=int,default=12)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--dt',type=float,default=.002);p.add_argument('--stretching',type=float,default=3.5)
    p.add_argument('--sgs',choices=['wale','none'],default='wale');p.add_argument('--upwind-fraction',type=float,default=.1)
    p.add_argument('--discard-time',type=float,default=100.);p.add_argument('--minimum-statistics-duration',type=float,default=500.)
    a=p.parse_args();out=a.output
    if a.checkpoint_every<1 or a.discard_time<0 or a.minimum_statistics_duration<=0:p.error('checkpoint interval and statistics duration must be positive, discard time nonnegative')
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    c=Cylinder3DConfig(nx=a.nx,ny=a.ny,nz=a.nz,mesh_type='body-fitted',body_fitted_stretching=a.stretching,time_step=a.dt,max_steps=a.steps,span=np.pi)
    settings=TransientSettings(sgs=a.sgs,upwind_fraction=a.upwind_fraction)
    repo=Path(__file__).resolve().parents[1];sources=[Path(__file__).resolve()]+[repo/'src/tensorfvm'/f for f in ('body_fitted_transient3d.py','cylinder3900_statistics.py','cylinder3900_checkpoint.py','body_fitted3d.py','body_fitted.py','solver3d.py','backend_registry.py')]
    archive=out/'source-snapshot';archive.mkdir();bindings={}
    for src in sources:
        target=archive/src.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);bindings[str(src.relative_to(repo))]=sha(target)
    dump(out/'config.json',dict(config=asdict(c),settings=asdict(settings),sources=bindings))
    s=BodyFittedTransient3D(c,settings);moments=FieldMoments(s,a.discard_time)
    if a.initialize_from:restore_checkpoint(a.initialize_from,s,moments,bindings)
    start=time.perf_counter();failure=None
    try:
        for i in range(a.steps):
            h=s.step()
            if s.time>=a.discard_time:moments.add(s)
            if (i+1)%a.checkpoint_every==0:save_checkpoint(out/'checkpoint.npz',s,moments,bindings)
            if (i+1)%50==0:print(json.dumps(h),flush=True)
    except (RuntimeError,ValueError) as e:failure=str(e)
    save_checkpoint(out/'checkpoint.npz',s,moments,bindings)
    dump(out/'history.json',s.history);dump(out/'forces.json',s.forces)
    st=statistics(s.forces,c.diameter,c.inlet_velocity,a.discard_time,a.minimum_statistics_duration)
    dump(out/'statistics.json',st)
    with (out/'wall-pressure.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['x','y','z','theta_degrees','pressure','Cp','area_x','area_y','area_z'])
        for j in torch.where(s.wall)[0]:
            xyz=s.fc[j];theta=np.degrees(np.arctan2(float(xyz[1])-c.cylinder_y,float(xyz[0])-c.cylinder_x))%360
            w.writerow([*xyz.tolist(),theta,float(s.pressure[s.owner[j]]),2*float(s.pressure[s.owner[j]])/(c.density*c.inlet_velocity**2),*s.area[j].tolist()])
    _,visc,_=s.transport(s.velocity,s.flux)
    normal=s.area[s.wall]/s.area_mag[s.wall,None]
    traction=visc[s.wall]/s.area_mag[s.wall,None]
    tangent=traction-(traction*normal).sum(1)[:,None]*normal
    utau=torch.sqrt(torch.linalg.vector_norm(tangent,dim=1)/c.density)
    y=(s.owner_offset[s.wall]*normal).sum(1).abs()
    yp=y*utau/c.kinematic_viscosity
    wall_resolution=dict(min_wall_distance=float(y.min()),max_wall_distance=float(y.max()),yplus_max=float(yp.max()),yplus_95=float(torch.quantile(yp,.95)),wall_resolved_target_yplus=1.,wall_resolution_passed=bool((yp<=1).all()),state='instantaneous startup diagnostic, not time-averaged qualification')
    summary=dict(wall_resolution=wall_resolution,case='3d-body-fitted-cylinder-Re3900',Re=c.reynolds,mesh_cells=s.N,steps=len(s.history),field_statistical_samples=moments.count,wall_time=time.perf_counter()-start,
                 final=s.history[-1]if s.history else None,failure=failure,numerical_short_run_passed=bool(s.history and failure is None and all(h['continuity']<settings.pressure_tolerance and h['cfl']<1 and h['momentum_ledger']<1e-10 for h in s.history)),physical_accuracy_qualified=False,
                 force=s.forces[-1]if s.forces else None,statistics=st,
                 blockers=['short startup record; no statistically stationary LES established','grid/time/domain/span sensitivity not established','reference Cp and velocity raw data not yet ingested','CPU only; CUDA pressure solver and multi-rank geometry kernel pending'],
                 discretization='single-rank CPU shared-face FV; forward Euler; central/upwind blend; LS full deviatoric stress; WALE; nonorthogonal face projection; collocated cell pressure correction')
    dump(out/'summary.json',summary)
    bindings_artifact={str(f.relative_to(out)):sha(f)for f in out.rglob('*')if f.is_file()}
    dump(out/'manifest.json',dict(schema='tensorfvm.body-fitted-transient3d/1',source_sha256=bindings,artifact_sha256=bindings_artifact))
    print(json.dumps(summary,indent=2));return 0 if failure is None else 2
if __name__=='__main__':raise SystemExit(main())
