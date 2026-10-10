#!/usr/bin/env python3
"""Build real Re3900 meshes and inspect wall metrics without pressure LU."""
import argparse,gc,hashlib,json,os,platform,resource,shutil,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy
import torch
from tensorfvm.body_fitted_transient3d import BodyFittedTransient3D,TransientSettings
from tensorfvm.solver3d import Cylinder3DConfig

class GeometryProbe(BodyFittedTransient3D):
    def _build_pressure(self):
        # Intentionally skip both global pressure CSR and LU. All other native
        # face geometry and LS/WALE operators are the current real kernel.
        self.pressure_matrix=None;self.pressure_lu=None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',default='128,64,24,6;128,64,24,7;128,64,24,8;192,96,32,7;192,96,32,8;64,32,12,5.5;64,32,12,7;64,32,12,8')
    a=p.parse_args();out=a.output
    if out.exists()and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);repo=Path(__file__).resolve().parents[1]
    sources=sorted(set([Path(__file__).resolve()]+[Path(m.__file__).resolve()for name,m in list(sys.modules.items())if(name=='tensorfvm'or name.startswith('tensorfvm.'))and getattr(m,'__file__',None)and Path(m.__file__).suffix=='.py']))
    bindings={}
    for src in sources:
        dest=out/'source-snapshot'/src.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);bindings[str(src.relative_to(repo))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    env=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),torch_threads=torch.get_num_threads(),threads={k:os.environ.get(k)for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),device='cpu')
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n');results=[]
    for case in a.cases.split(';'):
        nx,ny,nz,stretch=map(float,case.split(','));nx,ny,nz=map(int,(nx,ny,nz));t0=time.perf_counter()
        c=Cylinder3DConfig(nx=nx,ny=ny,nz=nz,mesh_type='body-fitted',body_fitted_stretching=stretch,length=24.,height=20.,cylinder_x=8.,cylinder_y=10.,span=np.pi,time_step=.002,max_steps=1)
        s=GeometryProbe(c,TransientSettings());setup=time.perf_counter()-t0
        t1=time.perf_counter();_,visc,nu=s.transport(s.velocity,s.flux)
        normal=s.area[s.wall]/s.area_mag[s.wall,None];y=(s.owner_offset[s.wall]*normal).sum(1).abs()
        traction=visc[s.wall]/s.area_mag[s.wall,None];tangent=traction-(traction*normal).sum(1)[:,None]*normal
        yp=y*torch.sqrt(torch.linalg.vector_norm(tangent,dim=1)/c.density)/c.kinematic_viscosity
        face_nu=.5*(nu[s.owner]+nu[s.safe]);face_nu[s.wall]=c.kinematic_viscosity;face_nu[s.outlet]=0
        rate=torch.zeros(s.N,dtype=torch.float64);fr=face_nu*s.coeff
        rate.index_add_(0,s.owner,fr);rate.index_add_(0,s.neighbor[s.interior],fr[s.interior]);diff_rate=float((2*rate/s.volumes).max())
        adv=torch.zeros_like(rate);adv.index_add_(0,s.owner,s.flux.abs());adv.index_add_(0,s.neighbor[s.interior],s.flux[s.interior].abs());adv_rate=float((adv/(2*s.volumes)).max())
        cross_wall=s.mesh.cylinder_face_vertices[0];chord=torch.linalg.vector_norm(cross_wall[:,1,:2]-cross_wall[:,0,:2],dim=1);angle=2*torch.asin(chord/(2*c.cylinder_radius))
        x=s.centers[:,0]-c.cylinder_x;yrel=s.centers[:,1]-c.cylinder_y;wake=(x>0)&(x<5)&(yrel.abs()<1)
        h=s.volumes[wake].pow(1/3)
        # Native tensor storage only, deduplicated by underlying storage pointer.
        storages={}
        for obj in (s,s.mesh):
            for val in vars(obj).values():
                if isinstance(val,torch.Tensor):storages[val.untyped_storage().data_ptr()]=val.untyped_storage().nbytes()
        row=dict(config=asdict(c),cells=s.N,pressure_matrix_built=False,pressure_factorization_built=False,geometry_setup_s=setup,transport_probe_s=time.perf_counter()-t1,
                 minimum_volume=float(s.volumes.min()),closed_cell_area_max=float(s.integrate(s.area).abs().max()),native_tensor_storage_bytes=sum(storages.values()),
                 cumulative_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                 wall_distance_D=dict(min=float(y.min()),max=float(y.max()),p95=float(torch.quantile(y,.95))),spanwise_spacing_D=c.span/nz,
                 cylinder_delta_theta_radians=dict(min=float(angle.min()),max=float(angle.max())),cylinder_arc_spacing_D=dict(min=float((angle*c.cylinder_radius).min()),max=float((angle*c.cylinder_radius).max())),
                 initial_yplus=dict(max=float(yp.max()),p95=float(torch.quantile(yp,.95)),state='unprojected homogeneous startup with imposed no-slip: not time-averaged wall qualification'),
                 initial_nu_total=dict(min=float(nu.min()),max=float(nu.max()),molecular=c.kinematic_viscosity),
                 explicit_diffusion_rate_per_time=diff_rate,explicit_diffusion_dt_screen_limit=.5/diff_rate,advective_dt_cfl_one=1/adv_rate,
                 recommended_initial_dt_with_20_percent_screen_margin=min(.5/diff_rate,1/adv_rate)*.2,
                 wake_region_volume_length_D=dict(min=float(h.min()),p50=float(torch.quantile(h,.5)),p95=float(torch.quantile(h,.95))),
                 physical_accuracy_qualified=False)
        results.append(row);(out/'probes.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
        del s,visc,nu,normal,y,traction,tangent,yp,face_nu,rate,fr,adv,cross_wall,chord,angle,x,yrel,wake,h;gc.collect()
    artifacts={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest()for f in out.rglob('*')if f.is_file()}
    (out/'manifest.json').write_text(json.dumps(dict(schema='tensorfvm.cylinder3900.geometry-preflight/1',source_sha256=bindings,artifact_sha256=artifacts),indent=2)+'\n')
if __name__=='__main__':main()
