"""Read-only actual solver exercise plus independent NumPy face operators."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from tensorfvm.solver3d import Cylinder3DConfig,Cylinder3DSolver

def np_grad(p,fluid,dx,dy,dz):
    p=np.where(fluid,p,0);gx=np.zeros((*p.shape[:-1],p.shape[-1]+1));gy=np.zeros((p.shape[0],p.shape[1]+1,p.shape[2]));mx=fluid[...,:-1]&fluid[...,1:];my=fluid[:,:-1]&fluid[:,1:];mz=fluid&np.roll(fluid,-1,axis=0)
    gx[...,1:-1]=(p[...,1:]-p[...,:-1])/dx*mx;gx[...,-1]=-2*p[...,-1]/dx;gy[:,1:-1]=(p[:,1:]-p[:,:-1])/dy*my;gz=(np.roll(p,-1,axis=0)-p)/dz*mz
    return gx,gy,gz

def np_div(faces,fluid,dx,dy,dz):
    x,y,z=faces;return np.where(fluid,(x[...,1:]-x[...,:-1])/dx+(y[:,1:]-y[:,:-1])/dy+(z-np.roll(z,1,axis=0))/dz,0)

def array(x):return x.detach().cpu().numpy()
def exercise(device):
    c=Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,time_step=.001,pressure_iterations=500,device=device);s=Cylinder3DSolver(c);fluid=array(s.mesh.fluid);spacing=(s.mesh.dx,s.mesh.dy,s.mesh.dz);rng=np.random.default_rng(724);rows=[]
    for label,p in [('random',rng.normal(size=fluid.shape)),('checkerboard',(-1.)**np.indices(fluid.shape).sum(0))]:
        p=np.where(fluid,p,0);pt=torch.tensor(p,dtype=torch.float64,device=device);grad=np_grad(p,fluid,*spacing);actual=tuple(array(g) for g in s._face_gradient(pt));op=-np_div(grad,fluid,*spacing);image=array(s._pressure_operator(pt));error=max(float(np.max(abs(g-a))) for g,a in zip(grad,actual));operror=float(np.max(abs(op-image)));assert error<1e-12 and operror<1e-12;assert np.linalg.norm(image)>0;rows.append(dict(case=label,gradient_error=error,operator_error=operror,operator_norm=float(np.linalg.norm(image))))
    initial=sum(array(s.face_velocity[0])[...,-1].ravel())-sum(array(s.face_velocity[0])[...,0].ravel());steps=[]
    for _ in range(3):
        record=s.step();faces=tuple(array(f) for f in s.face_velocity);div=np_div(faces,fluid,*spacing);actual=array(s._face_divergence(s.face_velocity));g=np_grad(array(s.pressure),fluid,*spacing);pred=tuple(array(t) for t in s.last_tentative_faces);correction_error=max(float(np.max(abs(after-before+c.time_step/c.density*grad))) for after,before,grad in zip(faces,pred,g));assert correction_error<1e-12;assert np.max(abs(div-actual))<1e-12
        x,y,z=faces;mx=fluid[...,:-1]&fluid[...,1:];my=fluid[:,:-1]&fluid[:,1:];mz=fluid&np.roll(fluid,-1,axis=0);wall=max(np.max(abs(x[...,1:-1][~mx]),initial=0),np.max(abs(y[:,1:-1][~my]),initial=0),np.max(abs(z[~mz]),initial=0));assert wall==0
        globalflux=(x[...,-1].sum()-x[...,0].sum())*spacing[1]*spacing[2]+(y[:,-1].sum()-y[:,0].sum())*spacing[0]*spacing[2];integrated=div.sum()*np.prod(spacing);assert abs(globalflux-integrated)<1e-12
        continuity=float(np.max(abs(div)))*c.diameter/c.inlet_velocity;assert abs(continuity-record['continuity'])<1e-12;assert continuity<1e-6;assert record['pressure_converged']==1;steps.append(dict(time=record['time'],continuity=continuity,solid_face_flux_max=wall,face_correction_error=correction_error,global_flux=globalflux,integrated_divergence=integrated))
    np.savez('/tmp/tensorfvm-projection-raw-'+device+'.npz',pressure=array(s.pressure),fluid=fluid,face_u=faces[0],face_v=faces[1],face_w=faces[2],tentative_u=pred[0],tentative_v=pred[1],tentative_w=pred[2],spacing=spacing,dt=c.time_step,density=c.density)
    return dict(device=device,operators=rows,steps=steps)

if __name__=='__main__':
    import inspect
    src=Path(inspect.getfile(Cylinder3DSolver));result=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),cases=[exercise('cpu')],physical_accuracy_qualified=False)
    if torch.cuda.is_available():result['cases'].append(exercise('cuda'))
    else:result['gpu_validation']='unavailable; not executed'
    Path('/tmp/tensorfvm-projection-independent-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
