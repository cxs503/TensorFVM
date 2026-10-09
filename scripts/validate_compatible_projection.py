"""Read-only actual solver exercise plus independent NumPy face operators."""
import json,hashlib,argparse
from dataclasses import asdict
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
def exercise(device,output):
    c=Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,time_step=.001,pressure_iterations=500,device=device);s=Cylinder3DSolver(c);fluid=array(s.mesh.fluid);spacing=(s.mesh.dx,s.mesh.dy,s.mesh.dz);rng=np.random.default_rng(724);rows=[]
    for label,p in [('random',rng.normal(size=fluid.shape)),('checkerboard',(-1.)**np.indices(fluid.shape).sum(0))]:
        p=np.where(fluid,p,0);pt=torch.tensor(p,dtype=torch.float64,device=device);grad=np_grad(p,fluid,*spacing);actual=tuple(array(g) for g in s._face_gradient(pt));op=-np_div(grad,fluid,*spacing);image=array(s._pressure_operator(pt));error=max(float(np.max(abs(g-a))) for g,a in zip(grad,actual));operror=float(np.max(abs(op-image)));assert error<1e-12 and operror<1e-12;assert np.linalg.norm(image)>0;rows.append(dict(case=label,gradient_error=error,operator_error=operror,operator_norm=float(np.linalg.norm(image))))
    initial=sum(array(s.face_velocity[0])[...,-1].ravel())-sum(array(s.face_velocity[0])[...,0].ravel());steps=[]
    for _ in range(3):
        record=s.step();faces=tuple(array(f) for f in s.face_velocity);div=np_div(faces,fluid,*spacing);actual=array(s._face_divergence(s.face_velocity));g=np_grad(array(s.pressure),fluid,*spacing);pred=tuple(array(t) for t in s.last_tentative_faces);correction_error=max(float(np.max(abs(after-before+c.time_step/c.density*grad))) for after,before,grad in zip(faces,pred,g));assert correction_error<1e-12;assert np.max(abs(div-actual))<1e-12
        x,y,z=faces;mx=fluid[...,:-1]&fluid[...,1:];my=fluid[:,:-1]&fluid[:,1:];mz=fluid&np.roll(fluid,-1,axis=0);wall=max(np.max(abs(x[...,1:-1][~mx]),initial=0),np.max(abs(y[:,1:-1][~my]),initial=0),np.max(abs(z[~mz]),initial=0));assert wall==0
        globalflux=(x[...,-1].sum()-x[...,0].sum())*spacing[1]*spacing[2]+(y[:,-1].sum()-y[:,0].sum())*spacing[0]*spacing[2];integrated=div.sum()*np.prod(spacing);assert abs(globalflux-integrated)<1e-12
        continuity=float(np.max(abs(div)))*c.diameter/c.inlet_velocity;assert abs(continuity-record['continuity'])<1e-12;assert continuity<1e-6;assert record['pressure_converged']==1;steps.append(dict(time=record['time'],continuity=continuity,solid_face_flux_max=wall,face_correction_error=correction_error,global_flux=globalflux,integrated_divergence=integrated))
    np.savez(output/('projection-raw-'+device+'.npz'),pressure=array(s.pressure),fluid=fluid,face_u=faces[0],face_v=faces[1],face_w=faces[2],tentative_u=pred[0],tentative_v=pred[1],tentative_w=pred[2],spacing=spacing,dt=c.time_step,density=c.density)
    return dict(device=device,config=asdict(c),operators=rows,steps=steps,raw_scope="last step pressure, corrected and tentative faces")

def audit_legacy(root):
    folder=root/'docs/compatible-projection/legacy'
    report_path=folder/'report.json';raw_path=folder/'cpu-step.npz'
    report=json.loads(report_path.read_text())
    if hashlib.sha256(raw_path.read_bytes()).hexdigest()!=report['raw_sha256']:
        raise ValueError('legacy raw hash mismatch')
    with np.load(raw_path,allow_pickle=False) as raw:
        velocity=raw['velocity'];fluid=raw['fluid'];dx,dy,dz=raw['spacing']
        def derivative(a,axis,h):
            if axis==0:return (np.roll(a,-1,axis=0)-np.roll(a,1,axis=0))/(2*h)
            forward=np.concatenate((np.take(a,range(1,a.shape[axis]),axis=axis),np.take(a,[-1],axis=axis)),axis=axis)
            backward=np.concatenate((np.take(a,[0],axis=axis),np.take(a,range(a.shape[axis]-1),axis=axis)),axis=axis)
            return (forward-backward)/(2*h)
        div=derivative(velocity[...,0],2,dx)+derivative(velocity[...,1],1,dy)+derivative(velocity[...,2],0,dz)
        c=report['config'];continuity=float(np.max(abs(div[fluid])))*2*c['cylinder_radius']/c['inlet_velocity']
        q=raw['pressure_unknown'];a=raw['operator'];dg=raw['negative_div_gradient']
        relative=float(np.linalg.norm((a-dg)[q])/np.linalg.norm(a[q]))
        pulse=raw['pulse']
        gp=[derivative(pulse,2,dx),derivative(pulse,1,dy),derivative(pulse,0,dz)]
        rebuilt=-np.where(fluid,derivative(gp[0],2,dx)+derivative(gp[1],1,dy)+derivative(gp[2],0,dz),0)
        if np.max(abs(rebuilt-dg))>1e-12 or abs(relative-report['operator_relative_L2_difference'])>1e-12 or abs(continuity-report['metric']['continuity'])>1e-12:
            raise ValueError('legacy fields do not reconstruct reported failure')
        if continuity<1e-6 or relative<.1 or report['physical_accuracy_qualified'] is not False:
            raise ValueError('legacy failure evidence unexpectedly qualified')
    return {'legacy_commit':report['legacy_commit'],'continuity':continuity,'operator_relative_L2_difference':relative,
            'raw_sha256':report['raw_sha256'],'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'scope':'independent reconstruction of archived failure fields'}

if __name__=='__main__':
    import inspect
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/compatible-projection/independent')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    src=Path(inspect.getfile(Cylinder3DSolver));root=src.parents[2]
    paths=[src,root/'src/tensorfvm/runtime.py',root/'src/tensorfvm/mesh3d.py',root/'src/tensorfvm/benchmark_cylinder3d.py',Path(__file__).resolve()]
    result=dict(source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},cases=[exercise('cpu',args.output)],physical_accuracy_qualified=False)
    if torch.cuda.is_available():result['cases'].append(exercise('cuda',args.output))
    else:result['gpu_validation']='unavailable; not executed'
    raw_paths=[args.output/('projection-raw-'+case['device']+'.npz') for case in result['cases']]
    result['raw_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths}
    result['legacy_failure']=audit_legacy(root)
    result['torch_version']=str(torch.__version__)
    result['numpy_version']=str(np.__version__)
    result['gpu_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
