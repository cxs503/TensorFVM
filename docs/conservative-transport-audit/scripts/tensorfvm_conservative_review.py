"""Read-only independent finite-volume flux and analytic stress helpers."""
import numpy as np

def div_vector_flux(faces,spacing):
    x,y,z=faces;dx,dy,dz=spacing
    return (x[...,1:,:]-x[...,:-1,:])/dx+(y[:,1:,:,:]-y[:,:-1,:,:])/dy+(z-np.roll(z,1,axis=0))/dz

def upwind_faces(u,q,fluid,inlet):
    result=[]
    for axis,f in enumerate(q):
        l,r=pair(u,axis);l=l.copy();r=r.copy()
        if axis==0:l[...,0,:]=inlet;r[...,-1,:]=inlet
        if axis==1:l[:,0]=inlet;r[:,-1]=inlet
        result.append(f[...,None]*np.where(f[...,None]>=0,l,r))
    return result

def global_boundary_flux(faces,spacing):
    x,y,z=faces;dx,dy,dz=spacing
    return (x[...,-1,:].sum(axis=(0,1))-x[...,0,:].sum(axis=(0,1)))*dy*dz+(y[:,-1,:,:].sum(axis=(0,1))-y[:,0,:,:].sum(axis=(0,1)))*dx*dz

def mms_fields(x,y,z):
    # u=(xy,x²,0); μ=1+x. div τ=(2y, 3+6x, 0).
    xx,yy,zz=np.meshgrid(x,y,z,indexing='ij');u=np.stack((xx*yy,xx*xx,np.zeros_like(xx)),axis=-1);mu=1+xx;exact=np.stack((2*yy,3+6*xx,np.zeros_like(xx)),axis=-1)
    return np.transpose(u,(2,1,0,3)),np.transpose(mu,(2,1,0)),np.transpose(exact,(2,1,0,3))

def pair(a,axis):
    if axis==0:return np.concatenate((a[...,:1,:],a),axis=-2),np.concatenate((a,a[...,-1:,:]),axis=-2)
    if axis==1:return np.concatenate((a[:,:1],a),axis=1),np.concatenate((a,a[:,-1:]),axis=1)
    return a,np.roll(a,-1,axis=0)
def face_fluid(fluid,axis):
    a,b=pair(fluid[...,None],axis);a=a[...,0].copy();b=b[...,0].copy()
    if axis==0:a[...,0]=False;b[...,-1]=False
    if axis==1:a[:,0]=False;b[:,-1]=False
    return a,b

def cell_gradient(u,fluid,inlet,spacing):
    values=[]
    for axis in range(3):
        l,r=pair(u,axis);a,b=face_fluid(fluid,axis);f=np.where((a&b)[...,None],.5*(l+r),0)
        if axis==0:f[...,0,:]=inlet;f[...,-1,:]=l[...,-1,:]
        if axis==1:f[:,0]=inlet;f[:,-1]=inlet
        values.append(f)
    x,y,z=values;return np.stack(((x[...,1:,:]-x[...,:-1,:])/spacing[0],(y[:,1:]-y[:,:-1])/spacing[1],(z-np.roll(z,1,axis=0))/spacing[2]),axis=-1)

def stress(u,mu,fluid,inlet,molecular,spacing):
    cg=cell_gradient(u,fluid,inlet,spacing);out=[]
    for axis,h in enumerate(spacing):
        l,r=pair(u,axis);a,b=face_fluid(fluid,axis);gl,gr=pair(cg.reshape(*cg.shape[:-2],9),axis);g=.5*(gl+gr).reshape(*gl.shape[:-1],3,3);wall=a^b
        normal=(r-l)/h;normal=np.where((a&~b)[...,None],-2*l/h,normal);normal=np.where((~a&b)[...,None],2*r/h,normal);g=np.where((a&b)[...,None,None],g,0);g[...,axis,:]=np.where(wall[...,None],0,g[...,axis,:])
        if axis==0:
            normal[...,0,:]=2*(r[...,0,:]-inlet)/h;normal[...,-1,:]=0;g[...,0,:,:]=0;g[...,-1,:,:]=gr.reshape(*gr.shape[:-1],3,3)[...,-1,:,:]
        if axis==1:
            normal[:,0]=2*(r[:,0]-inlet)/h;normal[:,-1]=2*(inlet-l[:,-1])/h;g[:,0]=0;g[:,-1]=0
        g[..., :,axis]=normal;ml,mr=pair(mu[...,None],axis);mf=.5*(ml[...,0]+mr[...,0]);internal=wall.copy()
        if axis==0:internal[...,0]=False;internal[...,-1]=False
        if axis==1:internal[:,0]=False;internal[:,-1]=False
        mf=np.where(internal,molecular,mf);tau=mf[...,None]*(g[..., :,axis]+g[...,axis,:]);out.append(np.where((a|b)[...,None],tau,0))
    return out

def independent_wall(pf,visc,fluid,spacing):
    force=np.zeros(3);moment=np.zeros(3);records=[]
    for axis,area in enumerate((spacing[1]*spacing[2],spacing[0]*spacing[2],spacing[0]*spacing[1])):
        a,b=face_fluid(fluid,axis);mask=a^b
        if axis==0:mask[...,0]=False;mask[...,-1]=False
        if axis==1:mask[:,0]=False;mask[:,-1]=False
        sign=a.astype(float)-b.astype(float);k,j,i=np.indices(a.shape);pos=np.stack(((i+(0 if axis==0 else .5))*spacing[0],(j+(0 if axis==1 else .5))*spacing[1],(k+(1 if axis==2 else .5))*spacing[2]),axis=-1)
        forcefacet=(pf[axis]-visc[axis])*sign[...,None]*area;force+=forcefacet[mask].sum(0);moment+=np.cross(pos[mask],forcefacet[mask]).sum(0)
        records.append((pos[mask],forcefacet[mask]))
    return force,moment,records

def array(t):return t.detach().cpu().numpy()

def actual(device):
    import torch
    from tensorfvm.solver3d import Cylinder3DConfig
    from tensorfvm.conservative3d import ConservativeCylinder3DSolver
    c=Cylinder3DConfig(nx=32,ny=24,nz=4,max_steps=3,pressure_iterations=500,time_step=.001,device=device);s=ConservativeCylinder3DSolver(c);fluid=array(s.mesh.fluid);sp=(s.mesh.dx,s.mesh.dy,s.mesh.dz);volume=np.prod(sp);inlet=np.array([c.inlet_velocity,0,0]);rows=[]
    for _ in range(3):
        oldfaces=tuple(array(f).copy() for f in s.face_velocity);record=s.step();t=s.last_transport;u=array(t['old_velocity']);mu=array(t['dynamic_viscosity']);conv=tuple(array(f) for f in t['convective_flux']);visc=tuple(array(f) for f in t['viscous_flux']);pf=tuple(array(f) for f in t['pressure_flux']);expected=stress(u,mu,fluid,inlet,c.density*c.kinematic_viscosity,sp);convection=upwind_faces(u,oldfaces,fluid,inlet);conv_error=max(float(np.max(abs(c.density*expected-actual))) for expected,actual in zip(convection,conv));assert conv_error<1e-12;stress_error=max(float(np.max(abs(a-b))) for a,b in zip(expected,visc));assert stress_error<1e-11
        force,moment,facets=independent_wall(pf,visc,fluid,sp);assert np.max(abs(force-array(t['body_force'])))<1e-11;assert np.max(abs(moment-array(t['body_moment'])))<1e-10
        def integ(f):return np.where(fluid[...,None],div_vector_flux(f,sp),0).sum(axis=(0,1,2))*volume
        convforce=-integ(conv);viscforce=integ(visc);pforce=-integ(pf);p0=(c.density*u).sum(axis=(0,1,2))*volume;p1=(c.density*array(s.velocity)).sum(axis=(0,1,2))*volume;res=p1-p0-c.time_step*(convforce+viscforce+pforce);assert np.max(abs(res))<1e-10
        # Full face census accounts for external boundaries + opposite wall force.
        totalboundary=-global_boundary_flux(conv,sp)+global_boundary_flux(visc,sp)-global_boundary_flux(pf,sp)-force
        assert np.max(abs(convforce+viscforce+pforce-totalboundary))<1e-10
        rows.append(dict(time=record['time'],stress_numpy_error=stress_error,convection_numpy_error=conv_error,momentum_residual_Ns=float(np.max(abs(res))),wall_force_N=force.tolist(),wall_moment_Nm=moment.tolist(),face_continuity=record['continuity'],center_divergence=record['center_velocity_divergence'],hypothetical_reconstruction_defect_Ns=record['hypothetical_reconstruction_defect_Ns']))
    np.savez('/tmp/tensorfvm-conservative-raw-'+device+'.npz',fluid=fluid,old_velocity=u,velocity=array(s.velocity),pressure=array(s.pressure),dynamic_viscosity=mu,old_face_u=oldfaces[0],old_face_v=oldfaces[1],old_face_w=oldfaces[2],conv_x=conv[0],conv_y=conv[1],conv_z=conv[2],visc_x=visc[0],visc_y=visc[1],visc_z=visc[2],pressure_x=pf[0],pressure_y=pf[1],pressure_z=pf[2],spacing=sp,dt=c.time_step,density=c.density)
    return dict(device=device,steps=rows)

def mms(device):
    import torch
    from tensorfvm.solver3d import Cylinder3DConfig
    from tensorfvm.conservative3d import ConservativeCylinder3DSolver
    c=Cylinder3DConfig(nx=32,ny=24,nz=8,max_steps=1,device=device);s=ConservativeCylinder3DSolver(c);s.mesh.fluid[:]=True;s._build_face_masks();sp=(s.mesh.dx,s.mesh.dy,s.mesh.dz);x=(np.arange(c.nx)+.5)*sp[0];y=(np.arange(c.ny)+.5)*sp[1];z=(np.arange(c.nz)+.5)*sp[2];u,mu,exact=mms_fields(x,y,z);ut=torch.tensor(u,dtype=torch.float64,device=device);mt=torch.tensor(mu,dtype=torch.float64,device=device);flux=s.stress_fluxes(ut,mt);a=array(s.vector_divergence(flux));interior=(slice(None),slice(3,-3),slice(3,-3),slice(None));error=float(np.max(abs(a[interior]-exact[interior])));assert error<1e-10
    # μΔu misses transpose/cross-gradient contributions by a finite amount.
    wrong=np.zeros_like(exact);wrong[...,1]=2*mu;difference=float(np.max(abs(exact[interior]-wrong[interior])));assert difference>1
    return dict(device=device,variable_mu_cross_component_MMS_max_error=error,mu_laplacian_wrong_model_difference=difference)

if __name__=='__main__':
    import torch,json,inspect,hashlib
    from pathlib import Path
    from tensorfvm.conservative3d import ConservativeCylinder3DSolver
    result={'cases':[actual('cpu')],'MMS':[mms('cpu')],'physical_accuracy_qualified':False}
    if torch.cuda.is_available():result['cases'].append(actual('cuda'));result['MMS'].append(mms('cuda'))
    p=Path(inspect.getfile(ConservativeCylinder3DSolver));result['source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();result['dependencies_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [p.parent/'solver3d.py',p.parent/'runtime.py',p.parent/'mesh3d.py',Path(__file__)]};result['raw_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path('/tmp/tensorfvm-conservative-raw-'+case['device']+'.npz') for case in result['cases']]};Path('/tmp/tensorfvm-conservative-independent-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
