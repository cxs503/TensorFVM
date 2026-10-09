"""Independent NumPy periodic MAC face checks; external solver adapter pending."""
import numpy as np

def divergence(faces,spacing):
    # Tensor spatial axes z,y,x; u/v/w stored at the owner's +axis face.
    result=np.zeros_like(faces[0])
    for field,axis,h in zip(faces,(2,1,0),spacing):result+=(field-np.roll(field,1,axis=axis))/h
    return result

def gradient(p,spacing):
    return tuple((np.roll(p,-1,axis=axis)-p)/h for axis,h in zip((2,1,0),spacing))

def energy(faces,rho,volume):return .5*rho*volume*sum(np.sum(f*f) for f in faces)
def momentum(faces,rho,volume):return rho*volume*np.array([np.sum(f) for f in faces])

def independent_projection(faces,spacing):
    rhs=divergence(faces,spacing);shape=rhs.shape;wave=0
    for axis,h in zip((2,1,0),spacing):
        n=shape[axis];s=[1,1,1];s[axis]=n;wave=wave+(4*np.sin(np.pi*np.fft.fftfreq(n))**2).reshape(s)/h**2
    r=np.fft.fftn(rhs);wave[0,0,0]=1.;p=np.fft.ifftn(-r/wave).real;p-=p.mean();return tuple(f-g for f,g in zip(faces,gradient(p,spacing))),p

def dual_convection(faces,spacing):
    results=[];allflux=[]
    for a,ua in enumerate(faces):
        axisa=(2,1,0)[a];fluxes=[];conv=np.zeros_like(ua)
        for b,ub in enumerate(faces):
            axisb=(2,1,0)[b];q=.5*(ub+np.roll(ub,-1,axis=axisa));transport=.5*(ua+np.roll(ua,-1,axis=axisb));flux=q*transport;fluxes.append(flux);conv+=(flux-np.roll(flux,1,axis=axisb))/spacing[b]
        results.append(conv);allflux.append(fluxes)
    return tuple(results),allflux

def full_stress(faces,mu,spacing):
    normal=[];shear={};operator=[np.zeros_like(a) for a in faces];dissipation=0.
    for a,ua in enumerate(faces):
        axis=(2,1,0)[a];d=(ua-np.roll(ua,1,axis=axis))/spacing[a];tau=2*mu*d;normal.append(tau);operator[a]+=(np.roll(tau,-1,axis=axis)-tau)/spacing[a];dissipation+=np.sum(2*mu*d*d)
    for a in range(3):
        for b in range(a+1,3):
            axisa=(2,1,0)[a];axisb=(2,1,0)[b];strain=(np.roll(faces[a],-1,axis=axisb)-faces[a])/spacing[b]+(np.roll(faces[b],-1,axis=axisa)-faces[b])/spacing[a];edge=.25*(mu+np.roll(mu,-1,axis=axisa)+np.roll(mu,-1,axis=axisb)+np.roll(np.roll(mu,-1,axis=axisa),-1,axis=axisb));tau=edge*strain;shear[(a,b)]=tau;operator[a]+=(tau-np.roll(tau,1,axis=axisb))/spacing[b];operator[b]+=(tau-np.roll(tau,1,axis=axisa))/spacing[a];dissipation+=np.sum(edge*strain*strain)
    return tuple(operator),float(dissipation),normal,shear

def array(x):return x.detach().cpu().numpy()
def exercise(device):
    import torch
    from tensorfvm.periodic_mac import PeriodicMACConfig,PeriodicMACSolver
    c=PeriodicMACConfig(nx=16,ny=12,nz=8,time_step_s=.001,device=device);s=PeriodicMACSolver(c);rng=np.random.default_rng(903);raw=rng.normal(scale=.1,size=(3,*s.shape));p=rng.normal(size=s.shape);tensor=torch.tensor(raw,dtype=torch.float64,device=device);pt=torch.tensor(p,dtype=torch.float64,device=device);g=gradient(p,s.spacing);actualg=array(s.gradient(pt));d=divergence(raw,s.spacing);actuald=array(s.divergence(tensor));gerr=float(np.max(abs(actualg-np.asarray(g))));derr=float(np.max(abs(actuald-d)));assert gerr<1e-12 and derr<1e-12
    adjoint=sum(np.sum(raw[a]*g[a]) for a in range(3))+np.sum(d*p);assert abs(adjoint)<1e-10
    corrected,pressure,record=s.project(tensor);expected,pp=independent_projection(raw,s.spacing);projecterr=float(np.max(abs(array(corrected)-np.asarray(expected))));assert projecterr<1e-12;closep=float(np.max(abs(array(pressure)-pp*c.density_kg_m3/c.time_step_s)));assert closep<1e-9
    before=energy(raw,c.density_kg_m3,s.volume);after=energy(expected,c.density_kg_m3,s.volume);removed=energy(tuple(a-b for a,b in zip(raw,expected)),c.density_kg_m3,s.volume);orthogonal_error=abs(before-after-removed);assert orthogonal_error<1e-11;assert after<=before;assert np.max(abs(momentum(raw,c.density_kg_m3,s.volume)-momentum(expected,c.density_kg_m3,s.volume)))<1e-11
    s.velocity=corrected;x,y,z=s.centers();s.dynamic_viscosity=.01+.004*torch.sin(x)*torch.cos(y)
    mu=array(s.dynamic_viscosity);npstress,diss,normals,shears=full_stress(expected,mu,s.spacing);ts=s.full_stress(corrected);top=array(s.stress_divergence(ts));stress_error=float(np.max(abs(top-np.asarray(npstress))));assert stress_error<1e-11;work=sum(np.sum(expected[a]*npstress[a]) for a in range(3));stress_power_error=abs(work+diss)*s.volume;assert stress_power_error<1e-10
    npconv,flux=dual_convection(expected,s.spacing);tf=s.convective_fluxes(corrected);convflux_error=float(np.max(abs(array(tf)-c.density_kg_m3*np.array(flux))));assert convflux_error<1e-12;convpower=sum(np.sum(expected[a]*npconv[a]) for a in range(3))*s.volume*c.density_kg_m3;assert abs(convpower)<1e-11
    rows=[]
    for _ in range(3):
        r=s.step();q=s.last_raw;old=array(q['old_velocity']);new=array(q['candidate_velocity']);mid=.5*(old+new);conv,_=dual_convection(mid,s.spacing);stressop,loss,*_=full_stress(mid,mu,s.spacing);tent=old+c.time_step_s*(-c.density_kg_m3*np.array(conv)+np.array(stressop))/c.density_kg_m3;check,indpressure=independent_projection(tent,s.spacing);nonlinear=float(np.max(abs(np.array(check)-new)));assert nonlinear<=c.nonlinear_tolerance*1.01;assert np.max(abs(tent-array(q['tentative_velocity'])))<1e-12
        k0=energy(old,c.density_kg_m3,s.volume);k1=energy(new,c.density_kg_m3,s.volume);convwork=-s.volume*c.density_kg_m3*np.sum(mid*np.array(conv));viscwork=-s.volume*loss;defect=k1-k0-c.time_step_s*(convwork+viscwork);assert abs(defect)<1e-10;assert k1<=k0;div=float(np.max(abs(divergence(new,s.spacing))));assert div<1e-11
        rows.append(dict(time_s=s.time,energy_balance_defect_J=float(defect),nonlinear_residual_m_s=nonlinear,max_divergence=div,energy_J=k1,viscous_power_W=float(viscwork),convective_power_W=float(convwork)))
    np.savez('/tmp/tensorfvm-mac-raw-'+device+'.npz',old=old,candidate=new,midpoint=mid,tentative=tent,pressure=array(q['pressure']),mu=mu,spacing=s.spacing,dt=c.time_step_s,density=c.density_kg_m3)
    return dict(device=device,gradient_error=gerr,divergence_error=derr,adjoint_error=float(abs(adjoint)),projection_error=projecterr,orthogonal_energy_error_J=orthogonal_error,stress_error=stress_error,stress_negative_work_identity_error=stress_power_error,convective_flux_error=convflux_error,convective_power_W=float(convpower),steps=rows)

if __name__=='__main__':
    import torch,json,hashlib,inspect
    from pathlib import Path
    from tensorfvm.periodic_mac import PeriodicMACSolver
    cases=[exercise('cpu')]
    if torch.cuda.is_available():cases.append(exercise('cuda'))
    src=Path(inspect.getfile(PeriodicMACSolver));report=dict(cases=cases,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_accuracy_qualified=False,distributed_supported=False);Path('/tmp/tensorfvm-mac-independent-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
