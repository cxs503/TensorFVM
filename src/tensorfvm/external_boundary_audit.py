"""Independent NumPy audit of external D2Q9 boundary evidence; no LBM imports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

C = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def scales(dx, dt, rho, thickness, nu, speed, diameter):
    require(all(math.isfinite(x) and x > 0 for x in (dx,dt,rho,thickness,nu,diameter)), 'invalid SI scales')
    cs=dx/(math.sqrt(3)*dt)
    return dict(sound_speed_m_s=cs,tau=.5+3*nu*dt/dx**2,mach=speed/cs,reynolds=speed*diameter/nu,
                lattice_mass_kg=rho*dx**2*thickness,lattice_impulse_Ns=rho*dx**3*thickness/dt)

def mask(shape, center, radius, periodic=False):
    y,x=np.indices(shape);xx=x-center[0];yy=y-center[1]
    if periodic:
        xx=(xx+shape[1]/2)%shape[1]-shape[1]/2;yy=(yy+shape[0]/2)%shape[0]-shape[0]/2
    return xx**2+yy**2<=radius**2

def field(f, solid):
    f=np.asarray(f,dtype=float)
    require(f.shape==(9,*solid.shape) and np.isfinite(f).all() and f.min()>=0, 'invalid population')
    require(np.max(np.abs(f[:,solid]))==0, 'fluid population inside solid mask')
    return float(f.sum()),np.einsum('iyx,ia->a',f,C)

def close(a,b,tol,message):
    require(np.max(np.abs(np.asarray(a)-np.asarray(b)))<=tol,message)

def moving(raw):
    state=raw['state'];r=raw['report'];cfg=state['config'];si=r['si']
    unit=scales(si['dx_m'],si['dt_s'],si['rho_kg_m3'],si['thickness_m'],si['nu_m2_s'],si['velocity_m_s'],2*cfg['radius']*si['dx_m'])
    close(unit['tau'],cfg['tau'],1e-14,'tau inconsistent with SI viscosity')
    close(np.asarray(cfg['velocity'])*si['dx_m']/si['dt_s'],[si['velocity_m_s'],0],1e-14,'velocity conversion')
    solid=mask((cfg['ny'],cfg['nx']),state['center'],cfg['radius'],True)
    mass,p=field(state['f'],solid)
    initial_solid=mask(solid.shape,cfg['center'],cfg['radius'],True)
    old_mass=float((~initial_solid).sum());old_p=old_mass*np.asarray(cfg['fluid_velocity'])
    wall=np.zeros(2);conversion=np.zeros(2);reservoir=np.zeros(2);maxp=maxm=0.
    for h in state['history']:
        conv=np.asarray(h['removed_momentum'])-h['added_momentum']
        close(conv,h['conversion_impulse_on_solid'],1e-12,'conversion impulse')
        dp=np.asarray(h['momentum'])-old_p
        err=dp+np.asarray(h['wall_impulse_on_solid'])+conv-h['reservoir_momentum']
        me=h['mass']-old_mass-h['boundary_mass_change']-h['added_mass']+h['removed_mass']-h['reservoir_mass']
        maxp=max(maxp,float(np.max(np.abs(err))));maxm=max(maxm,abs(me))
        wall+=h['wall_impulse_on_solid'];conversion+=conv;reservoir+=h['reservoir_momentum']
        old_p=np.asarray(h['momentum']);old_mass=h['mass']
    require(maxp<1e-9 and maxm<1e-8,'moving step ledger does not close')
    close(p,old_p,1e-10,'raw final momentum');close(mass,old_mass,1e-9,'raw final mass')
    close((wall+conversion)*unit['lattice_impulse_Ns'],r['solid_impulse_si_Ns'],1e-10,'SI body impulse')
    return dict(**unit,fluid_mass_kg=mass*unit['lattice_mass_kg'],excluded_nodes=int(solid.sum()),
        body_wall_impulse_Ns=(wall*unit['lattice_impulse_Ns']).tolist(),body_conversion_impulse_Ns=(conversion*unit['lattice_impulse_Ns']).tolist(),
        global_reservoir_impulse_Ns=(reservoir*unit['lattice_impulse_Ns']).tolist(),max_step_momentum_error_Ns=maxp*unit['lattice_impulse_Ns'],
        max_step_mass_error_kg=maxm*unit['lattice_mass_kg'],reservoir_absolute_exchange_kg=sum(abs(h['reservoir_mass']) for h in state['history'])*unit['lattice_mass_kg'],
        local_mass_conservation_qualified=False,steady_qualified=False,sensitivity_type='combined time-step and compressibility or changed motion Reynolds number')

def fixed(raw):
    dx=raw['dx_m'];dt=raw['dt_s'];driven='drive_impulse_Ns' in raw
    nu=raw['nu_m2_s'] if driven else raw['viscosity_m2_s'];speed=0. if driven else math.hypot(*raw['initial_velocity_m_s'])
    solid=mask((raw['n'],raw['n']),np.asarray(raw['center_m'])/dx,raw['radius_m']/dx)
    m0,p0=field(raw['initial_population'],solid);mf,pf=field(raw['final_population'],solid)
    unit=scales(dx,dt,raw['rho_kg_m3'],raw['thickness_m'],nu,speed,2*raw['radius_m'])
    body=np.asarray(raw['solid_impulse_Ns']);drive=np.asarray(raw.get('drive_impulse_Ns',[0,0]))
    residual=(pf-p0)*unit['lattice_impulse_Ns']+body-drive
    require(abs(mf-m0)<1e-8 and np.max(np.abs(residual))<1e-8,'fixed raw conservation failure')
    ff=np.asarray(raw['final_population']);density=ff.sum(0)
    velocity=np.einsum('iyx,ia->ayx',ff,C)[:,~solid]/density[~solid]
    final_speed=float(np.max(np.linalg.norm(velocity,axis=0)))*dx/dt
    h=raw['history'];forces=np.array([x['force_x_N'] if driven else x['force_N'][0] for x in h])
    close(forces.sum()*dt,body[0],1e-8,'integrated body force')
    drift=None
    if driven:
        early=[x['force_x_N'] for x in h if 1.5<x['time_s']<=1.75];late=[x['force_x_N'] for x in h if x['time_s']>1.75]
        require(bool(early) and bool(late),'missing late transient windows')
        drift=abs(np.mean(late)/np.mean(early)-1)
        close(drift,raw['late_window_drift'],1e-12,'late drift')
        require(raw['steady_qualified']==bool(drift<.01),'steady badge mismatch')
    return dict(**unit,fluid_mass_kg=mf*unit['lattice_mass_kg'],excluded_nodes=int(solid.sum()),
        final_max_speed_m_s=final_speed,final_max_mach=final_speed/unit['sound_speed_m_s'],final_max_reynolds=final_speed*2*raw['radius_m']/nu,
        body_impulse_Ns=body.tolist(),drive_impulse_Ns=drive.tolist(),raw_momentum_error_Ns=float(np.max(np.abs(residual))),
        late_window_relative_drift=drift,steady_qualified=bool(driven and drift<.01),sensitivity_type='transient impulse' if not driven else 'late transient window; steady requires drift < 1%',
        velocity_scale_definition='initial uniform velocity' if not driven else 'zero initial velocity; Mach/Re above are initial values, not evolved flow values')

def audit(root):
    root=Path(root);outputs=[];bindings={}
    for rel,kind in [('docs/assets/curved-boundary/study.json','fixed'),('docs/assets/curved-boundary/driven-study.json','fixed'),('examples/moving_boundary/evidence/study.json','moving')]:
        path=root/rel;manifest=json.loads(path.read_text());bindings[rel]=sha(path)
        sources=manifest['source_sha256']
        if isinstance(sources,str):sources={'src/tensorlbm/moving_boundary_2d.py':sources}
        for source,digest in sources.items():
            require(sha(root/source)==digest,'source hash mismatch '+source);bindings[source]=digest
        for entry in manifest['cases']:
            file=entry.get('file',entry.get('name','')+'.json');rawpath=path.parent/file
            require(sha(rawpath)==entry['sha256'],'raw hash mismatch '+file);bindings[str(rawpath.relative_to(root))]=entry['sha256']
            raw=json.loads(rawpath.read_text());metrics=(moving if kind=='moving' else fixed)(raw)
            outputs.append(dict(file=str(rawpath.relative_to(root)),**metrics,raw_audit_passed=True,physical_accuracy_qualified=False))
    return dict(schema='tensorfvm.external-boundary-cross-audit/1',auditor='independent numpy; no TensorLBM imports',source_and_raw_sha256=bindings,auditor_source_sha256=sha(__file__),cases=outputs,
        fvm_same_boundary_simulation_performed=False,physical_accuracy_qualified=False,temporal_convergence_qualified=False,
        eos_note='Changing dx/dt changes SI sound speed. Fixed-grid dt refinement is combined temporal/compressibility sensitivity.',
        reservoir_note='Global rescaling closes global bookkeeping; it does not establish local swept-volume conservation.')
