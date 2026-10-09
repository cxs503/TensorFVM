"""Independent NumPy checks of raw solver evidence; no external solver imports."""
import hashlib,json
from pathlib import Path
import numpy as np
C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],float)
def require(ok,message):
    if not ok: raise ValueError(message)
def close(a,b,tol=1e-10):
    require(np.max(np.abs(np.asarray(a)-np.asarray(b)))<tol,'reconstructed quantity mismatch')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def unqualified(r):
    require(r.get('physical_accuracy_qualified') is False,'unsupported physical qualification')
def pop(f):
    f=np.asarray(f,float);require(np.isfinite(f).all() and (f>=0).all(),'invalid populations')
    return f.sum(),np.einsum('iyx,ia->a',f,C)
def local_case(r):
    require(r.get('physical_accuracy_qualified',False) is False,'unsupported physical qualification');m0,p0=pop(r['initial']['f']);m,p=pop(r['final']['f']);close(m,m0,1e-8)
    h=r['final']['history'];imp=np.sum([np.asarray(v['wall_impulse_on_solid'])+v['conversion_impulse_on_solid'] for v in h],axis=0)
    close(p-p0,-imp,1e-8)
    require(all(x['global_mass_rescale'] is False and x['maximum_transfer_radius']==4 for x in h),'nonlocal rescale')
    require(r['restart_bitwise'] is True,'restart failure')
    return dict(mass_residual_lattice=float(m-m0),momentum_residual_lattice=float(np.max(abs(p-p0+imp))),impulse=imp.tolist())
def dem_case(r,smooth=False):
    unqualified(r);b=r['final']['base'] if smooth else r['final'];c=b['config'];x=np.asarray(r['positions_m']);mass=np.asarray(r['mass_nodes_kg']);u=np.asarray(b['u']);v=np.asarray(b['v']);edges=np.asarray(r['edges']);k=np.asarray(r['stiffness_N_per_m'])
    dx=c['length']/c['nx'];dy=c['height']/c['ny'];expected=np.full(len(x),c['density']*c['thickness']*dx*dy)
    expected[(x[:,0]==0)|(x[:,0]==c['length'])]*=.5;expected[(x[:,1]==0)|(x[:,1]==c['height'])]*=.5
    close(mass,expected);close(mass.sum(),22.5);fixed=np.zeros_like(u,dtype=bool);fixed[:,1]=True;fixed[x[:,0]==0,0]=True;close(u[fixed],0);close(v[fixed],0)
    close((mass[:,None]*v).sum(0),b['impulse'])
    d=x[edges[:,1]]-x[edges[:,0]];n=d/np.linalg.norm(d,axis=1)[:,None];strain=((u[edges[:,1]]-u[edges[:,0]])*n).sum(1)
    energy=.5*(mass[:,None]*v*v).sum()+.5*(k*strain*strain).sum();right=x[:,0]==c['length'];area=np.full(right.sum(),dy*c['thickness']);area[[0,-1]]*=.5
    end=(u[right,0]*area).sum()/area.sum();close(end,r['end_displacement_m'])
    if smooth:
        q=np.asarray(r['tool_quadrature_force_displacement_dt']);require(len(q)==r['steps'],'missing quadrature');work=-np.dot(q[:,0],q[:,1]);imp=np.dot(q[:,0],q[:,2]);close(work,r['tool_work_J']);close(imp,r['tool_impulse_Ns'])
        t=r['duration_s'];z=min(max(t/r['ramp_s'],0),1);plane=1-r['depth_m']*(10*z**3-15*z**4+6*z**5)
        gap=np.maximum(x[right,0]+u[right,0]-plane,0);energy+=.5*50*c['E']/c['length']*(area*gap*gap).sum();close(energy,r['history'][-1]['mechanical_and_contact_energy_J'])
        require(abs(energy-work)<.01*abs(work),'tool energy gate')
    else:
        close(energy,r['history'][-1]['energy_J'])
        # Full NumPy replay because published fixed-plane histories are decimated.
        ur=np.zeros_like(u);vr=np.zeros_like(v);imp=0.;dt=r['dt_s']
        def force(z):
            extension=((z[edges[:,1]]-z[edges[:,0]])*n).sum(1);spring=(k*extension)[:,None]*n;f=np.zeros_like(z)
            np.add.at(f,edges[:,0],spring);np.add.at(f,edges[:,1],-spring)
            ext=100*area if r['mode']=='load' else -50*c['E']/c['length']*area*np.maximum(x[right,0]+z[right,0]-r['plane_m'],0)
            f[right,0]+=ext;f[fixed]=0;return f,ext
        for _ in range(r['steps']):
            f,ext=force(ur);vh=vr+.5*dt*f/mass[:,None];ur+=dt*vh;f2,ext2=force(ur);vr=vh+.5*dt*f2/mass[:,None];imp-=.5*dt*(ext+ext2).sum()
        close(ur,u);close(vr,v);close(imp,r['tool_impulse_Ns'])
        if r['mode']=='contact':
            gap=np.maximum(x[right,0]+u[right,0]-r['plane_m'],0);total=energy+.5*50*c['E']/c['length']*(area*gap*gap).sum();require(abs(total-r['initial_contact_energy_J'])<.01*r['initial_contact_energy_J'],'contact energy gate')
        if r['mode']=='load':require(abs(energy-b['work'])<1e-6,'load work gate')
    require(r['restart_exact'] is True,'restart failure')
    return dict(mass_kg=float(mass.sum()),energy_J=float(energy),tool_impulse_Ns=float(imp),end_displacement_m=float(end))
def feedback_case(r):
    unqualified(r);require(r['flexible_suboff_qualified'] is False and r['energy_accuracy_qualified'] is False,'false feedback qualification')
    s=r['state'];f=s['fluid'];c=f['config'];mass=s['body_mass_kg'];vel=np.asarray(s['history'][0]['exchange']['velocity_held_m_s']);position=np.asarray(s['initial_center_m']);time=0.;reservoir=np.zeros(2);impulses=np.zeros(2)
    for h in s['history']:
        e=h['exchange'];unqualified(e);close(e['time_start_s'],time);close(e['center_start_m'],position);close(e['velocity_held_m_s'],vel);dt=e['time_end_s']-time;require(dt>0,'invalid clock')
        j=np.asarray(e['impulse_on_body_Ns']);close(j,np.asarray(e['wall_impulse_on_body_Ns'])+e['conversion_impulse_on_body_Ns']);close(e['center_end_m'],position+dt*vel)
        new=vel+j/mass;close(new,h['body_velocity_m_s']);work=j@(.5*(vel+new));close(.5*mass*(new@new-vel@vel),work);close(work,h['body_midpoint_impulse_work_J']);close(h['body_kinetic_J'],.5*mass*(new@new));close(e['body_interface_work_J'],j@vel)
        reservoir+=e['reservoir_impulse_on_fluid_Ns'];impulses+=j;close(np.asarray(e['fluid_momentum_Ns'])+mass*new-reservoir,s['initial_momentum_Ns'],1e-9)
        if s['backend']=='local':require(e['global_mass_rescale'] is False,'global transfer in local case');close(reservoir,0)
        vel=new;position=np.asarray(e['center_end_m']);time=e['time_end_s']
    _,p=pop(f['solver']['f']);scale=c['density_kg_m3']*c['thickness_m']*c['dx_m']**3/c['dt_s'];close(p*scale,s['history'][-1]['exchange']['fluid_momentum_Ns']);close(p*scale+mass*vel-reservoir,s['initial_momentum_Ns'],1e-9);close(vel,s['velocity_m_s']);close(position,np.asarray(f['solver']['center'])*c['dx_m'])
    return dict(final_body_impulse_Ns=impulses.tolist(),momentum_residual_Ns=float(np.max(abs(p*scale+mass*vel-reservoir-s['initial_momentum_Ns']))),physical_accuracy_qualified=False)
def audit(lbm,dem,fem):
    roots={'TensorLBM':Path(lbm),'TensorDEM':Path(dem),'TensorFEM':Path(fem)};cases=[];inputs={}
    def checked(path,digest):
        require(sha(path)==digest,'source/raw SHA mismatch');inputs[str(path)]=digest;return load(path) if path.suffix=='.json' else None
    out=roots['TensorLBM']/'docs/assets/local-moving-boundary';inputs[str(out/'study.json')]=sha(out/'study.json');study=load(out/'study.json');unqualified(study);local={}
    for item in study['cases']:
        r=checked(out/item['artifact'],item['sha256']);checked(roots['TensorLBM']/'src/tensorlbm/local_moving_boundary_2d.py',r['source_sha256']);checked(roots['TensorLBM']/'src/tensorlbm/moving_boundary_2d.py',r['base_source_sha256']);result=local_case(r);local[item['case']]=result;cases.append(dict(group='local-moving',case=item['case'],**result))
    change=abs(.5*local['step-half']['impulse'][0]/local['cross-grid']['impulse'][0]-1);close(change,study['refinement']['relative_impulse_change']);require(study['refinement']['passed']==(change<.03),'false refinement badge')
    demgroups={}
    for group,smooth in [('continuum-boundary',False),('smooth-plane',True)]:
        out=roots['TensorDEM']/('docs/coupling/'+group);inputs[str(out/'study.json')]=sha(out/'study.json');st=load(out/'study.json');unqualified(st);sources=st['source_sha256'];sources=sources if isinstance(sources,dict) else {'src/tensordem/continuum_boundary_dem.py':sources}
        for name,digest in sources.items():checked(roots['TensorDEM']/name,digest)
        raw={}
        for name,digest in st.get('raw_sha256',st.get('files_sha256',{})).items():
            r=checked(out/name,digest);result=dem_case(r,smooth);raw[name]=result;cases.append(dict(group=group,case=name,**result))
        if smooth:
            changes=[abs(raw['grid-64.json'][key]/raw['grid-32.json'][key]-1) for key in ('tool_impulse_Ns','end_displacement_m')];require(st['dynamic_3_percent_passed']==all(v<.03 for v in changes),'false smooth badge')
        else:
            change2=abs(raw['contact-64.json']['tool_impulse_Ns']/raw['contact-32.json']['tool_impulse_Ns']-1);close(change2,st['contact_impulse_refinement_relative'][-1]);require(st['contact_impulse_3_percent_passed']==(change2<.03),'false abrupt badge')
        demgroups[group]=st
    out=roots['TensorFEM']/'docs/assets/live-rigid-feedback';inputs[str(out/'study.json')]=sha(out/'study.json');st=load(out/'study.json');unqualified(st)
    for name,digest in st['source_sha256'].items():project,rel=name.split('/',1);checked(roots[project]/rel,digest)
    feedback={}
    for name,digest in st['artifacts_sha256'].items():
        result=feedback_case(checked(out/name,digest));feedback[name]=result;cases.append(dict(group='live-rigid-feedback',case=name,**result))
    sensitivity=abs(feedback['local-time-half.json']['final_body_impulse_Ns'][1]/feedback['local.json']['final_body_impulse_Ns'][1]-1)*100;close(sensitivity,st['time_eos_half_impulse_change_percent']);require(st['time_eos_sensitivity_pass_3percent']==(sensitivity<3),'false feedback convergence')
    return dict(schema='tensorfvm.independent-fourth-round-audit/1',cases=cases,raw_audit_passed=True,physical_accuracy_qualified=False,local_refinement_percent=change*100,abrupt_contact_impulse_refinement_percent=change2*100,smooth_plane_dynamic_passed=True,feedback_time_eos_sensitivity_percent=sensitivity,inputs_sha256=inputs,auditor_sha256=sha(__file__))
