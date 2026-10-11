"""Independent NumPy raw-field and full-step reconstruction; no solver imports."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .core import sha


def close(a,b,tol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite audit input")
    e=float(np.max(abs(a-b),initial=0))
    if e>tol:raise ValueError(f"independent reconstruction mismatch {e} > {tol}")
    return e


def relative(a,b,maximum=False):
    a,b=np.asarray(a),np.asarray(b)
    denominator=np.max(abs(b)) if maximum else np.sqrt(np.sum(b*b))
    return float((np.max(abs(a-b)) if maximum else np.sqrt(np.sum((a-b)**2)))/denominator)


def d(u,h):
    return sum((u[a]-np.roll(u[a],1,axis=2-a))/h[a] for a in range(3))


def g(p,h):
    return np.stack([(np.roll(p,-1,axis=2-a)-p)/h[a] for a in range(3)])


def mac_acceleration(u,h,mu,rho):
    convection=[]
    for a in range(3):
        total=np.zeros_like(u[a])
        for b in range(3):
            adv=.5*(u[b]+np.roll(u[b],-1,axis=2-a))
            transported=.5*(u[a]+np.roll(u[a],-1,axis=2-b))
            flux=adv*transported
            total+=(flux-np.roll(flux,1,axis=2-b))/h[b]
        convection.append(total)
    # Constant mu: the complete symmetric stress still uses staggered normal/edge sites.
    normal=[2*mu*(u[a]-np.roll(u[a],1,axis=2-a))/h[a] for a in range(3)]
    shear={}
    for a in range(3):
        for b in range(a+1,3):
            shear[a,b]=mu*((np.roll(u[a],-1,axis=2-b)-u[a])/h[b]+
                            (np.roll(u[b],-1,axis=2-a)-u[b])/h[a])
    force=[]
    for a in range(3):
        value=(np.roll(normal[a],-1,axis=2-a)-normal[a])/h[a]
        for b in range(3):
            if a!=b:
                t=shear[min(a,b),max(a,b)]
                value+=(t-np.roll(t,1,axis=2-b))/h[b]
        force.append(value)
    return np.array(convection),np.array(force)


def projected(u,h,rho,dt):
    rhs=d(u,h)
    symbol=0
    for a in range(3):
        n=u.shape[3-a];shape=[1,1,1];shape[2-a]=n
        symbol=symbol+(4*np.sin(np.pi*np.fft.fftfreq(n))**2/h[a]**2).reshape(shape)
    symbol[0,0,0]=1
    p=np.fft.ifftn(-np.fft.fftn(rhs)/symbol).real*rho/dt;p-=p.mean()
    return u-dt/rho*g(p,h),p


def audit(directory,source_check=True):
    directory=Path(directory);summary=json.loads((directory/"summary.json").read_text())
    root=Path(__file__).resolve().parents[3]
    if summary["error_limit"]!=.03:raise ValueError("3% gate changed")
    if source_check:
        for n,h in summary["provenance"]["source_sha256"].items():
            if sha(root/n)!=h:raise ValueError("source digest mismatch: "+n)
    manifest=directory/"manifest.json"
    if manifest.exists():
        for n,h in json.loads(manifest.read_text())["artifacts_sha256"].items():
            if sha(directory/n)!=h:raise ValueError("artifact digest mismatch: "+n)
    for case,role,count in (("poiseuille","spatial",3),("taylor-green","spatial",3),
                            ("taylor-green","temporal",3)):
        if sum(r["case"]==case and r["role"]==role for r in summary["runs"])!=count:
            raise ValueError("required refinement study missing")
    if sum(r["role"]=="negative-control" for r in summary["runs"])!=2:
        raise ValueError("required negative controls missing")
    audited=[]
    for r in summary["runs"]:
        sub=directory/r["directory"];f=np.load(sub/"fields.npz",allow_pickle=False)
        if json.loads((sub/"result.json").read_text())!=r:
            raise ValueError("summary/run record mismatch")
        history=json.loads((sub/"history.json").read_text());c=r["config"];computed={}
        if not all(np.isfinite(f[k]).all() for k in f.files):raise ValueError("nonfinite raw arrays")
        if r["case"]=="poiseuille":
            x,y=f["x_m"],f["y_m"];u,v=f["u_m_s"],f["v_m_s"];uf,vf=f["u_faces_m_s"],f["v_faces_m_s"]
            close(u,.5*(uf[:,1:]+uf[:,:-1]));close(v,.5*(vf[1:]+vf[:-1]))
            nu=c["inlet_velocity"]*c["height"]/c["reynolds"];mu=c["density"]*nu
            exact=6*c["inlet_velocity"]*y/c["height"]*(1-y/c["height"])
            section=np.argmin(abs(x-5*c["height"]));region=(x>=3*c["height"])&(x<=5*c["height"])
            px=f["p_pa"][:,region].mean(0);xx=x[region]-x[region].mean()
            slope=np.dot(xx,px-px.mean())/np.dot(xx,xx);eslope=-12*mu*c["inlet_velocity"]/c["height"]**2
            pref=px.mean()+eslope*xx;wall=-mu*np.stack((u[0],u[-1]))/(.5*c["height"]/c["ny"]);shear=6*mu*c["inlet_velocity"]/c["height"]
            close(f["reference_u_profile_m_s"],exact);close(f["reference_pressure_curve_pa"],pref)
            close(f["pressure_curve_pa"],px);close(f["profile_u_m_s"],u[:,section])
            div=(uf[:,1:]-uf[:,:-1])/(c["length"]/c["nx"])+(vf[1:]-vf[:-1])/(c["height"]/c["ny"])
            close(div,f["divergence_s_inv"])
            inlet=uf[:,0].sum();outlet=uf[:,-1].sum();length=region.sum()*c["length"]/c["nx"]
            pf=-slope*c["height"]*length;wf=wall[:,region].sum()*c["length"]/c["nx"]
            computed=dict(velocity_l2=relative(u[:,section],exact),velocity_linf=relative(u[:,section],exact,True),
              pressure_gradient=abs(slope-eslope)/abs(eslope),pressure_curve_l2=relative(px-px.mean(),pref-px.mean()),
              wall_shear_linf=float(np.max(abs(wall[:,region]+shear))/shear),force_balance=abs(pf+wf)/abs(pf),
              relative_mass_imbalance=abs(outlet-inlet)/abs(inlet),scaled_divergence=float(np.max(abs(div))*c["height"]/c["inlet_velocity"]),
              solver_residual=max(history[-1][k] for k in ("continuity","momentum","mass_imbalance")),
              convergence_failure=0. if r["converged"] else 1.)
            close(f["lower_wall_traction_pa"],wall[0]);close(f["upper_wall_traction_pa"],wall[1])
        elif r["case"]=="taylor-green":
            n=c["nx"];h=np.array(c["lengths_m"])/[c["nx"],c["ny"],c["nz"]];dt=c["time_step_s"];rho=c["density_kg_m3"];mu=c["viscosity_pa_s"]
            z,y,x=np.meshgrid((np.arange(c["nz"])+.5)*h[2],(np.arange(n)+.5)*h[1],(np.arange(n)+.5)*h[0],indexing="ij")
            initial=np.stack((np.sin(x+h[0]/2)*np.cos(y),-np.cos(x)*np.sin(y+h[1]/2),np.zeros_like(x)))
            exact=initial*np.exp(-2*mu/rho*r["time_s"]);ep=rho/4*(np.cos(2*x)+np.cos(2*y))*np.exp(-4*mu/rho*(r["time_s"]-dt/2));ep-=ep.mean()
            eigen=sum(4*np.sin(h[a]/2)**2/h[a]**2 for a in (0,1));semi=initial*np.exp(-mu/rho*eigen*r["time_s"])
            close(f["initial_faces_m_s"],initial);close(f["reference_velocity_faces_m_s"],exact);close(f["reference_pressure_pa"],ep)
            close(f["semidiscrete_reference_faces_m_s"],semi);states=f["accepted_faces_m_s"];vol=np.prod(h);residuals=[];defects=[];divs=[];convpowers=[];viscpowers=[];mom=[];energyinc=[];projinc=[]
            if len(states)!=len(history)+1:raise ValueError("missing accepted state")
            close(states[0],initial);close(states[-1],f["velocity_faces_m_s"])
            for i,(old,new) in enumerate(zip(states[:-1],states[1:])):
                mid=.5*(old+new);conv,force=mac_acceleration(mid,h,mu,rho)
                tentative=old+dt*(-conv+force/rho);check,p=projected(tentative,h,rho,dt)
                res=float(np.max(abs(check-new)));residuals.append(res)
                cp=-rho*vol*np.sum(mid*conv);vp=vol*np.sum(mid*force);pp=-vol*np.sum(mid*g(p,h))
                k0=.5*rho*vol*np.sum(old*old);k1=.5*rho*vol*np.sum(new*new)
                defect=k1-k0-dt*(cp+vp+pp);defects.append(abs(defect));divs.append(float(np.max(abs(d(new,h)))))
                convpowers.append(abs(cp));viscpowers.append(vp);mom.append(float(np.max(abs(rho*vol*np.sum(new-old,axis=(1,2,3))))));energyinc.append(k1-k0)
                projinc.append(.5*rho*vol*(np.sum(check*check)-np.sum(tentative*tentative)))
                record=history[i];close(record["time_s"],(i+1)*dt);close(record["nonlinear_residual_m_s"],res);close(record["energy_balance_defect_J"],defect)
                close(record["kinetic_after_J"],k1);close(record["convective_power_W"],cp);close(record["viscous_power_W"],vp);close(record["max_divergence_s_inv"],divs[-1])
            close(p,f["pressure_pa_3d"]);close(f["p_pa"],f["pressure_pa_3d"][0]);close(f["divergence_s_inv"],d(states[-1],h))
            computed=dict(velocity_l2=relative(states[-1],exact),velocity_linf=relative(states[-1],exact,True),
              pressure_l2=relative(f["pressure_pa_3d"],ep),pressure_linf=relative(f["pressure_pa_3d"],ep,True),
              temporal_velocity_l2=relative(states[-1],semi),scaled_divergence=max(divs),
              nonlinear_residual=max(residuals),energy_defect=max(defects),energy_increase=max(0.,max(energyinc)),
              momentum_change=max(mom),convective_power=max(convpowers),positive_viscous_power=max(0.,max(viscpowers)),
              pressure_failure=0. if all(q["projection"]["pressure_converged"] for q in history) else 1.,
              projection_energy_increase=max(0.,max(projinc)))
        else:
            n=c["nx"];dx=c["length_m"]/n;dt=c["time_step_s"];scale=dx/dt;tau=c["tau"]
            y,x=np.meshgrid(f["y_m"],f["x_m"],indexing="ij")
            C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
            # Reproduce the existing production lattice's declared float32 constants.
            W=np.array([4/9,*([1/9]*4),*([1/36]*4)],dtype=np.float32).astype(np.float64)
            states=f["accepted_populations"]
            rho_initial=1+3*.25*(np.cos(2*x)+np.cos(2*y))/scale**2
            u_initial=np.stack((np.sin(x)*np.cos(y),-np.cos(x)*np.sin(y)))/scale
            ci=np.einsum("qa,ayx->qyx",C,u_initial)
            init=W[:,None,None]*rho_initial*(1+3*ci+4.5*ci*ci-1.5*np.sum(u_initial*u_initial,axis=0))
            close(states[0],init)
            for old,new in zip(states[:-1],states[1:]):
                rho=old.sum(0);u=np.einsum("qyx,qa->ayx",old,C)/rho
                cu=np.einsum("qa,ayx->qyx",C,u)
                eq=W[:,None,None]*rho*(1+3*cu+4.5*cu*cu-1.5*np.sum(u*u,axis=0))
                post=old-(old-eq)/tau
                predicted=np.stack([np.roll(post[q],tuple(C[q,::-1]),axis=(0,1)) for q in range(9)])
                close(predicted,new,1e-12)
            final=states[-1];rho=final.sum(0);u=np.einsum("qyx,qa->ayx",final,C)/rho*scale;p=(rho-rho.mean())*scale**2/3
            eu=np.stack((np.sin(x)*np.cos(y),-np.cos(x)*np.sin(y)))*np.exp(-2*.1*r["time_s"])
            ep=.25*(np.cos(2*x)+np.cos(2*y))*np.exp(-4*.1*r["time_s"]);ep-=ep.mean()
            close(f["populations"],final);close(f["p_pa"],p);close(f["u_m_s"],u[0]);close(f["v_m_s"],u[1])
            close(f["reference_pressure_pa"],ep);close(f["reference_u_m_s"],eu[0]);close(f["reference_v_m_s"],eu[1])
            if len(states)!=len(history)+1:raise ValueError("missing LBM history")
            for i,state in enumerate(states[1:]):
                close(history[i]["time_s"],(i+1)*dt)
                close(history[i]["relative_mass_drift"],abs(state.sum()-states[0].sum())/states[0].sum())
            computed=dict(velocity_l2=relative(u,eu),velocity_linf=relative(u,eu,True),
                          pressure_l2=relative(p,ep),pressure_linf=relative(p,ep,True))
            if source_check:
                repo=Path(r["provenance"]["repository"])
                for name,digest in r["provenance"]["source_sha256"].items():
                    if sha(repo/name)!=digest:raise ValueError("LBM source changed")
        if set(computed)!={q["name"] for q in r["metrics"]}:raise ValueError("metric audit coverage incomplete")
        for q in r["metrics"]:
            close(computed[q["name"]],q["value"],2e-10)
            if q["name"] in ("velocity_l2","velocity_linf","pressure_l2","pressure_linf","pressure_gradient","pressure_curve_l2","wall_shear_linf","force_balance","temporal_velocity_l2") and q["limit"]!=.03:
                raise ValueError("physical accuracy gate changed")
            if q["passed"]!=(math.isfinite(computed[q["name"]]) and computed[q["name"]]<q["limit"]):
                raise ValueError("false metric badge")
        if r["passed"]!=all(q["passed"] for q in r["metrics"]):raise ValueError("false run badge")
        audited.append(dict(directory=r["directory"],raw_reconstruction_passed=True,benchmark_passed=r["passed"],steps=len(history)))
    regular=[r for r in summary["runs"] if r["role"] not in ("negative-control","comparison")]
    controls=[r for r in summary["runs"] if r["role"]=="negative-control"]
    orders={}
    for key,case,role,metric in [("channel_velocity","poiseuille","spatial","velocity_l2"),
      ("channel_pressure_gradient","poiseuille","spatial","pressure_gradient"),
      ("tg_velocity","taylor-green","spatial","velocity_l2"),("tg_pressure","taylor-green","spatial","pressure_l2"),
      ("tg_time","taylor-green","temporal","temporal_velocity_l2")]:
        runs=[r for r in summary["runs"] if r["case"]==case and r["role"]==role]
        vals=[next(q["value"] for q in r["metrics"] if q["name"]==metric) for r in runs]
        orders[key]=[math.log(a/b)/math.log(runs[i]["spacing_m"]/runs[i+1]["spacing_m"]) for i,(a,b) in enumerate(zip(vals,vals[1:]))]
        close(orders[key],summary["observed_orders"][key])
    second=all(1.8<p<2.2 for k in ("tg_velocity","tg_pressure","tg_time") for p in orders[k])
    passed=all(r["passed"] for r in regular) and all(not r["passed"] for r in controls) and second
    if summary["passed"]!=passed:raise ValueError("false suite qualification")
    return dict(schema="tensorfvm.independent-benchmark-audit/1",raw_reconstruction_passed=True,
       benchmark_passed=passed,cases=audited,summary_sha256=sha(directory/"summary.json"),
       auditor_sha256=sha(Path(__file__)),scope="NumPy analytic references, authoritative raw states and every accepted MAC/BGK step; no solver imports")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--skip-source-check",action="store_true",help="audit archived fields without requiring original source checkouts")
    args=parser.parse_args();print(json.dumps(audit(args.directory,not args.skip_source_check),indent=2))


if __name__=="__main__":main()
