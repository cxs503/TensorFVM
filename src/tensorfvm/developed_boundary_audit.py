"""Independent long-window force/drive audit, separate from historical evidence."""
import json
import math
from pathlib import Path
import numpy as np
from .external_boundary_audit import sha, require, close, mask, field, scales


def steady_gates(drift, reaction, drive):
    require(math.isfinite(drift) and drift >= 0 and math.isfinite(reaction) and math.isfinite(drive) and abs(drive)>0,'invalid steady metrics')
    imbalance=abs(reaction-drive)/abs(drive)
    return dict(window_drift_passed=bool(drift<.01),net_acceleration_passed=bool(imbalance<.01),
                reaction_drive_relative_imbalance=imbalance,steady_qualified=bool(drift<.01 and imbalance<.01))


def reconstruct(raw):
    dt=raw['dt_s'];dx=raw['dx_m'];window=raw['steady_window_s'];count=round(window/dt)
    force=np.asarray(raw['force_history_N'],dtype=float)
    steps=round(raw['duration_s']/dt)
    require(len(force)==steps and np.isfinite(force).all() and count>0 and len(force)>=2*count,'invalid force samples')
    solid=mask((raw['n'],raw['n']),np.asarray(raw['center_m'])/dx,raw['radius_m']/dx)
    m0,p0=field(raw['initial_population'],solid);mf,pf=field(raw['final_population'],solid)
    unit=scales(dx,dt,raw['rho_kg_m3'],raw['thickness_m'],raw['nu_m2_s'],0.,2*raw['radius_m'])
    fluidmass=m0*unit['lattice_mass_kg']
    midpoint=(np.arange(steps)+.5)*dt
    drive_impulses=fluidmass*raw['acceleration_m_s2']*np.sin(np.minimum(midpoint/raw['ramp_s'],1)*math.pi/2)**2*dt
    force_impulses=force*dt
    initial_p=p0[0]*unit['lattice_impulse_Ns']
    start=initial_p+float(np.sum(drive_impulses[:-count])-np.sum(force_impulses[:-count]))
    end=initial_p+float(np.sum(drive_impulses)-np.sum(force_impulses))
    drive=float(np.sum(drive_impulses[-count:])/window);reaction=float(np.sum(force_impulses[-count:])/window)
    late=float(np.mean(force[-count:]));previous=float(np.mean(force[-2*count:-count]));drift=abs(late/previous-1)
    gates=steady_gates(drift,reaction,drive)
    require(abs(mf-m0)/m0<1e-11,'relative mass drift exceeds 1e-11')
    close(end,pf[0]*unit['lattice_impulse_Ns'],1e-8,'raw momentum vs force/drive integration')
    close(np.sum(force_impulses),raw['solid_impulse_Ns'][0],1e-8,'body impulse')
    close(np.sum(drive_impulses),raw['drive_impulse_Ns'][0],1e-8,'drive impulse')
    close(drift,raw['late_window_drift'],1e-12,'force window drift')
    for key in ('window_drift_passed','net_acceleration_passed','steady_qualified'):
        require(type(raw.get(key)) is bool and raw[key]==gates[key],'incorrect qualification '+key)
    return dict(**unit,**gates,fluid_mass_kg=fluidmass,late_mean_drive_force_N=drive,late_mean_reaction_force_N=reaction,
        fluid_momentum_window_start_x_kg_m_s=start,fluid_momentum_window_end_x_kg_m_s=end,
        fluid_momentum_rate_x_N=(end-start)/window,late_window_relative_drift=drift,
        raw_momentum_residual_Ns=abs(end-pf[0]*unit['lattice_impulse_Ns']),raw_relative_mass_error=abs(mf-m0)/m0,raw_audit_passed=True)


def audit(root):
    root=Path(root);path=root/'docs/assets/curved-boundary/developed-study.json';study=json.loads(path.read_text())
    bindings={str(path.relative_to(root)):sha(path)}
    for source,digest in study['source_sha256'].items():
        require(sha(root/source)==digest,'source hash mismatch '+source);bindings[source]=digest
    cases=[]
    for entry in study['cases']:
        rawpath=path.parent/entry['file'];require(sha(rawpath)==entry['sha256'],'raw hash mismatch')
        bindings[str(rawpath.relative_to(root))]=entry['sha256'];result=reconstruct(json.loads(rawpath.read_text()))
        for key in ('window_drift_passed','net_acceleration_passed','steady_qualified'):
            require(entry[key]==result[key],'manifest qualification mismatch')
        cases.append(dict(file=entry['file'],**result))
    return dict(schema='tensorfvm.developed-boundary-cross-audit/1',cases=cases,source_and_raw_sha256=bindings,
        auditor_source_sha256=sha(__file__),shared_audit_source_sha256=sha(Path(__file__).with_name('external_boundary_audit.py')),
        thresholds=dict(force_window_drift=.01,reaction_drive_relative_imbalance=.01),
        fvm_same_boundary_simulation_performed=False,physical_accuracy_qualified=False,
        temporal_convergence_qualified=False,eos_note='dt refinement changes SI sound speed; combined dt/compressibility sensitivity')
