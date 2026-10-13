#!/usr/bin/env python3
"""Rebuild current steady RANS operators and audit saved SUBOFF field ledgers.

This independently sums flux/work from reconstructed production operators; it
is not an independent discretization, experiment, or physical qualification.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision-sensitivity',action='store_true',help='compare saved extended scalar states with their production float64 cast, at identical frozen closure coefficients')
    a=p.parse_args()
    root=a.run;cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['initialized'] or not summary['source_integrity_verified']:raise ValueError('verified initialized run required')
    if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('saved field changed')
    for relative,sha in json.loads((root/'sources.json').read_text()).items():
        if digest(root/'source-snapshot'/relative)!=sha:raise ValueError('producer source changed')
    for relative,sha in cfg['initial_bindings'].items():
        if digest(root/('initial-'+relative))!=sha:raise ValueError('initial input changed')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output
    write_json(out/'settings.json',dict(run=str(root),scope=__doc__,bindings={name:digest(root/name) for name in
        ('settings.json','summary.json','fields.npz','initial-input.msh','sources.json')},
        precision_sensitivity=a.precision_sensitivity,physical_suboff_benchmark_passed=False))
    m=NativeMesh3D.from_gmsh(root/'initial-input.msh',triangulate_faces=True)
    f=np.load(root/'fields.npz',allow_pickle=False);boundary=m.neighbor<0;inside=~boundary;wall=m.patch_labels=='wall'
    fixed=boundary&(m.patch_labels!='outlet');velocity=np.tile([1.,0.,0.],(len(m.owner),1));velocity[wall]=0.
    flow=NativeHybridNavierStokes3D(m,fixed,velocity,viscosity=cfg['nu'],convection_order=2,backflow_velocity=[1,0,0])
    policy=cfg.get('scalar_work_balance_policy','absolute')
    s=NativeHybridRans3D(flow,f['wall_distance'],cfg['nu'],k_dirichlet_mask=fixed,omega_dirichlet_mask=fixed,
        k_boundary=f['k_boundary'],omega_boundary=f['omega_boundary'],k_inflow=cfg['k_inflow'],omega_inflow=cfg['omega_inflow'],
        scalar_correction=cfg['scalar_correction'],scalar_work_balance_policy=policy)
    for name,value in [('owner',m.owner),('neighbor',m.neighbor),('vertices',m.vertices),('area_vectors',m.area_vectors)]:
        np.testing.assert_array_equal(f[name],value)
    s.restore_accepted_fields(f);current=s.current_residual();r=s.result();q=flow.volume_flux()
    def divergence(flux):
        answer=np.zeros((flow.N,)+flux.shape[1:],dtype=flux.dtype)
        np.add.at(answer,m.owner,flux);np.add.at(answer,m.neighbor[inside],-flux[inside]);return answer
    momentum_flux=r['momentum_flux_integrated'];traction=r['total_traction_integrated']
    local=divergence(momentum_flux);force=-traction[wall].sum(axis=0)
    balances=dict(mass_divergence_max=float(np.max(abs(divergence(q)/m.cell_volumes))),
        local_momentum_balance_max=float(np.max(abs(local))),global_momentum_balance_max=float(np.max(abs(momentum_flux[boundary].sum(axis=0)))),
        interface_traction_defect_max=float(np.max(abs(traction[inside]+r['neighbor_total_traction_integrated']))),
        energy_balance_error=float(r['energy_balance_error']),wall_force=force.tolist(),Cd_L2=float(2*force[0]),
        pressure_work=float(r['pressure_work']),viscous_energy=float(r['energy_form']),convection_power=float(r['convection_power']),external_work=float(r['external_work']))
    raw=dict(volume_flux=q,total_traction_integrated=traction,neighbor_total_traction_integrated=r['neighbor_total_traction_integrated'],
        momentum_flux_integrated=momentum_flux,local_momentum_balance=local)
    scalar={};c=current['transport']['closure']
    for name in ('k','omega'):
        op=NativeHybridScalar3D(m,cfg['nu']+c['sigma_'+name]*c['nu_t'],dirichlet_mask=fixed,
            correction=cfg['scalar_correction'],work_balance_policy=policy)
        state=np.r_[np.asarray(f[name],float),np.asarray(f[name+'_faces'],float)]
        linear=copy.copy(op);linear.mmp=None
        if op.mmp is not None:linear.diffusion,_=op.mmp.operator(state)
        ledger=linear._solution_result(state,q,c[name+'_source'],np.zeros(flow.N),np.zeros(flow.N),0.,cfg[name+'_inflow'],0.)
        saved=current['transport'][name]
        np.testing.assert_allclose(ledger['cell_balance'],saved['local_shared_flux_balance'],rtol=1e-12,atol=1e-12)
        scalar[name]={key:ledger[key] for key in ('cell_balance_max','interface_flux_defect_max','global_balance',
            'work_balance','work_balance_scale','relative_work_balance','absolute_work_gate_passed','work_balance_policy')}
        scalar[name].update(relative_current_residual=saved['relative_residual'],current_steady_ledger_passed=op.ledger_acceptable(ledger))
        for key in ('total_flux','neighbor_total_flux','cell_balance','work_balance_cell_terms','work_balance_interface_terms'):
            raw[name+'_'+key]=ledger[key]
        if a.precision_sensitivity:
            preserved=np.r_[f[name],f[name+'_faces']]
            extended=copy.copy(op);extended.mmp=None
            if op.mmp is not None:extended.diffusion,_=op.mmp.operator(preserved)
            sensitivity=extended._solution_result(preserved,q,c[name+'_source'],np.zeros(flow.N),np.zeros(flow.N),0.,cfg[name+'_inflow'],0.)
            scalar[name]['state_precision_sensitivity']=dict(
                scope='diagnostic only: identical current float64 SST closure; changing state precision also rebuilds its nonlinear MMP coefficients; not a full extended-precision SST solve',
                saved_dtype=str(preserved.dtype),saved_precision_bits=int(np.finfo(preserved.dtype).nmant+1),
                maximum_state_cast_difference=float(np.max(abs(preserved-state))),
                preserved_state_ledger={key:sensitivity[key] for key in ('cell_balance_max','interface_flux_defect_max',
                    'global_balance','work_balance','relative_work_balance')},
                maximum_local_ledger_change=float(np.max(abs(sensitivity['cell_balance']-ledger['cell_balance']))),
                maximum_cell_equation_change=float(np.max(abs(sensitivity['cell_equation_extended']-ledger['cell_equation_extended']))))
            raw[name+'_preserved_precision_cell_balance']=sensitivity['cell_balance']
            raw[name+'_preserved_precision_cell_equation']=sensitivity['cell_equation_extended']
    np.savez_compressed(out/'reconstructed-ledgers.npz',**raw);evidence.verify_sources()
    write_json(out/'audit.json',dict(source_integrity_verified=True,producer_sources_verified=True,
        reconstructed_sha256=digest(out/'reconstructed-ledgers.npz'),producer_solver_converged=summary.get('solver_converged',False),
        current_equations={k:v for k,v in current.items() if k!='transport'},flow=balances,scalar=scalar,
        numerical_qualification=False,physical_suboff_benchmark_passed=False,
        qualification='diagnostic audit only; all convergence, mesh and physical-reference gates remain required'))
    print(json.loads((out/'audit.json').read_text()),flush=True)


if __name__=='__main__':main()
