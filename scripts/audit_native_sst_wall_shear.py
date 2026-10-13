#!/usr/bin/env python3
"""Rebuild coupled wall-SST equations from saved fields, without re-solving them."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.verification.native_spalding import spalding_couette_solver
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.run
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if not summary['source_integrity_verified'] or sha(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('changed/unverified raw field')
    sources=json.loads((root/'sources.json').read_text())
    if 'src/tensorfvm/native_sst_wall3d.py' not in sources:raise ValueError('producer wall-formula snapshot missing')
    for name,digest in sources.items():
        if sha(root/'source-snapshot'/name)!=digest:raise ValueError('producer source changed')
    evidence=NativeRunEvidence(a.output,__file__)
    with np.load(root/'fields.npz') as f:
        m,flow,*_=spalding_couette_solver(cfg['layers'],mu=cfg['initial_bulk_viscosity'],nu=cfg['nu'])
        for name in ('vertices','owner','neighbor','area_vectors','face_centers','cell_centers','cell_volumes'):
            np.testing.assert_array_equal(f[name],getattr(m,name))
        s=NativeHybridRans3D(flow,f['wall_distance'],cfg['nu'],k_dirichlet_mask=f['k_dirichlet'],
            omega_dirichlet_mask=f['omega_dirichlet'],k_boundary=f['k_boundary'],omega_boundary=f['omega_boundary'],
            k_inflow=cfg['k_inflow'],omega_inflow=cfg['omega_inflow'],scalar_correction=cfg['scalar_correction'],
            scalar_precision=cfg['scalar_precision'],scalar_work_balance_policy=cfg['scalar_work_balance_policy'],
            wall_treatment=cfg['wall_treatment'])
        s.restore_accepted_fields(f)
        r=s.result();current=s.current_residual(with_ledgers=True);accepted=s._steady_ledger_acceptance(current,flow.result())
        np.testing.assert_allclose(r['total_traction_integrated'],f['total_traction_integrated'],rtol=1e-12,atol=1e-12)
        inside=m.neighbor>=0
        def div(value):
            out=np.zeros((flow.N,)+value.shape[1:],dtype=value.dtype)
            np.add.at(out,m.owner,value);np.add.at(out,m.neighbor[inside],-value[inside]);return out
        flow_check=dict(mass_divergence_max=float(np.max(abs(div(r['flux'])/m.cell_volumes))),
            local_momentum_balance_max=float(np.max(abs(div(r['momentum_flux_integrated'])))),
            global_momentum_balance_max=float(np.max(abs(r['momentum_flux_integrated'][~inside].sum(0)))))
        scalar={};arrays={};closure=current['transport']['closure'];wall=closure['wall_model']
        for name in ('k','omega'):
            ledger=current['transport'][name]['steady_ledger'];reaction=ledger['cell_constraint_reaction']
            reconstructed=div(ledger['total_flux'])-m.cell_volumes*closure[name+'_source']-reaction
            np.testing.assert_allclose(reconstructed,ledger['cell_balance'],rtol=0.,atol=1e-14)
            np.testing.assert_allclose(reconstructed,f[name+'_cell_balance'],rtol=0.,atol=1e-14)
            work=np.sum(ledger['work_balance_cell_terms'],dtype=np.longdouble)+np.sum(ledger['work_balance_interface_terms'],dtype=np.longdouble)
            np.testing.assert_allclose(work,ledger['work_balance'],rtol=1e-12,atol=1e-16)
            scalar[name]=dict(local_balance_max=float(np.max(abs(reconstructed))),constraint_reaction_sum=float(reaction.sum()),
                constraint_reaction_max=float(np.max(abs(reaction))),constraint_work=float(ledger['cell_constraint_work']),
                relative_work_balance=ledger['relative_work_balance'],cell_constraint_error=ledger['cell_constraint_error'])
            arrays[name+'_cell_balance']=reconstructed;arrays[name+'_constraint_reaction']=reaction
        for name in ('cells','omega','production','physical_wall_gradient_magnitude'):
            np.testing.assert_array_equal(wall[name],f['wall_'+name]);arrays['wall_'+name]=wall[name]
        np.testing.assert_array_equal(r['nu_t'],f['nu_t'])
        independent=dict(flow=flow_check,scalar=scalar,
            viscosity_min=float(flow.viscous.mu.min()),viscosity_max=float(flow.viscous.mu.max()),
            k_min=float(s.k.min()),k_max=float(s.k.max()),nu_t_max=float(s.closure['nu_t'].max()),
            wall_G_min=float(wall['production'].min()),wall_G_max=float(wall['production'].max()))
    np.savez_compressed(a.output/'reconstructed.npz',**arrays)
    evidence.verify_sources();write_json(a.output/'audit.json',dict(source_integrity_verified=True,
        bindings={name:sha(root/name) for name in ('fields.npz','settings.json','summary.json','sources.json')},
        current_equations={key:value for key,value in current.items() if key!='transport'},
        current_steady_ledgers=accepted,independent=independent,
        reconstructed_sha256=sha(a.output/'reconstructed.npz'),
        scope='saved-state reconstruction using production operators and independent flux/work sums; not an independent discretization or experiment',
        physical_suboff_benchmark_passed=False,physical_channel_benchmark_passed=False))
    print(independent,flush=True)


if __name__=='__main__':main()
