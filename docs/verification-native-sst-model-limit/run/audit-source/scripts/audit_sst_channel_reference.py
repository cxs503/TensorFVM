#!/usr/bin/env python3
"""Replay independent reference fields using the separately implemented production closure."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from tensorfvm.native_sst3d import sst2003_closure


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);root=p.parse_args().run
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert summary['source_integrity_verified']
    for name,digest in json.loads((root/'sources.json').read_text()).items():assert sha(root/'source-snapshot'/name)==digest
    assert sha(root/'model-source.html')==cfg['model_source_sha256']
    meta=json.loads((root/'dns-reference/provenance.json').read_text())
    for ref in meta['files'].values():assert sha(root/'dns-reference'/ref['csv'])==ref['csv_sha256']
    rows=[]
    for row in json.loads((root/'cases.json').read_text()):
        path=root/f"reference-n{row['n']}-fields.npz";assert sha(path)==row['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            y=f['y'];faces=f['faces'];volume=np.diff(faces);k=f['k'];w=f['omega'];nu=cfg['nu']
            assert np.all(k>=0) and np.all(w>0)
            def gradient(v,bd):return np.gradient(np.r_[bd,v,v[-1]],np.r_[0,y,2-y[-1]],edge_order=2)[1:-1]
            G=np.zeros((len(y),3,3));G[:,0,1]=f['shear']
            gk=np.zeros((len(y),3));gk[:,1]=gradient(k,0.)
            gw=np.zeros((len(y),3));gw[:,1]=gradient(w,row['wall_omega'])
            c=sst2003_closure(k,w,y,nu,G,gk,gw)
            np.testing.assert_allclose(c['nu_t'],f['nu_t'],rtol=1e-12,atol=1e-16)
            np.testing.assert_allclose((nu+c['nu_t'])*f['shear'],1-y,rtol=1e-12,atol=1e-14)
            uv=np.r_[0,np.cumsum(volume*f['shear'])]
            np.testing.assert_allclose(f['face_velocity'],uv,rtol=1e-12)
            np.testing.assert_allclose(f['velocity'],(uv[:-1]+uv[1:])/2,rtol=1e-12)
            cf=float(2/(volume@f['velocity'])**2);np.testing.assert_allclose(cf,row['Cf'],rtol=1e-12)
            scalar={}
            for name,value,sigma,production,destruction,boundary in [
                ('k',k,c['sigma_k'],c['production_limited'],c['k_destruction'],0.),
                ('omega',w,c['sigma_omega'],c['omega_production']+c['cross_diffusion_source'],c['omega_destruction'],row['wall_omega'])]:
                coeff=nu+sigma*c['nu_t'];flux=np.zeros(len(faces));flux[0]=coeff[0]*(value[0]-boundary)/y[0]
                flux[1:-1]=np.diff(value)/((faces[1:-1]-y[:-1])/coeff[:-1]+(y[1:]-faces[1:-1])/coeff[1:])
                source=volume*(production-destruction);residual=-np.diff(flux)-source
                scale=abs(flux[:-1])+abs(flux[1:])+volume*(abs(production)+abs(destruction))
                relative=float(np.max(abs(residual)/np.maximum(scale,1e-30)))
                replay_difference=float(np.max(abs(residual-f[name+'_defect'])/np.maximum(scale,1e-30)))
                assert replay_difference<5e-13
                if row['converged']:assert relative<1e-10
                scalar[name]=dict(production_closure_relative_local_residual=relative,
                    stored_residual_replay_relative_difference=replay_difference,
                    absolute_local_residual=float(np.max(abs(residual))),global_balance=float(residual.sum()))
            history=json.loads((root/f"reference-n{row['n']}-history.json").read_text())
            if row['converged']:
                assert len(history)>=2 and all(max(v['current_relative_local_residual'],v['unrelaxed_change'])<1e-10 for v in history[-2:])
            rows.append(dict(n=row['n'],Cf=cf,scalar=scalar))
    repo=Path(__file__).resolve().parents[1];sources={}
    for name in ('scripts/audit_sst_channel_reference.py','src/tensorfvm/native_sst3d.py'):
        dst=root/'audit-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,dst);sources[name]=sha(dst)
    result=dict(source_and_field_hashes_verified=True,reference_equations_replayed_with_production_closure=True,
        audit_sources=sources,cases=rows,native_absolute_ledger_gates_replaced=False,physical_suboff_benchmark_passed=False,
        scope='Cross-check of a separately solved 1D model; fine-grid omega absolute cancellation is reported, not certified under 3D ledger gates')
    (root/'artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
