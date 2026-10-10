#!/usr/bin/env python3
"""Common Metric gates for a source-bound 3-D cylinder run, not LES approval."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import Metric,evaluate,write_json,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();d=a.input
    if (d/'acceptance.json').exists():p.error('acceptance evidence already exists; preserve prior audit')
    summary=json.loads((d/'summary.json').read_text());history=json.loads((d/'history.json').read_text());config=json.loads((d/'config.json').read_text())['config']
    if not history:p.error('no completed physical step to audit')
    # Independent final flux replay uses checkpoint owner/neighbor/actual volumes.
    z=np.load(d/'checkpoint.npz',allow_pickle=False);o=z['owner'];n=z['neighbor'];q=z['flux'];v=z['volumes'];mass=np.bincount(o,weights=q,minlength=len(v))-np.bincount(n[n>=0],weights=q[n>=0],minlength=len(v))
    records,passed=evaluate([
        Metric('recorded_all_step_continuity',max(h['continuity']for h in history),1e-8,'all completed steps, actual shared-face divergence'),
        Metric('independent_final_face_continuity',float(np.max(np.abs(mass/v))),1e-8,'checkpoint-native face-flux continuity replay'),
        Metric('advective_cfl',max(h['cfl']for h in history),1.,'all completed steps'),
        Metric('explicit_diffusion_screen',max(h['diffusion_cfl']for h in history),.5,'conservative screen, not general nonorthogonal stability proof'),
        Metric('momentum_ledger',max(h['momentum_ledger']for h in history),1e-10,'recorded actual step boundary/cell momentum balance'),
        Metric('boundary_mass_imbalance',max(abs(h['mass_imbalance'])for h in history)/(config['inlet_velocity']*config['height']*config['span']),1e-8,'normalized physical-domain boundary flux')])
    resolution,resolved=evaluate([Metric('instantaneous_wall_yplus',summary['wall_resolution']['yplus_max'],1.,'instantaneous startup wall indicator; long-time resolution still requires stationary statistics')])
    scripts=[Path(__file__).resolve()]+[Path(mod.__file__).resolve()for name,mod in list(sys.modules.items())if(name=='tensorfvm'or name.startswith('tensorfvm.'))and getattr(mod,'__file__',None)and Path(mod.__file__).suffix=='.py']
    repo=Path(__file__).resolve().parents[1];bindings={}
    for src in sorted(set(scripts)):
        target=d/'acceptance-source-snapshot'/src.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);bindings[str(src.relative_to(repo))]=sha(target)
    write_json(d/'acceptance.json',dict(schema='tensorfvm.common-3d-case-acceptance/1',common_module='tensorfvm.verification.core',numerical_metrics=records,numerical_passed=passed,
        resolution_metrics=resolution,instantaneous_resolution_passed=resolved,physical_accuracy_qualified=False,
        physical_metrics=dict(Cd_relative_error=None,St_relative_error=None,Cpb_relative_error=None,profile_relative_error=None,grid_independence=None,time_independence=None,stationarity=None),
        reason='startup stability and resolution audit only; no qualified long-time CFD statistics',source_sha256=bindings,input_manifest_sha256=sha(d/'manifest.json')))
    manifest=json.loads((d/'manifest.json').read_text());manifest['acceptance_source_sha256']=bindings;manifest['artifact_sha256']={str(f.relative_to(d)):sha(f)for f in d.rglob('*')if f.is_file()and f.name!='manifest.json'};write_json(d/'manifest.json',manifest)
    print(json.dumps(dict(numerical_passed=passed,instantaneous_resolution_passed=resolved,physical_accuracy_qualified=False)))
if __name__=='__main__':main()
