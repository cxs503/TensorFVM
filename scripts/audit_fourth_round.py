import argparse,json
from pathlib import Path
from tensorfvm.fourth_round_audit import audit
p=argparse.ArgumentParser();p.add_argument('--lbm-root',required=True);p.add_argument('--dem-root',required=True);p.add_argument('--fem-root',required=True);p.add_argument('--output',default='docs/external-boundary/fourth-round/report.json');a=p.parse_args();r=audit(a.lbm_root,a.dem_root,a.fem_root);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('cases','inputs_sha256')}));print('Cases:',len(r['cases']))
