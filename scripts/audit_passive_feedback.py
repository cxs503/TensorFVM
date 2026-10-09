import argparse,json
from pathlib import Path
from tensorfvm.passive_feedback_audit import audit,audit_balanced
p=argparse.ArgumentParser();p.add_argument('--fem-root',required=True);p.add_argument('--lbm-root',required=True);a=p.parse_args();r=audit({'TensorFEM':Path(a.fem_root),'TensorLBM':Path(a.lbm_root)});r['balanced_local']=audit_balanced(Path(a.lbm_root));out=Path('docs/external-boundary/fifth-round');out.mkdir(parents=True,exist_ok=True);(out/'passive-report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['cases'],indent=2))
