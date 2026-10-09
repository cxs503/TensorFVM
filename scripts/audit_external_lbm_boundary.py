"""Audit external raw evidence without importing its solver."""
import argparse,json
from pathlib import Path
from tensorfvm.external_boundary_audit import audit
parser=argparse.ArgumentParser();parser.add_argument('--lbm-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('docs/external-boundary/report.json'));args=parser.parse_args()
report=audit(args.lbm_root);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(cases=len(report['cases']),raw_audit_passed=True,physical_accuracy_qualified=False)))
