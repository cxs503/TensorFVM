#!/usr/bin/env python3
"""Audit complete restart histories without equating termination with convergence."""
import argparse,json
from pathlib import Path
from tensorfvm.openfoam_history import convergence_audit
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);a=p.parse_args()
result=convergence_audit(a.case)
(a.case/'convergence-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
