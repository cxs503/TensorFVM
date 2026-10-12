#!/usr/bin/env python3
"""Run a prepared Foundation 12 case with explicit mesh and layer gates.

The default performs meshing only. --solve enables the configured pilot run;
execution and mesh quality never establish physical benchmark acceptance.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--foam-wrapper', default='tensorfvm-foam12')
p.add_argument('--solve', action='store_true')
a = p.parse_args()
case = a.case.resolve()
if (case/'constant/polyMesh').exists():
    p.error('refusing to overwrite an existing mesh; prepare a fresh case')

def run(command, log, *extra):
    with (case/log).open('w') as stream:
        subprocess.run([a.foam_wrapper, command, '-case', str(case), *extra],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return (case/log).read_text()

run('blockMesh', 'log.blockMesh')
run('snappyHexMesh', 'log.backgroundRefinement', '-dict',
    str(case/'system/backgroundRefinementDict'), '-overwrite')
mesh_log = run('snappyHexMesh', 'log.snappyHexMesh', '-overwrite')
standard = run('checkMesh', 'log.checkMesh')
extended = run('checkMesh', 'log.checkMesh.extended', '-allTopology', '-allGeometry')
# checkMesh can return zero even when it prints "Failed ... mesh checks".
quality_ok = 'Mesh OK.' in standard and 'Failed ' not in standard
rows = re.findall(r'^wall\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$', mesh_log, re.M)
layers = float(rows[-1][1]) if rows else None
layer_ok = layers is not None and layers >= 5
status = dict(standard_mesh_ok=quality_ok, extended_mesh_ok='Mesh OK.' in extended,
              mean_layers=layers, mean_layer_gate=layer_ok,
              solve_requested=a.solve, solver_completed=False,
              physical_benchmark_passed=False)
(case/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')
if not quality_ok or not layer_ok:
    raise SystemExit('Mesh/layer gate failed; inspect run-status.json and logs.')
if a.solve:
    run('foamRun', 'log.foamRun')
    status['solver_completed'] = True
    (case/'run-status.json').write_text(json.dumps(status, indent=2)+'\n')
print(json.dumps(status, indent=2))
