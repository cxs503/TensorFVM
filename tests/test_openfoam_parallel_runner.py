"""Guard against silent continuation from stale OpenFOAM checkpoints."""
from pathlib import Path
import json
import subprocess
import sys
import pytest


@pytest.mark.parametrize('start,index,period,partial', [(0,0,200,False),(100,100,300,False),(0,0,200,True)])
def test_block_persists_endpoint_and_rejects_missing_checkpoint(tmp_path,start,index,period,partial):
    case = tmp_path / 'case'
    (case / 'system').mkdir(parents=True)
    checkpoint=case / 'processor0' / str(start)
    (checkpoint / 'uniform').mkdir(parents=True)
    (checkpoint / 'U').touch()
    (checkpoint / 'uniform/time').write_text(f'index {index};\n')
    (case / 'log.checkMesh').write_text('Mesh OK.\n')
    (case / 'log.snappyHexMesh').write_text('wall 100 6 0.001 100\n')
    control = case / 'system/controlDict'
    control.write_text('startFrom startTime; endTime 600;\nwriteInterval 600;\n'
                       'functions { forces { writeInterval 1; } yPlus { writeInterval 600; } }\n')
    # A solver may exit zero without producing a checkpoint: reject it.
    wrapper = tmp_path / 'fake-foam'
    wrapper.write_text(f'''#!{sys.executable}
from pathlib import Path
if {partial}:
    field=Path({str(case/'processor0'/str(start+200)/'U')!r})
    field.parent.mkdir()
    field.write_text('velocity without pressure or turbulence fields')
''')
    wrapper.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'scripts/run_openfoam_parallel.py'
    result = subprocess.run([sys.executable, str(script), '--case', str(case),
                             '--foam-wrapper', str(wrapper), '--ranks', '1',
                             '--cpu-set', '0', '--block', '200', '--max-iteration', '600'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert f'without complete checkpoint {start+200}' in result.stderr
    assert control.read_text().count(f'writeInterval {period};') == 2
    assert 'writeInterval 1;' in control.read_text()
    assert not (case / 'convergence-audit.json').exists()


def test_iteration_limit_with_saved_fields_is_not_success(tmp_path):
    case = tmp_path / 'case'
    (case / 'system').mkdir(parents=True)
    (case / 'processor0/0').mkdir(parents=True)
    (case / 'processor0/0/U').touch()
    (case / 'log.checkMesh').write_text('Mesh OK.\n')
    (case / 'log.snappyHexMesh').write_text('wall 100 6 0.001 100\n')
    (case / 'system/controlDict').write_text(
        'startFrom startTime; endTime 600; writeInterval 600;\n')
    wrapper = tmp_path / 'fake-foam'
    wrapper.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
if 'foamRun' in sys.argv:
    case=Path(sys.argv[sys.argv.index('-case')+1])
    (case/'processor0/200').mkdir()
    for field in ['U','p','k','omega','nut','phi']:
        (case/'processor0/200'/field).write_text('checkpoint fixture')
    for function,file,columns in [('forceCoeffs','forceCoeffs.dat',6),('forces','forces.dat',13)]:
        folder=case/'postProcessing'/function/'0'
        folder.mkdir(parents=True)
        (folder/file).write_text('200 '+' '.join(['0.1']*(columns-1))+'\\n')
''')
    wrapper.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'scripts/run_openfoam_parallel.py'
    result = subprocess.run([sys.executable, str(script), '--case', str(case),
                             '--foam-wrapper', str(wrapper), '--ranks', '1',
                             '--cpu-set', '0', '--block', '200', '--max-iteration', '200'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'without numerical convergence' in result.stderr
    audit = json.loads((case / 'convergence-audit.json').read_text())
    assert not audit['numerically_converged']
    assert not audit['physical_benchmark_passed']
