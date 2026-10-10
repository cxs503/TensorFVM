"""Export a completed checkpoint with its archived producer, without advancing it."""
import argparse
import json
import shutil
import sys
import hashlib
import types
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be empty')
    producer = json.loads((run/'producer.json').read_text())
    archive = run/'source-snapshot'
    for name, expected in producer['source_sha256'].items():
        if hashlib.sha256((archive/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived producer mismatch: '+name)
    root = Path(__file__).resolve().parents[1]
    restored = {}
    # These imports were omitted by the old snapshot inventory. They are
    # unchanged tracked files: recover their exact git-base bytes explicitly.
    for name in ['src/tensorfvm/external_anderson.py', 'src/tensorfvm/external_coupled.py']:
        if name in producer['source_sha256']:
            continue
        content = subprocess.check_output(['git','show',producer['git_base']+':'+name],cwd=root)
        if (root/name).read_bytes() != content:
            raise ValueError('Unarchived dependency changed since producer base: '+name)
        restored[name] = content
    out.mkdir(parents=True, exist_ok=True)
    for name in ['checkpoint.npz', 'mesh.msh', 'config.json', 'producer.json', 'mesh-quality.json']:
        shutil.copy2(run/name, out/name)
    for source in run.glob('mesh-producer-*.json'):
        shutil.copy2(source,out/source.name)
    shutil.copytree(archive, out/'source-snapshot', dirs_exist_ok=True)
    archive = out/'source-snapshot'
    for name, content in restored.items():
        (archive/name).write_bytes(content)
    # Import the numerical producer from the checkpoint's immutable snapshot.
    sys.path.insert(0, str(archive/'src'))
    # The producer archive intentionally omits unrelated 3-D modules that the
    # package initializer imports. Load the archived numerical modules directly.
    package = types.ModuleType('tensorfvm')
    package.__path__ = [str(archive/'src/tensorfvm')]
    sys.modules['tensorfvm'] = package
    import numpy as np
    import torch
    from dataclasses import replace
    from tensorfvm.solver import SolverConfig
    from tensorfvm.multi_element_flow import MultiElementFlowSolver
    from tensorfvm.benchmark_30p30n import export_result
    from tensorfvm.verification.core import sha, write_json, artifact_manifest
    settings = producer['numerical_settings']
    base = MultiElementFlowSolver
    if settings.get('coupled'):
        from tensorfvm.multi_element_coupled import CoupledMultiElementSolver
        base = CoupledMultiElementSolver
    if settings.get('pressure_gradient') == 'least-squares':
        from tensorfvm.multi_element_least_squares import LeastSquaresMultiElementSolver, LeastSquaresCoupledMultiElementSolver
        base = LeastSquaresCoupledMultiElementSolver if settings.get('coupled') else LeastSquaresMultiElementSolver
    if settings.get('pressure_gradient') == 'gauss-skew-corrected':
        from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver, SkewPressureCoupledMultiElementSolver
        base = SkewPressureCoupledMultiElementSolver if settings.get('coupled') else SkewPressureMultiElementSolver
    if settings.get('sa_newton'):
        from tensorfvm.sa_newton import NewtonSaMixin
        class ArchivedSolver(NewtonSaMixin, base):
            pass
        base = ArchivedSolver
    torch.set_num_threads(1)
    # Read only the copied, atomically replaced complete checkpoint.
    with np.load(out/'checkpoint.npz') as checkpoint:
        history = json.loads(str(checkpoint['history'].item()))
        if len(history) != int(checkpoint['iteration']):
            raise ValueError('Checkpoint history mismatch')
        config = SolverConfig(**json.loads((out/'config.json').read_text()))
        config = replace(config, mesh_file=str(out/'mesh.msh'), max_iterations=len(history))
        if len(history) >= 100 or producer.get('initialization'):
            config = replace(config, pseudo_time_step=None)
        solver = base(config)
        for key, target in [('velocity', solver.velocity), ('pressure', solver.p), ('mass_flux', solver.mass_flux), ('nu_tilde', solver.nu_tilde)]:
            value = checkpoint[key]
            if value.shape != tuple(target.shape) or not np.isfinite(value).all():
                raise ValueError('Invalid checkpoint field: '+key)
            target.copy_(torch.from_numpy(value))
    solver.turbulent_kinematic_viscosity = solver._sa_eddy_viscosity(solver.nu_tilde)
    solver.history = history
    solver.convection_blend = history[-1].get('convection_blend', 0.)
    solver.converged = False
    export_result(solver.solve(), out)
    if hasattr(solver, 'reconstructed_pressure_faces'):
        import csv
        face_pressure=solver.reconstructed_pressure_faces(solver.p)
        with (out/'wall-face-pressure.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['face','p'])
            writer.writerows((int(i),float(face_pressure[i])) for i in torch.nonzero(solver.mesh.masks['wall']).flatten())
    note = dict(iterations=len(history), converged=False, physics_accepted=False,
                operation='Export completed checkpoint; zero additional solver steps',
                checkpoint_sha256=sha(out/'checkpoint.npz'), source_run=str(run))
    write_json(out/'checkpoint-export.json', note)
    name = str(Path(__file__).resolve().relative_to(root))
    shutil.copy2(__file__, out/'source-snapshot'/name)
    hashes = dict(producer['source_sha256']); hashes[name] = sha(Path(__file__))
    for dependency, content in restored.items():
        hashes[dependency] = hashlib.sha256(content).hexdigest()
    write_json(out/'restored-dependencies.json', dict(git_base=producer['git_base'], source_sha256={n:hashes[n] for n in restored}))
    write_json(out/'manifest.json', dict(source_snapshot_root='source-snapshot', source_sha256=hashes, artifacts_sha256=artifact_manifest(out)))
    print(json.dumps(note))


if __name__ == '__main__':
    main()
