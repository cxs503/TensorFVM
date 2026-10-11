#!/usr/bin/env python3
"""Integrate source-bound SUBOFF preparation and pressure-compatibility evidence."""
import argparse,json,shutil
from pathlib import Path
from tensorfvm.verification.core import write_json,sha


def main():
    repo=Path(__file__).resolve().parents[1];docs=repo/'docs';out=docs/'verification-suboff-development';out.mkdir(exist_ok=True)
    projection={name:json.loads((docs/'verification-suboff-projection'/name/'result.json').read_text())for name in ['coarse','medium']}
    compatibility=json.loads((docs/'verification-transient-pressure-compatibility/summary.json').read_text())
    tests=['tests/test_body_fitted_transient3d.py','tests/test_body_fitted3d.py','tests/test_external_mesh_contract.py','tests/test_multi_element_flow.py','tests/test_three_element.py','tests/test_native_mesh3d.py','tests/test_native_projection3d.py','tests/test_suboff_geometry.py']
    # Supplied by the completed integration test run; bind the exact checked
    # sources and retain the real pytest output, not a manufactured CFD gate.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-log',type=Path,default=out/'pytest-output.txt')
    args=parser.parse_args()
    log=args.test_log.read_text()
    if '62 passed,'not in log:raise RuntimeError('expected completed integration test evidence absent')
    (out/'pytest-output.txt').write_text(log)
    write_json(out/'validation.json',dict(command='env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python -m pytest '+' '.join(tests)+' -q',passed=62,failed=0,source_sha256={p:sha(repo/p)for p in tests},physical_accuracy_qualified=False))
    rows=[]
    for name,r in projection.items():
        c=r['coupling'];rows.append(f"| {name} | {r['geometry']['cells']:,} | {c['native_face_divergence_max']:.4g} | {c['interior_max_normal_velocity_difference']:.5g} | {c['interpolated_cell_velocity_divergence_max']:.5g} | {r['projection_contract_passed']} |")
    report='''# SUBOFF parallel development: first integrated delivery

**SUBOFF viscous-flow accuracy remains unqualified. This delivery prepares actual body-fitted geometry, general static 3D operators, native-face initialization and restricted pressure-boundary corrections. It does not publish a validated SUBOFF Navier--Stokes result.**

## Parallel implementation

| Workstream | Implemented result | Qualification scope |
| --- | --- | --- |
| Geometry and mesh | TensorLBM bare AFF-1 profile port with original source SHA; Gmsh OCC body-fitted tetrahedra, 6,810 / 13,977 cells | Bare geometry and closed positive mesh only; no appendages or prism boundary layer |
| General 3D interface | NativeMesh3D: Gmsh, unique shared faces, owner/neighbor, actual volumes/centers/area vectors and physical patches; NativeOperators3D: conservative integration and affine reconstruction | Linear convex tetra/hex/wedge/pyramid with planar faces; rejects unsupported topology |
| Pressure compatibility | Same nonzero pressure Neumann data in LS, face flux and conservative traction; optional dt-independent initialization in the cylinder kernel | Restricted opt-in correction; complete coupling and spatial pressure convergence still fail |
| Integrated SUBOFF projection | Full nonorthogonal mixed-boundary scalar operator and CPU sparse LU on both actual SUBOFF meshes | Native-face initialization diagnostic; scalar is velocity potential, not physical pressure |

## Actual SUBOFF meshes and computed initialization

[Geometry preparation and dimensions](../verification-suboff-mesh/report.md); [native topology and affine operator audit](../verification-suboff-native-contract/report.md).

![Actual SUBOFF surface and volume-grid evidence](../verification-suboff-mesh/medium/mesh.png)

[Coarse initialization report](../verification-suboff-projection/coarse/report.md) and [medium report](../verification-suboff-projection/medium/report.md) archive the input mesh, raw fields, source snapshots, mixed-boundary data, independent affine manufacture and actual LU costs.

![True tetrahedral section, velocity potential and cell speed](../verification-suboff-projection/medium/initialization.png)

No viscosity, Reynolds number or physical time evolution is used in this SUBOFF initialization. At the wall the potential normal derivative removes native normal penetration; inlet/farfield prescribed normal flux is retained, and outlet potential is zero. Each plot explicitly distinguishes velocity potential from physical pressure. Native-face mass conservation does not make the independently corrected cell velocity incompressible.

| Mesh | Cells | Native divergence U/L | Max cell/face normal difference / U | Interpolated cell divergence U/L | Projection contract |
| --- | ---: | ---: | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

Affine scalar tests on the actual SUBOFF mesh give gradient errors about 1.3e-13, with prescribed analytic Neumann/Dirichlet data. These are independent operator responses, not physical benchmark accuracy. Both SUBOFF numerical-flow and physical-accuracy qualifications remain false.

## Pressure--velocity compatibility correction

[Full six-run report](../verification-transient-pressure-compatibility/report.md) includes old startup, projected startup on three body-fitted cylinder grids, complete time-step halving, unsmoothed pressure/velocity fields and failed common-module gates.

The 24,576-cell cylinder maximum normal cell/face difference decreases from 23.90% to 1.428%; 55,296 cells give 1.018%. However, interpolated-cell divergence remains 3.876/3.626 U/D. Medium-to-fine wall Cp differs 14.76%, exceeding 3%. All six common acceptance results remain false. The complete dt/half-dt wall Cp difference is 0.01568% and initial projected fields are bitwise identical. A conservative initialization improves the defect; it does not prove a matching pressure--velocity operator or long-time physical correctness.

The correction is deliberately opt-in via `TransientSettings(initial_projection=True)`; default existing initial conditions remain unchanged. A checkpoint regression verifies projected initialization evidence and subsequent field/history/statistical continuation bitwise. The velocity potential is distinct from actual transient pressure.

## Integration verification

**62 tests passed**, with one existing PyTorch index_reduce beta warning. They cover native geometry, mixed-boundary affine pressure and projection, shared-face conservation, legacy three-element paths, failed stability gates and both initialization/restart paths. [validation.json](validation.json) binds test sources; [pytest-output.txt](pytest-output.txt) preserves the actual completed output. A passed regression suite does not supersede failed physical or cell/face gates.

## Reproduction

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/prepare_suboff_mesh.py --output results/suboff-mesh
python scripts/verify_native_mesh3d.py results/suboff-mesh/suboff.msh --output results/suboff-mesh-audit
python scripts/initialize_suboff_native.py --mesh results/suboff-mesh/suboff.msh --output results/suboff-initialization
python scripts/verify_transient_pressure_compatibility.py --output results/pressure-compatibility
python scripts/run_body_fitted_cylinder3900.py --output results/cylinder-hodge --nx 64 --ny 32 --nz 12 --stretching 7 --dt 1e-5 --steps 50 --initial-projection
```

Use a fresh output directory. Benchmark/mesh extras supply NumPy, SciPy, Matplotlib and Gmsh. Tested interpreter is `/data/TensorFEM/.venv/bin/python`. CPU reference factorization is measured; GPU acceleration is untested in these paths.

## Next development gate

1. General native 3D conservative viscous momentum and its pressure response; independent manufactured stress, local momentum, boundary-flux and cell/face checks.
2. Low-Re bare-SUBOFF diagnostic with actual pressure, viscous force, histories and demonstrated convergence; distinguish this from experimental-Reynolds comparisons.
3. Boundary-layer and wake refinement, domain sensitivity, verified turbulence/wall treatment and primary reference ingestion before a 3% physical accuracy claim.
4. AFF-8 appendages with conforming junction geometry/mesh, followed by the same verification sequence.

This delivery does not infer drag/Cp from the startup potential or claim that geometry-only tests validate turbulent SUBOFF flow. The unresolved compatibility errors remain visible and determine the next work.
'''
    (out/'report.md').write_text(report)
    numerical_sources=['src/tensorfvm/native_projection3d.py','src/tensorfvm/native_mesh3d.py','src/tensorfvm/native_operators3d.py','src/tensorfvm/suboff_geometry.py','src/tensorfvm/suboff_mesh.py','src/tensorfvm/body_fitted_transient3d.py','scripts/initialize_suboff_native.py','scripts/prepare_suboff_mesh.py','scripts/verify_native_mesh3d.py','scripts/verify_transient_pressure_compatibility.py','scripts/run_body_fitted_cylinder3900.py',str(Path(__file__).resolve().relative_to(repo))]
    source_records={}
    for rel in numerical_sources+tests:
        target=out/'integration-source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/rel,target);source_records[rel]=sha(target)
    inputs={str(p.relative_to(docs)):sha(p)for p in docs.glob('verification-suboff-*/**/manifest.json')if 'verification-suboff-development'not in str(p)}
    for p in [docs/'verification-suboff-native-contract/audit.json',docs/'verification-transient-pressure-compatibility/manifest.json']:
        inputs[str(p.relative_to(docs))]=sha(p)
    write_json(out/'manifest.json',dict(schema='tensorfvm.suboff-development-integration/1',integration_source_sha256=source_records,input_evidence_sha256=inputs,artifact_sha256={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'}))
    print(json.dumps(dict(tests_passed=62,suboff_numerical_flow_qualified=False,physical_accuracy_qualified=False)))

if __name__=='__main__':main()
