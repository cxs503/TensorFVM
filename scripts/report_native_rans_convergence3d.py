#!/usr/bin/env python3
"""Report actual nonzero 3D NS/SST convergence and pseudo-step consistency."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('docs/verification-native-rans3d'));a=p.parse_args();root=a.directory
    names=['converged-dt-01','converged','converged-dt-10'];cases={name:json.loads((root/name/'result.json').read_text()) for name in names}
    fields={name:np.load(root/name/'fields.npz') for name in names};ref=fields['converged'];volume=ref['cell_volumes']
    keys=['momentum_current_residual','current_reconstructed_divergence_max','stored_native_divergence_max','face_flux_consistency_max','velocity_change','modified_pressure_change','flux_change','k_change','omega_change','k_current_nonlinear_residual','omega_current_nonlinear_residual','modified_pressure_gauge_residual']
    records=[]
    for name,result in cases.items():
        last=result['history'][-1];actual=fields[name]
        errors={}
        for key in ['velocity','pressure','k','omega','flux']:
            v=actual[key];r=ref[key];w=np.ones(len(v)) if key=='flux' else volume
            w=w.reshape((len(w),)+(1,)*(v.ndim-1));errors[key]=float(np.sqrt(np.sum(w*(v-r)**2)/np.sum(w*r*r)))
        gates={key:dict(actual=last[key],limit=1e-7,passed=last[key]<1e-7) for key in keys}
        records.append(dict(case=name,pseudo_dt=result['pseudo_time_step'],iterations=len(result['history']),numerically_converged=result['numerically_converged'],gates=gates,field_relative_difference_to_dt1=errors,physical_benchmark_passed=False))
    passed=all(r['numerically_converged'] and all(g['passed'] for g in r['gates'].values()) for r in records)
    comparison=dict(kind='Numerical NS/SST coupling convergence; not physical validation',all_unchanged_gates_passed=passed,gate=1e-7,inner_momentum_gate=1e-8,field_consistency_limit=1e-6,field_consistency_passed=max(e for r in records for e in r['field_relative_difference_to_dt1'].values())<1e-6,physical_benchmark_passed=False,cases=records)
    (root/'convergence.json').write_text(json.dumps(comparison,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for name,case in cases.items():
        history=case['history'];x=[r['iteration'] for r in history]
        for ax,key in zip(axes,['k_current_nonlinear_residual','omega_current_nonlinear_residual','current_reconstructed_divergence_max']):
            ax.semilogy(x,[max(r[key],1e-16) for r in history],label=f"dt={case['pseudo_time_step']}");ax.axhline(1e-7,color='black',ls='--',lw=.7);ax.set(xlabel='Outer iteration',ylabel=key.replace('_',' '));ax.grid(alpha=.25)
    axes[0].legend();fig.tight_layout();fig.savefig(root/'convergence.png',dpi=180);fig.savefig(root/'convergence.svg');plt.close(fig)
    # Actual central-plane intersections of the native box hexes. Fields are
    # piecewise cell values, not a manufactured reference or raster smoothing.
    vertices=ref['vertices'];cells=ref['cells'];ids=[];polygons=[]
    for ci,cell in enumerate(cells):
        pts=vertices[cell]
        if pts[:,2].min()<=.5<pts[:,2].max():
            xy=np.unique(pts[:,:2],axis=0);angle=np.arctan2(xy[:,1]-xy[:,1].mean(),xy[:,0]-xy[:,0].mean());polygons.append(xy[np.argsort(angle)]);ids.append(ci)
    ids=np.array(ids);fig,axes=plt.subplots(2,3,figsize=(12,7))
    values=[None,ref['pressure'][ids],np.linalg.norm(ref['velocity'][ids],axis=1),ref['k'][ids],ref['omega'][ids],ref['wall_distance'][ids]]
    titles=['Actual z=0.5 native cell cuts','Physical pressure / rho U²','Cell speed / U','k / U²','omega L / U','Nearest wall distance / L']
    for ax,value,title in zip(axes.ravel(),values,titles):
        pc=PolyCollection(polygons,facecolors='none',edgecolors='black',linewidths=.7) if value is None else PolyCollection(polygons,array=value,cmap='coolwarm' if 'pressure' in title else 'viridis',edgecolors='black',linewidths=.3)
        ax.add_collection(pc);ax.set(xlim=(0,1),ylim=(0,1),xlabel='x/L',ylabel='y/L',title=title);ax.set_aspect('equal')
        if value is not None:fig.colorbar(pc,ax=ax)
    fig.tight_layout();fig.savefig(root/'converged-fields.png',dpi=180);fig.savefig(root/'converged-fields.svg');plt.close(fig)
    rows='\n'.join(f"| {r['pseudo_dt']} | {r['iterations']} | {r['gates']['k_current_nonlinear_residual']['actual']:.6g} | {r['gates']['omega_current_nonlinear_residual']['actual']:.6g} | {r['gates']['current_reconstructed_divergence_max']['actual']:.6g} | {r['numerically_converged']} |" for r in records)
    errors='\n'.join(f"| {r['pseudo_dt']} | "+' | '.join(f"{r['field_relative_difference_to_dt1'][key]:.6g}" for key in ['velocity','pressure','k','omega','flux'])+' |' for r in records)
    text=f'''# Nonzero native 3D NS/SST numerical convergence

**Numerical coupling convergence is verified; physical benchmark accuracy remains unqualified.** The real 27-cell unit channel has uniform inlet U=1, two prescribed no-slip walls, physical-pressure outlet p=0 and prescribed uniform velocity on z boundaries. Re=20, rho=1 and molecular viscosity=.05. SST-2003 exact production, original constants, and low-Re wall omega are unchanged. This is a coupling diagnostic, not a turbulent reference experiment.

## Original limitation and correction

The [original two-outer-step run](nonzero-channel/report.md) remains preserved. Its pseudo dt=1e-4 did not advance turbulence close to steady balance; omega's current nonlinear residual stayed near 1. The new runs increase pseudo dt and outer iterations. Initial k=.001 and omega=1 remain identical; no reference solution supplies initial fields or sources. No equation, wall condition, positivity clipping or acceptance threshold was changed.

Outer gates remain **1e-7**, inner momentum gate **1e-8**. Each outer acceptance checks current momentum after updating viscosity, stored and reconstructed continuity, current Rhie–Chow flux consistency, velocity/modified-pressure/flux/k/omega changes and recomputed original nonlinear k/omega equations. Gauge is checked when applicable. The frozen implicit transport ledger is reported separately and does not qualify nonlinear convergence.

| Pseudo dt | Actual outer iterations | Current k residual | Current omega residual | Reconstructed divergence U/L | Numerically converged |
| ---: | ---: | ---: | ---: | ---: | --- |
{rows}

![Actual nonlinear residual histories](convergence.png)

## Actual converged mesh and fields

![Native central-plane cells and actual dt=1 fields](converged-fields.png)

Fields are actual computed cell values; the native cell intersections remain visible. Physical pressure is converted consistently from pi=p+2rho k/3. Isotropic and anisotropic stress terms reconstruct the same total native face force. Raw arrays and all imported producer source snapshots are archived separately in each case directory.

## Independent pseudo-step comparison

Relative field differences below compare converged dt=.1 and dt=10 solutions against dt=1. Cell fields use volume weighting; native face flux uses its Euclidean norm. Numerical consistency limit is 1e-6, separate from the physical benchmark's requested 3% criterion.

| Pseudo dt | Velocity | Physical pressure | k | omega | Native flux |
| ---: | ---: | ---: | ---: | ---: | ---: |
{errors}

## Qualification scope and reproduction

All unchanged nonlinear gates passed: **{passed}**. Pseudo-step field consistency passed: **{comparison['field_consistency_passed']}**. Physical benchmark passed: **False**. No experimental drag/Cp reference, below-3% accuracy, spatial refinement, high-Re wall y+, SUBOFF result or GPU validation is claimed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_native_rans_integration3d.py --outer 100 --pseudo-dt 1 --turbulence-relaxation .7 --tolerance 1e-7 --output NEW_EMPTY_DIRECTORY
PYTHONPATH=src python scripts/report_native_rans_convergence3d.py
```

[convergence.json](convergence.json) lists every gate, threshold and actual value. Detailed histories, original current-source ledgers and measured elapsed times appear in the three case result.json files. Source bindings are captured at launch. Physical validation still requires independent benchmark data and wall/mesh convergence.
'''
    (root/'report.md').write_text(text)
    snapshot=root/'report-source-snapshot';snapshot.mkdir(exist_ok=True)
    source=Path(__file__).resolve();(snapshot/source.name).write_bytes(source.read_bytes())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(report_source_sha256=digest(source),input_sha256={str(p.relative_to(root)):digest(p) for name in names for p in [root/name/'result.json',root/name/'fields.npz']},artifact_sha256={name:digest(root/name) for name in ['report.md','convergence.json','convergence.png','convergence.svg','converged-fields.png','converged-fields.svg']})
    (root/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(comparison,indent=2));return 0 if passed and comparison['field_consistency_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
