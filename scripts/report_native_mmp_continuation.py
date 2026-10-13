#!/usr/bin/env python3
"""Audit and plot thin-layer scalar controls; these are not SUBOFF flow fields."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_sections3d import cell_section_polygons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    summary=json.loads((root/'summary.json').read_text())
    assert summary['source_integrity_verified']
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        assert hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==digest
    rows=json.loads((root/'cases.json').read_text());refined=[]
    for row in rows:
        path=root/(row['case']+'-fields.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            internal=f['neighbor']>=0
            balance=(f['storage']+f['destruction']-f['production']).copy()
            np.add.at(balance,f['owner'],f['total_flux'])
            np.add.at(balance,f['neighbor'][internal],-f['total_flux'][internal])
            np.testing.assert_allclose(balance,f['cell_balance'],rtol=0,atol=1e-12)
            work=f['work_balance_cell_terms'].sum()+f['work_balance_interface_terms'].sum()
            np.testing.assert_allclose(work,f['work_balance'],rtol=1e-12,atol=1e-16)
            if row['correction']=='nonlinear_mmp':
                assert f['correction_energy']>=0
                history=json.loads((root/(row['case']+'-history.json')).read_text())
                if row['converged']:
                    assert float(f['final_smoothing'])==0
                    assert float(f['current_nonlinear_residual'])<1e-12
                    assert history[-1]['change']<1e-12
                    polish=[v for v in history if v.get('fixed_point_polish')]
                    if polish:
                        assert len(polish)>=2
                        assert all(v['unrelaxed'] and v['change']<1e-12 and v['true_linear_relative_residual']<1e-9 for v in polish[-2:])
                    assert max(row['cell_balance_max'],row['interface_flux_defect_max'],abs(row['global_balance']),abs(work))<1e-8
                else:assert not bool(f['converged'])
            if row['case'].startswith('suboff_layer'):
                error=float(np.max(np.abs(f['cell_values']-1-f['cell_centers']@np.array([.1,-.2,.3]))))
                np.testing.assert_allclose(error,row['affine_max_absolute_error'],rtol=1e-6,atol=1e-15)
                if row['correction']=='nonlinear_mmp':refined.append(row)
    assert summary['all_requested_controls_converged']==all(r['converged'] for r in rows)
    refined.sort(key=lambda r:r['cells'])
    decreasing=len(refined)>=3 and all(a['affine_max_absolute_error']>b['affine_max_absolute_error'] for a,b in zip(refined,refined[1:]))
    if refined:
        finest=refined[-1]
        with np.load(root/(finest['case']+'-fields.npz'),allow_pickle=False) as f:
            m=SimpleNamespace(vertices=f['vertices'],cells=list(range(len(f['cell_volumes']))),
                face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'])
            err=np.asarray(f['cell_values']-1-f['cell_centers']@np.array([.1,-.2,.3]),float)
            fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
            for ax,axis,value,label in zip(axes[:2],[2,0],[0.,.47],['Actual axial section z/L=0','Affine scalar error; section x/L=0.47']):
                polygons,ids=cell_section_polygons(m,axis=axis,value=value)
                pc=PolyCollection(polygons,edgecolors='.35',linewidths=.25,
                    **(dict(facecolors='none') if axis==2 else dict(array=err[ids],cmap='coolwarm')))
                ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(label)
                ax.set_xlabel('x/L' if axis==2 else 'y/L');ax.set_ylabel('y/L' if axis==2 else 'z/L')
                if axis==0:fig.colorbar(pc,ax=ax,label='Scalar error (not pressure)')
            axes[2].loglog([r['cells'] for r in refined],[r['affine_max_absolute_error'] for r in refined],'o-')
            for r in refined:
                axes[2].annotate('converged' if r['converged'] else 'REJECTED',(r['cells'],r['affine_max_absolute_error']),xytext=(4,5),textcoords='offset points',fontsize=8)
            axes[2].set(xlabel='Native cells',ylabel='Maximum absolute affine scalar error');axes[2].grid(True)
            axes[2].margins(x=.15,y=.25)
            fig.suptitle('SUBOFF thin-layer scalar control — no exterior flow or Cp/Cd')
            fig.savefig(root/'thin-layer-controls.png',dpi=180);fig.savefig(root/'thin-layer-controls.pdf');plt.close(fig)
    output=dict(source_and_field_integrity_verified=True,saved_cases=len(rows),
        all_requested_controls_converged=summary['all_requested_controls_converged'],
        three_grid_affine_error_decreasing=decreasing,physical_suboff_benchmark_passed=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
