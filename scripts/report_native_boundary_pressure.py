#!/usr/bin/env python3
"""Audit sampled-function pressure trace evidence; no flow qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);root=p.parse_args().directory
    assert json.loads((root/'summary.json').read_text())['source_integrity_verified']
    for file,digest in json.loads((root/'sources.json').read_text()).items():
        assert hashlib.sha256((root/'source-snapshot'/file).read_bytes()).hexdigest()==digest
    rows=json.loads((root/'cases.json').read_text())
    for row in rows:
        file=root/f"r{row['refinement']}-fields.npz";assert hashlib.sha256(file.read_bytes()).hexdigest()==row['fields_sha256']
        with np.load(file,allow_pickle=False) as f:
            ids=f['face_indices'];fc=f['face_centers'][ids];cc=f['cell_centers']
            exact=1+.2*np.sin(2*np.pi*fc[:,0])+.3*fc[:,1]**2-.1*fc[:,2]
            cell=1+.2*np.sin(2*np.pi*cc[:,0])+.3*cc[:,1]**2-.1*cc[:,2]
            np.testing.assert_allclose(exact,f['reference_pressure'],atol=1e-14,rtol=0)
            np.testing.assert_allclose(cell,f['cell_pressure'],atol=1e-14,rtol=0)
            order=np.searchsorted(f['owner_cells'],f['owner'][ids])
            replay=cell[f['owner'][ids]]+np.sum((fc-cc[f['owner'][ids]])*f['gradient'][order],axis=1)
            np.testing.assert_allclose(replay,f['pressure'],atol=1e-14,rtol=0)
            area=np.linalg.norm(f['area_vectors'][ids],axis=1)
            error=np.sqrt(np.sum(area*(f['pressure']-exact)**2)/np.sum(area*exact**2))
            np.testing.assert_allclose(error,row['trace_relative_l2'],rtol=1e-12)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for name,label in [('trace_relative_l2','Connected-cell linear trace'),('owner_relative_l2','Adjacent cell value')]:
        ax.loglog([r['refinement'] for r in rows],[r[name]*100 for r in rows],'o-',label=label)
    ax.set(xlabel='Directional refinement factor',ylabel='Area-weighted relative L2 error (%)',title='Sampled smooth pressure function; NOT a CFD flow result');ax.grid(True,which='both');ax.legend()
    fig.savefig(root/'trace-refinement.png',dpi=180);fig.savefig(root/'trace-refinement.pdf');plt.close(fig)
    result=dict(source_and_fields_verified=True,sampled_function_only=True,wall_Cp_certified=False,physical_suboff_benchmark_passed=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
