"""Plot frozen independent SST-reference results without rerunning the solver."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_reference(directory):
    root=Path(directory);rows=json.loads((root/'cases.json').read_text())
    native=json.loads((root/'native-comparison.json').read_text())
    cfg=json.loads((root/'settings.json').read_text());re_tau=cfg['Re_tau']
    reference=json.loads((root/'dns-reference/provenance.json').read_text())
    ref=reference['files']['mean'];path=root/'dns-reference'/ref['csv']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['csv_sha256']:raise ValueError('changed DNS data')
    dns=np.loadtxt(path,delimiter=',',skiprows=1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].plot(dns[:,1],dns[:,2],'k-',label='Lee–Moser DNS')
    selected={min(r['n'] for r in rows),128,max(r['n'] for r in rows)}
    for row in rows:
        path=root/f"reference-n{row['n']}-fields.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['fields_sha256']:raise ValueError('changed reference fields')
        if row['n'] in selected:
            with np.load(path,allow_pickle=False) as f:axes[0].plot(f['y']*re_tau,f['velocity'],label=f"Independent SST n={row['n']}")
    axes[0].set(xscale='log',xlim=(.1,re_tau*1.05),xlabel='y+',ylabel='U+',title='Independent model; displayed y+ >= 0.1')
    axes[0].legend(fontsize=8);axes[0].grid(True)
    axes[1].plot([r['n'] for r in rows],[r['Cf'] for r in rows],'o-',label='Independent 1D SST')
    native.sort(key=lambda r:r['n']);axes[1].plot([r['n'] for r in native],[r['Cf'] for r in native],'x--',label='Native hybrid 3D')
    axes[1].axhline(reference['Cf'],color='k',label='DNS');axes[1].axhspan(reference['Cf']*.97,reference['Cf']*1.03,color='g',alpha=.1,label='±3% nominal band')
    axes[1].set(xscale='log',xlabel='Wall-normal cells',ylabel='Cf',title='Independent model limit vs native grids');axes[1].legend(fontsize=8);axes[1].grid(True)
    fig.savefig(root/'model-discretization-comparison.png',dpi=180);fig.savefig(root/'model-discretization-comparison.pdf');plt.close(fig)
    path=root/'postprocessing-source/sst_channel_report.py';path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,path)
    (root/'postprocessing-sources.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        display_minimum_y_plus=.1,numerical_fields_changed=False),indent=2)+'\n')
