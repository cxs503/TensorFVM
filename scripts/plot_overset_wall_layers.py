#!/usr/bin/env python3
"""Plot archived actual section polygons together and zoom into wall layers."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
out=args.directory/'wall-layer-plots'
if out.exists():raise ValueError('new plot destination required')
out.mkdir();inputs={};artifacts=[]
for folder in sorted(args.directory.glob('level-*')):
    if not (folder/'radially-section.npz').exists():continue
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,color in [('cartesian','#666666'),('radially','#157ca0')]:
        path=folder/(name+'-section.npz');raw=np.load(path)
        inputs[str(path.relative_to(args.directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
        polygons=[raw['vertices'][a:b] for a,b in zip(raw['offsets'][:-1],raw['offsets'][1:])]
        for ax in axes:
            rgba=np.zeros((len(polygons),4));rgba[raw['status']==2]=[1,.65,.1,.22]
            ax.add_collection(PolyCollection(polygons,facecolors=rgba,edgecolors=color,linewidths=.32 if name=='cartesian' else .55))
    axes[0].set_xlim(-3,3);axes[0].set_ylim(-3,3);axes[0].set_title('Both actual grids; z=0 section')
    axes[1].set_xlim(.9,1.65);axes[1].set_ylim(-.4,.4);axes[1].set_title('Actual near-wall layers and overlap')
    for ax in axes:ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y')
    axes[0].legend(handles=[Patch(facecolor='none',edgecolor='#666666',label='Cartesian grid'),Patch(facecolor='none',edgecolor='#157ca0',label='Body-fitted layers'),Patch(facecolor='#ffbf65',label='Fringe')],loc='upper left',fontsize=8)
    fig.suptitle('Mesh geometry only; background HOLE outlines retained')
    target=out/(folder.name+'.png');fig.savefig(target,dpi=180,bbox_inches='tight');plt.close(fig);artifacts.append(target.name)
source=Path(__file__);shutil.copy2(source,out/source.name)
(out/'report.md').write_text('# Actual body-fitted layers and Cartesian background\n\nThe plots overlay archived native section polygons. Background HOLE outlines are retained; orange marks fringe cells. These are mesh plots, not computed flow fields.\n\n'+''.join(f'![{name}]({name})\n\n' for name in artifacts))
manifest=dict(inputs=inputs,outputs={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
