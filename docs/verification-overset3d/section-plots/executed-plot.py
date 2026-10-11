#!/usr/bin/env python3
"""Draw archived real z=0 section polygons without rebuilding overset geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap,BoundaryNorm


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    output=args.directory/'section-plots'
    if output.exists():raise ValueError('plot output already exists')
    output.mkdir();colors=ListedColormap(['#c4e6c3','#d5d5d5','#edaf66']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    inputs={}
    for level in (1,2,3):
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
        for ax,name,title in zip(axes,('cartesian','radially'),('Cartesian background','Radial wedge shell')):
            path=args.directory/f'level-{level}'/(name+'-section.npz');raw=np.load(path);inputs[str(path.relative_to(args.directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
            points=raw['vertices'];offsets=raw['offsets'];polygons=[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
            coll=PolyCollection(polygons,array=raw['status'].astype(float),cmap=colors,norm=norm,edgecolors='#444444',linewidths=.22)
            ax.add_collection(coll);ax.autoscale_view();ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y');ax.set_title(title)
        fig.colorbar(coll,ax=axes,ticks=[0,1,2],label='0 active; 1 hole; 2 fringe',shrink=.75)
        fig.suptitle(f'Level {level}: actual z=0 cell sections; static geometry, no flow solve')
        fig.savefig(output/f'level-{level}-mesh.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (output/'input-sha256.json').write_text(json.dumps(inputs,indent=2)+'\n');shutil.copy2(__file__,output/'executed-plot.py')
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
