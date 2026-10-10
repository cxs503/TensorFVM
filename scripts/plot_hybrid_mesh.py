"""Reproduce the hybrid wall/trailing-edge detail from saved physical cells."""
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib.collections import PolyCollection
from tensorfvm.verification.report import plt
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    summary=json.loads((args.output/'summary.json').read_text());row=summary['runs'][-1]
    with np.load(args.output/row['directory']/'fields.npz') as fields:polys=fields['cell_polygons_m'].copy()
    quad=~np.all(polys[:,2]==polys[:,3],axis=1);c=row['config'];x,y=c['airfoil_x'],c['airfoil_y'];angle=-np.radians(row['mesh_generation']['geometric_incidence_deg']);te=np.array([x+.25+.75*np.cos(angle),y+.75*np.sin(angle)])
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for axis,limits in zip(axes,[(x-.1,x+1.15,y-.22,y+.17),(te[0]-.07,te[0]+.1,te[1]-.04,te[1]+.04)]):
        axis.add_collection(PolyCollection(polys,facecolors=np.where(quad,'#d6ecff','#fff4d9'),edgecolors='#405060',linewidths=.23,rasterized=True));axis.set(xlim=limits[:2],ylim=limits[2:],xlabel='x / c',ylabel='y / c');axis.set_aspect('equal')
    axes[0].set_title('Quad wall layers / triangular outer cells');axes[1].set_title('Trailing-edge fan and wake refinement');fig.tight_layout();fig.savefig(args.output/'hybrid-mesh-detail.png',dpi=220);plt.close(fig)
    manifest=json.loads((args.output/'manifest.json').read_text());manifest['source_sha256']['scripts/plot_hybrid_mesh.py']=sha(Path(__file__));manifest['artifacts_sha256']=artifact_manifest(args.output);write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
