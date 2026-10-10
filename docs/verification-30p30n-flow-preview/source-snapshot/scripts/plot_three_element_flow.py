"""Show source geometry and actual saved cell flow fields, with convergence status."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.benchmark_30p30n import PROFILE_DIRECTORY,_read_profile
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def geometry(output):
    output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
    for name,color in zip(('slat','main','flap'),('tab:blue','tab:orange','tab:green')):
        body=np.array(_read_profile(PROFILE_DIRECTORY/f'{name}.dat'));ax.fill(body[:,0],body[:,1],facecolor=color,edgecolor='black',linewidth=.7,label=name)
    ax.set_xlim(-.03,1.04);ax.set_ylim(-.15,.09);ax.set_aspect('equal');ax.set_xlabel('x / C');ax.set_ylabel('y / C');ax.legend();ax.grid(alpha=.2);ax.set_title('30P30N geometry: slat, main element and flap; fluid slots retained')
    fig.savefig(output/'geometry.png',dpi=300);fig.savefig(output/'geometry.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--geometry-only',action='store_true');a=p.parse_args();out=a.output;geometry(out)
    if a.geometry_only:return
    summary=json.loads((out/'benchmark.json').read_text());data=np.genfromtxt(out/'fields.csv',delimiter=',',names=True);nodes=np.genfromtxt(out/'nodes.csv',delimiter=',',names=True);cells=np.genfromtxt(out/'cells.csv',delimiter=',',skip_header=1);vertices=np.c_[nodes['x'],nodes['y']];polys=[vertices[row[1:][np.isfinite(row[1:])].astype(int)] for row in cells]
    converged=summary['converged'];status='NUMERICALLY CONVERGED; reference accuracy unverified' if converged else 'UNCONVERGED DIAGNOSTIC: not a validated flow solution'
    with PdfPages(out/'flow-report.pdf') as pdf:
        image=plt.imread(out/'geometry.png');fig,ax=plt.subplots(figsize=(12,4));ax.imshow(image);ax.axis('off');pdf.savefig(fig);plt.close(fig)
        for field,label in [('speed','Velocity magnitude / U_inf'),('p','Pressure / (rho U_inf^2)')]:
            fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);collection=PolyCollection(polys,array=data[field],cmap='turbo',edgecolors='none',rasterized=True);ax.add_collection(collection);fig.colorbar(collection,ax=ax,label=label)
            for name in ('slat','main','flap'):
                body=np.array(_read_profile(PROFILE_DIRECTORY/f'{name}.dat'));ax.fill(body[:,0],body[:,1],color='white',edgecolor='black',linewidth=.7)
            ax.set_xlim(-.15,1.3);ax.set_ylim(-.3,.2);ax.set_aspect('equal');ax.set_xlabel('x / C');ax.set_ylabel('y / C');ax.set_title(f'30P30N: {label}\n{status}',fontsize=11,color='darkred' if not converged else 'black');fig.savefig(out/f'{field}.png',dpi=300);fig.savefig(out/f'{field}.svg');pdf.savefig(fig);plt.close(fig)
    (out/'visual-report.md').write_text('# 30P30N geometry and actual flow preview\n\n'+status+'\n\nThese pressure and velocity fields are actual saved SIMPLE/SA cell fields after '+str(summary['iterations'])+' completed iterations, not the overset manufactured scalar solution. Re=5,000,000, alpha=0 degrees. This flow solve uses one body-fitted Gmsh mesh; it does not implement overset Navier-Stokes coupling. They must not be used as accepted aerodynamic results.\n\n![Geometry](geometry.png)\n\n![Velocity](speed.png)\n\n![Pressure](p.png)\n\n[Raw fields](fields.csv) | [Residual history](history.json) | [PDF](flow-report.pdf)\n')
    manifest=json.loads((out/'manifest.json').read_text());key='scripts/plot_three_element_flow.py';manifest['source_sha256'][key]=sha(Path(__file__));target=out/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes())
    for name in ('slat','main','flap'):
        key=f'src/tensorfvm/data/30p30n/{name}.dat';source=PROFILE_DIRECTORY/f'{name}.dat';manifest['source_sha256'][key]=sha(source);target=out/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    manifest['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()
