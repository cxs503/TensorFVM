#!/usr/bin/env python3
"""Check saved Couette force/work arrays and plot native wall-boundary evidence."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.native_sections3d import cell_section_polygons
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evidence=NativeRunEvidence(a.output,__file__);records=[]
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    for root in a.runs:
        cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
        if not summary['source_integrity_verified'] or sha(root/'fields.npz')!=summary['fields_sha256']:
            raise ValueError('changed/unverified raw fields')
        for name,value in json.loads((root/'sources.json').read_text()).items():
            if sha(root/'source-snapshot'/name)!=value:raise ValueError('changed producer source')
        with np.load(root/'fields.npz') as f:
            for key in ('velocity','face_velocity','pressure','flux','momentum_flux','total_traction','wall_target_traction'):
                if not np.isfinite(f[key]).all():raise ValueError('nonfinite raw field')
            np.testing.assert_allclose(np.sum(f['face_velocity']*f['area_vectors'],axis=1),f['flux'],atol=1e-14,rtol=1e-14)
            ids=f['wall_faces'];area=np.linalg.norm(f['area_vectors'],axis=1)
            normal=f['area_vectors'][ids]/area[ids,None]
            actual=f['total_traction'][ids];actual=actual-normal*np.sum(actual*normal,axis=1)[:,None]
            defect=float(np.max(abs(actual-f['wall_target_traction'])))
            cells=f['velocity'][f['owner'][ids]];faces=f['face_velocity'][ids]
            wall=np.zeros_like(cells);wall[:,0]=cfg['wall_speed'];target=f['wall_target_traction']
            work=dict(boundary_power=float(np.sum(faces*target)),wall_motion_power=float(np.sum(wall*target)),
                sample_dissipation=-float(np.sum((cells-wall)*target)),sampling_transfer_power=float(np.sum((faces-cells)*target)),
                traction_defect_max=defect)
            for key,value in work.items():np.testing.assert_allclose(value,summary['wall'][key],rtol=1e-12,atol=1e-14)
            error=float(np.max(abs(f['velocity']-f['reference_velocity'])))
            np.testing.assert_allclose(error,summary['independent']['velocity_max_error'],rtol=1e-12,atol=1e-15)
            record=dict(run=str(root),bindings={key:sha(root/key) for key in ('fields.npz','settings.json','summary.json','sources.json')},
                settings=cfg,summary=summary,saved_force_and_work_rechecked=True)
            records.append(record)
            y=f['cell_centers'][:,1];order=np.argsort(y);label=root.name
            line,=ax.plot(f['reference_velocity'][order,0],y[order],label=label+' reference')
            ax.plot(f['velocity'][:,0],y,'o',ms=3,color=line.get_color())
            mesh=SimpleNamespace(vertices=f['vertices'],face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'],cells=list(range(len(y))))
            polygons,cell_ids=cell_section_polygons(mesh,axis=2,value=.37)
            fields,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
            for axis in axes:axis.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x',ylabel='y')
            axes[0].add_collection(PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.5));axes[0].set_title('Actual mesh section z=0.37')
            for axis,key,title in zip(axes[1:],('velocity','pressure'),('Streamwise velocity','Pressure (floor on colour range: 1e-6)')):
                values=f[key][cell_ids];values=values[:,0] if values.ndim==2 else values
                collection=PolyCollection(polygons,array=values,edgecolors='none',cmap='viridis' if key=='velocity' else 'coolwarm')
                if key=='pressure':limit=max(float(np.max(abs(values))),1e-6);collection.set_clim(-limit,limit)
                axis.add_collection(collection);axis.set_title(title);fields.colorbar(collection,ax=axis,shrink=.7)
            fields.suptitle(label+' — boundary verification; constant bulk viscosity, no SST')
            fields.savefig(a.output/(label+'-fields.png'),dpi=160);fields.savefig(a.output/(label+'-fields.pdf'));plt.close(fields)
    ax.set(xlabel='Streamwise velocity',ylabel='y',title='Lines: scalar shear-balance reference; markers: native cell solutions')
    ax.legend(fontsize=7);ax.grid(True,alpha=.25);fig.savefig(a.output/'velocity-profiles.png',dpi=160);fig.savefig(a.output/'velocity-profiles.pdf');plt.close(fig)
    evidence.verify_sources();write_json(a.output/'comparison.json',dict(records=records,
        scope='boundary verification at each explicit sampling distance; not SST or mesh independence',physical_suboff_benchmark_passed=False))


if __name__=='__main__':main()
