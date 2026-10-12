#!/usr/bin/env python3
"""Archive mesh/solver evidence from this project's ASCII OpenFOAM pilot.

The generated metadata deliberately makes no physical benchmark pass claim.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.openfoam_history import function_history,convergence_audit

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();case=a.case;out=a.output
out.mkdir(parents=True,exist_ok=True);evidence=out/'evidence';evidence.mkdir(exist_ok=True)

def scalar_patch(path, owners=None, count=None):
    text=path.read_text();wall=re.search(r'\bwall\s*\{(.*?)\}',text,re.S).group(1)
    match=re.search(r'value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)',wall,re.S)
    if match:
        values=np.fromstring(match.group(2),sep=' ')
        assert len(values)==int(match.group(1))
        return values
    uniform=re.search(r'value\s+uniform\s+([\d.eE+-]+)\s*;',wall)
    if uniform and (count is not None or owners is not None):
        return np.full(count if count is not None else len(owners),float(uniform[1]))
    if owners is not None and re.search(r'type\s+zeroGradient\s*;',wall):
        match=re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)',text,re.S)
        if match:
            values=np.fromstring(match.group(2),sep=' ')
            assert len(values)==int(match.group(1))
            return values[owners]
    raise ValueError(f'expected an explicit wall scalar array: {path}')

folder=case/'constant/polyMesh'
points=np.array([np.fromstring(s,sep=' ') for s in re.findall(r'^\(([^()]+)\)$',(folder/'points').read_text(),re.M)])
patch=re.search(r'\bwall\s*\{(.*?)\}',(folder/'boundary').read_text(),re.S).group(1)
start=int(re.search(r'startFace\s+(\d+)',patch).group(1));count=int(re.search(r'nFaces\s+(\d+)',patch).group(1))
face_rows=re.findall(r'^\d+\(([^()]+)\)$',(folder/'faces').read_text(),re.M)[start:start+count]
faces=[points[np.fromstring(row,sep=' ',dtype=int)] for row in face_rows]
centers=np.array([face.mean(0) for face in faces])
area_vectors=np.array([np.cross(face,np.roll(face,-1,axis=0)).sum(0)/2 for face in faces])
areas=np.linalg.norm(area_vectors,axis=1)
layers=scalar_patch(case/'0/nSurfaceLayers',count=count)
assert len(layers)==count
visible=centers[:,1]>0
fig,axes=plt.subplots(2,1,figsize=(11,5),layout='constrained')
for ax in axes:
    coll=PolyCollection([f[:,[0,2]] for f,keep in zip(faces,visible) if keep],array=layers[visible],
                        cmap='viridis',edgecolors=(0,0,0,.25),linewidths=.12,clim=(0,6))
    ax.add_collection(coll);ax.set_aspect('equal');ax.set_xlabel('x/L');ax.set_ylabel('z/L')
axes[0].set(xlim=(-.015,1.015),ylim=(-.07,.07),title='Actual hull surface mesh: colour = number of prism layers')
axes[1].set(xlim=(.72,1.005),ylim=(-.065,.065),title='Stern detail (surface facets, not a volume slice)')
fig.colorbar(coll,ax=axes,label='Actual layers');fig.savefig(out/'mesh-layers.png',dpi=180);plt.close(fig)
meshlog=(case/'log.checkMesh').read_text();extended=(case/'log.checkMesh.extended').read_text()
metrics=dict(cells=int(re.search(r'^\s*cells:\s*(\d+)',meshlog,re.M).group(1)),wall_faces=count,
    mean_layers_face_weighted=float(layers.mean()),
    area_fraction_with_layers=float(areas[layers>0].sum()/areas.sum()),
    area_fraction_six_layers=float(areas[layers==6].sum()/areas.sum()),
    face_layer_histogram={str(int(v)):int(n) for v,n in zip(*np.unique(layers,return_counts=True))},
    standard_mesh_ok='Mesh OK.' in meshlog,extended_mesh_ok='Mesh OK.' in extended,
    max_nonorthogonality_deg=float(re.search(r'non-orthogonality Max:\s*(\S+)',meshlog).group(1)),
    max_skewness=float(re.search(r'Max skewness =\s*(\S+)',meshlog).group(1)),
    solver_completed=False,numerically_converged=False,physical_benchmark_passed=False)
metrics['wall_surface_area_L2']=float(areas.sum())
for name,mask in [('missing_layers',layers==0),('incomplete_layers',layers<6)]:
    metrics[name]=dict(faces=int(mask.sum()),area_fraction=float(areas[mask].sum()/areas.sum()),
        face_centres_min_L=centers[mask].min(0).tolist() if mask.any() else None,
        face_centres_max_L=centers[mask].max(0).tolist() if mask.any() else None)
force_paths=list((case/'postProcessing/forceCoeffs').glob('*/forceCoeffs.dat'))
if force_paths:
    data,force_paths=function_history(case,'forceCoeffs','forceCoeffs.dat');cd=data[-1,2];reference=.09312081297064881
    np.savetxt(evidence/'forceCoeffs.dat.gz',data,header='Time Cm Cd Cl Cl_front Cl_rear')
    audit=convergence_audit(case)
    (out/'convergence-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    metrics['numerically_converged']=audit['numerically_converged']
    metrics.update(last_iteration=float(data[-1,0]),last_cd=float(cd),reference_cd=reference,
        provisional_drag_relative_error_percent=float(abs(cd/reference-1)*100))
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');ax.plot(data[:,0],data[:,2],label='OpenFOAM pilot')
    ax.axhline(reference,color='k',ls='--',label='AFF-1 reference under stated material assumptions')
    ax.set(xlabel='SIMPLE iteration',ylabel='Cd (frontal area)',title='Drag history: '+('numerical gates passed; accuracy requires validation' if metrics['numerically_converged'] else 'not numerically converged'))
    ax.legend(fontsize=8);fig.savefig(out/'drag-history.png',dpi=160);plt.close(fig)
    log=(case/'log.foamRun').read_text();metrics['solver_completed']=bool(re.search(r'^End\s*$',log,re.M))
    iterations=re.findall(r'^Time = (\d+)s',log,re.M)
    if iterations:
        last=log.split('Time = '+iterations[-1]+'s')[-1]
        metrics['last_initial_residuals']={field:float(value) for field,value in re.findall(r'Solving for (\w+), Initial residual = ([\d.eE+-]+)',last)}
        # Pressure is solved several times: retain its first, not its last,
        # correction residual as the nonlinear outer-loop indicator.
        prs=re.findall(r'Solving for p, Initial residual = ([\d.eE+-]+)',last)
        if prs:metrics['last_initial_residuals']['p']=float(prs[0])
    time_dir=case/str(int(data[-1,0]))
    if (time_dir/'p').exists():
        owner_text=(folder/'owner').read_text()
        owner_match=re.search(r'\n(\d+)\s*\n\((.*?)\)',owner_text,re.S)
        owners=np.fromstring(owner_match.group(2),sep=' ',dtype=int)
        assert len(owners)==int(owner_match.group(1))
        cp=2*scalar_patch(time_dir/'p',owners[start:start+count])
        force_files=list((case/'postProcessing/forces').glob('*/forces.dat'))
        if force_files:
            force_data,force_files=function_history(case,'forces','forces.dat')
            np.savetxt(evidence/'forces.dat.gz',force_data,header='Time pressure_xyz viscous_xyz pressure_moment_xyz viscous_moment_xyz')
            force_values=force_data[-1]
            integrated_pressure=(cp[:,None]*area_vectors/2).sum(0)
            mismatch=float(np.max(np.abs(integrated_pressure-force_values[1:4])))
            metrics['pressure_force_reintegration_max_abs_error']=mismatch
            assert mismatch<1e-9, 'wall Cp reconstruction disagrees with OpenFOAM pressure forces'
            denominator=.5*np.pi*(.508/4.356)**2/4
            metrics['cd_pressure']=float(force_values[1]/denominator)
            metrics['cd_viscous']=float(force_values[4]/denominator)
            pressure_regions={
                'forebody_x_lt_0.25':centers[:,0]<.25,
                'midbody_0.25_to_0.75':(centers[:,0]>=.25)&(centers[:,0]<.75),
                'afterbody_x_ge_0.75':centers[:,0]>=.75,
                'zero_layer_faces':layers==0}
            metrics['pressure_drag_regions']={name:dict(
                cd_pressure=float(np.sum(cp[mask]*area_vectors[mask,0]/2)/denominator),
                area_fraction=float(areas[mask].sum()/areas.sum()))
                for name,mask in pressure_regions.items()}
        np.savetxt(evidence/'wall-cp-pilot.csv.gz',np.c_[centers,cp],delimiter=',',header='x_L,y_L,z_L,Cp',comments='')
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');ax.scatter(centers[:,0],cp,s=.5,alpha=.2)
        ax.set(xlabel='x/L',ylabel='Cp = 2p (U=1)',title='All wall faces: Re=13.218M (not the Re=12M tap comparison)')
        fig.savefig(out/'cp-pilot.png',dpi=160);plt.close(fig)
    yp_files=sorted((case/'postProcessing/yPlus').glob('*/yPlus.dat'),key=lambda p:float(p.parent.name))
    if yp_files:
        yp=yp_files[-1]
        with gzip.open(evidence/'yPlus.dat.gz','wb') as stream:stream.write(yp.read_bytes())
        rows=np.array([line.split() for line in yp.read_text().splitlines() if line.strip() and not line.startswith('#')])
        metrics['last_yplus']={key:float(val) for key,val in zip(['minimum','maximum','mean'],rows[-1,2:5])}
    if (time_dir/'yPlus').exists():
        yp_wall=scalar_patch(time_dir/'yPlus',count=count)
        metrics['yplus_outliers']=dict(
            maximum_face_center_L=centers[np.argmax(yp_wall)].tolist(),
            area_fraction_gt_100=float(areas[yp_wall>100].sum()/areas.sum()),
            area_fraction_gt_200=float(areas[yp_wall>200].sum()/areas.sum()),
            max_on_six_layer_faces=float(yp_wall[layers==6].max()) if np.any(layers==6) else None,
            max_on_zero_layer_faces=float(yp_wall[layers==0].max()) if np.any(layers==0) else None,
            area_fraction_gt_100_on_incomplete_layers=float(areas[(yp_wall>100)&(layers<6)].sum()/areas.sum()))
        regions={'body_x_lt_0.85':centers[:,0]<.85,'tail_x_ge_0.85':centers[:,0]>=.85,
                 'all':np.ones(count,dtype=bool)}
        metrics['yplus_regions']={}
        for name,mask in regions.items():
            vals=yp_wall[mask];weights=areas[mask]
            metrics['yplus_regions'][name]=dict(
                face_quantiles_0_10_50_90_100=np.quantile(vals,[0,.1,.5,.9,1]).tolist(),
                area_weighted_mean=float(np.average(vals,weights=weights)),
                area_fraction_40_to_50=float(weights[(vals>=40)&(vals<=50)].sum()/weights.sum()),
                area_fraction_30_to_100=float(weights[(vals>=30)&(vals<=100)].sum()/weights.sum()),
                area_fraction_with_layers=float(weights[layers[mask]>0].sum()/weights.sum()))
        np.savetxt(evidence/'wall-yplus.csv.gz',np.c_[centers,yp_wall,layers,areas],delimiter=',',
                   header='x_L,y_L,z_L,yplus,layers,area_L2',comments='')
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        ax.scatter(centers[:,0],yp_wall,s=.5,alpha=.25)
        ax.axhspan(40,50,color='green',alpha=.2,label='Nominal target 40-50')
        ax.set(xlabel='x/L',ylabel='y+',yscale='log',title='Computed wall y+: '+('numerically converged' if metrics['numerically_converged'] else 'not numerically converged'))
        ax.legend();fig.savefig(out/'yplus-pilot.png',dpi=160);plt.close(fig)
for name in ['log.blockMesh','log.backgroundRefinement','log.snappyHexMesh','log.checkMesh','log.checkMesh.extended']+[p.name for p in case.glob('log.foamRun*')]:
    path=case/name
    if path.exists():
        with gzip.open(evidence/(name+'.gz'),'wb') as target:target.write(path.read_bytes())
initial_manifest={}
for directory in ['system','constant','0']:
    target=evidence/'case-input'/directory;target.mkdir(parents=True,exist_ok=True)
    for path in (case/directory).iterdir():
        if path.is_file() and (directory!='0' or path.name in ['U','p','k','omega','nut']):
            if directory=='0':
                initial_manifest[path.name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,archived=path.stat().st_size<262144)
                if not initial_manifest[path.name]['archived']:continue
            shutil.copy2(path,target/path.name)
(evidence/'initial-field-manifest.json').write_text(json.dumps(initial_manifest,indent=2)+'\n')
shutil.copy2(case/'provenance.json',evidence/'provenance.json')
if (case/'restart-provenance.json').exists():shutil.copy2(case/'restart-provenance.json',evidence/'restart-provenance.json')
for name in ['initialization-map.json','continuation-stages.json']:
    if (case/name).exists():shutil.copy2(case/name,evidence/name)
for audit_path in case.glob('convergence-audit-*.json'):
    shutil.copy2(audit_path,evidence/audit_path.name)
discarded=case/'noncheckpointed-attempt'
if discarded.exists():
    target=evidence/'noncheckpointed-attempt';target.mkdir(exist_ok=True)
    shutil.copy2(discarded/'reason.json',target/'reason.json')
    with gzip.open(target/'log.foamRun.gz','wb') as stream:
        stream.write((discarded/'log.foamRun').read_bytes())
metrics['mesh_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['points','faces','owner','neighbour','boundary']}
(out/'status.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
