#!/usr/bin/env python3
"""Independently replay archived point interpolation; no connectivity builder import."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.verification.core import Metric,evaluate


def prescribed(points):
    x,y,z=np.asarray(points).T
    return np.column_stack((1+.3*np.sin(.7*x)*np.cos(.6*y)+.2*np.cos(.8*z),
        .8+.2*np.cos(.5*x+.4*y)+.15*np.sin(.9*z),1.2+.25*np.sin(.6*y+.3*z)+.1*np.cos(.7*x)))


def audit(directory):
    recorded=json.loads((directory/'manifest.json').read_text())
    for name,expected in recorded.items():
        actual=hashlib.sha256((directory/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('producer artifact hash mismatch: '+name)
    summary=json.loads((directory/'summary.json').read_text());rows=[];metrics=[]
    for record in summary['rows']:
        if not record['completed']:raise ValueError('incomplete producer level')
        folder=directory/f"level-{record['level']}";background=np.load(folder/'background-native.npz');body=np.load(folder/'body-native.npz')
        cells=background['cells'].reshape(-1,8);vertices=background['vertices'][cells]
        closest=np.minimum(np.maximum(np.zeros(3),vertices.min(axis=1)),vertices.max(axis=1))
        intersecting=np.linalg.norm(closest,axis=1)<=1*(1+1e-12)
        unblanked=int(np.count_nonzero(intersecting & (background['status']!=1)))
        metrics.append(Metric(f"level-{record['level']}-solid-intersecting-background-not-hole",unblanked,.5,'Cartesian cell-box sphere intersection requires HOLE'))
        for direction,donor,receiver in (('background-from-body',body,background),('body-from-background',background,body)):
            raw=np.load(folder/(direction+'.npz'));ids=raw['receiver_indices'];di=raw['donor_indices'];w=raw['weights'];p=receiver['cell_centers'][ids]
            source=prescribed(donor['cell_centers']);truth=prescribed(p);mapped=np.einsum('ij,ijk->ik',w,source[di]);volume=receiver['cell_volumes'][ids]
            nonlinear=float(np.sqrt(np.sum(volume[:,None]*(mapped-truth)**2)/np.sum(volume[:,None]*truth**2)))
            component_errors=np.sqrt(np.sum(volume[:,None]*(mapped-truth)**2,axis=0)/np.sum(volume[:,None]*truth**2,axis=0))
            affine_source=np.column_stack((1+2*donor['cell_centers'][:,0]-3*donor['cell_centers'][:,1]+.7*donor['cell_centers'][:,2],-.2+donor['cell_centers'][:,0]+.4*donor['cell_centers'][:,1]-donor['cell_centers'][:,2]))
            affine_exact=np.column_stack((1+2*p[:,0]-3*p[:,1]+.7*p[:,2],-.2+p[:,0]+.4*p[:,1]-p[:,2]))
            constant=float(np.max(abs(w.sum(axis=1)-1)))
            affine=float(np.max(abs(np.einsum('ij,ijk->ik',w,affine_source[di])-affine_exact)))
            coordinate=float(np.max(np.linalg.norm(np.einsum('ij,ijk->ik',w,donor['cell_centers'][di])-p,axis=1)))
            unsafe=int(np.count_nonzero(donor['status'][di]!=0));unsafe_containing=int(np.count_nonzero(donor['status'][raw['containing_cells']]!=0))
            distances=np.linalg.norm(p,axis=1)
            separation=np.einsum('rkj,rj->rk',donor['cell_centers'][di],p/distances[:,None])-1
            safety_defect=float(max(0,-separation.min()))
            containment_defect=0.
            # Reconstruct the complete native cell boundary without importing its locator.
            fowner,fneighbor=donor['owner'],donor['neighbor']
            for point,ci in zip(p,raw['containing_cells']):
                faces=np.flatnonzero((fowner==ci)|(fneighbor==ci));sign=np.where(fowner[faces]==ci,1.,-1.)
                normalized=sign*np.einsum('ij,ij->i',point-donor['face_centers'][faces],donor['area_vectors'][faces])/np.linalg.norm(donor['area_vectors'][faces],axis=1)
                containment_defect=max(containment_defect,float(max(0,normalized.max())))
            prefix=f"level-{record['level']}-{direction}"
            checks=[('constant',constant,1e-12),('affine',affine,1e-10),('coordinate-reproduction',coordinate,1e-10),('negative-weight',float(max(0,-w.min())),1e-11),('inactive-donors',unsafe,.5),('inactive-containing-cells',unsafe_containing,.5),('sphere-clearance-defect',safety_defect,1e-11),('native-containment-defect',containment_defect,1e-10),('nonlinear-L2',nonlinear,.03),('saved-field-replay',float(np.max(abs(mapped-raw['transferred_receiver_field']))),1e-12),('saved-source-replay',float(np.max(abs(source-raw['donor_field']))),1e-12),('saved-reference-replay',float(np.max(abs(truth-raw['reference_receiver_field']))),1e-12),('saved-points-replay',float(np.max(abs(p-raw['receiver_points']))),1e-12)]
            checks.extend((f'nonlinear-component-{i}-L2',float(error),.03) for i,error in enumerate(component_errors))
            expected_receivers=np.flatnonzero(receiver['status']==2)
            checks.extend((('receiver-duplicate-count',len(ids)-len(np.unique(ids)),.5),('receiver-fringe-set-mismatch',len(np.setxor1d(ids,expected_receivers)),.5),('receiver-weight-replay',float(np.max(abs(volume-raw['receiver_volume_weights']))),1e-14)))
            metrics.extend(Metric(prefix+'-'+name,float(value),float(limit),'Independent raw geometry and point-map replay') for name,value,limit in checks)
            rows.append(dict(level=record['level'],direction=direction,relative_l2=nonlinear,constant_max=constant,affine_max=affine,coordinate_max=coordinate,minimum_sphere_tangent_clearance=float(separation.min()),native_containment_max_defect=containment_defect))
    for direction in ('background-from-body','body-from-background'):
        errors=[r['relative_l2'] for r in rows if r['direction']==direction]
        for i in range(2):metrics.append(Metric(f'{direction}-refinement-{i+1}-error-ratio',errors[i+1]/errors[i],1,'Strict decrease across actual archived refinement levels'))
    records,passed=evaluate(metrics)
    return dict(passed=passed,metrics=records,replayed_rows=rows,producer_artifacts_checked=len(recorded),producer_manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),method='Independent raw arrays; no connectivity builder or locator imported',flow_solved=False,fvm_lbm_coupling_executed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args();result=audit(args.directory)
    auditdir=args.directory/'independent-audit'
    if auditdir.exists():raise ValueError('audit destination already exists')
    auditdir.mkdir();root=Path(__file__).resolve().parents[1]
    for name in ('scripts/audit_overset3d.py','src/tensorfvm/verification/core.py'):
        target=auditdir/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    (auditdir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={str(p.relative_to(auditdir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(auditdir.rglob('*')) if p.is_file()}
    (auditdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
