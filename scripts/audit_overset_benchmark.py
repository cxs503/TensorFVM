"""Independently replay saved overset FV diffusion and donor rows with NumPy."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import write_json,sha,artifact_manifest


def audit(output):
    summary=json.loads((output/'summary.json').read_text());records=[]
    for row in summary['runs']:
        directory=output/row['directory'];f=np.load(directory/'fields.npz');c=np.load(directory/'connectivity.npz')
        r=row['config']['body_radius'];residuals=[];physical_flux=0.;wall_area=0.
        states=[];centers=[];values=[]
        for k in (0,1):
            p=f[f'g{k}_polygons'];nextp=np.roll(p,-1,axis=1)
            cross=p[:,:,0]*nextp[:,:,1]-nextp[:,:,0]*p[:,:,1];area=cross.sum(1)/2
            center=((p+nextp)*cross[:,:,None]).sum(1)/(6*area[:,None])
            np.testing.assert_allclose(area,f[f'g{k}_volumes'],rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(center,f[f'g{k}_centers'],rtol=1e-11,atol=1e-12)
            if not np.all(area>0):raise ValueError('Nonpositive cell area')
            state=f[f'g{k}_state'];np.testing.assert_array_equal(state,c[f'g{k}_state'])
            u=f[f'g{k}_solution'];truth=(center[:,0]**2+center[:,1]**2-r*r)*(1+.2*center[:,0]+.1*center[:,1])
            np.testing.assert_allclose(truth,f[f'g{k}_exact'],rtol=1e-12,atol=1e-12)
            states.append(state);centers.append(center);values.append(u)
            active=state==0;res=-( -(4+1.6*center[:,0]+.8*center[:,1])*area)
            known={};edges=[]
            for owner,polygon in enumerate(p):
                for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                    key=tuple(sorted((tuple(a),tuple(b))))
                    if key in known:
                        fi=known[key]
                        if edges[fi][1]!=-1:raise ValueError('Nonmanifold edge')
                        edges[fi][1]=owner
                    else:
                        known[key]=len(edges);edges.append([owner,-1,a,b])
            for owner,neighbor,a,b in edges:
                delta=b-a;S=np.array([delta[1],-delta[0]]);length=np.linalg.norm(S);cf=(a+b)/2
                if neighbor>=0:
                    distance=np.dot(center[neighbor]-center[owner],S)/length
                    if distance<=0:raise ValueError('Negative diffusion projection')
                    flux=length/distance*(u[owner]-u[neighbor])
                    if active[owner]:
                        if state[neighbor]==1:raise ValueError('Active cell touches blanked hole')
                        res[owner]+=flux
                    if active[neighbor]:
                        if state[owner]==1:raise ValueError('Active cell touches blanked hole')
                        res[neighbor]-=flux
                elif active[owner]:
                    wall=k==1 and np.linalg.norm(cf)<r*1.1
                    if wall:
                        np.testing.assert_allclose(np.linalg.norm([a,b],axis=1),r,atol=1e-12,rtol=0)
                        boundary=0.;wall_area-=np.dot(cf,S)/2
                    else:boundary=(np.dot(cf,cf)-r*r)*(1+.2*cf[0]+.1*cf[1])
                    distance=np.dot(cf-center[owner],S)/length
                    if distance<=0:raise ValueError('Nonpositive boundary projection')
                    flux=length/distance*(u[owner]-boundary);res[owner]+=flux;physical_flux+=flux
            if k==0:
                # Rectangle cells may be blanked near the component; none of the
                # actually calculated/receiver polygons may touch the solid.
                minimum=np.linalg.norm(np.maximum(np.maximum(p.min(1),0),-p.max(1)),axis=1)
                if np.any((state!=1)&(minimum<=r)):raise ValueError('Cartesian cell enters physical solid')
                full_source=float(np.dot(-(4+1.6*center[:,0]+.8*center[:,1]),area))
            residuals.extend(res[active]);error=np.sqrt(np.dot((u[active]-truth[active])**2,area[active])/np.dot(truth[active]**2,area[active]))
            np.testing.assert_allclose(error,row['computed'][f'grid_{k}_relative_l2_error'],rtol=1e-10,atol=1e-12)
        donor_residual=0.;affine_error=0.;receivers=set()
        for k in (0,1):
            rg=int(c[f's{k}_receiver_grid']);dg=int(c[f's{k}_donor_grid']);rc=c[f's{k}_receiver_cells'];dc=c[f's{k}_donor_cells'];w=c[f's{k}_weights']
            if rg==dg or not np.all(states[dg][dc]==0) or not np.all(states[rg][rc]==2):raise ValueError('Invalid active donor/fringe relation')
            np.testing.assert_allclose(w.sum(1),1,atol=1e-12,rtol=0)
            if np.any(w< -1e-12):raise ValueError('Extrapolation weight')
            affine_error=max(affine_error,float(np.max(abs((centers[dg][dc]*w[:,:,None]).sum(1)-centers[rg][rc]))))
            donor_residual=max(donor_residual,float(np.max(abs((values[dg][dc]*w).sum(1)-values[rg][rc]))))
            for i in rc:
                if (rg,int(i)) in receivers:raise ValueError('Duplicate constraint')
                receivers.add((rg,int(i)))
        if receivers!={(k,int(i)) for k,s in enumerate(states) for i in np.flatnonzero(s==2)}:raise ValueError('Orphan receivers')
        body=c['body_polygon'];q=np.roll(body,-1,axis=0);cross=body[:,0]*q[:,1]-q[:,0]*body[:,1];body_area=cross.sum()/2;body_moment=((body+q)*cross[:,None]).sum(0)/6
        np.testing.assert_allclose(body_area,wall_area,atol=1e-12,rtol=0)
        physical_source=full_source+4*body_area+1.6*body_moment[0]+.8*body_moment[1]
        global_error=abs(physical_flux-physical_source)/abs(physical_source)
        np.testing.assert_allclose(global_error,row['computed']['physical_global_conservation_defect_relative'],rtol=1e-8,atol=1e-12)
        maximum=float(np.max(np.abs(residuals)))
        passed=bool(maximum<1e-9 and donor_residual<1e-10 and affine_error<1e-10)
        if not passed:raise ValueError('Independent FV or donor replay failed')
        record=dict(resolution=row['resolution'],passed=passed,independent_equation_max_residual=maximum,donor_constraint_max_residual=donor_residual,affine_coordinate_max_error=affine_error,physical_global_conservation_relative_error=global_error,strict_local_conservation=False,navier_stokes_verified=False)
        write_json(directory/'audit.json',record);records.append(record);f.close();c.close()
    write_json(output/'audit.json',dict(passed=all(r['passed'] for r in records),runs=records,scope='NumPy replay from raw polygons, areas, centroids, face incidence, full ACTIVE diffusion, physical wall/outer boundaries and saved donor constraints; solver and connectivity modules are not imported. Scalar equations only.'))
    manifest=json.loads((output/'manifest.json').read_text());manifest['source_sha256']['scripts/audit_overset_benchmark.py']=sha(Path(__file__));manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();audit(a.output)
