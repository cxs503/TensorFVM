"""Independent NumPy replay of saved NACA overset nonorthogonal diffusion."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def exact(p):
    x,y=np.asarray(p).T
    return 1+x*x+y*y+.2*x*y+.1*x+.15*y


def audit(output):
    summary=json.loads((output/'summary.json').read_text());records=[]
    for row in summary['runs']:
        directory=output/row['directory'];f=np.load(directory/'fields.npz');c=np.load(directory/'connectivity.npz');states=[];centers=[];values=[];errors=[];physical=0.;residuals=[]
        for k in (0,1):
            polygons=f[f'g{k}_polygons'];other=np.roll(polygons,-1,axis=1);cross=polygons[:,:,0]*other[:,:,1]-other[:,:,0]*polygons[:,:,1];volume=cross.sum(1)/2;center=((polygons+other)*cross[:,:,None]).sum(1)/(6*volume[:,None])
            np.testing.assert_allclose(volume,f[f'g{k}_volumes'],rtol=1e-10,atol=1e-13);np.testing.assert_allclose(center,f[f'g{k}_centers'],atol=1e-11,rtol=1e-10)
            if np.any(volume<=0):raise ValueError('Invalid polygon area')
            state=f[f'g{k}_state'];np.testing.assert_array_equal(state,c[f'g{k}_state']);u=f[f'g{k}_solution'];states.append(state);centers.append(center);values.append(u)
            np.testing.assert_allclose(exact(center),f[f'g{k}_exact'],atol=1e-11,rtol=1e-11)
            adjacency=[set() for _ in volume];boundaries=[[] for _ in volume];known={};faces=[]
            for owner,poly in enumerate(polygons):
                for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                    if np.array_equal(a,b):continue
                    key=tuple(sorted((tuple(a),tuple(b))))
                    if key in known:
                        face=faces[known[key]]
                        if face[1]!=-1:raise ValueError('Nonmanifold cell edge')
                        neighbor=face[0];face[1]=owner;adjacency[owner].add(neighbor);adjacency[neighbor].add(owner)
                    else:
                        known[key]=len(faces);e=b-a;faces.append([owner,-1,(a+b)/2,np.array([e[1],-e[0]])])
            for o,n,cf,S in faces:
                if n<0 and state[o]==0:boundaries[o].append(cf)
            gradient=np.zeros((len(volume),2))
            for i in np.flatnonzero(state!=1):
                neighbors=[j for j in sorted(adjacency[i]) if state[j]!=1];points=np.array([center[j] for j in neighbors]+boundaries[i]);delta=points-center[i]
                field=np.r_[u[neighbors],exact(np.array(boundaries[i])) if boundaries[i] else []]-u[i]
                weights=1/(delta*delta).sum(1);gradient[i]=np.linalg.solve((delta.T*weights)@delta,(delta.T*weights)@field)
            residual=4*volume
            for o,n,cf,S in faces:
                if state[o]!=0 and (n<0 or state[n]!=0):continue
                if n>=0 and (state[o]==1 or state[n]==1):raise ValueError('Active/hole adjacency')
                d=center[n]-center[o] if n>=0 else cf-center[o];projection=S@d
                if projection<=0:raise ValueError('Nonpositive face projection')
                a=S@S/projection;T=S-a*d
                if n>=0:
                    lam=(cf-center[o])@S/projection;g=(1-lam)*gradient[o]+lam*gradient[n];flux=a*(u[o]-u[n])-T@g
                    if state[o]==0:residual[o]+=flux
                    if state[n]==0:residual[n]-=flux
                else:
                    flux=a*(u[o]-exact(cf[None])[0])-T@gradient[o];residual[o]+=flux;physical+=flux
            residuals.extend(residual[state==0]);truth=exact(center);error=float(np.sqrt(np.dot((u[state==0]-truth[state==0])**2,volume[state==0])/np.dot(truth[state==0]**2,volume[state==0])));errors.append(error)
            np.testing.assert_allclose(error,row['computed'][f'grid_{k}_relative_l2_error'],rtol=1e-9,atol=1e-12)
            if k==0:background_area=volume.sum()
        receiver_set=set();constraint=0.;affine=0.;interpolation=[]
        for k in (0,1):
            rg=int(c[f's{k}_receiver_grid']);dg=int(c[f's{k}_donor_grid']);rc=c[f's{k}_receiver_cells'];dc=c[f's{k}_donor_cells'];w=c[f's{k}_weights']
            if rg==dg or not np.all(states[dg][dc]==0) or not np.all(states[rg][rc]==2):raise ValueError('Invalid donor states')
            np.testing.assert_allclose(w.sum(1),1,atol=1e-12,rtol=0)
            if np.any(w<0):raise ValueError('Negative interpolation weights')
            constraint=max(constraint,float(np.max(abs((values[dg][dc]*w).sum(1)-values[rg][rc]))));affine=max(affine,float(np.max(abs((centers[dg][dc]*w[:,:,None]).sum(1)-centers[rg][rc]))))
            t=exact(centers[rg][rc]);donor_truth=exact(centers[dg][dc].reshape(-1,2)).reshape(dc.shape);interpolation.append(float(np.linalg.norm((donor_truth*w).sum(1)-t)/np.linalg.norm(t)))
            for i in rc:
                if (rg,int(i)) in receiver_set:raise ValueError('Duplicate receiver')
                receiver_set.add((rg,int(i)))
        if receiver_set!={(k,int(i)) for k,s in enumerate(states) for i in np.flatnonzero(s==2)}:raise ValueError('Orphan receivers')
        body=c['body_polygon'];q=np.roll(body,-1,axis=0);area=abs(np.sum(body[:,0]*q[:,1]-q[:,0]*body[:,1])/2);source=-4*(background_area-area);global_error=abs(physical-source)/abs(source)
        np.testing.assert_allclose(global_error,row['computed']['physical_global_conservation_defect_relative'],atol=1e-10,rtol=1e-8)
        # Independent dense analytic profile check in body coordinates.
        angle=-np.radians(row['config']['incidence_deg']);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);local=(body-[1.5,1.5])@R+[.25,0]
        x=(1-np.cos(np.linspace(0,np.pi,4097)))/2;y=.6*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4);profile=np.r_[np.c_[x[::-1],y[::-1]],np.c_[x[1:],-y[1:]]];a,b=profile[:-1],profile[1:];segments=b-a;deviation=0.
        for start in range(0,len(local),32):
            d=local[start:start+32,None]-a;fraction=np.clip(np.sum(d*segments,axis=-1)/np.sum(segments*segments,axis=-1),0,1);deviation=max(deviation,float(np.linalg.norm(d-fraction[:,:,None]*segments,axis=-1).min(1).max()))
        maximum=float(np.max(np.abs(residuals)));passed=bool(maximum<1e-8 and constraint<1e-10 and affine<1e-10 and deviation<5e-6)
        if not passed:raise ValueError('Independent equation or NACA geometry audit failed')
        record=dict(resolution=row['resolution'],passed=passed,independent_equation_max_residual=maximum,donor_constraint_max_residual=constraint,affine_coordinate_max_error=affine,quadratic_interpolation_relative_l2_errors=interpolation,body_max_deviation_m=deviation,component_relative_l2_errors=errors,physical_global_balance_relative_error=global_error,navier_stokes_verified=False,strict_local_conservation=False)
        write_json(directory/'audit.json',record);records.append(record);f.close();c.close()
    write_json(output/'audit.json',dict(passed=True,runs=records,scope='Independent raw polygon geometry, dense analytic NACA0012 conformity, NumPy LS gradients/full nonorthogonal scalar diffusion, two-way ACTIVE donors and unique-domain global scalar flux; no overset production module imported.'))
    m=json.loads((output/'manifest.json').read_text());m['source_sha256']['scripts/audit_naca_overset.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();audit(a.output)
