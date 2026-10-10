"""Generic static overset manufactured diffusion with nonorthogonal correction.

Actual scalar equations on polygonal grids; no fluid pressure/velocity claim.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix,diags,block_diag
from scipy.sparse.linalg import spsolve
from .overset import ACTIVE,HOLE,FRINGE


def exact_value(p):
    p=np.asarray(p);x,y=p[...,0],p[...,1]
    return 1+x*x+y*y+.2*x*y+.1*x+.15*y


def face_geometry(grid):
    edges={};faces=[]
    for owner,p in enumerate(grid.polygons):
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            key=tuple(sorted((tuple(a),tuple(b))))
            if key in edges:
                f=faces[edges[key]]
                if f[1]!=-1:raise ValueError('Nonmanifold face')
                f[1]=owner
            else:
                edge=b-a;edges[key]=len(faces);faces.append([owner,-1,(a+b)/2,np.array([edge[1],-edge[0]])])
    return faces


@dataclass
class ScalarResult:
    fields:tuple
    exact_fields:tuple
    metrics:dict
    connectivity:object


def solve_manufactured_diffusion(conn):
    grids,states=conn.grids,conn.states
    offsets=np.cumsum([0]+[len(g.centers) for g in grids]);N=offsets[-1]
    face_sets=[face_geometry(g) for g in grids];gx=[];gy=[];constant=[]
    # LS gradients contain no reference data on either interpolation boundary.
    for g,state,faces in zip(grids,states,face_sets):
        boundary=[[] for _ in g.centers]
        for o,n,cf,S in faces:
            if n<0 and state[o]==ACTIVE:boundary[o].append(cf)
        rows=[];cols=[];xx=[];yy=[];bc=np.zeros((len(state),2))
        for i in np.flatnonzero(state!=HOLE):
            nb=[int(j) for j in g.neighbors[i] if state[j]!=HOLE]
            points=[g.centers[j] for j in nb]+boundary[i]
            d=np.array(points)-g.centers[i]
            if len(d)<2:raise ValueError('Insufficient gradient stencil')
            w=1/np.sum(d*d,axis=1);gram=(d.T*w)@d
            if np.linalg.cond(gram)>1e12:raise ValueError('Singular gradient stencil')
            coefficients=np.linalg.solve(gram,d.T*w)
            for k,j in enumerate(nb):rows.append(i);cols.append(j);xx.append(coefficients[0,k]);yy.append(coefficients[1,k])
            rows.append(i);cols.append(i);xx.append(-coefficients[0].sum());yy.append(-coefficients[1].sum())
            if boundary[i]:bc[i]=coefficients[:,len(nb):]@exact_value(np.array(boundary[i]))
        shape=(len(state),len(state));gx.append(coo_matrix((xx,(rows,cols)),shape=shape).tocsr());gy.append(coo_matrix((yy,(rows,cols)),shape=shape).tocsr());constant.append(bc)
    Gx,Gy=block_diag(gx,format='csr'),block_diag(gy,format='csr');bc=np.concatenate(constant)
    ro=[];co=[];val=[];rhs=np.zeros(N);flux_rows=[];flux_cols=[];flux_data=[];physical=[]
    for k,(g,state,faces) in enumerate(zip(grids,states,face_sets)):
        for o,n,cf,S in faces:
            if state[o]!=ACTIVE and (n<0 or state[n]!=ACTIVE):continue
            if n>=0 and (state[o]==HOLE or state[n]==HOLE):raise ValueError('Active cell touches hole')
            d=(g.centers[n]-g.centers[o]) if n>=0 else cf-g.centers[o]
            projection=np.dot(S,d)
            if projection<=0:raise ValueError('Nonpositive face projection')
            a=np.dot(S,S)/projection;T=S-a*d
            oi=offsets[k]+o;ni=offsets[k]+n if n>=0 else -1
            lam=np.dot(cf-g.centers[o],S)/projection if n>=0 else 0.
            gradient=(1-lam)*np.array([oi]) # weights assembled below
            q= a*(coo_matrix(([1.,-1.],([0,0],[oi,ni])),shape=(1,N)).tocsr()) if n>=0 else coo_matrix(([a],([0],[oi])),shape=(1,N)).tocsr()
            q-=T[0]*((1-lam)*Gx[oi]+lam*Gx[ni]) if n>=0 else T[0]*Gx[oi]
            q-=T[1]*((1-lam)*Gy[oi]+lam*Gy[ni]) if n>=0 else T[1]*Gy[oi]
            qconst=-np.dot(T,(1-lam)*bc[oi]+lam*bc[ni]) if n>=0 else -a*float(exact_value(cf))-np.dot(T,bc[oi])
            for cell,sign in ((o,1.),(n,-1.)):
                if cell<0 or state[cell]!=ACTIVE:continue
                row=offsets[k]+cell;ro.extend([row]*len(q.indices));co.extend(q.indices);val.extend(sign*q.data);rhs[row]-=sign*qconst
            if n<0:physical.append((q,qconst))
        active=np.flatnonzero(state==ACTIVE);rhs[offsets[k]+active]+=-4*g.volumes[active]
        for i in np.flatnonzero(state==HOLE):ro.append(offsets[k]+i);co.append(offsets[k]+i);val.append(1.)
    received=set()
    for s in conn.stencils:
        for i,donors,w in zip(s.receiver_cells,s.donor_cells,s.weights):
            row=offsets[s.receiver_grid]+i
            if row in received:raise ValueError('Duplicate constraint')
            received.add(row);ro.append(row);co.append(row);val.append(1.)
            for j,v in zip(donors,w):ro.append(row);co.append(offsets[s.donor_grid]+j);val.append(-v)
    expected={int(offsets[k]+i) for k,state in enumerate(states) for i in np.flatnonzero(state==FRINGE)}
    if expected!=received:raise ValueError('Missing donor constraints')
    A=coo_matrix((val,(ro,co)),shape=(N,N)).tocsr();u=spsolve(A,rhs)
    if not np.isfinite(u).all():raise ValueError('Nonfinite scalar solution')
    fields=tuple(u[offsets[k]:offsets[k+1]] for k in range(2));truth=tuple(exact_value(g.centers) for g in grids)
    m=dict(relative_algebraic_residual=float(np.linalg.norm(A@u-rhs)/np.linalg.norm(rhs)),strict_local_conservation=False,navier_stokes_verified=False)
    for k,(g,state,v,t) in enumerate(zip(grids,states,fields,truth)):
        active=state==ACTIVE;m[f'grid_{k}_relative_l2_error']=float(np.sqrt(np.dot((v[active]-t[active])**2,g.volumes[active])/np.dot(t[active]**2,g.volumes[active])))
    exchanged=conn.interpolate(fields);m['maximum_donor_constraint_residual']=max(float(np.max(abs(exchanged[k][state==FRINGE]-fields[k][state==FRINGE]))) for k,state in enumerate(states))
    body=conn.body_polygon;q=np.roll(body,-1,axis=0);area=abs(np.sum(body[:,0]*q[:,1]-q[:,0]*body[:,1])/2)
    source=-4*(grids[0].volumes.sum()-area);physical_flux=sum(float((q@u)[0]+b) for q,b in physical)
    m.update(physical_global_conservation_defect_relative=abs(physical_flux-source)/abs(source),physical_boundary_diffusive_flux=physical_flux,physical_domain_integrated_source=float(source),active_cells=sum(int(np.sum(s==ACTIVE)) for s in states),fringe_cells=len(received))
    return ScalarResult(fields,truth,m,conn)
