"""Actual coupled two-grid finite-volume diffusion verification.

Scalar -Laplacian(u)=f; this is not an incompressible Navier--Stokes solver.
Point-value donor constraints do not enforce equal interface fluxes.  Both the
algebraic residual and unique physical-domain diffusion balance are saved.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .overset import ACTIVE, HOLE, FRINGE


@dataclass
class OversetPoissonResult:
    fields: tuple
    exact_fields: tuple
    metrics: dict
    connectivity: object


def manufactured_value(points, radius=.3):
    p=np.asarray(points);x,y=p[...,0],p[...,1]
    return (x*x+y*y-radius*radius)*(1+.2*x+.1*y)


def manufactured_source(points):
    p=np.asarray(points)
    return -(4+1.6*p[...,0]+.8*p[...,1])


def _faces(grid):
    edges={}; result=[]
    for owner,poly in enumerate(grid.polygons):
        poly=np.asarray(poly)
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            key=tuple(sorted((tuple(a),tuple(b))))
            if key in edges:
                k=edges.pop(key); result[k][1]=owner
            else:
                e=b-a; S=np.array([e[1],-e[0]])
                edges[key]=len(result);result.append([owner,-1,(a+b)/2,S])
    return result


def solve_overset_poisson(connectivity, *, radius=.3):
    """Solve actual FV equations and simultaneous fringe interpolation rows.

    Domain is an origin-centred cylinder outside r=radius in a rectangular
    background. Component grid is the annulus. All physical outer boundaries
    prescribe the manufactured scalar; cylinder wall prescribes scalar zero.
    """
    grids=connectivity.grids; states=connectivity.states
    if len(grids)!=2 or not np.isfinite(radius) or radius<=0:
        raise ValueError('Expected background plus origin-centred circular component')
    inner=np.asarray(grids[1].boundaries.get('inner',[]),dtype=bool)
    outer=np.asarray(grids[1].boundaries.get('outer',[]),dtype=bool)
    if inner.shape!=(len(grids[1].centers),) or outer.shape!=inner.shape:
        raise ValueError('Component must expose annular inner/outer cell masks')
    if np.any(np.asarray(states[1])[outer]==ACTIVE):
        raise ValueError('Component outer boundary must be fringe constrained')
    # Actual polygon vertices define the inner circular approximation; reject
    # a changed centre/radius instead of silently changing the exact reference.
    inner_vertices=np.concatenate([grids[1].polygons[i] for i in np.flatnonzero(inner)])
    vertex_radii=np.linalg.norm(inner_vertices,axis=1)
    if not np.isclose(vertex_radii.min(),radius,rtol=1e-10,atol=1e-12):
        raise ValueError('Manufactured radius does not match origin-centred annulus')
    ring_vertices=inner_vertices[np.isclose(vertex_radii,radius,rtol=1e-10,atol=1e-12)]
    if len(ring_vertices)<8 or np.linalg.norm(np.unique(ring_vertices,axis=0).mean(axis=0))>1e-10:
        raise ValueError('Manufactured annulus must be centred at the origin')
    body=np.asarray(connectivity.body_polygon)
    if not np.allclose(np.linalg.norm(body,axis=1),radius,rtol=1e-10,atol=1e-12):
        raise ValueError('Hole-cutting body polygon must match manufactured radius')
    vertices=np.concatenate(grids[0].polygons);lo=vertices.min(axis=0);hi=vertices.max(axis=0)
    if not np.isclose(np.sum(grids[0].volumes),np.prod(hi-lo),rtol=1e-11,atol=1e-12):
        raise ValueError('Background must cover its full rectangular domain')
    offsets=np.cumsum([0]+[len(g.centers) for g in grids]);N=int(offsets[-1])
    rows=[];cols=[];data=[];rhs=np.zeros(N)
    def add(i,j,v):rows.append(int(i));cols.append(int(j));data.append(float(v))
    face_sets=[_faces(g) for g in grids]
    for k,(g,state,faces) in enumerate(zip(grids,states,face_sets)):
        centers=np.asarray(g.centers);vol=np.asarray(g.volumes)
        active=np.asarray(state)==ACTIVE
        rhs[offsets[k]:offsets[k+1]][active]=manufactured_source(centers[active])*vol[active]
        for owner,neigh,cf,S in faces:
            length=np.linalg.norm(S)
            if neigh>=0:
                d=centers[neigh]-centers[owner];dist=np.dot(d,S/length)
                if dist<=0:raise ValueError('nonpositive diffusion face distance')
                conductance=length/dist
                for cell,other in ((owner,neigh),(neigh,owner)):
                    if active[cell]:
                        if state[other]==HOLE:raise ValueError('active cell directly touches a hole')
                        add(offsets[k]+cell,offsets[k]+cell,conductance)
                        add(offsets[k]+cell,offsets[k]+other,-conductance)
            elif active[owner]:
                dist=np.dot(cf-centers[owner],S/length)
                if dist<=0:raise ValueError('nonpositive boundary distance')
                conductance=length/dist
                # Component physical inner boundary is stationary homogeneous
                # scalar Dirichlet. This must not be called an NS no-slip test.
                wall=(k==1 and np.linalg.norm(cf)<radius*1.1)
                value=0. if wall else float(manufactured_value(cf,radius))
                add(offsets[k]+owner,offsets[k]+owner,conductance)
                rhs[offsets[k]+owner]+=conductance*value
        for cell in np.flatnonzero(np.asarray(state)==HOLE):
            add(offsets[k]+cell,offsets[k]+cell,1.)
    constrained=set()
    for stencil in connectivity.stencils:
        rk=stencil.receiver_grid;dk=stencil.donor_grid
        for receiver,donors,weights in zip(stencil.receiver_cells,stencil.donor_cells,stencil.weights):
            row=offsets[rk]+receiver
            if row in constrained:raise ValueError('duplicate receiver constraint')
            constrained.add(row);add(row,row,1.)
            for donor,weight in zip(donors,weights):add(row,offsets[dk]+donor,-weight)
    expected=set(int(offsets[k]+i) for k,s in enumerate(states) for i in np.flatnonzero(np.asarray(s)==FRINGE))
    if constrained!=expected:raise ValueError('fringe cells missing donor equations')
    A=coo_matrix((data,(rows,cols)),shape=(N,N)).tocsr();u=spsolve(A,rhs)
    if not np.all(np.isfinite(u)):raise RuntimeError('nonfinite coupled diffusion solution')
    fields=tuple(u[offsets[k]:offsets[k+1]] for k in range(len(grids)))
    exact=tuple(manufactured_value(g.centers,radius) for g in grids)
    weighted_error=weighted_exact=0.;max_error=0.;interface=0.;interface_absolute=0.;physical=0.;source=0.
    for k,(g,s,faces,values,truth) in enumerate(zip(grids,states,face_sets,fields,exact)):
        active=np.asarray(s)==ACTIVE
        for owner,neigh,cf,S in faces:
            length=np.linalg.norm(S)
            if neigh>=0:
                conductance=length/np.dot(g.centers[neigh]-g.centers[owner],S/length)
                for cell,other in ((owner,neigh),(neigh,owner)):
                    if active[cell] and s[other]==FRINGE:
                        flux=conductance*(values[cell]-values[other]);interface+=flux;interface_absolute+=abs(flux)
            elif active[owner]:
                wall=(k==1 and np.linalg.norm(cf)<radius*1.1)
                boundary=0. if wall else float(manufactured_value(cf,radius))
                physical+=length/np.dot(cf-g.centers[owner],S/length)*(values[owner]-boundary)
        source+=float(np.dot(manufactured_source(g.centers[active]),g.volumes[active]))
        error=values[active]-truth[active]
        weighted_error+=float(np.dot(error*error,g.volumes[active]));weighted_exact+=float(np.dot(truth[active]**2,g.volumes[active]))
        max_error=max(max_error,float(np.max(abs(error))))
    interpolated=connectivity.interpolate(fields)
    constraint_error=max(float(np.max(abs(interpolated[k][np.asarray(s)==FRINGE]-fields[k][np.asarray(s)==FRINGE]))) if np.any(np.asarray(s)==FRINGE) else 0. for k,s in enumerate(states))
    # The two fringe surfaces enclose different regions. Their net flux is
    # generally nonzero even for the exact solution, because the overlap has
    # a volumetric source. Only the physical rectangle-minus-body balance
    # below is called a global conservation defect.
    inner_area=0.;inner_moment_x=0.;inner_moment_y=0.
    for owner,neigh,cf,S in face_sets[1]:
        if neigh<0 and np.linalg.norm(cf)<radius*1.1:
            inner_area-=.5*np.dot(cf,S)
            inner_moment_x-=.5*(cf[0]**2+S[1]**2/12)*S[0]
            inner_moment_y-=.5*(cf[1]**2+S[0]**2/12)*S[1]
    physical_source=float(np.dot(manufactured_source(grids[0].centers),grids[0].volumes)+4*inner_area+1.6*inner_moment_x+.8*inner_moment_y)
    conservation_defect=physical-physical_source
    metrics={
        'maximum_donor_constraint_residual':constraint_error,
        'relative_l2_error':float(np.sqrt(weighted_error/weighted_exact)),
        'maximum_absolute_error':max_error,
        'relative_algebraic_residual':float(np.linalg.norm(A@u-rhs)/max(np.linalg.norm(rhs),1e-30)),
        'raw_interface_net_flux':float(interface),
        'physical_domain_integrated_source':physical_source,
        'physical_global_conservation_defect':float(conservation_defect),
        'physical_global_conservation_defect_relative':float(abs(conservation_defect)/max(abs(physical_source),1e-30)),
        'interface_absolute_flux':float(interface_absolute),
        'physical_boundary_diffusive_flux':float(physical),
        'active_integrated_source':float(source),
        'summed_fv_balance_defect':float(physical+interface-source),
        'active_cells':int(sum(np.count_nonzero(np.asarray(s)==ACTIVE) for s in states)),
        'fringe_cells':len(constrained),
        'strict_local_conservation':False,
        'navier_stokes_verified':False,
        'error_norm_domain':'Both active component domains; overlap counted twice',
        'wall_condition':'Scalar homogeneous Dirichlet on polygonal circular boundary',
    }
    for k,(g,s,v,t) in enumerate(zip(grids,states,fields,exact)):
        active=np.asarray(s)==ACTIVE
        metrics[f'grid_{k}_relative_l2_error']=float(np.sqrt(np.dot((v[active]-t[active])**2,g.volumes[active])/np.dot(t[active]**2,g.volumes[active])))
    return OversetPoissonResult(fields,exact,metrics,connectivity)
