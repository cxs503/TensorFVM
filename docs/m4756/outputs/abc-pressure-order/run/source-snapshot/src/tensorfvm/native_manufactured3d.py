"""Independent analytic ABC forcing and boundary quadrature for 3D NS checks.

Prescribed velocity at a SUBOFF wall is manufactured data, not physical
no-slip submarine flow. No discrete production residual defines the force.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .native_face_geometry3d import polygon_triangles


def abc_fields(points,*,amplitude=.5,wavenumber=.7,nu=.1,rho=1.):
    points=np.asarray(points,dtype=float);X,Y,Z=(points*wavenumber).T
    U=amplitude;k=wavenumber
    u=U*np.column_stack((np.sin(Z)+np.cos(Y),np.sin(X)+np.cos(Z),np.sin(Y)+np.cos(X)))
    G=np.zeros((len(points),3,3));G[:,0,1]=-U*k*np.sin(Y);G[:,0,2]=U*k*np.cos(Z);G[:,1,0]=U*k*np.cos(X);G[:,1,2]=-U*k*np.sin(Z);G[:,2,0]=-U*k*np.sin(X);G[:,2,1]=U*k*np.cos(Y)
    p=rho*U**2*np.sin(X)*np.cos(Y)*np.sin(Z)
    gp=rho*U**2*k*np.column_stack((np.cos(X)*np.cos(Y)*np.sin(Z),-np.sin(X)*np.sin(Y)*np.sin(Z),np.sin(X)*np.cos(Y)*np.cos(Z)))
    force=np.einsum('cij,cj->ci',G,u)+gp/rho+nu*k*k*u
    return dict(velocity=u,pressure=p,velocity_gradient=G,pressure_gradient=gp,acceleration=force)


def exact_boundary_flux(mesh,field,order=8):
    """Duffy tensor Gauss vector flux on convex planar native polygons."""
    if any(len(mesh.face_vertices[i]) != 3 for i in np.flatnonzero(mesh.neighbor < 0)):
        means = exact_boundary_velocity_means(mesh, field, order)
        return np.einsum('fk,fk->f', means, mesh.area_vectors)
    a,w=leggauss(order);a=(a+1)/2;w=w/2
    bary=np.array([[1-r,r*(1-s),r*s] for r in a for s in a])
    weights=np.array([2*r*wr*ws for r,wr in zip(a,w) for s,ws in zip(a,w)])
    b=np.flatnonzero(mesh.neighbor<0);result=np.zeros(len(mesh.owner))
    for start in range(0,len(b),256):
        ids=b[start:start+256]
        if any(len(mesh.face_vertices[i])!=3 for i in ids):raise ValueError('boundary quadrature requires native triangle facets')
        triangles=mesh.vertices[np.asarray([mesh.face_vertices[i] for i in ids])]
        nodes=np.einsum('qj,fjk->fqk',bary,triangles)
        velocity=field(nodes.reshape(-1,3))['velocity'].reshape(len(ids),len(weights),3)
        mean=np.einsum('q,fqk->fk',weights,velocity)
        result[ids]=np.einsum('fk,fk->f',mean,mesh.area_vectors[ids])
    return result



def exact_boundary_velocity_means(mesh, field, order=8):
    """Independent prescribed boundary area means; no interior reference use."""
    a,w=leggauss(order);a=(a+1)/2;w=w/2
    bary=np.array([[1-r,r*(1-s),r*s] for r in a for s in a])
    weights=np.array([2*r*wr*ws for r,wr in zip(a,w) for s,ws in zip(a,w)])
    b=np.flatnonzero(mesh.neighbor<0);result=np.zeros((len(mesh.owner),3))
    if any(len(mesh.face_vertices[i]) != 3 for i in b):
        # Fan triangles have the true geometry and area weights; do not apply
        # the triangle covariance/quadrature to a four-vertex face directly.
        pieces=[];face_ids=[];areas=[]
        for face in b:
            triangles=polygon_triangles(mesh.vertices[list(mesh.face_vertices[face])])
            pieces.extend(triangles);face_ids.extend([face]*len(triangles))
            areas.extend(np.linalg.norm(.5*np.cross(triangles[:,1]-triangles[:,0],
                                                    triangles[:,2]-triangles[:,0]),axis=1))
        pieces=np.asarray(pieces);face_ids=np.asarray(face_ids);areas=np.asarray(areas)
        for start in range(0,len(pieces),256):
            stop=min(start+256,len(pieces));triangles=pieces[start:stop]
            nodes=np.einsum('qj,fjk->fqk',bary,triangles)
            values=field(nodes.reshape(-1,3))['velocity'].reshape(len(triangles),len(weights),3)
            means=np.einsum('q,fqk->fk',weights,values)
            np.add.at(result,face_ids[start:stop],areas[start:stop,None]*means)
        result[b]/=np.linalg.norm(mesh.area_vectors[b],axis=1)[:,None]
        return result
    for start in range(0,len(b),256):
        ids=b[start:start+256]
        if any(len(mesh.face_vertices[i])!=3 for i in ids):raise ValueError('boundary quadrature requires native triangle facets')
        triangles=mesh.vertices[np.asarray([mesh.face_vertices[i] for i in ids])]
        nodes=np.einsum('qj,fjk->fqk',bary,triangles)
        values=field(nodes.reshape(-1,3))['velocity'].reshape(len(ids),len(weights),3)
        result[ids]=np.einsum('q,fqk->fk',weights,values)
    return result

def volume_weighted_error(actual,reference,volumes,*,remove_mean=False):
    a=np.asarray(actual);r=np.asarray(reference);v=np.asarray(volumes)
    if a.shape!=r.shape or a.shape[0]!=len(v) or not np.isfinite(a).all() or not np.isfinite(r).all() or np.any(v<=0):raise ValueError('finite matching fields and positive volumes required')
    if remove_mean:
        if a.ndim!=1:raise ValueError('gauge removal is scalar only')
        a=a-np.sum(a*v)/v.sum();r=r-np.sum(r*v)/v.sum()
    w=v.reshape((len(v),)+(1,)*(a.ndim-1))
    denominator=np.sum(w*r*r)
    if denominator<=0:raise ValueError('nonzero reference norm required')
    return float(np.sqrt(np.sum(w*(a-r)**2)/denominator))
