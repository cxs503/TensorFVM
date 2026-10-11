"""Local affine numerical field transfer for initialization, not flux coupling."""
import numpy as np
from scipy.spatial import cKDTree


def transfer_cell_values(source_centers, source_values, target_centers, *, neighbors=16):
    """Transfer point samples without analytic data or a conservation claim.

    Coincident centers copy exactly. Other targets use a distance-weighted local
    affine fit. The output is an initial guess and must be re-solved on the new
    mesh; it is unsuitable as a conservative overset exchange operator.
    """
    source=np.asarray(source_centers,float);target=np.asarray(target_centers,float)
    values=np.asarray(source_values,float)
    if (source.ndim!=2 or source.shape[1]!=3 or target.ndim!=2 or target.shape[1]!=3
            or values.ndim not in (1,2) or len(values)!=len(source) or len(source)<4
            or neighbors<4 or not all(np.isfinite(a).all() for a in (source,target,values))):
        raise ValueError('finite 3D centers, aligned scalar/vector samples and >=4 neighbors required')
    distance,indices=cKDTree(source).query(target,k=min(neighbors,len(source)))
    result=np.empty((len(target),)+values.shape[1:]);copied=0
    tolerance=32*np.finfo(float).eps*max(float(np.ptp(source,axis=0).max()),1.)
    for cell,(d,ids) in enumerate(zip(distance,indices)):
        if d[0]<=tolerance:
            result[cell]=values[ids[0]];copied+=1;continue
        offset=source[ids]-target[cell]
        scale=np.sqrt(np.mean(offset**2,axis=0))
        if np.any(scale<=tolerance):raise ValueError('rank-deficient local transfer geometry')
        basis=np.column_stack((np.ones(len(ids)),offset/scale))
        weights=d[0]/d
        weighted=weights if values.ndim==1 else weights[:,None]
        coefficient,_,rank,_=np.linalg.lstsq(basis*weights[:,None],values[ids]*weighted,rcond=1e-12)
        if rank<4:raise ValueError('rank-deficient local transfer geometry')
        result[cell]=coefficient[0]
    return result,dict(exact_center_copies=copied,affine_fits=len(target)-copied,
        source_cells=len(source),target_cells=len(target),neighbors=min(neighbors,len(source)),
        conservative_transfer=False,flow_solved=False)
