"""Explicit Foundation-12 blended omega/G wall values and wall-cell reduction.

This is a boundary constitutive component, not an integrated RANS wall model.
Callers must supply the actual wall-normal velocity-gradient magnitude, wall
distance, and the unmodified current production G. Momentum traction, k's
boundary condition, the cell omega constraint, and production replacement
must be coupled consistently before claiming an integrated wall treatment.

Formula and area weighting reference: OpenFOAM Foundation 12 (20250206),
omegaWallFunctionFvPatchScalarField::calculate and
wallCellWallFunctionFvPatchScalarField::patchFieldsToWallCellField.
Only blended=True and ordinary conformal full-wall cells are covered.
"""
import numpy as np


def foundation12_blended_omega_wall(k, distance, nu, normal_gradient_magnitude,
                                   production, *, beta1=.075, cmu=.09,
                                   kappa=.41, E=9.8):
    """Return specific G and omega for each wall face (SI or consistent units).

    y_plus_k is the k-derived coordinate used by this omega condition; it is
    not the Spalding stress-derived y+. k=0 uses the analytic viscous limit,
    avoiding the literal formula's 0/log(0) arithmetic. No k clipping or
    replacement of the Foundation blend by a hypotenuse formula is applied.
    The log-law pole at E*y_plus_k=1 is rejected for nonzero wall gradient.
    """
    constants=np.asarray([beta1,cmu,kappa,E],float)
    if not np.isfinite(constants).all() or np.any(constants<=0):
        raise ValueError('positive finite wall constants required')
    k,y,nu,grad,G=np.broadcast_arrays(*[np.asarray(a,float) for a in
        (k,distance,nu,normal_gradient_magnitude,production)])
    if not all(np.isfinite(a).all() for a in (k,y,nu,grad,G)):
        raise ValueError('finite wall inputs required')
    if np.any(k<0) or np.any(y<=0) or np.any(nu<=0) or np.any(grad<0):
        raise ValueError('k and gradient must be nonnegative; distance and nu positive')
    rey=y*np.sqrt(k)/nu;yp=cmu**.25*rey
    lam=np.exp(-rey/11);turb=-np.expm1(-rey/11)
    ustar=np.sqrt(lam*nu*grad+turb*np.sqrt(cmu)*k)
    omega_vis=6*nu/(beta1*y*y)
    omega_log=ustar/(np.sqrt(cmu)*kappa*y)
    omega=lam*omega_vis+turb*omega_log
    # Evaluate the turbulent term only where its numerator can be nonzero.
    active=(k>0)&(grad>0);up=np.ones_like(yp)
    np.log(E*yp,out=up,where=active);up=np.where(active,up/kappa,1.)
    if np.any(active & (up==0)):
        raise ValueError('Foundation-12 blended production is singular at E*y_plus_k=1')
    g_turb=np.zeros_like(G)
    np.divide(turb*(ustar*grad*y/up)**2,nu*kappa*yp,out=g_turb,where=active)
    value=lam*G+g_turb
    if not np.isfinite(omega).all() or not np.isfinite(value).all():
        raise ValueError('nonfinite Foundation-12 wall value; no clipping applied')
    return dict(omega=omega,production=value,y_plus_k=yp,viscous_fraction=lam,
                turbulent_fraction=turb,u_star_blended=ustar)


def area_average_wall_cells(owner, area, omega, production):
    """Reduce all participating wall faces to full wall-cell constraints.

    Uses face AREA weights, including when faces from multiple patches share
    a cell. Splitting a planar polygon into triangles must not change its
    contribution. This does not implement partially covered/nonconformal
    wall-cell fractions. Returned sorted cells and omega are suitable for
    NativeHybridScalar3D's explicit cell_constraints interface.
    """
    owner=np.asarray(owner)
    if owner.ndim!=1 or not np.issubdtype(owner.dtype,np.integer) or np.any(owner<0):
        raise ValueError('nonnegative one-dimensional integer wall owners required')
    area,omega,production=[np.asarray(v,float) for v in (area,omega,production)]
    if any(v.shape!=owner.shape or not np.isfinite(v).all() for v in (area,omega,production)):
        raise ValueError('one finite area, omega and production per wall face required')
    if np.any(area<=0) or np.any(omega<=0):raise ValueError('positive face area and omega required')
    cells,inverse=np.unique(owner,return_inverse=True)
    total=np.bincount(inverse,weights=area,minlength=len(cells))
    averaged=[np.bincount(inverse,weights=area*v,minlength=len(cells))/total for v in (omega,production)]
    return dict(cells=cells,omega=averaged[0],production=averaged[1],wall_area=total)
