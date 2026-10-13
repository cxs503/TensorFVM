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


class NativeSSTWall3D:
    """Selected hybrid Spalding + Foundation-12 blended wall coupling.

    This combines the native cell-sampled momentum boundary with the pinned
    omega/G wall formula; it is not identical to OpenFOAM's full discretization.
    The physical-wall gradient estimate is |U_cell-U_wall|/distance, not the
    gradient of the auxiliary hybrid face trace. Bulk closure remains SST-2003.
    """
    def __init__(self,momentum_wall):
        from .native_spalding3d import NativeSpaldingWall3D
        if not isinstance(momentum_wall,NativeSpaldingWall3D):
            raise TypeError('native Spalding momentum wall required')
        self.momentum=momentum_wall;self.mesh=momentum_wall.mesh

    def validate_transport(self,k_mask,omega_mask,nu,rho,k_neumann,omega_neumann):
        w=self.momentum
        if np.any(k_mask[w.mask]) or np.any(omega_mask[w.mask]):
            raise ValueError('wall model requires free scalar faces with zero diffusive flux')
        for data in (k_neumann,omega_neumann):
            if np.any(np.broadcast_to(np.asarray(data,float),(w.F,))[w.mask]!=0):
                raise ValueError('wall model requires zero k/omega wall diffusive flux')
        if rho!=w.density or not np.allclose(np.asarray(nu)[w.owner],w.molecular_viscosity/rho,rtol=1e-14,atol=0.):
            raise ValueError('SST and momentum wall molecular properties differ')

    def apply(self,closure,k,cell_velocity):
        w=self.momentum;velocity=np.asarray(cell_velocity,float)
        if velocity.shape!=(w.N,3) or not np.isfinite(velocity).all():
            raise ValueError('finite current cell velocity required by SST wall model')
        k=np.asarray(k);div=closure['velocity_divergence']
        # Foundation G is deviatoric shear production. Its k*div(U) term
        # sits OUTSIDE Pk's limiter; keep this distinction at wall cells.
        deviatoric=closure['nu_t']*(closure['strain_squared']-2/3*div*div)
        gradient=np.linalg.norm(velocity[w.owner]-w.wall_velocity,axis=1)/w.distance
        face=foundation12_blended_omega_wall(k[w.owner],w.distance,w.molecular_viscosity/w.density,
                                            gradient,deviatoric[w.owner])
        cell=area_average_wall_cells(w.owner,w.area,face['omega'],face['production'])
        ids=cell['cells'];out=dict(closure)
        for key in ('production_raw','production_limited','k_source'):out[key]=closure[key].copy()
        dilatation=2/3*k[ids]*div[ids]
        out['production_raw'][ids]=cell['production']-dilatation
        out['production_limited'][ids]=np.minimum(cell['production'],closure['production_limit'][ids])-dilatation
        out['k_source'][ids]=out['production_limited'][ids]-closure['k_destruction'][ids]
        # omega has its independent G/nu_t production expression. Its wall
        # rows are constrained and their original-equation reactions retained.
        out['wall_model']=dict(**cell,faces=w.faces.copy(),face_omega=face['omega'],
            face_production=face['production'],y_plus_k=face['y_plus_k'],
            physical_wall_gradient_magnitude=gradient,
            gradient_sampling='|U_cell-U_wall|/distance; physical wall, not auxiliary face trace',
            variant='native SST-2003 bulk; Foundation-12 blended omega/G wall formulas',
            wall_formula_precision='float64',physical_benchmark_passed=False)
        return out
