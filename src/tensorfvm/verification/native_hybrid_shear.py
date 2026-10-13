"""Unforced variable-viscosity Couette flow; a stress audit, not an SST model."""
import numpy as np
from .native_hybrid_flow import channel_mesh
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means, volume_weighted_error


def shear_fields(points, kind='smooth'):
    y=np.asarray(points)[:,1]
    if kind=='smooth':
        viscosity=1+y
        speed=np.log1p(y)/np.log(2.)
        stress=1/np.log(2.)
    elif kind=='jump':
        viscosity=np.where(y<.5,1.,4.)
        speed=np.where(y<.5,y,.5+(y-.5)/4)/.625
        stress=1/.625
    else:
        raise ValueError('shear kind must be smooth or jump')
    velocity=np.zeros((len(y),3));velocity[:,0]=speed
    return dict(velocity=velocity,viscosity=viscosity,shear_stress=stress)


def shear_solver(n,kind='smooth'):
    if not isinstance(n,int) or n<2 or n%2:
        raise ValueError('positive even wall-normal subdivisions required')
    mesh=channel_mesh((2,n,1));b=mesh.neighbor<0;z=mesh.face_centers[:,2]
    symmetry=b & ((z<1e-10)|(z>1-1e-10))
    fixed=np.repeat(b[:,None],3,axis=1);fixed[symmetry,:2]=False
    field=lambda points:shear_fields(points,kind)
    exact=field(mesh.cell_centers)
    boundary=exact_boundary_velocity_means(mesh,field)
    solver=NativeHybridStokes3D(mesh,fixed,boundary,viscosity=exact['viscosity'])
    return mesh,solver,exact


def shear_metrics(mesh,result,exact):
    b=mesh.neighbor<0;y=mesh.face_centers[:,1]
    lower=b & (y<1e-10);upper=b & (y>1-1e-10)
    lower_force=float(result['total_traction_integrated'][lower,0].sum())
    upper_force=float(result['total_traction_integrated'][upper,0].sum())
    tau=exact['shear_stress']
    row=dict(cells=len(mesh.cells),numerically_converged=result['numerically_converged'],
        velocity_relative_l2=volume_weighted_error(result['velocity'],exact['velocity'],mesh.cell_volumes),
        pressure_maximum_absolute_error=float(np.max(np.abs(result['pressure']))),
        lower_force=lower_force,upper_force=upper_force,exact_stress=tau,
        lower_force_relative_error=abs(lower_force+tau)/tau,
        upper_force_relative_error=abs(upper_force-tau)/tau,
        energy_balance_error=result['energy_balance_error'],
        interface_traction_defect_max=result['interface_traction_defect_max'],
        global_force_balance_error=float(np.max(np.abs(result['total_traction_integrated'][b].sum(0)))))
    row['velocity_and_wall_force_3_percent_passed']=bool(row['numerically_converged'] and
        max(row[k] for k in ('velocity_relative_l2','lower_force_relative_error','upper_force_relative_error'))<.03)
    return row
