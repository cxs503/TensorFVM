"""Independent diagnostics from native collocated FV fields (no smoothing)."""
import numpy as np


def flux_coupling(owner, neighbor, area, volumes, flux, face_velocity, speed):
    owner, neighbor = np.asarray(owner), np.asarray(neighbor)
    area, volumes, flux = map(np.asarray, (area, volumes, flux))
    interior = neighbor >= 0
    interpolated = (np.asarray(face_velocity) * area).sum(axis=1)
    def divergence(q):
        net = np.bincount(owner, weights=q, minlength=len(volumes))
        net -= np.bincount(neighbor[interior], weights=q[interior], minlength=len(volumes))
        return float(np.max(np.abs(net / volumes)))
    defect = np.abs(flux-interpolated)[interior]/np.linalg.norm(area[interior],axis=1)/speed
    return dict(native_face_divergence_max=divergence(flux),
                interpolated_cell_velocity_divergence_max=divergence(interpolated),
                interior_max_normal_velocity_difference=float(defect.max()),
                interior_p95_normal_velocity_difference=float(np.quantile(defect,.95)),
                physical_coupling_qualified=False)


def angular_pressure(pressure, nx, ny, nz):
    """Span-mean native innermost ring; high-frequency content is diagnostic.

    A high-frequency fraction is not a general turbulence acceptance limit.
    Here it detects the startup mesh-locked modes, including k near Nyquist.
    """
    wall=np.asarray(pressure).reshape(nz,ny,nx)[:,0].mean(axis=0)
    fluctuation=wall-wall.mean()
    power=np.abs(np.fft.rfft(fluctuation))**2
    total=power[1:].sum()
    curvature=np.roll(wall,1)-2*wall+np.roll(wall,-1)
    return dict(wall_pressure_min=float(wall.min()),wall_pressure_max=float(wall.max()),
                high_angular_frequency_energy_fraction=float(power[(nx+3)//4:].sum()/total)if total else 0.,
                angular_second_difference_relative_l2=float(np.linalg.norm(curvature)/np.linalg.norm(fluctuation))if total else 0.,
                pressure_coupling_qualified=False)
