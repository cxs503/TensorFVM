"""Axial finite-volume heat conduction in rods with nested, stepped cross sections."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve


def _profile(value, length, name, dtype=float):
    raw = np.asarray(value)
    if dtype is float and np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    values = np.broadcast_to(np.asarray(value, dtype=dtype), (length,))
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _rod_operator(dx, area, conductivity, capacity, film_coefficient, shoulder_areal_heat_capacity):
    dx = np.asarray(dx, dtype=float)
    if dx.ndim != 1 or not dx.size or not np.isfinite(dx).all() or np.any(dx <= 0):
        raise ValueError("dx must be a nonempty vector of finite positive widths")
    n = dx.size
    area = _profile(area,n,"area")
    k = _profile(conductivity,n,"conductivity")
    c = _profile(capacity,n,"volumetric heat capacity")
    if np.any(area <= 0) or np.any(k <= 0) or np.any(c <= 0):
        raise ValueError("area, conductivity and volumetric heat capacity must be positive")
    h = _profile(film_coefficient,n-1,"film coefficient")
    cw = _profile(shoulder_areal_heat_capacity,n-1,"shoulder areal heat capacity")
    if np.any(h < 0) or np.any(cw < 0):
        raise ValueError("film coefficient and shoulder heat capacity must be nonnegative")
    exposed = np.abs(np.diff(area))
    film = h*exposed
    volume = area*dx
    g = 2*k*area/dx
    # Interleave cell means and shared face temperatures. Each half-cell
    # conducts from its own cell to the face through its own section area.
    diagonal = np.empty(2*n-1)
    diagonal[::2] = 2*g
    diagonal[1::2] = g[:-1]+g[1:]+film
    links = np.repeat(-g,2)[1:-1]
    operator = sp.diags((links, diagonal, links), (-1,0,1),format="csc")
    mass = np.empty(2*n-1)
    mass[::2] = c*volume
    mass[1::2] = cw*exposed
    return operator,mass,volume,g,film


def _load(volume, g, film, ends, source, ambient, dtype):
    n = len(g)
    if np.shape(ends) != (2,):
        raise ValueError("provide left and right end temperatures")
    ends = _profile(ends,2,"end temperatures",dtype)
    rhs = np.empty(2*n-1,dtype=dtype)
    rhs[::2] = volume*_profile(source,n,"source",dtype)
    rhs[1::2] = film*_profile(ambient,n-1,"ambient temperature",dtype)
    rhs[0] += g[0]*ends[0]
    rhs[-1] += g[-1]*ends[1]
    return rhs, ends


def solve_harmonic_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                               end_temperatures, angular_frequency, *, source=0.,
                               film_coefficient=0., ambient_temperature=0.,
                               shoulder_areal_heat_capacity=0.):
    """Solve a rod with perfectly bonded concentric sections and prescribed end phasors.

    Cell inputs are width (m), area (m²), conductivity (W/(m K)), volumetric
    capacity (J/(m³ K)) and source (W/m³). At each internal face the absolute
    area change exposes a shoulder to a film (W/(m² K)) with surface storage
    (J/(m² K)). Film, capacity and ambient accept scalars or n-1 values.
    Under exp(+iωt), return cell temperatures, all n+1 face temperatures,
    power entering each cell from its left and power leaving its right (W).
    """
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    operator,mass,volume,g,film = _rod_operator(dx,area,conductivity,volumetric_heat_capacity,
                                               film_coefficient,shoulder_areal_heat_capacity)
    rhs, ends = _load(volume,g,film,end_temperatures,source,ambient_temperature,complex)
    state = spsolve(operator+sp.diags(1j*angular_frequency*mass),rhs)
    cell = state[::2]
    faces = np.r_[ends[0],state[1::2],ends[1]]
    return cell,faces,g*(faces[:-1]-cell),g*(cell-faces[1:])


def solve_transient_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                                end_temperatures, time, initial_temperature, *, source=0.,
                                film_coefficient=0., ambient_temperature=0.,
                                shoulder_areal_heat_capacity=0., initial_shoulder_temperature=None,
                                theta=.5):
    """Advance the stepped rod with theta-weighted cell and shoulder storage.

    Uses the geometry, units, power signs and four outputs of the harmonic solve,
    each output now indexed first by time. End temperatures, ambient and source
    may be functions sampled at the supplied times. By default the initial face
    temperature balances its two neighbouring cells and its film instantaneously.
    initial_shoulder_temperature (scalar or n-1 values) overrides only faces
    carrying positive heat capacity. Massless faces always satisfy their balance.
    """
    time = np.asarray(time,dtype=float)
    if time.ndim != 1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):
        raise ValueError("time must be a finite strictly increasing vector")
    if not .5 <= theta <= 1.:
        raise ValueError("theta must be in [0.5,1]")
    operator,mass,volume,g,film = _rod_operator(dx,area,conductivity,volumetric_heat_capacity,
                                                film_coefficient,shoulder_areal_heat_capacity)
    n = len(g)
    def forcing(t):
        values = [v(float(t)) if callable(v) else v for v in (end_temperatures,source,ambient_temperature)]
        return _load(volume,g,film,*values,float)

    states = np.empty((len(time),2*n-1))
    ends = np.empty((len(time),2))
    rhs,ends[0] = forcing(time[0])
    initial = _profile(initial_temperature,n,"initial temperature")
    states[0,::2] = initial
    initial_faces = (rhs[1::2]+g[:-1]*initial[:-1]+g[1:]*initial[1:])/(g[:-1]+g[1:]+film)
    if initial_shoulder_temperature is not None:
        supplied = _profile(initial_shoulder_temperature,n-1,"initial shoulder temperature")
        initial_faces = np.where(mass[1::2]>0,supplied,initial_faces)
    states[0,1::2] = initial_faces
    M = sp.diags(mass,format="csc")
    previous_dt = None
    for step, dt in enumerate(np.diff(time)):
        if dt != previous_dt:
            lu = splu(M+theta*dt*operator)
            previous_dt = dt
        next_rhs, ends[step+1] = forcing(time[step+1])
        u = states[step]
        states[step+1] = lu.solve(mass*u+dt*((1-theta)*(rhs-operator@u)+theta*next_rhs))
        rhs = next_rhs
    cells = states[:,::2]
    faces = np.column_stack((ends[:,0],states[:,1::2],ends[:,1]))
    return cells,faces,g*(faces[:,:-1]-cells),g*(cells-faces[:,1:])
