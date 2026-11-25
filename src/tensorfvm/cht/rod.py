"""Axial finite-volume heat conduction in rods with nested, stepped cross sections."""
from dataclasses import dataclass

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


@dataclass
class _Layout:
    left: np.ndarray
    right: np.ndarray
    shoulders: np.ndarray
    power_ends: np.ndarray

    @property
    def cells(self):
        # Indices in the system after the two prescribed ends are removed.
        return self.left

    def output(self, state, ends, g, flux_order, return_traces):
        if self.power_ends.size:
            # Appended unknowns are temperatures at ends with specified power.
            ends = ends.copy()
            ends[...,self.power_ends] = state[...,-self.power_ends.size:]
            state = state[...,:-self.power_ends.size]
        full = np.concatenate((ends[..., :1],state,ends[..., 1:]),axis=-1)
        cell = state[...,self.cells]
        west, east = full[...,self.left], full[...,self.right]
        faces = np.concatenate((ends[..., :1],full[...,self.shoulders],ends[..., 1:]),axis=-1)
        left = g*(west-cell)
        right = g*(cell-east)
        if flux_order == 2:
            correction = left-right
            left = left+correction
            right = right-correction
        result = (cell,faces,left,right)
        if return_traces:
            result += ((east[..., :-1],west[..., 1:]),)
        return result


def _rod_operator(dx, area, conductivity, capacity, film_coefficient, shoulder_areal_heat_capacity, flux_order, contact_resistance, end_types):
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
    if flux_order not in (1,2):
        raise ValueError("flux_order must be 1 or 2")
    resistance = _profile(contact_resistance,n-1,"contact resistance")
    if np.any(resistance < 0):
        raise ValueError("contact resistance must be nonnegative")
    split = resistance > 0
    left = 2*np.arange(n)+np.r_[0,np.cumsum(split)]
    right = left+2
    count = right[-1]+1
    shoulders = np.where(area[:-1]>=area[1:],right[:-1],left[1:])
    types = np.asarray(end_types)
    if types.shape != (2,) or not np.isin(types,["temperature","power"]).all():
        raise ValueError("end_types must specify temperature or power for each end")
    power_ends = np.flatnonzero(types == "power")
    layout = _Layout(left,right,shoulders,power_ends)
    links = np.empty(count-1)
    links[left] = links[left+1] = g
    links[right[:-1][split]] = np.minimum(area[:-1],area[1:])[split]/resistance[split]
    diagonal = np.r_[links,0.] + np.r_[0.,links]
    diagonal[shoulders] += film
    full = sp.diags((-links, diagonal, -links), (-1,0,1),format="csc")
    if flux_order == 2:
        indices = left[:,None]+np.arange(3)
        curvature = sp.csr_matrix((np.tile([1.,-2.,1.],n),indices.ravel(),3*np.arange(n+1)),
                                  shape=(n,count))
        full = full + curvature.T @ sp.diags(g) @ curvature
    endpoints = np.array([0,count-1])
    unknown = np.r_[np.arange(1,count-1),endpoints[power_ends]]
    operator = full[unknown][:,unknown].tocsc()
    end_load = -full[unknown][:,endpoints]
    if power_ends.size:
        # Dirichlet values enter through eliminated conductances, Neumann
        # powers enter their own massless endpoint energy balances.
        end_load = end_load @ sp.diags((types=="temperature").astype(float))
        end_load += sp.csc_matrix((np.ones(power_ends.size),
                                  (np.arange(count-2,len(unknown)),power_ends)),
                                  shape=(len(unknown),2))
    mass = np.zeros(len(unknown))
    mass[layout.cells] = c*volume
    mass[shoulders-1] = cw*exposed
    return operator,mass,volume,g,film,end_load,layout


def _load(volume, end_load, layout, film, ends, source, ambient, dtype):
    n = len(volume)
    if np.shape(ends) != (2,):
        raise ValueError("provide left and right end temperatures")
    ends = _profile(ends,2,"end boundary values",dtype)
    rhs = np.zeros(end_load.shape[0],dtype=dtype)
    rhs[layout.cells] = volume*_profile(source,n,"source",dtype)
    rhs[layout.shoulders-1] = film*_profile(ambient,n-1,"ambient temperature",dtype)
    rhs += end_load @ ends
    return rhs, ends


def solve_steady_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                             end_temperatures, *, source=0., film_coefficient=0.,
                             ambient_temperature=0., shoulder_areal_heat_capacity=0.,
                             flux_order=1, contact_resistance=0., return_traces=False,
                             end_types=("temperature","temperature"), mean_temperature=None):
    """Steady temperatures and rightward powers with the harmonic solver's conventions.

    When both ends prescribe power and every exposed film is insulating, the
    inward end powers plus volumetric heating must sum to zero. In this case
    mean_temperature fixes the heat-capacity-weighted mean of cells and storing
    shoulders, in K. Other boundary combinations determine the absolute level.
    """
    operator,mass,volume,g,film,end_load,layout = _rod_operator(
        dx,area,conductivity,volumetric_heat_capacity,film_coefficient,
        shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types)
    rhs, ends = _load(volume,end_load,layout,film,end_temperatures,source,ambient_temperature,float)
    if layout.power_ends.size==2 and not np.any(film):
        if mean_temperature is None or not np.isscalar(mean_temperature) or not np.isfinite(mean_temperature):
            raise ValueError("two power ends without film exchange require a finite mean_temperature")
        if abs(np.sum(rhs)) > 64*np.finfo(float).eps*np.sum(abs(rhs)):
            raise ValueError("incompatible steady load: net inward end power plus source must vanish")
        # Pin one cell during the solve and then shift every trace by the
        # same constant to impose the prescribed total stored energy.
        state = np.zeros(len(mass))
        if len(state)>1:
            state[1:] = spsolve(operator[1:,1:],rhs[1:])
        state += mean_temperature-(mass @ state)/mass.sum()
    else:
        if mean_temperature is not None:
            raise ValueError("mean_temperature only applies to an unanchored steady rod")
        state = spsolve(operator,rhs)
    return layout.output(state,ends,g,flux_order,return_traces)


def solve_harmonic_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                               end_temperatures, angular_frequency, *, source=0.,
                               film_coefficient=0., ambient_temperature=0.,
                               shoulder_areal_heat_capacity=0., flux_order=1, contact_resistance=0., return_traces=False,
                               end_types=("temperature","temperature")):
    """Solve a rod with concentric sections and prescribed end phasors.

    end_types specifies "temperature" (K) or "power" (inward W) for each
    value of end_temperatures. Zero power insulates an end. Both reported
    cell powers point right, so inward power at the right end changes sign.
    Cell inputs are width (m), area (m²), conductivity (W/(m K)), volumetric
    capacity (J/(m³ K)) and source (W/m³). At each internal face the absolute
    area change exposes a shoulder to a film (W/(m² K)) with surface storage
    (J/(m² K)). Film, capacity and ambient accept scalars or n-1 values.
    flux_order=2 reconstructs each face power from a quadratic matching the
    cell mean and its two faces. The default uses half-cell differences.
    Under exp(+iωt), return cell temperatures, all n+1 face temperatures,
    power entering each cell from its left and power leaving its right (W).
    contact_resistance in m² K/W applies to the overlapping (smaller) area.
    The larger section owns the exposed shoulder, whose temperature is reported
    at each split face (left trace for equal areas). With return_traces=True,
    append a pair of arrays giving left and right material traces at internal faces.
    """
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    operator,mass,volume,g,film,end_load,layout = _rod_operator(dx,area,conductivity,volumetric_heat_capacity,
                                               film_coefficient,shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types)
    rhs, ends = _load(volume,end_load,layout,film,end_temperatures,source,ambient_temperature,complex)
    state = spsolve(operator+sp.diags(1j*angular_frequency*mass),rhs)
    return layout.output(state,ends,g,flux_order,return_traces)


def solve_transient_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                                end_temperatures, time, initial_temperature, *, source=0.,
                                film_coefficient=0., ambient_temperature=0.,
                                shoulder_areal_heat_capacity=0., initial_shoulder_temperature=None,
                                theta=.5, flux_order=1, contact_resistance=0., return_traces=False,
                                method="theta", return_step_heat=False, end_types=("temperature","temperature")):
    """Advance the stepped rod with theta-weighted cell and shoulder storage.

    Uses the geometry, units, power signs and four outputs of the harmonic solve,
    each output now indexed first by time. End temperatures, ambient and source
    may be functions sampled at the supplied times. By default the initial face
    temperature balances its two neighbouring cells and its film instantaneously.
    initial_shoulder_temperature (scalar or n-1 values) overrides only faces
    carrying positive heat capacity. Massless faces always satisfy their balance.
    method="sdirk2" uses a two-stage second-order L-stable method. Its inputs
    are sampled at the internal stage and step end. method="rannacher" takes
    two backward Euler half steps on the first interval and Crank-Nicolson
    steps thereafter. Both half steps contribute to the reported heat for
    the first interval. method="positive_cn" uses internal Crank-Nicolson
    substeps with dt <= min(2*C_i/K_ii) over positive-capacity nodes. For
    flux_order=1 this preserves nonnegative temperatures with nonnegative
    initial and sampled boundary/source data. Returned times remain the
    requested times. With return_step_heat=True,
    append a dictionary of integrated joules per interval: left/right (positive
    toward increasing x), source, ambient (entering the shoulder), and contact
    (toward increasing x across the overlap, for each internal face).
    """
    time = np.asarray(time,dtype=float)
    if time.ndim != 1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):
        raise ValueError("time must be a finite strictly increasing vector")
    if not .5 <= theta <= 1.:
        raise ValueError("theta must be in [0.5,1]")
    if method not in ("theta","sdirk2","rannacher","positive_cn") or (method != "theta" and theta != .5):
        raise ValueError("choose a supported method; only theta integration accepts a theta override")
    if method == "positive_cn" and flux_order != 1:
        raise ValueError("positive_cn requires flux_order=1")
    weight = 1.-1./np.sqrt(2.) if method == "sdirk2" else theta
    operator,mass,volume,g,film,end_load,layout = _rod_operator(dx,area,conductivity,volumetric_heat_capacity,
                                                film_coefficient,shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types)
    n = len(g)
    def forcing(t):
        values = [v(float(t)) if callable(v) else v for v in (end_temperatures,source,ambient_temperature)]
        load, boundary = _load(volume,end_load,layout,film,*values,float)
        return load, boundary, values[1], values[2]

    states = np.empty((len(time),len(mass)))
    ends = np.empty((len(time),2))
    rhs,ends[0],src,air = forcing(time[0])
    initial = _profile(initial_temperature,n,"initial temperature")
    states[0,layout.cells] = initial
    # At t0 a supplied storing-face temperature is a constraint on adjacent
    # massless faces, because quadratic reconstructions couple the face traces.
    known = np.zeros(len(mass),dtype=bool)
    known[layout.cells] = True
    if initial_shoulder_temperature is not None:
        supplied = _profile(initial_shoulder_temperature,n-1,"initial shoulder temperature")
        ids = layout.shoulders-1
        storing = mass[ids]>0
        states[0,ids[storing]] = supplied[storing]
        known[ids[storing]] = True
    free = ~known
    if np.any(free):
        load = rhs[free] - operator[free][:,known] @ states[0,known]
        states[0,free] = spsolve(operator[free][:,free],load)
    M = sp.diags(mass,format="csc")
    previous_dt = None
    if return_step_heat:
        step_heat = {key:np.zeros((len(time)-1,size)) for key,size in
                     (("left",n),("right",n),("source",n),("ambient",n-1),("contact",n-1))}
        area_values = _profile(area,n,"area")
        larger_left = area_values[:-1]>=area_values[1:]
        resistance = _profile(contact_resistance,n-1,"contact resistance")
        split = resistance>0
        contact_g = np.minimum(area_values[:-1],area_values[1:])[split]/resistance[split]
    # Eliminating massless faces of the first-order conductance network
    # preserves nonpositive off-diagonals and can only lower its diagonal.
    # This bound therefore makes the explicit CN half nonnegative too.
    max_dt = np.inf
    if method == "positive_cn":
        active = mass>0
        max_dt = np.min(2*mass[active]/operator.diagonal()[active])
    for step, interval in enumerate(np.diff(time)):
        subdivisions = max(1,int(np.ceil(interval/max_dt)))
        start = time[step]
        u,boundary_start = states[step],ends[step]
        for part in range(1,subdivisions+1):
            stop = time[step+1] if part == subdivisions else time[step]+part*interval/subdivisions
            dt = stop-start
            if dt != previous_dt:
                lu = splu(M+weight*dt*operator)
                previous_dt = dt
            stage = u
            stage_ends = boundary_start
            if method == "sdirk2" or (method == "rannacher" and step == 0):
                rhs,stage_ends,src,air = forcing(start+weight*dt)
                stage = lu.solve(mass*u+weight*dt*rhs)
            next_rhs,next_ends,next_src,next_air = forcing(stop)
            next_u = lu.solve(mass*u+dt*((1-weight)*(rhs-operator@stage)+weight*next_rhs))
            if return_step_heat:
                # Flux, storage and film operators are linear, so their time
                # integrals can be evaluated on the stage-weighted state.
                mean = (1-weight)*stage+weight*next_u
                boundary = (1-weight)*stage_ends+weight*next_ends
                _,faces,left,right,(up,down) = layout.output(mean,boundary,g,flux_order,True)
                step_heat["left"][step] += dt*left
                step_heat["right"][step] += dt*right
                step_heat["source"][step] += dt*volume*((1-weight)*np.asarray(src)+weight*np.asarray(next_src))
                step_heat["ambient"][step] += dt*film*((1-weight)*np.asarray(air)+weight*np.asarray(next_air)-faces[1:-1])
                contact = np.where(larger_left,left[1:],right[:-1])
                contact[split] = contact_g*(up[split]-down[split])
                step_heat["contact"][step] += dt*contact
            rhs,src,air = next_rhs,next_src,next_air
            u,boundary_start,start = next_u,next_ends,stop
        states[step+1],ends[step+1] = u,boundary_start
    output = layout.output(states,ends,g,flux_order,return_traces)
    return output+(step_heat,) if return_step_heat else output
