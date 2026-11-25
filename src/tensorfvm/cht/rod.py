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
    free_ends: np.ndarray
    anchored: bool
    end_film: np.ndarray
    left_fraction: np.ndarray
    left_storage: np.ndarray
    right_storage: np.ndarray

    @property
    def cells(self):
        # Indices in the system after the two prescribed ends are removed.
        return self.left

    def output(self, state, ends, g, flux_order, return_traces):
        if self.free_ends.size:
            # Appended unknowns are temperatures at power and convection boundaries.
            ends = ends.copy()
            ends[...,self.free_ends] = state[...,-self.free_ends.size:]
            state = state[...,:-self.free_ends.size]
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


def _rod_operator(dx, area, conductivity, capacity, film_coefficient, shoulder_areal_heat_capacity, flux_order, contact_resistance, end_types, end_film_coefficient, end_areal_heat_capacity, interface_heat_capacity, interface_left_fraction, interface_capacity_left_fraction):
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
    sheet_capacity = _profile(interface_heat_capacity,n-1,"interface heat capacity")
    if np.any(sheet_capacity < 0):
        raise ValueError("interface heat capacity must be nonnegative")
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
    if types.shape != (2,) or not np.isin(types,["temperature","power","robin"]).all():
        raise ValueError("end_types must specify temperature, power or robin for each end")
    free_ends = np.flatnonzero(types != "temperature")
    h_end = _profile(end_film_coefficient,2,"end film coefficient")
    if np.any(h_end < 0):
        raise ValueError("end film coefficient must be nonnegative")
    end_film = h_end*area[[0,-1]]*(types=="robin")
    anchored = free_ends.size < 2 or np.any(end_film) or np.any(film)
    def fraction_profile(value,name):
        if value is None:
            return (area[:-1]>=area[1:]).astype(float)
        fraction = _profile(value,n-1,name)
        if np.any((fraction<0) | (fraction>1)):
            raise ValueError(f"{name} must be in [0,1]")
        return fraction
    left_fraction = fraction_profile(interface_left_fraction,"interface left fraction")
    capacity_fraction = fraction_profile(interface_capacity_left_fraction,"interface capacity left fraction")
    layout = _Layout(left,right,shoulders,free_ends,anchored,end_film,left_fraction,
                     sheet_capacity*capacity_fraction,sheet_capacity*(1-capacity_fraction))
    links = np.empty(count-1)
    links[left] = links[left+1] = g
    links[right[:-1][split]] = np.minimum(area[:-1],area[1:])[split]/resistance[split]
    diagonal = np.r_[links,0.] + np.r_[0.,links]
    diagonal[[0,-1]] += end_film
    diagonal[shoulders] += film
    full = sp.diags((-links, diagonal, -links), (-1,0,1),format="csc")
    if flux_order == 2:
        indices = left[:,None]+np.arange(3)
        curvature = sp.csr_matrix((np.tile([1.,-2.,1.],n),indices.ravel(),3*np.arange(n+1)),
                                  shape=(n,count))
        full = full + curvature.T @ sp.diags(g) @ curvature
    endpoints = np.array([0,count-1])
    unknown = np.r_[np.arange(1,count-1),endpoints[free_ends]]
    operator = full[unknown][:,unknown].tocsc()
    end_load = -full[unknown][:,endpoints]
    if free_ends.size:
        # Dirichlet values enter through eliminated conductances, Neumann
        # powers enter their own massless endpoint energy balances.
        end_load = end_load @ sp.diags((types=="temperature").astype(float))
        end_load += sp.csc_matrix((np.where(types=="power",1.,end_film)[free_ends],
                                  (np.arange(count-2,len(unknown)),free_ends)),
                                  shape=(len(unknown),2))
    mass = np.zeros(len(unknown))
    end_capacity = _profile(end_areal_heat_capacity,2,"end areal heat capacity")
    if np.any(end_capacity<0) or np.any(end_capacity[types=="temperature"]):
        raise ValueError("end heat capacity must be nonnegative and zero on fixed-temperature ends")
    mass[count-2:] = (end_capacity*area[[0,-1]])[free_ends]
    mass[layout.cells] = c*volume
    mass[shoulders-1] = cw*exposed
    mass[right[:-1]-1] += layout.left_storage
    mass[left[1:]-1] += layout.right_storage
    return operator,mass,volume,g,film,end_load,layout


def _load(volume, end_load, layout, film, ends, source, ambient, interface_power, trace_power, dtype):
    n = len(volume)
    if np.shape(ends) != (2,):
        raise ValueError("provide left and right end boundary values")
    ends = _profile(ends,2,"end boundary values",dtype)
    rhs = np.zeros(end_load.shape[0],dtype=dtype)
    rhs[layout.cells] = volume*_profile(source,n,"source",dtype)
    rhs[layout.shoulders-1] = film*_profile(ambient,n-1,"ambient temperature",dtype)
    power = _profile(interface_power,n-1,"interface power",dtype)
    extra_left,extra_right = trace_power
    deposited = np.stack((layout.left_fraction*power+_profile(extra_left,n-1,"left trace power",dtype),
                          (1-layout.left_fraction)*power+_profile(extra_right,n-1,"right trace power",dtype)))
    rhs[layout.right[:-1]-1] += deposited[0]
    rhs[layout.left[1:]-1] += deposited[1]
    rhs += end_load @ ends
    return rhs, ends, deposited


def solve_steady_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                             end_temperatures, *, source=0., source_gradient=None, source_curvature=None, film_coefficient=0.,
                             ambient_temperature=0., shoulder_areal_heat_capacity=0.,
                             flux_order=1, contact_resistance=0., return_traces=False,
                             end_types=("temperature","temperature"), end_film_coefficient=0., end_areal_heat_capacity=0., mean_temperature=None, interface_power=0., interface_heat_capacity=0., interface_left_fraction=None, interface_capacity_left_fraction=None, trace_power=(0.,0.)):
    """Steady temperatures and rightward powers with the harmonic solver's conventions.

    source gives the cell mean in W/m³. With flux_order=2, source_gradient
    and source_curvature optionally specify the first and second spatial
    derivatives at each cell midpoint (W/m⁴ and W/m⁵). The resulting steady
    reconstruction integrates a cellwise quadratic generation profile.
    When neither end fixes temperature and all films are insulating, the
    inward end powers, interface power and volumetric heating must sum to zero. In this case
    mean_temperature fixes the heat-capacity-weighted mean of cells and storing
    shoulders and ends, in K. Other boundary combinations determine the absolute level.
    """
    operator,mass,volume,g,film,end_load,layout = _rod_operator(
        dx,area,conductivity,volumetric_heat_capacity,film_coefficient,
        shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types,end_film_coefficient,end_areal_heat_capacity,interface_heat_capacity,interface_left_fraction,interface_capacity_left_fraction)
    rhs, ends, _ = _load(volume,end_load,layout,film,end_temperatures,source,ambient_temperature,interface_power,trace_power,float)
    offsets = None
    if source_gradient is not None or source_curvature is not None:
        if flux_order != 2:
            raise ValueError("source derivatives require flux_order=2")
        n = len(volume)
        width = np.asarray(dx)
        slope = _profile(0. if source_gradient is None else source_gradient,n,"source gradient")
        curvature = _profile(0. if source_curvature is None else source_curvature,n,"source curvature")
        # For z measured from the midpoint, remove the cell mean from
        # slope*z + curvature*z²/2. Integrating -k*T'' twice gives cubic
        # and quartic corrections to the two rightward face powers.
        odd = volume*width*slope/12
        even = volume*width**2*curvature/120
        west, east = odd-even, odd+even
        offsets = west, east
        rhs[layout.cells] += west-east
        rhs[layout.left[1:]-1] -= west[1:]
        rhs[layout.right[:-1]-1] += east[:-1]
        if layout.free_ends.size:
            rhs[-len(layout.free_ends):] += np.array([-west[0],east[-1]])[layout.free_ends]
    if not layout.anchored:
        if mean_temperature is None or not np.isscalar(mean_temperature) or not np.isfinite(mean_temperature):
            raise ValueError("unanchored ends without film exchange require a finite mean_temperature")
        if abs(np.sum(rhs)) > 64*np.finfo(float).eps*np.sum(abs(rhs)):
            raise ValueError("incompatible steady load: net inward power plus source must vanish")
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
    result = layout.output(state,ends,g,flux_order,return_traces)
    if offsets is None:
        return result
    cell, faces, left, right, *traces = result
    return (cell, faces, left+offsets[0], right+offsets[1], *traces)


def solve_harmonic_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                               end_temperatures, angular_frequency, *, source=0.,
                               film_coefficient=0., ambient_temperature=0.,
                               shoulder_areal_heat_capacity=0., flux_order=1, contact_resistance=0., return_traces=False,
                               end_types=("temperature","temperature"), end_film_coefficient=0.,
                               end_areal_heat_capacity=0., interface_power=0., interface_heat_capacity=0., interface_left_fraction=None, interface_capacity_left_fraction=None, trace_power=(0.,0.)):
    """Solve a rod with concentric sections and prescribed end phasors.

    end_types selects "temperature" (K), "power" (inward W), or "robin"
    (ambient K) for each value of end_temperatures. At a Robin end the inward
    power equals end_film_coefficient times end area times ambient minus wall
    temperature, h*A*(Tamb-Twall). Supply a scalar h or a west/east pair
    in W/(m² K). Only Robin ends use their coefficient. end_areal_heat_capacity
    (J/(m² K), scalar or pair) adds surface storage to power or Robin ends.
    External input then supplies both storage and conduction. Both reported
    cell powers point right, so inward power at the right end changes sign.
    Cell inputs are width (m), area (m²), conductivity (W/(m K)), volumetric
    capacity (J/(m³ K)) and source (W/m³). At each internal face the absolute
    area change exposes a shoulder to a film (W/(m² K)) with surface storage
    (J/(m² K)). Film, capacity and ambient accept scalars or n-1 values.
    interface_power supplies W at each internal face (scalar or n-1 values).
    interface_heat_capacity adds lumped J/K at the same face (scalar or n-1
    values), in addition to the capacity on its exposed area. With a contact
    jump, both default to the shoulder trace, left for equal areas.
    interface_left_fraction (scalar or one value per interface, in [0,1])
    deposits this fraction of the applied power on the left trace and the
    remainder on the right. interface_capacity_left_fraction independently
    partitions interface_heat_capacity in the same way. The exposed-area
    shoulder capacity always belongs to the larger section. trace_power is a
    (left, right) pair of additional signed powers in W. Each entry may be
    scalar or one value per interface, with independent phases in harmonic runs.
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
                                               film_coefficient,shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types,end_film_coefficient,end_areal_heat_capacity,interface_heat_capacity,interface_left_fraction,interface_capacity_left_fraction)
    rhs, ends, _ = _load(volume,end_load,layout,film,end_temperatures,source,ambient_temperature,interface_power,trace_power,complex)
    state = spsolve(operator+sp.diags(1j*angular_frequency*mass),rhs)
    return layout.output(state,ends,g,flux_order,return_traces)


def solve_transient_stepped_rod(dx, area, conductivity, volumetric_heat_capacity,
                                end_temperatures, time, initial_temperature, *, source=0.,
                                film_coefficient=0., ambient_temperature=0.,
                                shoulder_areal_heat_capacity=0., initial_shoulder_temperature=None,
                                theta=.5, flux_order=1, contact_resistance=0., return_traces=False,
                                method="theta", return_step_heat=False, end_types=("temperature","temperature"), end_film_coefficient=0.,
                               end_areal_heat_capacity=0., initial_end_temperature=None, initial_trace_temperatures=(None,None), load_jump_times=(), interface_power=0., interface_heat_capacity=0., interface_left_fraction=None, interface_capacity_left_fraction=None, trace_power=(0.,0.)):
    """Advance the stepped rod with theta-weighted cell and shoulder storage.

    Uses the geometry, units, power signs and four outputs of the harmonic solve,
    each output now indexed first by time. End temperatures, ambient and source
    may be functions sampled at the supplied times. By default the initial face
    temperature balances its two neighbouring cells and its film instantaneously.
    initial_shoulder_temperature (scalar or n-1 values) overrides only faces
    carrying positive shoulder or interface heat capacity. Massless faces always satisfy their balance.
    initial_trace_temperatures is a (left, right) pair. Each entry may be
    None, a scalar or n-1 temperatures for storing material traces. Prescribed
    temperatures on a shared degree of freedom must agree.
    initial_end_temperature (scalar or pair) sets the temperature of storing
    ends. By default those ends initially balance the applied heat input.
    method="sdirk2" uses a two-stage second-order L-stable method. Its inputs
    are sampled at the internal stage and step end. method="rannacher" takes
    two backward Euler half steps on the first interval and Crank-Nicolson
    steps thereafter. Both half steps contribute to the reported heat for
    the first interval. method="positive_cn" uses internal Crank-Nicolson
    substeps with dt <= min(2*C_i/K_ii) over positive-capacity nodes. For
    flux_order=1 this preserves nonnegative temperatures with nonnegative
    initial and sampled boundary/source data. Returned times remain the
    requested times. load_jump_times declares distinct interior times at which
    the end, source or ambient loads change discontinuously. Callables give
    the new value at the jump. The previous step uses its left limit; massless
    traces at the event are recomputed with the new forcing. Rannacher damping
    also restarts immediately after each jump. With return_step_heat=True,
    append a dictionary of integrated joules per interval: left/right (positive
    toward increasing x), source, ambient (entering the shoulder), and contact
    (toward increasing x across the overlap, for each internal face).
    trace_power may be a callable returning the pair at time t.
    interface_power may also be callable. The total signed interface input in joules
    is recorded per face in step_heat["interface"]. trace_left and trace_right
    give the portions deposited on each trace, including split interface_power.
    end_input gives external
    heat entering each end, including surface storage.
    """
    time = np.asarray(time,dtype=float)
    if time.ndim != 1 or len(time)<2 or not np.isfinite(time).all() or np.any(np.diff(time)<=0):
        raise ValueError("time must be a finite strictly increasing vector")
    jumps = np.asarray(load_jump_times,dtype=float)
    if (jumps.ndim!=1 or not np.isfinite(jumps).all() or len(np.unique(jumps))!=len(jumps)
            or np.any(jumps<=time[0]) or np.any(jumps>=time[-1])):
        raise ValueError("load_jump_times must contain distinct finite interior times")
    if not .5 <= theta <= 1.:
        raise ValueError("theta must be in [0.5,1]")
    if method not in ("theta","sdirk2","rannacher","positive_cn") or (method != "theta" and theta != .5):
        raise ValueError("choose a supported method; only theta integration accepts a theta override")
    if method == "positive_cn" and flux_order != 1:
        raise ValueError("positive_cn requires flux_order=1")
    weight = 1.-1./np.sqrt(2.) if method == "sdirk2" else theta
    operator,mass,volume,g,film,end_load,layout = _rod_operator(dx,area,conductivity,volumetric_heat_capacity,
                                                film_coefficient,shoulder_areal_heat_capacity,flux_order,contact_resistance,end_types,end_film_coefficient,end_areal_heat_capacity,interface_heat_capacity,interface_left_fraction,interface_capacity_left_fraction)
    n = len(g)
    def forcing(t):
        values = [v(float(t)) if callable(v) else v for v in (end_temperatures,source,ambient_temperature,interface_power,trace_power)]
        load, boundary, deposited = _load(volume,end_load,layout,film,*values,float)
        return load, boundary, values[1], values[2], deposited

    states = np.empty((len(time),len(mass)))
    ends = np.empty((len(time),2))
    rhs,ends[0],src,air,face_power = forcing(time[0])
    initial = _profile(initial_temperature,n,"initial temperature")
    states[0,layout.cells] = initial
    # At t0 a supplied storing-face temperature is a constraint on adjacent
    # massless faces, because quadratic reconstructions couple the face traces.
    known = np.zeros(len(mass),dtype=bool)
    known[layout.cells] = True
    initial_west,initial_east = initial_trace_temperatures
    for values,ids,name in ((initial_shoulder_temperature,layout.shoulders-1,"initial shoulder temperature"),
                            (initial_west,layout.right[:-1]-1,"initial left trace temperature"),
                            (initial_east,layout.left[1:]-1,"initial right trace temperature")):
        if values is None:
            continue
        supplied = _profile(values,n-1,name)
        storing = mass[ids]>0
        fixed = ids[storing]
        overlap = known[fixed]
        if np.any(states[0,fixed[overlap]]!=supplied[storing][overlap]):
            raise ValueError("conflicting initial temperatures for the same storing trace")
        states[0,fixed] = supplied[storing]
        known[fixed] = True
    if initial_end_temperature is not None:
        supplied = _profile(initial_end_temperature,2,"initial end temperature")
        ids = np.arange(len(mass)-layout.free_ends.size,len(mass))
        storing = mass[ids]>0
        states[0,ids[storing]] = supplied[layout.free_ends[storing]]
        known[ids[storing]] = True
    free = ~known
    if np.any(free):
        load = rhs[free] - operator[free][:,known] @ states[0,known]
        states[0,free] = spsolve(operator[free][:,free],load)
    M = sp.diags(mass,format="csc")
    previous_dt = None
    if return_step_heat:
        step_heat = {key:np.zeros((len(time)-1,size)) for key,size in
                     (("left",n),("right",n),("source",n),("ambient",n-1),("interface",n-1),("trace_left",n-1),("trace_right",n-1),("contact",n-1),("end_input",2))}
        types = np.asarray(end_types)
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
    events = set(jumps)
    segments = np.union1d(time,jumps)
    massless = mass==0
    if events and np.any(massless):
        trace_factor = splu(operator[massless][:,massless].tocsc())
        coupling = operator[massless][:,~massless]
    step = 0
    u,boundary_start = states[0],ends[0]
    for begin,end in zip(segments,segments[1:]):
        interval = end-begin
        subdivisions = max(1,int(np.ceil(interval/max_dt)))
        start = begin
        damp = method=="rannacher" and (begin==time[0] or begin in events)
        for part in range(1,subdivisions+1):
            stop = end if part==subdivisions else begin+part*interval/subdivisions
            dt = stop-start
            if dt != previous_dt:
                lu = splu(M+weight*dt*operator)
                previous_dt = dt
            stage = u
            stage_ends = boundary_start
            if method == "sdirk2" or damp:
                stage_time = start+weight*dt
                if not start<stage_time<stop:
                    raise ValueError("internal stage has no representable time")
                rhs,stage_ends,src,air,face_power = forcing(stage_time)
                stage = lu.solve(mass*u+weight*dt*rhs)
            next_rhs,next_ends,next_src,next_air,next_power = forcing(np.nextafter(stop,start) if stop in events else stop)
            next_u = lu.solve(mass*u+dt*((1-weight)*(rhs-operator@stage)+weight*next_rhs))
            if return_step_heat:
                # Flux, storage and film operators are linear, so their time
                # integrals can be evaluated on the stage-weighted state.
                mean = (1-weight)*stage+weight*next_u
                boundary = (1-weight)*stage_ends+weight*next_ends
                _,faces,left,right,(up,down) = layout.output(mean,boundary,g,flux_order,True)
                step_heat["left"][step] += dt*left
                step_heat["right"][step] += dt*right
                external = np.where(types=="power",boundary,[left[0],-right[-1]])
                external = np.where(types=="robin",layout.end_film*(boundary-faces[[0,-1]]),external)
                step_heat["end_input"][step] += dt*external
                step_heat["source"][step] += dt*volume*((1-weight)*np.asarray(src)+weight*np.asarray(next_src))
                step_heat["ambient"][step] += dt*film*((1-weight)*np.asarray(air)+weight*np.asarray(next_air)-faces[1:-1])
                mean_power = (1-weight)*face_power+weight*next_power
                step_heat["trace_left"][step] += dt*mean_power[0]
                step_heat["trace_right"][step] += dt*mean_power[1]
                step_heat["interface"][step] += dt*mean_power.sum(axis=0)
                # Remove power deposited on the smaller branch itself when
                # inferring the transfer across an ideal (unsplit) overlap.
                contact = np.where(larger_left,left[1:]-mean_power[1],right[:-1]+mean_power[0])
                # For a common trace, the contact also delivers heat stored
                # on the smaller side. Use the actual increment for this step.
                change = next_u[layout.shoulders-1]-u[layout.shoulders-1]
                contact_heat = np.where(larger_left,layout.right_storage,-layout.left_storage)*change
                contact_heat[split] = 0.
                contact[split] = contact_g*(up[split]-down[split])
                step_heat["contact"][step] += dt*contact+contact_heat
            rhs,src,air,face_power = next_rhs,next_src,next_air,next_power
            u,boundary_start,start = next_u,next_ends,stop
        if end in events:
            rhs,boundary_start,src,air,face_power = forcing(end)
            if np.any(massless):
                u[massless] = trace_factor.solve(rhs[massless]-coupling @ u[~massless])
        if end==time[step+1]:
            step += 1
            states[step],ends[step] = u,boundary_start
    output = layout.output(states,ends,g,flux_order,return_traces)
    return output+(step_heat,) if return_step_heat else output
