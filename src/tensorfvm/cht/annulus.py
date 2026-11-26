"""Cell-centred radial finite volumes for harmonic conduction in a material annulus."""
import numpy as np
from scipy.linalg import solve_banded


def _half_shell_source_drop(distance, face_radius, conductivity):
    """Temperature drop from a face to a cell center per unit generation."""
    x = distance/face_radius
    # (x - log1p(x))/x² loses precision for thin shells.
    ratio = np.empty_like(x)
    thin = abs(x) < 1e-4
    ratio[thin] = .5 - x[thin]/3 + x[thin]**2/4
    ratio[~thin] = (x[~thin]-np.log1p(x[~thin]))/x[~thin]**2
    return distance**2*(.5+ratio)/(2*conductivity)


def _half_shell_resistances(r,k,cell_location):
    width = np.diff(r)
    midpoint = r[:-1]+.5*width
    if cell_location not in ("midpoint", "volume_centroid"):
        raise ValueError("cell_location must be midpoint or volume_centroid")
    # r-weighted centroid minus midpoint, without subtracting powers of radii.
    shift = width*(width/midpoint)/12 if cell_location == "volume_centroid" else 0.
    center = midpoint + shift
    left = np.log1p((.5*width+shift)/r[:-1])/(2*np.pi*k)
    right = np.log1p((.5*width-shift)/center)/(2*np.pi*k)
    return width,shift,left,right


def _validate_axial_wavenumber(axial_wavenumber):
    if np.ndim(axial_wavenumber) != 0 or not np.isrealobj(axial_wavenumber) or not np.isfinite(axial_wavenumber):
        raise ValueError("axial_wavenumber must be a finite real scalar")


def _sheet_modal_conductance(radius, mode, wave, sheet_conductivity, axial_conductivity):
    if axial_conductivity is None:
        axial_conductivity = sheet_conductivity
    elif (np.ndim(axial_conductivity) != 0 or not np.isrealobj(axial_conductivity)
          or not np.isfinite(axial_conductivity) or axial_conductivity < 0):
        raise ValueError("sheet_axial_conductivity must be a finite nonnegative real scalar")
    return sheet_conductivity*(mode/radius)**2 + axial_conductivity*wave**2


def _directional_profile(value, radial, name):
    if value is None:
        return radial
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    profile = np.broadcast_to(np.asarray(value, dtype=float), radial.shape)
    if not np.isfinite(profile).all() or (profile <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return profile


def _modal_conductance(radii, circumferential, axial, volume, mode, axial_wavenumber):
    angular = 2*np.pi*circumferential*mode**2*np.log1p(np.diff(radii)/radii[:-1])
    return angular + axial*volume*axial_wavenumber**2


def solve_annular_conduction(radii, conductivity, volumetric_heat_capacity, wall_temperatures,
                             angular_frequency, *, contact_resistance=0., source=0., azimuthal_mode=0,
                             circumferential_conductivity=None, axial_conductivity=None,
                             axial_wavenumber=0., film_coefficients=(None, None), cell_location="midpoint",
                             wall_fluxes=(None, None), distributed_source=False, mean_temperature=None,
                             return_contact_temperatures=False, return_wall_temperatures=False):
    """Return cell phasors and radial face heat rates per unit axial length.

    conductivity is radial. Optional positive circumferential_conductivity and
    axial_conductivity (W/m/K) accept scalar or per-cell values, each defaulting
    to conductivity. The material axes follow cylindrical coordinates.

    Boundary, temperature and volumetric source amplitudes multiply
    cos(m*angle)*exp(i*(axial_wavenumber*z + omega*time)), with real axial
    wavenumber in rad/m. Face rates are 2πr times the local
    radial flux amplitude, positive outward (W/m). For m=0 they are the
    circumferential integrals. Volumetric heat capacity may vanish in any cell;
    it then conducts without storing heat. Radii are cell faces in m, contact resistance
    is areal (m² K/W) on internal faces. For each wall, a None film coefficient
    prescribes surface temperature; a finite nonnegative film coefficient
    (W/m²/K) uses the supplied temperature as ambient. Zero insulates.
    A wall_fluxes entry supplies inward heat flux in W/m² instead. For that
    wall use None in both wall_temperatures and film_coefficients.
    With cell_location="volume_centroid", each temperature is located at
    integral(r² dr)/integral(r dr) and both half-cell resistances use that
    radius. The default "midpoint" uses the arithmetic radial midpoint.
    distributed_source=True integrates a uniform net generation density
    (applied source minus storage, angular loss and axial loss) across each half-shell
    when relating cell temperatures to face power. For an unanchored axisymmetric
    problem with zero axial wavenumber and either zero frequency or all capacities zero, supply the volume-weighted mean
    cell temperature. Wall inputs and integrated generation must sum to zero.
    With return_wall_temperatures=True, append the material surface temperatures
    (inner wall, outer wall) in K before any contact-temperature output.
    With return_contact_temperatures=True, append an array of shape (2, n-1)
    in K. Rows give the inner- and outer-material surface temperatures at
    each internal face in order of increasing radius.

    """
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or r[0] <= 0 or np.any(np.diff(r) <= 0):
        raise ValueError("radii must be increasing positive finite faces")
    if not np.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("angular_frequency must be finite and nonnegative")
    if not isinstance(azimuthal_mode, (int, np.integer)) or azimuthal_mode < 0:
        raise ValueError("azimuthal_mode must be a nonnegative integer")
    _validate_axial_wavenumber(axial_wavenumber)
    n = len(r)-1
    k, cv = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in
             (conductivity, volumetric_heat_capacity))
    if not np.isfinite(k).all() or np.any(k <= 0) or not np.isfinite(cv).all() or np.any(cv < 0):
        raise ValueError("conductivity must be positive and volumetric capacity nonnegative, both finite")
    kt = _directional_profile(circumferential_conductivity, k, "circumferential_conductivity")
    kz = _directional_profile(axial_conductivity, k, "axial_conductivity")
    contact = np.broadcast_to(np.asarray(contact_resistance, dtype=float), (n-1,))
    if len(wall_temperatures) != 2 or len(wall_fluxes) != 2:
        raise ValueError("supply one boundary condition per wall")
    walls = np.zeros(2, dtype=complex)
    density = np.broadcast_to(np.asarray(source, dtype=complex), (n,))
    if not np.isfinite(contact).all() or np.any(contact < 0):
        raise ValueError("contact resistance must be finite and nonnegative")
    if not np.isfinite(density).all():
        raise ValueError("heat generation must be finite")

    width,shift,left,right = _half_shell_resistances(r,k,cell_location)
    conductance = 1/np.r_[left[0], right[:-1]+contact/(2*np.pi*r[1:-1])+left[1:], right[-1]]
    if len(film_coefficients) != 2:
        raise ValueError("supply one film coefficient per wall")
    inward = np.zeros(2, dtype=complex)
    for index, wall, h, flux in zip((0,-1), wall_temperatures, film_coefficients, wall_fluxes):
        if flux is not None:
            if wall is not None or h is not None or np.ndim(flux) != 0 or not np.isfinite(flux):
                raise ValueError("a wall flux must be finite and requires None for temperature and film")
            conductance[index] = 0.
            inward[index] = 2*np.pi*r[index]*flux
            continue
        if wall is None or np.ndim(wall) != 0 or not np.isfinite(wall):
            raise ValueError("a temperature boundary requires a finite wall temperature")
        walls[index] = wall
        if h is None:
            continue
        if not np.isfinite(h) or h < 0:
            raise ValueError("film coefficients must be finite and nonnegative")
        conductance[index] = (0. if h == 0 else
                              1/(1/conductance[index] + 1/(2*np.pi*r[index]*h)))
    needs_mean = (angular_frequency == 0 or not cv.any()) and azimuthal_mode == 0 and axial_wavenumber == 0 and not conductance[[0,-1]].any()
    if needs_mean and mean_temperature is None:
        raise ValueError("a steady axisymmetric flux-driven annulus needs a temperature reference")
    if mean_temperature is not None:
        if not needs_mean or np.ndim(mean_temperature) != 0 or not np.isfinite(mean_temperature):
            raise ValueError("a finite mean temperature applies only to an unanchored steady axisymmetric mode")
    volume = np.pi*width*(r[1:]+r[:-1])
    lateral = _modal_conductance(r, kt, kz, volume, azimuthal_mode, axial_wavenumber)
    sink = lateral + 1j*angular_frequency*cv*volume
    anchored = conductance[[0, -1]] > 0
    if distributed_source:
        # Integrating dQ/dr=2πr*g and dT/dr=-Q/(2πkr) gives these
        # temperature-drop coefficients for unit net generation g.
        dl, dr = .5*width+shift, .5*width-shift
        factor_l = _half_shell_source_drop(dl, r[:-1], k)
        factor_r = -_half_shell_source_drop(-dr, r[1:], k)

    def result(field, rates):
        values = (field, rates)
        if not (return_contact_temperatures or return_wall_temperatures):
            return values
        # Recover each cell's two surface values using its own material law.
        at_left = field + rates[:-1]*left
        at_right = field - rates[1:]*right
        if distributed_source:
            net = density - (sink/volume)*field
            at_left += factor_l*net
            at_right -= factor_r*net
        if return_wall_temperatures:
            surfaces = np.array([at_left[0], at_right[-1]])
            for side in (0, 1):
                if wall_fluxes[side] is None and film_coefficients[side] is None:
                    surfaces[side] = walls[side]
            values += (surfaces,)
        if return_contact_temperatures:
            values += (np.stack((at_right[:-1], at_left[1:])),)
        return values

    if not sink.any() and anchored.all():
        # Balance fixes differences between face rates. The total drop
        # between the two ambient temperatures fixes their common offset.
        resistance = 1/conductance
        generation = density*volume
        source_drop = np.zeros(n+1, dtype=complex)
        if distributed_source:
            source_drop[:-1] += factor_l*density
            source_drop[1:] += factor_r*density
        from_inner = np.cumsum(resistance[:-1])
        to_outer = np.cumsum(resistance[:0:-1])[::-1]
        total_resistance = resistance.sum()
        # Each source sends heat to both walls in inverse proportion to the
        # two path resistances. Sum its signed contribution at each face
        # directly to retain leakage far below the total generated power.
        toward_outer = generation * (from_inner / total_resistance)
        toward_inner = generation * (to_outer / total_resistance)
        through = (walls[0] - walls[-1] - source_drop.sum()) / total_resistance
        rates = (through + np.r_[0., np.cumsum(toward_outer)]
                 - np.r_[np.cumsum(toward_inner[::-1])[::-1], 0.])
        drops = rates*resistance + source_drop
        field = np.where(from_inner <= to_outer, walls[0] - np.cumsum(drops[:-1]),
                         walls[-1] + np.cumsum(drops[:0:-1])[::-1])
        return result(field, rates)
    if needs_mean or (not sink.any() and anchored.sum() == 1):
        # A flux boundary fixes one face rate. March the energy balance
        # from that wall, including generation in each cell.
        generation = density*volume
        if anchored[0]:
            rates = -inward[-1] - np.r_[np.cumsum(generation[::-1])[::-1], 0.]
        else:
            rates = inward[0] + np.r_[0., np.cumsum(generation)]
        if needs_mean:
            net = generation.sum()+inward.sum()
            throughput = np.abs(generation).sum()+np.abs(inward).sum()
            if abs(net) > 1e-12*throughput:
                raise ValueError("incompatible wall inputs and volume generation for steady conduction")
            rates[-1] = -inward[-1]
        drops = rates[1:-1]/conductance[1:-1]
        if distributed_source:
            drops += factor_r[:-1]*density[:-1] + factor_l[1:]*density[1:]
        if needs_mean:
            field = np.r_[0j, -np.cumsum(drops)]
            field += mean_temperature - np.average(field, weights=volume)
        elif anchored[0]:
            first_drop = rates[0]/conductance[0]
            if distributed_source:
                first_drop += factor_l[0]*density[0]
            field = walls[0] - np.cumsum(np.r_[first_drop, drops])
        else:
            last_drop = rates[-1]/conductance[-1]
            if distributed_source:
                last_drop += factor_r[-1]*density[-1]
            field = walls[-1] + np.cumsum(np.r_[last_drop, drops[::-1]])[::-1]
        return result(field, rates)
    # Face power = offset + outward*T_inner - inward_coefficient*T_outer.
    outward = conductance[1:].astype(complex)
    inward_coefficient = conductance[:-1].astype(complex)
    offset = np.zeros(n+1, dtype=complex)
    offset[0] = conductance[0]*walls[0]+inward[0]
    offset[-1] = -conductance[-1]*walls[-1]-inward[-1]
    if distributed_source:
        offset[:-1] -= conductance[:-1]*factor_l*density
        offset[1:] -= conductance[1:]*factor_r*density
        outward += conductance[1:]*factor_r*(sink/volume)
        inward_coefficient -= conductance[:-1]*factor_l*(sink/volume)
    bands = np.zeros((3,n), dtype=complex)
    bands[0,1:] = -inward_coefficient[1:]
    bands[2,:-1] = -outward[:-1]
    bands[1] = outward+inward_coefficient+sink
    load = density*volume - np.diff(offset)
    field = solve_banded((1,1), bands, load)
    rates = offset.copy()
    rates[1:] += outward*field
    rates[:-1] -= inward_coefficient*field
    return result(field, rates)


def solve_annular_sheet(radii, conductivity, volumetric_heat_capacity, wall_temperatures, angular_frequency,
                        sheet_face, *, resistance=0., inner_resistance_fraction=.5, sheet_capacity=0.,
                        sheet_conductivity=0., sheet_axial_conductivity=None, sheet_film=0., ambient=0., sheet_source=0.,
                        circumferential_conductivity=None, axial_conductivity=None,
                        source=0., azimuthal_mode=0, axial_wavenumber=0., wall_fluxes=(None,None), film_coefficients=(None,None),
                        cell_location="midpoint", distributed_source=False, mean_temperature=None,
                        return_contact_temperatures=False, return_wall_temperatures=False):
    """Harmonic conduction with one storing sheet at internal face index sheet_face.

    Bulk circumferential_conductivity and axial_conductivity follow
    solve_annular_conduction, independently of the sheet properties.

    Return cell temperatures, radial rates on the inner and outer domains
    (each including its two boundary faces, outward positive, W/m), and sheet
    temperature. With return_wall_temperatures, append the two exterior material
    surface temperatures in radial order. With return_contact_temperatures, append the two material-side
    temperature phasors (inner, outer) at the contact radius. The sheet has areal heat capacity (J/m²/K), circumferential
    conductivity times thickness (W/K), reservoir film (W/m²/K) and applied
    generation (W/m²). sheet_axial_conductivity (W/K) defaults to sheet_conductivity.
    Sheet properties are uniform in angle; loads are the
    selected complex modal amplitudes with axial wavenumber in rad/m. The mean datum for a floating steady
    zero mode is the volume-weighted temperature of the material cells.
    """
    edges = np.asarray(radii,dtype=float)
    if edges.ndim!=1 or not isinstance(sheet_face,(int,np.integer)) or not 0<sheet_face<len(edges)-1:
        raise ValueError("sheet_face must index an internal radius")
    if not np.isfinite([resistance,inner_resistance_fraction,sheet_capacity,sheet_conductivity,sheet_film]).all():
        raise ValueError("sheet properties must be finite")
    if min(resistance,sheet_capacity,sheet_conductivity,sheet_film)<0 or not 0<=inner_resistance_fraction<=1:
        raise ValueError("sheet properties must be nonnegative and resistance fraction in [0,1]")
    if not np.isfinite([ambient,sheet_source]).all():
        raise ValueError("sheet load and ambient must be finite")
    n = edges.size-1
    k,c,p = (np.broadcast_to(np.asarray(x),(n,)) for x in (conductivity,volumetric_heat_capacity,source))
    kt = _directional_profile(circumferential_conductivity, k, "circumferential_conductivity")
    kz = _directional_profile(axial_conductivity, k, "axial_conductivity")
    if any(len(pair)!=2 for pair in (wall_temperatures,film_coefficients,wall_fluxes)):
        raise ValueError("supply one boundary condition per wall")
    j = sheet_face
    radius = edges[j]
    circum = 2*np.pi*radius
    _validate_axial_wavenumber(axial_wavenumber)
    sheet_loss = _sheet_modal_conductance(radius, azimuthal_mode, axial_wavenumber,
                                         sheet_conductivity, sheet_axial_conductivity)
    impedance = sheet_film + 1j*angular_frequency*sheet_capacity + sheet_loss
    load = sheet_source+sheet_film*ambient
    options = dict(axial_wavenumber=axial_wavenumber,azimuthal_mode=azimuthal_mode,cell_location=cell_location,distributed_source=distributed_source)
    ri,ro = resistance*inner_resistance_fraction,resistance*(1-inner_resistance_fraction)

    def result(field, inner_rates, outer_rates, sheet):
        values = (field, inner_rates, outer_rates, sheet)
        if return_wall_temperatures:
            values += (base_walls + unit_walls*sheet,)
        if return_contact_temperatures:
            traces = sheet + np.array([ri*inner_rates[-1], -ro*outer_rates[0]])/circum
            return (*values, traces)
        return values

    def response(unit):
        # Unit sheet-temperature response uses homogeneous exterior and body loads.
        w = tuple(0. if unit and v is not None else v for v in wall_temperatures)
        q = tuple(0. if unit and v is not None else v for v in wall_fluxes)
        inner = solve_annular_conduction(edges[:j+1],k[:j],c[:j],(w[0],float(unit)),angular_frequency,
                wall_fluxes=(q[0],None),film_coefficients=(film_coefficients[0],1/ri if ri else None),
                circumferential_conductivity=kt[:j], axial_conductivity=kz[:j],
                source=0. if unit else p[:j],return_wall_temperatures=return_wall_temperatures, **options)
        outer = solve_annular_conduction(edges[j:],k[j:],c[j:],(float(unit),w[1]),angular_frequency,
                wall_fluxes=(None,q[1]),film_coefficients=(1/ro if ro else None,film_coefficients[1]),
                circumferential_conductivity=kt[j:], axial_conductivity=kz[j:],
                source=0. if unit else p[j:],return_wall_temperatures=return_wall_temperatures, **options)
        surfaces = np.array([inner[2][0], outer[2][-1]]) if return_wall_temperatures else None
        return np.r_[inner[0], outer[0]], inner[1], outer[1], surfaces

    base,bi,bo,base_walls = response(False)
    unit,ui,uo,unit_walls = response(True)
    closed = ((not angular_frequency or (not c.any() and not sheet_capacity))
              and not azimuthal_mode and not axial_wavenumber and not sheet_film and
              all(q is not None or h==0 for q,h in zip(wall_fluxes,film_coefficients)))
    volume = np.pi*np.diff(edges)*(edges[1:]+edges[:-1])
    if closed:
        if mean_temperature is None or np.ndim(mean_temperature)!=0 or not np.isfinite(mean_temperature):
            raise ValueError("an unanchored sheet requires a finite mean temperature")
        inputs = np.r_[p*volume,circum*sheet_source,bi[0],-bo[-1]]
        if abs(inputs.sum())>1e-12*np.abs(inputs).sum():
            raise ValueError("incompatible wall inputs, sheet and volume generation")
        sheet = mean_temperature-np.average(base,weights=volume)
        return result(base+sheet, bi, bo, sheet)
    if mean_temperature is not None:
        raise ValueError("mean_temperature applies only to a floating steady zero mode")
    # Integrate the two bulk balances to evaluate their net response.
    # This preserves the small storage admittance near zero frequency under
    # insulated end conditions, where differences of radial rates cancel.
    sink = 1j*angular_frequency*c*volume + _modal_conductance(edges, kt, kz, volume, azimuthal_mode, axial_wavenumber)
    forcing = circum*load+bi[0]-bo[-1]+np.sum(volume*p-sink*base)
    admittance = circum*impedance+uo[-1]-ui[0]+sink@unit
    sheet = forcing/admittance
    field = base+unit*sheet
    inner_rates,outer_rates = bi+ui*sheet,bo+uo*sheet
    # Under a prescribed wall power the boundary rate is known exactly.
    # Integrate the cell balances from that end to avoid multiplying tiny
    # roundoff in unit-response fluxes by a large low-frequency temperature.
    net = volume*p-sink*field
    if wall_fluxes[0] is not None or film_coefficients[0]==0:
        inner_rates[1:] = bi[0]+np.cumsum(net[:j])
    if wall_fluxes[1] is not None or film_coefficients[1]==0:
        outer_rates[:-1] = bo[-1]-np.cumsum(net[j:][::-1])[::-1]
    return result(field, inner_rates, outer_rates, sheet)


def solve_transient_annular_sheet(radii, conductivity, volumetric_heat_capacity, wall_temperatures,
                                  times, initial_temperature, sheet_face, *, initial_sheet_temperature=None,
                                  sheet_capacity, resistance=0., inner_resistance_fraction=.5,
                                  sheet_conductivity=0., sheet_axial_conductivity=None, azimuthal_mode=0, axial_wavenumber=0., sheet_film=0., ambient=0.,
                                  circumferential_conductivity=None, axial_conductivity=None,
                                  sheet_source_intervals=0., theta=.5, wall_fluxes=(None,None), source_intervals=0.,
                                  film_coefficients=(None,None), return_wall_temperatures=False,
                                  return_contact_temperatures=False, method="theta", return_interval_temperatures=False,
                                  source_times=None, wall_times=None):
    """Advance a real circumferential and axial Fourier mode with a storing sheet.

    Bulk circumferential_conductivity and axial_conductivity follow
    solve_annular_conduction, independently of the sheet properties.

    axial_wavenumber is in rad/m. The sheet conductivity times thickness
    (W/K) applies circumferentially. sheet_axial_conductivity (W/K) defaults
    to the same value.

    Walls and reservoir ambient accept constants, nodal samples or one-sided
    (interval,2) samples. wall_times optionally supplies their time grid
    (default: output times). Wall traces at a switch use the right limit.
    A None wall temperature instead takes a prescribed inward wall_fluxes
    entry (W/m²), either constant or sampled at the output times. A finite
    nonnegative film_coefficients entry (W/m²/K) treats the corresponding
    wall temperature as a fluid temperature; zero insulates that wall.
    Sheet generation in W/m² is constant or has shape (interval,2), giving its
    one-sided start and end values. Optional source_times defines the sheet and
    material source intervals independently of output times, with the same
    first and last instant. Integrate on the union of both grids; return
    interval heats and weighted means on the requested output grid.
    Material generation source_intervals (W/m³) accepts a constant, a static
    per-cell profile, or start/end profiles shaped (interval,2,cell). A final
    dimension of one broadcasts a time-varying source across all cells.
    With zero sheet capacity, its algebraic temperature follows the current
    source and adjacent cells. At a switch the sheet trace is right-continuous
    (the final output uses the last interval end). Positive capacity requires
    an initial sheet temperature. Cell temperatures refer to radial volume centroids. Returns cell and sheet
    temperature histories (K), interval radial heats on each side (J/m, both
    directed toward increasing radius, with all faces including the walls),
    and interval sheet-to-reservoir heat (J/m). Rates use the 2πr modal
    normalization of solve_annular_conduction. theta=.5 gives Crank–Nicolson.
    method="sdirk2" selects a two-stage L-stable method with linearly
    interpolated loads and consistent stage quadrature for all heats.
    return_wall_temperatures appends instantaneous material surface temperatures
    at the bore and exterior (shape time × 2), using the radial half-cell resistances.
    return_contact_temperatures appends the inner and outer material traces at
    the sheet (shape time × 2), after optional wall temperatures. Massless
    contact traces use the same one-sided convention as sheet temperatures.
    return_interval_temperatures appends a dict with cell, sheet, contact and
    wall temperatures in K, averaged with the actual interval heat quadrature.
    The wall array has one row per output interval and columns (bore, exterior).
    """
    r = np.asarray(radii,dtype=float)
    t = np.asarray(times,dtype=float)
    if r.ndim!=1 or len(r)<3 or not np.isfinite(r).all() or r[0]<=0 or np.any(np.diff(r)<=0):
        raise ValueError("radii must be increasing positive finite faces")
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError("times must be a finite strictly increasing vector")
    j = sheet_face
    if not isinstance(j,(int,np.integer)) or not 0<j<len(r)-1:
        raise ValueError("sheet_face must index an internal radius")
    _validate_axial_wavenumber(axial_wavenumber)
    if not isinstance(azimuthal_mode,(int,np.integer)) or azimuthal_mode<0:
        raise ValueError("azimuthal_mode must be a nonnegative integer")
    if not np.isfinite([sheet_capacity,resistance,inner_resistance_fraction,sheet_conductivity,sheet_film,theta]).all():
        raise ValueError("sheet properties and theta must be finite")
    if min(sheet_capacity,resistance,sheet_conductivity,sheet_film)<0 or not 0<=inner_resistance_fraction<=1 or not .5<=theta<=1:
        raise ValueError("require nonnegative sheet properties, fraction in [0,1] and theta in [.5,1]")
    if method not in ("theta","sdirk2") or (method=="sdirk2" and theta!=.5):
        raise ValueError("use theta or sdirk2; sdirk2 does not accept a theta override")
    n = len(r)-1
    k,c,u = (np.broadcast_to(np.asarray(v,dtype=float),(n,)) for v in (conductivity,volumetric_heat_capacity,initial_temperature))
    kt = _directional_profile(circumferential_conductivity, k, "circumferential_conductivity")
    kz = _directional_profile(axial_conductivity, k, "axial_conductivity")
    if np.any(k<=0) or np.any(c<=0) or not all(np.isfinite(v).all() for v in (k,c,u)):
        raise ValueError("material properties must be positive and initial temperatures finite")
    if any(len(pair)!=2 for pair in (wall_temperatures,wall_fluxes,film_coefficients)):
        raise ValueError("supply two wall conditions")
    if initial_sheet_temperature is None:
        if sheet_capacity>0:
            raise ValueError("a storing sheet requires a finite initial sheet temperature")
        initial_sheet_temperature = 0.
    if np.ndim(initial_sheet_temperature)!=0 or not np.isfinite(initial_sheet_temperature):
        raise ValueError("initial sheet temperature must be finite")
    for temp,flux,h in zip(wall_temperatures,wall_fluxes,film_coefficients):
        if h is not None and (np.ndim(h)!=0 or not np.isfinite(h) or h<0 or flux is not None):
            raise ValueError("a wall film needs a finite nonnegative coefficient and an ambient temperature")
        if (temp is None)==(flux is None):
            raise ValueError("supply either temperature or inward heat flux for each wall")
    output_times = t
    def load_grid(values,name):
        grid = t if values is None else np.asarray(values,dtype=float)
        if (grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all()
                or np.any(np.diff(grid)<=0) or grid[0]!=t[0] or grid[-1]!=t[-1]):
            raise ValueError(f"{name} must increase strictly and span the requested output times")
        return grid

    source_grid = load_grid(source_times,"source_times")
    wall_grid = load_grid(wall_times,"wall_times")

    def wall_history(value):
        v = np.asarray(value,dtype=float)
        if v.shape==wall_grid.shape:
            v = np.column_stack((v[:-1],v[1:]))
        if v.shape not in ((),(len(wall_grid)-1,2)) or not np.isfinite(v).all():
            raise ValueError("wall histories must be finite and scalar, nodal or shaped (interval,2)")
        return np.broadcast_to(v,(len(wall_grid)-1,2))
    west,east = (wall_history(v if v is not None else q) for v,q in zip(wall_temperatures,wall_fluxes))
    fluid = wall_history(ambient)
    drive = np.asarray(sheet_source_intervals,dtype=float)
    if drive.shape not in ((),(len(source_grid)-1,2)):
        raise ValueError("sheet source must be constant or shaped (interval,2)")
    drive = np.broadcast_to(drive,(len(source_grid)-1,2))
    if not all(np.isfinite(v).all() for v in (west,east,fluid,drive)):
        raise ValueError("wall, ambient and source histories must be finite")

    body = np.asarray(source_intervals,dtype=float)
    if body.shape not in ((),(n,),(len(source_grid)-1,2,1),(len(source_grid)-1,2,n)) or not np.isfinite(body).all():
        raise ValueError("material source must be finite and scalar, per cell or shaped (interval,2,cell)")
    body = np.broadcast_to(body,(len(source_grid)-1,2,n))
    t = np.union1d(np.union1d(output_times,source_grid),wall_grid)
    def subdivide(grid,history):
        if np.array_equal(t,grid):
            return history
        # The subinterval starts on the right of a possible discontinuity.
        idx = np.searchsorted(grid,t[:-1],side="right")-1
        f = (np.column_stack((t[:-1],t[1:]))-grid[idx,None])/np.diff(grid)[idx,None]
        if history.ndim==3:
            f = f[:,:,None]
        return (1-f)*history[idx,:1]+f*history[idx,1:]
    drive,body = subdivide(source_grid,drive),subdivide(source_grid,body)
    west,east,fluid = (subdivide(wall_grid,v) for v in (west,east,fluid))
    width,_,left,right = _half_shell_resistances(r,k,"volume_centroid")
    circum = 2*np.pi*r[j]
    links = np.r_[left[0],right[:-1]+left[1:],right[-1]]
    links = np.r_[links[:j],right[j-1]+inner_resistance_fraction*resistance/circum,
                  left[j]+(1-inner_resistance_fraction)*resistance/circum,links[j+1:]]
    g = 1/links
    # Write each boundary power as a known load plus its cell coefficient.
    for side,h in zip((0,-1),film_coefficients):
        if h is not None:
            g[side] = 0. if h==0 else 1/(links[side]+1/(2*np.pi*r[side]*h))
    west_factor,east_factor = g[0],-g[-1]
    if wall_fluxes[0] is not None:
        g[0],west_factor = 0.,2*np.pi*r[0]
    if wall_fluxes[1] is not None:
        g[-1],east_factor = 0.,-2*np.pi*r[-1]
    volume = np.pi*width*(r[:-1]+r[1:])
    capacity = np.insert(c*volume,j,circum*sheet_capacity)
    sheet_loss = _sheet_modal_conductance(r[j], azimuthal_mode, axial_wavenumber,
                                         sheet_conductivity, sheet_axial_conductivity)
    sink = np.insert(_modal_conductance(r, kt, kz, volume, azimuthal_mode, axial_wavenumber),j,
                     circum*(sheet_film + sheet_loss))
    diagonal = g[:-1]+g[1:]+sink
    state = np.empty((len(t),n+1))
    state[0] = np.insert(u,j,initial_sheet_temperature)
    heat = np.empty((len(t)-1,n+2))
    if return_interval_temperatures:
        interval_state = np.empty((len(t)-1,n+1))
    reservoir = np.empty(len(t)-1)
    bands = np.zeros((3,n+1))
    sdirk = method=="sdirk2"
    weight = 1-1/np.sqrt(2) if sdirk else theta
    bands[0,1:] = bands[2,:-1] = -weight*g[1:-1]

    def loads(i,fraction):
        w,e,a = ((1-fraction)*v[i,0]+fraction*v[i,1] for v in (west,east,fluid))
        wall_input = np.array([west_factor*w,east_factor*e])
        power = volume*((1-fraction)*body[i,0]+fraction*body[i,1])
        supply = (1-fraction)*drive[i,0]+fraction*drive[i,1]
        applied = np.insert(power,j,circum*(supply+sheet_film*a))
        applied[0] += wall_input[0]
        applied[-1] -= wall_input[1]
        return applied,wall_input,a

    def internal_power(temperature):
        return np.diff(g*np.diff(np.r_[0.,temperature,0.]))-sink*temperature

    for i,dt in enumerate(np.diff(t)):
        old = state[i]
        if sheet_capacity==0:
            # A source jump changes a massless trace before the next step.
            # Use its right limit for quadrature while retaining cell storage.
            old[j] = (g[j]*old[j-1]+g[j+1]*old[j+1]+circum*(drive[i,0]+sheet_film*fluid[i,0]))/diagonal[j]
        # Increment form avoids subtracting two large storage terms.
        bands[1] = capacity/dt+weight*diagonal
        # For affine forcing SDIRK's weighted load equals its midpoint value.
        applied,wall_input,avg_a = loads(i,.5 if sdirk else theta)
        rhs = applied+internal_power(old)
        if sdirk:
            stage_load,_,_ = loads(i,weight)
            first_delta = solve_banded((1,1),bands,weight*(stage_load+internal_power(old)))
            rhs += (1-weight)*internal_power(first_delta)
        delta = solve_banded((1,1),bands,rhs)
        state[i+1] = old+delta
        stage = old+weight*delta
        if sdirk:
            stage += (1-weight)*first_delta
        if return_interval_temperatures:
            interval_state[i] = stage
        heat[i] = -dt*g*np.diff(np.r_[0.,stage,0.])
        heat[i,[0,-1]] += dt*wall_input
        reservoir[i] = dt*circum*sheet_film*(stage[j]-avg_a)
    def wall_surfaces(states, west_input, east_input):
        surfaces = np.empty((len(states),2))
        for column,index,data in ((0,0,west_input),(1,-1,east_input)):
            if wall_fluxes[column] is not None:
                inward = 2*np.pi*r[index]*data
            elif film_coefficients[column] is None:
                surfaces[:,column] = data
                continue
            else:
                inward = g[index]*(data-states[:,index])
            surfaces[:,column] = states[:,index]+links[index]*inward
        return surfaces

    if return_interval_temperatures:
        f = .5 if sdirk else theta
        interval_wall = wall_surfaces(interval_state, (1-f)*west[:,0]+f*west[:,1],
                                      (1-f)*east[:,0]+f*east[:,1])
    keep = np.searchsorted(t,output_times)
    if len(t)!=len(output_times):
        heat = np.add.reduceat(heat,keep[:-1],axis=0)
        reservoir = np.add.reduceat(reservoir,keep[:-1])
        if return_interval_temperatures:
            def output_means(values):
                integral = np.add.reduceat(values*np.diff(t)[:,None],keep[:-1],axis=0)
                return integral/np.diff(output_times)[:,None]
            interval_state = output_means(interval_state)
            interval_wall = output_means(interval_wall)
        state = state[keep]
        t = output_times
    result = np.delete(state,j,axis=1),state[:,j],heat[:,:j+1],heat[:,j+1:],reservoir
    if return_wall_temperatures:
        wall_samples = (np.r_[v[:,0],v[-1,1]][keep] for v in (west,east))
        result += (wall_surfaces(state, *wall_samples),)
    if return_contact_temperatures:
        inner_flux = g[j]*(state[:,j-1]-state[:,j])/circum
        outer_flux = g[j+1]*(state[:,j]-state[:,j+1])/circum
        result += (np.column_stack((state[:,j]+inner_resistance_fraction*resistance*inner_flux,
                                    state[:,j]-(1-inner_resistance_fraction)*resistance*outer_flux)),)
    if return_interval_temperatures:
        dt = np.diff(t)
        inner_flux = heat[:,j]/(circum*dt)
        outer_flux = heat[:,j+1]/(circum*dt)
        mean_sheet = interval_state[:,j]
        mean_contact = np.column_stack((mean_sheet+inner_resistance_fraction*resistance*inner_flux,
                                        mean_sheet-(1-inner_resistance_fraction)*resistance*outer_flux))
        result += (dict(cell=np.delete(interval_state,j,axis=1),sheet=mean_sheet,contact=mean_contact,wall=interval_wall),)
    return result
