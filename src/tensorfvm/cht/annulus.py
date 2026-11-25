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


def solve_annular_conduction(radii, conductivity, volumetric_heat_capacity, wall_temperatures,
                             angular_frequency, *, contact_resistance=0., source=0., azimuthal_mode=0,
                             film_coefficients=(None, None), cell_location="midpoint",
                             wall_fluxes=(None, None), distributed_source=False, mean_temperature=None):
    """Return cell phasors and radial face heat rates per unit axial length.

    Boundary, temperature and volumetric source amplitudes multiply
    cos(m*angle)*exp(i*omega*time). Face rates are 2πr times the local
    radial flux amplitude, positive outward (W/m). For m=0 they are the
    circumferential integrals. Radii are cell faces in m, contact resistance
    is areal (m² K/W) on internal faces. For each wall, a None film coefficient
    prescribes surface temperature; a finite nonnegative film coefficient
    (W/m²/K) uses the supplied temperature as ambient. Zero insulates.
    A wall_fluxes entry supplies inward heat flux in W/m² instead. For that
    wall use None in both wall_temperatures and film_coefficients.
    With cell_location="volume_centroid", each temperature is located at
    integral(r² dr)/integral(r dr) and both half-cell resistances use that
    radius. The default "midpoint" uses the arithmetic radial midpoint.
    distributed_source=True integrates a uniform net generation density
    (applied source minus storage and angular loss) across each half-shell
    when relating cell temperatures to face power. For the steady axisymmetric
    case with no temperature anchor, supply the desired volume-weighted mean
    cell temperature. Wall inputs and integrated generation must sum to zero.

    """
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or r[0] <= 0 or np.any(np.diff(r) <= 0):
        raise ValueError("radii must be increasing positive finite faces")
    if not np.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("angular_frequency must be finite and nonnegative")
    if not isinstance(azimuthal_mode, (int, np.integer)) or azimuthal_mode < 0:
        raise ValueError("azimuthal_mode must be a nonnegative integer")
    n = len(r)-1
    k, cv = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in
             (conductivity, volumetric_heat_capacity))
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in (k,cv)):
        raise ValueError("material properties must be finite and positive")
    contact = np.broadcast_to(np.asarray(contact_resistance, dtype=float), (n-1,))
    if len(wall_temperatures) != 2 or len(wall_fluxes) != 2:
        raise ValueError("supply one boundary condition per wall")
    walls = np.zeros(2, dtype=complex)
    density = np.broadcast_to(np.asarray(source, dtype=complex), (n,))
    if not np.isfinite(contact).all() or np.any(contact < 0):
        raise ValueError("contact resistance must be finite and nonnegative")
    if not np.isfinite(density).all():
        raise ValueError("heat generation must be finite")

    width = np.diff(r)
    midpoint = r[:-1]+.5*width
    if cell_location not in ("midpoint", "volume_centroid"):
        raise ValueError("cell_location must be midpoint or volume_centroid")
    # r-weighted centroid minus midpoint, without subtracting powers of radii.
    shift = width*(width/midpoint)/12 if cell_location == "volume_centroid" else 0.
    center = midpoint + shift
    left = np.log1p((.5*width+shift)/r[:-1])/(2*np.pi*k)
    right = np.log1p((.5*width-shift)/center)/(2*np.pi*k)
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
    needs_mean = angular_frequency == 0 and azimuthal_mode == 0 and not conductance[[0,-1]].any()
    if needs_mean and mean_temperature is None:
        raise ValueError("a steady axisymmetric flux-driven annulus needs a temperature reference")
    if mean_temperature is not None:
        if not needs_mean or np.ndim(mean_temperature) != 0 or not np.isfinite(mean_temperature):
            raise ValueError("a finite mean temperature applies only to an unanchored steady axisymmetric mode")
    volume = np.pi*width*(r[1:]+r[:-1])
    lateral = 2*np.pi*k*azimuthal_mode**2*np.log1p(width/r[:-1])
    sink = lateral + 1j*angular_frequency*cv*volume
    # Face power = offset + outward*T_inner - inward_coefficient*T_outer.
    outward = conductance[1:].astype(complex)
    inward_coefficient = conductance[:-1].astype(complex)
    offset = np.zeros(n+1, dtype=complex)
    offset[0] = conductance[0]*walls[0]+inward[0]
    offset[-1] = -conductance[-1]*walls[-1]-inward[-1]
    if distributed_source:
        # Integrating dQ/dr=2πr*g and dT/dr=-Q/(2πkr) gives these
        # temperature-drop coefficients for unit net generation g.
        dl, dr = .5*width+shift, .5*width-shift
        factor_l = _half_shell_source_drop(dl, r[:-1], k)
        factor_r = -_half_shell_source_drop(-dr, r[1:], k)
        offset[:-1] -= conductance[:-1]*factor_l*density
        offset[1:] -= conductance[1:]*factor_r*density
        outward += conductance[1:]*factor_r*(sink/volume)
        inward_coefficient -= conductance[:-1]*factor_l*(sink/volume)
    bands = np.zeros((3,n), dtype=complex)
    bands[0,1:] = -inward_coefficient[1:]
    bands[2,:-1] = -outward[:-1]
    bands[1] = outward+inward_coefficient+sink
    load = density*volume - np.diff(offset)
    if needs_mean:
        generation = density*volume
        net = generation.sum()+inward.sum()
        throughput = np.abs(generation).sum()+np.abs(inward).sum()
        if abs(net) > 1e-12*throughput:
            raise ValueError("incompatible wall inputs and volume generation for steady conduction")
        # Pin the first cell temporarily. Compatibility supplies the omitted
        # first balance equation; shifting the solution then fixes its mean.
        bands[1,0], load[0] = 1., 0.
        if n > 1:
            bands[0,1] = 0.
    field = solve_banded((1,1), bands, load)
    if needs_mean:
        field += mean_temperature - volume @ field / volume.sum()
    rates = offset.copy()
    rates[1:] += outward*field
    rates[:-1] -= inward_coefficient*field
    return field, rates
