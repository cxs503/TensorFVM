"""Finite-volume response of two storing interfaces in a conducting annulus."""
import numpy as np

from .annulus import solve_annular_conduction


def solve_storing_rings(radii, conductivity, sheet_faces, wall_temperatures, omega, *,
                        contact_resistances, sheet_capacities, sheet_generation=(0., 0.), source=0.,
                        sheet_film_coefficients=(0., 0.), sheet_ambient_temperatures=(0., 0.),
                        azimuthal_mode=0, volumetric_heat_capacity=None, film_coefficients=(None, None),
                        inner_resistance_fractions=(.5, .5), sheet_conductivities=(0., 0.),
                        return_wall_rates=False, return_bulk_storage=False, return_angular_conduction=False,
                        return_wall_temperatures=False, return_contact_temperatures=False):
    """Return two sheet temperatures and their incoming and outgoing radial fluxes.

    The cell faces include both sheets. Conductivity is scalar or per cell.
    Wall temperatures and areal heat generation (W/m²) are complex amplitudes of
    cos(mθ) exp(iωt). Omitting volumetric_heat_capacity gives quasi-steady bulk conduction.
    source is scalar or per-cell volumetric heat generation (W/m³), in the same mode.
    At either physical wall a positive film coefficient (W/m²/K) makes the supplied
    temperature an ambient phasor. None prescribes the surface temperature.
    A nonnegative scalar or per-cell capacity (J/(m³ K)) enables harmonic bulk storage.
    Zero-capacity cells conduct without storing heat, even beside cells with positive capacity.
    inner_resistance_fractions specifies the fraction of each contact resistance
    (m² K/W) on the smaller-radius side of its storing sheet (capacity in J/(m² K)).
    sheet_conductivities are circumferential conductivities times thickness (W/K),
    uniform around each sheet. They contribute Ks*(m/r)² to the modal admittance.
    Each sheet also receives h*(ambient - sheet temperature) in W/m², with h
    from sheet_film_coefficients and ambient from sheet_ambient_temperatures.
    A zero sheet film coefficient disables this exchange.
    Each output has one entry per sheet ordered from the bore outwards. Fluxes
    (W/m²) are positive towards increasing radius on both sides.
    With return_wall_rates, append the bore and exterior radial heat rates per unit
    axial length (W/m), both positive towards increasing radius. For angular modes
    above zero these are 2πr times the flux amplitudes.
    With return_bulk_storage, append three modal storage rates in W/m, ordered
    by radius, after any wall rates. Each equals iω times the radial integral
    of 2πr times capacity and temperature amplitude. Quasi-steady layers give zero.
    With return_angular_conduction, append three circumferential divergence
    amplitudes, integral(2π k m² T/r dr) per layer in W/m. They vanish for
    mode zero and enter each modal balance as radial input plus volume generation minus output and storage.
    With return_wall_temperatures, append the physical bore and exterior surface
    temperature phasors (K). With return_contact_temperatures, append a final
    (2, 2) array of material temperatures (K), one row per sheet, with the
    smaller-radius and larger-radius material in columns zero and one.
    """
    r = np.asarray(radii, dtype=float)
    faces = np.asarray(sheet_faces)
    if r.ndim != 1 or faces.shape != (2,) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("two integer sheet faces and a radial face array required")
    if not 0 < faces[0] < faces[1] < len(r)-1:
        raise ValueError("sheet faces must be increasing internal indices")
    if not np.isscalar(omega) or not np.isrealobj(omega) or not np.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative")
    k = np.broadcast_to(np.asarray(conductivity, dtype=float), (len(r)-1,))
    bulk_frequency = 0. if volumetric_heat_capacity is None else omega
    cv = np.broadcast_to(1. if volumetric_heat_capacity is None else volumetric_heat_capacity, k.shape)
    volume_source = np.broadcast_to(np.asarray(source, dtype=complex), k.shape)
    if not np.isfinite(volume_source).all():
        raise ValueError("volumetric heat generation must be finite")
    walls = np.asarray(wall_temperatures, dtype=complex)
    generation = np.asarray(sheet_generation, dtype=complex)
    sheet_h = np.asarray(sheet_film_coefficients)
    if (sheet_h.shape != (2,) or not np.isrealobj(sheet_h) or
            not np.isfinite(sheet_h).all() or np.any(sheet_h < 0)):
        raise ValueError("two finite nonnegative sheet film coefficients required")
    ambient = np.asarray(sheet_ambient_temperatures, dtype=complex)
    if ambient.shape != (2,) or not np.isfinite(ambient).all():
        raise ValueError("two finite sheet ambient phasors required")
    capacity = np.asarray(sheet_capacities, dtype=float)
    resistance = np.asarray(contact_resistances, dtype=float)
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (walls, generation, capacity, resistance)):
        raise ValueError("supply two finite values per boundary or sheet property")
    if np.any(capacity < 0) or np.any(resistance < 0):
        raise ValueError("sheet capacities and contact resistances must be nonnegative")

    sheet_k = np.asarray(sheet_conductivities, dtype=float)
    if sheet_k.shape != (2,) or not np.isfinite(sheet_k).all() or np.any(sheet_k < 0):
        raise ValueError("two finite nonnegative sheet conductivities required")
    fractions = np.asarray(inner_resistance_fractions, dtype=float)
    if fractions.shape != (2,) or not np.all((0 <= fractions) & (fractions <= 1)):
        raise ValueError("two inner resistance fractions in [0, 1] required")
    inner = resistance*fractions
    outer = resistance*(1-fractions)
    if np.shape(film_coefficients) != (2,) or any(
            h is not None and (not np.isrealobj(h) or not np.isfinite(h) or h <= 0)
            for h in film_coefficients):
        raise ValueError("supply a positive finite film coefficient or None for each physical wall")
    inner_film = [1/value if value else None for value in inner]
    outer_film = [1/value if value else None for value in outer]
    layer_films = [(film_coefficients[0], inner_film[0]), (outer_film[0], inner_film[1]),
                   (outer_film[1], film_coefficients[1])]
    ends = [0, *faces, len(r)-1]
    domains = []
    if return_bulk_storage:
        cell_storage = 1j*bulk_frequency*cv*np.pi*np.diff(r)*(r[:-1]+r[1:])
        storage_maps = np.zeros((3, 3), dtype=complex)
    if return_angular_conduction:
        cell_angular = 2*np.pi*k*azimuthal_mode**2*np.log1p(np.diff(r)/r[:-1])
        angular_maps = np.zeros((3, 3), dtype=complex)
    for layer, (a, b) in enumerate(zip(ends, ends[1:])):
        film = layer_films[layer]
        # The two unit temperature responses map end temperatures to radial rates.
        # The third column is the forced solution with homogeneous end data.
        response = np.zeros((2, 3), dtype=complex)
        for column in range(3 if np.any(volume_source[a:b]) else 2):
            boundary = (float(column == 0), float(column == 1))
            field, rate = solve_annular_conduction(
                r[a:b+1], k[a:b], cv[a:b], boundary, bulk_frequency, film_coefficients=film,
                source=volume_source[a:b] if column == 2 else 0.,
                azimuthal_mode=azimuthal_mode, cell_location="volume_centroid", distributed_source=True)
            response[:, column] = rate[[0, -1]]
            if return_bulk_storage:
                storage_maps[layer, column] = cell_storage[a:b] @ field
            if return_angular_conduction:
                angular_maps[layer, column] = cell_angular[a:b] @ field
        domains.append(response)

    area = 2*np.pi*r[faces]
    a, b, c = domains
    # Sheet generation supplies radial flux, storage, and circumferential conduction.
    matrix = np.array([[b[0, 0]-a[1, 1], b[0, 1]], [-b[1, 0], c[0, 0]-b[1, 1]]])
    admittance = sheet_h + 1j*omega*capacity + sheet_k*(azimuthal_mode/r[faces])**2
    matrix += np.diag(admittance*area)
    forcing = area*(generation + sheet_h*ambient) + [a[1, 0]*walls[0], -c[0, 1]*walls[1]]
    forcing += [a[1, 2]-b[0, 2], b[1, 2]-c[0, 2]]
    sheet = np.linalg.solve(matrix, forcing)
    middle = b @ [*sheet, 1.]
    inner_rates = a @ [walls[0], sheet[0], 1.]
    outer_rates = c @ [sheet[1], walls[1], 1.]
    incoming = np.array([inner_rates[1], middle[1]]) / area
    outgoing = np.array([middle[0], outer_rates[0]]) / area
    result = (sheet, incoming, outgoing)
    if return_wall_rates:
        result += (np.array([inner_rates[0], outer_rates[1]]),)
    if return_bulk_storage or return_angular_conduction:
        nodes = np.r_[walls[0], sheet, walls[1]]
    if return_bulk_storage:
        result += (storage_maps[:, 0]*nodes[:-1] + storage_maps[:, 1]*nodes[1:] + storage_maps[:, 2],)
    if return_angular_conduction:
        result += (angular_maps[:, 0]*nodes[:-1] + angular_maps[:, 1]*nodes[1:] + angular_maps[:, 2],)
    if return_wall_temperatures:
        surface = walls.copy()
        if film_coefficients[0] is not None:
            surface[0] -= inner_rates[0]/(2*np.pi*r[0]*film_coefficients[0])
        if film_coefficients[1] is not None:
            surface[1] += outer_rates[1]/(2*np.pi*r[-1]*film_coefficients[1])
        result += (surface,)
    if return_contact_temperatures:
        result += (np.column_stack((sheet + inner*incoming, sheet - outer*outgoing)),)
    return result
