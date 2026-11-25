"""Finite-volume response of two storing interfaces in a conducting annulus."""
import numpy as np

from .annulus import solve_annular_conduction


def solve_storing_rings(radii, conductivity, sheet_faces, wall_temperatures, omega, *,
                        contact_resistances, sheet_capacities, sheet_generation=(0., 0.),
                        azimuthal_mode=0, volumetric_heat_capacity=None):
    """Return two sheet temperatures and their incoming and outgoing radial fluxes.

    The cell faces include both sheets. Conductivity is scalar or per cell.
    Wall temperatures and areal heat generation (W/m²) are complex amplitudes of
    cos(mθ) exp(iωt). Omitting volumetric_heat_capacity gives quasi-steady bulk conduction.
    A positive scalar or per-cell capacity (J/(m³ K)) enables harmonic bulk storage.
    Each contact resistance (m² K/W)
    is split equally about the storing sheet (capacity in J/(m² K)).
    Each output has one entry per sheet ordered from the bore outwards. Fluxes
    (W/m²) are positive towards increasing radius on both sides.
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
    walls = np.asarray(wall_temperatures, dtype=complex)
    generation = np.asarray(sheet_generation, dtype=complex)
    capacity = np.asarray(sheet_capacities, dtype=float)
    resistance = np.asarray(contact_resistances, dtype=float)
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (walls, generation, capacity, resistance)):
        raise ValueError("supply two finite values per boundary or sheet property")
    if np.any(capacity < 0) or np.any(resistance < 0):
        raise ValueError("sheet capacities and contact resistances must be nonnegative")

    films = [2/value if value else None for value in resistance]
    boundary_films = [None, *films, None]
    ends = [0, *faces, len(r)-1]
    domains = []
    for layer, (a, b) in enumerate(zip(ends, ends[1:])):
        film = boundary_films[layer:layer+2]
        # The two unit Dirichlet responses map end temperatures to radial rates.
        columns = []
        for boundary in np.eye(2):
            _, rate = solve_annular_conduction(
                r[a:b+1], k[a:b], cv[a:b], boundary, bulk_frequency, film_coefficients=film,
                azimuthal_mode=azimuthal_mode, cell_location="volume_centroid", distributed_source=True)
            columns.append(rate[[0, -1]])
        domains.append(np.column_stack(columns))

    area = 2*np.pi*r[faces]
    a, b, c = domains
    # Area times (generation - iωCT) equals outgoing minus incoming power.
    matrix = np.array([[b[0, 0]-a[1, 1], b[0, 1]], [-b[1, 0], c[0, 0]-b[1, 1]]])
    matrix += np.diag(1j*omega*capacity*area)
    forcing = area*generation + [a[1, 0]*walls[0], -c[0, 1]*walls[1]]
    sheet = np.linalg.solve(matrix, forcing)
    middle = b @ sheet
    incoming = np.array([a[1] @ [walls[0], sheet[0]], middle[1]]) / area
    outgoing = np.array([middle[0], c[0] @ [sheet[1], walls[1]]]) / area
    return sheet, incoming, outgoing
