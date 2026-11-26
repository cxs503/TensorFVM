import numpy as np
import pytest

from tensorfvm.cht.storing_rings import solve_storing_rings


@pytest.mark.parametrize("resistance", [(0., 0.), (.3, .7)])
@pytest.mark.parametrize("fractions", [(.5, .5), (0., 1.), (1., 0.), (.2, .7)])
def test_dc_flux_and_sheet_location(resistance, fractions):
    radii = np.array([.3, .4, .6, .85, 1.])
    walls = [2.+.4j, -.2j]
    sheet, incoming, outgoing, traces = solve_storing_rings(
        radii, 2., [1, 3], walls, 0., contact_resistances=resistance, sheet_capacities=[.2, .4],
        inner_resistance_fractions=fractions, return_contact_temperatures=True)
    sheets = radii[[1, 3]]
    jumps = resistance/sheets
    rate_over_tau = (walls[0]-walls[1])/(np.log(radii[-1]/radii[0])/2 + sum(jumps))
    expected = walls[0]-rate_over_tau*(np.log(sheets/radii[0])/2 + jumps*fractions + [0, jumps[0]])
    np.testing.assert_allclose(sheet, expected, atol=1e-12)
    left = walls[0]-rate_over_tau*(np.log(sheets/radii[0])/2 + [0., jumps[0]])
    np.testing.assert_allclose(traces, np.column_stack((left, left-rate_over_tau*jumps)), atol=1e-12)
    np.testing.assert_allclose(incoming, rate_over_tau/sheets, atol=1e-12)
    np.testing.assert_allclose(outgoing, incoming, atol=1e-12)


def test_transparent_sheets_preserve_harmonic_bulk_flux():
    from tensorfvm.cht.annulus import solve_annular_conduction

    r = np.geomspace(.4, 1.1, 25)
    k = np.repeat([1., 2.5, .8], 8)
    cv = np.linspace(.7, 2.3, 24)
    walls = [1.+.4j, -.2j]
    faces = [8, 16]
    _, whole_rates = solve_annular_conduction(
        r, k, cv, walls, 8., azimuthal_mode=2, distributed_source=True, cell_location="volume_centroid")
    _, incoming, outgoing, wall_rates = solve_storing_rings(
        r, k, faces, walls, 8., volumetric_heat_capacity=cv, azimuthal_mode=2,
        sheet_capacities=(0., 0.), contact_resistances=(0., 0.), return_wall_rates=True)
    np.testing.assert_allclose(wall_rates, whole_rates[[0, -1]], rtol=1e-11)
    expected = whole_rates[faces]/(2*np.pi*r[faces])
    np.testing.assert_allclose(incoming, expected, rtol=1e-11)
    np.testing.assert_allclose(outgoing, expected, rtol=1e-11)


def test_uniform_angular_mode_is_invariant_to_sheet_conductivity():
    args = ([.3, .4, .6, .85, 1.], 2., [1, 3], [1.+.2j, -.1j], 4.)
    options = dict(contact_resistances=(.3, .7), sheet_capacities=(.2, .4), sheet_generation=(1., .6j))
    baseline = solve_storing_rings(*args, **options)
    actual = solve_storing_rings(*args, **options, sheet_conductivities=(10., .3))
    np.testing.assert_array_equal(actual, baseline)


def test_layer_storage_matches_single_domain_with_spatial_capacity():
    from tensorfvm.cht.annulus import solve_annular_conduction

    r = np.geomspace(.4, 1.1, 25)
    cv = np.linspace(.7, 2.3, 24)
    walls = [1.+.4j, -.2j]
    field, _ = solve_annular_conduction(r, 2., cv, walls, 8., azimuthal_mode=2,
        distributed_source=True, cell_location="volume_centroid")
    _, _, _, storage = solve_storing_rings(r, 2., [8, 16], walls, 8., azimuthal_mode=2,
        volumetric_heat_capacity=cv, sheet_capacities=(0., 0.), contact_resistances=(0., 0.),
        return_bulk_storage=True)
    expected = (8j*cv*np.pi*np.diff(r**2)*field).reshape(3, 8).sum(axis=1)
    np.testing.assert_allclose(storage, expected, rtol=1e-12)


@pytest.mark.parametrize("mode", [0, 2])
def test_integrated_angular_conduction_against_power_law(mode):
    edges = np.array([.5, .7, .9, 1.2])
    mesh = np.r_[np.concatenate([np.geomspace(a, b, 33)[:-1] for a, b in zip(edges, edges[1:])]), edges[-1]]
    k = 1.8
    # T=r^m is harmonic in a homogeneous annulus with transparent sheets.
    _, _, _, angular = solve_storing_rings(
        mesh, k, [32, 64], edges[[0, -1]]**mode, 0., azimuthal_mode=mode,
        sheet_capacities=(0., 0.), contact_resistances=(0., 0.), return_angular_conduction=True)
    np.testing.assert_allclose(angular, 2*np.pi*k*mode*np.diff(edges**mode), rtol=1e-4, atol=0.)


@pytest.mark.parametrize("films", [(7., None), (None, 3.), (7., 3.)])
def test_dc_wall_films_against_radial_resistances(films):
    edges = np.array([.4, .55, .9, 1.1])
    k = [1., 2.5, .8]
    contacts = np.array([.5, .9])
    ambient = np.array([1.+.2j, -.3j])
    _, _, _, rates, surface = solve_storing_rings(
        edges, k, [1, 2], ambient, 0., contact_resistances=contacts, sheet_capacities=(.25, .8),
        film_coefficients=films, return_wall_rates=True, return_wall_temperatures=True)
    film_resistance = sum(1/(r*h) for r, h in zip(edges[[0, -1]], films) if h is not None)
    resistance = sum(np.diff(np.log(edges))/k) + sum(contacts/edges[1:-1]) + film_resistance
    np.testing.assert_allclose(rates, 2*np.pi*(ambient[0]-ambient[1])/resistance, rtol=1e-12)
    film_drops = np.array([0. if h is None else 1/(r*h) for h, r in zip(films, edges[[0, -1]])])
    expected_surface = ambient + np.array([-1., 1.])*film_drops*(ambient[0]-ambient[1])/resistance
    np.testing.assert_allclose(surface, expected_surface, rtol=1e-11)


@pytest.mark.parametrize("heated_layer", [0, 1, 2])
def test_local_bulk_heater_with_transparent_sheets(heated_layer):
    from tensorfvm.cht.annulus import solve_annular_conduction

    mesh = np.geomspace(.5, 1.2, 25)
    source = np.zeros(24, dtype=complex)
    source[8*heated_layer:8*(heated_layer+1)] = .8+.35j
    _, rates = solve_annular_conduction(mesh, 2., 1.4, (0., 0.), 4., source=source,
        azimuthal_mode=2, cell_location="volume_centroid", distributed_source=True)
    _, incoming, outgoing, walls = solve_storing_rings(
        mesh, 2., [8, 16], (0., 0.), 4., contact_resistances=(0., 0.), sheet_capacities=(0., 0.),
        source=source, volumetric_heat_capacity=1.4, azimuthal_mode=2, return_wall_rates=True)
    expected = rates[[8, 16]]/(2*np.pi*mesh[[8, 16]])
    np.testing.assert_allclose(incoming, expected, rtol=1e-11)
    np.testing.assert_allclose(outgoing, expected, rtol=1e-11)
    np.testing.assert_allclose(walls, rates[[0, -1]], rtol=1e-11)


def test_uniform_sheet_temperature_under_balanced_reservoir_loading():
    t = .7-.2j
    h = np.array([1.3, 4.2])
    ambient = np.array([-.2j, .5+.1j])
    capacity = np.array([.25, .8])
    generation = 8j*capacity*t + h*(t-ambient)
    sheets, _, _, rates = solve_storing_rings([.4, .55, .9, 1.1], [1., 2.5, .8], [1, 2],
        [t, t], 8., contact_resistances=[.5, .9], sheet_capacities=capacity, sheet_generation=generation,
        sheet_film_coefficients=h, sheet_ambient_temperatures=ambient, return_wall_rates=True)
    np.testing.assert_allclose(sheets, t, atol=1e-12)
    np.testing.assert_allclose(rates, 0., atol=1e-12)


@pytest.mark.parametrize("sides", [(0,), (1,), (0, 1)])
def test_flux_reconstructs_generated_dirichlet_solution(sides):
    r = np.geomspace(.4, 1.1, 13)
    original = [1.+.2j, -.3j]
    props = dict(contact_resistances=(.5, .9), sheet_capacities=(.25, .8), source=.7+.2j,
                 azimuthal_mode=1, volumetric_heat_capacity=1.3, return_wall_rates=True,
                 return_bulk_storage=True, return_angular_conduction=True, return_wall_temperatures=True)
    expected = solve_storing_rings(r, 2., [4, 8], original, 8., **props)
    inward = [None, None]
    walls = list(original)
    for side in sides:
        inward[side] = (1 if side == 0 else -1)*expected[3][side]/(2*np.pi*r[[0,-1]][side])
        walls[side] = None
    actual = solve_storing_rings(r, 2., [4, 8], walls, 8., wall_fluxes=inward, **props)
    for result, target in zip(actual, expected):
        np.testing.assert_allclose(result, target, atol=1e-11)


def test_double_flux_requires_a_temperature_anchor_in_static_zero_mode():
    args = ([.4,.55,.9,1.1], 2., [1,2], (None,None), 0.)
    options = dict(contact_resistances=(.2,.5), sheet_capacities=(.4,.8), wall_fluxes=(0.,0.))
    with pytest.raises(ValueError, match="temperature reference"):
        solve_storing_rings(*args, **options)
    sheet, qi, qo, wall = solve_storing_rings(*args, **options, sheet_film_coefficients=(2.,0.),
        sheet_ambient_temperatures=(.3+.4j,0.), return_wall_temperatures=True)
    np.testing.assert_allclose(np.r_[sheet,wall], .3+.4j)
    np.testing.assert_allclose(np.r_[qi,qo], 0., atol=1e-12)


def test_floating_double_flux_preserves_heat_rates_under_large_datum_shift():
    r = [.4, .55, .9, 1.1]
    options = dict(contact_resistances=(.5, .9), sheet_capacities=(.25, .8), return_wall_rates=True,
                   return_wall_temperatures=True, return_contact_temperatures=True)
    args = (r, 2., [1,2], (None,None), 0.)
    q = (1.+.2j, -(.4/1.1)*(1.+.2j))
    base = solve_storing_rings(*args, **options, wall_fluxes=q, bore_temperature_reference=0.)
    shifted = solve_storing_rings(*args, **options, wall_fluxes=q, bore_temperature_reference=1e9)
    for i in (1,2,3):
        np.testing.assert_array_equal(shifted[i], base[i])
    for i in (0,4,5):
        np.testing.assert_array_equal(shifted[i], base[i]+1e9)
    with pytest.raises(ValueError, match="incompatible"):
        solve_storing_rings(*args, **options, wall_fluxes=q, source=.1, bore_temperature_reference=0.)


def test_volume_mean_reference_uses_shell_volumes_and_resists_conflicting_data():
    edges = np.array([.4, .5, .8, 1.2])
    volumes = np.diff(edges**2)
    centers = (2/3)*np.diff(edges**3)/volumes
    flux = (.6+.2j, -edges[0]/edges[-1]*(.6+.2j))
    slope = -edges[0]*flux[0]/2.
    mean = .7+.3j
    bore = mean - np.average(slope*np.log(centers/edges[0]), weights=volumes)
    args = (edges, 2., [1, 2], (None, None), 0.)
    options = dict(sheet_capacities=(0.,0.), contact_resistances=(0.,0.), wall_fluxes=flux)
    _, _, _, wall, field = solve_storing_rings(*args, **options, mean_temperature=mean,
        return_wall_temperatures=True, return_cell_temperatures=True)
    np.testing.assert_allclose(field, bore+slope*np.log(centers/edges[0]), rtol=0., atol=1e-13)
    np.testing.assert_allclose(wall, bore+slope*np.log(edges[[0,-1]]/edges[0]), rtol=0., atol=1e-13)
    with pytest.raises(ValueError, match="exactly one"):
        solve_storing_rings(*args, **options, mean_temperature=mean, bore_temperature_reference=bore)
