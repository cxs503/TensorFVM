import numpy as np
import pytest

from tensorfvm.cht.storing_rings import solve_storing_rings


@pytest.mark.parametrize("resistance", [(0., 0.), (.3, .7)])
@pytest.mark.parametrize("fractions", [(.5, .5), (0., 1.), (1., 0.), (.2, .7)])
def test_dc_flux_and_sheet_location(resistance, fractions):
    radii = np.array([.3, .4, .6, .85, 1.])
    walls = [2.+.4j, -.2j]
    sheet, incoming, outgoing = solve_storing_rings(
        radii, 2., [1, 3], walls, 0., contact_resistances=resistance, sheet_capacities=[.2, .4],
        inner_resistance_fractions=fractions)
    sheets = radii[[1, 3]]
    jumps = resistance/sheets
    rate_over_tau = (walls[0]-walls[1])/(np.log(radii[-1]/radii[0])/2 + sum(jumps))
    expected = walls[0]-rate_over_tau*(np.log(sheets/radii[0])/2 + jumps*fractions + [0, jumps[0]])
    np.testing.assert_allclose(sheet, expected, atol=1e-12)
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
    _, _, _, rates = solve_storing_rings(
        edges, k, [1, 2], ambient, 0., contact_resistances=contacts, sheet_capacities=(.25, .8),
        film_coefficients=films, return_wall_rates=True)
    film_resistance = sum(1/(r*h) for r, h in zip(edges[[0, -1]], films) if h is not None)
    resistance = sum(np.diff(np.log(edges))/k) + sum(contacts/edges[1:-1]) + film_resistance
    np.testing.assert_allclose(rates, 2*np.pi*(ambient[0]-ambient[1])/resistance, rtol=1e-12)


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
