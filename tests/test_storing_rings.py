import numpy as np
import pytest

from tensorfvm.cht.storing_rings import solve_storing_rings


@pytest.mark.parametrize("resistance", [(0., 0.), (.3, .7)])
def test_dc_flux_and_sheet_location(resistance):
    radii = np.array([.3, .4, .6, .85, 1.])
    walls = [2.+.4j, -.2j]
    sheet, incoming, outgoing = solve_storing_rings(
        radii, 2., [1, 3], walls, 0., contact_resistances=resistance, sheet_capacities=[.2, .4])
    sheets = radii[[1, 3]]
    jumps = resistance/sheets
    rate_over_tau = (walls[0]-walls[1])/(np.log(radii[-1]/radii[0])/2 + sum(jumps))
    expected = walls[0]-rate_over_tau*(np.log(sheets/radii[0])/2 + jumps/2 + [0, jumps[0]])
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
    _, incoming, outgoing = solve_storing_rings(
        r, k, faces, walls, 8., volumetric_heat_capacity=cv, azimuthal_mode=2,
        sheet_capacities=(0., 0.), contact_resistances=(0., 0.))
    expected = whole_rates[faces]/(2*np.pi*r[faces])
    np.testing.assert_allclose(incoming, expected, rtol=1e-11)
    np.testing.assert_allclose(outgoing, expected, rtol=1e-11)
