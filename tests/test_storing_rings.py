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
