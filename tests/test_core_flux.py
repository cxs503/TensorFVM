"""Core flux registry: four families plus the wired overset transfer."""
import numpy as np
import pytest
import torch

from tensorfvm.core.flux import FluxFunction, get_flux, list_families

FAMILIES = {"vof_plic", "vof_cicsam", "vof_thinc", "compressible_slope",
            "overset_transfer"}


def test_five_families_registered():
    assert set(list_families()) == FAMILIES


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        get_flux("no_such_family")


def test_thinc_family_delegates():
    flux = get_flux("vof_thinc")
    assert isinstance(flux, FluxFunction)
    alpha = torch.full((3,), 0.5, dtype=torch.float64)
    out = flux.flux(alpha, torch.full((3,), 0.5, dtype=torch.float64))
    # uniform cell: interface at the centre, the strip [0.5-c, 0.5] carries
    # (nearly) the whole left half, clamped to courant
    assert torch.allclose(out, torch.full((3,), 0.5, dtype=torch.float64), atol=1e-3)


def test_compressible_slope_family_delegates():
    flux = get_flux("compressible_slope")
    q = torch.zeros(6, 4)
    out = flux.flux(q, 0, "minmod")
    assert out.shape == q.shape
    with pytest.raises(ValueError):
        flux.flux(q, 0, "not-a-limiter")


def test_cicsam_family_delegates():
    flux = get_flux("vof_cicsam")
    a = torch.full((4,), 0.5, dtype=torch.float64)
    cour = torch.full((4,), 0.25, dtype=torch.float64)
    gamma = torch.zeros(4, dtype=torch.float64)
    out = flux.flux(a, a, a, cour, gamma)
    assert out.shape == a.shape


def test_overset_transfer_is_wired_and_conserving():
    from tensorfvm.overset_flux import PlanarFluxTransfer
    donor = np.array([[[0.0, 0.0], [1.0, 1.0]]])
    receiver = np.array([[[0.0, 0.0], [0.5, 1.0]],
                        [[0.5, 0.0], [1.0, 1.0]]])
    transfer = PlanarFluxTransfer.build(donor, receiver)
    flux = get_flux("overset_transfer", transfer)
    out = flux.flux(np.array([2.0]))
    # oriented conservation: received total = -sent total (normal_sign=-1)
    assert np.allclose(out, [-1.0, -1.0])
    assert np.allclose(flux.balance(np.array([2.0]), out), 0.0)
