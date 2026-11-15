"""Tests for the KMM180 closure-free turbulence diagnostic (m22266).

The "管到底" diagnostic answers whether the vacuous PASS (m20573) can be
repaired by running the closure-free solver longer / finer.  These tests pin
the honest finding: the solver does not develop resolved Reynolds shear at the
available resolution, so the closure-free ``u+`` grades the initial condition
and the PASS is not repairable (falsified hypothesis).

The assertions target the *discriminating power* of the diagnostic, not "the
code ran":

* a short closure-free run produces no resolved Reynolds shear (``uv_share``
  stays ~0, ``uv_max`` stays O(1e-3) or below) -- the C1 precondition is never
  met;
* the diagnostic's ``conclusion`` is ``VACUOUS_PASS_NOT_REPAIRABLE`` when C1
  is not met;
* the diagnostic writes a machine-readable ``turbulence_diag.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.channel_spectral_dns import (
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
)


def run_short_diag(nx=16, ny=64, nz=16, steps=300, noise=0.5, seed=11):
    """Run the closure-free solver briefly and return the log-layer shear share."""
    config = ChannelConfig(re_tau=180.0, nx=nx, ny=ny, nz=nz)
    solver = SpectralChannel(config)
    solver.initialize(noise=noise, seed=seed)
    for _ in range(steps):
        solver.step()
    acc = StatisticsAccumulator(solver)
    for _ in range(20):
        solver.step()
        acc.add()
    st = acc.folded()
    mask = (st.y_plus >= 30) & (st.y_plus <= 36)
    target = 1.0 - st.y_plus / 180.0
    share = -st.uv_plus[mask] / target[mask]
    return dict(uv_share=float(share.mean()),
                uv_max=float(np.abs(st.uv_plus).max()),
                u_rms=float(np.sqrt(np.maximum(st.uu_plus, 0)).max()))


def test_closure_free_run_does_not_develop_reynolds_shear():
    """A short closure-free run produces no resolved Reynolds shear (C1 unmet)."""
    d = run_short_diag()
    # the resolved Reynolds-shear share in the log layer is ~0 (not >= 0.5)
    assert d['uv_share'] < 0.05
    # the uv_plus peak is O(1e-3) or below -- no turbulent stress
    assert d['uv_max'] < 1e-2
    # the fluctuation intensity is small (perturbation has not grown)
    assert d['u_rms'] < 1.0


def test_diag_conclusion_is_vacuous_not_repairable_when_c1_unmet(tmp_path):
    """The diagnostic writes a turbulence_diag.json with the C1 verdict."""
    from scripts.diag_kmm180_turbulence import main as diag_main
    out = tmp_path / 'diag'
    old_argv = sys.argv
    sys.argv = ['diag_kmm180_turbulence.py',
                '--nx', '16', '--ny', '64', '--nz', '16',
                '--steps', '200', '--probe-every', '100', '--out', str(out)]
    try:
        rc = diag_main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    summary = json.loads((out / 'turbulence_diag.json').read_text())
    assert summary['c1_uv_share_met'] is False
    assert 'VACUOUS_PASS_NOT_REPAIRABLE' in summary['conclusion']
    assert summary['c1_uv_share_max'] < 0.05
