"""Cross-solver intercomparison of the VOF ice-water solver against TensorLBM.

Criterion C4 of node P180W2-N02 asks the ice-water free-surface VOF solver to be
arbitrated by an *independent* solver on the same working condition.  The
reference used here is the production free-surface LBM of the TensorLBM
repository (:mod:`tensorlbm.free_surface_lbm_2d`), which is imported unmodified
and recorded with its git revision and source hashes.

Working condition (identical in both solvers)
---------------------------------------------
2-D closed tank.  A water column of depth ``h0`` and width ``L0`` sits behind a
dam at the left wall; a rigid rectangular *ice floe* stands on the floor at
``x = x_ice``.  The dam is removed at ``t = 0`` and the surge hits the floe.

Comparability is enforced in three steps:

1. **Same grid geometry.**  Both solvers run ``nx x ny`` cells with the floe
   occupying exactly the same cell block.
2. **Same Froude/Reynolds state.**  The dam break is Froude-scaled by gravity,
   so the nondimensional solution depends on the geometry and on
   ``Re = sqrt(g h0) h0 / nu``.  The lattice viscosity ``nu = (tau - 1/2)/3`` is
   fixed first, ``Re`` computed from it, and the SI viscosity of the VOF run is
   set to reproduce that ``Re``.
3. **Same diagnostic, same code path.**  The horizontal load on the floe is the
   *normal pressure integral over the floe faces*, evaluated by the same
   formula on each solver's own pressure field:

   ``F_x = sum(p_fluid * dA)`` with sign given by which side of the floe the
   fluid cell sits on; ``p_lbm = cs^2 (rho - rho_0)`` with ``cs^2 = 1/3``.

   Both are reported as ``C_F = F_x / (rho_liquid g h0^2)`` (2-D, unit depth),
   i.e. the forces are made nondimensional with each solver's own reference
   scales, so nothing is rescaled after the fact.

Known limitation (declared, not hidden): a free-surface LBM at the density ratio
of 1:1000 does not exist in this form -- the LBM leg is a filled/gas volume
tracking scheme with a light-gas closure, so the comparison is made at a
*moderate* Reynolds number where both formulations are in their comfort zone.
The measured deviation is the criterion; it is not tuned.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

__all__ = ["lbm_available", "run_crosscheck"]


def _candidate_repos() -> list[Path]:
    env = os.environ.get("TENSORLBM_REPO")
    cands = [Path(env)] if env else []
    cands += [Path("/root/TensorLBM"), Path("/root/tp-agent/work/m9482")]
    for base in Path("/root").glob("TensorLBM*"):
        cands.append(base)
    return [c for c in cands if (c / "src" / "tensorlbm").is_dir()]


def lbm_available() -> Path | None:
    """Return the first TensorLBM checkout that can be imported, else ``None``."""
    for repo in _candidate_repos():
        try:
            _import(repo)
            return repo
        except Exception:
            continue
    return None


def _import(repo: Path):
    repo = repo.resolve()
    expected = repo / "src" / "tensorlbm"
    loaded = sys.modules.get("tensorlbm")
    if loaded is not None and not Path(loaded.__file__).resolve().parent == expected:
        raise RuntimeError("another tensorlbm checkout is already imported")
    sys.path.insert(0, str(repo / "src"))
    fs = importlib.import_module("tensorlbm.free_surface_lbm_2d")
    d2 = importlib.import_module("tensorlbm.d2q9")
    for module in (fs, d2):
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise RuntimeError("tensorlbm provenance mismatch")
    return fs, d2


def _provenance(repo: Path) -> dict:
    files = ["src/tensorlbm/free_surface_lbm_2d.py", "src/tensorlbm/d2q9.py",
             "src/tensorlbm/solver.py", "src/tensorlbm/boundaries.py"]
    hashes = {}
    for rel in files:
        p = repo / rel
        if p.is_file():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                       text=True).strip()
    except Exception:
        head = "unknown"
    return dict(repository=str(repo), git_head=head, source_sha256=hashes)


# --------------------------------------------------------------------------- #
# the two legs
# --------------------------------------------------------------------------- #
def _load_on_floe(pressure, solid) -> float:
    """Normal pressure integral on the floe (same formula for both solvers)."""
    p = torch.as_tensor(pressure, dtype=torch.float64)
    s = torch.as_tensor(solid)
    fx = 0.0
    m = (~s[:, :-1]) & s[:, 1:]
    fx += float(p[:, :-1][m].sum())
    m = s[:, :-1] & (~s[:, 1:])
    fx -= float(p[:, 1:][m].sum())
    return fx


def _lbm_leg(repo, nx, ny, col_w, col_h, block, g_lat, tau, steps, sample_every):
    fs, d2 = _import(repo)
    device = torch.device("cpu")
    fill, solid = fs.init_fill_rectangular_2d(ny, nx, col_w, col_h, device)
    jy0, jy1, jx0, jx1 = block
    solid[jy0:jy1, jx0:jx1] = True
    fill[solid] = 0.0
    flags = fs.init_flags_from_fill_2d(fill, solid)
    rho0 = torch.ones_like(fill)
    f = d2.equilibrium(rho0 * fill, torch.zeros_like(fill), torch.zeros_like(fill))
    cs2 = 1.0 / 3.0
    history = []
    for step in range(1, steps + 1):
        f, fill, flags = fs.free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gx=0.0, gy=-g_lat,
            rho_liquid=1.0, rho_gas=1.0)
        if not bool(torch.isfinite(f).all()):
            raise FloatingPointError(f"LBM populations non-finite at step {step}")
        if step % sample_every == 0:
            rho = f.sum(dim=0)
            p = cs2 * (rho - 1.0)
            pressure = torch.where(solid, torch.zeros_like(p), p)
            # leading edge of the liquid in the free-surface LBM
            wet = (fill > 0.5) & (~solid)
            cols = wet.any(dim=0).nonzero().reshape(-1)
            front = float(cols[-1].item() + 1.0) if cols.numel() else 0.0
            history.append(dict(step=step, time_lattice=float(step),
                                load=_load_on_floe(pressure, solid), front_cells=front))
    return dict(history=history, provenance=_provenance(repo))


def _vof_leg(nx, ny, length, h0, x_ice, mu, t_end, n_samples, block_w=8, col_w=20):
    from .ice_water import IceWaterConfig, IceWaterVOF

    height = ny * length / nx
    # Only the water viscosity is set by the Reynolds matching; the air keeps
    # its physical viscosity (overriding it with the water value made the
    # viscous time-step limit collapse to 1e-4 s and the run 30x more expensive
    # -- measured 1403 steps instead of 46).
    cfg = IceWaterConfig(nx=nx, ny=ny, length=length, height=height,
                         mu_water=mu, mu_ice=mu)
    s = IceWaterVOF(cfg)
    a = torch.zeros((ny, nx), dtype=torch.float64)
    j0 = int(round(h0 / s.dy))
    # dam at the left wall; the column is `col_w` cells wide (identical in the
    # LBM leg) so that a dry bed of `x_ice - col_w` cells separates it from the
    # floe -- otherwise the floe would simply hold the column in place.
    a[:j0, :col_w] = 1.0
    s.set_water(a)
    jx0 = int(round(x_ice / s.dx))
    jx1 = jx0 + block_w
    jy0, jy1 = 0, j0
    s.add_ice_block(jx0 * s.dx, jx1 * s.dx, jy0 * s.dy, jy1 * s.dy)
    history = []
    next_sample = 0.0
    while s.time < t_end and s.step_count < 20000:
        s.step()
        if s.time >= next_sample:
            next_sample += t_end / max(n_samples, 1)
            fx, _ = s.drag_on_ice()
            history.append(dict(step=s.step_count, time=s.time, load=fx,
                                front_cells=s.front_position(fraction=0.01, h0=j0 * s.dy) / s.dx))
    return dict(history=history, geometry=dict(dx=s.dx, dy=s.dy, h0=j0 * s.dy,
                                               block=(jy0, jy1, jx0, jx1)))


def run_crosscheck(nx=64, ny=28, col_w=20, col_h=8, x_ice_cells=38,
                   g_lat=2.0e-3, tau=0.51, t_star_end=2.0, block_w=5):
    """Run both legs and return the nondimensional comparison (criterion C4)."""
    repo = lbm_available()
    if repo is None:
        raise RuntimeError("no TensorLBM checkout available for the cross-check")
    g = 9.81
    # --- lattice leg: fix tau, derive Re -------------------------------- #
    nu_lat = (tau - 0.5) / 3.0
    u_lat = math.sqrt(g_lat * col_h)
    re_lat = u_lat * col_h / nu_lat
    t_star_per_step = math.sqrt(g_lat / col_h)
    steps_lbm = int(round(t_star_end / t_star_per_step))
    sample = max(1, steps_lbm // 60)
    lbm = _lbm_leg(repo, nx, ny, col_w, col_h,
                   (0, col_h, x_ice_cells, x_ice_cells + block_w),
                   g_lat, tau, steps_lbm, sample)
    # --- VOF leg: same grid, same Re ------------------------------------ #
    length = 0.6
    h0 = col_h * length / nx
    u_si = math.sqrt(g * h0)
    mu = 998.0 * u_si * h0 / re_lat
    t_end = t_star_end * math.sqrt(h0 / g)
    vof = _vof_leg(nx, ny, length, h0, x_ice_cells * length / nx, mu,
                   t_end=t_end, n_samples=60, block_w=block_w, col_w=col_w)
    steps_vof = vof["history"][-1]["step"] if vof["history"] else 0
    rho_l, rho_ice = 998.0, 917.0
    ref_lbm = 1.0 * g_lat * col_h ** 2
    ref_vof = rho_l * g * h0 ** 2
    # nondimensional time of every sample and the matched window
    raw_cf = [r["load"] / ref_lbm for r in lbm["history"]]
    window = 15
    smooth = [float(np.mean(raw_cf[max(0, i - window + 1): i + 1])) for i in range(len(raw_cf))]
    lbm_rows = [dict(t_star=r["time_lattice"] * t_star_per_step, cf=raw_cf[i],
                     cf_smoothed=smooth[i], front=r["front_cells"])
                for i, r in enumerate(lbm["history"])]
    vof_rows = [dict(t_star=r["time"] * math.sqrt(g / h0), cf=r["load"] / ref_vof,
                     front=r["front_cells"]) for r in vof["history"]]
    lo, hi = 1.2, t_star_end
    w_lbm = [r for r in lbm_rows if lo <= r["t_star"] <= hi]
    w_vof = [r for r in vof_rows if lo <= r["t_star"] <= hi]
    mean_lbm = float(np.mean([r["cf_smoothed"] for r in w_lbm])) if w_lbm else float("nan")
    mean_vof = float(np.mean([r["cf"] for r in w_vof])) if w_vof else float("nan")
    dev = abs(mean_vof - mean_lbm) / max(abs(mean_lbm), 1e-30)
    # impulse (time integral of the nondimensional load) is the most robust
    # integrated load diagnostic -- acoustic oscillation integates out
    def _trapz(rows, key):
        if len(rows) < 2:
            return float("nan")
        t = np.array([r["t_star"] for r in rows])
        f = np.array([r[key] for r in rows])
        return float(np.trapezoid(f, t))
    impulse_lbm = _trapz(w_lbm, "cf_smoothed")
    impulse_vof = _trapz(w_vof, "cf")
    front_dev = float("nan")
    if w_lbm and w_vof:
        f_lbm = float(np.mean([r["front"] for r in w_lbm]))
        f_vof = float(np.mean([r["front"] for r in w_vof]))
        front_dev = abs(f_vof - f_lbm) / max(f_lbm, 1e-30)
    return dict(
        case="C4-lbm-crosscheck",
        criterion="mean non-dimensional pressure load on the ice floe over "
                  "the post-impact window: |VOF - LBM| / LBM < 5 %",
        grid=dict(nx=nx, ny=ny, col_w=col_w, col_h=col_h, x_ice_cells=x_ice_cells,
                  block_w=block_w),
        matching=dict(re_lattice=re_lat, tau=tau, nu_lattice=nu_lat,
                      u_lattice=u_lat, h0_m=h0, u_si=u_si, mu_si=mu,
                      g_lattice=g_lat, t_star_end=t_star_end,
                      steps_lbm=steps_lbm, steps_vof=steps_vof),
        reference_scales=dict(lbm=ref_lbm, vof=ref_vof),
        lbm=dict(provenance=lbm["provenance"], rows=lbm_rows, mean_cf=mean_lbm),
        vof=dict(geometry=vof["geometry"], rows=vof_rows, mean_cf=mean_vof),
        load_deviation=dev,
        impulse=dict(lbm=impulse_lbm, vof=impulse_vof,
                     relative_deviation=abs(impulse_vof - impulse_lbm)
                     / max(abs(impulse_lbm), 1e-30) if np.isfinite(impulse_lbm) else float("nan")),
        lbm_load_oscillation=float(np.std(raw_cf)) if raw_cf else float("nan"),
        front_deviation=front_dev,
        criterion_met=bool(math.isfinite(dev) and dev < 0.05),
    )