"""Multi-block overset driver for the three-element (slat + main + flap) airfoil.

Reuses the validated per-block curvilinear operators in tensorlbm.curve_grid and
the overset coupling in tensorlbm.overset.  Each block carries its own body-fitted
C-grid + wall; the Overset layer rebuilds each block's `recv` mask as

        recv = HOLE | FRINGE | FARFIELD

and exchanges velocity every step (hole/fringe <- donor bilinear, farfield <-
free stream).  The per-block Chorin projection solver is used UNCHANGED (it only
ever solves the ~recv cells with p=0 at the recv boundary -- exactly the
velocity-only overset the prototype validated, no pressure exchange, so no
double-Dirichlet over-constraint blow-up).

Time loop (per step):
    Overset.exchange()                 # hole/fringe <- donor, farfield <- U_inf
    for each block:
        predictor  (explicit conv + diffusion)
        re-impose wall = 0, recv = donor / U_inf
        project()  (Chorin, Poisson on solved cells, p = 0 at recv)
        re-impose wall = 0, recv = donor / U_inf
    sum forces over blocks -> total Cd / Cl

The adapter OBlock lets tensorlbm.overset (which indexes b['X'], b['wall'],
b['recv'], b['u'], ...) drive the SAME arrays the solver reads/writes, so in-place
overset writes (b['u'][idx] = v) are seen by curve_grid -- no field copying.
"""
import numpy as np
from tensorlbm import curve_grid as CG
from tensorlbm.cgrid_gen import make_three_element_cgrid
from tensorlbm.overset import Overset, cell_centers


class OBlock:
    """Dict-style adapter over a curve_grid.Block so tensorlbm.overset can drive
    the block's arrays.  In-place writes by Overset hit blk.u/blk.v/blk.p, so the
    solver and the overset exchange share one set of fields."""

    def __init__(self, blk, X, Y):
        self.blk = blk
        self.X = X
        self.Y = Y

    def __getitem__(self, k):
        b = self.blk
        if k == "X":
            return self.X
        if k == "Y":
            return self.Y
        if k == "wall":
            return b.wall
        if k == "recv":
            return b.recv
        if k == "u":
            return b.u
        if k == "v":
            return b.v
        if k == "p":
            return b.p
        if k == "nu_tilde":
            return getattr(b, "nu_tilde", None)
        if k == "_xc":
            return b._xc
        if k == "_yc":
            return b._yc
        if k == "_ni":
            return b._ni
        if k == "_nj":
            return b._nj
        if k == "hole":
            return b._hole
        if k == "fringe":
            return b._fringe
        if k == "farfield":
            return b._farfield
        raise KeyError(k)

    def __setitem__(self, k, v):
        b = self.blk
        if k == "recv":
            b.recv = v
        elif k == "u":
            b.u = v
        elif k == "v":
            b.v = v
        elif k == "p":
            b.p = v
        elif k == "nu_tilde":
            b.nu_tilde = v
        elif k == "X":
            self.X = v
        elif k == "Y":
            self.Y = v
        elif k in ("_xc", "_yc", "_ni", "_nj"):
            setattr(b, k, v)
        elif k == "hole":
            b._hole = v
        elif k == "fringe":
            b._fringe = v
        elif k == "farfield":
            b._farfield = v
        else:
            raise KeyError(k)

    def get(self, k, default=None):
        if k == "name":
            return getattr(self.blk, "name", default)
        try:
            return self[k]
        except KeyError:
            return default


def make_te_blocks(geom_dict, aoa_deg, U=1.0, Re=100.0, turbulent=False,
                   nu_tilde_inf=None, sa_scheme="upwind1", nu_t_cap=None,
                   trip=False, U_init=None, j_floor=0.0):
    """Build one curve_grid.Block per element, initialised to the (ramped) free
    stream, with no-slip walls.  Returns (blocks, oblocks) where oblocks are the
    Overset-facing adapters sharing the same arrays.  When `turbulent` is True a
    Spalart-Allmaras working variable nu_tilde is initialised to `nu_tilde_inf`
    (default 3*nu) everywhere and the block is tagged for SA coupling.  nu_t_cap
    bounds the eddy viscosity (so the diffusion-CFL dt stays finite); default is
    the larger of 10*nu and the free-stream nu_t, which is plenty for a resolved
    BL but keeps the explicit step stable on coarse grids.

    U_init (optional) is the free-stream SPEED the velocity field is initialised
    to.  CRITICAL for stability: it MUST match the ramped free-stream speed at
    step 1 (U*min(1,1/ramp)), otherwise the interior starts at the full free
    stream (U) while the far-field BC is clamped to the tiny ramped value, and the
    resulting interior/boundary velocity mismatch drives a huge spurious pressure
    (p ~ 1e1) that the PPM convection then amplifies to divergence within a few
    steps -- especially at high Re where molecular diffusion is far too weak to
    damp the mismatch.  When U_init is None the field is initialised to the full
    free stream U (use only with ramp=1 / impulsive start)."""
    aoa = np.deg2rad(aoa_deg)
    Us = U if U_init is None else U_init
    cu, cv = Us * np.cos(aoa), Us * np.sin(aoa)
    blocks, oblocks = [], []
    for el in geom_dict["elements"]:
        X, Y, wall = el["X"], el["Y"], el["wall"]
        # periodic_i: the C-grid wake cut (columns i=0 and i=ni-1) is the SAME
        # physical wake-centreline line and must be PERIODIC for the MAIN element
        # (its wake is (approximately) symmetric about the chord, so the two wake
        # halves meeting at the cut is physical).  The slat/flap are DEPLOYED,
        # cambered elements whose wakes trail ASYMMETRICALLY into the main's flow,
        # so a periodic cut would impose a false symmetry and create a pressure
        # discontinuity at the seam that rAU=1 then over-corrects to divergence.
        # Those elements keep the (overset-overwritten) Dirichlet cut instead.
        periodic = False
        blk = CG.make_cblock(X, Y, periodic_i=periodic, U=U, Lref=el["Lref"],
                             wall_mask=wall, j_floor=j_floor)
        blk.name = el["name"]
        ni, nj = blk.ni, blk.nj
        blk._nu = U * el["Lref"] / Re
        blk.u = np.full((ni, nj), cu)
        blk.v = np.full((ni, nj), cv)
        blk.u[wall] = 0.0
        blk.v[wall] = 0.0
        blk.p = np.zeros((ni, nj))
        xc, yc = cell_centers(X, Y)
        blk._xc, blk._yc = xc, yc
        blk._ni, blk._nj = ni, nj
        if turbulent:
            nui = 3.0 * blk._nu if nu_tilde_inf is None else nu_tilde_inf
            blk.nu_tilde = np.full((ni, nj), nui)
            blk._nu_tilde_inf = nui
            blk._sa_on = True
            blk._sa_scheme = sa_scheme
            blk._sa_trip = trip
            blk._nu_t_cap = (max(10.0 * blk._nu, nui)
                             if nu_t_cap is None else nu_t_cap)
        blocks.append(blk)
        oblocks.append(OBlock(blk, X, Y))
    return blocks, oblocks


def compute_dt(blocks, U=1.0, scheme="ppm", nu_hyp=0.0, cfl=0.20):
    """Stable explicit dt from the diffusion CFL on the FINEST (min-edge) cell of
    each block, turbulence-aware.  The diffusion limit is
        dt <= safety / (coeff_max * max_cell(1/hx^2 + 1/hy^2))
    where coeff_max = nu (+ nu_t_cap when turbulent) and the max is over the most
    stretched cell.  This uses the true min edge, not sqrt(|J|.min()) (the
    geometric-mean edge), which is far too loose on stretched C-grid cells and
    was letting the explicit diffusion diverge.  Hard-capped at 5e-2.

    `cfl` is the explicit convective/viscous safety factor.  The high-lift 30P30N
    C-grid has a near-TE wake cell whose fine (y+~50) normal edge + shear layer
    makes the local convective CFL grow once Ueff>~0.5; the single-block sweep
    shows the instability is purely CFL-limited (monotonic in `cfl`: 0.10 blows at
    step ~185, 0.05 at ~290, 0.04 survives to Ueff=1.0).  Use cfl<=0.04 for the
    Re=9e6 production run; cfl=0.20 is fine for low-Re validation grids."""
    dt = 5e-2
    for blk in blocks:
        max_inv = CG.c_max_inv_h2(blk)
        coeff_max = blk._nu
        if getattr(blk, "_sa_on", False):
            coeff_max = blk._nu + getattr(blk, "_nu_t_cap", 10.0 * blk._nu)
        # diffusion limit
        dt = min(dt, cfl / (coeff_max * max_inv))
        # convective CFL on the min edge
        hmin = 1.0 / np.sqrt(max_inv)
        dt = min(dt, cfl * hmin / max(U, 1e-12))
        if scheme == "central" and nu_hyp > 0.0:
            dt = min(dt, 0.10 * hmin ** 4 / nu_hyp)
    return dt


def build_poissons(blocks, dt, reg_r=0.0, mode="adjoint"):
    """Build the pressure-Poisson solver for every block (recv already rebuilt by
    Overset).  Uses the iterative AMG-backed variant for large grids."""
    poissons = []
    for blk in blocks:
        N = blk.ni * blk.nj
        if N <= 45000:
            poissons.append(CG.build_poisson(blk, blk.recv, reg_r=reg_r, dt=dt,
                                             mode=mode))
        else:
            poissons.append(CG.build_poisson_iterative(blk, blk.recv, reg_r=reg_r,
                                                       dt=dt, mode=mode))
    return poissons


def multiblock_forces(blocks, U=1.0, aoa_deg=0.0):
    """Sum the raw wall forces over all blocks and rotate into the incoming-flow
    frame.  Cd/Cl are normalised by 0.5*U^2*Lref_main (the main chord =
    reference length), so the three-element and single-element cases are directly
    comparable."""
    aoa = np.deg2rad(aoa_deg)
    Fxt = 0.0
    Fyt = 0.0
    for blk in blocks:
        _, _, Fx, Fy = CG.c_forces(blk, blk._nu, U)
        Fxt += Fx
        Fyt += Fy
    Lref = max(b.Lref for b in blocks)          # main chord (reference)
    Fdrag = Fxt * np.cos(aoa) + Fyt * np.sin(aoa)
    Flift = -Fxt * np.sin(aoa) + Fyt * np.cos(aoa)
    q = 0.5 * U * U * Lref
    return Fdrag / q, Flift / q, Fxt, Fyt, Lref


def _trip_mask_from_geom(el, blk, frac=0.07, band=0.02, jt=None):
    """Boolean (ni,nj) mask of the near-wall cells whose chordwise station (LE->TE,
    measured from the element's minimum-x point) lies in [frac, frac+band] of the
    local element chord.  These cells get nu_tilde forced large EACH step so the
    SA trip term (ft1) pins transition at that location.  jt = number of near-wall
    j-layers covered by the strip."""
    X = el["X"]; ni, nj = blk.ni, blk.nj
    xs = X[:, 0]                                  # wall node x (node line j=0)
    iLE = int(np.argmin(xs)); xLE = xs[iLE]
    Lref = max(abs(blk.Lref), 1e-6)
    chord_pos = (xs - xLE) / Lref
    on_strip = (chord_pos >= frac) & (chord_pos <= frac + band)
    if jt is None:
        jt = max(2, min(nj // 4, 10))
    mask = np.zeros((ni, nj), dtype=bool)
    # on_strip has length ni+1 (node line); cell i uses nodes i,i+1 -> drop last
    mask[on_strip[:-1], :jt] = True
    return mask


def run_multiblock(geom_dict, nsteps=300, aoa_deg=4.0, Re=100.0, scheme="ppm",
                   cb="pair", reg_r=0.0, n_fringe=1, nu_hyp=0.0, mode="rhie2",
                   U=1.0, field_every=0, out="te_run", ramp=50, save_every=0,
                   record_every=100, Rhole=0.08, turbulent=False,
                   nu_tilde_inf=None, sa_scheme="upwind1", nu_t_cap=None,
                   trip=False, trip_strip_frac=0.07, trip_band=0.02,
                   wall_fn=True, cfl=0.10, own_margin_frac=1.5,
                   nu_hyp_local=0.0, j_floor=0.0, p_exact=False,
                   proj_ucap=0.0, p_cap=0.0,                    p_scale=0.0, p_scale_ref=0.1,
                   p_wall_nojump=False, p_phys=False, p_phys_k=1.0,
                   p_phys_pred=True, no_wake_sym=False):
    """Drive the multi-block overset solve.  Returns a dict with the force
    histories and the final blocks/overset (for plotting or restart)."""
    aoa = np.deg2rad(aoa_deg)
    cu0, cv0 = U * np.cos(aoa), U * np.sin(aoa)
    # Never let the ramp exceed the run so short validation runs still record.
    ramp = min(ramp, max(1, nsteps - 1))
    # Initialise the velocity to the ramped free-stream speed at step 1 so the
    # interior matches the far-field BC (avoids the interior/boundary mismatch
    # that blows the explicit projection up at high Re -- see make_te_blocks).
    U_init = U * min(1.0, 1.0 / ramp)
    blocks, oblocks = make_te_blocks(geom_dict, aoa_deg, U=U, Re=Re,
                                  turbulent=turbulent, nu_tilde_inf=nu_tilde_inf,
                                  sa_scheme=sa_scheme, nu_t_cap=nu_t_cap,
                                  trip=trip, U_init=U_init, j_floor=j_floor)
    # Exact-projection + velocity-correction cap (stability for the exact solve).
    for _blk in blocks:
        _blk._p_exact = p_exact
        _blk._proj_ucap = proj_ucap
        _blk._p_cap = p_cap
        _blk._p_scale = p_scale
        _blk._p_scale_ref = p_scale_ref
        _blk._p_scale_mode = "wall"
        _blk._p_wall_nojump = p_wall_nojump
        _blk._p_phys = p_phys
        _blk._p_phys_k = p_phys_k
        _blk._p_phys_pred = p_phys_pred
        if p_phys:
            # Min edge per block -> wall CFL inflation factor f = k*0.5*U*dt/hmin
            # so the EXACT-projection pressure (structurally inflated by 1/CFL)
            # can be de-inflated to physical O(rho*U^2) for the force & predictor.
            # k is a calibration multiplier (default 1.0 -> f=0.5*U*dt/hmin).
            _blk._hmin = 1.0 / np.sqrt(CG.c_max_inv_h2(_blk))
    # Pre-compute the trip-strip mask (fixed transition location) for each block.
    # Stored on the block so it survives the run and is re-applied every step.
    for bi, blk in enumerate(blocks):
        if trip and getattr(blk, "_sa_on", False):
            blk._trip_mask = _trip_mask_from_geom(geom_dict["elements"][bi], blk,
                                                  frac=trip_strip_frac,
                                                  band=trip_band)
        else:
            blk._trip_mask = None
    ov = Overset(oblocks, n_fringe=n_fringe, U=U, aoa_deg=aoa_deg, Rhole=Rhole,
                 own_margin_frac=own_margin_frac)
    dt = compute_dt(blocks, U=U, scheme=scheme, nu_hyp=nu_hyp, cfl=cfl)
    poissons = build_poissons(blocks, dt, reg_r=reg_r, mode=mode)
    # Precompute per-block dt-bounding quantities for the ADAPTIVE dt used below.
    # In adjoint mode the Poisson operator L = c_div o c_grad is dt-INDEPENDENT;
    # only blk._rAU = dt/J carries dt, so we can shrink dt every step (to hold the
    # local convective CFL <= cfl even when the overset fringe injects high-velocity
    # donor values into the fine near-wall / near-TE cells) by simply rewriting
    # _rAU -- the Poisson matrix built once stays valid for any dt.
    _blk_maxinv = [CG.c_max_inv_h2(b) for b in blocks]
    _blk_coeff = []
    for b in blocks:
        cm = b._nu
        if getattr(b, "_sa_on", False):
            cm = b._nu + getattr(b, "_nu_t_cap", 10.0 * b._nu)
        _blk_coeff.append(cm)
    hmin_global = 1.0 / np.sqrt(max(_blk_maxinv))
    # Pre-compute per-block biharmonic eigenvalue bound kmax4(i,j) for the LOCAL,
    # grid-scale-limited 4th-order dissipation (nu_hyp_local).  This damps ONLY the
    # grid-scale checkerboard that the explicit collocated projection leaves
    # divergence-free and that central/PPM convection amplifies at high Re -- it is
    # what makes Re=9e6 wall-bounded flow stable here (see c_hypervis_local).
    if nu_hyp_local and nu_hyp_local > 0.0:
        _kmax4 = [CG.c_kmax4(b) for b in blocks]
        print(f"[mb] local 4th-order dissipation enabled: C_hyp={nu_hyp_local}")
    else:
        _kmax4 = None
    # Wake-centreline SYMMETRY mask per block: the j=0 inner line is the body arc
    # (wall) over the element and the WAKE CENTRELINE (a symmetry plane) on the
    # upstream/downstream portions.  The symmetry plane needs zero normal velocity
    # (no penetration) at every solved centreline cell; without it the collocated
    # div carried a spurious normal flux through the centreline that the projection
    # could not eliminate, seeding the wake jet.  Restricted to solved (non-recv)
    # cells so we never clobber an overset donor / farfield value.  Computed once
    # (wall is fixed); AND-ed with the per-step recv in the loop.
    _cl_base = []
    for blk in blocks:
        j0 = np.zeros((blk.ni, blk.nj), dtype=bool); j0[:, 0] = True
        _cl_base.append(j0 & (~blk.wall))

    print(f"[mb] blocks={len(blocks)} aoa={aoa_deg} Re={Re} scheme={scheme} "
          f"mode={mode} reg_r={reg_r} n_fringe={n_fringe}")
    for bi, blk in enumerate(blocks):
        nm = geom_dict["elements"][bi]["name"]
        print(f"  {nm}: ni={blk.ni} nj={blk.nj} "
              f"solved={int((~blk.recv).sum())} recv={int(blk.recv.sum())} "
              f"nu={blk._nu:.3e}")
    print(ov.summary())
    print(f"  dt={dt:.3e}")

    cd_hist, cl_hist, t_hist = [], [], []
    t_acc = 0.0
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        cu, cv = Ueff * np.cos(aoa), Ueff * np.sin(aoa)
        # ADAPTIVE explicit dt: cap the LOCAL convective CFL at `cfl` using the
        # current max velocity (so a high-velocity overset fringe / suction peak
        # cannot push a fine cell past the stability limit).  The Poisson operator
        # is dt-independent in adjoint mode, so we only rewrite blk._rAU = dt/J.
        umax = max(float(np.sqrt(b.u ** 2 + b.v ** 2).max()) for b in blocks)
        dt = min(5e-2, cfl * hmin_global / max(umax, U),
                 min(cfl / (cm * mi) for cm, mi in zip(_blk_coeff, _blk_maxinv)))
        for b in blocks:
            # Adaptive dt only rewrites _rAU in adjoint mode (the Poisson matrix
            # is dt-independent there).  Rhie-Chow carries its own spatially
            # varying rAU (built once) -- do NOT overwrite it.
            if getattr(b, "_poisson_mode", "adjoint") == "adjoint":
                b._rAU = dt * np.ones_like(b.J)
        # 1. overset exchange: hole/fringe <- donor, farfield <- free stream
        ov.exchange(U=Ueff, aoa_deg=aoa_deg)
        for bi, blk in enumerate(blocks):
            m = ov.masks[bi]
            wall = blk.wall
            far = m["farfield"]
            # 2. predictor (explicit)
            # Global CONSTANT diffusion cap = coeff_max = nu (+ nu_t_cap when
            # turbulent).  dt was derived so coeff_max is stable on the finest
            # (min-edge) cell, so the actual coefficient nu+nu_t (<= coeff_max,
            # since nu_t is capped) is never capped and therefore stays SMOOTH --
            # no grad(coeff) jumps, which is what made the per-cell field cap
            # destabilise.  Applied to both laminar and turbulent momentum.
            blk._diff_cap = blk._nu + getattr(blk, "_nu_t_cap", 0.0)
            if getattr(blk, "_sa_on", False):
                # turbulent viscosity from the current nu_tilde -> effective
                # (molecular + turbulent) viscosity used by c_diffusion
                nu_t = CG.sa_nut(blk.nu_tilde, blk._nu)
                blk._nu_eff_max = blk._diff_cap
                blk._nu_eff = blk._nu + nu_t
            duc, dvc = CG.c_conv(blk, blk.u, blk.v, scheme=scheme)
            du, dv = CG.c_diffusion(blk, blk.u, blk.v)
            # LOCAL, grid-scale-limited 4th-order DISSIPATION (correct - sign):
            # damps ONLY the grid-scale checkerboard the collocated projection
            # leaves divergence-free; molecular+eddy viscosity ~1e-7 is far too
            # weak to do this at Re=9e6, so without it the run diverges.  Applied
            # for every scheme (not just "central"), since the instability is in
            # the pressure-velocity decoupling, not the convection scheme.
            if _kmax4 is not None:
                hdu, hdv = CG.c_hypervis_local(blk, blk.u, blk.v,
                                                nu_hyp_local, dt, _kmax4[bi])
                du = du + hdu; dv = dv + hdv
            ustar = blk.u + dt * (-duc + du)
            vstar = blk.v + dt * (-dvc + dv)
            # SIMPLE momentum coupling (physical-pressure mode).  The carried
            # blk.p is the EXACT projection pressure, structurally inflated by
            # 1/CFL_wall = 2*h/(rho*U*dt) (rho=1).  The de-inflation factor
            # f = 0.5*U*dt/hmin (a SMALL number ~0.05 here) recovers the physical
            # O(rho*U^2) pressure p_phys = f*blk.p.  The predictor term
            # -dt*grad(p_phys) = -dt*f*grad(blk.p) is then O(CFL*U) -- a
            # well-balanced, stable contribution that carries physical pressure
            # into the momentum step so the projected correction p' becomes a
            # small SUPERPOSITION on a physical p^n (SIMPLE), not the sole
            # pressure.  The velocity-correction gradient in project() stays
            # inflated (uses blk.p as-is) so divergence is still removed exactly.
            if getattr(blk, "_p_phys", False):
                blk._p_phys_f = (0.5 * U * dt / max(getattr(blk, "_hmin", dt),
                                                   1e-30)) * getattr(blk,
                                                                     "_p_phys_k", 1.0)
                if getattr(blk, "_p_phys_pred", True):
                    _gpx, _gpy = CG.c_grad(blk, blk.p)
                    ustar = ustar - dt * (_gpx * blk._p_phys_f)
                    vstar = vstar - dt * (_gpy * blk._p_phys_f)
            ustar[wall] = 0.0
            vstar[wall] = 0.0
            # ONLY the far field is clamped to free stream; the hole/fringe keep
            # the donor-interpolated velocity filled by exchange() -- overwriting
            # them with the free stream (the old single-block pattern) destroys
            # the overset velocity coupling and makes the rig drift.
            ustar[far] = cu
            vstar[far] = cv
            blk.u, blk.v = ustar, vstar
            # wall BC: no-slip (low-Re) OR log-law wall-function (high-Re, y+~30-60
            # grid).  The wall function sets the wall-cell tangential velocity to
            # the log-law value (slipping) and zeroes the normal component; the
            # shear is then recovered by c_forces from the cell gradient.
            if wall_fn:
                CG.c_wall_function(blk, blk._nu)
            else:
                blk.u[wall] = 0.0
                blk.v[wall] = 0.0
            # wake-centreline symmetry: zero the normal (v) velocity at solved
            # j=0 non-wall cells before projection so the divergence field the
            # projection sees already respects the symmetry plane.
            cl = _cl_base[bi] & (~blk.recv)
            if not no_wake_sym:
                blk.v[cl] = 0.0
            # 3. projection (Chorin) -- unchanged single-block solver
            CG.project(blk, dt, poissons[bi])
            if cb != "none":
                blk.p = CG.c_checkerboard_project(blk, blk.p, mode=cb)
                blk.u = CG.c_checkerboard_project(blk, blk.u, mode=cb)
                blk.v = CG.c_checkerboard_project(blk, blk.v, mode=cb)
            # re-impose wake-centreline symmetry on the projected (post-checker
            # board) velocity so the final field is divergence-free AND symmetric.
            cl = _cl_base[bi] & (~blk.recv)
            if not no_wake_sym:
                blk.v[cl] = 0.0
            # 3b. SA nu_tilde transport (uses the projected, divergence-free
            #     velocity).  Boundaries re-imposed here; hole/fringe are
            #     overwritten by the overset exchange() below.
            if getattr(blk, "_sa_on", False):
                rhs, _ = CG.sa_rhs(blk, blk.u, blk.v, blk.nu_tilde, blk._nu,
                                   scheme=blk._sa_scheme, d=None,
                                   nu_eff_max=getattr(blk, "_nu_eff_max", None),
                                   trip=getattr(blk, "_sa_trip", False))
                blk.nu_tilde = blk.nu_tilde + dt * rhs
                blk.nu_tilde = np.maximum(blk.nu_tilde, 0.0)
                # cap nu_tilde so nu_t <= nu_t_cap (keeps coeff <= coeff_max and
                # the explicit step stable on the finest cell)
                blk.nu_tilde = np.minimum(blk.nu_tilde, blk._nu_t_cap)
                blk.nu_tilde[wall] = 0.0
                blk.nu_tilde[far] = blk._nu_tilde_inf
                # fixed-transition trip strip: force nu_tilde large at the strip
                # every step so the SA ft1 term pins transition at x/c.  Applied
                # AFTER the cap (== nu_t_cap) and the wall/far BCs.
                if getattr(blk, "_trip_mask", None) is not None:
                    blk.nu_tilde[blk._trip_mask] = blk._nu_t_cap
        # 4. re-impose overset BCs after projection (hole/fringe <- donor,
        #    farfield <- free stream) and pin interface pressure continuity so
        #    the two-way Chimera seam does not round-trip a residual divergence.
        ov.exchange(U=Ueff, aoa_deg=aoa_deg)
        for blk in blocks:
            if wall_fn:
                CG.c_wall_function(blk, blk._nu)
            else:
                blk.u[blk.wall] = 0.0
                blk.v[blk.wall] = 0.0
        ov.exchange_pressure()

        if not all(np.isfinite(b.u).all() and np.isfinite(b.v).all()
                   and np.isfinite(b.p).all() for b in blocks):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp and step % record_every == 0:
            cd, cl, _, _, _ = multiblock_forces(blocks, U=U, aoa_deg=aoa_deg)
            t_acc += dt
            t = t_acc
            cd_hist.append(cd)
            cl_hist.append(cl)
            t_hist.append(t)
            if step % (record_every * 5) == 0 or step == nsteps:
                um = 0.0
                loc = (0, 0, 0)
                for bi, b in enumerate(blocks):
                    sp = float(np.sqrt(b.u ** 2 + b.v ** 2).max())
                    if sp > um:
                        um = sp
                        kk = int(np.argmax(b.u ** 2 + b.v ** 2))
                        loc = (bi, kk // b.u.shape[1], kk % b.u.shape[1])
                names = [b.name for b in blocks]
                print(f"  step{step:5d} t={t:7.3f} Cd={cd:+.4f} "
                      f"Cl={cl:+.4f} |u|max={um:.3f} @"
                      f"{names[loc[0]]}({loc[1]},{loc[2]})")
        if field_every > 0 and (step % field_every == 0 or step == nsteps):
            _render_multiblock(blocks, geom_dict, out, step, aoa_deg)
        if save_every > 0 and step % save_every == 0:
            d = dict(step=step, dt=dt, aoa=aoa_deg)
            for bi, b in enumerate(blocks):
                d[f"u{bi}"] = b.u
                d[f"v{bi}"] = b.v
                d[f"p{bi}"] = b.p
            np.savez(f"/workspace/{out}_ckpt.npz", **d)

    if nan:
        print("RESULT: NaN")
        return dict(nan=True, cd=cd_hist, cl=cl_hist, t=t_hist, blocks=blocks,
                    ov=ov)
    cd = np.array(cd_hist)
    cl = np.array(cl_hist)
    t = np.array(t_hist)
    if cd.size == 0:
        cd_mean = cl_mean = float("nan")
        t0 = 0.0
        print(f"\nRESULT multiblock aoa={aoa_deg} Re={Re} mode={mode}")
        print("  (no force records collected -- run too short)")
    else:
        n0 = max(1, len(cd) // 3)
        if n0 >= len(cd):
            n0 = max(0, len(cd) - 1)
        cd_mean = float(np.mean(cd[n0:]))
        cl_mean = float(np.mean(cl[n0:]))
        t0 = t[n0]
        print(f"\nRESULT multiblock aoa={aoa_deg} Re={Re} mode={mode}")
        print(f"  Cd_mean={cd_mean:.4f}  Cl_mean={cl_mean:+.4f}  "
              f"(window t>={t0:.3f})")
    return dict(nan=False, cd=cd_hist, cl=cl_hist, t=t_hist, cd_mean=cd_mean,
                cl_mean=cl_mean, blocks=blocks, ov=ov, dt=dt)


def _render_multiblock(blocks, geom_dict, out, step, aoa_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    fig, axes = plt.subplots(1, len(blocks), figsize=(6 * len(blocks), 5))
    if len(blocks) == 1:
        axes = [axes]
    for ax, blk, el in zip(axes, blocks, geom_dict["elements"]):
        xc, yc = blk.xc, blk.yc
        xmin, xmax = xc.min(), xc.max()
        ymin, ymax = yc.min(), yc.max()
        gx = np.linspace(xmin, xmax, 220)
        gy = np.linspace(ymin, ymax, 160)
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([xc.ravel(), yc.ravel()])
        spd = np.sqrt(blk.u ** 2 + blk.v ** 2)
        zu = griddata(pts, spd.ravel(), (GX, GY), method="linear")
        im = ax.imshow(zu, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap="viridis", vmin=0, vmax=1.4)
        fig.colorbar(im, ax=ax, label="|u|")
        # recv mask outline (hole+fringe+farfield)
        recv = blk.recv
        ax.scatter(xc[recv].ravel(), yc[recv].ravel(), s=2, c="red", alpha=0.4,
                   marker=".", label="recv (hole/fringe/far)")
        ax.plot(xc[:, 0], yc[:, 0], "k-", lw=1.2, label="wall")
        ax.set_aspect("equal")
        ax.set_title(f"{el['name']} |u| (step {step})")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    fname = f"/workspace/{out}_step{step:05d}.png"
    fig.savefig(fname, dpi=120)
    plt.close()
    print(f"  rendered {fname}")


if __name__ == "__main__":
    te = make_three_element_cgrid()
    run_multiblock(te, nsteps=300, aoa_deg=4.0, Re=100.0, scheme="ppm",
                   cb="pair", reg_r=0.0, n_fringe=1, mode="adjoint",
                   ramp=50, field_every=300)
