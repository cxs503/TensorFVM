"""Multi-block overset (Chimera) coupling for the curvilinear FVM solver.

The single-block operators (curve_grid.c_div/c_grad/c_lap/project, and the torch
siblings) are per-block and cross-block-free.  This module adds the ONLY missing
piece for a three-element (slat + main + flap) solve: the inter-block coupling.

Design (mirrors the proven overset_cylinder_fvm.py prototype, generalised from
regular/polar grids to arbitrary curvilinear C-grids):

  * Each block keeps its own body-fitted C-grid and its OWN wall (the body arc).
  * A block's `recv` boundary is rebuilt to be

        recv = HOLE | FRINGE | FARFIELD

    so the EXISTING single-block solver runs UNCHANGED on each block (it always
    solved only ~recv, with p=0 at the recv boundary -- the velocity-only overset
    coupling the prototype validated).  No operator or solver is touched.

  * HOLE     : cells owned by ANOTHER block (that block's body is nearer), i.e.
               ceded -- never solved, velocity held (donor / freestream).
  * FRINGE   : a thin (n_fringe-cell) band of THIS block's own cells adjacent to a
               hole -- the overset interface, velocity Dirichlet from the donor.
  * FARFIELD : this block's outer boundary cells that no other block covers --
               clamped to the free stream.
  * SOLVED   : everything else (own wall + interior); the near-body region THIS
               block is responsible for.

Ownership uses "nearest body wall": a point belongs to the element whose surface
is closest.  This is the robust, general replacement for the prototype's analytic
polar/Cartesian sampling.

Donor location for arbitrary C-grids uses a cKDTree of donor cell centres to get
candidates, then an exact inverse-bilinear map inside the candidate quad; the
interpolation uses the standard cell-centred bilinear weights.

This module is numpy/CPU and used only at SETUP time (the mesh is static): it
produces per-block boolean masks + a cached donor table that the time loop (both
numpy and torch) consumes every step.
"""
import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def cell_centers(X, Y):
    """(ni, nj) cell-centre coordinates from the (ni+1, nj+1) node arrays."""
    xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    return xc, yc


def wall_points(X, Y, wall):
    """Cell centres of the no-slip wall (j=0 arcs) -- the samples of the body
    surface used for nearest-body ownership."""
    ni = X.shape[0] - 1
    idx = np.nonzero(wall[:, 0])[0]
    bx, by = [], []
    for i in idx:
        bx.append(0.25 * (X[i, 0] + X[i + 1, 0] + X[i + 1, 1] + X[i, 1]))
        by.append(0.25 * (Y[i, 0] + Y[i + 1, 0] + Y[i + 1, 1] + Y[i, 1]))
    return np.column_stack([bx, by]), idx


def inv_bilinear(px, py, qx, qy):
    """Inverse bilinear: local coords (xi, eta) of (px,py) inside the quad whose
    corners are qx/qy = [Q00, Q10, Q11, Q01] (grid order (i,j),(i+1,j),(i+1,j+1),
    (i,j+1)).  Returns (xi, eta) with both in [0,1] if inside, else None.

    Solves  d = xi*a + eta*b + xi*eta*c  (d=P-Q00, a=Q10-Q00, b=Q01-Q00,
    c=Q11-Q10-Q01+Q00)  via the standard closed-form quadratic in eta."""
    x0, x1, x2, x3 = qx
    y0, y1, y2, y3 = qy
    ax = x1 - x0; ay = y1 - y0
    bx = x3 - x0; by = y3 - y0
    cx = (x2 - x1) - (x3 - x0); cy = (y2 - y1) - (y3 - y0)
    dx = px - x0; dy = py - y0

    def cr(u, v, w, z):
        return u * z - v * w

    ab = cr(ax, ay, bx, by)
    cb = cr(cx, cy, bx, by)
    ca = cr(cx, cy, ax, ay)
    db = cr(dx, dy, bx, by)
    da = cr(dx, dy, ax, ay)
    A = ab * cb
    B = ab * ab + da * cb - ca * db
    C = da * ab
    tol = 1e-14 * (abs(ab) + abs(cb) + abs(ax) + abs(ay) + 1.0)
    if abs(A) < tol:
        # degenerate (parallelogram): linear in eta
        if abs(B) < tol:
            return None
        eta = -C / B
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return None
        sq = np.sqrt(disc)
        eta = None
        for cand in ((-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)):
            if -1e-9 <= cand <= 1.0 + 1e-9:
                eta = cand; break
        if eta is None:
            return None
    den = ab + eta * cb
    if abs(den) < tol:
        return None
    xi = db / den
    if not (-1e-7 <= xi <= 1.0 + 1e-7 and -1e-7 <= eta <= 1.0 + 1e-7):
        return None
    xi = float(np.clip(xi, 0.0, 1.0)); eta = float(np.clip(eta, 0.0, 1.0))
    # forward-reconstruction check: a NON-CONVEX quad can yield a spurious root
    # that lands in [0,1]^2 without actually containing P -- reject those.
    fx = x0 + xi * ax + eta * bx + xi * eta * cx
    fy = y0 + xi * ay + eta * by + xi * eta * cy
    scale = abs(px) + abs(py) + abs(ax) + abs(ay) + abs(bx) + abs(by) + 1.0
    if (fx - px) ** 2 + (fy - py) ** 2 > (1e-8 * scale) ** 2:
        return None
    return xi, eta


def _dilate(mask, n):
    """Binary dilation by n 4-neighbour layers (on an (ni,nj) cell mask)."""
    m = mask.copy()
    for _ in range(int(n)):
        p = np.zeros_like(m)
        p[1:, :] |= m[:-1, :]; p[:-1, :] |= m[1:, :]
        p[:, 1:] |= m[:, :-1]; p[:, :-1] |= m[:, 1:]
        m = m | p
    return m


# --------------------------------------------------------------------------- #
# overset system
# --------------------------------------------------------------------------- #
class Overset:
    """Owns the per-block hole/fringe/farfield classification and the cached
    donor-interpolation table for a static multi-block mesh.

    build() consumes a list of block specs (dicts with 'X','Y','wall' and the
    live field arrays 'u','v','p','recv','ni','nj') in place: it WRITES each
    block's new 'recv' mask (= hole|fringe|farfield) and stores its masks.
    overset_exchange() then fills the hole/fringe (donor) and farfield
    (freestream) velocities every time step.
    """

    def __init__(self, blocks, n_fringe=2, U=1.0, aoa_deg=0.0, Rhole=None,
                 own_margin_frac=1.5, wake_hole=True):
        self.blocks = blocks
        self.n_fringe = int(n_fringe)
        self.U = float(U)
        self.aoa = float(np.deg2rad(aoa_deg))
        # wake_hole: an embedded (smaller) element's OWN near-wake -- the cells
        # downstream of its own trailing edge on the j=0 centreline -- is ceded
        # to the background (main) instead of being solved by the small element.
        # WHY: a C-grid element's j=0 line is its wake-cut SYMMETRY plane, so its
        # solved centreline cells must carry v=0.  But the small element is fully
        # embedded in the background, whose flow at the slat's physical location is
        # NOT symmetric (it has v != 0 -- the upwash).  When the small element's
        # centreline extends past its TE into the background's domain, the donor
        # (background) injects v != 0 into the small element's centreline fringe,
        # breaking the symmetry and seeding an O(1) divergence at the centreline
        # that the projection cannot remove (it is re-injected every step) -> the
        # coupled run diverges exponentially at that cell.  Ceding the small
        # element's own near-wake (i > its TE_i) to the background removes the
        # small element's centreline from the overlap region entirely: the small
        # element solves only its body + immediate near-field, and the background
        # carries the wake.  Only applied where a valid donor (hasd) exists, so
        # the largest block (main), which has no donor downstream of its TE, is
        # never affected.
        self.wake_hole = wake_hole
        # Rhole: max distance-to-an-OTHER-wall for a cell to be classed HOLE.
        # Without it, every cell whose nearest wall is another element (which,
        # for these compact rigs with Rf >> element spacing, is the ENTIRE far
        # field of every block) becomes a hole filled by a donor -- so ~60% of
        # each block is donor-driven and the overset seam drifts/NaNs on the
        # (1-dt) residual.  Standard Chimera only cedes cells actually SHADOWED
        # by another body; far-field overlap stays solved (free stream).  Set
        # Rhole to e.g. 0.4*chord to enable this; None keeps the legacy behaviour.
        self.Rhole = Rhole
        # own_margin_frac: a covered cell is ceded to a LARGER covering block only
        # if it lies beyond own_margin_frac * own_chord from its OWN wall -- i.e.
        # the cell's own near-field (high-res slot/LE region) stays solved by the
        # small element, while its far-field overlap with the larger background
        # (main) is ceded to the background.  This stops the small element's huge
        # C-grid (Rf=18*chord, entirely inside the main's domain) from being solved
        # independently and double-covering the main -> the seam discontinuity that
        # was diverging the coupled run.
        self.own_margin_frac = own_margin_frac
        self._build()

    # ---- setup ---------------------------------------------------------- #
    def _build(self):
        nb = len(self.blocks)
        self.fringe_depth = []
        # per-block cell-centre coords (cached; may already exist on the block)
        for b in self.blocks:
            xc, yc = cell_centers(b["X"], b["Y"])
            b["_xc"] = xc; b["_yc"] = yc
            b["_ni"] = xc.shape[0]; b["_nj"] = xc.shape[1]
        # per-block wall sample trees (for ownership)
        self.wall_trees = []
        self.wall_pts = []
        for b in self.blocks:
            wp, _ = wall_points(b["X"], b["Y"], b["wall"])
            self.wall_pts.append(wp)
            self.wall_trees.append(cKDTree(wp))

        # ownership: for each cell of each block, the nearest-wall block id
        self.owner = []
        self.dmin = []
        # per-block distance to OWN wall (for the far-field-cede criterion) and the
        # block reference lengths (for the size hierarchy: small cedes to large).
        self.own_dmin = []
        self.Lref_arr = [float(getattr(b.blk, "Lref", 1.0)) for b in self.blocks]
        for bi, b in enumerate(self.blocks):
            P = np.column_stack([b["_xc"].ravel(), b["_yc"].ravel()])
            dmin = None; own = None
            for k in range(nb):
                d, _ = self.wall_trees[k].query(P)
                if dmin is None:
                    dmin = d; own = np.full(P.shape[0], k, dtype=np.int64)
                else:
                    upd = d < dmin
                    dmin = np.where(upd, d, dmin)
                    own = np.where(upd, k, own)
            self.owner.append(own.reshape(b["_ni"], b["_nj"]))
            self.dmin.append(dmin.reshape(b["_ni"], b["_nj"]))
            # distance from each cell to THIS block's own wall
            d_own, _ = self.wall_trees[bi].query(P)
            self.own_dmin.append(d_own.reshape(b["_ni"], b["_nj"]))

        # donor table: for every cell of block B, the best donor (block,cell,xi,eta)
        # among ALL OTHER blocks -- built ONCE (mesh static).
        self.donor = []                      # list of dicts per block
        for bi, b in enumerate(self.blocks):
            others = [k for k in range(nb) if k != bi]
            if not others:
                self.donor.append(None); continue
            ctrs = []; tags = []; offs = {}
            base = 0
            for k in others:
                ob = self.blocks[k]
                ctrs.append(np.column_stack([ob["_xc"].ravel(), ob["_yc"].ravel()]))
                tags.append(np.full(ob["_xc"].size, k, dtype=np.int64))
                offs[k] = base
                base += ob["_xc"].size
            ctrs = np.vstack(ctrs); tags = np.concatenate(tags)
            tree = cKDTree(ctrs)
            P = np.column_stack([b["_xc"].ravel(), b["_yc"].ravel()])
            K = 12
            _, idxs = tree.query(P, k=min(K, ctrs.shape[0]))
            if idxs.ndim == 1:
                idxs = idxs[:, None]
            ni, nj = b["_ni"], b["_nj"]
            dv = np.zeros(P.shape[0], dtype=bool)
            db_ = np.full(P.shape[0], -1, dtype=np.int64)
            ci_ = np.full(P.shape[0], -1, dtype=np.int64)   # base donor cell (x)
            cj_ = np.full(P.shape[0], -1, dtype=np.int64)   # base donor cell (y)
            cfx = np.zeros(P.shape[0]); cfy = np.zeros(P.shape[0])
            for n in range(P.shape[0]):
                if dv[n]:
                    continue
                px, py = P[n]
                for q in idxs[n]:
                    k = int(tags[q])
                    ob = self.blocks[k]
                    oni, onj = ob["_ni"], ob["_nj"]
                    flat = int(q) - offs[k]          # flat cell index within block k
                    oi, oj = divmod(flat, onj)
                    # keep donors strictly interior so the cell-centred bilinear
                    # (which spans oi-1..oi+1) never leaves the donor grid
                    if oi < 1 or oj < 1 or oi + 1 >= oni or oj + 1 >= onj:
                        continue
                    if oi + 1 >= ob["_ni"] or oj + 1 >= ob["_nj"]:
                        continue
                    qx = [ob["X"][oi, oj], ob["X"][oi + 1, oj],
                          ob["X"][oi + 1, oj + 1], ob["X"][oi, oj + 1]]
                    qy = [ob["Y"][oi, oj], ob["Y"][oi + 1, oj],
                          ob["Y"][oi + 1, oj + 1], ob["Y"][oi, oj + 1]]
                    se = inv_bilinear(px, py, qx, qy)
                    if se is not None:
                        xi, eta = se
                        # node-fractional -> cell-CENTRE-fractional (shift 0.5)
                        ux = oi + xi - 0.5; uy = oj + eta - 0.5
                        i0 = int(np.floor(ux)); i0 = min(max(i0, 0), oni - 2)
                        j0 = int(np.floor(uy)); j0 = min(max(j0, 0), onj - 2)
                        dv[n] = True; db_[n] = k; ci_[n] = i0; cj_[n] = j0
                        cfx[n] = ux - i0; cfy[n] = uy - j0
                        break
            self.donor.append(dict(valid=dv.reshape(ni, nj), blk=db_.reshape(ni, nj),
                                   i=ci_.reshape(ni, nj), j=cj_.reshape(ni, nj),
                                   fx=cfx.reshape(ni, nj), fy=cfy.reshape(ni, nj)))

        # classification -> rebuilt recv mask per block, in place.
        # KEY: a cell is only HOLE if it is ALSO covered by another grid (hasd),
        # so every hole cell has a donor -- otherwise a "owned-by-other but not
        # covered" cell (in an inter-element gap) would be pinned to free stream
        # and inject a bogus patch into the block's near field.
        #
        # Hole cut (size-hierarchical Chimera):
        #   hole = hasd & ~wall & ( (owner != bi)                          # (a) nearest wall is another block  -> standard slot/near-field cut
        #                           | (cover_L > own_L                      # (b) a LARGER block covers this cell
        #                              & own_dmin > own_margin) )           #     and it is far from THIS block's wall -> far-field overlap ceded to background
        # (a) cedes each cell to the block whose wall is nearest (the classic
        #     Chimera cut, so the slot is solved by exactly one block).
        # (b) additionally cedes a small element's far-field OVERLAP with the
        #     larger background (main) to that background, while keeping the
        #     small element's own high-res near-field solved.  Without (b) the
        #     small element's whole C-grid (Rf=18*chord, inside the main) is
        #     solved independently -> double-covers the main and the seam
        #     discontinuity diverges the coupled run.
        # The largest block (main) never triggers (b) (cover_L can't exceed its
        # own Lref) and wins (a) only where its wall is nearest, so it never
        # cedes its own region.
        self.masks = []
        for bi, b in enumerate(self.blocks):
            ni, nj = b["_ni"], b["_nj"]
            owner = self.owner[bi]
            wall = b["wall"]
            own_L = self.Lref_arr[bi]
            own_margin = self.own_margin_frac * own_L
            orig_recv = b["recv"].copy()          # outer boundary BEFORE overwrite
            dn = self.donor[bi]
            hasd = dn["valid"] if dn is not None else np.zeros_like(owner, dtype=bool)
            # single-block (no donor) -> cover_L all zero, so no hole/fringe is
            # carved and the block is solved in full (used for single-block tests).
            _donor_L = (np.asarray(self.Lref_arr)[dn["blk"]]
                        if dn is not None else np.zeros_like(owner, dtype=float))
            cover_L = np.where(hasd, _donor_L, 0.0)  # Lref of covering donor block
            hole = hasd & (~wall) & ((owner != bi)
                                   | ((cover_L > own_L) & (self.own_dmin[bi] > own_margin)))
            # backward-compat: if Rhole given, also require the cell to be within
            # Rhole of an OTHER wall (constrains the (a) branch to near-body cells).
            if self.Rhole is not None:
                hole = hole & (self.dmin[bi] < self.Rhole)
            # wake_hole: cede this element's OWN near-wake (downstream of its TE
            # on the j=0 centreline) to the background.  Only where a valid donor
            # exists, so the largest block (no donor downstream of its TE) is
            # untouched.  See __init__ docstring for the divergence mechanism.
            if self.wake_hole:
                wi = np.nonzero(wall[:, 0])[0]
                te_i = int(wi.max()) if wi.size else ni - 1
                le_i = int(wi.min()) if wi.size else 0
                igrid = np.arange(ni).reshape(-1, 1)
                # Cede the embedded element's ENTIRE near-wake -- both upstream
                # (i < LE_i) and downstream (i > TE_i) of its body on the j=0
                # centreline -- to the background.  The element then solves only
                # its body + immediate near-field; the background carries the wake.
                # BOTH ends must be ceded: the upstream centreline sits in the
                # background's upwash (v != 0) just as the downstream one does, and
                # either end left solved seeds the same centreline-divergence mode
                # (verified: ceding only the downstream end moved the blow-up to the
                # upstream LE / near-field seam).  Only where a larger block covers
                # the cell (cover_L > own_L), so the largest block is never holed.
                hole = hole | (hasd & (~wall) & (cover_L > own_L)
                               & ((igrid < le_i) | (igrid > te_i)))
            band = _dilate(hole, self.n_fringe)
            fringe = (~hole) & (~wall) & hasd & (band | orig_recv)
            farfield = orig_recv & (~hole) & (~fringe)
            recv = hole | fringe | farfield
            solved = ~recv
            # fringe-depth (1 = adjacent to hole .. n_fringe = outermost) for the
            # sponge relaxation in exchange(): the inner fringe (closest to the
            # solved cells) is only weakly updated each step so the overset
            # velocity feedback is damped instead of round-tripping and growing.
            fd = np.zeros((ni, nj), dtype=float)
            if self.n_fringe > 1:
                prev = hole
                for k in range(1, self.n_fringe + 1):
                    cur = _dilate(hole, k)
                    fd[(cur & (~prev)) & fringe] = float(k)
                    prev = cur
            else:
                fd[fringe] = 1.0
            b["recv"] = recv
            b["hole"] = hole
            b["fringe"] = fringe
            b["farfield"] = farfield
            self.fringe_depth.append(fd)
            self.masks.append(dict(hole=hole, fringe=fringe, farfield=farfield,
                                   recv=recv, solved=solved))

    # ---- per-step exchange --------------------------------------------- #
    def exchange(self, U=None, aoa_deg=None, sponge_a0=0.3):
        """Fill hole/fringe velocities from the donors and farfield from the free
        stream.  Call this BEFORE each block's predictor/projection.

        Sponge: fringe cells are RELAXED toward the donor value
            u_fringe = (1 - a) * u_old + a * u_donor
        with a ramping from `sponge_a0` (inner fringe, next to the solved cells)
        to 1.0 (outer fringe, next to the far field).  Holes (fully inside the
        donor) and the far field are hard-set.  The partial inner-fringe update
        damps the overset velocity feedback so the two-way coupling does not
        round-trip and grow (the flap-velocity monotonic drift observed without
        it).  The relaxation uses the start-of-step snapshot for u_old."""
        U = self.U if U is None else U
        aoa = self.aoa if aoa_deg is None else np.deg2rad(aoa_deg)
        cu, cv = U * np.cos(aoa), U * np.sin(aoa)
        # snapshot donors first (so a block with several acceptors reads a
        # consistent donor state)
        snap = [(b["u"].copy(), b["v"].copy()) for b in self.blocks]
        nf = max(1, self.n_fringe - 1)
        for bi, b in enumerate(self.blocks):
            m = self.masks[bi]
            dn = self.donor[bi]
            # farfield -> free stream (hard)
            b["u"][m["farfield"]] = cu
            b["v"][m["farfield"]] = cv
            # hole -> donor bilinear (hard, fully inside the donor)
            if dn is not None and m["hole"].any():
                ii, jj = np.nonzero(m["hole"])
                for (i, j) in zip(ii, jj):
                    if not dn["valid"][i, j]:
                        b["u"][i, j] = cu; b["v"][i, j] = cv
                        continue
                    k = dn["blk"][i, j]; i0 = dn["i"][i, j]; j0 = dn["j"][i, j]
                    x = dn["fx"][i, j]; e = dn["fy"][i, j]
                    ob = self.blocks[k]; su, sv = snap[k]
                    i1 = i0 + 1; j1 = j0 + 1
                    w00 = (1 - x) * (1 - e); w10 = x * (1 - e)
                    w01 = (1 - x) * e;       w11 = x * e
                    b["u"][i, j] = (w00 * su[i0, j0] + w10 * su[i1, j0]
                                    + w01 * su[i0, j1] + w11 * su[i1, j1])
                    b["v"][i, j] = (w00 * sv[i0, j0] + w10 * sv[i1, j0]
                                    + w01 * sv[i0, j1] + w11 * sv[i1, j1])
            # fringe -> donor bilinear, RELAXED by the sponge coefficient
            if dn is not None and m["fringe"].any():
                fd = self.fringe_depth[bi]
                ii, jj = np.nonzero(m["fringe"])
                for (i, j) in zip(ii, jj):
                    if not dn["valid"][i, j]:
                        b["u"][i, j] = (1.0 - sponge_a0) * snap[bi][0][i, j] \
                            + sponge_a0 * cu
                        b["v"][i, j] = (1.0 - sponge_a0) * snap[bi][1][i, j] \
                            + sponge_a0 * cv
                        continue
                    k = dn["blk"][i, j]; i0 = dn["i"][i, j]; j0 = dn["j"][i, j]
                    x = dn["fx"][i, j]; e = dn["fy"][i, j]
                    ob = self.blocks[k]; su, sv = snap[k]
                    i1 = i0 + 1; j1 = j0 + 1
                    w00 = (1 - x) * (1 - e); w10 = x * (1 - e)
                    w01 = (1 - x) * e;       w11 = x * e
                    dnu = (w00 * su[i0, j0] + w10 * su[i1, j0]
                           + w01 * su[i0, j1] + w11 * su[i1, j1])
                    dnv = (w00 * sv[i0, j0] + w10 * sv[i1, j0]
                           + w01 * sv[i0, j1] + w11 * sv[i1, j1])
                    depth = fd[i, j]
                    a = sponge_a0 + (1.0 - sponge_a0) * ((depth - 1.0) / nf) ** 2 \
                        if nf > 0 else 1.0
                    b["u"][i, j] = (1.0 - a) * snap[bi][0][i, j] + a * dnu
                    b["v"][i, j] = (1.0 - a) * snap[bi][1][i, j] + a * dnv
        # ---- optional SA working variable nu_tilde (mirrors the velocity fill) ----
        if all(b.get("nu_tilde") is not None for b in self.blocks):
            snap_nu = [b["nu_tilde"].copy() for b in self.blocks]
            for bi, b in enumerate(self.blocks):
                m = self.masks[bi]; dn = self.donor[bi]
                inf = getattr(b.blk, "_nu_tilde_inf", 0.0)
                b["nu_tilde"][m["farfield"]] = inf
                if dn is not None and m["hole"].any():
                    for (i, j) in zip(*np.nonzero(m["hole"])):
                        if not dn["valid"][i, j]:
                            b["nu_tilde"][i, j] = inf; continue
                        k = dn["blk"][i, j]; i0 = dn["i"][i, j]; j0 = dn["j"][i, j]
                        x = dn["fx"][i, j]; e = dn["fy"][i, j]; sn = snap_nu[k]
                        i1 = i0 + 1; j1 = j0 + 1
                        w00 = (1-x)*(1-e); w10 = x*(1-e); w01 = (1-x)*e; w11 = x*e
                        b["nu_tilde"][i, j] = (w00*sn[i0, j0] + w10*sn[i1, j0]
                                              + w01*sn[i0, j1] + w11*sn[i1, j1])
                if dn is not None and m["fringe"].any():
                    fd = self.fringe_depth[bi]
                    for (i, j) in zip(*np.nonzero(m["fringe"])):
                        if not dn["valid"][i, j]:
                            b["nu_tilde"][i, j] = (1.0 - sponge_a0) * snap_nu[bi][i, j] + sponge_a0 * inf
                            continue
                        k = dn["blk"][i, j]; i0 = dn["i"][i, j]; j0 = dn["j"][i, j]
                        x = dn["fx"][i, j]; e = dn["fy"][i, j]; sn = snap_nu[k]
                        i1 = i0 + 1; j1 = j0 + 1
                        w00 = (1-x)*(1-e); w10 = x*(1-e); w01 = (1-x)*e; w11 = x*e
                        dnf = (w00*sn[i0, j0] + w10*sn[i1, j0]
                               + w01*sn[i0, j1] + w11*sn[i1, j1])
                        depth = fd[i, j]
                        a = sponge_a0 + (1.0 - sponge_a0) * ((depth - 1.0) / nf) ** 2 \
                            if nf > 0 else 1.0
                        b["nu_tilde"][i, j] = (1.0 - a) * snap_nu[bi][i, j] + a * dnf

    def exchange_pressure(self):
        """Make the overset interface PRESSURE-CONTINUOUS: set each block's
        fringe (and hole) pressure to the donor block's bilinearly-interpolated
        pressure, reusing the SAME donor table as exchange().

        ROOT CAUSE of two-way drift (see overset_cylinder_fvm.joint_project): with
        plain velocity-Dirichlet coupling each block is projected with p=0 at its
        recv ring, so the interface pressure is a FREE O(1) value on the donor side
        but 0 on the acceptor side -- an O(1) PRESSURE JUMP.  That jump is not
        divergence, but the next exchange cannot remove it, so the fringe velocity
        (continuous) gets re-corrected against two different pressure references and
        leaves a residual divergence every step that ROUND-TRIPS AND GROWS (the
        acceptor velocity climbs monotonically -- exactly the flap u_max growth
        seen here).  Pinning p_acceptor(fringe) = p_donor(fringe) makes the seam
        pressure-continuous -> the round-trip leak vanishes -> the drift stops.
        Only recv cells are overwritten (they are not solved), so each block stays
        divergence-free on its solved cells."""
        for bi, b in enumerate(self.blocks):
            m = self.masks[bi]
            dn = self.donor[bi]
            acc = m["fringe"] | m["hole"]
            if dn is None or not acc.any():
                continue
            ii, jj = np.nonzero(acc)
            for (i, j) in zip(ii, jj):
                if not dn["valid"][i, j]:
                    continue
                k = int(dn["blk"][i, j]); i0 = int(dn["i"][i, j]); j0 = int(dn["j"][i, j])
                x = dn["fx"][i, j]; e = dn["fy"][i, j]
                ob = self.blocks[k]; sp = ob["p"]
                i1 = i0 + 1; j1 = j0 + 1
                w00 = (1 - x) * (1 - e); w10 = x * (1 - e)
                w01 = (1 - x) * e;       w11 = x * e
                b["p"][i, j] = (w00 * sp[i0, j0] + w10 * sp[i1, j0]
                                + w01 * sp[i0, j1] + w11 * sp[i1, j1])

    def summary(self):
        s = []
        for bi, b in enumerate(self.blocks):
            m = self.masks[bi]
            s.append("%s: hole=%d fringe=%d farfield=%d solved=%d wall=%d" % (
                b.get("name", str(bi)), int(m["hole"].sum()), int(m["fringe"].sum()),
                int(m["farfield"].sum()), int(m["solved"].sum()), int(b["wall"].sum())))
        return "\n".join(s)