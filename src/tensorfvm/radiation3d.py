"""Three-dimensional surface-to-surface view factors for planar polygons.

Why this module exists
----------------------
:mod:`tensorfvm.radiation` covers **two-dimensional** enclosures only (infinitely
long strips, crossed-strings construction, exact for that limit).  That scope was
declared explicitly in the delivered debt entry ``docs/VERIFICATION_DEBT_LEDGER.md``
§4b item D8 ("明示未做: 三维视图因子、遮挡扣除"), so any three-dimensional thermal
model in this repository -- a cargo-hold corner, an ice-channel wall, a deck
frustum, a cooled equipment bay -- had no way to compute radiative exchange
*between* surfaces at all.

Two methods, and why both are here
----------------------------------
1. **Area quadrature of an exact inner integral** (the default,
   :func:`exchange_area`).  The inner integral over the target polygon is
   evaluated in closed form by Lambert's contour sum

   .. math:: F_{dA_i \\to A_j} = \\frac{1}{2\\pi}\\sum_{k}\\gamma_k\\,
             \\hat{\\mathbf{n}}_i\\cdot\\frac{\\mathbf{r}_k\\times\\mathbf{r}_{k+1}}
             {\\lVert\\mathbf{r}_k\\times\\mathbf{r}_{k+1}\\rVert},

   with :math:`\\gamma_k` the angle subtended by edge ``k`` and
   :math:`\\mathbf{r}_k` the vector from the observation point to vertex ``k``.
   Being exact in the inner variable kills the :math:`1/r^2` kernel, so the
   remaining 2-D quadrature over the emitting polygon only sees a bounded
   integrand.  This method works for **every** configuration, including the ones
   that matter for an enclosure (faces that share an edge).

2. **Double contour integration** (:func:`exchange_area_contour`), the classic
   Mitalas-Stephenson identity

   .. math:: A_i F_{ij} = \\frac{1}{2\\pi}\\oint_{\\Gamma_i}\\oint_{\\Gamma_j}
             \\ln r\\;(\\mathrm{d}\\mathbf{l}_i\\cdot\\mathrm{d}\\mathbf{l}_j).

   It is *exact* (measured: machine precision) for polygons that are disjoint and
   not coplanar, and it is an independent method, which is exactly what the
   verification driver needs to cross-check method 1.  It is **not** valid when
   the polygons touch or are coplanar -- measured failures are recorded below.

Measured validity boundary of the contour method (do not "fix" these by eye)
---------------------------------------------------------------------------
The ``ln r`` identity silently produces wrong numbers, not NaN, for degenerate
pairs.  Measured on unit squares (``docs/verification-radiation3d/summary.json``):

============================  =========  =================  ==============
configuration                 contour    truth              verdict
============================  =========  =================  ==============
parallel opposed, d = 1        0.199825   0.199825 (closed)  PASS 3e-16
perpendicular, gap 0.5         0.076137   0.076137 (quad)    PASS 1e-16
perpendicular, shared edge    -0.277421   0.200044 (exact)   FAIL  +239 %
coplanar, shared edge         -0.405078   0                  FAIL  +405 %
coplanar, gap 0.5             -0.232028   0                  FAIL  +232 %
============================  =========  =================  ==============

:func:`contour_pair_is_valid` encodes exactly this boundary (touching or
coplanar => invalid) and :func:`exchange_area` never calls the contour path for
such a pair.  Filling the enclosure values in from the summation rule instead --
i.e. "make the rows sum to one" -- would hide a >200 % pair error; the driver
asserts both the summation rule *and* the independent closed forms.

Geometry contract
-----------------
A polygon's vertex order defines its normal (Newell's construction) and its
contour direction (counter-clockwise about that normal).  Exchange is computed
**two-sided**: reversing a polygon's vertex order does not change any physical
number (the driver asserts this invariance), so enclosures may be built with
either all-outward or all-inward normals.

Known limits (honest)
---------------------
* **No occlusion / blocking.**  A third surface standing between two polygons is
  not subtracted; the caller owns contraposed geometry.  This is the second half
  of D8 and is *not* delivered here.
* Planar convex polygons only; non-planar vertex sets raise ``ValueError``.
* Curved surfaces are faceted; the error is ``O(1/N^2)`` in the facet count
  (verified against the coaxial-disk closed form).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Polygon3D", "box_enclosure", "regular_polygon_disk",
    "point_form_factor", "exchange_area", "exchange_area_contour",
    "exchange_area_matrix", "view_factor_matrix",
    "contour_pair_is_valid", "reciprocity_residual", "summation_residual",
    "coaxial_disk_form_factor", "perpendicular_common_edge_form_factor",
    "parallel_rectangles_form_factor",
]

_EPS = 1e-12


def _as_points(vertices: Iterable[Sequence[float]]) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must be an (n, 3) array")
    if v.shape[0] < 3:
        raise ValueError("a polygon needs at least three vertices")
    if not np.all(np.isfinite(v)):
        raise ValueError("vertices must be finite")
    return v


def _newell(v: np.ndarray) -> np.ndarray:
    """Newell normal (unnormalised); its norm is twice the polygon area."""
    nxt = np.roll(v, -1, axis=0)
    return np.array([
        np.sum((v[:, 1] - nxt[:, 1]) * (v[:, 2] + nxt[:, 2])),
        np.sum((v[:, 2] - nxt[:, 2]) * (v[:, 0] + nxt[:, 0])),
        np.sum((v[:, 0] - nxt[:, 0]) * (v[:, 1] + nxt[:, 1])),
    ])


@dataclass(frozen=True)
class Polygon3D:
    """A planar, convex polygon in three dimensions.

    The vertex order carries the normal (right-hand rule via Newell) and the
    contour direction, so no separate orientation flag exists that could disagree
    with the geometry.
    """

    vertices: Tuple[Tuple[float, float, float], ...]
    planarity_tol: float = 1e-9

    def __init__(self, vertices: Iterable[Sequence[float]], planarity_tol: float = 1e-9):
        v = _as_points(vertices)
        raw = _newell(v)
        twice_area = float(np.linalg.norm(raw))
        if twice_area <= _EPS:
            raise ValueError("degenerate polygon (zero area)")
        scale = max(float(np.ptp(v[:, k])) for k in range(3))
        offset = np.abs((v - v[0]) @ (raw / twice_area))
        if float(offset.max()) > planarity_tol * max(scale, _EPS) + 1e-14:
            raise ValueError(
                f"polygon is not planar: max out-of-plane offset {offset.max():.3e}")
        object.__setattr__(self, "vertices", tuple(tuple(float(x) for x in p) for p in v))
        object.__setattr__(self, "planarity_tol", float(planarity_tol))

    # -- geometry -----------------------------------------------------------
    @property
    def coords(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=float)

    @property
    def normal(self) -> np.ndarray:
        raw = _newell(self.coords)
        return raw / np.linalg.norm(raw)

    @property
    def area(self) -> float:
        return 0.5 * float(np.linalg.norm(_newell(self.coords)))

    @property
    def centroid(self) -> np.ndarray:
        v = self.coords
        n_hat = self.normal
        acc = np.zeros(3)
        w_tot = 0.0
        for i in range(1, len(v) - 1):
            a, b, c = v[0], v[i], v[i + 1]
            w = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
            acc += w * (a + b + c) / 3.0
            w_tot += w
        if w_tot <= _EPS:
            raise ValueError("degenerate polygon (zero area)")
        c = acc / w_tot
        return c - n_hat * float((c - v[0]) @ n_hat)      # remove round-off drift

    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        v = self.coords
        return [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]

    def __str__(self) -> str:                                  # pragma: no cover
        return (f"Polygon3D(n={len(self.vertices)}, area={self.area:.6g}, "
                f"centroid={np.round(self.centroid, 6).tolist()})")


# ---------------------------------------------------------------------------
# method 1: exact inner integral (Lambert contour sum) + outer quadrature
# ---------------------------------------------------------------------------
def point_form_factor(point: Sequence[float], target: Polygon3D,
                      emitter_normal: Optional[Sequence[float]] = None) -> float:
    """``F_{dA -> target}`` for an emitting element at ``point`` (0 <= F <= 1).

    Lambert's contour sum, exact for a planar convex polygon and free of the
    ``1/r^2`` kernel singularity: the only degeneracy left is ``|r_k x r_{k+1}| = 0``
    (the observation point on the line through an edge), where the edge subtends
    zero angle and contributes nothing.

    ``emitter_normal`` is the normal of the *emitting* surface at ``point`` (a
    planar emitter has one).  The projected solid angle is measured with respect
    to that normal; passing the *target's* normal instead is the single easiest
    way to get a plausible-looking but wrong answer, so the argument is required
    rather than inferred.

    The sign of the raw contour sum follows the orientation of the target loop as
    seen from the point, hence the magnitude is returned: exchange in this module
    is two-sided (see the module docstring), which also makes the result invariant
    under reversing a polygon's vertex order.
    """
    if emitter_normal is None:
        raise ValueError("emitter_normal is required (normal at the emitting element)")
    x = np.asarray(point, dtype=float)
    v = target.coords
    r = v - x
    r_next = np.roll(r, -1, axis=0)
    cross = np.cross(r, r_next)
    twice_tri = np.linalg.norm(cross, axis=1)
    r_norm = np.linalg.norm(r, axis=1)
    r_next_norm = np.linalg.norm(r_next, axis=1)
    denom = r_norm * r_next_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_gamma = np.where(denom > _EPS, np.einsum("ij,ij->i", r, r_next) / denom, 1.0)
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    n_hat = np.asarray(emitter_normal, dtype=float)
    n_norm = float(np.linalg.norm(n_hat))
    if n_norm <= _EPS:
        raise ValueError("emitter_normal must be non-zero")
    n_hat = n_hat / n_norm
    proj = np.where(twice_tri > _EPS, (cross @ n_hat) / np.where(twice_tri > _EPS, twice_tri, 1.0), 0.0)
    return abs(float(np.dot(gamma, proj))) / (2.0 * math.pi)


def _triangle_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss rule on the unit triangle via the Duffy transformation."""
    xs, ws = np.polynomial.legendre.leggauss(order)
    xi = 0.5 * (xs + 1.0)
    w = 0.5 * ws
    a, b = np.meshgrid(xi, xi, indexing="ij")
    wa, wb = np.meshgrid(w, w, indexing="ij")
    # (a, b) in [0,1]^2 -> (u, v) = (a(1-b), a b) covers the unit triangle u+v <= 1
    # with jacobian a, so sum(weights) = 1/2 = the triangle area.
    u = a * (1.0 - b)
    v = a * b
    weight = wa * wb * a
    return np.stack([u.ravel(), v.ravel()], axis=1), weight.ravel()


def _polygon_rule(poly: Polygon3D, order: int, panels: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature points and weights on a planar polygon (fan from the centroid)."""
    v = poly.coords
    c = poly.centroid
    tri_uv, tri_w = _triangle_rule(order)
    if panels > 1:
        # subdivide the unit square before the Duffy map to concentrate points
        sub = [i / panels for i in range(panels + 1)]
        xs, ws = np.polynomial.legendre.leggauss(order)
        pts: List[np.ndarray] = []
        wts: List[np.ndarray] = []
        for i in range(panels):
            for j in range(panels):
                lo_a, hi_a = sub[i], sub[i + 1]
                lo_b, hi_b = sub[j], sub[j + 1]
                a = lo_a + 0.5 * (hi_a - lo_a) * (xs + 1.0)
                b = lo_b + 0.5 * (hi_b - lo_b) * (xs + 1.0)
                wa = 0.5 * (hi_a - lo_a) * ws
                wb = 0.5 * (hi_b - lo_b) * ws
                A, B = np.meshgrid(a, b, indexing="ij")
                WA, WB = np.meshgrid(wa, wb, indexing="ij")
                pts.append(np.stack([(A * (1.0 - B)).ravel(), (A * B).ravel()], axis=1))
                wts.append((WA * WB * A).ravel())
        tri_uv = np.concatenate(pts, axis=0)
        tri_w = np.concatenate(wts, axis=0)
    points = []
    weights = []
    for i in range(len(v)):
        p0 = c
        p1 = v[i]
        p2 = v[(i + 1) % len(v)]
        # (u, v) barycentric: p0 + u*(p1-p0) + v*(p2-p0) has jacobian
        # |cross(p1-p0, p2-p0)| = 2 * triangle area, and sum(tri_w) = 1/2,
        # so the weights below sum to the triangle area.
        jac = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        pts = p0 + tri_uv[:, :1] * (p1 - p0) + tri_uv[:, 1:2] * (p2 - p0)
        points.append(pts)
        weights.append(tri_w * jac)
    return np.concatenate(points, axis=0), np.concatenate(weights, axis=0)


def exchange_area(p_i: Polygon3D, p_j: Polygon3D, order: int = 24,
                  panels: int = 1, method: str = "auto") -> float:
    """``A_i F_{ij}``, positive for mutually visible polygons (two-sided).

    ``method``: ``"auto"`` (contour path when valid, otherwise quadrature),
    ``"quadrature"``, or ``"contour"``.  ``panels`` refines the outer quadrature
    of the quadrature path only.

    Accuracy is measured, not assumed: for disjoint non-coplanar polygons the
    quadrature path agrees with the independent contour path to machine
    precision, and for enclosure configurations (faces sharing an edge) it
    reproduces the exact symmetry value of the cube to ``1e-13``.
    """
    if order < 2:
        raise ValueError("order >= 2")
    if method not in ("auto", "quadrature", "contour"):
        raise ValueError("method must be 'auto', 'quadrature' or 'contour'")
    if p_i is p_j:
        return 0.0
    if method == "contour" or (method == "auto" and contour_pair_is_valid(p_i, p_j)):
        return exchange_area_contour(p_i, p_j, order=max(order, 16))
    xs, ws = _polygon_rule(p_i, order, panels)
    n_i = p_i.normal
    vals = np.array([point_form_factor(x, p_j, n_i) for x in xs])
    return float(np.dot(ws, vals))


def exchange_area_matrix(polygons: Sequence[Polygon3D], order: int = 24,
                         panels: int = 1, method: str = "auto") -> np.ndarray:
    """Matrix ``AF[i, j] = A_i F_ij`` with a zero diagonal."""
    polys = list(polygons)
    n = len(polys)
    af = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            af[i, j] = exchange_area(polys[i], polys[j], order=order,
                                     panels=panels, method=method)
    return af


def view_factor_matrix(polygons: Sequence[Polygon3D], order: int = 24,
                       panels: int = 1, method: str = "auto",
                       symmetrise: bool = False) -> np.ndarray:
    """``F[i, j]`` for a set of polygons (rows are the emitting surfaces).

    ``symmetrise=True`` replaces ``A_i F_ij`` by ``(A_i F_ij + A_j F_ji)/2``, which
    enforces reciprocity exactly while preserving the summation rule; use it
    before :class:`tensorfvm.radiation.GrayEnclosure`, which requires
    ``sum_j F_ij = 1``.
    """
    polys = list(polygons)
    af = exchange_area_matrix(polys, order=order, panels=panels, method=method)
    if symmetrise:
        af = 0.5 * (af + af.T)
    areas = np.array([p.area for p in polys])
    return af / areas[:, None]


def reciprocity_residual(af: np.ndarray) -> float:
    """``max |A_i F_ij - A_j F_ji| / max |AF|`` (0 = reciprocal)."""
    af = np.asarray(af, dtype=float)
    scale = float(np.max(np.abs(af)))
    if scale <= 0.0:
        raise ValueError("degenerate exchange-area matrix")
    return float(np.max(np.abs(af - af.T)) / scale)


def summation_residual(f: np.ndarray) -> float:
    """``max_i |sum_j F_ij - 1|`` (summation rule for a closed enclosure)."""
    return float(np.max(np.abs(np.asarray(f, dtype=float).sum(axis=1) - 1.0)))


# ---------------------------------------------------------------------------
# method 2: double contour integration (independent cross-check)
# ---------------------------------------------------------------------------
def _G(x: float) -> float:
    """Antiderivative of ``ln|x|``, with ``G(0) = 0``."""
    ax = abs(x)
    return x * math.log(ax) - x if ax > 0.0 else 0.0


def _collinear_ln_integral(a0: float, a1: float, b0: float, b1: float) -> float:
    """``int_{a0}^{a1} int_{b0}^{b1} ln|s-t| dt ds`` with the inner integral exact."""
    a_lo, a_hi = sorted((a0, a1))
    b_lo, b_hi = sorted((b0, b1))
    breaks = [a_lo] + [x for x in (b_lo, b_hi) if a_lo < x < a_hi] + [a_hi]
    xs, ws = np.polynomial.legendre.leggauss(48)
    total = 0.0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi <= lo:
            continue
        s = lo + 0.5 * (hi - lo) * (xs + 1.0)
        f = np.array([_G(si - b_hi) - _G(si - b_lo) for si in s])
        total += 0.5 * (hi - lo) * float(np.dot(ws, f))
    return total


def _is_collinear(a0, a1, b0, b1, scale: float) -> bool:
    ta = a1 - a0
    tb = b1 - b0
    la = np.linalg.norm(ta)
    lb = np.linalg.norm(tb)
    if la <= _EPS or lb <= _EPS:
        return False
    if np.linalg.norm(np.cross(ta / la, tb / lb)) > 1e-9:
        return False
    n_hat = ta / la
    worst = 0.0
    for b in (b0, b1):
        w = b - a0
        worst = max(worst, float(np.linalg.norm(w - float(w @ n_hat) * n_hat)))
    return worst <= 1e-9 * max(scale, _EPS)


def _segment_pair_exchange(a0, a1, b0, b1, order: int, scale: float) -> float:
    """``int int ln r (dl_i . dl_j)`` over one pair of segments."""
    ta_vec = a1 - a0
    tb_vec = b1 - b0
    la = float(np.linalg.norm(ta_vec))
    lb = float(np.linalg.norm(tb_vec))
    if la <= _EPS or lb <= _EPS:
        return 0.0
    ta = ta_vec / la
    tb = tb_vec / lb
    dot = float(ta @ tb)
    if _is_collinear(a0, a1, b0, b1, scale):
        s0, s1 = 0.0, la
        t0 = float((b0 - a0) @ ta)
        t1 = float((b1 - a0) @ ta)
        return dot * _collinear_ln_integral(s0, s1, t0, t1)
    xs, ws = np.polynomial.legendre.leggauss(order)
    pa = a0 + np.outer(0.5 * la * (xs + 1.0), ta)
    pb = b0 + np.outer(0.5 * lb * (xs + 1.0), tb)
    d = pb[None, :, :] - pa[:, None, :]
    r = np.linalg.norm(d, axis=2)
    with np.errstate(divide="ignore"):
        lg = np.log(r)
    if not np.all(np.isfinite(lg)):
        lg = np.where(np.isfinite(lg), lg, 0.0)
    return dot * float(np.einsum("i,j,ij->", ws, ws, lg)) * 0.25 * la * lb


def _min_edge_distance(p_i: Polygon3D, p_j: Polygon3D) -> float:
    """Smallest distance between any two edges (shares a vertex => ~0)."""
    best = np.inf
    for a0, a1 in p_i.edges():
        for b0, b1 in p_j.edges():
            d = a1 - a0
            e = b1 - b0
            r = a0 - b0
            aa = float(d @ d)
            bb = float(d @ e)
            cc = float(e @ e)
            denom = aa * cc - bb * bb
            s = 0.0
            if abs(denom) > 1e-18:
                s = min(1.0, max(0.0, (bb * float(e @ r) - cc * float(d @ r)) / denom))
            t = min(1.0, max(0.0, float((d @ (a0 + s * d - b0)) / cc))) if cc > _EPS else 0.0
            if aa > _EPS:
                s = min(1.0, max(0.0, float((e * t - r) @ d) / aa)) if cc > _EPS else s
            best = min(best, float(np.linalg.norm((a0 + s * d) - (b0 + t * e))))
            for pa in (a0, a1):
                for qb in (b0, b1):
                    best = min(best, float(np.linalg.norm(pa - qb)))
    return best


def contour_pair_is_valid(p_i: Polygon3D, p_j: Polygon3D, tol: float = 1e-9,
                          separation: float = 0.1) -> bool:
    """Is the ``ln r`` contour identity trustworthy for this pair?

    ``False`` for the degenerate families measured to break it (module docstring
    table): polygons that touch (shared vertex/edge), polygons that come closer
    than ``separation`` x their own size, and coplanar polygons.  The failures are
    silent -- a wrong number, never NaN -- which is why a gate exists instead of a
    tolerance on the result.

    Measured degradation of the contour path for perpendicular unit squares with a
    gap ``g`` (relative to the quadrature path, itself anchored on the exact
    touching value): ``g=1e-1`` 1.2e-06, ``g=1e-2`` 1.5e-02, ``g=1e-3`` 8.8e-02,
    ``g=1e-6`` 3.6e-01.  The ``separation = 0.1`` default sits above the knee; the
    gap is caused by the logarithmic edge singularity, which the plain Gauss
    quadrature of this path cannot resolve.
    """
    scale = max(math.sqrt(p_i.area), math.sqrt(p_j.area))
    if _min_edge_distance(p_i, p_j) <= separation * scale:
        return False
    n_i, n_j = p_i.normal, p_j.normal
    if abs(abs(float(n_i @ n_j)) - 1.0) <= tol:
        gap = abs(float((p_j.centroid - p_i.centroid) @ n_i))
        if gap <= tol * scale:
            return False
    return True


def exchange_area_contour(p_i: Polygon3D, p_j: Polygon3D, order: int = 16) -> float:
    """``A_i F_ij`` by the Mitalas-Stephenson double contour integral.

    Exact for disjoint, non-coplanar polygons.  For touching or coplanar polygons
    the identity is **invalid** and this function will happily return a wrong
    number: use :func:`contour_pair_is_valid` (or ``method="auto"``).
    """
    if order < 4:
        raise ValueError("order >= 4")
    if p_i is p_j:
        return 0.0
    scale = max(math.sqrt(p_i.area), math.sqrt(p_j.area))
    total = 0.0
    for a0, a1 in p_i.edges():
        for b0, b1 in p_j.edges():
            total += _segment_pair_exchange(a0, a1, b0, b1, order, scale)
    # the contour integral is signed (flipping either loop flips the sign); this
    # module's contract is two-sided exchange, so the magnitude is returned
    return abs(total) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# geometry constructors
# ---------------------------------------------------------------------------
def box_enclosure(size: Sequence[float] = (1.0, 1.0, 1.0),
                  origin: Sequence[float] = (0.0, 0.0, 0.0)) -> List[Polygon3D]:
    """Six faces of an axis-aligned box, contours CCW about their **outward** normals.

    Order: ``-x, +x, -y, +y, -z, +z``.  Exchange here is two-sided, so the same
    numbers come out if every face is reversed.
    """
    lx, ly, lz = (float(x) for x in size)
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise ValueError("box sizes must be positive")
    ox, oy, oz = (float(x) for x in origin)
    x0, y0, z0 = ox, oy, oz
    x1, y1, z1 = ox + lx, oy + ly, oz + lz
    faces = [
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],   # -x
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],   # +x
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],   # -y
        [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],   # +y
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],   # -z
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],   # +z
    ]
    return [Polygon3D(f) for f in faces]


def regular_polygon_disk(radius: float, n: int, centre: Sequence[float],
                         normal: Sequence[float] = (0.0, 0.0, 1.0),
                         phase: float = 0.0) -> Polygon3D:
    """Planar ``n``-gon inscribed in a disk -- the facet model of a circular disk."""
    if radius <= 0.0 or n < 3:
        raise ValueError("radius must be positive and n >= 3")
    n_hat = np.asarray(normal, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n_hat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    c = np.asarray(centre, dtype=float)
    ang = phase + 2.0 * math.pi * np.arange(n) / n
    return Polygon3D([c + radius * (math.cos(a) * e1 + math.sin(a) * e2) for a in ang])


# ---------------------------------------------------------------------------
# independent closed forms used as verification anchors
# ---------------------------------------------------------------------------
def parallel_rectangles_form_factor(a: float, b: float, d: float) -> float:
    """``F_{12}`` between two identical, parallel, directly opposed ``a x b`` rectangles.

    Hamilton & Morgan (1952) closed form.  Reproduces the standard table values
    used by the driver: ``F(1x1, d=1) = 0.1998248957``, ``F(1x1, d=0.5) =
    0.4152532836``.
    """
    if a <= 0.0 or b <= 0.0 or d <= 0.0:
        raise ValueError("dimensions must be positive")
    x, y = a / d, b / d
    t1 = math.log(math.sqrt((1 + x * x) * (1 + y * y) / (1 + x * x + y * y)))
    t2 = x * math.sqrt(1 + y * y) * math.atan(x / math.sqrt(1 + y * y))
    t3 = y * math.sqrt(1 + x * x) * math.atan(y / math.sqrt(1 + x * x))
    t4 = x * math.atan(x) + y * math.atan(y)
    return (2.0 / (math.pi * x * y)) * (t1 + t2 + t3 - t4)


def perpendicular_common_edge_form_factor(w1: float, w2: float, h: float) -> float:
    """``F_{12}`` between two perpendicular rectangles sharing an edge ``h``.

    Hottel-Sarofim closed form for rectangles ``w1 x h`` and ``w2 x h``.  Checked
    against the limits it must obey (``w2 -> 0`` gives ``0``; ``w2 -> inf``
    approaches ``1/2``), so a transcription slip in the log group shows up as a
    limit failure rather than a quiet bias.  For ``w1 = w2 = h`` it must return the
    cube's symmetry value ``(1 - 0.199825)/4 = 0.2000437761``.
    """
    if w1 <= 0.0 or w2 <= 0.0 or h <= 0.0:
        raise ValueError("edge lengths must be positive")
    x, y = w1 / h, w2 / h
    term = (x * math.atan(1.0 / x) + y * math.atan(1.0 / y)
            - math.hypot(x, y) * math.atan(1.0 / math.hypot(x, y)))
    psi = (((1.0 + x * x) * (1.0 + y * y)) / (1.0 + x * x + y * y)
           * (x * x * (1.0 + x * x + y * y) / ((1.0 + x * x) * (x * x + y * y))) ** (x * x)
           * (y * y * (1.0 + x * x + y * y) / ((1.0 + y * y) * (x * x + y * y))) ** (y * y))
    return (term + 0.25 * math.log(psi)) / (math.pi * x)


def coaxial_disk_form_factor(r1: float, r2: float, h: float) -> float:
    """``F_{12}`` between two coaxial parallel disks of radii ``r1``, ``r2`` at ``h``."""
    if r1 <= 0.0 or r2 <= 0.0 or h <= 0.0:
        raise ValueError("radii and spacing must be positive")
    big_r1, big_r2 = r1 / h, r2 / h
    s = 1.0 + (1.0 + big_r2 ** 2) / big_r1 ** 2
    radicand = s * s - 4.0 * (r2 / r1) ** 2
    if radicand < 0.0:
        raise ValueError("closed form undefined for this geometry")
    return 0.5 * (s - math.sqrt(radicand))