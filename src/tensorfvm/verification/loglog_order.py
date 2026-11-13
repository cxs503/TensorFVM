"""Three-grid log--log-slope order verification for cell-centred finite volumes.

Why this module exists
----------------------
Order-of-accuracy claims in TensorFVM used to be produced by one of two *ad hoc*
estimators:

* a two-point successive ratio, ``p = ln(e_coarse/e_fine)/ln(r)`` with ``r = 2``
  hard-wired (``case_cavity_re100.observed_order``,
  ``benchmark_poiseuille_channel.observed_order``,
  ``verification.core.observed_orders``, ...).  Two points and one unknown leave
  **zero** degrees of freedom: the result carries no uncertainty and cannot
  distinguish a clean ``C h^p`` decay from a pre-asymptotic or non-monotone
  series -- it always returns a plausible-looking number;
* the Richardson/GCI family (:mod:`tensorfvm.verification.richardson`,
  :mod:`tensorfvm.verification.multi_grid`), which is well cross-validated but
  estimates ``p`` from *differences of the functional* and therefore needs a
  finite limit and a monotone one-sided approach.

This module implements the third and most common practice in the FV literature:
the **log--log least-squares slope of a known discretisation error** over three
(or more) grids.  With ``n`` levels and two fitted parameters (``p``, ``C``) the
fit has ``n-2`` degrees of freedom, so a three-grid study *does* report a
standard error -- and this module reports it instead of hiding it.  Because the
estimator is used to audit solvers it is verified against synthetic data of
known order before it is applied to anything.

The three load-bearing conventions for **cell-centred** FV
----------------------------------------------------------
1. **The unknown is a cell average, not a point value.**
   ``u_h,i = (1/V_i) Int_cell u``.  Comparing ``u_h,i`` against ``u(x_i)`` (the
   point value at the cell centre) injects a *reference* error
   ``u_h - u(x_i) = h^2/24 u''(x_i) + O(h^4)``.  For a second-order scheme this
   only changes the constant (for the truncation error by exactly the factor
   :data:`POINT_VALUE_CONSTANT_FACTOR` = 2), i.e. the error *magnitude* -- which
   is what the 3 % / 15 % acceptance gates compare -- moves by a factor of two
   while the *order* still reads 2.  For a *higher*-order scheme the same term
   **caps the observable order at 2**: a formally fourth-order cell-centred
   scheme reads as second order.  Both effects are measured here (criteria
   ``C5``--``C7``).
2. **The abscissa is the cell size, not the cell count.**
   ``e ~ h^p`` with ``h = (V/N)^(1/d)``.  Fitting against the number of cells
   returns ``p/d`` under an isotropic refinement (``p/2`` in 2-D, criterion
   ``C10``) -- a 2nd-order scheme then "measures" first order and a genuinely
   degraded scheme cannot be told apart from a dimension artefact.
3. **Norms are volume weighted.**  On a non-uniform mesh the unweighted cell
   mean is not the ``L2`` norm.  Its effect on the *order* is negligible on a
   fixed smooth mesh family (the weight--error correlation is ``h``-independent;
   criterion ``C11`` asserts ``|dp| <= 0.05``, this module's declared claim)
   while the reported *magnitude* moves by the measured ratio
   ``l2/l2_unweighted``.  Weighting is therefore not an order question -- it is a
   threshold question.

What this module deliberately does **not** do
---------------------------------------------
It does not decide whether a solver is "converged".  A slope is an order only
inside the asymptotic range, so :func:`loglog_order` withholds the number
(``verdict='WITHHELD'``) whenever the series is non-monotone, the refinement
ratio is too small, or two adjacent levels are identical to round-off.  Unlike
``verification.annular_convergence.observed_order`` this module *returns* the
withheld study rather than raising: a guarded study has to be printable in a
table, and ``nan`` with a recorded reason is more honest than an exception that
a caller swallows.

Command line
------------
``python -m tensorfvm.verification.loglog_order`` runs the whole cross-validation
and writes ``summary.json`` + ``fig_loglog_order.png`` next to this repository's
``docs/verification-three-grid-loglog-cellcentre/`` directory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MIN_RATIO",
    "ORDER_TOLERANCE",
    "ASYMPTOTIC_TOLERANCE",
    "POINT_VALUE_CONSTANT_FACTOR",
    "LogLogStudy",
    "loglog_order",
    "cell_error_norms",
    "mm_value",
    "mm_derivative",
    "mm_cell_average",
    "mm_cell_average_second_derivative",
    "uniform_faces",
    "stretched_faces",
    "truncation_uniform",
    "truncation_stretched",
    "truncation_ladder",
    "solution_error_ladder",
    "stretched_weights",
    "run",
    "figure",
    "main",
]

#: Minimum adjacent refinement ratio that still carries order information (same
#: value as :mod:`tensorfvm.verification.richardson`).
MIN_RATIO = 1.01

#: Declared a priori: the observed order must sit within this *relative* band of
#: the formal order.  It is the owner's 15 % standard applied to the order itself
#: -- a solver whose measured order is 15 % off its formal order has not been
#: verified, whatever its absolute error.
ORDER_TOLERANCE = 0.15

#: Declared a priori: the adjacent pairwise slopes of a ladder must agree to this
#: fraction of the formal order for the ladder to be called asymptotic (the 25 %
#: convention already used by
#: ``verification.annular_convergence.observed_order(asymptotic_tolerance=...)``).
ASYMPTOTIC_TOLERANCE = 0.25

#: Analytic ratio of the *cell-average* truncation constant of the 2nd-order
#: face-difference operator to its *point-value* one: ``(h^2/12)/(h^2/24) = 2``
#: (criterion ``C7``), quoted so the trap's size is visible without re-deriving.
POINT_VALUE_CONSTANT_FACTOR = 2.0


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LogLogStudy:
    """Three-or-more-grid log--log least-squares order study (dict serialisable).

    Spacings and errors are stored **coarse -> fine** whatever order they were
    supplied in.  ``order`` is the least-squares slope of ``ln e`` against
    ``ln h``; ``order_stderr`` is defined for ``n >= 3`` (``nan`` for two levels);
    ``pairwise`` holds the adjacent two-point slopes, i.e. exactly what the
    *ad hoc* estimators of this repository compute, so their spread can be read
    next to the fitted value.
    """

    spacings: Tuple[float, ...]
    errors: Tuple[float, ...]
    order: float
    order_stderr: float
    r2: float
    intercept: float
    pairwise: Tuple[float, ...]
    spread: float
    monotone: bool
    asymptotic: bool
    theory_order: Optional[float]
    relative_deviation: float
    tolerance: float
    withheld_reason: str
    verdict: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def loglog_order(
    spacings: Sequence[float],
    errors: Sequence[float],
    *,
    theory_order: Optional[float] = None,
    tolerance: float = ORDER_TOLERANCE,
    asymptotic_tolerance: float = ASYMPTOTIC_TOLERANCE,
) -> LogLogStudy:
    """Observed order from the log--log slope of ``error`` against ``spacing``.

    ``spacings`` and ``errors`` must have the same length ``n >= 3`` and are
    accepted in any order (the ladder is sorted coarse -> fine internally; a
    silent sign flip from an inverted ladder is a classic way to publish ``-p``).

    Guards, evaluated before any fit (each yields ``verdict='WITHHELD'`` with
    ``order=nan`` and the rule recorded in ``withheld_reason``):

    ``non-monotone``
        the error sequence does not decrease strictly with refinement.  A
        least-squares slope through such a sequence is a number, not an order
        (the committed Sod artifact of this repository is a real example: its
        fitted exponent is negative while the finest pair gives ``+1.84``);
    ``refinement-ratio-too-small``
        an adjacent ratio below :data:`MIN_RATIO` cannot resolve an order;
    ``stagnant``
        two adjacent levels have identical errors to round-off, so the slope is
        dominated by noise.

    ``asymptotic`` compares the two finest adjacent pairwise slopes; a ladder
    whose slopes disagree beyond ``asymptotic_tolerance * theory_order`` is
    reported with ``asymptotic=False`` (pre-asymptotic).  That is a *qualified*
    order, not a FAIL: ``verdict`` is ``PASS``/``FAIL`` only against a declared
    ``theory_order``, and the caller decides what a pre-asymptotic PASS means.
    """
    h = np.asarray(spacings, dtype=float).ravel()
    e = np.asarray(errors, dtype=float).ravel()
    if h.shape != e.shape:
        raise ValueError("spacings and errors must have the same length")
    if h.size < 3:
        raise ValueError("at least three grid levels are required")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(e))):
        raise ValueError("non-finite spacing or error")
    if not np.all(e > 0.0):
        raise ValueError("errors must be strictly positive (a logarithm is fitted)")
    if not np.all(h > 0.0):
        raise ValueError("spacings must be strictly positive")
    order_idx = np.argsort(-h)                       # coarse -> fine
    h, e = h[order_idx], e[order_idx]
    if np.any(np.diff(h) == 0.0):
        raise ValueError("spacings must be distinct")

    ratios = h[:-1] / h[1:]
    monotone = bool(np.all(np.diff(e) < 0.0))
    withheld = ""
    if not monotone:
        withheld = "non-monotone"
    elif np.any(ratios < MIN_RATIO):
        withheld = "refinement-ratio-too-small"
    elif np.any(np.abs(np.diff(e)) <= 1e-14 * np.abs(e[:-1])):
        withheld = "stagnant"

    pairwise = tuple(
        float(math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]))
        for i in range(len(e) - 1)
    )
    shared = dict(
        spacings=tuple(float(v) for v in h),
        errors=tuple(float(v) for v in e),
        pairwise=pairwise,
        monotone=monotone,
        theory_order=None if theory_order is None else float(theory_order),
        tolerance=float(tolerance),
    )

    if withheld:
        return LogLogStudy(
            order=float("nan"), order_stderr=float("nan"), r2=float("nan"),
            intercept=float("nan"), spread=float("nan"), asymptotic=False,
            relative_deviation=float("nan"), withheld_reason=withheld,
            verdict="WITHHELD", **shared,
        )

    x, y = np.log(h), np.log(e)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    dof = len(e) - 2
    stderr = math.sqrt(ss_res / dof / sxx) if dof >= 1 and sxx > 0.0 else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    spread = abs(pairwise[-1] - pairwise[-2]) if len(pairwise) >= 2 else float("nan")
    scale = abs(theory_order) if theory_order is not None else max(abs(float(slope)), 1e-12)
    asymptotic = bool(spread <= asymptotic_tolerance * scale)

    if theory_order is None:
        deviation, verdict = float("nan"), "INFO"
    else:
        deviation = abs(float(slope) - float(theory_order)) / abs(float(theory_order))
        verdict = "PASS" if deviation <= tolerance else "FAIL"

    return LogLogStudy(
        order=float(slope), order_stderr=float(stderr), r2=float(r2),
        intercept=float(intercept), spread=float(spread),
        asymptotic=asymptotic, relative_deviation=float(deviation),
        withheld_reason="", verdict=verdict, **shared,
    )


# --------------------------------------------------------------------------- #
# norms for cell-centred data
# --------------------------------------------------------------------------- #
def cell_error_norms(error, volumes) -> Dict[str, float]:
    """Discrete norms of a cell-centred error field.

    The *density* norms (``l1``/``l2``) divide by the total measure, so on a
    uniform mesh they are the ordinary ``L1``/``L2`` norms and on any mesh family
    they converge to the continuous norm.  ``l2_unweighted`` is the plain cell
    mean -- what ``float(np.sqrt(np.mean(e**2)))`` computes -- and agrees with
    ``l2`` on a uniform mesh but not on a stretched one.  ``l2_integral`` keeps
    the ``sqrt(V)`` scaling; on a fixed domain its ``h``-scaling equals that of
    ``l2`` (the total volume is constant), so it is reported for completeness
    only.
    """
    e = np.asarray(error, dtype=float).ravel()
    v = np.asarray(volumes, dtype=float).ravel()
    if e.shape != v.shape:
        raise ValueError("error and volumes must have the same shape")
    if not np.all(v > 0.0):
        raise ValueError("cell volumes must be positive")
    total = float(v.sum())
    e2 = e * e
    return dict(
        l1=float(np.sum(v * np.abs(e)) / total),
        l2=float(math.sqrt(np.sum(v * e2) / total)),
        linf=float(np.max(np.abs(e))),
        l2_integral=float(math.sqrt(np.sum(v * e2))),
        l2_unweighted=float(math.sqrt(np.mean(e2))),
    )


# --------------------------------------------------------------------------- #
# manufactured solution (periodic, smooth, rich derivative spectrum)
# --------------------------------------------------------------------------- #
#: ``(harmonic number, amplitude)``: ``u(x) = sum a_k sin(2 pi k x)``.  Three
#: harmonics keep every derivative appearing in the truncation error non-zero, so
#: no single harmonic can accidentally cancel the leading error term.
HARMONICS: Tuple[Tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.3), (3.0, 0.1))


def _omega(k: float) -> float:
    return 2.0 * math.pi * k


def mm_value(x) -> np.ndarray:
    """Manufactured solution ``u(x)`` (point value)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k, a in HARMONICS:
        out = out + a * np.sin(_omega(k) * x)
    return out


def mm_derivative(x) -> np.ndarray:
    """``u'(x)`` in closed form."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k, a in HARMONICS:
        out = out + a * _omega(k) * np.cos(_omega(k) * x)
    return out


def mm_cell_average(left, right) -> np.ndarray:
    """Exact cell average ``(1/V) Int u`` -- closed form, no quadrature error.

    A cell average is the quantity a cell-centred FV scheme actually stores, so
    it is the only admissible reference for one (see the module docstring).
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.zeros_like(left)
    for k, a in HARMONICS:
        w = _omega(k)
        out = out + a * (np.cos(w * left) - np.cos(w * right)) / w
    return out / (right - left)


def mm_cell_average_second_derivative(left, right) -> np.ndarray:
    """Exact cell average of ``u''``: ``(u'(b) - u'(a))/V`` (no quadrature)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return (mm_derivative(right) - mm_derivative(left)) / (right - left)


def uniform_faces(n: int) -> np.ndarray:
    """``n`` periodic cells of size ``1/n`` on ``[0, 1]`` (faces ``0..n``)."""
    return np.linspace(0.0, 1.0, n + 1)


def stretched_faces(n: int, amplitude: float = 0.6) -> np.ndarray:
    """Smooth non-uniform faces: ``x(xi) = xi + a sin(2 pi xi)/(2 pi)``.

    ``dx/dxi = 1 + a cos(2 pi xi) >= 1 - a > 0`` for ``a < 1`` (here ``0.4``), so
    the map is monotone and periodic (``x(0)=0``, ``x(1)=1``).  The family is
    *self-similar*: refining doubles ``n`` without changing the stretch, which is
    what makes the weighting question of criterion ``C11`` well posed.
    """
    xi = np.arange(n + 1, dtype=float) / n
    return xi + amplitude * np.sin(2.0 * math.pi * xi) / (2.0 * math.pi)


# --------------------------------------------------------------------------- #
# cell-centred FV operators and their truncation error
# --------------------------------------------------------------------------- #
VARIANTS = ("second_order", "fourth_order", "fourth_order_cellavg", "shifted")


def truncation_uniform(u, n: int, variant: str = "second_order") -> Tuple[np.ndarray, np.ndarray]:
    """Truncation error of the cell-centred 1-D FV Laplacian on a uniform mesh.

    ``L_h(u)_i = (F_{i+1/2} - F_{i-1/2}) / h`` with the face derivative ``F``

    ``second_order``          ``(u_{i+1} - u_i) / h``                     -> ``p = 2``
    ``fourth_order``          ``(u_{i+2} - 27u_{i+1} + 27u_i - u_{i-1}) / (24h)``
                              -- the textbook 4th-order *midpoint* derivative:
                              ``p = 4`` for **point values**, but only ``p = 2``
                              when its inputs are **cell averages** (it then
                              carries ``h^2/12 u3`` instead of cancelling it);
    ``fourth_order_cellavg``  ``(u_{i-1} - 15u_i + 15u_{i+1} - u_{i+2}) / (12h)``
                              -- the same four cells with coefficients
                              re-derived so that *cell averages* are
                              differentiated to 4th order (its Laplacian is the
                              classic ``(-1, 16, -30, 16, -1)/(12 h^2)``
                              stencil);
    ``shifted``               ``(u_i - u_{i-1}) / h`` -- the flux of face
                              ``i+1/2`` taken from the *previous* face, i.e. a
                              displaced-flux defect (first-order mechanism) -> ``p = 1``

    The two 4-point stencils differ only in their coefficients and each is 4th
    order for exactly one kind of input: this is the **discretisation** side of
    the cell-average trap (criterion ``C7``), complementing the measurement side
    (``C6``).

    Returns ``(tau, faces)`` with ``faces`` the cell faces (``n+1`` values, the
    last coinciding with the first by periodicity) and ``tau`` the truncation
    error per cell, ``L_h(u) - (exact cell average of u'')``.  ``u`` may be the
    exact **cell averages** or the exact **point values**; the two differ by
    exactly a factor two for ``second_order`` (criterion ``C7``).
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.size != n:
        raise ValueError("u must hold one value per cell")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    h = 1.0 / n
    i = np.arange(n)
    if variant == "second_order":
        face = (u[(i + 1) % n] - u[i]) / h
    elif variant == "fourth_order":
        face = (u[(i - 1) % n] - 27.0 * u[i] + 27.0 * u[(i + 1) % n] - u[(i + 2) % n]) / (24.0 * h)
    elif variant == "fourth_order_cellavg":
        face = (u[(i - 1) % n] - 15.0 * u[i] + 15.0 * u[(i + 1) % n] - u[(i + 2) % n]) / (12.0 * h)
    else:
        face = (u[i] - u[(i - 1) % n]) / h
    tau = (face - np.roll(face, 1)) / h
    return tau, uniform_faces(n)


def truncation_stretched(u, faces) -> Tuple[np.ndarray, np.ndarray]:
    """Truncation error of the 2nd-order cell-centred FV Laplacian, non-uniform mesh.

    The face derivative uses the distance between adjacent **cell centroids**
    (``(u_{i+1} - u_i)/(x^c_{i+1} - x^c_i)``) and the divergence uses the cell
    width ``V_i`` -- the textbook cell-centred scheme on an arbitrary mesh.
    Returns ``(tau, volume)`` with ``tau = L_h(u) - (exact cell average of u'')``.
    """
    u = np.asarray(u, dtype=float).ravel()
    faces = np.asarray(faces, dtype=float).ravel()
    n = u.size
    if faces.size != n + 1:
        raise ValueError("faces must hold n+1 values")
    xc = 0.5 * (faces[:-1] + faces[1:])
    volume = np.diff(faces)
    i = np.arange(n)
    j = (i + 1) % n
    dist = np.where(i + 1 < n, xc[j] - xc[i], xc[0] + 1.0 - xc[n - 1])
    face = (u[j] - u[i]) / dist
    tau = (face - np.roll(face, 1)) / volume
    return tau, volume


def _periodic_operator(n: int, variant: str, kappa: float):
    """Sparse periodic matrix of ``L_h - kappa I`` acting on cell averages."""
    import scipy.sparse as sp

    h = 1.0 / n
    if variant == "second_order":
        stencil = {0: -2.0 / h ** 2, 1: 1.0 / h ** 2, -1: 1.0 / h ** 2}
    elif variant == "fourth_order":
        c = 1.0 / (24.0 * h ** 2)
        stencil = {0: -54.0 * c, 1: 28.0 * c, -1: 28.0 * c, 2: -1.0 * c, -2: -1.0 * c}
    elif variant == "fourth_order_cellavg":
        c = 1.0 / (12.0 * h ** 2)
        stencil = {0: -30.0 * c, 1: 16.0 * c, -1: 16.0 * c, 2: -1.0 * c, -2: -1.0 * c}
    else:
        raise ValueError("only the second- and the two fourth-order operators are solvable")
    rows, cols, vals = [], [], []
    idx = np.arange(n)
    for off, coef in stencil.items():
        rows.append(idx)
        cols.append((idx + off) % n)
        vals.append(np.full(n, coef))
    rows, cols, vals = (np.concatenate(v) for v in (rows, cols, vals))
    vals = vals.copy()
    vals[rows == cols] -= kappa
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def truncation_ladder(variant: str, n_cells: Sequence[int] = (16, 32, 64), *,
                      cell_average: bool = True) -> Dict[str, object]:
    """Truncation-error ladder of a uniform-mesh operator.

    ``cell_average=True`` feeds the exact cell averages into the stencil (the
    admissible data for a cell-centred scheme); ``False`` feeds the exact cell
    centre point values, which is what a naive driver does.
    """
    formal = dict(second_order=2.0, fourth_order=4.0,
                  fourth_order_cellavg=4.0, shifted=1.0)[variant]
    spacings, errors, rows = [], [], []
    for n in n_cells:
        faces = uniform_faces(n)
        left, right = faces[:-1], faces[1:]
        volume = right - left
        data = mm_cell_average(left, right) if cell_average else mm_value(0.5 * (left + right))
        tau, _ = truncation_uniform(data, n, variant)
        exact_source = mm_cell_average_second_derivative(left, right)
        norms = cell_error_norms(tau - exact_source, volume)
        spacings.append(1.0 / n)
        errors.append(norms["l2"])
        rows.append(dict(n=int(n), h=1.0 / n, l2=norms["l2"], linf=norms["linf"]))
    study = loglog_order(spacings, errors, theory_order=formal)
    return dict(spacings=spacings, errors=errors, levels=rows,
                study=study.as_dict(), theory_order=formal, cell_average=cell_average)


def solution_error_ladder(n_cells: Sequence[int], variant: str = "second_order", *,
                          kappa: float = 1.0, reference: str = "cell_average") -> Dict[str, object]:
    """Solve ``u'' - kappa u = f`` with a cell-centred FV scheme and fit the error.

    ``f`` is the *exact* cell average of ``u'' - kappa u`` for the manufactured
    solution, and ``kappa > 0`` makes the matrix non-singular (a pure Laplacian
    has the constant null space, which makes the error defined only up to an
    additive constant -- another FV trap this harness removes rather than hides).

    ``reference`` selects the comparison field and is the point of the
    experiment:

    ``cell_average``  exact cell averages: admissible for a cell-centred scheme,
                      returns the formal order;
    ``point_value``   ``u(x_i)`` at the cell centres: injects ``h^2/24 u''`` and
                      caps the observable order at 2 (criteria ``C5``/``C6``).
    """
    import scipy.sparse.linalg as spla

    if reference not in ("cell_average", "point_value"):
        raise ValueError("reference must be 'cell_average' or 'point_value'")
    formal = 2.0 if variant == "second_order" else 4.0
    rows, spacings, errors = [], [], []
    for n in n_cells:
        faces = uniform_faces(n)
        left, right = faces[:-1], faces[1:]
        volume = right - left
        exact_avg = mm_cell_average(left, right)
        source = mm_cell_average_second_derivative(left, right) - kappa * exact_avg
        solved = spla.spsolve(_periodic_operator(n, variant, kappa), source)
        ref = exact_avg if reference == "cell_average" else mm_value(0.5 * (left + right))
        norms = cell_error_norms(solved - ref, volume)
        h = float(np.mean(volume))                     # (V/N)^(1/d) in 1-D
        spacings.append(h)
        errors.append(norms["l2"])
        rows.append(dict(n=int(n), h=h, l2=norms["l2"], l1=norms["l1"],
                         linf=norms["linf"], l2_unweighted=norms["l2_unweighted"]))
    study = loglog_order(spacings, errors, theory_order=formal)
    return dict(variant=variant, reference=reference, theory_order=formal,
                study=study.as_dict(), levels=rows)


def stretched_weights(n_cells: Sequence[int] = (32, 64, 128)) -> Dict[str, object]:
    """Density vs unweighted norm of the same truncation error on a stretched family."""
    spacings, weighted, unweighted, rows = [], [], [], []
    for n in n_cells:
        faces = stretched_faces(n)
        left, right = faces[:-1], faces[1:]
        exact_avg = mm_cell_average(left, right)
        tau, volume = truncation_stretched(exact_avg, faces)
        exact_source = mm_cell_average_second_derivative(left, right)
        norms = cell_error_norms(tau - exact_source, volume)
        h = float(np.mean(volume))
        spacings.append(h)
        weighted.append(norms["l2"])
        unweighted.append(norms["l2_unweighted"])
        rows.append(dict(n=int(n), h=h, l2=norms["l2"],
                         l2_unweighted=norms["l2_unweighted"],
                         ratio=norms["l2"] / norms["l2_unweighted"]))
    weighted_study = loglog_order(spacings, weighted)
    unweighted_study = loglog_order(spacings, unweighted)
    return dict(levels=rows,
                weighted=weighted_study.as_dict(),
                unweighted=unweighted_study.as_dict(),
                order_difference=abs(weighted_study.order - unweighted_study.order),
                magnitude_ratio=rows[-1]["ratio"])


# --------------------------------------------------------------------------- #
# verification driver
# --------------------------------------------------------------------------- #
def run() -> Dict[str, object]:
    """Run the whole cross-validation: estimator self-test first, then the ladders."""
    # ---- C1: the estimator recovers a synthetic order exactly -------------- #
    synthetic = {}
    for p in (1.0, 2.0, 3.0):
        h = np.array([1 / 8, 1 / 16, 1 / 32])
        synthetic[str(int(p))] = loglog_order(h, 0.7 * h ** p, theory_order=p).as_dict()

    # ---- C2: the guards withhold instead of inventing an order ------------ #
    guards = dict(
        non_monotone=loglog_order([0.5, 0.25, 0.125], [1e-2, 3e-2, 2e-3]).as_dict(),
        stagnant=loglog_order([0.5, 0.25, 0.125], [1e-2, 1e-3, 1e-3 * (1.0 - 1e-15)]).as_dict(),
        tiny_ratio=loglog_order([0.5, 0.4995, 0.125], [1e-2, 4e-3, 1e-3]).as_dict(),
    )

    tau_second = truncation_ladder("second_order", cell_average=True)
    tau_second_point = truncation_ladder("second_order", cell_average=False)
    tau_shifted = truncation_ladder("shifted", cell_average=True)
    sol_second = solution_error_ladder((16, 32, 64), "second_order", reference="cell_average")
    sol_fourth = solution_error_ladder((16, 32, 64), "fourth_order_cellavg",
                                       reference="cell_average")
    sol_fourth_point = solution_error_ladder((16, 32, 64), "fourth_order_cellavg",
                                             reference="point_value")
    sol_fourth_native = solution_error_ladder((16, 32, 64), "fourth_order",
                                              reference="cell_average")
    tau_fourth = truncation_ladder("fourth_order_cellavg", cell_average=True)
    tau_fourth_native = truncation_ladder("fourth_order", cell_average=True)
    constant_ratio = tau_second["errors"][-1] / tau_second_point["errors"][-1]

    # ---- C9: cross-check against the Richardson/GCI family ---------------- #
    from .richardson import richardson_gci

    values = tau_second["errors"]
    rich = richardson_gci(values[-1], values[-2], values[-3],
                          tuple(reversed(tau_second["spacings"])), theory_order=2.0)

    # ---- C10: the abscissa trap (fitting against cell count) ------------- #
    # A 2-D 2nd-order error on three isotropically refined grids: the true order
    # is 2 against the cell size h = L/sqrt(N) and 1 against the cell count N.
    counts = np.array([256.0, 1024.0, 4096.0])
    h_2d = counts ** -0.5
    e_2d = 0.5 * h_2d ** 2
    p_vs_h = loglog_order(h_2d, e_2d, theory_order=2.0)
    p_vs_count = abs(math.log(e_2d[0] / e_2d[1]) / math.log(counts[1] / counts[0]))
    count_ladder = loglog_order(counts, e_2d)
    abscissa = dict(vs_h=p_vs_h.as_dict(),
                    vs_count=dict(order=p_vs_count, expected=1.0),
                    count_ladder=count_ladder.as_dict())

    stretched = stretched_weights()

    criteria = []

    def add(cid, claim, measured, threshold, passed):
        criteria.append(dict(id=cid, claim=claim, measured=measured,
                             threshold=threshold,
                             verdict="PASS" if passed else "FAIL"))

    worst = max(abs(s["relative_deviation"]) for s in synthetic.values())
    add("C1", "loglog_order recovers a synthetic order p=1,2,3 from three grids",
        worst, "<= 1e-9 relative", worst <= 1e-9)

    expected_guards = dict(non_monotone="non-monotone", stagnant="stagnant",
                           tiny_ratio="refinement-ratio-too-small")
    add("C2", "non-monotone / stagnant / tiny-ratio ladders are WITHHELD, not fitted",
        {k: v["withheld_reason"] for k, v in guards.items()},
        "each ladder withheld by its own named guard",
        all(guards[k]["verdict"] == "WITHHELD"
            and guards[k]["withheld_reason"] == reason
            for k, reason in expected_guards.items()))

    add("C3", "2nd-order FV Laplacian: truncation-error order = 2 (three grids)",
        tau_second["study"]["order"], "within 15 % of 2",
        tau_second["study"]["verdict"] == "PASS")

    add("C4", "2nd-order FV Helmholtz solution error vs exact cell averages: order = 2",
        sol_second["study"]["order"], "within 15 % of 2",
        sol_second["study"]["verdict"] == "PASS")

    add("C5", "path A: 4th-order cell-average stencil + cell-average reference -> order 4",
        dict(solution=sol_fourth["study"]["order"], truncation=tau_fourth["study"]["order"]),
        "both within 15 % of 4",
        sol_fourth["study"]["verdict"] == "PASS" and tau_fourth["study"]["verdict"] == "PASS")

    masked = sol_fourth_point["study"]["order"]
    add("C6", "path B: same 4th-order solution measured at cell-centre POINT values reads 2",
        masked, "within 15 % of 2 and >15 % away from 4",
        abs(masked - 2.0) / 2.0 <= ORDER_TOLERANCE and abs(masked - 4.0) / 4.0 > ORDER_TOLERANCE)

    native = sol_fourth_native["study"]["order"]
    add("C7", "path C: textbook POINT-VALUE 4-point stencil fed cell averages reads 2, not 4",
        dict(solution=native, truncation=tau_fourth_native["study"]["order"]),
        "within 15 % of 2 and >15 % away from 4",
        abs(native - 2.0) / 2.0 <= ORDER_TOLERANCE and abs(native - 4.0) / 4.0 > ORDER_TOLERANCE)

    add("C8", "2nd-order truncation constant: cell-average / point-value = 2 (analytic)",
        constant_ratio, "2.00 +/- 2 %",
        abs(constant_ratio - POINT_VALUE_CONSTANT_FACTOR) <= 0.02 * POINT_VALUE_CONSTANT_FACTOR)

    add("C9", "displaced-flux negative control: truncation order = 1",
        tau_shifted["study"]["order"], "within 15 % of 1",
        tau_shifted["study"]["verdict"] == "PASS")

    add("C10", "log-log slope and Richardson/GCI observed order agree on the same data",
        dict(loglog=tau_second["study"]["order"], richardson=rich.order),
        "|dp|/p <= 15 %",
        abs(tau_second["study"]["order"] - rich.order) / 2.0 <= ORDER_TOLERANCE)

    add("C11", "abscissa: p=2 reads 2 vs h, p/d=1 vs cell count (estimator refuses a count)",
        dict(vs_h=abscissa["vs_h"]["order"], vs_count=abscissa["vs_count"]["order"],
             count_ladder=abscissa["count_ladder"]["withheld_reason"]),
        "2 vs h (within 15 %), 1 vs count (within 15 %), count ladder WITHHELD",
        abs(abscissa["vs_h"]["order"] - 2.0) / 2.0 <= ORDER_TOLERANCE
        and abs(abscissa["vs_count"]["order"] - 1.0) <= ORDER_TOLERANCE
        and abscissa["count_ladder"]["verdict"] == "WITHHELD")

    add("C12", "volume weighting is order-neutral on a self-similar stretched mesh family",
        dict(order_difference=stretched["order_difference"],
             magnitude_ratio=stretched["magnitude_ratio"]),
        "|dp| <= 0.05 (magnitude ratio reported, not asserted)",
        stretched["order_difference"] <= 0.05)

    return dict(
        theory=dict(module="tensorfvm.verification.loglog_order", min_ratio=MIN_RATIO,
                    order_tolerance=ORDER_TOLERANCE,
                    asymptotic_tolerance=ASYMPTOTIC_TOLERANCE),
        synthetic=synthetic, guards=guards,
        tau_second=tau_second, tau_second_point=tau_second_point, tau_shifted=tau_shifted,
        tau_fourth=tau_fourth, tau_fourth_native=tau_fourth_native,
        sol_second=sol_second, sol_fourth=sol_fourth, sol_fourth_point=sol_fourth_point,
        sol_fourth_native=sol_fourth_native,
        constant_ratio=constant_ratio,
        richardson=dict(observed_order=rich.order, convergence=rich.convergence,
                        fallback_reason=rich.fallback_reason, gci=rich.gci),
        abscissa=abscissa, stretched=stretched, criteria=criteria,
        all_pass=all(c["verdict"] == "PASS" for c in criteria),
    )


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def figure(payload: Dict[str, object], path: Path) -> Path:
    """Log--log error ladders plus observed-vs-formal order panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    series = (
        ("tau_second", "2nd-order truncation, cell averages", "o-", 2.0),
        ("tau_second_point", "2nd-order truncation, point values", "s--", 2.0),
        ("tau_shifted", "displaced flux (1st-order control)", "^-", 1.0),
        ("sol_second", "2nd-order solution, cell averages", "d-", 2.0),
        ("sol_fourth", "4th-order solution, cell-average stencil", "o-", 4.0),
        ("sol_fourth_point", "4th-order solution, point-value reference", "s--", 4.0),
        ("sol_fourth_native", "4th-order solution, point-value stencil (path C)", "v:", 4.0),
    )
    for key, label, style, formal in series:
        rows = payload[key]["levels"]
        hs = [r["h"] for r in rows]
        es = [r["l2"] for r in rows]
        ax.loglog(hs, es, style, label=label, lw=1.3, ms=5)
    href = np.array([1 / 64, 1 / 16])
    anchor = payload["sol_second"]["levels"][-1]["l2"]
    for power, style in ((1, ":"), (2, "--"), (4, "-.")):
        ax.loglog(href, anchor * (href / href[-1]) ** power, style, color="k", lw=0.8,
                  label=f"$h^{power}$")
    ax.set_xlabel("cell size $h$")
    ax.set_ylabel(r"volume-weighted $L_2$ error")
    ax.set_title("log--log error ladders (cell-centred FV)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    labels = ["2nd\ntrunc.", "2nd\nsolution", "4th\npath A", "4th\npath B", "4th\npath C",
              "1st\ncontrol"]
    measured = [payload["tau_second"]["study"]["order"],
                payload["sol_second"]["study"]["order"],
                payload["sol_fourth"]["study"]["order"],
                payload["sol_fourth_point"]["study"]["order"],
                payload["sol_fourth_native"]["study"]["order"],
                payload["tau_shifted"]["study"]["order"]]
    formal = [2, 2, 4, 4, 4, 1]
    xpos = np.arange(len(labels))
    ax2.bar(xpos - 0.2, formal, 0.4, label="formal order", color="0.7")
    ax2.bar(xpos + 0.2, measured, 0.4, label="observed (log--log slope)", color="C3")
    for x, m in zip(xpos, measured):
        ax2.text(x + 0.2, m + 0.06, f"{m:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("order")
    ax2.set_title("observed vs formal order")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or [])
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[3]
    out = root / "docs" / "verification-three-grid-loglog-cellcentre"
    out.mkdir(parents=True, exist_ok=True)
    payload = run()
    (out / "summary.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    figure(payload, out / "fig_loglog_order.png")
    verdict = "ALL PASS" if payload["all_pass"] else "FAIL PRESENT"
    print(f"criteria: {len(payload['criteria'])} -> {verdict}")
    for c in payload["criteria"]:
        print(f"  {c['id']:>4} {c['verdict']:<5} {c['claim']}")
    print(f"artefacts: {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())