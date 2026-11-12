"""1-D shallow-water solver over an arbitrary bed (well-balanced, wet/dry).

Why this module exists
----------------------
``shallow_water.swe1d`` integrates the depth-averaged equations on a **flat**
bed only (the module says so, and the Ritter/Stoker benchmark legitimately
needs nothing more).  Every practical shallow-water question -- harbour
seiching over a dredged channel, a tsunami running up a slope, a dam break
over a sill -- needs the bed-slope source term, and the source term is exactly
where naive schemes fail: the hydrostatic balance ``h + z = const, u = 0``
(*lake at rest*) is a **steady exact solution** of the continuous equations,
so a correct discretisation must keep it at rest to round-off for every bed,
and a scheme that does not is silently manufacturing spurious currents of
order ``g (dz/dx) dt`` everywhere.

Method
------
Audusse, Bouchut, Bristeau, Klein & Perthame (2004, *SIAM J. Sci. Comput.*
**25**, 2050, doi:10.1137/S1064827503426693) hydrostatic reconstruction:

* MUSCL reconstruction of the *water surface* ``eta = h + z`` and of ``u``
  (minmod; the two boundary cells get zero slope), piecewise-constant bed;
* at a face, ``z_f = max(z_L, z_R)`` and the two **starred** depths are
  ``h*_side = max(0, eta_side - z_f)``;
* the HLL flux of ``(h, hu)`` is evaluated on the starred states;
* the momentum flux seen by a cell carries the pressure correction
  ``+ g/2 (h_side^2 - h*_side^2)`` on its own side of the face.

The correction terms are *not* the same on the two sides of a face, so the
scheme is conservative in water volume and non-conservative in momentum --
that is the price of the well-balanced property (the bed legitimately exerts a
force), and it is what Audusse et al. prescribe.

Lake-at-rest proof for this implementation: ``eta`` is constant, so minmod
returns zero slopes, ``eta_side = eta`` and ``u = 0``.  Then
``h*_L = h*_R = eta - z_f``, so the HLL mass flux is 0 and the HLL momentum
flux is ``g/2 (eta - z_f)^2``; the correction makes the flux a cell sees at
either of its faces ``g/2 (eta - z_side)^2 = g/2 h_i^2``, identical on both
faces, hence ``d(hu)/dt = 0``.  Dry cells stay dry because
``eta_side - z_f <= 0`` there.

``well_balanced=False`` builds the *negative control* used by the verification
driver: no reconstruction of the star states, no pressure correction, and the
bed slope enters as the centred source ``-g h (dz/dx)_central``.  It is kept
in-tree on purpose -- a gate that cannot fail proves nothing -- and it is
never used by a benchmark case.

Reference for the wet/dry treatment: Audusse et al. (2004) §3; Toro (2009)
ch. 13; Xing & Shu (2011, *J. Comput. Phys.* **230**, 6037) for the
well-balanced WENO/MUSCL framework this reconstruction belongs to.
"""

from __future__ import annotations

import numpy as np

from .swe1d import ShallowWater1D

__all__ = ["ShallowWaterBed1D"]


class ShallowWaterBed1D(ShallowWater1D):
    """Well-balanced 1-D shallow-water solver with a bed ``z(x)``.

    Parameters
    ----------
    bed : array_like, optional
        Cell-centred bed elevation, shape ``(n,)``.  Defaults to a flat bed at
        ``z = 0``, in which case every operator reduces to the flat-bed solver
        of :class:`~tensorfvm.shallow_water.swe1d.ShallowWater1D` (this is
        checked bit-for-bit by ``tests/test_shallow_water_well_balanced.py``).
    well_balanced : bool, optional
        ``True`` (default) selects the hydrostatic-reconstruction scheme;
        ``False`` selects the naive centred-source negative control.
    """

    def __init__(self, n: int, x_min: float, x_max: float, bed=None,
                 gravity: float = 9.81, cfl: float = 0.4, order: int = 2,
                 dry_depth: float = 1e-12, well_balanced: bool = True,
                 manning_n: float = 0.0, slope: float = 0.0):
        super().__init__(n, x_min, x_max, gravity, cfl, order, dry_depth,
                         manning_n=manning_n, slope=slope)
        self.bed = np.zeros(self.n)
        if bed is not None:
            self.set_bed(bed)
        self.well_balanced = bool(well_balanced)
        self._eta0 = self.water_surface().copy()

    # ------------------------------------------------------------------ bed
    def set_bed(self, bed) -> None:
        """Install the cell-centred bed elevation ``z``."""
        bed = np.asarray(bed, dtype=float)
        if bed.shape != (self.n,):
            raise ValueError(f"bed must have shape ({self.n},), got {bed.shape}")
        if not np.all(np.isfinite(bed)):
            raise ValueError("bed elevation must be finite")
        self.bed = bed.copy()

    def set_state(self, h, u=None, bed=None) -> None:
        """Set the depth (and optional velocity, and optional bed)."""
        if bed is not None:
            self.set_bed(bed)
        super().set_state(h, u)
        self._eta0 = self.water_surface().copy()

    def water_surface(self) -> np.ndarray:
        """``eta = h + z`` (the quantity a lake at rest holds constant)."""
        return self.h + self.bed

    # ------------------------------------------------------ reconstruction
    def _bed_face_states(self):
        """Reconstructed ``(hL, uL, hR, uR, hsL, hsR)`` on the ``n + 1`` faces.

        ``hL``/``hR`` are the reconstructed depths relative to the cell's own
        face bed (they feed the pressure correction); ``hsL``/``hsR`` are the
        hydrostatic-reconstruction star depths (they feed the Riemann solver).
        """
        eta = self.h + self.bed
        _, u = self.primitive()
        se = self._minmod_slope(eta)
        su = self._minmod_slope(u)
        n = self.n
        half = 0.5 * self.dx

        etaL = np.empty(n + 1)
        etaR = np.empty(n + 1)
        etaL[0], etaR[0] = eta[0], eta[0]
        etaL[-1], etaR[-1] = eta[-1], eta[-1]
        etaL[1:-1] = eta[:-1] + half * se[:-1]
        etaR[1:-1] = eta[1:] - half * se[1:]

        uL = np.empty(n + 1)
        uR = np.empty(n + 1)
        uL[0], uR[0] = u[0], u[0]
        uL[-1], uR[-1] = u[-1], u[-1]
        uL[1:-1] = u[:-1] + half * su[:-1]
        uR[1:-1] = u[1:] - half * su[1:]

        zL = np.empty(n + 1)
        zR = np.empty(n + 1)
        zL[0] = zR[0] = self.bed[0]
        zL[-1] = zR[-1] = self.bed[-1]
        zL[1:-1] = self.bed[:-1]
        zR[1:-1] = self.bed[1:]

        if self.well_balanced:
            z_face = np.maximum(zL, zR)
        else:
            z_face = 0.5 * (zL + zR)

        hL = np.maximum(etaL - zL, 0.0)
        hR = np.maximum(etaR - zR, 0.0)
        if self.well_balanced:
            hsL = np.maximum(etaL - z_face, 0.0)
            hsR = np.maximum(etaR - z_face, 0.0)
        else:
            hsL, hsR = hL, hR

        uL = np.where(hsL > self.dry_depth, uL, 0.0)
        uR = np.where(hsR > self.dry_depth, uR, 0.0)
        return hL, uL, hR, uR, hsL, hsR

    # ------------------------------------------------------------- residual
    def rhs(self, h: np.ndarray, hu: np.ndarray, dt: float | None = None):
        """Semi-discrete right-hand side ``d/dt (h, hu)`` with bed terms.

        ``dt`` (optional) is forwarded to the bottom-friction reversal guard;
        with ``manning_n == 0`` and ``slope == 0`` nothing is added.
        """
        saved = self.h, self.hu
        self.h, self.hu = h, hu
        try:
            hL, uL, hR, uR, hsL, hsR = self._bed_face_states()
        finally:
            self.h, self.hu = saved

        flux_h, flux_hu = self._hll_flux(hsL, uL, hsR, uR)
        dh = -(flux_h[1:] - flux_h[:-1]) / self.dx

        if self.well_balanced:
            g = self.gravity
            # pressure correction on the cell's own side of each face:
            # cell i sees ``corrL`` on its right face and ``corrR`` on its left.
            corr_l = 0.5 * g * (hL * hL - hsL * hsL)
            corr_r = 0.5 * g * (hR * hR - hsR * hsR)
            # cell i sees the left-side correction on its right face and the
            # right-side correction on its left face.
            right = flux_hu + corr_l
            left = flux_hu + corr_r
            dhu = -(right[1:] - left[:-1]) / self.dx
        else:
            g = self.gravity
            dz = np.empty(self.n)
            dz[1:-1] = (self.bed[2:] - self.bed[:-2]) / (2.0 * self.dx)
            dz[0] = (self.bed[1] - self.bed[0]) / self.dx
            dz[-1] = (self.bed[-1] - self.bed[-2]) / self.dx
            dhu = -(flux_hu[1:] - flux_hu[:-1]) / self.dx - g * h * dz
        if self.has_sources:
            dhu = dhu + self._source_momentum(h, hu, dt)
        return dh, dhu

    # ---------------------------------------------------------- diagnostics
    def max_spurious_velocity(self) -> float:
        """``max |u|`` -- the lake-at-rest thermometer."""
        return float(np.max(np.abs(self.primitive()[1])))

    def surface_drift(self) -> float:
        """``max |eta - eta_0|`` against the initial surface."""
        return float(np.max(np.abs(self.water_surface() - self._eta0)))

    def lake_at_rest_state(self, surface: float) -> None:
        """Initialise a lake at rest with the given free-surface elevation."""
        surface = float(surface)
        h = np.maximum(surface - self.bed, 0.0)
        self.set_state(h, None)
        self._eta0 = self.water_surface().copy()

    def summary(self) -> dict:
        out = super().summary()
        out.update(
            bed_min=float(self.bed.min()),
            bed_max=float(self.bed.max()),
            well_balanced=self.well_balanced,
        )
        return out