"""3-D periodic-hill solver (streamwise- and spanwise-periodic).

Round 33 (mission 23249, TensorFVM).  Companion to ``periodic_hill3d`` (mesh).
A CPU reference fractional-step (predictor / pressure-projection) incompressible
solver on the periodic-extruded terrain-following hill mesh.  It is the 3-D
analogue of the 2-D ``periodic_hill`` solver:

* **no inlet/outlet** -- the streamwise direction is periodic, so the cylinder
  case's inlet/outlet pair is replaced by periodic wrap;
* **body-force integral controller** -- a uniform streamwise body force holds
  the bulk velocity above the crest at the target value (the 2-D solver's
  ``body_force_relaxation``), as ERCOFTAC prescribes;
* **no-slip** on the hill surface and the flat top wall;
* an optional Wale SGS term for the turbulent runs (``sgs``).

It is *not* a qualified Re=2800 LES: the SGS constant, the upwind fraction and
the time step need qualification on a refined grid before any number from it
is graded against the DNS reference.  The flat-bottom limit (``height_m = 0``)
must reproduce 3-D Poiseuille flow; that is the correctness gate.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .body_fitted_transient3d import TransientSettings
from .periodic_hill3d import BodyFittedHillMesh3D, PeriodicHill3DConfig


class BodyFittedTransient3DHill:
    """Fractional-step solver for the 3-D periodic hill (x, z periodic)."""

    def __init__(self, config: PeriodicHill3DConfig,
                 settings: TransientSettings | None = None):
        self.config = config
        self.settings = settings or TransientSettings()
        c = config
        self.mesh = BodyFittedHillMesh3D.build(c, torch.device("cpu"))
        m = self.mesh
        self.centers = m.centers.reshape(-1, 3)
        self.volumes = m.cell_volumes.flatten()
        self.N = len(self.volumes)
        count = c.nx * c.ny
        zo = torch.arange(self.N)
        zn = (zo + count) % self.N
        zs = torch.zeros((self.N, 3), dtype=torch.float64)
        zs[:, 2] = m.spanwise_face_areas[1:].flatten()
        zc = self.centers.clone()
        zc[:, 2] += m.dz / 2
        self.owner = torch.cat((m.lateral_owner.flatten(), zo))
        self.neighbor = torch.cat((m.lateral_neighbor.flatten(), zn))
        self.area = torch.cat((m.lateral_face_area_vectors.reshape(-1, 3), zs))
        self.fc = torch.cat((m.lateral_face_centers.reshape(-1, 3), zc))
        labs = list(m.lateral_labels) * c.nz + ["interior"] * self.N
        self.interior = self.neighbor >= 0
        self.boundary = ~self.interior
        self.wall = torch.tensor([x in ("hill", "top") for x in labs])
        self.outlet = torch.zeros(self.N, dtype=torch.bool)
        self.slip = torch.zeros(self.N, dtype=torch.bool)
        self.prescribed = torch.zeros(self.N, dtype=torch.bool)
        self.safe = self.neighbor.clamp_min(0)
        self.d = self.centers[self.safe] - self.centers[self.owner]
        # minimum image in the streamwise (periodic) direction: the seam face
        # connects cells a full period apart in raw coordinates
        self.d[:, 0] = self.d[:, 0] - c.length_m * torch.round(self.d[:, 0] / c.length_m)
        self.d[self.boundary] = (self.fc[self.boundary]
                                 - self.centers[self.owner[self.boundary]])
        self.d[-self.N:, 2] = m.dz
        self.owner_offset = self.fc - self.centers[self.owner]
        self.neighbor_offset = self.fc - self.centers[self.safe]
        self.neighbor_offset[-count:, 2] = -m.dz / 2
        projection = (self.area * self.d).sum(1)
        if not bool((projection > 0).all()):
            raise ValueError("nonpositive owner-neighbor face projection")
        self.coeff = (self.area * self.area).sum(1) / projection
        self.tangent = self.area - self.coeff[:, None] * self.d
        self.area_mag = torch.linalg.vector_norm(self.area, dim=1)
        self.distance = torch.linalg.vector_norm(self.d, dim=1)
        self.pressure_weight = ((self.owner_offset * self.d).sum(1)
                               / self.distance.square())
        self.pressure_skew = (self.owner_offset
                             - self.pressure_weight[:, None] * self.d)
        self.velocity = torch.zeros((self.N, 3), dtype=torch.float64)
        self.velocity[:, 0] = c.bulk_velocity
        self.pressure = torch.zeros(self.N, dtype=torch.float64)
        self.n_faces = len(self.owner)
        self.flux = torch.zeros(self.n_faces, dtype=torch.float64)
        self.time = 0.0
        self.body_force = torch.zeros(3, dtype=torch.float64)
        self._bulk_integral = 0.0
        self.history = []
        self.forces = []
        self.pressure_neumann = torch.zeros(self.N, dtype=torch.float64)
        self.pressure_outlet_values = torch.zeros(self.N, dtype=torch.float64)
        self._build_ls()
        self._build_pressure()

    # -- bulk-velocity band (ERCOFTAC: H to 3.035 H above the crest) -------- #
    @property
    def _band_mask(self):
        c = self.config
        y = self.centers[:, 1]
        return (y >= c.height_m) & (y <= 3.035 * c.height_m)

    def bulk_velocity(self):
        """Streamwise bulk velocity over the ERCOFTAC band, volume-weighted."""
        mask = self._band_mask
        vol = self.volumes[mask]
        return float((self.velocity[mask, 0] * vol).sum() / vol.sum())

    def update_body_force(self):
        """Proportional-integral controller on the band bulk velocity."""
        c = self.config
        err = c.bulk_velocity - self.bulk_velocity()
        self._bulk_integral += err * c.time_step_s
        gain = c.density_kg_m3 * c.bulk_velocity / c.height_m
        self.body_force[0] = gain * (c.body_force_relaxation * err
                                     + (1.0 - c.body_force_relaxation)
                                     * self._bulk_integral / c.time_step_s)
        return float(self.body_force[0])

    # -- least-squares gradient (pattern from the cylinder solver) ---------- #
    def _build_ls(self):
        w = 1.0 / self.distance.square().clamp_min(1e-300)
        normal = torch.zeros((self.N, 3, 3), dtype=torch.float64)
        outer = self.d[:, :, None] * self.d[:, None, :] * w[:, None, None]
        normal.index_add_(0, self.owner, outer)
        normal.index_add_(0, self.neighbor[self.interior], outer[self.interior])
        self.ls_inverse = torch.linalg.inv(normal)
        self.ls_weight = w

    def gradient(self, field, pressure=False):
        scalar = field.ndim == 1
        f = field[:, None] if scalar else field
        delta = f[self.safe] - f[self.owner]
        if pressure:
            delta[self.boundary] = (self.pressure_neumann[self.boundary, None]
                                    * self.distance[self.boundary, None])
        else:
            target = torch.zeros((int(self.boundary.sum()), f.shape[1]),
                             dtype=f.dtype)
            target[self.wall[self.boundary]] = 0
            delta[self.boundary] = target - f[self.owner[self.boundary]]
        rhs = torch.zeros((self.N, f.shape[1], 3), dtype=f.dtype)
        val = (delta[:, :, None] * self.d[:, None, :]
               * self.ls_weight[:, None, None])
        rhs.index_add_(0, self.owner, val)
        rhs.index_add_(0, self.neighbor[self.interior], val[self.interior])
        g = torch.einsum("nij,ncj->nci", self.ls_inverse, rhs)
        return g[:, 0] if scalar else g

    def _build_pressure(self):
        o = self.owner.numpy(); n = self.neighbor.numpy()
        a = self.coeff.numpy(); i = self.interior.numpy()
        row = np.r_[o[i], n[i], o[i], n[i]]
        col = np.r_[o[i], n[i], n[i], o[i]]
        val = np.r_[a[i], a[i], -a[i], -a[i]]
        self.pressure_matrix = coo_matrix((val, (row, col)),
                                         shape=(self.N, self.N)).tocsc()
        self.pressure_lu = splu(self.pressure_matrix)

    def integrate(self, face_values):
        if face_values.ndim == 1:
            out = torch.zeros(self.N, dtype=torch.float64)
        else:
            out = torch.zeros((self.N, face_values.shape[1]), dtype=torch.float64)
        out.index_add_(0, self.owner, face_values)
        out.index_add_(0, self.neighbor[self.interior],
                       face_values[self.interior])
        return out

    def face_values(self, velocity):
        values = 0.5 * (velocity[self.owner] + velocity[self.safe])
        values[self.wall] = 0
        return values

    def predict_flux(self, velocity):
        return (self.face_values(velocity) * self.area).sum(1)

    def momentum_predictor_flux(self, tentative_velocity):
        return (self.flux + self.predict_flux(tentative_velocity)
                - self.predict_flux(self.velocity))

    def pressure_flux_gradient(self, p):
        g = self.gradient(p, pressure=True)
        gf = 0.5 * (g[self.owner] + g[self.safe])
        gf[self.boundary] = g[self.owner[self.boundary]]
        diff = p[self.safe] - p[self.owner]
        diff[self.boundary] = 0
        q = self.coeff * diff + (gf * self.tangent).sum(1)
        q[self.boundary] = (self.pressure_neumann[self.boundary]
                            * self.area_mag[self.boundary])
        return q

    def eddy_viscosity(self, gradient):
        if self.settings.sgs == "none":
            return torch.zeros(self.N, dtype=torch.float64)
        s = 0.5 * (gradient + gradient.transpose(1, 2))
        g2 = gradient @ gradient
        sd = 0.5 * (g2 + g2.transpose(1, 2))
        sd = sd - (torch.eye(3, dtype=torch.float64)[None]
                       * sd.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] / 3)
        ss = s.square().sum((1, 2)); dd = sd.square().sum((1, 2))
        numerator = dd.pow(1.5); denominator = ss.pow(2.5) + dd.pow(1.25)
        return ((self.settings.wale_constant * self.volumes.pow(1 / 3)).square()
                * numerator / denominator.clamp_min(1e-60))

    def transport(self, velocity, flux):
        c = self.config
        g = self.gradient(velocity)
        nu = c.nu + self.eddy_viscosity(g)
        gf = 0.5 * (g[self.owner] + g[self.safe])
        gf[self.boundary] = g[self.owner[self.boundary]]
        mu = c.density_kg_m3 * 0.5 * (nu[self.owner] + nu[self.safe])
        mu[self.wall] = c.density_kg_m3 * c.nu
        delta = velocity[self.safe] - velocity[self.owner]
        delta[self.wall] = -velocity[self.owner[self.wall]]
        defect = delta - torch.einsum("fcj,fj->fc", gf, self.d)
        gf += (defect[:, :, None] * self.d[:, None, :]
                / self.distance.square()[:, None, None])
        stress = mu[:, None, None] * (
            gf + gf.transpose(1, 2)
            - 2 / 3 * torch.eye(3, dtype=torch.float64)[None]
            * gf.diagonal(dim1=1, dim2=2).sum(1)[:, None, None])
        viscous = torch.einsum("fij,fj->fi", stress, self.area)
        up = velocity[self.owner].clone()
        reverse = flux < 0
        up[reverse & self.interior] = velocity[self.neighbor[reverse & self.interior]]
        central = self.face_values(velocity)
        transported = ((1 - self.settings.upwind_fraction) * central
                        + self.settings.upwind_fraction * up)
        convection = c.density_kg_m3 * flux[:, None] * transported
        return convection, viscous, nu

    @torch.no_grad()
    def step(self):
        c = self.config
        self.update_body_force()
        u0 = self.velocity.clone()
        conv, visc, nu = self.transport(u0, self.flux)
        face_nu = 0.5 * (nu[self.owner] + nu[self.safe])
        face_nu[self.wall] = c.nu
        rate = torch.zeros(self.N, dtype=torch.float64)
        face_rate = face_nu * self.coeff
        rate.index_add_(0, self.owner, face_rate)
        rate.index_add_(0, self.neighbor[self.interior], face_rate[self.interior])
        diffusion_cfl = float((2 * c.time_step_s * rate / self.volumes).max())
        if diffusion_cfl > self.settings.diffusion_cfl_limit:
            raise RuntimeError(f"explicit diffusion dt screen failed: {diffusion_cfl:.6g}")
        bf = self.body_force
        tentative = u0 + c.time_step_s / c.density_kg_m3 * (
            self.integrate(visc - conv) / self.volumes[:, None] + bf[None, :])
        p, q, it, div = self.project(self.momentum_predictor_flux(tentative))
        facep = p[:, None] * self.area
        u = tentative - c.time_step_s / c.density_kg_m3 * self.integrate(facep) / self.volumes[:, None]
        cfl = float((c.time_step_s * (q.abs()[self.owner]
                 + q.abs()[self.neighbor[self.interior]]).max()
                 / (2 * self.volumes)).max()) if False else float(
            (c.time_step_s * q.abs() / (2 * self.volumes)).max())
        if not bool(torch.isfinite(u).all()) or cfl >= 1:
            raise RuntimeError(f"nonfinite field or CFL >= 1: {cfl}")
        self.velocity = u
        self.pressure = p
        self.flux = q
        self.time += c.time_step_s
        row = dict(step=len(self.history) + 1, time=self.time, continuity=div,
                   projection_iterations=it, cfl=cfl, diffusion_cfl=diffusion_cfl,
                   bulk_velocity=self.bulk_velocity(),
                   max_speed=float(torch.linalg.vector_norm(u, dim=1).max()),
                   body_force_x=float(bf[0]))
        self.history.append(row)
        return row

    def project(self, tentative_flux, *, correction_scale=None):
        c = self.config
        dt = c.time_step_s / c.density_kg_m3 if correction_scale is None else float(correction_scale)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite pressure correction scale required")
        rhs = -self.integrate(tentative_flux) / dt
        p = self.pressure.clone()
        before = None
        for k in range(self.settings.nonorthogonal_iterations):
            nonorth = self.pressure_flux_gradient(p)
            diff = p[self.safe] - p[self.owner]
            diff[self.boundary] = 0
            nonorth -= self.coeff * diff
            p = torch.from_numpy(self.pressure_lu.solve(
                (rhs + self.integrate(nonorth)).numpy()))
            q = tentative_flux - dt * self.pressure_flux_gradient(p)
            residual = float((self.integrate(q) / self.volumes).abs().max())
            if before is None:
                before = residual
            if residual < self.settings.pressure_tolerance:
                return p, q, k + 1, residual
            if not np.isfinite(residual):
                break
        raise RuntimeError(f"nonorthogonal pressure projection failed: "
                           f"max divergence {residual:.6g}, first {before:.6g}")


__all__ = ["BodyFittedTransient3DHill"]
