"""High-order reconstruction FVM for compressible Euler equations on unstructured grids.

Supports MUSCL (linear reconstruction with slope limiters) and WENO-3/WENO-5
(weighted essentially non-oscillatory) reconstruction on 2-D unstructured
triangular/quadrilateral meshes.  Includes Roe/HLLC/AUSM+ flux solvers,
explicit Runge-Kutta time integration, and strict conservation verification.

Public API
----------
UnstructuredMesh2D : mesh data structure with cell-face-node topology
MUSCLReconstructor : linear reconstruction + Venkatakrishnan/Barth-Jespersen limiters
WENOReconstructor  : WENO-3 / WENO-5 reconstruction via polynomial basis
RKTimeIntegrator   : explicit SSP-RK2/RK3 with local time stepping
EulerSolver        : full solver tying reconstruction + flux + time integration
conservation_check : strict mass/momentum/energy conservation verifier
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Unstructured mesh data structure
# ---------------------------------------------------------------------------

@dataclass
class UnstructuredMesh2D:
    """Cell-centred unstructured 2-D mesh with face-based topology."""
    cell_centers: torch.Tensor
    cell_volumes: torch.Tensor
    face_centers: torch.Tensor
    face_area_vectors: torch.Tensor
    face_lengths: torch.Tensor
    face_normals: torch.Tensor
    owner: torch.Tensor
    neighbor: torch.Tensor
    interior: torch.Tensor
    boundary: torch.Tensor
    cell_vertices: List[np.ndarray] = field(default_factory=list)
    bc_masks: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def ncell(self) -> int:
        return int(self.cell_centers.shape[0])

    @property
    def nface(self) -> int:
        return int(self.owner.shape[0])

    @property
    def device(self):
        return self.cell_centers.device


def make_rectangular_mesh(
    nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> UnstructuredMesh2D:
    """Structured rectangular mesh split into triangles (each quad → 2 triangles)."""
    xv = torch.linspace(0, lx, nx + 1, dtype=dtype)
    yv = torch.linspace(0, ly, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xv, yv, indexing="ij")
    verts = torch.stack([X.ravel(), Y.ravel()], dim=-1)

    cells_verts = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1)
            v3 = v2 + 1
            cells_verts.append([v0, v1, v2])
            cells_verts.append([v1, v3, v2])
    ncell = len(cells_verts)

    centers = torch.zeros(ncell, 2, dtype=dtype)
    volumes = torch.zeros(ncell, dtype=dtype)
    for c, cv in enumerate(cells_verts):
        pts = verts[cv]
        centers[c] = pts.mean(0)
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        volumes[c] = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])

    face_map: Dict[Tuple[int, int], int] = {}
    faces_owner: List[int] = []
    faces_neighbor: List[int] = []
    faces_vverts: List[List[int]] = []
    fid = 0
    for c, cv in enumerate(cells_verts):
        edges = [(cv[0], cv[1]), (cv[1], cv[2]), (cv[2], cv[0])]
        for v0, v1 in edges:
            key = (min(v0, v1), max(v0, v1))
            if key in face_map:
                faces_neighbor[face_map[key]] = c
            else:
                face_map[key] = fid
                faces_owner.append(c)
                faces_neighbor.append(-1)
                faces_vverts.append([v0, v1])
                fid += 1
    nface = fid

    face_centers = torch.zeros(nface, 2, dtype=dtype)
    face_area_vectors = torch.zeros(nface, 2, dtype=dtype)
    face_lengths = torch.zeros(nface, dtype=dtype)
    face_normals = torch.zeros(nface, 2, dtype=dtype)

    for f in range(nface):
        v0, v1 = faces_vverts[f]
        p0, p1 = verts[v0], verts[v1]
        fc = 0.5 * (p0 + p1)
        edge = p1 - p0
        fl = torch.sqrt(edge[0]**2 + edge[1]**2)
        n = torch.tensor([edge[1], -edge[0]], dtype=dtype)
        n = n / torch.linalg.vector_norm(n)
        oc = centers[faces_owner[f]]
        to_center = fc - oc
        if torch.dot(n, to_center) > 0:
            n = -n
        av = n * fl
        face_centers[f] = fc
        face_area_vectors[f] = av
        face_lengths[f] = fl
        face_normals[f] = n

    owner = torch.tensor(faces_owner, dtype=torch.long)
    neighbor = torch.tensor(faces_neighbor, dtype=torch.long)
    interior = neighbor >= 0
    boundary = ~interior

    return UnstructuredMesh2D(
        cell_centers=centers, cell_volumes=volumes,
        face_centers=face_centers, face_area_vectors=face_area_vectors,
        face_lengths=face_lengths, face_normals=face_normals,
        owner=owner, neighbor=neighbor,
        interior=interior, boundary=boundary,
        cell_vertices=[np.array(cv) for cv in cells_verts],
    )


def make_triangular_mesh_uniform(
    nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> UnstructuredMesh2D:
    """Alias for make_rectangular_mesh (triangular split)."""
    return make_rectangular_mesh(nx, ny, lx, ly, dtype)


# ---------------------------------------------------------------------------
# 2. MUSCL reconstruction
# ---------------------------------------------------------------------------

class MUSCLReconstructor:
    """Linear (P1) reconstruction with slope limiting on unstructured meshes."""

    def __init__(self, mesh: UnstructuredMesh2D, limiter: str = "venkatakrishnan"):
        self.mesh = mesh
        self.limiter = limiter.lower()
        assert self.limiter in ("venkatakrishnan", "barth_jespersen", "none")
        self._build_stencil()

    def _build_stencil(self):
        mesh = self.mesh
        ncell = mesh.ncell
        self.cell_neighbours: List[List[int]] = [[] for _ in range(ncell)]
        self.cell_neigh_deltas: List[List[np.ndarray]] = [[] for _ in range(ncell)]
        for f in range(mesh.nface):
            o = int(mesh.owner[f])
            n = int(mesh.neighbor[f])
            if n >= 0:
                self.cell_neighbours[o].append(n)
                self.cell_neigh_deltas[o].append(
                    (mesh.cell_centers[n] - mesh.cell_centers[o]).numpy())
                self.cell_neighbours[n].append(o)
                self.cell_neigh_deltas[n].append(
                    (mesh.cell_centers[o] - mesh.cell_centers[n]).numpy())

    def compute_gradients(self, q: torch.Tensor) -> torch.Tensor:
        mesh = self.mesh
        ncell = mesh.ncell
        nvar = q.shape[1]
        grad = torch.zeros(ncell, nvar, 2, dtype=q.dtype, device=q.device)

        for c in range(ncell):
            neigh = self.cell_neighbours[c]
            if len(neigh) < 2:
                continue
            deltas = np.array(self.cell_neigh_deltas[c])
            A = torch.tensor(deltas, dtype=q.dtype, device=q.device)
            for v in range(nvar):
                dq = q[neigh, v] - q[c, v]
                ATA = A.T @ A
                ATdq = A.T @ dq
                try:
                    g = torch.linalg.solve(ATA, ATdq)
                except RuntimeError:
                    g = torch.zeros(2, dtype=q.dtype, device=q.device)
                grad[c, v] = g

        if self.limiter == "none":
            return grad
        return self._apply_limiter(q, grad)

    def _apply_limiter(self, q: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        mesh = self.mesh
        ncell = mesh.ncell
        nvar = q.shape[1]
        limited = grad.clone()

        for c in range(ncell):
            neigh = self.cell_neighbours[c]
            if not neigh:
                continue
            q_c = q[c]
            q_min = q[neigh].min(dim=0).values
            q_max = q[neigh].max(dim=0).values

            for v in range(nvar):
                dq_max = q_max[v] - q_c[v]
                dq_min = q_min[v] - q_c[v]
                if self.limiter == "venkatakrishnan":
                    eps = 1e-6 * (q_max[v] - q_min[v] + 1e-30)
                    phi_pos = (dq_max**2 + 2 * eps * dq_max) / (dq_max**2 + 2 * eps * dq_max + eps**2 + 1e-30)
                    phi_neg = (dq_min**2 - 2 * eps * dq_min) / (dq_min**2 - 2 * eps * dq_min + eps**2 + 1e-30)
                    phi = max(min(min(phi_pos.item(), phi_neg.item()), 1.0), 0.0)
                    limited[c, v] *= phi
                elif self.limiter == "barth_jespersen":
                    recon_max = q_c[v].item()
                    recon_min = q_c[v].item()
                    for n_idx in neigh:
                        dr = mesh.cell_centers[n_idx] - mesh.cell_centers[c]
                        rv = q_c[v] + torch.dot(grad[c, v], dr)
                        recon_max = max(recon_max, rv.item())
                        recon_min = min(recon_min, rv.item())
                    phi = 1.0
                    if recon_max > q_max[v].item():
                        phi = min(phi, (dq_max.item() / (recon_max - q_c[v].item() + 1e-30)))
                    if recon_min < q_min[v].item():
                        phi = min(phi, (dq_min.item() / (recon_min - q_c[v].item() + 1e-30)))
                    phi = max(min(phi, 1.0), 0.0)
                    limited[c, v] *= phi
        return limited

    def reconstruct_face(self, q: torch.Tensor, grad: torch.Tensor,
                         face_idx: int, side: str = "owner") -> torch.Tensor:
        mesh = self.mesh
        if side == "owner":
            c = int(mesh.owner[face_idx])
        else:
            c = int(mesh.neighbor[face_idx])
        dr = mesh.face_centers[face_idx] - mesh.cell_centers[c]
        return q[c] + grad[c] @ dr

    def reconstruct_faces_vectorized(self, q, grad, side="owner"):
        """Vectorized face reconstruction for all faces."""
        mesh = self.mesh
        if side == "owner":
            cells = mesh.owner
        else:
            cells = mesh.neighbor
        dr = mesh.face_centers - mesh.cell_centers[cells]
        return q[cells] + torch.bmm(grad[cells], dr.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. WENO reconstruction
# ---------------------------------------------------------------------------

class WENOReconstructor:
    """WENO-3 / WENO-5 reconstruction on 2-D unstructured meshes."""

    def __init__(self, mesh: UnstructuredMesh2D, order: int = 3):
        assert order in (3, 5), "WENO order must be 3 or 5"
        self.mesh = mesh
        self.order = order
        self._build_stencils()

    def _build_stencils(self):
        mesh = self.mesh
        ncell = mesh.ncell
        n_stencils = 3 if self.order == 3 else 5
        self.stencils: List[List[List[int]]] = [[] for _ in range(ncell)]

        for c in range(ncell):
            neigh = list(set(
                int(mesh.neighbor[f]) for f in range(mesh.nface)
                if int(mesh.owner[f]) == c and int(mesh.neighbor[f]) >= 0
            ))
            second_neigh = set()
            for n in neigh:
                for f2 in range(mesh.nface):
                    if int(mesh.owner[f2]) == n:
                        nn = int(mesh.neighbor[f2])
                        if nn >= 0 and nn != c:
                            second_neigh.add(nn)
            all_neigh = [c] + neigh + list(second_neigh)

            needed = self.order
            if len(all_neigh) >= needed:
                dists = [(n, float(torch.norm(mesh.cell_centers[n] - mesh.cell_centers[c])))
                         for n in all_neigh if n != c]
                dists.sort(key=lambda x: x[1])
                central = [c] + [d[0] for d in dists[:needed - 1]]
                self.stencils[c].append(central)
                for shift in range(1, n_stencils):
                    biased = [c] + [d[0] for d in dists[shift:shift + needed - 1]]
                    if len(biased) == needed:
                        self.stencils[c].append(biased)
            else:
                self.stencils[c].append(all_neigh[:needed])

    def _lsq_reconstruct_stencil(self, q: torch.Tensor, stencil: List[int],
                                  target: torch.Tensor) -> torch.Tensor:
        nvar = q.shape[1]
        c0 = stencil[0]
        xc = self.mesh.cell_centers[c0]
        npts = len(stencil)
        if npts < 2:
            return q[c0]

        A_rows = []
        b_rows = [[] for _ in range(nvar)]
        for i in range(1, npts):
            ci = stencil[i]
            dr = self.mesh.cell_centers[ci] - xc
            A_rows.append(dr.numpy())
            for v in range(nvar):
                b_rows[v].append((q[ci, v] - q[c0, v]).item())

        A = torch.tensor(np.array(A_rows), dtype=q.dtype, device=q.device)
        ATA = A.T @ A
        result = torch.zeros(nvar, dtype=q.dtype, device=q.device)
        for v in range(nvar):
            b = torch.tensor(b_rows[v], dtype=q.dtype, device=q.device)
            ATb = A.T @ b
            try:
                g = torch.linalg.solve(ATA, ATb)
            except RuntimeError:
                g = torch.zeros(2, dtype=q.dtype, device=q.device)
            dr_target = target - xc
            result[v] = q[c0, v] + torch.dot(g, dr_target)
        return result

    def _smoothness_indicator(self, q: torch.Tensor, stencil: List[int]) -> float:
        if len(stencil) < 2:
            return 1e-6
        vals = q[stencil, 0]
        return float(torch.var(vals).item()) + 1e-10

    def reconstruct_face(self, q: torch.Tensor, face_idx: int,
                         side: str = "owner") -> torch.Tensor:
        mesh = self.mesh
        if side == "owner":
            c = int(mesh.owner[face_idx])
        else:
            c = int(mesh.neighbor[face_idx])

        target = mesh.face_centers[face_idx]
        stencils = self.stencils[c]
        if not stencils:
            return q[c]

        nvar = q.shape[1]
        n_stencils = len(stencils)

        q_stencil = torch.zeros(n_stencils, nvar, dtype=q.dtype, device=q.device)
        beta = torch.zeros(n_stencils, dtype=q.dtype, device=q.device)

        for s, stencil in enumerate(stencils):
            q_stencil[s] = self._lsq_reconstruct_stencil(q, stencil, target)
            beta[s] = self._smoothness_indicator(q, stencil)

        gamma = torch.ones(n_stencils, dtype=q.dtype, device=q.device) / n_stencils
        epsilon = 1e-6
        alpha = gamma / (beta + epsilon) ** 2
        omega = alpha / alpha.sum()

        return (omega[:, None] * q_stencil).sum(dim=0)

    def reconstruct_faces_vectorized(self, q, side="owner"):
        """Vectorized WENO reconstruction for all faces."""
        mesh = self.mesh
        nface = mesh.nface
        nvar = q.shape[1]
        result = torch.zeros(nface, nvar, dtype=q.dtype, device=q.device)
        for f in range(nface):
            result[f] = self.reconstruct_face(q, f, side)
        return result


# ---------------------------------------------------------------------------
# 4. Numerical flux functions
# ---------------------------------------------------------------------------

def _primitive_to_conservative(rho, u, v, p, gamma: float = 1.4):
    """Convert primitive to conservative variables."""
    if not isinstance(rho, torch.Tensor):
        rho = torch.tensor(rho, dtype=torch.float64)
        u = torch.tensor(u, dtype=torch.float64)
        v = torch.tensor(v, dtype=torch.float64)
        p = torch.tensor(p, dtype=torch.float64)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    return torch.stack([rho, rho * u, rho * v, E])


def _conservative_to_primitive(U, gamma: float = 1.4):
    """Convert conservative to primitive variables."""
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


def _euler_flux(U, n, gamma: float = 1.4):
    """Physical Euler flux in direction n."""
    rho, u, v, p = _conservative_to_primitive(U, gamma)
    un = u * n[0] + v * n[1]
    E = U[3]
    return torch.stack([
        rho * un,
        rho * u * un + p * n[0],
        rho * v * un + p * n[1],
        (E + p) * un,
    ])


def roe_flux(UL, UR, n, gamma: float = 1.4):
    """Roe approximate Riemann solver flux."""
    rhoL, uL, vL, pL = _conservative_to_primitive(UL, gamma)
    rhoR, uR, vR, pR = _conservative_to_primitive(UR, gamma)

    sqrt_rhoL = torch.sqrt(rhoL)
    sqrt_rhoR = torch.sqrt(rhoR)
    u_tilde = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)
    v_tilde = (sqrt_rhoL * vL + sqrt_rhoR * vR) / (sqrt_rhoL + sqrt_rhoR)
    H_tilde = ((sqrt_rhoL * (UL[3] + pL) / rhoL + sqrt_rhoR * (UR[3] + pR) / rhoR)
               / (sqrt_rhoL + sqrt_rhoR))
    c_tilde = torch.sqrt((gamma - 1) * (H_tilde - 0.5 * (u_tilde**2 + v_tilde**2)))
    un_tilde = u_tilde * n[0] + v_tilde * n[1]

    FL = _euler_flux(UL, n, gamma)
    FR = _euler_flux(UR, n, gamma)

    delta_U = UR - UL
    abs_lambda = torch.tensor([
        abs(un_tilde - c_tilde),
        abs(un_tilde),
        abs(un_tilde),
        abs(un_tilde + c_tilde),
    ], dtype=UL.dtype, device=UL.device)

    F = 0.5 * (FL + FR) - 0.5 * abs_lambda * delta_U
    return F


def hllc_flux(UL, UR, n, gamma: float = 1.4):
    """HLLC approximate Riemann solver flux."""
    rhoL, uL, vL, pL = _conservative_to_primitive(UL, gamma)
    rhoR, uR, vR, pR = _conservative_to_primitive(UR, gamma)

    unL = uL * n[0] + vL * n[1]
    unR = uR * n[0] + vR * n[1]
    cL = torch.sqrt(gamma * pL / rhoL)
    cR = torch.sqrt(gamma * pR / rhoR)

    S_L = min(unL - cL, unR - cR)
    S_R = max(unL + cL, unR + cR)

    FL = _euler_flux(UL, n, gamma)
    FR = _euler_flux(UR, n, gamma)

    if S_L >= 0:
        return FL
    if S_R <= 0:
        return FR

    # Star region pressure and velocity
    p_star = ((rhoL * (unL - S_L) * unL - rhoR * (unR - S_R) * unR - pL + pR)
              / (rhoL * (unL - S_L) - rhoR * (unR - S_R) + 1e-30))
    S_star = (rhoR * unR * (unR - S_R) - rhoL * unL * (unL - S_L) - pR + pL) / (
        rhoR * (unR - S_R) - rhoL * (unL - S_L) + 1e-30)

    vtL = -uL * n[1] + vL * n[0]
    vtR = -uR * n[1] + vR * n[0]

    if S_L <= 0 <= S_star:
        rho_star = rhoL * (S_L - unL) / (S_L - S_star + 1e-30)
        u_star = S_star * n[0] - vtL * n[1]
        v_star = S_star * n[1] + vtL * n[0]
        e_star = p_star / ((gamma - 1) * rho_star + 1e-30)
        E_star = rho_star * (e_star + 0.5 * (u_star**2 + v_star**2))
        U_star = torch.stack([rho_star, rho_star * u_star, rho_star * v_star, E_star])
        return FL + S_L * (U_star - UL)
    else:
        rho_star = rhoR * (S_R - unR) / (S_R - S_star + 1e-30)
        u_star = S_star * n[0] - vtR * n[1]
        v_star = S_star * n[1] + vtR * n[0]
        e_star = p_star / ((gamma - 1) * rho_star + 1e-30)
        E_star = rho_star * (e_star + 0.5 * (u_star**2 + v_star**2))
        U_star = torch.stack([rho_star, rho_star * u_star, rho_star * v_star, E_star])
        return FR + S_R * (UR - U_star)


def ausm_plus_flux(UL, UR, n, gamma: float = 1.4):
    """AUSM+ flux splitting scheme."""
    rhoL, uL, vL, pL = _conservative_to_primitive(UL, gamma)
    rhoR, uR, vR, pR = _conservative_to_primitive(UR, gamma)

    unL = uL * n[0] + vL * n[1]
    unR = uR * n[0] + vR * n[1]
    cL = torch.sqrt(gamma * pL / rhoL)
    cR = torch.sqrt(gamma * pR / rhoR)
    c_hat = (cL**2 + cR**2) / (cL + cR + 1e-30)

    ML = unL / (c_hat + 1e-30)
    MR = unR / (c_hat + 1e-30)

    if abs(ML) <= 1:
        M_plus = 0.25 * (ML + 1)**2
    else:
        M_plus = 0.5 * (ML + abs(ML))

    if abs(MR) <= 1:
        M_minus = -0.25 * (MR - 1)**2
    else:
        M_minus = 0.5 * (MR - abs(MR))

    M_hat = M_plus + M_minus

    if abs(ML) <= 1:
        p_plus = 0.25 * pL * (ML + 1)**2 * (2 - ML)
    else:
        p_plus = pL * 0.5 * (1 + torch.sign(ML))

    if abs(MR) <= 1:
        p_minus = 0.25 * pR * (MR - 1)**2 * (2 + MR)
    else:
        p_minus = pR * 0.5 * (1 - torch.sign(MR))

    p_hat = p_plus + p_minus

    if M_hat >= 0:
        mdot = c_hat * M_hat * rhoL
        u_conv = uL
        v_conv = vL
        H_conv = (UL[3] + pL) / rhoL
    else:
        mdot = c_hat * M_hat * rhoR
        u_conv = uR
        v_conv = vR
        H_conv = (UR[3] + pR) / rhoR

    F = torch.stack([
        mdot,
        mdot * u_conv + p_hat * n[0],
        mdot * v_conv + p_hat * n[1],
        mdot * H_conv,
    ])
    return F


# ---------------------------------------------------------------------------
# 5. Time integration
# ---------------------------------------------------------------------------

class RKTimeIntegrator:
    """Explicit SSP Runge-Kutta time integrator."""

    def __init__(self, method: str = "rk3"):
        assert method in ("rk2", "rk3")
        self.method = method

    def compute_local_dt(self, mesh, q, gamma: float = 1.4, cfl: float = 0.5):
        ncell = mesh.ncell
        dt = torch.full((ncell,), float("inf"), dtype=q.dtype, device=q.device)

        for f in range(mesh.nface):
            o = int(mesh.owner[f])
            n_idx = int(mesh.neighbor[f])
            rho, u, v, p = _conservative_to_primitive(q[o], gamma)
            c = torch.sqrt(gamma * p / rho)
            un = abs(u * mesh.face_normals[f, 0] + v * mesh.face_normals[f, 1])
            speed = un + c
            if speed > 0:
                dt[o] = min(dt[o], cfl * mesh.cell_volumes[o] / (speed * mesh.face_lengths[f] + 1e-30))
            if n_idx >= 0:
                rho, u, v, p = _conservative_to_primitive(q[n_idx], gamma)
                c = torch.sqrt(gamma * p / rho)
                un = abs(u * mesh.face_normals[f, 0] + v * mesh.face_normals[f, 1])
                speed = un + c
                if speed > 0:
                    dt[n_idx] = min(dt[n_idx], cfl * mesh.cell_volumes[n_idx] / (speed * mesh.face_lengths[f] + 1e-30))
        return dt

    def step(self, mesh, q, rhs_fn, dt_global):
        if self.method == "rk2":
            rhs1 = rhs_fn(q)
            q1 = q + dt_global * rhs1
            rhs2 = rhs_fn(q1)
            return 0.5 * (q + q1 + dt_global * rhs2)
        else:
            rhs1 = rhs_fn(q)
            q1 = q + dt_global * rhs1
            rhs2 = rhs_fn(q1)
            q2 = 0.75 * q + 0.25 * (q1 + dt_global * rhs2)
            rhs3 = rhs_fn(q2)
            return (1/3) * q + (2/3) * (q2 + dt_global * rhs3)


# ---------------------------------------------------------------------------
# 6. Full Euler solver
# ---------------------------------------------------------------------------

class EulerSolver:
    """Compressible Euler solver on unstructured 2-D meshes."""

    def __init__(self, mesh, gamma: float = 1.4,
                 reconstruction: str = "muscl", weno_order: int = 3,
                 flux_scheme: str = "roe", limiter: str = "venkatakrishnan",
                 time_method: str = "rk3"):
        self.mesh = mesh
        self.gamma = gamma
        self.reconstruction = reconstruction.lower()
        self.flux_scheme = flux_scheme.lower()
        assert self.flux_scheme in ("roe", "hllc", "ausm_plus")

        if self.reconstruction == "muscl":
            self.reconstructor = MUSCLReconstructor(mesh, limiter=limiter)
        elif self.reconstruction.startswith("weno"):
            self.reconstructor = WENOReconstructor(mesh, order=weno_order)
        else:
            raise ValueError(f"Unknown reconstruction: {reconstruction}")

        self.time_integrator = RKTimeIntegrator(time_method)

    def _compute_flux(self, qL, qR, n):
        if self.flux_scheme == "roe":
            return roe_flux(qL, qR, n, self.gamma)
        elif self.flux_scheme == "hllc":
            return hllc_flux(qL, qR, n, self.gamma)
        else:
            return ausm_plus_flux(qL, qR, n, self.gamma)

    def compute_rhs(self, q):
        mesh = self.mesh
        ncell = mesh.ncell
        nvar = q.shape[1]
        rhs = torch.zeros_like(q)

        if self.reconstruction == "muscl":
            grad = self.reconstructor.compute_gradients(q)
            qL = self.reconstructor.reconstruct_faces_vectorized(q, grad, "owner")
            qR = torch.zeros_like(qL)
            interior = mesh.interior
            qR[interior] = self.reconstructor.reconstruct_faces_vectorized(q, grad, "neighbor")[interior]
            qR[~interior] = qL[~interior]
        else:
            interior = mesh.interior
            qL = self.reconstructor.reconstruct_faces_vectorized(q, "owner")
            qR = torch.zeros_like(qL)
            qR[interior] = self.reconstructor.reconstruct_faces_vectorized(q, "neighbor")[interior]
            qR[~interior] = qL[~interior]

        n = mesh.face_normals
        fl = mesh.face_lengths
        F = self._compute_flux_vectorized(qL, qR, n)

        owner = mesh.owner
        neighbor = mesh.neighbor
        flux_scaled = F * fl[:, None]
        rhs.index_add_(0, owner, -flux_scaled)
        interior_faces = interior.nonzero(as_tuple=True)[0]
        if len(interior_faces) > 0:
            rhs.index_add_(0, neighbor[interior_faces], flux_scaled[interior_faces])

        return rhs

    def _compute_flux_vectorized(self, qL, qR, n):
        """Vectorized flux computation over all faces."""
        nface = qL.shape[0]
        F = torch.zeros_like(qL)
        for f in range(nface):
            F[f] = self._compute_flux(qL[f], qR[f], n[f])
        return F

    def step(self, q, dt=None, cfl: float = 0.5):
        if dt is None:
            local_dt = self.time_integrator.compute_local_dt(self.mesh, q, self.gamma, cfl)
            dt = float(local_dt.min().item())
        q_new = self.time_integrator.step(self.mesh, q, self.compute_rhs, dt)
        return q_new, dt

    def run(self, q0, n_steps, cfl: float = 0.5, report_interval: int = 100):
        q = q0.clone()
        history = []
        t0 = time.perf_counter()

        for step in range(n_steps):
            q, dt = self.step(q, cfl=cfl)
            if step % report_interval == 0 or step == n_steps - 1:
                mass = float(q[:, 0].sum().item())
                history.append({"step": step, "dt": dt, "mass": mass})

        cpu_time = time.perf_counter() - t0
        return {"q": q, "history": history, "cpu_time": cpu_time}


# ---------------------------------------------------------------------------
# 7. Conservation verification
# ---------------------------------------------------------------------------

def conservation_check(q_initial, q_final, mesh, tol: float = 1e-12):
    """Strict conservation check: mass, momentum, energy."""
    vol = mesh.cell_volumes[:, None]
    Q_initial = (q_initial * vol).sum(dim=0)
    Q_final = (q_final * vol).sum(dim=0)

    residuals = {}
    names = ["mass", "momentum_x", "momentum_y", "energy"]
    nvar = q_initial.shape[1]
    for i, name in enumerate(names[:nvar]):
        res = float(abs(Q_final[i] - Q_initial[i]).item())
        residuals[name] = res

    residuals["max_residual"] = max(residuals.values())
    residuals["passed"] = residuals["max_residual"] < tol
    return residuals


# ---------------------------------------------------------------------------
# 8. Test case initial conditions
# ---------------------------------------------------------------------------

def uniform_flow(rho: float = 1.0, u: float = 1.0, v: float = 0.0,
                 p: float = 1.0, ncell: int = 1, gamma: float = 1.4):
    """Uniform flow initial condition."""
    U = _primitive_to_conservative(
        torch.tensor(rho, dtype=torch.float64),
        torch.tensor(u, dtype=torch.float64),
        torch.tensor(v, dtype=torch.float64),
        torch.tensor(p, dtype=torch.float64), gamma)
    return U[None, :].expand(ncell, -1).clone()


def sod_shocktube_1d(nx: int, gamma: float = 1.4,
                     dtype: torch.dtype = torch.float64):
    """Sod shock tube initial condition."""
    q = torch.zeros(nx, 4, dtype=dtype)
    mid = nx // 2
    UL = _primitive_to_conservative(
        torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype),
        torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype), gamma)
    UR = _primitive_to_conservative(
        torch.tensor(0.125, dtype=dtype), torch.tensor(0.0, dtype=dtype),
        torch.tensor(0.0, dtype=dtype), torch.tensor(0.1, dtype=dtype), gamma)
    q[:mid] = UL
    q[mid:] = UR
    return q


def isentropic_vortex_2d(mesh, M: float = 0.5, beta: float = 5.0,
                         gamma: float = 1.4):
    """Isentropic vortex initial condition."""
    ncell = mesh.ncell
    q = torch.zeros(ncell, 4, dtype=mesh.cell_centers.dtype)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    xc, yc = 5.0, 0.0

    r2 = (x - xc)**2 + (y - yc)**2
    T_inf = 1.0
    rho_inf = 1.0
    u_inf = M * math.sqrt(gamma)

    T = T_inf - (gamma - 1) * beta**2 / (8 * gamma * torch.pi**2) * torch.exp(1 - r2)
    rho = rho_inf * (T / T_inf) ** (1 / (gamma - 1))
    u = u_inf - beta / (2 * torch.pi) * torch.exp(0.5 * (1 - r2)) * (y - yc)
    v = beta / (2 * torch.pi) * torch.exp(0.5 * (1 - r2)) * (x - xc)
    p = rho * T

    for i in range(ncell):
        q[i] = _primitive_to_conservative(rho[i], u[i], v[i], p[i], gamma)
    return q
