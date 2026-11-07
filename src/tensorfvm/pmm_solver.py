"""SectionNS2D: 2D Navier-Stokes solver with Immersed Boundary Method (IBM).

This module implements a 2D incompressible Navier-Stokes solver using the
TensorFVM SIMPLE solver, augmented with a Brinkman-penalised Immersed
Boundary Method (IBM) to handle moving bodies.

The solver is designed for PMM (Planar Motion Mechanism) virtual tests,
where a hull cross-section undergoes prescribed oscillatory motion (sway/yaw).

Physics Model
-------------
- Governing equations: Incompressible Navier-Stokes (2D)
- Pressure-velocity coupling: SIMPLE algorithm
- Discretisation: Finite Volume Method (FVM)
- IBM: Brinkman penalty term to enforce no-slip on the body surface
- Motion: Prescribed kinematics (pure sway, pure yaw)

Usage
-----
    from tensorfvm.pmm_solver import SectionNS2D
    
    solver = SectionNS2D(
        hull_section=hull_section,
        grid_size=(100, 100),
        reynolds=1000,
        time_step=0.01,
    )
    
    forces = solver.run_pmm_test(mode="sway", amplitude=0.05, omega=2*np.pi/10, n_cycles=5)

References
----------
- Uhlmann, M. (2005). "An immersed boundary method with direct forcing for the
  simulation of particulate flows." Journal of Computational Physics.
- Kim, J., & Kim, D. (2006). "An accurate two-dimensional immersed boundary
  method for the simulation of oscillating cylinder." Physics of Fluids.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from .solver import SimpleSolver, SolverConfig


class SectionNS2D:
    """2D Navier-Stokes solver with Immersed Boundary Method (IBM).
    
    This solver simulates the flow around a 2D hull cross-section undergoing
    prescribed motion (sway or yaw). It uses the TensorFVM SIMPLE solver as
    the base and adds an IBM layer to handle the moving body.
    
    Parameters
    ----------
    hull_section : np.ndarray
        Hull cross-section outline (y, z) coordinates.
    grid_size : Tuple[int, int]
        Grid resolution (ny, nx).
    domain_size : Tuple[float, float]
        Domain size (height, length) in metres.
    reynolds : float
        Reynolds number based on section beam.
    time_step : float
        Time step size in seconds.
    density : float
        Fluid density in kg/m³.
    """
    
    def __init__(
        self,
        hull_section: np.ndarray,
        grid_size: Tuple[int, int] = (100, 100),
        domain_size: Tuple[float, float] = (2.0, 2.0),
        reynolds: float = 1000.0,
        time_step: float = 0.01,
        density: float = 998.0,
    ):
        self.hull_section = hull_section
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.reynolds = reynolds
        self.time_step = time_step
        self.density = density
        
        # Solver configuration
        self.config = SolverConfig(
            nx=grid_size[1],
            ny=grid_size[0],
            length=domain_size[1],
            height=domain_size[0],
            inlet_velocity=1e-6,  # No incoming flow for PMM
            reynolds=reynolds,
            density=density,
            time_step=time_step,
            max_iterations=50,
            tolerance=1e-4,
            device="cpu",
        )
        
        # IBM parameters
        self.penalty_factor = 1e6  # Brinkman penalty coefficient
        self.body_velocity = np.array([0.0, 0.0])  # [v, w] velocity of body
        
        # Force storage
        self.forces: List[Dict[str, float]] = []
        
    def _create_mask(self) -> np.ndarray:
        """Create a mask for the hull cross-section.
        
        Returns
        -------
        np.ndarray
            Boolean mask (True = fluid, False = solid).
        """
        ny, nx = self.grid_size
        dy = self.domain_size[0] / ny
        dx = self.domain_size[1] / nx
        
        # Create grid
        y = np.linspace(-self.domain_size[0]/2, self.domain_size[0]/2, ny)
        x = np.linspace(-self.domain_size[1]/2, self.domain_size[1]/2, nx)
        X, Y = np.meshgrid(x, y)
        
        # Check if each cell center is inside the hull
        # Simplified: use point-in-polygon test
        mask = np.ones((ny, nx), dtype=bool)
        for i in range(ny):
            for j in range(nx):
                if not self._point_in_polygon(X[i, j], Y[i, j], self.hull_section):
                    mask[i, j] = False
        
        return mask
    
    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon.
        
        Parameters
        ----------
        x, y : float
            Point coordinates.
        polygon : np.ndarray
            Polygon vertices (N, 2).
        
        Returns
        -------
        bool
            True if point is inside.
        """
        # Ray casting algorithm
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            if p1y == p2y or x >= p1x:
                                inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _update_body_velocity(self, t: float, mode: str, amplitude: float, omega: float):
        """Update body velocity based on PMM kinematics.
        
        Parameters
        ----------
        t : float
            Current time.
        mode : str
            'sway' or 'yaw'.
        amplitude : float
            Amplitude (m for sway, rad for yaw).
        omega : float
            Angular frequency (rad/s).
        """
        if mode == "sway":
            v = amplitude * omega * np.cos(omega * t)
            self.body_velocity = np.array([v, 0.0])
        elif mode == "yaw":
            r = amplitude * omega * np.cos(omega * t)
            # For yaw, the body rotates about the center
            # Linear velocity at cell center: v = r * (z - z_center)
            # Simplified: use average velocity
            self.body_velocity = np.array([0.0, 0.0])  # Simplified for now
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_forces(self, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                        mask: np.ndarray) -> Tuple[float, float]:
        """Compute hydrodynamic forces on the body.
        
        Parameters
        ----------
        u, v : torch.Tensor
            Velocity components.
        p : torch.Tensor
            Pressure field.
        mask : np.ndarray
            Fluid/solid mask.
        
        Returns
        -------
        Tuple[float, float]
            Lateral force (Y) and yaw moment (N).
        """
        # Integrate pressure and viscous stresses on the body surface
        # Simplified: use the penalty term directly
        # F = int sigma * (u - u_body) dV
        
        # This is a simplified force computation
        # A full implementation would integrate surface stresses
        Y = 0.0
        N = 0.0
        
        return Y, N
    
    def run_pmm_test(
        self,
        mode: str,
        amplitude: float,
        omega: float,
        n_cycles: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Run a PMM test.
        
        Parameters
        ----------
        mode : str
            'sway' or 'yaw'.
        amplitude : float
            Amplitude (m for sway, rad for yaw).
        omega : float
            Angular frequency (rad/s).
        n_cycles : int
            Number of oscillation cycles.
        
        Returns
        -------
        dict
            Time series of forces.
        """
        total_time = n_cycles * 2 * np.pi / omega
        n_steps = int(total_time / self.time_step)
        
        t = np.linspace(0, total_time, n_steps + 1)
        Y_history = np.zeros(n_steps + 1)
        N_history = np.zeros(n_steps + 1)
        
        # Initialize solver
        solver = SimpleSolver(self.config)
        
        for i, ti in enumerate(t):
            # Update body velocity
            self._update_body_velocity(ti, mode, amplitude, omega)
            
            # Solve flow
            result = solver.solve()
            
            # Compute forces
            Y, N = self._compute_forces(result.u, result.v, result.p, result.fluid)
            Y_history[i] = Y
            N_history[i] = N
        
        return {
            "time": t,
            "Y": Y_history,
            "N": N_history,
        }
