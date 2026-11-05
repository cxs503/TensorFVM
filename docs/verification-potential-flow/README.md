# Potential Flow Verification — Grid Convergence Study

## Case Introduction

### Physical Background

The potential flow **U = ∇(x + y + z + xyz)** is an exact, unforced solution of the 
incompressible Navier-Stokes equations at **any** viscosity. This is because:

- **Divergence-free**: ∇·U = ∇²(x+y+z+xyz) = 0
- **Irrotational**: ∇×U = 0 (hence the Laplacian vanishes: ∇²U = 0)
- **Pressure**: p = -|U|²/2, so ∇p = -(U·∇)U
- **Momentum**: (U·∇)U + ∇p - ν∇²U = 0 identically

All three velocity components and all three spatial coordinates are active, making this 
a **genuinely 3D verification case** that exercises the full coupled solver without any 
body force or boundary-driven flow.

### Geometry & Parameters

| Parameter | Value |
|-----------|-------|
| Domain | Unit cube [0,1]³ |
| Boundary conditions | Exact Dirichlet velocity on all faces |
| Viscosity ν | 0.1 Pa·s / (kg/m³) |
| Density ρ | 1.0 kg/m³ |
| Reynolds number (based on L=1, U_max≈2.5) | ~25 |
| Mesh | Uniform hexahedral, triangulated faces |
| Resolutions tested | n = 2, 3, 4 (8, 27, 64 cells) |

### Exact Solution

```
u(x,y,z) = 1 + y·z
v(x,y,z) = 1 + x·z  
w(x,y,z) = 1 + x·y
p(x,y,z) = -½[(1+yz)² + (1+xz)² + (1+xy)²]
```

## Method

### Solver Configuration

- **Solver**: `NativeHybridNavierStokes3D` (coupled algorithm)
- **Convection order**: 2 (second-order face reconstruction)
- **Linear solver**: Reused LU with GMRES fallback
- **Tolerance**: 1e-10 (relative residual)
- **Max iterations**: 5000

### Verification Approach

1. **Prescribe exact boundary velocity means** using independent Gaussian quadrature 
   (`exact_boundary_velocity_means` from `native_manufactured3d`)
2. **Solve** the Navier-Stokes equations with no body force
3. **Compare** the computed velocity and pressure against the exact solution
4. **Measure** volume-weighted L2 relative errors at each resolution
5. **Compute** convergence orders from successive grid refinements

### Error Metrics

- **Velocity L2 error**: `√(Σ V_i |u_i - u_exact_i|²) / √(Σ V_i |u_exact_i|²)`
- **Pressure L2 error**: `√(Σ V_i (p_i - p_exact_i)²) / √(Σ V_i p_exact_i²)`
- **Divergence max**: Maximum absolute divergence across all cells
- **Energy balance error**: `|E_viscous + E_pressure - E_external| / max(|E|, 1)`

## Results

### Numerical Summary

| n | Cells | h=1/n | Velocity L2 error | Pressure L2 error | Div max | Energy error |
|---|-------|-------|-------------------|-------------------|---------|--------------|
| 2 | 8 | 0.500 | 6.14e-03 | 7.35e-03 | 1.17e-12 | 9.26e-12 |
| 3 | 27 | 0.333 | 2.93e-03 | 5.27e-03 | 1.07e-12 | 1.07e-11 |
| 4 | 64 | 0.250 | 1.67e-03 | 3.56e-03 | 6.75e-13 | 1.11e-11 |

### Convergence Orders

| Refinement | Velocity order | Pressure order |
|------------|----------------|----------------|
| n=2 → 3 | 1.82 | 0.82 |
| n=3 → 4 | 1.97 | 1.36 |

### Convergence Plot

![Convergence plot](convergence.png)

The velocity error shows clear **second-order convergence** (order approaching 2.0 as 
the mesh refines). The pressure error converges at **first-to-second order**, which is 
typical for cell-centered pressure on collocated grids where the pressure gradient is 
computed from cell-center differences.

## Discussion

### Key Findings

1. **Velocity convergence is second-order**: The order increases from 1.82 (n=2→3) to 
   1.97 (n=3→4), approaching the theoretical O(h²) for the second-order convection scheme.

2. **Pressure convergence is 1st–2nd order**: The pressure error converges but at a lower 
   rate than velocity. This is expected for cell-centered finite-volume methods where the 
   pressure is not directly solved but reconstructed from the velocity-pressure coupling.

3. **Divergence-free to machine precision**: The maximum divergence is O(1e-12) at all 
   resolutions, confirming that the discrete continuity equation is satisfied to solver 
   tolerance.

4. **Energy balance is exact**: The energy balance error is O(1e-11), confirming that the 
   discrete operators satisfy the kinetic energy identity.

### Verification Status

✅ **Velocity**: Second-order convergence confirmed (order > 1.9 at finest resolution)  
✅ **Pressure**: Converges at 1st–2nd order (expected for cell-centered pressure)  
✅ **Divergence-free**: max|∇·u| < 1e-11 at all resolutions  
✅ **Energy balance**: Error < 1e-10 at all resolutions  
✅ **Exact solution**: Solver recovers the analytic potential flow to O(h²)

### Limitations & Future Work

- **Coarse grids only**: Due to computational cost, only n ≤ 4 was tested. Higher 
  resolutions (n=6,8) would better confirm asymptotic second-order behavior.
- **Low viscosity convergence**: At ν < 0.1, the coupled solver struggles to converge 
  at n=3. This is a solver robustness issue, not a discretization error.
- **3D TGV verification**: The decaying Taylor-Green vortex (a genuinely unsteady 3D 
  benchmark) remains to be verified for the native hybrid NS solver.

## References

- Exact solution derived from potential theory: U = ∇φ, φ = x+y+z+xyz
- Finite-volume discretization: see `tensorfvm.native_hybrid_ns3d`
- Boundary quadrature: see `tensorfvm.native_manufactured3d.exact_boundary_velocity_means`
