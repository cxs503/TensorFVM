#!/usr/bin/env python
"""Generate report JSON for API submission."""
import base64
import json
import time
from pathlib import Path

OUT = Path("artifacts/heated-cavity-benchmark")

# Load summary
summary = json.loads((OUT / "summary.json").read_text())

# Encode figures
figures = []
for name in ["nu_comparison", "nu_convergence", "nu_profile_Ra1e5", "velocity_comparison"]:
    png_path = OUT / f"{name}.png"
    if png_path.exists():
        b64 = base64.b64encode(png_path.read_bytes()).decode()
        captions = {
            "nu_comparison": "Mean Nusselt number comparison: de Vahl Davis (1983) reference vs TensorFVM (n=128) for Ra=10³-10⁶",
            "nu_convergence": "Mesh convergence: relative error in Nu vs resolution for all Rayleigh numbers",
            "nu_profile_Ra1e5": "Local Nusselt number profile along the hot wall at Ra=10⁵, n=128",
            "velocity_comparison": "Velocity extrema (|u|_max, |v|_max) comparison: reference vs TensorFVM",
        }
        figures.append({"name": name, "caption": captions.get(name, ""), "png_base64": b64})

# Build markdown report
markdown = """# de Vahl Davis (1983) Heated Cavity Benchmark — TensorFVM

## 1. Problem Description

Natural convection in a differentially heated square cavity is a canonical benchmark for buoyancy-driven flow solvers. The geometry is a unit square [0,1]² with:
- **Left wall**: Hot, T = +ΔT/2 (dimensionless θ = +0.5)
- **Right wall**: Cold, T = -ΔT/2 (dimensionless θ = -0.5)
- **Top/Bottom walls**: Adiabatic, no-slip
- **Fluid**: Air, Pr = 0.71
- **Rayleigh numbers**: Ra ∈ {10³, 10⁴, 10⁵, 10⁶}

**Reference**: de Vahl Davis, G. (1983), "Natural convection of air in a square cavity: a bench mark numerical solution", *Int. J. Numer. Methods Fluids* 3, 249-264.

## 2. Method & Implementation

**Solver**: TensorFVM `StaggeredMAC` — staggered (MAC) finite-volume discretisation with:
- Van Leer limited conservative fluxes for convection
- DCT-II diagonalisation for pressure Poisson (pure Neumann)
- Explicit forward-Euler time marching to steady state
- Boussinesq buoyancy coupling in momentum equation
- Second-order wall-normal temperature gradient for Nu computation

**Non-dimensionalisation**: Free-fall scaling with H=1, ΔT, U=√(gβΔTH), giving:
- ν = √(Pr/Ra), α = 1/√(Pr·Ra)
- Ra = gβΔTH³/(να), Pr = ν/α

**Mesh**: Uniform n×n cells, n ∈ {32, 64, 128}, with grid sequencing (coarse solution interpolates to fine).

## 3. Parameters

| Parameter | Value |
|---|---|
| Prandtl number (Pr) | 0.71 |
| Rayleigh numbers | 10³, 10⁴, 10⁵, 10⁶ |
| Mesh resolutions | 32², 64², 128² |
| CFL | 0.5 |
| Convergence tolerance | 2×10⁻⁶ |
| Wall gradient order | Second-order (quadratic fit) |

## 4. Results — Reference vs Computed

### Mean Nusselt Number (primary metric)

| Ra | Nu_ref (de Vahl Davis) | Nu_got (TensorFVM, n=128) | Error | <5%? |
|---|---|---|---|---|
| 10³ | 1.118 | 1.1179 | 0.01% | ✓ |
| 10⁴ | 2.243 | 2.2463 | 0.15% | ✓ |
| 10⁵ | 4.519 | 4.5320 | 0.29% | ✓ |
| 10⁶ | 8.800 | 8.9009 | 1.15% | ✓ |

### Velocity Extrema (secondary metric, normalised by α/H)

| Ra | |u|_max ref | |u|_max got | u_err% | |v|_max ref | |v|_max got | v_err% |
|---|---|---|---|---|---|---|
| 10³ | 3.649 | 3.647 | 0.06% | 3.697 | 3.694 | 0.07% |
| 10⁴ | 16.178 | 16.194 | 0.10% | 19.617 | 19.637 | 0.10% |
| 10⁵ | 34.73 | 34.976 | 0.71% | 68.59 | 68.658 | 0.10% |
| 10⁶ | 64.63 | 65.337 | 1.09% | 219.36 | 220.707 | 0.62% |

**All cases pass the <5% tolerance criterion.**

## 5. Convergence & Sensitivity

### Mesh Convergence
The Nu error decreases monotonically with mesh refinement:
- Ra=10⁴: 1.28% (n=32) → 0.36% (n=64) → 0.15% (n=128)
- Ra=10⁵: 4.99% (n=32) → 1.07% (n=64) → 0.29% (n=128)
- Ra=10⁶: 21.2% (n=32) → 4.51% (n=64) → 1.15% (n=128)

Higher Ra requires finer mesh due to thinner boundary layers.

### Convergence Behavior
All cases converged via the stationary Nu criterion (|ΔNu/Nu| < 0.1% for 2 consecutive checks). Grid sequencing from coarse to fine reduced wall-clock time by ~40% compared to cold starts.

## 6. Notes & Limitations

- **Bug fix**: Corrected velocity reference values for Ra=10⁴ in `reference.py` (was 4.535/6.776, corrected to 16.178/19.617 per de Vahl Davis Table 3). Also corrected Nu for Ra=10³ (was 1.334, corrected to 1.118).
- **Explicit time stepping**: The explicit scheme requires small CFL for stability at high Ra; implicit would be more efficient.
- **2D only**: The benchmark is inherently 2D; 3D effects (turbulence at higher Ra) are not captured.
- **Steady state**: The benchmark targets steady solutions; time-dependent behaviour at Ra>10⁶ is not addressed.

## Reproduction
```bash
bash scripts/repro_heated_cavity.sh
```
"""

report = {
    "title": "de Vahl Davis (1983) Heated Cavity Benchmark — TensorFVM",
    "summary": "Natural convection square cavity benchmark at Ra=10³-10⁶, Pr=0.71. All four cases pass <5% tolerance: Nu errors 0.01%-1.15%. Fixed reference data bug (Ra=10⁴ velocity values).",
    "scenario_key": "CASE-HEATED-CAVITY-FVM",
    "mission_id": 15410,
    "tags": ["benchmark", "natural-convection", "de-vahl-davis", "TensorFVM", "heated-cavity"],
    "markdown": markdown,
    "figures": figures,
}

Path("/tmp/report.json").write_text(json.dumps(report, indent=2))
print(f"Report written to /tmp/report.json ({len(json.dumps(report))} bytes)")
print(f"Figures: {len(figures)}")
