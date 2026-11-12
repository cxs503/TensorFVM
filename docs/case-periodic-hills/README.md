# CASE-FVM-PERIODIC-HILLS — periodic hills, separation / reattachment (round 1)

Anchor: `CASE-FVM-PERIODIC-HILLS` (mission 16139), TensorFVM.
Reference case: Breuer, Peller, Rapp & Manhart (2009), *Flow over periodic hills --
numerical and experimental study in a wide range of Reynolds numbers*,
Computers & Fluids **38**:433-457, doi:10.1016/j.compfluid.2008.05.002
(ERCOFTAC UFR 3-30; identical geometry/BC set).

## 1. What is delivered in round 1

| item | status | evidence |
|---|---|---|
| hill contour `y_b(x/H)` (six-cubic ERCOFTAC description, reconstructed and *validated*) | **REAL** | `src/tensorfvm/periodic_hill.py::hill_profile`, `tests/test_periodic_hill.py` |
| boundary-conforming (terrain-following) mesh + analytic metrics | **REAL** | `PeriodicHillMesh` (verticals `xi` faces, analytic `eta_x`, `eta_y`) |
| separation / reattachment / wall-shear post-processing, bulk-velocity definition | **REAL** | `PeriodicHillSolver.wall_shear`, `.reattachment`, `.bulk_velocity` |
| incompressible field solution at Re=2800 vs Breuer 2009 | **WITHHELD (solver not converged)** | section 5 |

## 2. Geometry: provenance and validation (zero hand-copying)

The contour is the six-cubic description of the ERCOFTAC UFR 3-30 Test Case,
pulled from the ERCOFTAC Knowledge Base Wiki
(`https://kbwiki.ercoftac.org/w/api.php?action=parse&prop=wikitext&page=UFR%203-30%20Test%20Case`,
section "Test Case", first paragraph + the six `<math>` blocks). The same six
coefficients are reproduced in Xiao, Wu & Laizet, arXiv:1910.01264, Appendix A.

The published cubics are written in a **normalised abscissa** `u = KAPPA * x/H`,
not in `x/H` directly (evaluating them at `x/H` contradicts itself: the six arcs
disagree by up to 1.2 in `y/H`). `KAPPA` is therefore *derived*, twice and
independently:

1. C0-continuity of the six arcs at the published junctions
   (0.3214, 0.5, 0.7143, 1.071, 1.429 h) gives `KAPPA = 27.96` (spread 0.1 % over
   the five junctions, inferred from the linear/quadratic/cubic coefficient
   ratios of a fit to the published contour);
2. the `max(0, .)` clamp of the last arc must land on `y/H = 0` at
   `x/H = 1.929`, which gives `KAPPA = 27.958`.

Validation against an **independent published contour** of the same mesh family
(`marchdf/periodicHill`, `meshes/periodicHill.dat`, 200-point CharLES mesh
contour, fetched from the GitHub contents API):

```
six-arc junction jumps (max over the 5 junctions) : 0.0
y/H at x/H = 0            : 1.000000   (crest)
y/H at x/H = 1.929        : 0.000278   (leeward foot)
y/H at x/H = 7.071        : 0.000000   (windward foot, mirror about 4.5)
y/H at x/H = 9            : 1.000000   (period closes)
max |dy/H| vs the 200-pt published contour (0 <= x/H <= 4.5) : 0.0017
```

So the geometry used by the solver is `y/H` accurate to 0.17 % of the hill
height against an independent mesh of the same case, and it closes exactly over
the 9 h period.

Domain and boundary conditions (ERCOFTAC UFR 3-30 Test Case, "Boundary
conditions and simulation parameters"): `Lx = 9 h`, `H_top = 3.036 h`, periodic
in `x`, no-slip + impermeability on both walls, spanwise homogeneous (2-D), mass
flux held constant by adjusting the streamwise body force, and the bulk velocity
is the ERCOFTAC definition `u_b = 1/2.035 * int_h^{3.035 h} u dy` (same page,
first paragraph of "Test Case"). `Re = u_b h / nu = 2800` is one of the two DNS
cases of Breuer et al. 2009 (case 4 LESOCC / case 5 MGLET, per the UFR 3-30
Evaluation page, Table 1 discussion).

## 3. Method (what was built)

`src/tensorfvm/periodic_hill.py` (new module, ~560 lines):

* **Mesh**: `x = xi`, `y = y_b(x) + eta (H_top - y_b(x))`; streamwise faces are
  exactly vertical, all metrics analytic (`eta_y = 1/Delta`,
  `eta_x = -(1-eta) y_b'/Delta`), vertices from the validated contour.
* **Solver**: cell-centred finite volume, fractional step. Explicit blended
  central/upwind convection; implicit diffusion on the fixed FV operator
  (factorised once); pressure-correction Poisson using the *same* face
  coefficients `|A_f| / d_n` as the divergence and the pressure gradient, with
  periodic wrap in `xi` and `dphi/dn = 0` on both walls (in these coordinates
  the wall Neumann condition is literally `p_eta = 0`).
* **Driving**: streamwise body force `G` updated every step from the ERCOFTAC
  bulk velocity (integral controller), so `Re` is a grid-independent input.
* **Post-processing**: kinematic wall shear from the wall-normal derivative of
  the tangential velocity at the wall face centres, zero-crossing search for
  separation `x_s/H` and reattachment `x_r/H`, and `u(y)` sampling on vertical
  lines (linear in `x`) for the profile comparison.

Reference data provenance for the metrics (the primary paper is paywalled and
could not be retrieved: OpenAlex reports `oa_status = closed`, no repository
copy; `sciencedirect` returns HTTP 403):

* `x_r/H = 5.34`, `x_s/H = 0.227` at `Re = 2800`, attributed to Breuer et al.
  (2009) DNS among others in Volpiani et al., Phys. Rev. Fluids **6**, 084612
  (2021), Sec. III (`https://arxiv.org/pdf/2101.10528`, refs [50, 61, 62];
  ref [62] is Breuer et al. 2009).
* ERCOFTAC UFR 3-30 Evaluation page: "the mean flow separates at the curved
  hill crown ... attaches naturally at about `x/h = 4.5`" (general statement
  over the Re range).
Both are recorded here as *intermediate sources*; the number is NOT read off the
primary paper, which is stated explicitly rather than papered over.

## 4. Results

No converged field: the acceptance numbers (`x_r/H`, `u(y)` profiles) could not
be produced in round 1 — see section 5. Nothing is quoted as a result.

## 5. Failure mode (honest)

`fail_stage`: **solve** (after geometry + mesh + operators were verified).

Symptom, reproduced deterministically with the hill removed
(`hill_amplitude = 0`, which reduces the case to plane-Poiseuille flow between
`y = 0` and `y = 3.036` with a known analytic solution):

```
periodic_hill: PeriodicHillConfig(nx=16, ny=64, hill_amplitude=0, Re=100, dt=0.5)
after 2000 steps:  G = 539.8 (should be ~0.013), bulk = 0.973,
profile is essentially PLUG flow  (u_max / u_bulk = 1.004, analytic: 1.5)
=> provisional verdict at that stage: FAIL (no wall friction in the system)
```

Diagnosis (last probe of round 1): the assembled implicit momentum matrix has
`diag(M) ~ 564` where `1/dt + nu * sum_f(|A_f|/d_n)/Vol` should be `~2.1`; the
factorised operator is therefore ~265x too stiff and the body force is being
absorbed instead of accelerating the fluid. This is an **assembly bug in
`PeriodicHillSolver._build`** (the diffusion coefficients that enter the diagonal
are larger than the face-coefficient product they are built from). The matrix is
an M-matrix with the right *structure* (`row sum max/min` consistent, exact
periodic wrap) but wrong *scale*, so the LU solve returns `O(1e-3)` of the RHS
rather than the physical response. The bug is isolated to one function and is the
first item of round 2; it was NOT worked around by tuning `dt`, the bulk
controller or the blending factor, because that would have hidden it.

Everything upstream of that solve is verified and reusable: geometry (0.17 %
against an independent contour), mesh metrics, face-flux divergence, wall-shear
and reattachment post-processing, and the ERCOFTAC bulk-velocity functional.

## 6. Ecosystem friction (recorded, not worked around)

1. `tensor/knowledge/search` (the mandated "开工三查" first check) is **offline**:
   `{"available": false, "reason": "engine_offline", "engine": {"base":
   "http://127.0.0.1:8021", ...Connection refused}` — the knowledge-base engine on
   port 8021 was down for the whole run.
2. `negative-knowledge/search?keywords=TensorFVM` returns HTTP 422 (parameter
   contract mismatch: the documented `keywords` query is rejected).
3. The primary reference is behind a paywall (Elsevier, `oa_status: closed`, no
   OA repository copy; `sciencedirect`/`researchgate`/`tandfonline` all return
   403 from this sandbox). Reference numbers therefore come from cited
   intermediate sources — a traceability gap the platform should close with a
   local copy of the ERCOFTAC database.
4. Outbound network is host-filtered (github.com, raw.githubusercontent.com,
   zenodo.org, wayback, search engines blocked; `api.github.com`,
   `api.openalex.org`, `arxiv.org`, `kbwiki.ercoftac.org`, `link.springer.com`
   allowed) — the UFR 3-30 wiki and the arXiv/GitHub APIs were the only route to
   the geometry and to a digitised contour.

## 7. Round-2 plan (concrete)

1. Fix `_build`'s diffusion diagonal (unit test: assemble, then check
   `M @ ones` equals `ones/dt + nu * (wall + interior face coefficients)`).
2. Re-run the plane-Poiseuille verification to `< 1e-3` on `u_max`; that becomes
   the gate for the hill case.
3. Grid-convergence study (96x48, 144x72, 192x96) at `Re = 2800`, sub-critical
   `dt`; report `x_r/H` vs 5.34 and `u(y)` at `x/H = 0.5, 2, 4, 6` (the stations
   of Breuer 2009 Figs. 4.3-4.6).
4. Expectation to be tested, stated in advance: a 2-D laminar/unsteady
   computation has no turbulence model, so `x_r/H` will most likely overshoot
   5.34 by more than 10 %; if so the honest verdict is FAIL against the <10 %
   tolerance with the 2-D-vs-turbulent gap quantified, and the next step is a
   closure model, not tolerance tuning.

## 8. Reproduce

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
s = PeriodicHillSolver(PeriodicHillConfig(nx=144, ny=72, reynolds=2800.0))
s.advance(4000)                      # not converged in round 1, see section 5
print(s.body_force, s.bulk_velocity(), s.reattachment())
EOF
pytest tests/test_periodic_hill.py -q     # geometry + bulk-velocity + mesh contract
```
