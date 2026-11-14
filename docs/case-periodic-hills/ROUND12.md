# CASE-FVM-PERIODIC-HILLS — round 12: the SST transport was solving the wrong equation

Anchor: `CASE-FVM-PERIODIC-HILLS` (mission 20920, selfstart, TensorFVM).
Reference: Breuer, Peller, Rapp & Manhart (2009), *Computers & Fluids* **38**:433,
doi:10.1016/j.compfluid.2008.05.002 / ERCOFTAC UFR 3-30, `Re = u_b h / nu = 2800`.
Acceptance (feature ledger): `x_r/H` and the `y/H = 0.5` mean profile within 10 %.

## 1. Why round 12 exists

Round 11 closed with an **honest FAIL** and, crucially, with a *model-class*
diagnosis: `x_r/H(SST) = 2.05` vs 5.34, `max nu_t/nu = 342` — "SST is
over-diffusive at Re_b = 2800". Round 10 had verified only the *algebraic*
closure (does the 2-D provider produce the same `nu_t`, `F1`, `F2`, production
and destruction as `native_sst3d.sst2003_closure`: yes, pointwise identical).

What no round had ever checked is the **transport** of `k` and `omega`: how the
two scalars are actually discretised. That is where the answer was.

## 2. Defects found (all in `src/tensorfvm/periodic_hill_sst.py`)

### D-A — the diffusion coefficient carried a second, spurious `1/sigma`

`_transport_step` was called with `mu = nu + sigma_k nu_t` **and** assembled
`-(1/sigma) * diffusion_operator(mu)`. The operator actually solved was

    div( (nu + sigma nu_t)/sigma grad phi )  =  div( (nu/sigma + nu_t) grad phi )

not the documented `div((nu + sigma nu_t) grad phi)`. Consequences:

| equation | intended eddy diffusivity | solved | molecular part |
|---|---|---|---|
| `k` | `0.85 nu_t` (inner) / `1.0 nu_t` (outer) | `1.0 nu_t` | `1.18 nu` |
| `omega` | `0.50 nu_t` / `0.856 nu_t` | **`1.0 nu_t`** | **`2.0 nu`** |

For `omega` that is a **factor 2** in the eddy diffusivity: the specific
dissipation rate is smeared by twice the intended amount, the shear layer
carries a lower `omega`, hence `nu_t = a1 k / max(a1 omega, sqrt(S2) F2)` is
*inflated*, and an inflated eddy viscosity is exactly what shortens a separation
bubble. The failure signature of round 11 (`nu_t/nu = 342`, `x_r/H = 2.05`)
follows.

### D-B — the `omega` wall Dirichlet value was documented but never imposed

The module docstring (round 10) says both walls are Dirichlet
`k = 0` **and** `omega = wall_omega = 60 nu/(beta1 d^2)` (the low-Re condition
`native_sst3d.wall_omega` already uses on the 3-D native solver). The diffusion
operator is homogeneous-Dirichlet by construction, so in the 2-D hill run the
wall value was silently **zero**: `omega` had no wall data at all. On a 96x48
hill mesh `wall_omega ~ 7.1e2` versus an equilibrium near-wall
`omega ~ 2e1` — a factor ~35 that simply was not there.

Both defects push the same way (too much eddy viscosity near and above the
leeward slope), which is why one number moved both.

### D-C — internal-face pairing of the diffusion operator (fixed here, SST only)

The north face coefficient was face-averaged, the south face coefficient used
the *P-cell* value `mu[j]`, so the same face had two different conductances.
The volume-weighted operator was therefore not symmetric. The SST operator now
averages both directions (`_diffusion_operator`), verified by
`test_diffusion_operator_is_conservative`. **The momentum operator in
`periodic_hill.py` has the same pattern and was deliberately NOT touched** —
changing it would move every published momentum number of rounds 4-11 without a
re-run; it is recorded here as a known limitation of the parent solver.

Also fixed in the same edit: the blended `gamma` / `sigma_k` / `sigma_omega`
fields of the closure are now used (rounds 10-11 used the inner-layer constants
`GAMMA1`, `SIGMA_K1`, `SIGMA_W1` everywhere), and the `omega` production is
written as `gamma * (P_k / nu_t)` instead of `gamma P_k / max(nu_t, 1e-12)` so
it stays finite as `nu_t -> 0`.

## 3. Verification (this round's deliverable)

`tests/test_periodic_hill_sst_transport_coefficients.py` — 5 tests, each pinning
a *discretisation identity* rather than a hill number:

| test | what it pins | independent reference |
|---|---|---|
| `test_diffusion_operator_is_not_double_scaled` | `A_k = I/dt - D(nu + sigma_k nu_t) + diag(dest)`, unit coefficient | a 5-point conservative Laplacian assembled in the test from the mesh metric (not by calling the module under test); the defective variant `D((nu+sigma nu_t)/sigma)` is asserted to differ by >1e-6 |
| `test_diffusion_operator_is_conservative` | one coefficient per face; `diag(V) L` symmetric; interior row sums 0 | exact (1e-15) |
| `test_update_uses_blended_sigma_and_gamma` | `update()` hands `nu + sigma_k nu_t` and `nu + sigma_omega nu_t` (blended) to the operator, and `gamma*(P_k/nu_t)` to the omega equation | closure fields, 1e-12 |
| `test_omega_wall_dirichlet_is_imposed` | a constant `omega` equal to the wall value is an **exact** backward-Euler fixed point (inhomogeneous Dirichlet reproduction) | exact, 1e-12; the wall-less variant breaks it by >1e-3 |
| `test_update_imposes_wall_omega_at_both_walls` | `60 nu/(beta1 d^2)` at both walls, `d` = first-cell wall distance; `k` stays homogeneous | closed form, 1e-12 |

Suite status: `tests/test_periodic_hill_sst_transport_coefficients.py` 5 passed;
with `test_periodic_hill_round10.py` 10 passed; with
`test_periodic_hill_round9.py` + `test_periodic_hill_round11.py` 32 passed
(the pre-existing hill gates are unaffected — none of them asserts the buggy
coefficient).

## 4. Physics verdict — WITHHELD at commit time

The re-scored time-averaged run (96x48, `dt = 0.02`, 20000 steps, burn-in 8000,
`samples = 600`, identical protocol to round 11 so the comparison is
apples-to-apples) was dispatched to the cluster as job **cj3059**
(`scripts/round11_sst_long_average.py --closure sst --tag r12-96x48`, artefact
`round11_sst_r12-96x48.json`, scored by `scripts/round12_analysis.py` →
`round12_comparison.json`, `fig_round12_profile_y05.png`,
`fig_round12_series.png`). At the time of committing this round the job was
still **running**, so this document reports **no** `x_r/H` for the fixed closure.

Honest statement of what is and is not established:

* **established**: rounds 10-11 measured a solver that did not solve the
  documented SST equations (`nu_t` diffusivity, no `omega` wall data, inner-only
  blending constants). Round 11's verdict is therefore **not** a model-class
  verdict, and the "SST over-diffusive at Re_b = 2800" claim in `ROUND11.md` §6
  is **withdrawn** as unsupported until the fixed closure is scored.
* **not established**: that the fixed SST reproduces `x_r/H = 5.34 +- 10 %`.
  That is a measurement, not an argument, and it is pending.
* the acceptance anchor `CASE-FVM-PERIODIC-HILLS` **stays OPEN**.

## 5. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_periodic_hill_sst_transport_coefficients.py -q
# short transient (local): 48x24
PYTHONPATH=src python scripts/round11_sst_long_average.py --closure sst \
    --tag r12-48x24 --nx 48 --ny 24 --steps 3000 --burn-in 1500 --sample-every 20
# acceptance protocol (cluster): 96x48
PYTHONPATH=src python scripts/round11_sst_long_average.py --closure sst \
    --tag r12-96x48 --nx 96 --ny 48 --steps 20000 --burn-in 8000 --sample-every 20
PYTHONPATH=src python scripts/round12_analysis.py
```

## 6. Note on the runtime environment (read before trusting any `python -m pytest`)

`/usr/lib/python3.11/site-packages/__editable__.tensorfvm-0.1.0.pth` points at
**another agent's worktree** (`/root/tp-agent/work/m20057/src`). A bare
`python -m pytest` therefore imports `tensorfvm` from that tree, and a defect fix
made here is invisible to the test run (observed: the new tests failed with
`AttributeError` while the class was patched in this worktree). Always run
`PYTHONPATH=<this worktree>/src python -m pytest ...`, and prefer
`run_compute --stage_dir` (which stages this worktree) over `ref=main` for
physics artefacts.
