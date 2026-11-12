# Ethier--Steinman 3-D manufactured verification -- measured status (honest)

Metric required by the case sheet: **velocity L_inf relative error < 1%** at
**Re = 1** and **Re = 100**.  Status of this delivery: **gate NOT met** (honest
FAIL), with the measurements and the diagnosis below.  Nothing is patched or
rounded up; `study.json` carries the raw records.

## 1. What is verified

The closed form used as the reference is

    u_i = -a [ e^{a x_i} sin(a (x_j + x_k)) + e^{a x_k} cos(a (x_i + x_j)) ] e^{-d^2 t}

with `(i,j,k)` cyclic.  It is **not copied**: it is re-derived and re-audited
inside the repo (`reference_audit`), and the audit numbers are in
`study.json -> reference_audit`:

| property | measured |
|---|---|
| analytic `div u` | 0.0 (exact, machine zero) |
| `lap(u) + a^2 u` | 0.0 (machine zero, exact eigenfunction) |
| `curl` of the unsteady momentum residual at `nu = d^2/a^2` | 9e-10 (FD round-off) |
| same curl at `nu = 1` (unforced relation violated) | 2.0 (Re=100 member) |
| atom-algebra gradient vs finite difference | 2e-10 |
| atom-algebra Laplacian vs finite difference | ~1e-7 |

The parameter relation `nu = d^2/a^2` is therefore **verified from the solution
itself**, not assumed.  The publisher PDF of the reference (DOI
10.1002/fld.1650190502) is closed access (Semantic Scholar `openAccess =
CLOSED`), so the original tables/pages could not be quoted; that part of the
case sheet is reported as WITHHELD rather than手抄 from secondary sources.

## 2. Measured ladder (Re = 1, a = d = 1, nu = 1, frozen t = 0.25 s)

| n | dx | velocity L_inf rel | velocity L2 rel | max div | Newton it | s |
|---|---|---|---|---|---|---|
| 4 | 0.5000 | 6.238e-02 | 5.731e-02 | 0.0 | 5 | 1.5 |
| 6 | 0.3333 | 3.884e-02 | 3.490e-02 | 0.0 | 5 | 23.4 |

Observed order of the gated metric: **1.14** (one refinement step only, so the
order is indicative, not asymptotic).  Extrapolating that slope, the 1% gate
needs `n ~ 24` (coupled direct LU on ~1.7e5 unknowns), which does not fit in the
local execution limit of this task; the run is therefore reported as FAIL at the
resolutions actually computed.

## 3. Re = 100 -- honest FAIL with diagnosis

Two parameterisations of the same relation were run:

| tag | a | d | nu | Re | cell Re `a h / nu` at n=4/n=6 | outcome |
|---|---|---|---|---|---|---|
| re100 | 1.0 | 0.1 | 0.01 | 100 | 50 / 33 | Newton `iteration_limit` |
| re100_resolved | 0.2 | 0.02 | 0.01 | 100 | 10 / 6.7 | Newton `iteration_limit` at 8 iterations; with 25 iterations it converges but the velocity error at n=4 is 137% |

Diagnosis (the ecosystem finding): the native hybrid 3-D backend discretises the
convective term with a central (order-2) hybrid upwind reconstruction and solves
the *steady* frozen problem.  Two consequences:

1. **Cell-Reynolds limit.** The scheme needs `a h / nu = O(1)`; with
   `Re = 1/nu = 100` this forces `n >= 200` per unit length (8e6 cells) for the
   paper-scale wavenumber `a = 1`.  The coupled direct LU (`splu`) cannot reach
   that size here, and the segregated SIMPLEC path does not converge either
   (measured: momentum residual stalls at 0.50 after 400 outer iterations at
   n = 4, nu = 0.01).
2. **Weak-balance ill-conditioning.** Lowering the wavenumber (`a = 0.2`) keeps
   the cell Reynolds number small, but at `nu = 0.01` the frozen steady problem
   becomes a balance of three small terms (`nu a^2`, `d^2`, `a^3`), so the
   velocity is recovered only to ~140% at `n = 4`.  The Re = 100 member is thus
   *solvable* but not *accuracy-qualified* by this backend.

Both statements come from runs recorded in `study.json` (`records[*].failure_stage`
/ `failure_reason`), not from hand-waving.

## 4. What would close the gap (carry-over for the next round)

* Replace the steady frozen formulation by the transient integrator, or
* stabilise the convection (upwind/GLS) and accept order 1->2 with a measured
  ladder, or
* keep the coupled solver but use `coupled_linear_method='block_amg'` and run
  `n = 24..32` on the cluster (the ladder slope 1.14 predicts 0.9% at n = 24).

`scripts/ethier-steinman-3d/run_study.py --n-list 24,32 --cases re1` is the
exact command to close item 3 once the iterative linear path is wired for this
case.

## 5. Reproduce

    PYTHONPATH=src python3 scripts/ethier-steinman-3d/run_study.py \
        --n-list 4,6 --cases re1,re100,re100_resolved --time 0.25 --max-iterations 8
    PYTHONPATH=src python3 -m pytest tests/test_ethier_steinman.py -q