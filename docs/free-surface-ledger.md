# Free-surface VOF boundary-flux witness: verification debt closure

Module under test: `src/tensorfvm/free_surface_bc.py`
Reproduce: `python3 scripts/verify_free_surface_ledger.py` (artifact `runs/free_surface_ledger/ledger.json`)
Tests: `tests/test_free_surface_ledger.py` (12 cases)

## 1. What was owed

`free_surface_bc` documented `outflow_vof_flux` as the quantity that lets
"the global volume ledger be closed in the anchor validation". The only test
was `test_outflow_vof_flux_zero_velocity` — a zero-velocity case, where every
formula returns `0.0`. So the module's headline claim was **unverified**: no
test exercised a non-trivial velocity field, and no reference value existed.

The shipped implementation was also analytically wrong on any non-trivial
field:

```python
east_out = max(0.0, float(u[:, -1].sum().item())) * dy      # column sum, then clamp
a_e      = float(alpha[:, -1].mean().item())                # column mean alpha
return dt * (east_out * a_e + ...)
```

Two independent defects:

1. **Clamp after summing.** `max(0, Σ u)` cancels a co-located inflow segment
   against an outflow segment in the same boundary column. Physically the two
   must be split *per face*; the out-flowing liquid leaves regardless of what
   enters next to it.
2. **Mean volume fraction.** The total column discharge was multiplied by the
   *arithmetic mean* of `alpha` over the boundary column, so a non-uniform
   profile is weighted wrongly (the correct weight is each face's own upwind
   `alpha`).

Measured on random fields the retired formula is off by **38 %–99 %**
(`min_retired_rel_error = 0.380`, worst `0.990` at `64×96`); on a hand-built
column with half outflow / half inflow it returns **exactly zero outflow**
while liquid is leaving.

## 2. What the fix is

The transport operator `VOFTransport.advect` reconstructs boundary face values
from a zero-gradient ghost cell, so the boundary face value is *exactly* the
adjacent interior cell value, and the flux-form update telescopes:

```
(Σ α_new − Σ α_ref)·dx·dy = − net_vof_flux
```

for **any** velocity field (divergence-free or not) and for any face
reconstruction, because the same face value enters both the last interior face
and the boundary face.

`free_surface_bc` now exposes that witness directly:

| symbol | meaning |
|---|---|
| `outflow_vof_flux(...)` | gross liquid volume leaving, in/out split **per face**, each face weighted by **its own** upwind `alpha` |
| `net_vof_flux(...)` (new) | signed net leaving = outflow − inflow; the ledger quantity |
| `FreeSurfaceBC.vof_net_flux(...)` (new) | same, with the BC's `dx, dy` |
| `FreeSurfaceBC.ledger_residual(...)` (new) | `ΔV + net`, the unaccounted liquid volume of one step |

Sign convention (the part that is easy to get wrong, and the bug found while
writing this): a boundary face's *leaving* flux is `+u·dy·α` on east/north but
`−u·dy·α` on west/south. All four faces are declared in the leaving sense
before splitting, so `gross = Σ max(leaving, 0)` and `net = Σ leaving` share one
definition.

Shape contract is now enforced (`u` must be `(ny, nx+1)`, `v` must be
`(ny+1, nx)`), because silent broadcasting of a transposed `v` would produce a
plausible-looking but wrong flux.

## 3. Evidence

`scripts/verify_free_surface_ledger.py` runs four checks on 8 random seeds plus
one refined `64×96` case.

| check | criterion | measured | verdict |
|---|---|---|---|
| per-face witness vs independent scalar face loop (gross) | rel ≤ 1e-12 | 2.3e-15 | pass |
| per-face witness vs independent scalar face loop (net) | rel ≤ 1e-12 | 2.2e-15 | pass |
| ledger closure vs `VOFTransport(scheme="upwind").advect` | residual / total volume ≤ 1e-13 | 2.8e-16 | pass |
| analytic uniform-stream translation `dV = dt·U·dy·(Σα_in − Σα_out)` | rel ≤ 1e-12 | 2.3e-13 | pass |
| negative control: retired column-sum formula rejected | rel > 1e-2 | 0.380 (min), 0.990 (worst) | pass |

Refinement case `64×96` (`dx = dy = 0.004`, `dt = 1e-4`): gross witness
`4.2e-16`, ledger scaled residual `1.4e-16`, retired formula `0.989`. The
identity does not degrade with grid size.

### Per-seed table

| seed | gross rel err | net rel err | ledger residual (scaled) | retired rel err | translation rel err |
|---|---|---|---|---|---|
| 1 | 1.6e-16 | 1.4e-15 | 5.4e-17 | 0.890 | 4.1e-14 |
| 2 | 2.2e-16 | 1.8e-15 | 4.4e-17 | 0.835 | 2.3e-13 |
| 3 | 1.6e-16 | 1.7e-15 | 1.1e-16 | 0.916 | 1.2e-14 |
| 4 | 1.4e-16 | 2.2e-16 | 1.7e-17 | 0.960 | 3.1e-15 |
| 5 | 0.0 | 1.5e-16 | 3.1e-17 | 0.957 | 3.8e-14 |
| 6 | 0.0 | 4.0e-16 | 1.2e-16 | 0.961 | 6.2e-14 |
| 7 | 0.0 | 1.6e-15 | 2.8e-16 | 0.931 | 1.4e-13 |
| 8 | 0.0 | 1.2e-16 | 1.6e-16 | 0.904 | 1.3e-14 |

`0.0` gross error on some seeds means `Σ max(leaving, 0)` happened to vanish
that column direction; the net column carries the information there.

## 4. Why the ledger residual is scaled, not relative

`ΔV = (Σα_new − Σα_ref)·dx·dy` is a *difference* of two `O(1)` sums, so
`|residual / ΔV|` is dominated by the cancellation floor of `Σα` (`≈ ε·N`) and
reads `~1e-11` even when the algebra is exact. The correct, well-conditioned
criterion divides by the total liquid volume `Σα·dx·dy`, which is where the
`1e-13` gate comes from. Reporting the residual relative to `ΔV` would have
falsely condemned a correct implementation.

## 5. Caveats and scope

- The ledger identity is exact **provided the `alpha ∈ [0,1]` clip in
  `advect` is inactive**. The verification therefore drives a discretely
  divergence-free MAC velocity from a corner streamfunction (cell divergence
  `1.3e-17`), which makes donor-cell advection bounded. On a divergent field the
  clip — not the boundary witness — controls the volume balance; that is a
  property of the operator, documented in `VOFTransport`, and out of scope here.
- `net_vof_flux` reproduces the **donor-cell** boundary face value, which is the
  value used by `scheme="upwind"`. Under `scheme="blend"` the boundary face
  value carries a Courant-weighted acceptor correction when `alpha` is strictly
  interior, so the ledger closure is certified for `upwind` only. `upwind` is
  the recommended scheme whenever the VOF Courant number exceeds `1/3`, which
  includes the two-phase solver default (0.5).
- This is a boundary-flux / bookkeeping module. It does not touch the
  pressure–velocity coupling, the CSF surface-tension source, or the two-phase
  transient projection (covered by `tests/test_two_phase_solver_verification.py`).
- `FreeSurfaceBC.apply_velocity_bc` (zero-gradient top/side, no-slip bottom) is
  unchanged and still only shape-tested; imposing it is the caller's job.

## 6. Error judgement against the acceptance standard

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| gross outflow vs independent per-face loop | exact discrete algebra | 2.3e-15 rel | yes |
| net outflow vs independent per-face loop | exact discrete algebra | 2.2e-15 rel | yes |
| one-step liquid-volume ledger vs `advect` | exact telescoping identity | 2.8e-16 of total volume | yes |
| uniform-stream `ΔV` vs closed form | analytic | 2.3e-13 rel | yes |
| retired formula vs same reference | exact discrete algebra | 38 %–99 % | **no — retired, replaced** |

The retired formula is retained only as a negative control in the test suite; it
must never re-enter the ledger path.