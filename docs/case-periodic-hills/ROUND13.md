# CASE-FVM-PERIODIC-HILLS — round 13: the momentum / SA-transport eta-face pairing defect

Anchor: `CASE-FVM-PERIODIC-HILLS` (status: **open**), TensorFVM, mission 20987.
Reference case: Breuer, Peller, Rapp & Manhart (2009), *Computers & Fluids*
**38**:433, doi:10.1016/j.compfluid.2008.05.002 (ERCOFTAC UFR 3-30), Re_b = 2800.
Reference values single-sourced in `src/tensorfvm/verification/periodic_hill_reference.py`
(`x_R/h = 5.41` from the KB wiki «Evaluation» page written by the case authors;
time-mean `u/U_b` at `y/H = 0.5` from the 10 DNS files distributed by M. Breuer).

This round has two parts:

* **A. close round 12's pending measurement** — the fixed SST transport (cj3059)
  was still running when round 12 committed, so its verdict was withheld. It has
  finished; the numbers are scored here.
* **B. fix the defect round 12 deliberately left open** — the momentum diffusion
  operator (and the SA transport operator, same bug) read a *different* viscosity
  on the two sides of every eta face.

---

## 1. Part A — the round-12 measurement, scored (no new physics claim)

Job **cj3059** (96×48, `dt = 0.02`, 20 000 steps, burn-in 8 000, `samples = 601`,
burst 96×48 protocol identical to rounds 9/11) finished `succeeded`. Artefacts
`round11_sst_r12-96x48.{json,npz}` were fetched from the job tarball and scored
with the round-12 script (no re-run, no re-digitisation):

| quantity | round 11 (defective transport) | round 12 (fixed transport) | reference |
|---|---|---|---|
| `x_r/H` | 2.046 | **3.144** | 5.41 (KB) / 5.34 (ledger) |
| error vs 5.41 | 62.2 % | **41.9 %** | — |
| `y/H = 0.5` stations inside 10 % | 1/7 | **0/7** | — |
| `max nu_t/nu` | 342.5 | **297.0** | — |
| stationarity (`spread_rel` of `x_r/H` over 601 samples) | ok | **2.5e-5** | — |
| gate `pass` | false | **false** | — |

Reading of the numbers:

* The transport fix moved `x_r/H` from 2.05 to 3.14 (**+54 %**), so the two
  defects round 12 found were real and material — they were *not* a model-class
  property of SST.
* The bubble is still **42 % short** of the DNS reattachment, i.e. the mean flow
  is still over-diffused: `nu_t/nu` peaks at 297 where a shear layer of thickness
  `~H` with `u' ~ 0.1 U_b` implies `O(30-60)`. The run is perfectly stationary
  (`spread_rel = 2.5e-5`), so this is not an averaging artefact.
* Therefore the anchor's acceptance (<10 % on `x_r/H` and on the `y/H = 0.5`
  profile) is still **not met**, and no claim about SST's model fidelity is made
  here: the solver still has a *measured* discretisation defect (part B), so a
  model-class verdict remains **WITHHELD**.

## 2. Part B — the defect: one eta face, two conductances

All three implicit diffusion operators of this solver are built as

```
L_h phi|_P = (1/Vol_P) * sum_f c_f mu_f (phi_N - phi_P)          (walls: Dirichlet)
```

An eta face sits on one level `L` but borders **two** cells: it is the *north*
face of row `L-1` and the *south* face of row `L`. Rounds 4–12 wrote

```python
cn_s = c_n[j]   * mu[j]                                  # south: the cell's OWN value
cn_n = c_n[j+1] * 0.5*(mu[j] + mu[j+1])                  # north: the face mean
```

so the two cells sharing a face used conductances that differ by
`0.5 * (mu[L] - mu[L-1])`. Round 12 fixed exactly this in the **SST transport**
operator (`src/tensorfvm/periodic_hill_sst.py`, `_diffusion_operator`) and wrote
in its ledger entry that the same asymmetry in the **momentum** operator was left
alone because it would move every momentum number of rounds 4–11. That is what
this round changes.

### 2.1 Reproduction before the fix (32×16 hill mesh, `nu_t/nu` up to 300)

Two independent measurements, both in `tests/test_periodic_hill_eta_face_viscosity.py`:

| measurement | value |
|---|---|
| matrix entries where legacy ≠ face-loop operator | **960** (max gap 0.944) |
| discrete energy identity `sum_P Vol_P phi_P (L phi)_P = -sum_f c_f mu_f (phi_N - phi_P)^2` | violated by **3.65 %** of the total dissipation at `max nu_t/nu = 300` (3.53 % at 100, 0 % at 0) |
| sign of the violation | follows `d mu / d eta`: under-dissipation where the shear layer thickens (`+3.5 %`), over-dissipation where it thins (`-3.2 %`) |

The identity is the stability property the implicit solve relies on: the operator
must be the divergence of a single face flux, which requires one `mu_f` per face.
Note what this is *not*: it is not "the scheme produces energy" — for the fields
tested the legacy operator still dissipates, it just dissipates the **wrong
amount**, by an amount that scales with the local viscosity jump. In the hill
problem `nu_t/nu` spans 300 across the free shear layer, i.e. the defect sits
exactly where the mean momentum balance determines the recirculation length.

### 2.2 Fix (minimal, single source of truth)

`src/tensorfvm/periodic_hill.py` gains

```python
def eta_face_viscosity(mu_cell) -> (ny+1, nx)      # level 0 and ny: wall-adjacent cell
                                                   # 0 < L < ny: 0.5*(mu[L-1] + mu[L])
```

and `_assemble_momentum` plus
`src/tensorfvm/periodic_hill_turbulence.py::SpalartAllmaras._diffusion_operator`
(Round-13'd here; the SA operator had the identical bug) now read `mu_eta[L]`
instead of `mu[j]`. Uniform `mu` degenerates to the old matrix **bit-for-bit**
(`0.5*(a+a) == a`), which is asserted, so rounds that only ever used a constant
viscosity are untouched.

### 2.3 Tests (8 new, all green; non-self-referential)

`tests/test_periodic_hill_eta_face_viscosity.py`

| test | what it binds to |
|---|---|
| `test_eta_face_values_pair_the_two_neighbouring_cells` | closed-form face values on a hand-written 3×2 array |
| `test_eta_face_rejects_invalid_viscosity` | 0/nan/1-D rejected |
| `test_momentum_operator_matches_an_independent_face_loop` | assembled matrix == a face loop written from the definition (1e-14) |
| `test_legacy_south_face_rule_is_not_the_face_loop` | negative control: the legacy rule differs by exactly `max c_n*0.5*|Δmu|/Vol` (analytic, 1e-12) and by > 1e-3 |
| `test_momentum_operator_is_energy_stable` | the dissipation identity, random field, 1e-10 |
| `test_legacy_south_face_rule_breaks_energy_stability` | the identity is violated with the analytic sign for decaying / rising `mu` |
| `test_uniform_viscosity_reproduces_the_round4_operator` | bit-for-bit backwards compatibility |
| `test_sa_transport_operator_uses_the_same_pairing` | SA operator == same face loop |

Regression: `tests/test_periodic_hill.py`, `test_periodic_hill_round2.py`,
`test_periodic_hill_round3.py` → **22 passed**; the round-12 SST coefficient
tests → **5 passed**. (Run with `PYTHONPATH=<this worktree>/src`; the editable
`.pth` in site-packages points at another agent's worktree — see ROUND12.md §6.)

## 3. Physics verdict — WITHHELD at commit time

The fixed-operator re-run is job **cj3108** (`scripts/round11_sst_long_average.py
--closure sst --tag r13-96x48`, 96×48, same protocol as cj3059, staged from this
worktree so the momentum fix is in the binary). At commit time it was **queued**
(queue position 26; 41 jobs in flight on the cluster). This document therefore
reports **no** post-fix `x_r/H`.

What *is* established: the operator was not the discrete divergence of a single
flux (measured, §2.1), and the effect is first order in the local `Δnu_t`, so it
must move the reattachment length. Whether it closes the 42 % gap is a
**measurement, not an argument** — score it with

```bash
# artifact lands in docs/case-periodic-hills/round11_sst_r13-96x48.json
PYTHONPATH=src python scripts/round12_analysis.py   # after pointing NEW at the r13 file
```

**Superseded numbers (honest scope statement).** Fixing the momentum operator
changes every published momentum result of rounds 4–11 that ran with
`nu_t ≠ 0` (rounds 4–6 laminar/channel gates used a constant viscosity and are
bit-for-bit unaffected; rounds 7–12 hill numbers are *not*). Those numbers stay
in the record as what the defective operator produced; nothing is re-labelled.

## 4. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_periodic_hill_eta_face_viscosity.py -q      # 8 passed
PYTHONPATH=src python -m pytest tests/test_periodic_hill.py tests/test_periodic_hill_round2.py \
                          tests/test_periodic_hill_round3.py -q                          # 22 passed
# defect metrics + figure
PYTHONPATH=src python docs/case-periodic-hills/../..  # see round13_operator_defect.json / fig_round13_eta_face_operator.png
```

## 5. Files

| path | role |
|---|---|
| `src/tensorfvm/periodic_hill.py` | `eta_face_viscosity` + momentum operator fix |
| `src/tensorfvm/periodic_hill_turbulence.py` | SA transport operator fix (same rule) |
| `tests/test_periodic_hill_eta_face_viscosity.py` | 8 tests (face loop, energy identity, negative control, compat) |
| `docs/case-periodic-hills/round11_sst_r12-96x48.{json,npz}` | cj3059 artefacts (fetched, scored) |
| `docs/case-periodic-hills/round12_comparison.json`, `fig_round12_*.png` | round-12 scoring, now with the real fixed-SST row |
| `docs/case-periodic-hills/round13_operator_defect.json` | defect metrics (§2.1) |
| `docs/case-periodic-hills/fig_round13_eta_face_operator.png` | violation vs `nu_t/nu`; matrix-entry histogram |

## 5b. Published report

Platform report id **19009** (mission 20987), scenario `case-fvm-periodic-hills`.

## 6. Error table (key quantity vs reference) — is it < 3 %?

| quantity | computed | reference | error | < 3 %? |
|---|---|---|---|---|
| `x_r/H` (round-12 transport fix, pre round-13 momentum fix) | 3.144 | 5.41 | **41.9 %** | **NO** |
| `y/H = 0.5` mean profile, 7 stations | — | DNS files | 0/7 inside 10 % | **NO** |
| eta-face operator vs face-loop definition | 0 | exact | **0 (1e-14)** | YES |
| dissipation identity of the fixed momentum operator | exact | closed form | 1e-10 | YES |
| `x_r/H` with the round-13 momentum fix | — | 5.41 | **WITHHELD** (job cj3108 queued) | n/a |

Honest summary: the *discretisation* claims of this round are verified to machine
precision; the *case acceptance* is still a FAIL (42 %) and the round-13 physics
number does not exist yet at commit time.