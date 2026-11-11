# P400W2-N01 — two-phase VOF verification face (dam break + Zalesak + mass gate)

Node of the FVM two-phase campaign: interface-transport verification with
**exact** references and machine-decidable gates — the Ritter (1892) dam-break
wave front, the Zalesak (1979) slotted disk under rigid rotation, and a global
mass ledger.

**Why this directory and not `benchmarks/twophase/`.** The node brief names
`benchmarks/twophase/`; this checkout's sparse-checkout cone is
`/src /tests /scripts /docs /outputs /pyproject.toml /README.*`, so files under
`benchmarks/` cannot be staged here (git refuses them as outside the cone). The
harness therefore lives beside the node's report in `docs/`, and the runnable
code sits in the package `src/tensorfvm/twophase_bench/`.

| what | where |
|---|---|
| library (CICSAM/HRIC transport, metrics, Ritter analytics, gates) | `src/tensorfvm/twophase_bench/` |
| harness (campaign runner, report renderer) | this directory (`run_benchmark.py`, `render_report.py`) |
| report (rendered from the JSON, no hand-copied numbers) | `report.md` |
| every measured number | `artifacts/twophase_verification.json` |
| figures | `artifacts/figures/*.png` |
| pytest | `tests/test_twophase_bench_vof.py` (15 cases) |

## Capability delta

Before this node the repository had (a) flux-corrected algebraic VOF with a
MUSCL reconstruction (`vof_phase0`), (b) geometric PLIC (`vof_plic`), and (c) a
donor-cell transport whose docstring *called itself* "CICSAM-like" without any
Hyper-C / ULTIMATE-QUICKEST face value. The normalised-variable family and its
boundedness machinery were absent, the Zalesak disk had never been scored on
perimeter fidelity, and the dam-break front had never been compared with Ritter
under a *commensurable* front definition.

This node adds:

* Hyper-C + CBC-limited ULTIMATE-QUICKEST face values, CICSAM (orientation
  blend) and HRIC-type (Courant blend) schemes, plus a Zalesak-1979 FCT limiter
  that restores the discrete maximum principle (measured over-fill
  `alpha_max = 1.01002`, 93 cell-steps, without it — §3.5 of the report);
* the divergence-free 2-D field of the Ritter similarity solution
  (`v = -y du/dx`), so a *volume fraction* — not just a depth — can be advected
  in the dam-break field, plus a depth-matched front locator;
* area (three definitions) and perimeter metrics with the closed-form slotted
  disk reference, calibrated on a sub-cell disk (0.3 % bias);
* a machine-decidable gate harness (C1–C4) and a report generator whose numbers
  are injected from the artefact JSON.

## Run

```bash
# full campaign (all legs; PLIC is the expensive one)
PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py \
    --threads 8 --zalesak-res 64 128 256 --zalesak-schemes upwind cicsam hric plic \
    --dam-cells 28 56 --dam-schemes upwind cicsam plic

# pieces (each run writes one JSON + its panels into its own --out directory)
... --only-zalesak ...      # algebraic schemes, 64/128/256
... --only-zalesak --zalesak-res 64 128 --zalesak-schemes plic
... --only-dam --dam-cells 28 56 --dam-schemes upwind cicsam
... --only-dam --dam-cells 28 56 --dam-schemes plic
# then fold them together (copies panels, re-evaluates the gates)
... --merge leg-a/twophase_verification.json leg-b/twophase_verification.json \
    --out docs/P400W2-N01-two-phase-vof-verification/artifacts

# report + tests
PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/render_report.py
PYTHONPATH=src python -m pytest tests/test_twophase_bench_vof.py -q
```

The `leg-*` directories written by split runs are intermediates: `--merge`
copies their panels into `artifacts/figures/` and folds every row into the
merged JSON, so they are not committed.

## Gates

| gate | criterion | where |
|---|---|---|
| C1 | dam-break front vs Ritter, `t in [0.05, 0.30] s`, relative error ≤ 10 %, at `dx = H/28` **and** `H/56` (scheme under test: cicsam; every scheme listed with its own verdict) | `campaign.evaluate_gates` |
| C2 | Zalesak, one revolution at N = 256: enclosed-area error ≤ 1 %, perimeter error ≤ 3 % (at least one of the node's algebraic schemes; per-scheme verdicts listed) | idem |
| C3 | `\|M(t)-M(0)\|/M(0) < 0.1 %` over every leg | idem |
| C4 | both dam-break resolutions inside the C1 band, finer not worse than coarser by more than the band | idem |
| C5 | ≥ 10 pytest cases | `tests/test_twophase_bench_vof.py` |

Headline numbers (report tables have all of them): front error 1.23 % (H/28) and
0.77 % (H/56); Zalesak N=256 area 0.02 % / perimeter 3.20 % for CICSAM and
0.00 % / 1.41 % for the HRIC-type blend; worst mass drift 7.4e-13.

## Three traps this benchmark encodes (all measured, not asserted)

1. **Ritter needs a semi-infinite reservoir.** For a 0.028 m wide column the wall
   rarefaction reaches the dam at `W/c0 = 0.038 s`; the front then decelerates and
   the experimental Martin & Moyce value at 0.3 s is ≈ 0.19 m against
   `2t√(gH) = 0.445 m`. The domain therefore carries 0.26 m of upstream reservoir
   so the analytic solution holds everywhere inside it, and `H/W = 2` is kept.
2. **The front must be compared at matched depth.** `h → 0` at the tip, so an
   `alpha > 1/2` front against the zero-depth tip carries an analytic lag
   `3t√(g dy/2)` — measured at 12 % of the front position at `t = 0.05 s` and
   19 % at `t = 0.3 s`. The gate uses the depth-matched Ritter abscissa; both
   numbers are reported side by side.
3. **A depth-averaged velocity cannot advect a 2-D volume fraction.** With
   `(u(x,t), 0)` the discrete divergence is non-zero and the water column dilates
   (measured: `alpha_max` 1.0 → 0.50 within 0.1 s). The field is completed with
   `v = -y ∂u/∂x`, which keeps the free surface a material surface and makes the
   discrete divergence of every cell zero.

Two further honesty notes: the NVD face value alone does **not** enforce a
discrete maximum principle in 2-D (hence the FCT limiter), and the compressive
branch costs smooth-field accuracy — on a rigidly rotating cosine hill the
measured relative L1 is 0.497 (CICSAM) / 0.519 (HRIC) against 0.198 for the pure
UQ branch, i.e. CICSAM is an interface scheme, not a smooth-data scheme.

## Scope limits

Interface-transport face only. Momentum coupling of a viscous 2-D collapse is a
different problem (Ritter assumes a semi-infinite reservoir; the repository's
coupled measurement lags it by 30–60 %,
`docs/vof-phase0/artifacts/phase0_report.json`) and is not re-litigated here.
3-D, surface tension and phase change are out of scope (owned by other nodes).
The geometric PLIC leg does *not* meet the 10 % front criterion on this
thin-film flow (16.83 % at H/28, 8.96 % at H/56) while it is the better scheme on
the Zalesak perimeter — reported as a boundary case of the two families, with the
mechanism stated as a hypothesis, not a conclusion (report §5, note 9).