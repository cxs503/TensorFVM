# Internal-flow momentum-ledger surface (`tensorfvm.internal_flow`)

Campaign node **P40W3B-N03** — maturity pass of the incompressible momentum budget:
three classical internal-flow cases (planar sudden expansion, backward-facing step,
differentially heated cavity) with a **control-volume momentum ledger** that itemises
every force acting on the fluid and checks that the sum vanishes.

* `REPORT.md` — generated report (tables, verdict, figures) from the case records.
* `results.json` — the machine-readable payload behind `REPORT.md`
  (criteria, tables, observed orders, figure list, record sha256 digests).
* `records/` — per-case JSON records (the hashed evidence behind every number);
  `records/figures/` — generated comparison figures and field contours.

## Module layout

| file | role |
|---|---|
| `src/tensorfvm/internal_flow/cases.py` | case builders + grid levels + three outlet treatments |
| `src/tensorfvm/internal_flow/ledger.py` | pure control-volume momentum ledger (x and y) |
| `src/tensorfvm/internal_flow/metrics.py` | wall shear / reattachment, wall Nusselt, Richardson |
| `src/tensorfvm/internal_flow/pipeline.py` | one case -> one JSON record |
| `src/tensorfvm/internal_flow/report.py` | criteria, tables, figures, markdown |
| `scripts/run_internal_flow_ledger.py` | `single` / `all` / `report` driver |
| `scripts/send_internal_flow_report.py` | publishes the platform report (base64 figures) |

## What the ledger integrates

For a cell box the fluid control volume surface is

1. **open cut faces** (box boundary inside the fluid): convective flux, pressure with the
   two-sided face pressure, normal/shear viscous stress;
2. **all wetted solid faces, including interior ones** — this is the defect that the
   first version of the ledger had: integrating only the four outer faces of a
   full-height box misses the inlet-channel wall shear of a sudden expansion and leaves a
   spurious 30–50 % residual;
3. **physical in/outflow boundaries**, which must be excluded from the wall set
   (`open_faces`) or the inlet pressure is counted twice.

Wall shear uses the solver's half-cell wall distance (`tau = 2 mu u / dy`), so the ledger
is consistent with the discrete equations of the flow solver; that discretisation is
first-order at the wall, which is why the residual itself converges at about first order
(synthetic plane-Poiseuille check: 6.25 % -> 3.1 % -> 1.6 % -> 0.8 % for
h = 1/8 -> 1/64).

## Reproduce

```bash
python scripts/run_internal_flow_ledger.py all --out docs/internal-flow-momentum-ledger/records --workers 4
python scripts/run_internal_flow_ledger.py report \
    --out docs/internal-flow-momentum-ledger/records --docs docs/internal-flow-momentum-ledger
python -m pytest tests/test_internal_flow_ledger.py -q
python scripts/send_internal_flow_report.py --mission-id 8275
```

## Evidence grades (see `reference.py`)

| anchor | grade | used as a gate? |
|---|---|---|
| Borda–Carnot `Cp = 2 s (1-s)` (exact momentum balance) | REAL | decomposition reference, not a gate |
| de Vahl Davis (1983) cavity `Nu/psi/u/v` (from `inc_bench.reference`) | REAL | yes (Nu, 15 % gate / 3 % target) |
| plane-channel `C_f = 12/Re_h` | REAL | sanity only |
| Armaly (1983) reattachment `x_r/S` | **WITHHELD** | orientation band only (±10 %) |
| laminar 2-D sudden-expansion measured `Cp` | **WITHHELD** | not used; two-route Cp agreement is the gate |