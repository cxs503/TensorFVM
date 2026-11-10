# Verified case: FVM two-phase chain (drift-flux · PLIC-VOF · cavitation · Stefan)

Node **P254W1-N01** of the TensorFVM campaign: a machine-verifiable surface for the
two-phase capability, built from four independently verified segments with one
negative control each.

| segment | model | reference | control |
|---|---|---|---|
| M1 | 1-D drift-flux, compressible gas in a vertical pipe | Wallis (1969) closed-form void fraction + DOP853 pressure integration | `G_g = 0` → single-phase liquid |
| M2 | 2-D PLIC-VOF interface transport | Martin & Moyce (1952) dam-break front and column height | `g = 0` → frozen interface |
| M3 | Rayleigh-Plesset bubble + cavitation void transport | Plesset (1949) linear solution; mass-source ledger | `p_v = p_inf` → static bubble |
| M4 | one-phase Stefan phase change (enthalpy method) | Carslaw & Jaeger similarity solution | `L = 0` → pure conduction |

## Contents

| path | what it is |
|---|---|
| `provenance.sqlite` | the SQL ledger: `runs` (commit, environment, config), `metrics` (value + comparator + limit + verdict), `artifacts`, `figures` (with SHA-256) |
| `report.json` | the report, generated **by querying the ledger** (no hand-copied numbers), figures embedded as base64 |
| `REPORT.md` | the same report in Markdown |
| `figures/*.png` | the four figures (drift profile, dam-break field + front, bubble oscillation + void transport, Stefan front + temperature) |

## Reproduce

```bash
python scripts/run_two_phase_chain.py            # production settings (~2 min, 1 thread)
python scripts/run_two_phase_chain.py --quick    # smoke settings (~15 s)
```

The script rewrites `provenance.sqlite`, `report.json`, `REPORT.md` and
`figures/`, and copies the report to `docs/reports/P254W1-N01_two_phase_chain.md`.
Any number in the report can be traced back with, e.g.

```sql
SELECT segment, name, value, unit, comparator, limit_value, passed, note
FROM metrics WHERE comparator IS NOT NULL ORDER BY segment, name;
```

## Verdicts (regenerate from the ledger, do not trust this table)

J1, J3, J4, J5 and J6 pass. **J2 is an honest FAIL**: the PLIC-VOF dam break
reproduces the Martin & Moyce surge front only at late times. The measured
diagnosis (recorded in the ledger) is that the transport scheme is *not* the
limiting factor — the algebraic FCT scheme on the same grid gives the same front
error while smearing the interface more — so the deficit lives in the
first-order upwind momentum advection of the shared projection kernel. The
per-band errors (`front_rel_error_band_*`) show the deficit shrinking with time.