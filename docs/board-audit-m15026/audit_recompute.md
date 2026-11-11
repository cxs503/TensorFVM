# Independent recomputation (audit m15190)

## Zalesak (from committed NPZ fields)

| scheme | N | geo err reported | geo err audited | audited vs independent ref | exact-field maxdiff | rel delta | mass drift rep/aud |
|---|---|---|---|---|---|---|---|
| cicsam | 128 | 0.04379 | 0.04379 | 0.04379 | 0 | 0.00% | 4.82e-16/4.82e-16 |
| cicsam | 64 | 0.11047 | 0.11047 | 0.11047 | 0 | 0.00% | 5.42e-14/5.42e-14 |
| plic | 128 | 0.03955 | 0.03955 | 0.03955 | 0 | 0.00% | 6.42e-16/6.42e-16 |
| plic | 64 | 0.09302 | 0.09302 | 0.09302 | 0 | 0.00% | 8.76e-15/8.76e-15 |

## Dam break (from committed leg series vs Martin & Moyce)

| grid | front err reported | front err audited (max) | mean | height err reported/audited | volume drift | control |
|---|---|---|---|---|---|---|
| [120, 40] | 0.9882 | 83.9578 | 2.5683 | 0.8450/9.5140 | 4.1e-16 | 0 |
| [240, 80] | 0.8477 | 5.5676 | 0.8248 | 0.8265/7.4280 | 1.3e-16 | 0 |

Gates reproduced from artefacts: **8/10** (reported 8/10; `--summary-only` rerun identical)
