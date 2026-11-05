# GCI band coverage audit

cases=1920 (pure=384, truncation=1536)

| alpha | cases | implementation rate | published rate | worst E |
|---|---|---|---|---|
| 0 | 384 | 1.000 | 1.000 | 0.000 |
| 0.01 | 384 | 1.000 | 1.000 | 0.100 |
| 0.05 | 384 | 1.000 | 1.000 | 0.223 |
| 0.1 | 384 | 1.000 | 1.000 | 0.120 |
| 0.3 | 384 | 1.000 | 0.969 | 0.243 |

| p | cases | implementation rate | published rate | worst E |
|---|---|---|---|---|
| 1 | 256 | 1.000 | 1.000 | 0.225 |
| 1.5 | 256 | 1.000 | 1.000 | 0.086 |
| 2 | 256 | 1.000 | 1.000 | 0.155 |
| 2.5 | 256 | 1.000 | 1.000 | 0.203 |
| 3 | 256 | 1.000 | 1.000 | 0.227 |
| 4 | 256 | 1.000 | 0.953 | 0.243 |

Data artifact: `outputs/gci-band-coverage/audit.json` holds the aggregates above
plus the worst-efficiency cases; the full 1920+1000 per-case tables are not
stored in git (artifact cap) and are rebuilt deterministically with
`PYTHONPATH=src python scripts/audit_gci_band_coverage.py gci-band-coverage --full`.
