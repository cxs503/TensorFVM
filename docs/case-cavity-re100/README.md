# CASE-CAVITY-RE100

Lid-driven square cavity at Re = 100 against Ghia, Ghia & Shin (1982), Tables I and II.

| file | what |
|---|---|
| `REPORT.md` | the six-section report (generated from the artefacts, zero hand-copy) |
| `research/oss_survey.md` | round-2 step ①: open-source survey (`~/oss-cache` + in-repo registry) |
| `research/method_survey.md` | round-2 step ②③: method genealogy and the selection that was implemented |
| `artifacts/cavity_re100_n{n}.json` | one run per resolution: pointwise ref/got/delta, field snapshot, 7 criteria |
| `artifacts/case_cavity_re100.json` | case verdict, convergence orders, per-resolution rows |
| `metrics.csv` | the report tables, machine written |
| `figures/` | profiles vs Ghia, error vs grid, psi contours + velocity field |

Reproduce::

    PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 32 64 128
    PYTHONPATH=src python scripts/render_case_cavity_re100_report.py
    PYTHONPATH=src python -m pytest tests/test_case_cavity_re100.py -q
