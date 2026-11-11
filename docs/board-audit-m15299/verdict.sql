-- Verdict for m15299 CASE-WOODWARD-COLELLA audit (m15394)
-- To be executed by platform when API is available

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorFVM',
    'm15299 CASE-WOODWARD-COLELLA',
    15394,
    4,
    '{"impl_files":["src/tensorfvm/compressible/cases.py","src/tensorfvm/compressible/double_blast.py","src/tensorfvm/compressible/riemann.py","tests/test_compressible_double_blast.py"],"entry":"src/tensorfvm/compressible/double_blast.py::run()","run_artifacts":"docs/woodward-colella-double-blast/artifacts/woodward_colella_double_blast.json (8.3MB)","tests":"test_compressible_double_blast.py: 4 passed in 28.78s","integration_callers":"PR#427 ddd5fbec merged to main","benchmark":"de_facto thresholds from artifact JSON; tensor_benchmark_cases: 0 rows for anchor (ledger gap, recommend registering)","repro_cmd":"PYTHONPATH=src python -m tensorfvm.compressible.double_blast --outdir docs/woodward-colella-double-blast/artifacts","external_anchor":"Woodward & Colella (1984) J. Comput. Phys. 54:115-173; arXiv:1902.07773 (Dumbser et al.); arXiv:2109.14340 (Fu et al.); Exact Riemann per Toro (2009) 3rd ed. Ch.4"}',
    'board-audit'
);
