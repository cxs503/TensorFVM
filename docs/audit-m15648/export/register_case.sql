
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'CASE-FORWARD-STEP',
       'Mach 3 wind tunnel with a forward-facing step (Woodward & Colella 1984 problem 2; 3x1 tunnel, step h=0.2 at x=0.6, gamma=1.4, rho=1.4, p=1, t=4)',
       'Woodward & Colella, J. Comput. Phys. 54 (1984) 115-173 problem 2 (t=4 density contours only); graded values = closed-form Rankine-Hugoniot / isentropic state for M1=3 recomputed in forward_step_benchmark._reference',
       'six closed-form criteria (p2/p1, rho2/rho1, u2/u1, M2, p_stag/p1, rho_stag/rho1) + mass drift + energy drift; graded grid of record 480x160',
       'each closed-form criterion <8% (pre-registered in docs/repro-fvm-forward-step/DESIGN.md); mass drift <1e-6; energy drift <1e-2',
       'TensorFVM', 'pending'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP');
