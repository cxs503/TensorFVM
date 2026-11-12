-- Board audit m16336: verdict for m16261 CASE-FVM-CHANNEL-KMM180
-- Audit date: 2026-11-12
-- Level: L4 (基准验证)
-- Verdict: confirmed (消账)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorFVM',
    'm16261 CASE-FVM-CHANNEL-KMM180',
    16336,
    4,
    '{"impl_files":["src/tensorfvm/verification/kmm1987.py:353L","src/tensorfvm/benchmark_channel_kmm1987.py:562L","src/tensorfvm/rans_validation/channel.py:240L"],"entry":"scripts/run_case_channel_kmm180.py + python -m tensorfvm.benchmark_channel_kmm1987","run_artifacts":["docs/case-fvm-channel-kmm180/benchmark.json","docs/case-fvm-channel-kmm180/metrics.csv","docs/case-fvm-channel-kmm180/stations.csv","docs/case-fvm-channel-kmm180/profiles.csv","docs/case-fvm-channel-kmm180/manifest.json","docs/case-fvm-channel-kmm180/REPORT.md","docs/case-fvm-channel-kmm180/fig_velocity_profiles.png","docs/case-fvm-channel-kmm180/fig_log_region_error.png"],"tests":["tests/test_channel_kmm1987.py:12/12 passed"],"integration_callers":["tensorfvm.verification.cases.channel_kmm180","tensorfvm.verification.__main__ --channel-kmm180"],"benchmark":{"criterion":"RMS rel dev u+(y+) y+>30 <5%","tolerance":0.05,"got_log_region_rms":0.03628,"verdict":"PASS"},"repro_cmd":"PYTHONPATH=src python scripts/run_case_channel_kmm180.py --out docs/case-fvm-channel-kmm180","external_anchor":"KMM1987 JFM 177:133; LM2015 JFM 774:395 Re_tau=182.088"}',
    'board-audit'
);
