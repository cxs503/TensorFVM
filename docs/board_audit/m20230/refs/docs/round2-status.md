# Round-2 status of the Re = 1000 ladder (honest scope note)

At the moment this mission closed, the runner had **two converged resolutions** in the
bundle (`n = 32` under-resolved control, `n = 64` production row, both gated at
`max|Δu|/Δt ≤ 1e-7`) and a **third one still marching**, started by grid continuation
from the n = 64 field:

| n | state at close | change rate | steps | checkpoint |
|---|---|---|---|---|
| 32 | converged | 9.76e-08 | 30 500 | `fields/Re1000_n32.pt` |
| 64 | converged | (see summary.json) | 58 750 | `fields/Re1000_n64.pt` |
| 128 | **not converged, excluded from every table** | 5.25e-06 | 54 750 | `fields/Re1000_n128.pt` (20k-step cadence) |

The n = 128 row is excluded exactly the way `docs/verification-lid-cavity-ladder/`
excluded its own n = 128 attempt: a row that has not reached the gate is not published
as a converged number. It is *not* lost — the checkpoint machinery restores it, so a
round-3 job can resume with

```bash
PYTHONPATH=src python3 scripts/run_cavity_re1000_m15458.py \
    --out artifacts/CASE-CAVITY-RE1000-FVM --reynolds 1000 --grids 64,128 --cold-grids 32
```

(the runner resumes `fields/Re1000_n128.pt` automatically and keeps the same gate).

Why it did not finish: the container CPU is shared (load average ≈ 385 during the run,
measured `date`/`uptime` at start), so n = 128 ran at ≈ 16 ms/step; extrapolating the
observed rate decay (5.3e-6 → 1e-7 needs ≈ 6 more decades-per-10k-steps intervals)
puts convergence about 15 min beyond this mission's wall clock.
