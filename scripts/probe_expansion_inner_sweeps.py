"""Inner linear-solve accuracy vs the discrete mirror-symmetry residual.

The momentum equations are solved with a *fixed* number of red-black
Gauss-Seidel sweeps (``momentum_sweeps``), so the outer SIMPLE fixed point is
the fixed point of an inexactly solved system and can inherit the sweep
ordering bias.  If the mirror residual of the centred geometry shrinks with the
sweep count the asymmetry is an inner-solver artifact; if it saturates it is a
property of the discretisation / the steady state itself.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tensorfvm.incompressible_baselines import (build_sudden_expansion,
                                                expansion_symmetry_metrics,
                                                sudden_expansion_metrics)

out = {}
for sweeps in (4, 20, 60):
    for level, re_d in ((6, 100.0),):
        s = build_sudden_expansion(level, re_D=re_d, inlet_window=(1.5, 2.5),
                                   poisson_solver="sparse")
        s.limited = False
        s.alpha_u, s.alpha_p = 0.8, 0.5
        t0 = time.time()
        r = s.run(max_iter=60000, tol=1e-8, check=10, record_every=0,
                  max_stall=10 ** 9, momentum_sweeps=sweeps, p_cycles=2)
        sym = expansion_symmetry_metrics(s)
        m = sudden_expansion_metrics(s)
        out[f"n{level}_re{re_d:g}_sweeps{sweeps}"] = dict(
            level=level, re_D=re_d, momentum_sweeps=sweeps,
            iterations=r["iterations"], continuity=r["continuity"],
            change=r["change"], wall_seconds=round(time.time() - t0, 1),
            symmetry=sym,
            Lr_upper=m["reattachment_over_D_upper"],
            Lr_lower=m["reattachment_over_D_lower"])
        print(f"sweeps={sweeps} n={level} Re={re_d:g} it={r['iterations']} "
              f"sym={sym['symmetry_rel']:.3e} Lr={m['reattachment_over_D_upper']}/"
              f"{m['reattachment_over_D_lower']} {time.time()-t0:.0f}s", flush=True)
dest = sys.argv[1] if len(sys.argv) > 1 else "docs/verification-expansion-symmetry"
os.makedirs(dest, exist_ok=True)
with open(os.path.join(dest, "inner_sweep_study.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print("wrote", dest, flush=True)
