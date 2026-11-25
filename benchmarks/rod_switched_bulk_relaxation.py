"""Temporal refinement of cell and face interval means during switched bulk cooling."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.rod import solve_transient_stepped_rod


def measure(n, method):
    # The switch always bisects an output interval. Both ends are insulated.
    times = np.r_[0., (np.arange(n)+.5)/n, 1.]
    switch, initial, rate = .5, .3, 3.
    first, second = 1.1, -.4
    at_switch = first+(initial-first)*math.exp(-rate*switch)
    t1, t2 = np.minimum(times,switch), np.maximum(times-switch,0.)
    state = np.where(times<=switch,first+(initial-first)*np.exp(-rate*t1),
                     second+(at_switch-second)*np.exp(-rate*t2))
    integral = (first*t1-(initial-first)*np.expm1(-rate*t1)/rate
                +second*t2-(at_switch-second)*np.expm1(-rate*t2)/rate)
    dt = np.diff(times)
    exact_mean = np.diff(integral)/dt
    dx,area,c = np.array([.2,.35,.45]),np.array([1.,.7,1.3]),np.array([1.2,2.4,.8])
    u,_,_,_,heat,mean = solve_transient_stepped_rod(dx,area,[.9,1.5,1.],c,[0.,0.],times,initial,
        end_types=("power","power"),heat_loss_coefficient=c*rate,
        bulk_ambient_temperature=lambda t:first if t<switch else second,
        load_jump_times=[switch],return_interval_temperatures=True,return_step_heat=True,method=method)
    exact_heat = (dx*area*c)*np.diff(state)[:,None]
    return dict(state_K=float(abs(u-state[:,None]).max()),
                mean_face_K=float(abs(mean["face"]-exact_mean[:,None]).max()),
                mean_cell_K=float(abs(mean["cell"]-exact_mean[:,None]).max()),
                cumulative_bulk_heat_J=float(abs((heat["bulk_exchange"]-exact_heat).cumsum(0)).max())), float(
                    abs(np.diff(u,axis=0)*dx*area*c-heat["bulk_exchange"]-heat["left"]+heat["right"]).max())


if __name__ == "__main__":
    rows = []
    for method in ("rannacher","sdirk2_start","sdirk2","theta"):
        previous = None
        for n in (16,32,64,128,256):
            errors,balance = measure(n,method)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            # BE startup gives a first-order mean on the damped interval.
            passed = balance<1e-12 and all(v>(.85 if method=="rannacher" and k.startswith("mean_") else 1.8)
                                           for k,v in orders.items())
            rows.append(dict(method=method,intervals=n+1,errors=errors,orders=orders,
                             cell_balance_J=balance,passed=passed))
            previous = errors
    version = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=version,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
