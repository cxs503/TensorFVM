"""Check switched Robin and power inputs against exact integration of a two-capacity network."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm

from tensorfvm.cht.rod import solve_transient_stepped_rod


def measure(steps,method):
    area,width,k,c,h,cw = 2.,.8,1.2,1.6,2.3,.7
    wall_mass,bulk_mass = area*cw,area*width*c
    film,g = area*h,2*area*k/width
    jumps = [.0625,.1875,.3125]
    levels = np.array([[.5,.2],[1.5,-.3],[-.2,1.1],[.7,.4]])
    def forcing(t):
        return levels[np.searchsorted(jumps,t,side="right")]
    time = np.linspace(0.,.5,steps+1)
    cell,face,_,_,heat = solve_transient_stepped_rod(
        [width],area,k,c,forcing,time,.1,method=method,return_step_heat=True,
        end_types=("robin","power"),end_film_coefficient=[h,0.],
        end_areal_heat_capacity=[cw,0.],initial_end_temperature=[.8,0.],load_jump_times=jumps)
    # Integrate wall and bulk energy balances directly on each plateau,
    # augmenting the state with cumulative film and imposed heat plus a constant.
    state = np.array([.8,.1,0.,0.,1.])
    reference = [state]
    for a,b in zip(time,time[1:]):
        ambient,power = forcing(a)
        matrix = np.zeros((5,5))
        matrix[0,[0,1,4]] = [-(g+film)/wall_mass,g/wall_mass,film*ambient/wall_mass]
        matrix[1,[0,1,4]] = [g/bulk_mass,-g/bulk_mass,power/bulk_mass]
        matrix[2,[0,4]] = [-film,film*ambient]
        matrix[3,4] = power
        state = expm((b-a)*matrix) @ state
        reference.append(state)
    ref = np.stack(reference)
    external = heat["end_input"]
    stored = wall_mass*face[:,0]+bulk_mass*cell[:,0]
    instant_power = np.array([forcing(t)[1] for t in time])
    return dict(steps=steps,method=method,
                temperature_error_K=float(np.abs(np.column_stack((face[:,0],cell[:,0]))-ref[:,:2]).max()),
                cumulative_heat_error_J=float(np.abs(external.cumsum(axis=0)-ref[1:,2:4]).max()),
                algebraic_trace_residual_W=float(np.abs(g*(face[:,-1]-cell[:,0])-instant_power).max()),
                energy_residual_J=float(np.abs(np.diff(stored)-external.sum(axis=1)).max()))


if __name__=="__main__":
    rows = []
    for method in ("theta","rannacher","sdirk2","positive_cn"):
        previous = {}
        for steps in (16,32,64,128):
            row = measure(steps,method)
            errors = {key:row[key] for key in ("temperature_error_K","cumulative_heat_error_J")}
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            row.update(orders=orders,passed=max(row["energy_residual_J"],row["algebraic_trace_residual_W"])<1e-11
                       and (steps<128 or min(orders.values())>1.8))
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
