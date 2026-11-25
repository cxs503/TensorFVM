"""Cooling of an initially hot thin sheet between two cold conducting branches."""
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import brentq

from tensorfvm.cht.rod import solve_transient_stepped_rod


LENGTH, STORAGE = .5, .8
AREA = np.array([1.,1.])
K = np.array([1.,2.])
C = 2*K
OBSERVE = np.array([.005,.01,.025,.05,.1,.2,.4])


def reference(times):
    # T=sin(z*(1-x/L))/sin(z) on each branch, normalized to one at the
    # sheet. Its flux jump supplies the sheet's rate of change of energy.
    bulk = LENGTH*(C*AREA).sum()
    def residual(z):
        return z*np.sin(z)-bulk/STORAGE*np.cos(z)
    roots = np.array([brentq(residual,m*np.pi,(m+.5)*np.pi) for m in range(64)])
    integral = (.5-np.sin(2*roots)/(4*roots))/np.sin(roots)**2
    # Projection of the initially hot sheet uses the heat-capacity inner
    # product, including its point mass in addition to the two rod integrals.
    amplitude = STORAGE/(STORAGE+bulk*integral)
    rate = (K[0]/C[0])*(roots/LENGTH)**2
    modes = amplitude*np.exp(-times[:,None]*rate)
    sheet = modes.sum(axis=1)
    mean = (1-np.cos(roots))/(roots*np.sin(roots))
    removed = STORAGE-(STORAGE*sheet+bulk*(modes @ mean))
    weights = AREA*K/(AREA*K).sum()
    return sheet,removed[:,None]*weights


def measure(n):
    times = np.union1d(np.arange(32*n+1)*.4/(32*n),OBSERVE).round(12)
    times = np.unique(times)
    indices = np.searchsorted(times,OBSERVE)
    capacities = np.zeros(2*n-1)
    capacities[n-1] = STORAGE
    cell,faces,_,_,heat = solve_transient_stepped_rod(
        np.full(2*n,LENGTH/n),np.repeat(AREA,n),np.repeat(K,n),np.repeat(C,n),
        [0.,0.],times,0.,interface_heat_capacity=capacities,initial_shoulder_temperature=1.,
        method="sdirk2",flux_order=2,return_step_heat=True)
    sheet,wall = reference(OBSERVE)
    outward = np.vstack((np.zeros(2),-heat["end_input"].cumsum(axis=0)))
    stored = cell @ np.repeat(C*AREA*LENGTH/n,n)+STORAGE*faces[:,n]
    return dict(sheet_K=float(abs(faces[indices,n]-sheet).max()),
                wall_J=float(abs(outward[indices]-wall).max()),
                balance_J=float(abs(stored+outward.sum(axis=1)-STORAGE).max()))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64):
        errors = measure(n)
        orders = {k:float(np.log2(rows[-1]["errors"][k]/errors[k])) for k in ("sheet_K","wall_J")} if rows else {}
        passed = errors["balance_J"]<1e-10 and all(v>0 for v in orders.values())
        rows.append(dict(cells_per_branch=n,errors=errors,orders=orders,
                         passed=passed and (n<64 or min(orders.values())>1.8)))
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,observation_times_s=OBSERVE.tolist(),runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
