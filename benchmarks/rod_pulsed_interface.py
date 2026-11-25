"""Pulse heating at an area step between equal-length branches with equal diffusivity."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.rod import solve_transient_stepped_rod


LENGTH = .5
AREA = np.array([1.,2.])
K = np.array([1.,1.5])
CAPACITY = 2*K
POWER = 3.
START, STOP = .023, .111
OBSERVE = np.array([0.,.018,START,.027,.05,.09,STOP,.117,.15,.23,.40])


def step_response(age):
    """Heater temperature and cumulative outward wall heat for a step switched on at zero."""
    if age<=0:
        return 0.,np.zeros(2)
    # Each branch has the same temperature profile. In coordinates from the
    # heater to the wall its boundary conditions are Neumann and Dirichlet.
    # Expand the steady linear profile in cos((j+1/2) π x/L).
    j = np.arange(128)
    z = (j+.5)*np.pi
    alpha = K[0]/CAPACITY[0]
    rate = alpha*(z/LENGTH)**2
    decay = np.exp(-age*rate)
    conductance = AREA*K/LENGTH
    sheet = POWER/conductance.sum()*(1-np.sum(2*decay/z**2))
    # Integral of the startup wall-flux deficit over infinite time is L²/(2α).
    lag = LENGTH**2/(2*alpha)-np.sum(2*(-1.)**j*decay/(z*rate))
    heat = POWER*conductance/conductance.sum()*(age-lag)
    return sheet,heat


def measure(n):
    times = np.union1d(np.round(np.linspace(0.,OBSERVE[-1],32*n+1),12),OBSERVE)
    ids = np.searchsorted(times,OBSERVE)
    sheet_power = np.zeros(2*n-1)
    sheet_power[n-1] = POWER
    cell,faces,_,_,heat = solve_transient_stepped_rod(
        np.full(2*n,LENGTH/n),np.repeat(AREA,n),np.repeat(K,n),np.repeat(CAPACITY,n),
        [0.,0.],times,0.,flux_order=2,method="sdirk2",return_step_heat=True,
        interface_power=lambda t:sheet_power*(START<=t<STOP),load_jump_times=(START,STOP))
    reference = [tuple(a-b for a,b in zip(step_response(t-START),step_response(t-STOP))) for t in OBSERVE]
    exact_sheet,exact_wall = map(np.asarray,zip(*reference))
    outward = np.vstack((np.zeros(2),-heat["end_input"].cumsum(axis=0)))
    stored = cell @ (LENGTH/n*np.repeat(AREA*CAPACITY,n))
    supplied = POWER*np.clip(times-START,0.,STOP-START)
    jump = np.isin(OBSERVE,[START,STOP])
    error = np.abs(faces[ids,n]-exact_sheet)
    return dict(sheet_smooth_K=float(error[~jump].max()),sheet_at_switch_K=float(error[jump].max()),
                wall_heat_J=float(abs(outward[ids]-exact_wall).max()),
                balance_J=float(abs(stored+outward.sum(axis=1)-supplied).max()))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64):
        errors = measure(n)
        orders = {k:float(np.log2(rows[-1]["errors"][k]/errors[k])) for k in
                  ("sheet_smooth_K","sheet_at_switch_K","wall_heat_J")} if rows else {}
        passed = errors["balance_J"]<1e-10 and all(v>0 for v in orders.values())
        if n==64:
            passed &= orders["sheet_smooth_K"]>1.7 and orders["sheet_at_switch_K"]>.85 and orders["wall_heat_J"]>1.7
        rows.append(dict(cells_per_branch=n,errors=errors,orders=orders,passed=passed))
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,observation_times_s=OBSERVE.tolist(),runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
