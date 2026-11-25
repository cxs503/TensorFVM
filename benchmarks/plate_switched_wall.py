"""A storing film, two unequal cells and independently switched wall and volume loads."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm

from tensorfvm.cht.energy import solve_transient_conduction


def measure(steps,theta):
    dx = np.array([.35,.65])
    c = np.array([1.2,2.1])
    k = np.array([.7,2.3])
    film,cw,contact = 2.4,.6,.17
    jumps = (.073,.134,.193)
    def ambient(t):
        return .6 if t<jumps[0] else -.8
    def east(t):
        return .3 if t<jumps[1] else 1.7
    def source(t):
        return np.array([.7,-.2]) if t<jumps[2] else np.array([-1.4,2.])
    bc = {"west":dict(type="robin",values=ambient,film_coefficient=film,wall_areal_heat_capacity=cw,
                      initial_wall_temperature=.4),"east":dict(type="neumann",values=east)}
    times = np.linspace(0.,.35,steps+1)
    start = [.8,-.6]
    field,walls,inner = solve_transient_conduction(dx,[1.],k,c,bc,times,start,source=source,theta=theta,
                contact_resistance_x=contact,load_jump_times=jumps,return_wall_power=True,return_interior_power=True)
    # The three storage nodes are connected by two series thermal resistances.
    g = 2*k[0]/dx[0]
    link = 1/(contact+sum(dx/(2*k)))
    mass = np.r_[cw,c*dx]
    matrix = np.zeros((4,4))
    matrix[:3,:3] = np.array([[-film-g,g,0.],[g,-g-link,link],[0.,link,-link]])/mass[:,None]
    state = np.array([.4,*start,1.])
    ref = [state[:3]]
    segments = np.union1d(times,jumps)
    for a,b in zip(segments,segments[1:]):
        power = dx*source(a)
        matrix[:3,-1] = np.r_[film*ambient(a),power[0],power[1]+east(a)]/mass
        state = expm((b-a)*matrix) @ state
        if b==times[len(ref)]:
            ref.append(state[:3])
    ref = np.array(ref)
    numerical_wall = field[:,0,0]+walls["west"][:,0]/g
    exact_contact = link*(ref[:,1]-ref[:,2])
    return dict(temperature=float(abs(field[:,0]-ref[:,1:]).max()),
                wall_temperature=float(abs(numerical_wall-ref[:,0]).max()),
                contact_power=float(abs(inner["x"][:,0,0]-exact_contact).max()))


if __name__=="__main__":
    rows = []
    for theta in (.5,1.):
        previous = {}
        for steps in (16,32,64,128):
            errors = measure(steps,theta)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            passed = steps<128 or min(orders.values())>(1.85 if theta==.5 else .9)
            rows.append(dict(steps=steps,theta=theta,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
