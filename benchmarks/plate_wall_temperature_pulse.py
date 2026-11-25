"""Sudden heating and cooling of a finite slab, checked against its mixed-wall heat kernel."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.energy import solve_transient_conduction


def measure(n,damped,steps_per_column=4,monotone=False):
    dx = 1/n
    start,stop = .017,.083
    edges = np.linspace(0.,1.,n+1)
    steps = steps_per_column*n
    time = np.linspace(0.,.2,steps+1)
    def pulse(t):
        return float(start<=t<stop)
    field,heat = solve_transient_conduction(np.full(n,dx),[1.],1.,1.,
            {"west":dict(type="dirichlet",values=pulse)},time,0.,
            load_jump_times=[start,stop],rannacher=damped,monotone_theta=monotone,return_face_heat=True)
    section = np.column_stack((heat["wall"]["west"][:,0],heat["interior"]["x"][:,0],
                               -heat["wall"]["east"][:,0]))
    cumulative = section.cumsum(axis=0)
    waves = np.pi*(np.arange(96)+.5)
    def integrated_step(t):
        # Downstream stored heat equals cumulative rightward heat crossing
        # the cut. Integrate the sine-series temperature exactly in space.
        if t<=0:
            return np.zeros_like(edges)
        return 1-edges-2*(np.cos(edges[:,None]*waves) @ (np.exp(-waves**2*t)/waves**2))
    temperature = transport = 0.
    for i in (steps//4,steps//2,3*steps//4,steps):
        exact = integrated_step(time[i]-start)-integrated_step(time[i]-stop)
        temperature = max(temperature,float(abs(field[i,0]+np.diff(exact)/dx).max()))
        transport = max(transport,float(abs(cumulative[i-1]-exact).max()))
    return dict(temperature=temperature,section_heat=transport,
                bound_violation=float(max(0.,field.max()-1.,-field.min())),
                cell_balance=float(abs(dx*np.diff(field[:,0],axis=0)+np.diff(section,axis=1)).max()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-steps",action="store_true",help="compare temperature bounds with one step per column")
    coarse = parser.parse_args().large_steps
    rows = []
    methods = ("cn","rannacher","monotone_theta") if coarse else ("cn","rannacher")
    for method in methods:
        previous = {}
        for n in (16,32,64,128):
            errors = measure(n,method=="rannacher",1 if coarse else 4,method=="monotone_theta")
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("temperature","section_heat") if previous}
            passed = errors["cell_balance"]<1e-12
            if coarse and method=="monotone_theta":
                passed &= errors["bound_violation"]<1e-12 and (n<128 or min(orders.values())>.8)
            if not coarse and method=="rannacher":
                passed &= errors["bound_violation"]<1e-10 and (n<128 or min(orders.values())>1.8)
            rows.append(dict(columns=n,steps=(1 if coarse else 4)*n,method=method,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
