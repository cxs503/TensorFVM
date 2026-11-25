"""Interval temperatures on both sides of a heated, resistive material contact."""
import json
from pathlib import Path
import subprocess

import numpy as np
from tensorfvm.cht.rod import solve_transient_stepped_rod


def measure(n, order, method):
    split = .4
    time = np.array([0., .11, .27, .4])
    switch = .153
    rate = lambda t: .8 if t<switch else -.5
    rise = lambda t: .8*t-1.3*np.maximum(t-switch,0.)
    primitive = .4*time**2-.65*np.maximum(time-switch,0.)**2
    mean_rise = np.diff(primitive)/np.diff(time)
    faces = np.r_[np.linspace(0.,split,n+1),np.linspace(split,1.,n+1)[1:]]
    dx = np.diff(faces)
    area,k,c,curve = (np.repeat(pair,n) for pair in ([1.5,.8],[1.2,2.1],[1.7,.9],[.3,-.2]))
    power,resistance,storage,fraction = .7,.24,.38,.35
    contact_jump = power*resistance/min(area[0],area[-1])
    offset = np.repeat([1.,1.-contact_jump],n)
    slope = -power/(area*k)

    def profile(z):
        return offset+z*(slope+curve*z)

    z = (faces[:-1]+faces[1:])/2-split
    cell = profile(z)+curve*dx**2/12
    west,east = profile(faces[:-1]-split),profile(faces[1:]-split)
    contact = np.zeros(2*n-1)
    contact[n-1] = 1.
    *_,heat,means = solve_transient_stepped_rod(dx,area,k,c,
        lambda t: np.array([west[0],east[-1]])+rise(t),time,cell,
        flux_order=order,method=method,source=lambda t: c*rate(t)-2*k*curve,
        contact_resistance=contact*resistance,interface_heat_capacity=contact*storage,
        initial_trace_temperatures=(east[:-1],west[1:]),interface_capacity_left_fraction=fraction,
        interface_left_fraction=fraction,interface_power=lambda t: contact*storage*rate(t),
        load_jump_times=[switch],return_step_heat=True,return_interval_temperatures=True)
    errors = {}
    for key,expected in (("cell",cell),("trace_left",east[:-1]),("trace_right",west[1:])):
        errors[key+"_K"] = float(abs(means[key]-expected-mean_rise[:,None]).max())
    for key,base in (("trace_left",1.),("trace_right",1.-contact_jump)):
        errors["contact_"+key+"_K"] = float(abs(means[key][:,n-1]-base-mean_rise).max())
    errors["contact_heat_J"] = float(abs(heat["contact"][:,n-1]-power*np.diff(time)).max())
    return errors


if __name__ == "__main__":
    rows = []
    for method in ("theta","sdirk2"):
        for order in (1,2):
            previous = None
            for n in (4,8,16,32):
                errors = measure(n,order,method)
                rates = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if order==1 and previous else {}
                passed = all(v>1.7 for v in rates.values()) if order==1 else max(errors.values())<1e-10
                rows.append(dict(cells_per_material=n,flux_order=order,method=method,
                                 errors=errors,orders=rates,passed=passed))
                previous = errors
    version = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=version,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
