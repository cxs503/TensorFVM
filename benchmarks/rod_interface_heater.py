"""Harmonic heat split from an interface heater with contact resistance and shoulder storage."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.rod import solve_harmonic_stepped_rod


def measure(n, area):
    length = np.array([.4,.6])
    k,capacity = np.array([1.5,2.]),np.array([3.,4.])
    omega,power = 7.,2.+3j
    resistance,h,storage = .3,1.2,.8
    small = int(area[0]>=area[1])
    exposed = abs(area[0]-area[1])
    # Each branch is grounded at its far wall. Its continuum thermal
    # admittance at the heater-side end follows from T''=(iωC/k)T.
    wave = np.sqrt(1j*omega*capacity/k)
    y = area*k*wave/np.tanh(wave*length)
    series = np.zeros(2)
    series[small] = resistance/min(area)
    sheet = power/((y/(1+series*y)).sum()+exposed*(h+1j*omega*storage))
    traces = sheet/(1+series*y)
    wall_power = traces*area*k*wave/np.sinh(wave*length)
    load = np.zeros(2*n-1,dtype=complex)
    load[n-1] = power
    contacts = np.zeros(2*n-1)
    contacts[n-1] = resistance
    _,faces,left,right = solve_harmonic_stepped_rod(np.repeat(length/n,n),np.repeat(area,n),
            np.repeat(k,n),np.repeat(capacity,n),[0.,0.],omega,flux_order=2,
            interface_power=load,contact_resistance=contacts,film_coefficient=h,
            shoulder_areal_heat_capacity=storage)
    return dict(sheet_K=float(abs(faces[n]-sheet)),
                wall_power_W=float(max(abs(np.array([-left[0],right[-1]])-wall_power))),
                balance_W=float(abs(left[n]-right[n-1]+exposed*(h+1j*omega*storage)*faces[n]-power)))


if __name__=="__main__":
    rows = []
    for areas in ([1.,2.],[2.,1.],[1.,1.]):
        previous = {}
        for n in (4,8,16,32,64):
            errors = measure(n,np.array(areas))
            orders = {key:float(np.log2(previous[key]/errors[key])) for key in ("sheet_K","wall_power_W")} if previous else {}
            rows.append(dict(area_m2=areas,cells_per_layer=n,errors=errors,orders=orders,
                             passed=errors["balance_W"]<1e-10 and all(v>1.8 for v in orders.values())))
            previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
