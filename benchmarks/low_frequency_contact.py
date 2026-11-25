"""Direct first-order thermal lag of a composite rod between two storing films."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.energy import solve_low_frequency_conduction


def measure(n):
    dx = 1/n
    m = n//2
    k,c = np.repeat([.8,3.],m),np.repeat([1.6,.7],m)
    resistance = .23
    contacts = np.zeros(n-1)
    contacts[m-1] = resistance
    h0,h1 = 2.,5.
    cw0,cw1 = .4,.9
    drive = .7+.35j
    bc = {side:dict(type="robin",values=drive,film_coefficient=h,wall_areal_heat_capacity=cw)
          for side,h,cw in (("west",h0,cw0),("east",h1,cw1))}
    steady,(u,walls,cuts) = solve_low_frequency_conduction(np.full(n,dx),[1.],k,c,bc,
                                contact_resistance_x=contacts[None,:],return_interior_power=True)
    # DC is uniform. At first order q'=-C*T0, T1'=-q/k and the contact
    # drop is R*q. Determine the one unknown west flux from the east film.
    def propagate(q0):
        q = q0
        value = -(q+cw0*drive)/h0
        means,flux = [],[q]
        for j in range(n):
            means.append(value-dx*q/(2*k[j])+dx**2*c[j]*drive/(6*k[j]))
            value += -dx*q/k[j]+.5*dx**2*c[j]*drive/k[j]
            q -= dx*c[j]*drive
            flux.append(q)
            if j==m-1:
                value -= resistance*q
        return q-h1*value-cw1*drive,np.array(means),np.array(flux)
    zero,one = propagate(0.)[0],propagate(1.)[0]
    _,exact,power = propagate(-zero/(one-zero))
    reported = np.r_[walls["west"],cuts["x"][0],-walls["east"]]
    return dict(steady_temperature=float(abs(steady[0]-drive).max()),
                temperature_lag_K_s=float(abs(u[0]-exact).max()),
                face_power_lag_W_s=float(abs(reported-power).max()),
                cell_balance_W_s=float(abs(np.diff(reported)+dx*c*drive).max()))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64,128):
        errors = measure(n)
        orders = {k:math.log2(rows[-1]["errors"][k]/errors[k]) for k in
                  ("temperature_lag_K_s",)} if rows else {}
        passed = (errors["steady_temperature"]<1e-11 and errors["cell_balance_W_s"]<1e-11
                  and errors["face_power_lag_W_s"]<1e-11
                  and (n<128 or orders["temperature_lag_K_s"]>1.9))
        rows.append(dict(cells=n,errors=errors,orders=orders,passed=passed))
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
