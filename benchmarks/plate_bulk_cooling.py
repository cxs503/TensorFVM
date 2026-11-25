"""Variable bulk heat exchange in an insulated anisotropic plate, at real and complex frequency."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensorfvm.cht.energy import solve_harmonic_conduction, solve_transient_conduction


def measure(n):
    widths = np.full(n,1/n)
    x = (np.arange(n)+.5)/n
    y = x[:,None]
    kx,ky,omega = .7,1.3,4.
    c = 1.+.4*x+.2*y
    loss = .3+1.7*x*y
    mode = np.cos(np.pi*x)*np.cos(2*np.pi*y)
    profile = .8+.3*mode
    diffusion = .3*np.pi**2*(kx+4*ky)*mode
    opts = dict(heat_loss_coefficient=loss,conductivity_y=ky)
    phase = .6+.8j
    harmonic,_,interior = solve_harmonic_conduction(widths,widths,kx,c,{},omega,
                 source=phase*((1j*omega*c+loss)*profile+diffusion),return_interior_power=True,**opts)
    faces = np.arange(1,n)/n
    qx = kx*.3*np.pi*np.sin(np.pi*faces)*np.cos(2*np.pi*y)
    qy = ky*.6*np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*faces[:,None])
    # Average the Fourier flux along each face, for comparison with power/area.
    qx *= np.sinc(1/n)
    qy *= np.sinc(.5/n)
    time = np.linspace(0.,.25,n+1)
    static = (loss-2*c)*profile+diffusion
    fields,power = solve_transient_conduction(widths,widths,kx,c,{},time,profile,
                                         source=lambda t:math.exp(-2*t)*static,return_interior_power=True,**opts)
    amplitude = np.exp(-2*time[:,None,None])
    net = amplitude*static-loss*fields
    residual = (np.diff(fields,axis=0)*c-.5*np.diff(time)[:,None,None]*(net[:-1]+net[1:])).mean(axis=(1,2))
    interval = np.diff(time)[:,None,None]
    exact_duration = amplitude[:-1]*(-np.expm1(-2*interval))/2
    section_error = 0.
    for direction,flux in (("x",qx),("y",qy)):
        integral = interval*(power[direction][:-1]+power[direction][1:])/2
        section_error = max(section_error,float(abs((integral*n-exact_duration*flux).cumsum(axis=0)).max()))
    return dict(cumulative_section_heat=section_error,transient_temperature=float(abs(fields-amplitude*profile).max()),
                harmonic_temperature=float(abs(harmonic-phase*profile).max()),
                harmonic_flux=float(max(abs(interior["x"]*n-phase*qx).max(),abs(interior["y"]*n-phase*qy).max())),
                heat_balance=float(abs(residual).max()))


if __name__=="__main__":
    rows = []
    for n in (8,16,32,64):
        errors = measure(n)
        orders = {k:math.log2(rows[-1]["errors"][k]/v) for k,v in errors.items()
                  if rows and k!="heat_balance"}
        passed = errors["heat_balance"]<1e-11 and (n<64 or min(orders.values())>1.85)
        rows.append(dict(cells_per_axis=n,errors=errors,orders=orders,passed=passed))
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
