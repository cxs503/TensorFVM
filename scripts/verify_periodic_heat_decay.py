"""Separate spatial and temporal errors in transient conduction on a periodic strip."""
import math

import numpy as np

from tensorfvm.cht.energy import solve_transient_conduction


def measure(n, steps, theta):
    end = 1/32
    dx = 1/n
    centers = (np.arange(n)+.5)*dx
    widths = np.full(n, dx)
    kx, ky, capacity = 2., .75, 3.
    # Cell means of sin(pi*x)*sin(2*pi*y) with fixed zero x walls.
    x_mode = np.sin(np.pi*centers)*np.sinc(.5/n)
    mode = (np.sin(2*np.pi*centers)*np.sinc(1/n))[:, None]*x_mode
    bc = {s: {"type":"dirichlet", "values":0.} for s in ("west", "east")}
    time = np.linspace(0, end, steps+1)
    history, power = solve_transient_conduction(widths, widths, kx, capacity, bc, time, mode,
                      theta=theta, conductivity_y=ky, periodic_y=True, return_wall_power=True)
    # Continuous and semi-discrete exponents give distinct error references.
    exact = math.exp(-end*np.pi**2*(kx+4*ky)/capacity)
    lattice = math.exp(-end*4/(capacity*dx**2)*(kx*np.sin(np.pi*dx/2)**2+ky*np.sin(np.pi*dx)**2))
    flux = -ky*2*np.pi*x_mode*exact
    return [float(abs(history[-1]-exact*mode).max()), float(abs(power["south"][-1]/dx-flux).max()),
            float(abs(history[-1]-lattice*mode).max())]


if __name__ == "__main__":
    for study, theta in (("space", .5), ("time", .5), ("time", 1.)):
        previous = None
        for level in (8, 16, 32, 64):
            n, steps = (level, 256) if study == "space" else (16, level)
            all_errors = measure(n, steps, theta)
            errors = np.array(all_errors[:2] if study == "space" else all_errors[2:])
            orders = np.log2(previous/errors) if previous is not None else []
            print(f"{study} n={n} steps={steps} theta={theta} errors={errors} orders={orders}")
            minimum = 1.8 if theta == .5 else .9
            if not np.isfinite(errors).all() or any(order < minimum for order in orders):
                raise SystemExit("decay verification failed")
            previous = errors
