"""Periodic generation in a strip: recover complex response from a real-time cycle."""

import numpy as np

from tensorfvm.cht.energy import solve_harmonic_conduction, solve_transient_conduction


def measure(steps):
    n, omega, k, capacity = 24, 8., 2., 3.
    widths = np.full(n, 1/n)
    x = (np.arange(n)+.5)/n
    mode = np.cos(np.pi*x)*np.exp(2j*np.pi*x[:, None])
    # Insulated x and periodic y. Cell averages are separable eigenvectors.
    source = (1+.4j)*mode+.5
    harmonic, power = solve_harmonic_conduction(widths, widths, k, capacity, {}, omega,
                                               source=source, periodic_y=True)
    discrete_rate = 4*k*n*n*(np.sin(np.pi/(2*n))**2+np.sin(np.pi/n)**2)
    exact = (1+.4j)*mode/(discrete_rate+1j*omega*capacity)+.5/(1j*omega*capacity)
    np.testing.assert_allclose(harmonic, exact, atol=1e-13, rtol=0)
    time = np.arange(steps+1)*(2*np.pi/(omega*steps))
    history, wall = solve_transient_conduction(widths, widths, k, capacity, {}, time, exact.real,
                    source=lambda t: (source*np.exp(1j*omega*t)).real, periodic_y=True,
                    return_wall_power=True)
    def response(samples):
        phases = np.exp(-1j*omega*time[:-1])
        return 2*np.tensordot(phases, samples[:-1], axes=(0, 0))/steps
    field = response(history)
    flux = response(wall["south"])
    projected = np.vdot(mode, field)/np.vdot(mode, harmonic)
    return np.array([abs(field-harmonic).max(), abs(abs(projected)-1), abs(np.angle(projected)),
                     abs(flux-power["south"]).max()*n])


if __name__ == "__main__":
    previous = None
    for steps in (16, 32, 64, 128):
        errors = measure(steps)
        orders = np.log2(previous/errors) if previous is not None else []
        print(f"steps={steps} errors(T,rel_amplitude,phase_rad,seam_flux)={errors} orders={orders}")
        if not np.isfinite(errors).all() or any(order < 1.8 for order in orders):
            raise SystemExit("periodic source convergence failed")
        previous = errors
