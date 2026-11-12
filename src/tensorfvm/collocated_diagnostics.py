"""Independent diagnostics from native collocated FV fields (no smoothing)."""
import numpy as np


def flux_coupling(owner, neighbor, area, volumes, flux, face_velocity, speed):
    owner, neighbor = np.asarray(owner), np.asarray(neighbor)
    area, volumes, flux = map(np.asarray, (area, volumes, flux))
    interior = neighbor >= 0
    interpolated = (np.asarray(face_velocity) * area).sum(axis=1)
    def divergence(q):
        net = np.bincount(owner, weights=q, minlength=len(volumes))
        net -= np.bincount(neighbor[interior], weights=q[interior], minlength=len(volumes))
        return float(np.max(np.abs(net / volumes)))
    defect = np.abs(flux-interpolated)[interior]/np.linalg.norm(area[interior],axis=1)/speed
    return dict(native_face_divergence_max=divergence(flux),
                interpolated_cell_velocity_divergence_max=divergence(interpolated),
                interior_max_normal_velocity_difference=float(defect.max()),
                interior_p95_normal_velocity_difference=float(np.quantile(defect,.95)),
                physical_coupling_qualified=False)


def angular_pressure(pressure, nx, ny, nz):
    """Span-mean native innermost ring; high-frequency content is diagnostic.

    A high-frequency fraction is not a general turbulence acceptance limit.
    Here it detects the startup mesh-locked modes, including k near Nyquist.
    """
    wall=np.asarray(pressure).reshape(nz,ny,nx)[:,0].mean(axis=0)
    fluctuation=wall-wall.mean()
    power=np.abs(np.fft.rfft(fluctuation))**2
    total=power[1:].sum()
    curvature=np.roll(wall,1)-2*wall+np.roll(wall,-1)
    return dict(wall_pressure_min=float(wall.min()),wall_pressure_max=float(wall.max()),
                high_angular_frequency_energy_fraction=float(power[(nx+3)//4:].sum()/total)if total else 0.,
                angular_second_difference_relative_l2=float(np.linalg.norm(curvature)/np.linalg.norm(fluctuation))if total else 0.,
                pressure_coupling_qualified=False)


def _smooth_axis(field, axis, kernel):
    """Apply a 1-D stencil ``kernel`` (len 3, centred) along ``axis``.

    ``kernel`` is the *smoothing* stencil ``(1, 2, 1)/4`` (low-pass, annihilates
    the Nyquist checkerboard mode) or the *second-difference* stencil
    ``(-1, 2, -1)/4`` (high-pass, the identity on the checkerboard mode -- see
    :func:`checkerboard_index` for why the latter is the wrong detector).

    The stencil is applied on the *full* field: interior cells use the centred
    three-point stencil, and the two edge cells use the one-sided stencil
    ``(2, 1)/3`` (low-pass) / ``(2, -1)/1`` (second) so a linear field is
    reproduced exactly and no artificial boundary residual is injected.
    """
    k = np.asarray(kernel, dtype=float)
    f = np.asarray(field, dtype=float)
    out = np.zeros_like(f)
    n = f.shape[axis]
    # interior: centred stencil
    sl = [slice(None)] * f.ndim
    for off, w in ((-1, k[0]), (0, k[1]), (1, k[2])):
        s = list(sl); t = list(sl)
        s[axis] = slice(1, -1) if off == 0 else (slice(0, -2) if off == -1 else slice(2, None))
        t[axis] = slice(1, -1)
        out[tuple(t)] += w * f[tuple(s)]
    # left edge (index 0): one-sided (2,1)/3 or (2,-1)/1
    if k[0] > 0:                       # low-pass
        w0, w1 = 2.0 / 3.0, 1.0 / 3.0
    else:                              # second-difference
        w0, w1 = 2.0, -1.0
    sl0 = list(sl); sl0[axis] = 0
    sl1 = list(sl); sl1[axis] = 1
    out[tuple(sl0)] = w0 * f[tuple(sl0)] + w1 * f[tuple(sl1)]
    # right edge (index n-1): one-sided (1,2)/3 or (-1,2)/1
    slm = list(sl); slm[axis] = n - 1
    sl2 = list(sl); sl2[axis] = n - 2
    out[tuple(slm)] = w1 * f[tuple(sl2)] + w0 * f[tuple(slm)]
    return out


def checkerboard_index(pressure, nx=None, ny=None, nz=None, stencil="lowpass"):
    """Checkerboard (odd-even) pressure-decoupling index ``CI = ||p - S(p)||/||p||``.

    ``S`` is a centred 1-D stencil applied along every grid axis.  The *correct*
    detector is the low-pass ``(1, 2, 1)/4``: on the Nyquist checkerboard mode
    ``p = (-1)^(i+j)`` it returns ``S(p) = 0`` so ``CI = 1`` (fully decoupled),
    and on a smooth (affine / analytic) field it returns ``CI ~ 0``.  The
    *second-difference* stencil ``(-1, 2, -1)/4`` is the **identity** on the
    checkerboard mode (``S(p) = p``, ``CI = 0``) -- it is the wrong detector and
    is kept here only to make that failure machine-checkable (see
    ``tests/test_checkerboard_index.py``).

    ``pressure`` may be a flat cell array (then ``nx/ny/nz`` give the reshape) or
    already a ``(nz, ny, nx)`` array.  ``stencil`` in ``{"lowpass", "second"}``.
    """
    p = np.asarray(pressure, dtype=float)
    if p.ndim == 1:
        if nx is None or ny is None or nz is None:
            raise ValueError("flat pressure needs nx/ny/nz")
        p = p.reshape(nz, ny, nx)
    kernel = np.array((1.0, 2.0, 1.0)) / 4.0 if stencil == "lowpass" else np.array((-1.0, 2.0, -1.0)) / 4.0
    sp = p.copy()
    for axis in range(p.ndim):
        sp = _smooth_axis(sp, axis, kernel)
    denom = float(np.linalg.norm(p))
    if denom == 0.0:
        return {"checkerboard_index": 0.0, "stencil": stencil, "denominator": 0.0}
    return {"checkerboard_index": float(np.linalg.norm(p - sp) / denom),
            "stencil": stencil, "denominator": denom}
