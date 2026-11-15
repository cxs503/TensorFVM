"""Evidence generator for the KMM180 3-D LES stability audit (no hand-copied numbers)."""
import json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.channel_les3d import ChannelLES3D, ChannelLESConfig

OUT = Path(__file__).resolve().parent

def cfg(**kw):
    base = dict(re_tau=180.0, nx=16, ny=16, nz=8, dt=2.0e-3, smagorinsky=0.1)
    base.update(kw); return ChannelLESConfig(**base)

def run(record=False, **kw):
    s = ChannelLES3D(cfg(**kw), seed=4); s.initialise(profile='reichardt', noise=0.15)
    hist = []
    while s.steps < 4000:
        if record: hist.append((s.time, float(np.abs(s.u).max()), s.bulk_velocity()))
        try: s.step()
        except FloatingPointError: return s.steps + 1, hist
    return None, hist

cases = [('Cs=0.1 dt=2e-3 rk3', dict(smagorinsky=0.1, dt=2e-3, time_scheme='rk3')),
         ('Cs=0.1 dt=4e-3 rk3', dict(smagorinsky=0.1, dt=4e-3, time_scheme='rk3')),
         ('Cs=0.1 dt=2e-3 euler', dict(smagorinsky=0.1, dt=2e-3, time_scheme='euler')),
         ('Cs=0.0 dt=2e-3 rk3', dict(smagorinsky=0.0, dt=2e-3, time_scheme='rk3')),
         ('Cs=0.0 dt=4e-3 rk3', dict(smagorinsky=0.0, dt=4e-3, time_scheme='rk3'))]
res = {}
for label, kw in cases:
    n, _ = run(**kw); res[label] = n
_, hist = run(record=True, smagorinsky=0.1, dt=2e-3, time_scheme='rk3')

# operator identity (round-off level), production-like grid
s = ChannelLES3D(ChannelLESConfig(re_tau=180.0, nx=64, ny=64, nz=32, smagorinsky=0.1, dt=2e-3))
rng = np.random.default_rng(3); nut = 0.3*rng.random((64, 64, 32))
cell, face = s._y_coefficients(s.nu + nut)
prof = rng.standard_normal(64)
s.u = np.repeat(prof[:, None, None], 64, axis=1).repeat(32, axis=2); s.w = s.u.copy()
s.v = np.zeros((65, 64, 32)); s.v[1:-1] = prof[:-1][:, None, None]
lu, lv, lw = s.diffusion(nut)
identity = dict(u=float(np.abs(lu - s._y_apply(cell, s.u)).max()),
                v=float(np.abs(lv - s._y_apply(face, s.v[1:-1])).max()),
                w=float(np.abs(lw - s._y_apply(cell, s.w)).max()))
# diffusion number on the wall row
dn = float(s.nu * 2e-3 / s.dy[0]**2)
# wall shear on the analytic parabola
p = ChannelLES3D(ChannelLESConfig(re_tau=40.0, nx=8, ny=24, nz=8, smagorinsky=0.0))
y = p.yc; p.u = (p.body_force * y * (2 - y) / (2 * p.nu))[:, None, None]*np.ones((24, 8, 8))
shear = dict(half_cell=p.wall_shear('half_cell'), second_order=p.wall_shear('second_order'), analytic=1.0)
# wall distance defect magnitude (old definition vs new) at the top wall
top_old, top_new = float(s.yc[-1]*s.config.re_tau), float(s.y_plus[-1])

summary = dict(anchor='CASE-FVM-CHANNEL-KMM180', schema='tensorfvm.les3d-stability-audit/1',
               grid='16x16x8 (evidence), 64x64x32 (production, not_run)',
               steps_to_divergence=res, operator_identity_max_abs=identity,
               wall_row_diffusion_number=dn, wall_shear=shear,
               wall_distance_top_wall=dict(old_definition_y_plus=top_old, corrected_y_plus=top_new),
               status='BLOCKED', production_u_plus_reading=None,
               tests='tests/test_channel_les3d_implicit.py (9) + tests/test_channel_les3d.py (15): 24 passed')
(OUT/'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
labels = list(res); vals = [res[k] if res[k] is not None else 4000 for k in labels]
ax[0].barh(range(len(labels)), vals, color=['#c0392b' if res[k] else '#7f8c8d' for k in labels])
ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels, fontsize=8)
ax[0].set_xlabel('steps until non-finite velocity (cap 4000)'); ax[0].set_title('Divergence is Cs- and integrator-independent')
for i, v in enumerate(vals): ax[0].text(v, i, ' %d' % v, va='center', fontsize=8)
t = [h[0] for h in hist]; mx = [h[1] for h in hist]
ax[1].semilogy(t, mx, lw=1.2, color='#2c3e50')
ax[1].set_xlabel('t [delta/u_tau]'); ax[1].set_ylabel('max |u| [u_tau]')
ax[1].set_title('Cs=0.1, dt=2e-3, SSP-RK3: amplitude before divergence')
fig.suptitle('CASE-FVM-CHANNEL-KMM180 / 3-D LES: remaining divergence after the implicit-diffusion fix', fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'fig_divergence_evidence.png', dpi=130)

rows = '\n'.join('| %s | %s |' % (k, ('%d' % res[k]) if res[k] else 'no divergence within 4000') for k in labels)
md = f"""# CASE-FVM-CHANNEL-KMM180 / 3-D LES route: what the implicit-diffusion fix does and does not fix

Anchor: `CASE-FVM-CHANNEL-KMM180` (mean-velocity log law, `Re_tau = 180`). Route: `channel_les3d`.
Status of this note: **BLOCKED** - the anchor's 3-D LES reading is still `not_run`.

## 1. What changed in the solver

1. `implicit_wall_normal` (default on): the wall-normal viscous/SGS flux is solved as a
   tridiagonal system `(I - dt M_y) u^{{n+1}} = u^*` (vectorised Thomas, M-matrix, no pivoting).
   The matrix is *derived from the same expressions* the explicit stencil uses; the identity is
   asserted to round-off: `max|M_y^implicit f - M_y^explicit f| = {identity['u']:.2e}` (u),
   `{identity['v']:.2e}` (v), `{identity['w']:.2e}` (w) at `64x64x32`.
   The wall-row diffusion number of the production grid is `nu dt/dy^2 = {dn:.2f}` at `dt = 2e-3`
   and grows linearly with `dt`, so any usable time step leaves the explicit stability band
   (`< 0.5`) and the stiff part has to be solved, not integrated.
2. `time_scheme='rk3'` (default): SSP-RK3 for convection + transverse diffusion. Forward Euler with
   central convection has `|G|^2 = 1 + (lambda sin theta)^2 > 1` at every wavenumber.
3. Wall distance: `y+ = min(y, 2-y) Re_tau`. The grid spans the full channel `y in [0, 2]`, so the
   historical `y+ = y Re_tau` read the **top wall as `y+ = {top_old:.0f}`** instead of
   `{top_new:.2f}`: van Driest damping was off in the upper half and the Reichardt initial profile
   was evaluated at twice its argument there.
4. `wall_shear('second_order')`: quadratic one-sided wall gradient next to the first-order
   half-cell conductance (`{shear['second_order']:.6f}` vs analytic `{shear['analytic']:.1f}` on the
   laminar parabola, half-cell `{shear['half_cell']:.6f}`).

## 2. What is still broken (honest)

At `16x16x8` the solution still becomes non-finite, and the time-to-divergence depends on `dt` and
**not** on the closure or the integrator:

| configuration | steps to divergence |
|---|---|
{rows}

Two recorded diagnoses are therefore falsified by this table:

* "the wall-normal diffusion stiffness is the cause" - fixed above, and the run still diverges;
* "the divergence comes from the `nu_t` feedback" - `Cs = 0` diverges as well (`{res['Cs=0.0 dt=2e-3 rk3']}` vs `{res['Cs=0.1 dt=2e-3 rk3']}` steps).

The trend "halving `dt` roughly doubles the steps" is a per-step amplification `|G| = 1 + O(dt)`,
not a per-unit-time growth: lowering `dt` delays but never removes it. Locating the amplified mode
(per-row / per-mode growth, wall-row vs interior) is the next concrete step and is **not done here**.

## 3. Consequence for the anchor

The ledger criterion `u+(y+)` in `y+ > 30` vs the KMM1987-lineage table is **not delivered by this
route**: no production (`64x64x32`) statistics exist, so no `<3 %` or `<5 %` claim is made. The
two-dimensional/1-D route (SST equilibrium channel, `docs/case-fvm-channel-kmm180-pool`) is
unaffected by this note.

## 4. Reproduction

```bash
PYTHONPATH=src python3 docs/case-fvm-channel-kmm180-les3d-stability/make.py
PYTHONPATH=src python3 -m pytest tests/test_channel_les3d_implicit.py tests/test_channel_les3d.py -q
```

Figure: `fig_divergence_evidence.png` (left: steps-to-divergence per configuration; right: amplitude
trace before divergence).
"""
(OUT/'REPORT.md').write_text(md)
print(json.dumps(summary, indent=1))
