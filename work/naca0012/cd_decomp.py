"""Pressure/friction drag and lift decomposition from RUNDIR/wall.dat (x y nx ny Cp Cf).

usage: cd_decomp.py RUNDIR [alpha]
Face lengths come from the mesh (RUNDIR/mesh.msh, same wall-face order as run_naca.wall_data).
Cf is the shear stress along t = (ny, -nx) (n = outward unit normal of the body surface).
"""
import sys, json, math, numpy as np, torch
from types import SimpleNamespace
from tensorfvm.multi_element import ThreeElementMesh

run = sys.argv[1]
alpha = math.radians(float(sys.argv[2]) if len(sys.argv) > 2 else json.load(open(f'{run}/config.json')).get('alpha', 0))
mesh = ThreeElementMesh(SimpleNamespace(mesh_file=f'{run}/mesh.msh', device='cpu', single_element=True))
wall = mesh.masks['main']
L = mesh.face_lengths[wall].numpy()
d = np.loadtxt(f'{run}/wall.dat')
x, y, nx, ny, cp, cf = d.T
assert len(d) == len(L), (len(d), len(L))
n = np.c_[nx, ny]                              # outward unit normal of the body (as written by run_naca)
t = np.c_[ny, -nx]
# force on the body per unit dynamic pressure: pressure acts along -n, shear along +t
F_p = (-(cp[:, None] * n) * L[:, None]).sum(0)
F_f = ((cf[:, None] * t) * L[:, None]).sum(0)
ca, sa = math.cos(alpha), math.sin(alpha)
rot = lambda F: (F[0] * ca + F[1] * sa, -F[0] * sa + F[1] * ca)
Dp, Lp = rot(F_p); Df, Lf = rot(F_f)
print(f'{run}: faces {len(L)}, perimeter {L.sum():.5f}')
print(f'  Cd_p {Dp:.5f}  Cd_f {Df:.5f}  Cd {Dp + Df:.5f}')
print(f'  Cl_p {Lp:.5f}  Cl_f {Lf:.5f}  Cl {Lp + Lf:.5f}')
# contributions by region (x bins) of the pressure drag
bins = [0, .005, .02, .05, .1, .2, .4, .6, .8, .95, 1.01]
up = ny > 0
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (x >= lo) & (x < hi)
    dp_u = (-(cp * nx * L)[sel & up]).sum() * ca + (-(cp * ny * L)[sel & up]).sum() * sa
    dp_l = (-(cp * nx * L)[sel & ~up]).sum() * ca + (-(cp * ny * L)[sel & ~up]).sum() * sa
    print(f'  x [{lo:.3f},{hi:.3f}): Cd_p upper {dp_u:+.5f} lower {dp_l:+.5f}')
