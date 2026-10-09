"""决定性验证: 用生产 run_multiblock 跑圆柱 (Re=100), 确认 Cd≈1.09。
若成立 -> 生产路径对圆柱(C-grid)正确, bug 是翼型特有(前缘/弯度/尾迹);
若不成立 -> bug 在 run_multiblock 通用路径。
用法: python3 _diag_cyl.py <nsteps> <reg_r>
"""
import sys, time
import numpy as np
sys.path.insert(0, "/workspace")
from tensorlbm import cgrid_gen as CGG, multiblock as MB
nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
reg_r = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
MODE = sys.argv[3] if len(sys.argv) > 3 else "rhie2"
PEXACT = (sys.argv[4] == "1") if len(sys.argv) > 4 else (MODE != "adjoint")
PSCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
PCAP = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0
UCAP = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
X, Y, wall, recv, Lref = CGG.make_cylinder_cgrid(Rb=1.0, Rf=20.0, ni=181, nj=71)
el = dict(name="cyl", X=X, Y=Y, wall=wall, recv=recv, Lref=Lref, ni=X.shape[0]-1,
         nj=X.shape[1]-1, theta=0.0, t=(0.0, 0.0))
geom = dict(elements=[el], Lref=Lref, placement=dict(real=True))
t0 = time.time()
res = MB.run_multiblock(geom, nsteps=nsteps, aoa_deg=0.0, Re=100.0, scheme="ppm",
                        turbulent=False, sa_scheme="upwind1", trip=False, ramp=1,
                        cb="pair", U=1.0, out="_cyl_", wall_fn=False,
                        cfl=0.2, Rhole=0.08, own_margin_frac=1.5, mode=MODE,
                        record_every=1000, reg_r=reg_r, j_floor=2.5e-6,
                        p_exact=PEXACT, p_wall_nojump=PEXACT,
                        proj_ucap=UCAP, p_cap=PCAP, p_scale=PSCALE, p_scale_ref=2e-3)
print(f"ran {time.time()-t0:.1f}s nan={res.get('nan')} Cl_end={res['cl'][-1]:.4f} Cd_end={res['cd'][-1]:.4f}")
print(f"  (expect Cd≈1.09 for cylinder Re=100)")
# Cp / force decomposition on final block  (physical Cp, referenced to far field)
b = res["blocks"][0]
U = 1.0; q_dyn = 0.5 * U * U
Lref = b.Lref; q = q_dyn * Lref
wall = b.wall[:, 0]
pfull = (b.p.reshape(b.ni, b.nj) * getattr(b, "_p_phys_f", 1.0))
pj0 = pfull[:, 0]                               # wall-face (j=0) pressure, (ni,)
pref = float(np.mean(pfull[:, -1]))            # far-field reference (recv line, j=nj-1)
pw = pj0[wall]                                 # body-cell wall pressures
Cp = (pw - pref) / q_dyn
Aex = b.Aef_x[:, 0]; Aey = b.Aef_y[:, 0]
Fx_p = -np.sum((pw - pref) * Aex[wall])
Fy_p = -np.sum((pw - pref) * Aey[wall])
print(f"  Lref={Lref:.3f} q_dyn={q_dyn:.3f} pref={pref:+.4e}")
print(f"  Cp(wall): min={Cp.min():+.3f} max={Cp.max():+.3f}  stag(front,+x)={Cp[:6]}")
print(f"  pressure Fx={Fx_p:+.4e} -> Cd_p={Fx_p/q:+.4f}  (expect Cd_total≈1.09 for cyl Re=100)")
ix_front = int(np.argmax(b.xc[:, 0])); ix_side = int(np.argmin(np.abs(b.xc[:, 0])))
print(f"  Cp_stag@front(i={ix_front},x={b.xc[ix_front,0]:+.2f})={Cp[ix_front]:+.3f}  Cp_side(i={ix_side})={Cp[ix_side]:+.3f}")
# sum of wall face lengths & perimeter check
Awall = np.sum(np.hypot(Aex[wall], Aey[wall]))
print(f"  sum|wall face|={Awall:.4f}  (cylinder circumference=pi*D={np.pi*Lref:.4f})")
# front stagnation among REAL body cells (exclude open-end recv columns i=0,ni-1)
body = wall & (~b.recv[:, 0])
xcw = np.where(body, b.xc[:, 0], -1e9)
ib_front = int(np.argmax(xcw))
Cpw = np.where(body, Cp, -1e9)
ib_top = int(np.argmax(Cpw)); ib_base = int(np.argmin(Cpw))
print(f"  REAL body cells={body.sum()}  front(x=+1): Cp={Cp[ib_front]:+.3f} @(i={ib_front},x={b.xc[ib_front,0]:+.2f})")
# locate cardinal points by geometry among body cells (global indices, orientation check)
def _near(tx, ty):
    d = (b.xc[:, 0] - tx) ** 2 + (b.yc[:, 0] - ty) ** 2
    d = np.where(body, d, 1e9)
    return int(np.argmin(d))
for name, (tx, ty) in [("front(+x)", (1.0, 0.0)), ("back(-x)", (-1.0, 0.0)),
                       ("top(+y)", (0.0, 1.0)), ("bot(-y)", (0.0, -1.0))]:
    ii = _near(tx, ty)
    print(f"  {name:9s} @(i={ii} x={b.xc[ii,0]:+5.2f} y={b.yc[ii,0]:+5.2f}) Cp={Cp[ii]:+6.3f}")
Cpw_min = np.where(body, Cp, 1e9)
jmin = int(np.argmin(Cpw_min))
print(f"  global-min body Cp={Cp[jmin]:+6.3f} @(i={jmin} x={b.xc[jmin,0]:+5.2f} y={b.yc[jmin,0]:+5.2f})")
print(f"  >> physics: windward Cp=+1.0, sides=0, leeward=-1.0; |Cp|~2x & inverted => SIGN+2x pressure bug")
# velocity probe at cardinal body cells (j=2, first interior) to test flow orientation
for name, (tx, ty) in [("front(+x)", (1.0, 0.0)), ("back(-x)", (-1.0, 0.0)),
                       ("top(+y)", (0.0, 1.0)), ("bot(-y)", (0.0, -1.0))]:
    ii = _near(tx, ty)
    uu = b.u[ii, 2]; vv = b.v[ii, 2]
    print(f"  vel[{name:9s}] i={ii}: u={uu:+.3f} v={vv:+.3f} |u|={np.hypot(uu,vv):.3f} (U=1.0, front stag u~0)")
