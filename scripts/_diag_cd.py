"""Cd 过估诊断：把力拆成压力阻力/摩擦阻力，沿物面输出 Cp、壁面剪切(分离指示)、
y+ 与 nu_tilde 场统计。用与 run_multiblock 相同的 c_forces 算法(blk.p, 未去膨胀)，
所以 Cd 分解与报告一致。

用法: python3 _diag_cd.py <nsteps> <pc> <ps> <sr> <wnj> [p_exact]
默认: 15000 8 0.05 2e-3 0 1   (CONV 稳定配置)
输出: _cd_<tag>.npz + 文本摘要(分离区/块分解)
"""
import sys, time
import numpy as np
sys.path.insert(0, "/workspace")
from tensorlbm import cgrid_gen as CGG, multiblock as MB, curve_grid as CG

nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
pc = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
ps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
sr = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-3
wnj = bool(int(float(sys.argv[5]))) if len(sys.argv) > 5 else False
p_exact = bool(int(float(sys.argv[6]))) if len(sys.argv) > 6 else True
ntc_mult = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0   # nu_t_cap = mult*nu_main
MODE = sys.argv[8] if len(sys.argv) > 8 else "rhie2"
U = 1.0
nu_main = U * 0.6837 / 9e6
ntc = None if ntc_mult <= 0 else nu_main * ntc_mult
tag = f"cd_n{nsteps}_pc{pc:g}_ps{ps:g}_sr{sr:g}_wnj{int(wnj)}_ntc{ntc_mult:g}_m{MODE}"

geom = CGG.make_three_element_cgrid_real(
    ni_m=121, nj_m=61, ni_s=61, nj_s=41, ni_f=61, nj_f=41,
    Rf_base=18.0, h_wall_abs=2.0e-4)
t0 = time.time()
res = MB.run_multiblock(geom, nsteps=nsteps, aoa_deg=8.1, Re=9e6, scheme="ppm",
                        turbulent=True, sa_scheme="upwind1", trip=True, ramp=1,
                        cb="pair", U=U, out="_cd_" + tag, wall_fn=True,
                        trip_strip_frac=0.07, trip_band=0.02, cfl=0.2, Rhole=0.08,
                        own_margin_frac=1.5, mode=MODE, record_every=500,
                        reg_r=0.0, j_floor=2.5e-6, p_exact=p_exact,
                        proj_ucap=1.0, p_cap=pc, p_scale=ps, p_scale_ref=sr,
                        p_wall_nojump=wnj, nu_t_cap=ntc)
_el = time.time() - t0
print(f"[cd {tag}] ran {_el:.1f}s nan={res.get('nan')}")


def wall_yplus(blk, nu):
    """复制 c_wall_function 的 fixed-point, 返回 (ni,) 体轴 y+ (仅壁面行 d_in)."""
    wall0 = blk.wall[:, 0]
    if not wall0.any():
        return np.zeros(blk.ni)
    Aex = blk.Aef_x[:, 0]; Aey = blk.Aef_y[:, 0]
    A = np.maximum(np.hypot(Aex, Aey), 1e-30)
    tx = -Aey / A; ty = Aex / A
    d = CG.c_walldist(blk)
    d_in = d[:, 1]
    ut = blk.u[:, 1] * tx + blk.v[:, 1] * ty
    absut = np.abs(ut)
    kappa, E = 0.41, 9.8
    yp0 = np.maximum(absut * d_in / nu, 1e-6)
    utau = np.maximum(absut * kappa / np.maximum(np.log(E * yp0), 1e-3), 1e-9)
    for _ in range(12):
        yp = np.maximum(utau * d_in / nu, 1e-6)
        utau = np.maximum(absut * kappa / np.maximum(np.log(E * yp), 1e-3), 1e-9)
    yp = utau * d_in / nu
    return yp


def force_decomp(blk, nu, U, Lref):
    j = 0
    wall = blk.wall[:, 0]
    pcol = blk.p[:, j]
    # checkerboard-free wall pressure: NON-wrapping 1-2-1 average (matches c_forces
    # in curve_grid.py).  np.roll would WRAP across the wake cut (C-grid j=0 TE
    # seam) and pollute the force -- do NOT wrap.
    p = pcol.copy()
    p[1:-1] = 0.25 * pcol[:-2] + 0.5 * pcol[1:-1] + 0.25 * pcol[2:]
    Aix = blk.Aef_x[:, j]; Aiy = blk.Aef_y[:, j]
    A = np.maximum(np.hypot(Aix, Aiy), 1e-30)
    nx = Aix / A; ny = Aiy / A
    tx = -ny; ty = nx
    Fpx = -np.sum(p[wall] * Aix[wall]); Fpy = -np.sum(p[wall] * Aiy[wall])
    gxu, gyu = CG.c_grad(blk, blk.u); gxv, gyv = CG.c_grad(blk, blk.v)
    ux = gxu[:, j]; uy = gyu[:, j]; vx = gxv[:, j]; vy = gyv[:, j]
    txx = 2.0 * nu * ux; txy = nu * (uy + vx); tyy = 2.0 * nu * vy
    Fvx = np.sum(txx[wall] * Aix[wall] + txy[wall] * Aiy[wall])
    Fvy = np.sum(txy[wall] * Aix[wall] + tyy[wall] * Aiy[wall])
    Fxt = Fpx + Fvx; Fyt = Fpy + Fvy
    q = 0.5 * U * U * Lref
    aoa = np.deg2rad(8.1)
    Cd = (Fxt * np.cos(aoa) + Fyt * np.sin(aoa)) / q
    Cl = (-Fxt * np.sin(aoa) + Fyt * np.cos(aoa)) / q
    # per-surface diagnostics (body-wall cells only)
    iw = np.where(wall)[0]
    Cp = p[wall] / q
    tauw = txx[wall] * tx[wall] + txy[wall] * ty[wall]   # 切向壁面剪切
    yp = wall_yplus(blk, nu)
    xc = blk.xc[wall, 0]; yc = blk.yc[wall, 0]
    # 沿物面弧长排序(用 x 粗糙排序 -> 实际按 i, C-grid j=0 从 TE 下表面->LE->TE 上表面)
    return dict(name=getattr(blk, "name", "?"), Cd=Cd, Cl=Cl,
                Cd_p=Fpx / q, Cd_f=Fvx / q, Cl_p=Fpy / q, Cl_f=Fvy / q,
                iw=iw, xc=xc, yc=yc, Cp=Cp, tauw=tauw, yp=yp[wall])


blocks = res["blocks"]
print(f"\n=== 总力 (multiblock_forces) ===")
Cd_t, Cl_t, Fxt, Fyt, Lref = MB.multiblock_forces(blocks, U, 8.1)
print(f"  Cd={Cd_t:.4f}  Cl={Cl_t:.4f}  Lref={Lref:.4f}")

out = {"tag": tag, "Cd_tot": Cd_t, "Cl_tot": Cl_t, "Lref": Lref}
surf = {}
Lref_all = max(b.Lref for b in blocks)   # 统一用主弦 Lref, 与 multiblock_forces 一致
print(f"\n=== 逐块力分解 (Cd=压力阻力+摩擦阻力, 归一化 Lref={Lref_all:.4f}) ===")
print(f"{'block':6s} {'Cd':>7s} {'Cd_p':>7s} {'Cd_f':>7s} {'Cl':>7s} {'Cl_p':>7s} {'Cl_f':>7s} "
      f"{'|u|max':>8s} {'yp_max':>7s} {'yp_mean':>7s} {'sep%':>6s}")
for b in blocks:
    d = force_decomp(b, b._nu, U, Lref_all)
    surf[d["name"]] = d
    umax = float(np.hypot(b.u, b.v).max())
    yp_max = float(d["yp"].max()); yp_mean = float(d["yp"].mean())
    sep = float((d["tauw"] < 0).mean() * 100)   # 壁面剪切为负的弧长占比(回流/分离)
    print(f"{d['name']:6s} {d['Cd']:7.4f} {d['Cd_p']:7.4f} {d['Cd_f']:7.4f} {d['Cl']:7.4f} "
          f"{d['Cl_p']:7.4f} {d['Cl_f']:7.4f} {umax:8.3f} {yp_max:7.2f} {yp_mean:7.2f} {sep:6.1f}")
    nui = b.nu_tilde
    nmax = float(nui.max()); nmean = float(nui[~b.recv].mean())
    print(f"      nu_tilde: max={nmax:.3e} mean(inner)={nmean:.3e}  "
          f"nu={b._nu:.3e} nu_t_cap={b._nu_t_cap:.3e}")
    tw = d["tauw"]; iw = d["iw"]
    flips = np.where(np.diff(np.sign(tw)) != 0)[0]
    if len(flips):
        fx = d["xc"][flips]
        print(f"      壁面剪切符号翻转(分离/再附) i={iw[flips].tolist()}  "
              f"x/c≈{[round(float(x), 3) for x in fx]}")

# 单独存 surf 供画图
np.savez(f"/workspace/_cd_surf_{tag}.npz",
         **{f"{nm}_{k}": v for nm, d in surf.items() for k, v in d.items() if k != "name"})
print(f"\n[cd {tag}] Cd_tot={Cd_t:.4f} Cl_tot={Cl_t:.4f}  saved _cd_surf_{tag}.npz")
