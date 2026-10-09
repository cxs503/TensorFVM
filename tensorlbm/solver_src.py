"""Full overset cylinder run with a selectable convective scheme.

Usage:
  python3 run_muscl_overset.py --scheme muscl --nj 48 --beta 2.5 --nsteps 8000
  python3 run_muscl_overset.py --scheme muscl --nj 96 --beta 1.2 --nsteps 12000
"""
import argparse
import overset_cylinder_fvm as M

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", default="muscl", choices=["upwind1", "muscl"])
ap.add_argument("--nj", type=int, default=48)
ap.add_argument("--ni", type=int, default=120, help="O-grid angular resolution")
ap.add_argument("--Nx", type=int, default=120)
ap.add_argument("--Ny", type=int, default=48)
ap.add_argument("--Lx", type=float, default=2.5)
ap.add_argument("--Ly", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=2.5)
ap.add_argument("--nsteps", type=int, default=8000)
ap.add_argument("--Rf_sponge", type=float, default=0.76, help="O-grid fringe inner radius (og.recv=r>=this); keep > --Rf to avoid a doubly-circular overlap band")
ap.add_argument("--Rf", type=float, default=0.72, help="background hole-recv outer radius (bg.recv=r in [Rh_hole,Rf])")
ap.add_argument("--Rf_og", type=float, default=0.95, help="O-grid outer radius (>Rf_sponge for a sponge band)")
ap.add_argument("--aoa", type=float, default=3.0)
ap.add_argument("--fsig", type=float, default=0.0, help="theta Shapiro filter sigma (time stepping)")
ap.add_argument("--reg_r", type=float, default=0.0, help="radial 2nd-order Poisson regularisation")
ap.add_argument("--fv", type=float, default=0.0, help="radial Shapiro filter sigma on O-grid velocity AFTER projection (kills velocity checkerboard)")
ap.add_argument("--fck", type=float, default=1.0, help="radial (-1)^j checkerboard removal alpha on O-grid velocity AFTER projection (0=off, 1=full)")
ap.add_argument("--hv", type=float, default=0.0, help="local 4th-order hyperviscosity CFL (damps grid-scale near-wall noise)")
ap.add_argument("--rc", action="store_true", help="use Rhie-Chow in projection (unstable on stretched O-grid)")
ap.add_argument("--coupling", default="oneway", choices=["oneway", "twoway", "og2bg", "bg2og"],
                help="overset coupling: oneway=stable no-shed; twoway=full Chimera (drifts); "
                     "og2bg=background carries wake, O-grid clamped stable (RECOMMENDED)")
ap.add_argument("--tw_blend", type=float, default=1.0,
                help="two-way back-coupling blend 0<w<=1 (1=hard takeover, <1 relaxes drift)")
ap.add_argument("--mci", default="none", choices=["none", "bg", "og", "both"],
                help="Mass-Conserving Interpolation: which fringe bands to make div-free")
ap.add_argument("--schwarz", type=int, default=1,
                help="overlapping-Schwarz coupling iterations (>=2 damps two-way drift)")
ap.add_argument("--out", default="probe_muscl")
a = ap.parse_args()

bg, og, hist, summary, dump = M.run_cylinder(
    nsteps=a.nsteps,
    Nx=a.Nx, Ny=a.Ny, Lx=a.Lx, Ly=a.Ly,
    ni=a.ni, nj=a.nj, R=0.1, Rf=a.Rf, Rf_og=a.Rf_og, beta=a.beta,
    aoa_deg=a.aoa, Rf_sponge=a.Rf_sponge, conv_scheme=a.scheme,
    conv_filter_sigma=a.fsig, reg_r=a.reg_r, fv_sigma=a.fv, fck=a.fck, rc=a.rc,
    hv_cfl=a.hv, coupling=a.coupling, tw_blend=a.tw_blend, mci=a.mci, schwarz=a.schwarz,
    probe_every=200, out_prefix=a.out,
    field_dump=[], field_every=999999,
)
print("\nSUMMARY:", summary)
