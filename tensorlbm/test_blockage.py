import sys
from overset_cylinder_fvm import run_cylinder

mode = sys.argv[1] if len(sys.argv) > 1 else "smallD"

if mode == "smallD":
    # 用户方案：圆柱直径减半 (R=0.06, D=0.12)，背景网格保持 600x240 不变 -> 总格数不增加
    # 径向半径按 0.6 同比例缩放；阻塞比 D/Ly = 0.12/1.0 = 12% (原 20%)
    # 注意：背景 ppD = D/hx = 0.12/(2.5/600) = 28.8 -> stair-step 比原 48ppD 更粗
    run_cylinder(nsteps=12000, Nx=600, Ny=240, Lx=2.5, Ly=1.0,
                 R=0.06, Rh_hole=0.252, Rf=0.36, Rf_og=0.468, Rf_sponge=0.3,
                 ni=240, nj=48, beta=2.5, coupling="bg2og",
                 out_prefix="test_smallD", probe_every=500, field_every=12000)

elif mode == "largeH":
    # 推荐方案：保持 D=0.2 / 48ppD 不变，仅把背景域加高 Ly 1.0->2.0 (5D->10D)
    # Ny 480 保持 48ppD；阻塞比 D/Ly = 0.2/2.0 = 10% (原 20%)；总格数 600x480
    run_cylinder(nsteps=12000, Nx=600, Ny=480, Lx=2.5, Ly=2.0,
                 R=0.1, Rh_hole=0.42, Rf=0.6, Rf_og=0.78, Rf_sponge=0.5,
                 ni=240, nj=48, beta=2.5, coupling="bg2og",
                 out_prefix="test_largeH", probe_every=500, field_every=12000)
