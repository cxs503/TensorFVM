"""Fusion check: prove our 30P30N geometry == TensorFVM's canonical geometry.

Our solver builds the three-element airfoil from tensorlbm/airfoil_data/gh_*.dat
(docstring: the Wolf-Dynamics / linuxguy123 30P-30N validation case).  TensorFVM
ships the SAME canonical coordinates in src/tensorfvm/data/30p30n/{slat,main,flap}.dat
(README cites linuxguy123/30P-30N-Validation-Case).  This script does the decisive
pointwise comparison so geometry can be ruled out as a confound for the Cd bug.

Run:  python3 _fuse_geom.py
"""
import numpy as np
import os

OUR = "airfoil_data"
TFVM = "/workspace/TensorFVM/src/tensorfvm/data/30p30n"
PAIRS = [("slat", "gh_Slat.dat", "slat.dat"),
         ("main", "gh_Main.dat", "main.dat"),
         ("flap", "gh_Flap.dat", "flap.dat")]


def main():
    print("=" * 70)
    print("30P30N geometry: OUR (gh_*.dat)  vs  TensorFVM (canonical .dat)")
    print("=" * 70)
    all_identical = True
    for name, gh, tf in PAIRS:
        a = np.loadtxt(os.path.join(OUR, gh))
        b = np.loadtxt(os.path.join(TFVM, tf))
        same_shape = a.shape == b.shape
        dx = np.max(np.abs(a[:, 0] - b[:, 0])) if same_shape else float("nan")
        dy = np.max(np.abs(a[:, 1] - b[:, 1])) if same_shape else float("nan")
        identical = same_shape and dx < 1e-12 and dy < 1e-12
        all_identical = all_identical and identical
        print(f"\n[{name}]")
        print(f"  our  : n={len(a)}  x[{a[:,0].min():.4f},{a[:,0].max():.4f}] "
              f"y[{a[:,1].min():.4f},{a[:,1].max():.4f}]")
        print(f"  tfvm : n={len(b)}  x[{b[:,0].min():.4f},{b[:,0].max():.4f}] "
              f"y[{b[:,1].min():.4f},{b[:,1].max():.4f}]")
        print(f"  shape_match = {same_shape}")
        print(f"  max|dx| = {dx:.2e}   max|dy| = {dy:.2e}")
        print(f"  IDENTICAL = {identical}")
    print("\n" + "=" * 70)
    verdict = ("PASS -- geometry is byte-for-byte identical; NOT a confound."
               if all_identical else
               "DIFFERS -- geometry mismatch may contribute to the Cd bug.")
    print("VERDICT:", verdict)
    print("=" * 70)
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
