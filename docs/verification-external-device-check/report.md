# 外流 CPU/CUDA 一致性辅助检查

32×12控制体、每例五个实际 SIMPLE 步。CPU SuperLU 与 CUDA Torch Krylov 求解相同离散方程。该检查不能证明稳态、3%物理精度或加速比。

[
  {
    "case": "cylinder",
    "grid": "32x12",
    "steps": 5,
    "relative_l2_errors": {
      "velocity": 9.309392399201723e-12,
      "p": 8.084044826811825e-11,
      "mass_flux": 1.8428400765261447e-11
    },
    "consistency_passed": true,
    "physical_passed": null
  },
  {
    "case": "naca",
    "grid": "32x12",
    "steps": 5,
    "relative_l2_errors": {
      "velocity": 1.7096398054672998e-10,
      "p": 6.777143802471188e-09,
      "mass_flux": 2.3660495562966973e-11
    },
    "consistency_passed": true,
    "physical_passed": null
  }
]
