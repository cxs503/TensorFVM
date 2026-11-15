# 证据目录（round-10）

- `ladder_profile_nx25600_weno5_cfl0.8.npz` / `ladder_metrics_nx25600_weno5_cfl0.8.json`
  —— round-7 作业 `cj3782` 的产出，此前从未取回本库（`R7-REF25600-CLOSURE.md` 因此记「仍未回填」）。
  本轮把它从集群工件中**取回并归档为旁证**：它是 CFL 0.8 放松时间步的变体，
  **不得**以标准档名入库（E0a 会判 fail），仅用于对 round-7 的 A0 探针做交叉核对。
- `rerun_nx6400_weno5.npz` / `rerun_nx6400_weno5.json`
  —— round-10 作业 `cj4143` 在同一作业内重跑的控制档（CFL 0.4），用于 E0b 仪器可复现性。
  作业未返回前本目录没有这两个文件，E0b 记 `not_measured`。
