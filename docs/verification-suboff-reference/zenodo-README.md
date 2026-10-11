# RANS resistance and actuator-disk self-propulsion of the DARPA SUBOFF AFF-8 submarine, code_saturne 9.1

Steady RANS (k-ω SST) hull resistance and a milestone-1 self-propulsion
study for the **DARPA SUBOFF AFF-8** configuration (bare hull + sail +
four cruciform stern appendages), computed with **code_saturne v9.1**
(EDF, GPL) on a half-model (symmetry plane z = 0), incompressible
segregated solver, SIMPLEC, wall functions.

Author: Florian Hermet (Simvia) — florian.hermet@simvia.tech

## Contents

Two independent cases:

- **RANS** — towed-hull resistance sweep at the 6 speeds tested in the
  reference tow-tank campaign (5.93 to 17.79 kn), validated against
  DTMB experimental data.
- **RANS_SELFPROP** — self-propulsion milestone 1: a thrust-only
  Hough–Ordway actuator disk at the propeller plane, driven by an
  in-run controller that converges to the self-propulsion point,
  restarted from the converged 5.93 kn towed field.

```
case_setup/
  RANS/
    setup.xml                 code_saturne case definition (notebook
                               variable UX = inlet speed, m/s; template
                               value 2.572, overridden per run — see
                               results/RANS/<run>/setup.xml for the
                               value actually used)
    run.cfg                    run configuration
    src/cs_user_extra_operations.cpp   resistance decomposition (below)
  RANS_SELFPROP/
    setup.xml, run.cfg
    src/cs_user_extra_operations.cpp   resistance decomposition + link
                                       to the actuator-disk controller
    src/cs_user_source_terms.cpp       Hough–Ordway actuator disk +
                                       self-propulsion controller (below)
results/
  RANS/
    resistance_validation.csv  CFD vs experiment, all 6 speeds (summary)
    run_exp_<speed>/
      resistance.csv           per-iteration force decomposition
      residuals.csv             solver residuals
      timer_stats.csv           per-iteration timing
      setup.xml                 resolved case (actual UX value used)
      logs/                     summary, performance.log, setup.log,
                                 compile.log, run_solver.log.zst
  RANS_SELFPROP/
    selfprop_5.93kn/
      resistance.csv, residuals.csv, timer_stats.csv, setup.xml
      selfprop.csv              controller history: T (disk thrust),
                                 R (measured half-body drag), imbalance
      logs/                     as above
fields_ensight/
  RANS_run_exp_<speed>_fields_ensight.tar.zst        (6 files)
  RANS_SELFPROP_selfprop_5.93kn_fields_ensight.tar.zst
    Full-field EnSight Gold output of the converged state: volume
    (Velocity, Pressure, k, omega, TurbVisc, total_pressure,
    Wall_distance, Courant/Fourier numbers) and boundary (per-zone
    boundary_stress, Shear_Stress, Yplus). Compressed with zstd -19;
    decompress with `tar --zstd -xf <file>` or `zstd -d <file> | tar -x`.
MESH/
  darpa_mesh.med              half-model mesh, 459 209 cells, symmetry
                               plane z = 0, flow along +x. Derived from
                               the public BARAM SUBOFF tutorial mesh
                               (OpenFOAM polyMesh) and converted to MED.
                               Wetted area from the mesh: 6.3367 m²,
                               matching the published AFF-8 value.
experimental_data/
  dtmb_tow_tank_table14.csv               Rt vs speed, DTMB tow tank
  exp_cp_hull_bare_table1_huang1992.csv    Cp, bare-hull, 19 taps
  exp_cp_hull_aff8_primary_fig9.csv        Cp, AFF-8, digitised Fig. 9
  exp_cp_profile_xL0978_bare_primary_fig10.csv  Cp radial profile, x/L=0.978
  exp_ux_xL0978_aff8_primary_fig16f.csv    Ux azimuthal-mean, x/L=0.978
  exp_cf_hull_aff8_primary_fig13.csv       Cf, AFF-8, digitised Fig. 13
  (transcribed/digitised from the cited reports below; not a
  redistribution of the source PDFs, which are not included here)
CHECKSUMS.sha256
```

## Resistance decomposition (RANS)

`cs_user_extra_operations.cpp` computes, every iteration:

- forces summed per zone (hull, sail, fins) and split viscous/pressure
  from the boundary stress, on the half body, then **doubled** (×2) for
  the symmetry-plane half-model to give full-body values;
- `q_ref = 0.5 · ρ · U_inlet²`, ρ = 998.2 kg/m³ (matches the GUI fluid
  property);
- `S_wet` = 6.3367 m² is the full-body wetted area, computed from the
  mesh itself (not hardcoded) and doubled the same way, reported per
  iteration as a sanity check.

`resistance.csv` columns: `iteration, U_ref, q_ref, S_wet, Rv, Rp, Rt,
Cf, Cp, Cd, Fx_hull, Fx_sail, Fx_fins` (Rv/Rp/Rt/Cf/Cp/Cd are full-body;
Fx_* are the half-body per-zone breakdown, N).

### Headline results (mean of the last 300 iterations of 2000)

| Speed (kn) | U (m/s) | Rt CFD (N) | Cd CFD | Rt exp. (N) | error |
|---:|---:|---:|---:|---:|---:|
| 5.93  | 3.0506 | 102.18 | 3.4719e-3 | 102.3 | −0.11 % |
| 10.00 | 5.1444 | 273.36 | 3.2660e-3 | 283.8 | −3.68 % |
| 11.85 | 6.0961 | 376.16 | 3.2005e-3 | 389.2 | −3.35 % |
| 13.92 | 7.1610 | 506.95 | 3.1258e-3 | 526.6 | −3.73 % |
| 16.00 | 8.2310 | 660.77 | 3.0839e-3 | 675.6 | −2.19 % |
| 17.79 | 9.1519 | 805.27 | 3.0400e-3 | 821.1 | −1.93 % |

Experimental reference: Liu & Huang, DTMB tow-tank campaign, 1998
(CRDKNSWC/HD-1298-11, DTIC ADA359226), Table 14.

## Self-propulsion, milestone 1: actuator disk

`cs_user_source_terms.cpp` applies a thrust-only momentum source term
over an annular disk at the propeller plane (x/L = 0.978, radii adapted
from the INSEAN E1619 propeller: D = 0.262 m, hub 0.226 D), with the
**Hough–Ordway** radial thrust distribution
f_x(r*) ∝ r*·√(1−r*), r* = (r − r_hub)/(R − r_hub) — thrust only, no
swirl, compatible with the symmetric half-model. No mesh modification:
592 cells flagged by geometric criteria.

The case restarts from the converged 5.93 kn towed field. An in-run
controller drives the disk to the self-propulsion point every 50
iterations (after a 200-iteration settling window):

```
T ← T + 0.35 · (R_half − T)
```

where `R_half` is the half-body drag measured live by
`cs_user_extra_operations.cpp` (shared via the global `cs_selfprop_rx`)
and `T` is the disk thrust (half-model). At convergence T = R: the
model's self-propulsion point. History in `selfprop.csv`
(`iteration, T_half_N, R_half_N, imbalance_pct`).

### Headline results (mean of the last 300 iterations)

| Quantity | This study (actuator disk) | Guo et al. 2023 (rotating E1619 blades)\* |
|---|---:|---:|
| T_sp (full body) | **113.9 N** | 113.0 N |
| J (equivalent)\*\* | 0.75–0.77 | 0.772 |
| n (equivalent)\*\* | 10.1–10.2 rps | 10.20 rps |
| t (thrust deduction) | 0.101 | 0.130 |

\* C. Guo, X. Wang, C. Chen, Y. Li, J. Hu, *Numerical Investigation of
Self-Propulsion Performance and Noise Level of DARPA Suboff Model*,
J. Mar. Sci. Eng. 2023, 11, 1206, https://doi.org/10.3390/jmse11061206
— STAR-CCM+, fully meshed rotating E1619 blades. Sezen et al. (cited
therein): J = 0.768, t = 0.131.

\*\* The actuator disk carries no rotation rate or blade geometry of
its own (thrust only); J and n are not computed here but quoted for
comparison, taken from the literature rotating-blade result at a
matching thrust. The gap on t is dominated by the reference towed
resistance (this study: −0.1 % vs tow-tank experiment; Guo et al.:
−3.7 %), not by the disk model itself.

The disk-thrust self-propulsion point matches the fully meshed
rotating-blade CFD of Guo et al. to **under 1 %**, on a 459 k-cell
pre-project mesh.

## Reproducing the runs

Solver: code_saturne 9.1, Docker image `simvia/code_saturne:latest`,
16 MPI ranks, `CS_MPIEXEC_OPTIONS="--mca btl sm,self,tcp"`.

```
docker run --rm --shm-size=48g \
  -v <workdir>:/home/code_saturne \
  -e CS_MPIEXEC_OPTIONS="--mca btl sm,self,tcp" \
  simvia/code_saturne:latest \
  code_saturne run --notebook-args UX=<speed_m_s> --id <run_id>
```

Place `darpa_mesh.med` in a `MESH/` directory next to the case as
referenced by `setup.xml`. RANS: 2000 pseudo-steady iterations per
speed. RANS_SELFPROP: restarted from the converged `run_exp_5.93kn`
checkpoint (not included here — regenerate by running RANS first),
then run with the actuator-disk source term until the controller
converges (~3500 iterations from restart in this study).

## Licensing

- `case_setup/*/src/*.cpp` derive from code_saturne user-function
  templates: **GPL-2.0-or-later**.
- Everything else (case setup, results, mesh, this README):
  **CC-BY 4.0**.

## Related resources

- SUBOFF geometry and towed-resistance reference data: T. T. Huang,
  H.-L. Liu, *Experiments of the DARPA SUBOFF Program*, DTRC/SHD-1298-02,
  Dec. 1989 (DTIC ADA218797); T. T. Huang et al., 19th Symposium on
  Naval Hydrodynamics, 1992 (DTIC ADA329405, hull Cp/Cf); H.-L. Liu,
  T. T. Huang, *Summary of DARPA SUBOFF Experimental Program Data*,
  CRDKNSWC/HD-1298-11, June 1998 (DTIC ADA359226, Table 14 resistance).
  DTIC accession pages: https://apps.dtic.mil/sti/citations/<accession>
- Self-propulsion / E1619 propeller validation reference: Guo et al.
  2023, https://doi.org/10.3390/jmse11061206
- code_saturne: https://www.code-saturne.org
- Companion write-up: https://simvia.tech/en/blog/code-saturne-naval-hydrodynamics-darpa-suboff
