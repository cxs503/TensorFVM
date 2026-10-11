# SUBOFF AFF-1 reference audit (2026-10-11)

## Status

Drag and Cp use **two independent, explicitly documented operating conditions**; they need not share Reynolds number. The drag literature reproduction parameters are available in [supplemental-notes.md](supplemental-notes.md). The 19-point Cp table remains a secondary-source candidate pending its primary-table audit. This reference package and the [validation specification](../../benchmarks/suboff_aff1_validation.json) do not establish a passed physical benchmark. Unavailable original experimental material parameters remain null.

## Experimental total resistance

Original source: Liu & Huang, *Summary of DARPA SUBOFF Experimental Program Data*, June 1998, CRDKNSWC/HD-1298-11, [official NTIS catalogue](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA359226.xhtml), [official PDF](https://apps.dtic.mil/sti/pdfs/ADA359226.pdf). The PDF endpoint was inaccessible; the original report text was read through an [OCR mirror](https://www.scribd.com/document/1003627685/Huang-Liu-Thomas-Experimental-Results-of-DARPA-Suboff). Table 14 gives AFF-1 resistance, model 5470, DTMB towing tank:

| Speed (knots) | Total resistance (N) |
|---:|---:|
| 5.92 | 87.40 |
| 10.00 | 242.2 |
| 11.84 | 332.9 |
| 13.92 | 451.5 |
| 16.00 | 576.9 |
| 17.99 | 697.0 |

The table lists tripwires at 5% chord. Its residual resistance coefficient is not total Ct. Measurement uncertainty, tank temperature, fluid density and viscosity have not been established from the original source. Convert knots using 1852/3600 when deriving SI values, preserving the speed's experimental rounding.

Independent corroboration: G. Budak (2023), [DOI 10.54926/gdt.1308809](https://doi.org/10.54926/gdt.1308809), [publisher PDF available for external download](https://dergipark.org.tr/en/download/article-file/3184219) (not redistributed; URL and SHA256 retained in the manifest), Tables 4–5. All six forces match. It explicitly uses Ct=Rt/(0.5 rho S U²), S=5.98 m² and reports Ct×1000 = 3.161, 3.070, 3.011, 2.954, 2.857, 2.730. Density/viscosity are unspecified, so these are independently published coefficients, not a source of complete original experimental fluid metadata. Another independent [2021 publication](https://doi.org/10.3390/jmse9121374), [external publisher PDF mirror](https://pdfs.semanticscholar.org/f47b/c3f261aad243de23e79324c957bd17cea3e7.pdf), Table 3 repeats the forces but Table 2 uses S=5.989 m². Its Cp condition differs from the independently verified wind-tunnel Re; do not import that metadata.

## Surface Cp: useful downloadable candidate, pending primary-table audit

[Zenodo DOI 10.5281/zenodo.22692726](https://zenodo.org/records/22692726), published 2026-09-10 by Simvia, contains `experimental_data__exp_cp_hull_bare_table1_huang1992.csv`. The overall dataset concerns AFF-8 CFD, but this individual experimental CSV explicitly identifies bare hull AFF-1. It contains 19 upper-meridian taps from x/L=.035 to .978, measured Cp, empty-tunnel axial-gradient correction, and corrected Cp. It claims exact transcription of Huang et al., 19th Naval Hydrodynamics Symposium, Table 1, p333, proceedings ADA329405. The conference was held in 1992; the CSV cites 1994 publication. No full primary Table 1 scan could be accessed in this audit.

The first four measured values match original 1998 Table 13d's example after rounding. This confirms a subset, not all 19 corrected values. Preserve the downloaded table unchanged. The declared correction equation is inconsistent with three rows:

| x/L | measured − empty correction − listed corrected |
|---:|---:|
| .181 | +.000020 |
| .857 | −.000003 |
| .956 | −.000360 |

Do not silently repair these differences or label all corrected entries authoritative before inspecting the original table. Nose is absent from this 19-tap candidate; it is not a full surface distribution. Original 1998 sample additionally includes measured nose Cp=.990007 and a 21-tap file, but only five example rows are printed. The complete government dataset was on a companion diskette.

Independent primary proceedings corroboration: P. Bull, [Twenty-First Symposium on Naval Hydrodynamics, pp1061–1076](https://www.nationalacademies.org/read/5870/chapter/74), confirms AFF-1 and AFF-8 zero incidence wind-tunnel experiments at Re_L=1.2×10^7, and 95% Cp uncertainty ±.015 absolute. [Morse & Mahesh 2023, JFM](https://doi.org/10.1017/jfm.2023.777), author PDF downloaded from the University of Michigan, independently confirms this experimental Re and Cp normalization (p−p∞)/(0.5 rho U∞²); Table 2 specifies a .00125D trip at .425D from the nose. Their own LES Re is lower and must not be substituted for these experiments.

For the 19 listed corrected values, the unweighted norm is .546488. An absolute .015 uncertainty per tap gives a conservative norm bound .015 sqrt(19)/.546488 = 11.9643%; correlation information is unavailable. Thus <3% agreement with nominal tabulated values can be a numerical comparison criterion, but it does not establish physical accuracy within 3% under experimental uncertainty. The original 1998 report states asymmetric general Cp bounds +.010/−.0085, with larger nose bounds; do not combine uncertainty statements from different sources without tracing definitions.

## Required next evidence

1. Obtain and inspect original Huang Table 1, including the three inconsistent corrected entries, correction convention and reference pressure.
2. Obtain original tow-tank fluid properties, speed uncertainty and drag uncertainty. Establish exact Re and normalization before promoting Ct.
3. Use separate drag and Cp operating conditions as specified. The drag reproduction at U=3.046 m/s uses independently published computational material parameters; Cp uses the documented wind-tunnel Re=1.2e7. Clearly distinguish original experimental metadata from literature reproduction choices.
4. Compare integrated drag and tap-sampled Cp to their own documented experimental conditions. Report pressure reference, tap interpolation, modeled trip/transition, domain and mesh/time uncertainty.

## Independent genuine 3D manufactured solution

Use a stationary ABC velocity on a three-dimensional mixed-cell domain, with exact boundary data, laminar viscosity and prescribed volumetric momentum forcing. Let X=kx, Y=ky, Z=kz, and

u=U(sin Z+cos Y, sin X+cos Z, sin Y+cos X),
p=rho U² sin X cos Y sin Z, K=|u|²/2.

Analytically div u=0, curl u=k u, laplacian u=−k²u, and (u·grad)u=grad K. Therefore incompressible Navier–Stokes is satisfied by f=grad(K+p/rho)+nu k²u (acceleration units). Evaluate the forcing analytically independently of the production FV operators. All three velocity components, all spatial directions and nonlinear pressure/velocity cancellation are exercised. Pressure gradients and face-normal boundary conditions follow exact derivatives; fix a pressure gauge consistently.

Apply exact Dirichlet velocity on every boundary, including curved immersed body surfaces, and compatible exact pressure boundary values/normal derivatives where the solver requires them. Such body boundary data is a manufactured mathematical case, not physical no-slip SUBOFF. Test at least three refined genuinely 3D mixed grids; report volume-weighted relative L2 errors in all velocity components and pressure after gauge removal, continuity and momentum residuals, observed refinement order and boundary flux error. Require velocity and pressure <3%, not merely sparse linear residuals. This verifies the native solver but cannot substitute for experimental SUBOFF drag/Cp acceptance. No manufactured computation is claimed by this reference-only audit.

## Reproducibility

`evidence-manifest.json` records hashes and intentionally null unavailable experimental metadata. Both experimental CSV SHA256 values match the downloadable Zenodo CHECKSUMS file. All three downloaded PDFs are cached locally under ignored `results/suboff-reference-source-pdfs/` and marked `external_download` in the manifest. The repository preserves their URLs and SHA256 values, not the full PDFs. CSV data and their upstream documentation remain unchanged.

## Attribution and redistribution

- **Simvia (2026), v1.0**, *DARPA SUBOFF AFF-8 Submarine: RANS Resistance and Actuator-Disk Self-Propulsion Simulations with code_saturne9.1*, [DOI10.5281/zenodo.22692726](https://zenodo.org/records/22692726): the two downloaded experimental CSVs, upstream README and checksums are redistributed **unmodified** under [CC BY4.0](https://creativecommons.org/licenses/by/4.0/), as declared by the record API. Original experimental authors and reports remain cited in the CSV headers. This license statement does not certify the transcription accuracy.
- **Nicholas Morse and Krishnan Mahesh (2023)**, *Tripping effects on model-scale studies of flow over the DARPA SUBOFF*, [DOI10.1017/jfm.2023.777](https://doi.org/10.1017/jfm.2023.777): original PDF and Cambridge article page explicitly declare [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). Downloaded PDF is unmodified and stored only in the ignored local cache to reduce repository size.
- **Jingyuan Li, Qinghe Zhang and Tongqing Chen (2021)**, *Numerical Investigation of Internal Solitary Wave Forces on Submarines in Continuously Stratified Fluids*, [DOI10.3390/jmse9121374](https://doi.org/10.3390/jmse9121374): original publisher PDF declares [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). The unmodified SemanticScholar-hosted publisher PDF is stored only in the ignored local cache.
- **Gokhan Budak (2023)**, *Non-dimensional Hydrodynamic Coefficients Determination of a Derived-Submarine Bare Hull Form*, [DOI10.54926/gdt.1308809](https://doi.org/10.54926/gdt.1308809): no explicit redistribution license was established; the PDF is not included in the published repository. Only its source URL, hash, citation and limited numerical metadata are retained.

The reference-specific CC BY attribution applies to downloaded third-party material; it does not replace the TensorFVM code license.
