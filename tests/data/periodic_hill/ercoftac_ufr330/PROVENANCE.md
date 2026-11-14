# Provenance — ERCOFTAC UFR 3-30 reference data (CASE-FVM-PERIODIC-HILLS)

## What these files are

`UFR3-30_C_2800_data_MB-001.dat` … `-010.dat` are the mean/statistics profiles
distributed by **Prof. Michael Breuer (Helmut-Schmidt-Universität Hamburg,
"Professur fuer Stroemungsmechanik")**, author of the reference paper of this
case, for **case 4 = Re = 2800 DNS, code LESOCC**:

> Breuer, M., Peller, N., Rapp, Ch., Manhart, M. (2009): *Flow over Periodic
> Hills — Numerical and Experimental Study in a Wide Range of Reynolds
> Numbers*, Computers & Fluids 38(2):433–457,
> doi:10.1016/j.compfluid.2008.05.002

Each file carries a 7-column header `y/h, U/U_b, V/U_b, uu/U_b^2, vv/U_b^2,
uv/U_b^2, k/U_b^2` and is a full wall-normal profile from the hill surface to
the channel top (`0 <= y/h <= 3.033`).

## Retrieval

* Source page: `https://kbwiki.ercoftac.org/w/index.php?title=UFR_3-30_Test_Case`
  (§ *CFD data available → Curvilinear grids → Case 4 DNS (LESOCC) Re = 2800*).
* Retrieval: file URLs resolved through the wiki's `api.php`
  (`prop=imageinfo&iiprop=url`), then fetched verbatim; **unmodified**.
* The reattachment length used as the acceptance target is quoted from
  `https://kbwiki.ercoftac.org/w/index.php?title=UFR_3-30_Evaluation`
  (§ *Separation and Reattachment Lengths*), a page written by the case's own
  authors (Rapp, Breuer, Manhart, Peller):

  > "The corresponding values of the reattachment lengths are
  > x_R/h = 5.24, 5.19, 5.41, 5.09, and 4.69 for Re = 700 to 10,595,
  > respectively."

  Re sequence 700, 1400, 2800, 5600, 10595 ⇒ **x_R/h(Re=2800) = 5.41**
  (also fixed by the adjacent sentence "the reattachment length increases again
  in this narrow Re-range [1400, 2800]", i.e. 5.19 → 5.41).

## Station assignment

The wiki does not print `x/h` inside the files. The 10 canonical UFR 3-30
stations `x/h = 0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8` are assigned in file order,
corroborated independently by (a) the repository's own geometric admission
filter, which rejects exactly `{0.05, 0.5, 1, 8}` because the sampling line is
inside the hill body there, and (b) the physical shape of the recovery: the
`y/H = 0.5` velocity rises monotonically to a maximum at the 9th station and
decelerates on the windward side of the next hill (10th). Any other ordering
gives a non-physical, non-monotone recovery.

## Caveats (honest)

* `y/H = 0.5` values are obtained by linear interpolation between the two
  nearest distributed samples; the local spacing is ~1e-2 in `y/H` and the
  velocity varies by < 1e-2 between them, so the interpolation error is far
  below the 10 % acceptance band.
* The separation length at Re = 2800 is **not** tabulated in the accessible
  source; the ledger's `x_s/H = 0.227` remains **unverified** and is reported as
  WITHHELD, not as a reference.
* The KB wiki does not state an explicit redistribution licence; the files are
  mirrored here for verification/reproducibility only, with full attribution,
  and can be re-fetched from the URLs above at any time.
