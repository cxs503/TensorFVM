# Provenance — ERCOFTAC UFR 3-30 Re = 700 reference data (CASE-FVM-PERIODIC-HILLS)

## What these files are

`UFR3-30_C_700_data_MB-001.dat` … `-010.dat` are the mean/statistics profiles
distributed by **Prof. Michael Breuer (Helmut-Schmidt-Universität Hamburg,
"Professur fuer Stroemungsmechanik")**, author of the reference paper of this
case, for **case 1 = Re = 700 DNS, code LESOCC**:

> Breuer, M., Peller, N., Rapp, Ch., Manhart, M. (2009): *Flow over Periodic
> Hills — Numerical and Experimental Study in a Wide Range of Reynolds
> Numbers*, Computers & Fluids 38(2):433–457,
> doi:10.1016/j.compfluid.2008.05.002

Each file carries a 7-column header `y/h, U/U_b, V/U_b, uu/U_b^2, vv/U_b^2,
uv/U_b^2, k/U_b^2` and is a full wall-normal profile from the hill surface to
the channel top (`y/h` up to `3.034`).  The streamwise stations are the 10
canonical UFR 3-30 positions `x/h = 0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8` (in
file order) — the same station set as the Re = 2800 set in
`../ercoftac_ufr330/`.

## Retrieval

* Source page: `https://kbwiki.ercoftac.org/w/index.php?title=UFR_3-30_Test_Case`
  (§ *CFD data available → Curvilinear grids → Case 1  DNS (LESOCC) Re = 700*).
* Retrieval (round 32, 2026-11-16): file URLs resolved through the wiki's
  `api.php` (`prop=imageinfo&iiprop=url`), then fetched verbatim;
  **unmodified**.
* The reattachment length used as the acceptance target at Re = 700
  (`x_R/h = 5.24`) is quoted from
  `https://kbwiki.ercoftac.org/w/index.php?title=UFR_3-30_Evaluation`
  (§ *Separation and Reattachment Lengths*), a page written by the case's own
  authors (Rapp, Breuer, Manhart, Peller):

  > "The corresponding values of the reattachment lengths are
  > x_R/h = 5.24, 5.19, 5.41, 5.09, and 4.69 for Re = 700 to 10,595,
  > respectively."

  The value is single-sourced in
  `tensorfvm.verification.periodic_hill_lower_re.KB_REATTACHMENT_BY_RE`;
  this directory adds the *profile* data, not a second copy of the number.

## Geometry admission (measured from the files)

At `x/h = 0.05, 0.5, 1` the profile starts at `y/h >= 0.45` (inside the hill
body) and at `x/h = 8` at `y/h = 0.453` (the hill surface rising again), so
`y/h = 0.5` is in the fluid only at `x/h = 2 .. 7`.  The loader's
`station_covers_y05` makes this exclusion explicit, mirroring the Re = 2800
module.
