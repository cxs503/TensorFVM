# CASE-FVM-CHANNEL-KMM180 — reference band at Re_tau ~ 180

Anchor: `CASE-FVM-CHANNEL-KMM180` (u+(y+) for y+ > 30 against the Kim, Moin & Moser 1987 table).

## 1. What changed

The KMM1987 table itself is still WITHHELD (no machine-readable copy in-tree). This run replaces the unquantified single-proxy substitution with a **measured reference band** made of three published Re_tau ~ 180 channel DNS:

| key | dataset | Re_tau | role |
| --- | --- | --- | --- |
| `mkm` | MKM1999 chan180 | 178.120 | same code as KMM1987 |
| `hj` | HJ2006 Re180 | 180.000 | independent group |
| `lm` | LM2015 Re182 | 182.088 | pre-existing proxy |

## 2. Method

* graded window: `30.00 <= y+ <= 36.00` (`30` to `0.2 Re_tau`, the only interval that is logarithmic at this Reynolds number);
* model profiles: the committed per-cell profiles of the anchor (`docs/case-fvm-channel-kmm180/profiles.csv`, sha256-pinned by that case's manifest) -- no solver re-run, so this audit adds reference information only;
* acceptance: worst-case RMS relative deviation over the three references `< 5 %` (strict version of the anchor criterion, which names one reference).

## 3. Reference-substitution uncertainty

Window RMS relative half-width of the envelope: **0.18 %**, maximum 0.18 % over 6 grid points.

| pair | RMS difference [%] | max difference [%] |
| --- | --- | --- |
| `mkm_vs_hj` | 0.35 | 0.36 |
| `mkm_vs_lm` | 0.07 | 0.09 |
| `hj_vs_lm` | 0.29 | 0.30 |

In this window the three datasets agree within the numbers above; the differences are far below the 5 % tolerance, which is what licenses grading the second decimal of the model error against any of them.

## 4. Case results (worst case over the band)

| case | RMS vs mkm [%] | RMS vs hj [%] | RMS vs lm [%] | worst RMS [%] | inside band | verdict | robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kepsilon_n128` | 6.37 | 6.04 | 6.31 | 6.37 | 0.00 | FAIL | yes |
| `kepsilon_n256` | 6.37 | 6.04 | 6.31 | 6.37 | 0.00 | FAIL | yes |
| `kepsilon_n64` | 6.37 | 6.04 | 6.31 | 6.37 | 0.00 | FAIL | yes |
| `komega_n128` | 13.54 | 13.23 | 13.48 | 13.54 | 0.00 | FAIL | yes |
| `komega_n256` | 13.51 | 13.21 | 13.45 | 13.51 | 0.00 | FAIL | yes |
| `komega_n64` | 13.51 | 13.20 | 13.46 | 13.51 | 0.00 | FAIL | yes |
| `sst_n128` | 5.72 | 5.39 | 5.66 | 5.72 | 0.00 | FAIL | yes |
| `sst_n256` | 6.17 | 5.84 | 6.11 | 6.17 | 0.00 | FAIL | yes |
| `sst_n64` | 4.67 | 4.33 | 4.61 | 4.67 | 0.00 | PASS | yes |

`inside band` = fraction of graded cells whose `u+` lies inside the envelope; `robust` = no reference flips the verdict.

## 5. Caveats

* The KMM1987 table is still not in the repository; no claim is made that the band equals it. `mkm1999_0178` is the closest published companion (same numerical method, two shared authors) and `hj2006_0180` sits at Re_tau = 180 exactly, so together with `lm2015_0182` they bracket the nominal Reynolds number instead of guessing it.
* The envelope is a *reference uncertainty*, not a model error: the graded RMS numbers keep the same definition as the anchor artifacts.
* Model verdicts are inherited from the committed solver profiles; this audit does not re-run the solver, so it cannot detect a stale profile (the anchor manifest sha256 does that).

