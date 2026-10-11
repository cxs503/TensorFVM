# Supplemental audit: independent drag and Cp cases

Checked 2026-10-11. This supplements the frozen evidence; it does not change existing files. Drag and Cp may use **two separately documented operating conditions**. Sharing Reynolds number is not an acceptance requirement.

## Drag case: independent published reproduction conditions now available

Ni, Chen & Wu, [Oxford Open Energy, DOI10.1093/ooenergy/oiaf014](https://academic.oup.com/ooenergy/article/doi/10.1093/ooenergy/oiaf014/8405192), published December 2025, volume 2026, explicitly sets density 998.2 kg/m³, dynamic viscosity 1.002e−3 Pa·s, inlet 3.046 m/s, SST k-omega and no-slip surfaces. It compares AFF-1 computed drag 88.71 N to experimental 87.4 N. Its force coefficient uses maximum cross-sectional area. These are stated numerical reproduction conditions; original DTMB temperature, density, viscosity and force uncertainty remain unverified. The experiment-force comparison can be reproduced transparently under this literature setup without requiring Cp at this Re.

Using L=4.356 m and D=.508 m gives nu=1.003806852e−6 m²/s, Re_L=1.321805681e7 and frontal area=.2026829916 m². Normalizing measured 87.4 N with those explicitly stated computational properties yields Cd=.093120813. This value is **derived**, not a newly discovered tabulated experimental coefficient. Use identical normalization for calculated and measured forces; report their relative error and retain the original force alongside it. A wetted-area Ct requires declaring its distinct area convention. Do not compare the frontal Cd to Budak's wetted Ct directly.

## Budak evidence

Budak 2023 retains a directly published EFD Ct=.003161 at 5.92 kn, wetted area5.98 m² and dynamic-pressure denominator. No density, viscosity or first-case Re were located anywhere in its original full paper. Budak & Beji 2016, original author paper text [available here](https://www.researchgate.net/publication/305308713_Computational_Resistance_Analyses_of_A_Generic_Submarine_Hull_Form_and_Its_Geometric_Variants), p83 explicitly identifies freshwater density998.2 kg/m³ in its own coefficient normalization. Its viscosity/first-case Re have not been established. This does not authorize filling missing experimental material data or importing properties into the 2023 table by assumption. No density was inferred from Ct.

## Cp primary scan and sensitivity

Targeted NASA NTRS and NAP searches did not identify original Huang Table1 p333. DTIC citations/PDF and NTIS endpoints were inaccessible. ONR lists the 1992 19th symposium but does not link a usable scan. Thus the three candidate CSV correction inconsistencies are still unverified against the primary table and remain unchanged.

The norm of the difference between the listed corrected column and measured-minus-empty-tunnel correction is .000360567608 in Cp; divided by the listed curve norm it is .000659789 = **.06598%**. This quantifies transcription/correction sensitivity only. It does not establish which column is physically authoritative. A provisional benchmark can show errors against both unchanged candidates and label its reference status honestly; the original ±.015 Cp measurement uncertainty remains separate and much larger. No provisional result should be labeled exact primary-data validation.

The published wind-tunnel Re_L=1.2e7 for Cp remains distinct from the drag reproduction Re above. These independent conditions are compatible with the user's separate drag/Cp acceptance goals.
