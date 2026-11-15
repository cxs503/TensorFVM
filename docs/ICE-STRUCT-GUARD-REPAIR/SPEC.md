# Ice-domain guardrail specification (JOUR-ICE-STRUCT-GUARD, rev. m22342)

## 1. What is guarded

The 12 quantities a level-ice resistance calculation consumes: 8 ice
properties (the m22209 delivery) plus 4 hull variables that were previously
unguarded even though the load formula multiplies them in.  Registry:
`tensorfvm.ice.parameters.ALL_ICE_PARAMETERS` (12 entries,
`ICE_RESISTANCE_INPUTS`).

## 2. Decision semantics

| class | meaning | `validate` verdict | `clamp` behaviour |
|---|---|---|---|
| inside typical band | normal operation | `pass` | unchanged |
| inside hard, outside soft | outside the calibration domain of the empirical models | `fail` (or `warn` when `allow_extrapolation=True`) | unchanged |
| outside hard bounds | outside the validity domain of the models / physically impossible | `fail` | clamped to the nearest hard bound, warning recorded |
| non-finite (`nan`, `±inf`) | not a physical measurement | `fail`, never in `validated_params` | `±inf` → bound; `nan` → rejected, never echoed back |

`require_ice_parameters` / `require_guarded_level_ice_resistance` turn the
verdict into a raised `IceParameterDomainError`, so a caller cannot forget the
`if`.

## 3. Bounds

| parameter | symbol | unit | hard min | hard max | soft min | soft max | typical | reference |
|---|---|---|---|---|---|---|---|---|
| `ice_thickness` | h | m | 0.01 | 10 | 0.05 | 5 | 0.3–2.5 | ISO 19906:2019 Annex C; Timco & Frederking (2012) |
| `ice_strength` | σ_c | MPa | 0.1 | 30 | 0.3 | 15 | 1–5 | Timco & Weeks (2010); ISO 19906:2019 |
| `ice_friction` | μ | — | 0 | 1 | 0.01 | 0.4 | 0.05–0.2 | Riska et al. (1997) VTT Publ. 319; ITTC Ice Committee (2017) |
| `ice_speed` | v | m/s | 0 | 15 | 0.1 | 8 | 0.5–5 | ITTC Ice Committee (2017); Riska et al. (1997) |
| `ice_density` | ρ_ice | kg/m³ | 800 | 920 | 850 | 918 | 880–910 | Timco & Frederking (2012); ISO 19906:2019 |
| `ice_elastic_modulus` | E | GPa | 0.1 | 20 | 1 | 12 | 3–8 | Timco & Weeks (2010); Schulson & Duval (2009) |
| `ice_poisson_ratio` | ν | — | 0 | 0.5 | 0.25 | 0.4 | 0.3–0.35 | Schulson & Duval (2009) |
| `ice_temperature` | T | °C | -60 | 0 | -45 | -1 | -30–-5 | Timco & Frederking (2012); WMO Sea Ice Nomenclature |
| `ship_length` | L | m | 10 | 500 | 30 | 400 | 50–350 | Lewis (1988) Principles of Naval Architecture; Riska et al. (1997) |
| `ship_beam` | B | m | 2 | 80 | 8 | 60 | 15–50 | Lewis (1988) Principles of Naval Architecture |
| `ship_draft` | T | m | 0.3 | 20 | 2 | 16 | 5–14 | Lewis (1988); Riska et al. (1997) |
| `ship_block_coefficient` | C_B | — | 0.3 | 0.9 | 0.45 | 0.88 | 0.55–0.85 | Lewis (1988) Principles of Naval Architecture |

The eight ice rows are the m22209 table verbatim
(`docs/FVM_ICE_GUARDRAILS/REPORT.md` §2.2).  The hull rows are declared here
for the first time; they are deliberately wide (an applicability envelope, not
a hull-design checker) and are documented as such in `parameters.py`.

## 4. Known scope limits of the specification

* Bounds are static; temperature- or salinity-dependent bounds and
  correlation-aware checks (thick ice with weak strength) are **not** modelled.
* `bow_angle_deg` is a shape descriptor, not part of the 12-parameter set, and
  is passed explicitly to the load entry point (default 30°, Lindqvist's
  nominal).
* Hard bounds state the *applicability limit of ship–ice models*, not physical
  impossibility: `h = 10 m` is the upper end of level-ice formulations, while
  ridge keels and glacier tongues are far thicker.
