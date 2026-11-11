# Provenance of the initial datum (zero hand-copy trail)

Case: **two interacting blast waves** (Woodward & Colella 1984, *J. Comput. Phys.*
**54**, 115-173, doi:10.1016/0021-9991(84)90142-6).

The 1984 article itself is **closed access** (OpenAlex `oa_status = closed`,
`any_repository_has_fulltext = false`, checked 2026-11-11 via
`https://api.openalex.org/works/doi:10.1016/0021-9991(84)90142-6`), so its own
tables could not be read. The initial datum is therefore taken from two
independent open-access restatements that both cite the 1984 paper and agree
digit for digit; each was fetched programmatically and the quoted sentence is
reproduced verbatim below (digits, not prose, are what the code uses).

| source | where | quoted text | retrieval |
|---|---|---|---|
| arXiv:1902.07773 (Dumbser et al., *J. Comput. Phys.* 2019), Sec. 5.1.4 "Woodward-Colella Problem" | `https://ar5iv.labs.arxiv.org/html/1902.07773` | "The initial data is the following: (ρ0,u0,p0) = {[1,0,10³], 0 ≤ x ≤ 0.1; [1,0,10⁻²], 0.1 < x < 0.9; [1,0,10²], 0.9 ≤ x ≤ 1}" … "the expected solution has been computed via the pure RPJ on a mesh of 3200 cells" … "T = 0.038" | 2026-11-11 |
| arXiv:2109.14340 (Fu et al.), Eq. (37), "Interacting blast waves" | `https://ar5iv.labs.arxiv.org/html/2109.14340` | "(ρ,u,p) = {(1,0,1000), if 0 ≤ x < 0.1; (1,0,0.01), if 0.1 ≤ x < 0.9; (1,0,100), if 0.9 ≤ x ≤ 1}" … "A reflective boundary condition is imposed at x = 0 and x = 1" … "until the final time t = 0.038" | 2026-11-11 |

Both restatements cite Woodward & Colella (1984). Crossref resolves the DOI to
the title "The numerical simulation of two-dimensional fluid flow with strong
shocks", *J. Comput. Phys.* 54 (1984) 115-173
(`https://api.crossref.org/works/10.1016/0021-9991(84)90142-6`).

The digit strings enter the code once, as named constants in
`src/tensorfvm/compressible/cases.py`
(`WC_BLAST_X1 = 0.1`, `WC_BLAST_X2 = 0.9`, `WC_BLAST_P_LEFT = 1000.0`,
`WC_BLAST_P_MIDDLE = 0.01`, `WC_BLAST_P_RIGHT = 100.0`, `WC_BLAST_RHO = 1.0`,
`WC_BLAST_T_END = 0.038`), and every number downstream (initial energy
`275.02`, exact Riemann front speeds) is computed from them, not typed.

## What could not be sourced

The 1984 paper's *published solution values* (its density profiles/contours at
`t = 0.038`) were not obtainable: the article is closed access, no open
repository copy exists, and the platform knowledge engine
(`/api/v1/tensor/knowledge/search`, backend `127.0.0.1:8021`) answered
`engine_offline` on 2026-11-11. The case is therefore verified against four
independent *computable* anchors (exact conservation, exact Riemann front
speeds and star states, a fine-grid reference solution, and the exact
y-independence of an extruded slab), and the comparison to the paper's own
printed numbers is declared **WITHHELD** rather than imitated from memory. This
is recorded as an ecosystem friction note in the report.