# fvm_sudden_expansion — planar sudden expansion, AR = 4

**Status: WITHHELD (no converged K within this session's budget).**

| item | value |
|---|---|
| geometry | inlet height D = 1, downstream height 4D, step face on the west boundary, 12D long |
| inlet | uniform over D (primary, consistent with the Borda-Carnot assumption); parabolic = sensitivity case |
| Reynolds | `Re_D = U_in D / nu = 100` |
| criterion | total-head loss `K` within 5 % of `(1 - 1/AR)^2 = 0.5625` |
| grids | D/10, D/20, D/40 |
| code | `src/tensorfvm/incompressible_baselines.py::build_sudden_expansion` |

Definition care (why the criterion needs a stated convention)
- Borda-Carnot assumes **uniform** velocity profiles and zero wall shear.  With a
  fully developed parabolic inlet the momentum balance gives `K ~ 0.46`, i.e. an
  18 % "error" that is a *definition difference*, not a numerical one.  The
  primary case therefore uses a uniform inlet, and both the total-head
  coefficient and the static-pressure recovery are reported, together with the
  identity check `K = recovery + (1 - (U_out/U_in)^2)`.
- The static pressure actually *recovers* across an expansion
  (`p_out > p_in`), so `(p_in - p_out)/(0.5 rho U^2)` is negative; the contract
  formula is only meaningful for the total head.  This is documented rather than
  silently reinterpreted.

No `K` value is claimed: the runs did not converge in this session (see
`docs/incompressible-baselines/VERIFICATION.md` §4.2 and §5).
