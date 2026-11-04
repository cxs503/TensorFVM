"""Curated mutation catalogue for the production 2-D finite-volume kernels.

Each entry is a single, semantically meaningful defect that a verification test
suite is *supposed* to catch: a sign flip in the Rhie--Chow damping, deletion of
the non-orthogonal cross term, a swapped interpolation weight, or replacement of
central face averaging by first-order upwind.  The catalogue is intentionally
textual and exact (``find`` must occur exactly once per file) so that drift of
the production source is detected instead of silently mutating the wrong line.

``expected`` records the prediction made before running the sweep.  A
prediction that fails is reported as-is in the audit; nothing is rewritten to
make the record look better.  ``scripts/audit_mutation_survival.py`` consumes
this catalogue, and ``tests/test_mutation_audit_contract.py`` locks it against
source drift.
"""

CATALOGUE = [
    dict(
        id="BF-RC-SIGN",
        file="src/tensorfvm/body_fitted.py",
        description="Rhie--Chow damping enters the face flux with the wrong sign",
        find="                                      - df * normal_pressure)",
        replace="                                      + df * normal_pressure)",
        tests=["tests/test_body_fitted.py", "tests/test_mutation_coverage.py"],
        rationale=(
            "The damping term is what prevents checkerboard pressure/velocity "
            "decoupling on collocated grids; flipping it must move the steady "
            "solution or destroy convergence."
        ),
        expected="killed",
    ),
    dict(
        id="BF-NONORTHO-DROP",
        file="src/tensorfvm/body_fitted.py",
        description="Non-orthogonal cross term dropped from the face normal pressure",
        find="        normal_pressure = self.k * dp + (self.T * self._interpolate(gp)).sum(-1)",
        replace="        normal_pressure = self.k * dp",
        tests=["tests/test_body_fitted.py"],
        rationale=(
            "On skewed body-fitted cells the projected-gradient correction "
            "carries the tangential part of the pressure gradient into the face "
            "flux.  If no test varies skewness, the deletion survives and the "
            "suite only certifies orthogonal-grid behaviour."
        ),
        expected="killed",
    ),
    dict(
        id="BF-DAMP-RATIO",
        file="src/tensorfvm/body_fitted.py",
        description="Pseudo-time damping ratio inverted (steady_df/df instead of df/steady_df)",
        find="            beta = (1 - df / steady_df).clamp(0, 1)",
        replace="            beta = (1 - steady_df / df).clamp(0, 1)",
        tests=["tests/test_body_fitted.py"],
        rationale=(
            "beta blends the retained face-flux memory with the unrelaxed "
            "operator; inverting the ratio selects the complementary blend, so "
            "the pseudo-time and velocity-relaxation equivalence tests must fail."
        ),
        expected="killed",
    ),
    dict(
        id="CP-WEIGHT-SWAP",
        file="src/tensorfvm/conservative_pressure.py",
        description="Projected-linear face weights applied to the wrong owner/neighbour",
        find="        values[self.f]=weights.reshape(shape)*field[self.oi]+(1-weights).reshape(shape)*field[self.ni]",
        replace="        values[self.f]=(1-weights).reshape(shape)*field[self.oi]+weights.reshape(shape)*field[self.ni]",
        tests=["tests/test_multi_element_flow.py"],
        rationale=(
            "Swapping the bracket keeps the weights admissible, so the existing "
            "bracketing guard cannot catch it; only a test that compares the "
            "interpolated face value against the exact linear field on a "
            "stretched grid can."
        ),
        expected="killed",
    ),
    dict(
        id="CP-OUTLET-DROP",
        file="src/tensorfvm/conservative_pressure.py",
        description="Outlet faces no longer zeroed in the Gauss pressure gradient",
        find="        face[self.mesh.masks['outlet']]=0",
        replace="        face[self.mesh.masks['outlet']]=face[self.mesh.masks['outlet']]",
        tests=["tests/test_multi_element_flow.py"],
        rationale=(
            "The gauge of a pressure gradient with a prescribed outlet pressure "
            "depends on this masking; a suite that only checks force balance can "
            "let it through."
        ),
        expected="killed",
    ),
    dict(
        id="STAG-UPWIND",
        file="src/tensorfvm/solver.py",
        description="Staggered u-face interpolation replaced by first-order upwind",
        find="        return ((0.5 * (self.u[:, :-1] + self.u[:, 1:])).masked_fill(~self.fluid, 0),",
        replace="        return ((self.u[:, :-1]).masked_fill(~self.fluid, 0),",
        tests=["tests/test_solver.py"],
        rationale=(
            "Upwinding biases every face value, so the channel-profile "
            "refinement test and the steady-state equivalence tests should react; "
            "if tolerances are loose enough it survives."
        ),
        expected="killed",
    ),
]