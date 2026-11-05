"""Curated mutation catalogue for the **3-D native** finite-volume core.

The pre-existing catalogue :mod:`tensorfvm.verification.mutation_catalogue`
covers only the 2-D production kernels (``body_fitted`` / ``solver``).  The 3-D
native path -- face/operator geometry, the compatible face-flux projection and
the collocated (Rhie--Chow type) pressure--velocity coupling -- was therefore
never subjected to a fault-injection audit, even though it carries the SUBOFF /
HLPW-style acceptance runs.

Each entry is one semantically meaningful defect with an exact ``find`` string
that must occur exactly once in the file, so source drift is detected instead
of mutating the wrong line.  ``expected`` is the prediction recorded *before*
seeing the result; an unexpected survivor is reported as-is.

``scripts/audit_mutation_survival3d.py`` consumes this catalogue.
"""

CATALOGUE3D = [
    dict(
        id="CV3-PFD-SIGN",
        file="src/tensorfvm/native_collocated3d.py",
        description="Compatible face-flux defect gets the wrong sign "
        "(interpolated gradient flux plus compact flux)",
        find="self.pressure_flux_defect=(self.interpolated_gradient_flux-self.compact_flux).tocsr()",
        replace="self.pressure_flux_defect=(self.interpolated_gradient_flux+self.compact_flux).tocsr()",
        tests=["tests/test_native_collocated3d.py"],
        rationale=(
            "The defect operator is what keeps the collocated pressure and the "
            "face flux compatible; reversing it turns the damping into an "
            "anti-damping, so a steady solve must lose convergence or move the "
            "pressure field."
        ),
        expected="killed",
    ),
    dict(
        id="CV3-PFD-DROP",
        file="src/tensorfvm/native_collocated3d.py",
        description="Compatible face-flux defect deleted: the compact two-point "
        "gradient is used as if it were a face flux",
        find="self.pressure_flux_defect=(self.interpolated_gradient_flux-self.compact_flux).tocsr()",
        replace="self.pressure_flux_defect=(-self.compact_flux).tocsr()",
        tests=["tests/test_native_collocated3d.py"],
        rationale=(
            "Deleting the interpolated-gradient part leaves an inconsistent "
            "face flux, i.e. exactly the checkerboard-prone operator this "
            "module exists to replace; the affine/consistency tests should "
            "notice."
        ),
        expected="killed",
    ),
    dict(
        id="CV3-SKEW-DROP",
        file="src/tensorfvm/native_collocated3d.py",
        description="Face skewness correction silently disabled regardless of the "
        "caller's flag",
        find="if not skew_correction:self.skew[:]=0.",
        replace="self.skew[:]=0.",
        tests=["tests/test_native_collocated3d.py"],
        rationale=(
            "On skewed native meshes the skew term is the second-order part of "
            "the face reconstruction.  If no test uses a skewed mesh with the "
            "flag on, the defect survives and the suite only certifies "
            "orthogonal behaviour."
        ),
        expected="killed",
    ),
    dict(
        id="OP3-RECON-SWAP",
        file="src/tensorfvm/native_operators3d.py",
        description="Owner/neighbour interpolation weights swapped in the "
        "affine-exact face reconstruction",
        find="result[i]=(1-weight)*v[m.owner[i]]+weight*v[m.neighbor[i]]",
        replace="result[i]=weight*v[m.owner[i]]+(1-weight)*v[m.neighbor[i]]",
        tests=["tests/test_native_mesh3d.py"],
        rationale=(
            "With the weights swapped the geometric weight (which is not 1/2 on "
            "a graded mesh) inverts; any test that checks interpolation on a "
            "non-uniform mesh or affine reproduction must fail."
        ),
        note=(
            "首次审计时该变异体 survived：被指向的 tests/test_native_mesh3d.py 只用单胞网格，"
            "没有内部面，权重路径从未被走到。已补 "
            "test_face_reconstruction_reduces_to_geometric_projection_on_nonuniform_pair 把存活项关闭。"
        ),
        expected="killed",
    ),
    dict(
        id="OP3-GAUSS-VOLUME",
        file="src/tensorfvm/native_operators3d.py",
        description="Gauss gradient returned as an integrated flux (cell volume "
        "division dropped)",
        find="return self.integrate(np.asarray(face_values)[:,None]*self.mesh.area_vectors)/self.mesh.cell_volumes[:,None]",
        replace="return self.integrate(np.asarray(face_values)[:,None]*self.mesh.area_vectors)",
        tests=["tests/test_native_mesh3d.py"],
        rationale=(
            "The gradient is a per-volume quantity; dropping the division "
            "scales it by the cell volume.  Only a test that compares against an "
            "analytic affine gradient on cells of different volume can see it."
        ),
        expected="killed",
    ),
]
