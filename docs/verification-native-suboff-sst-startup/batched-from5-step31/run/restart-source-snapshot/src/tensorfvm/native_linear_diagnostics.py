"""Original-system diagnostics; never relax acceptance tolerances or alter state."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, onenormest, splu


def linear_system_diagnostics(matrix, rhs, solution, *, estimate_condition=False):
    matrix = matrix.tocsc()
    rhs, solution = np.asarray(rhs), np.asarray(solution)
    absolute = abs(matrix)
    row_max = absolute.max(axis=1).toarray().ravel()
    column_max = absolute.max(axis=0).toarray().ravel()
    finite = np.isfinite(solution).all()
    clean = lambda value: float(value) if np.isfinite(value) else None
    def span(values):
        positive = values[values > 0]
        return dict(minimum_positive=clean(positive.min()) if len(positive) else None,
                    maximum=clean(values.max()), zero_count=int(np.sum(values == 0)))
    result = dict(size=matrix.shape[0], nonzeros=matrix.nnz,
                  row_maximum_span=span(row_max), column_maximum_span=span(column_max),
                  negative_diagonal_count=int(np.sum(matrix.diagonal() < 0)),
                  finite_solution=bool(finite), solution_max=None,
                  relative_rhs_residual=None, normwise_backward_error=None,
                  componentwise_backward_error=None)
    if finite:
        residual = matrix @ solution-rhs
        denominator = float(absolute.sum(axis=1).max())*np.linalg.norm(solution, np.inf)+np.linalg.norm(rhs, np.inf)
        component_scale = absolute @ np.abs(solution)+np.abs(rhs)
        component_error = np.divide(np.abs(residual), component_scale,
            out=np.where(np.abs(residual) == 0, 0., np.inf), where=component_scale > 0)
        result.update(solution_max=clean(np.max(np.abs(solution))),
                      relative_rhs_residual=clean(np.linalg.norm(residual)/max(np.linalg.norm(rhs), 1e-30)),
                      normwise_backward_error=clean(np.linalg.norm(residual, np.inf)/max(denominator, np.finfo(float).tiny)),
                      componentwise_backward_error=clean(np.max(component_error)))
    if estimate_condition:
        # A deterministic 1-norm estimate, not an exact condition number.
        factor = splu(matrix)
        inverse = LinearOperator(matrix.shape, dtype=float, matvec=factor.solve,
            rmatvec=lambda x: factor.solve(x, trans='T'), matmat=factor.solve,
            rmatmat=lambda x: factor.solve(x, trans='T'))
        result['condition_one_norm_estimate'] = clean(float(absolute.sum(axis=0).max())*onenormest(inverse, t=1))
    return result
