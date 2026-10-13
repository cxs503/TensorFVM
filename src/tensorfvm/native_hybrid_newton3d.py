"""Damped coupled Newton solver for the existing native hybrid NS residual.

A separate diagnostic/reference solver; it is not SIMPLE/SIMPLEC. Neither the
continuity constraint nor any physical force/flux is changed. Trial residuals
are rebuilt without mutating the accepted state, including upwind switches.
"""
import numpy as np
from scipy import sparse


def solve_newton(solver, *, max_iterations=50, tolerance=1e-8, max_backtracks=20, callback=None, allow_inexact_linear_steps=False):
    s=solver
    s.velocity_constraint_tolerance=tolerance
    if not np.isfinite(tolerance) or tolerance<=0 or max_iterations<1 or max_backtracks<0:
        raise ValueError('positive tolerance/iteration limit and nonnegative backtrack limit required')
    space=s.viscous.velocity_space;size=len(space.free);volumes=s.mesh.cell_volumes
    momentum_scale=max(np.linalg.norm(space.restrict(s.load-s.viscous.full_matrix@s.prescribed)),1.)
    continuity_scale=max(np.linalg.norm(s.divergence@s.state/volumes),1.)
    s.nonlinear_converged=False;s.newton_history=[];s.newton_failure=None

    def evaluate(state,pressure):
        full,source=s._current_system(state)
        momentum=space.restrict(full@state+source-s.divergence.T@pressure-s.load)
        mass=s.divergence@state
        gauge=(pressure[0]-s.gauge_value) if s.gauge else 0.
        merit=float(np.linalg.norm(np.r_[momentum/momentum_scale,mass/volumes/continuity_scale,gauge]))
        scale=max(np.linalg.norm(space.restrict(s.load-source-full@s.prescribed)),1.)
        row=dict(current_momentum_residual=float(np.linalg.norm(momentum)/scale),
                 divergence_max=float(np.max(np.abs(mass/volumes))),gauge_error=abs(gauge),
                 boundary_velocity_error=space.constraint_error(state-s.prescribed))
        return momentum,mass,gauge,merit,row

    current=evaluate(s.state,s.p)
    for iteration in range(max_iterations):
        momentum,mass,gauge,merit,metrics=current
        if iteration==0 and max(metrics.values())<tolerance:
            s.nonlinear_converged=True
            s.newton_history.append(dict(iteration=0,accepted=True,step_length=0.,backtracks=0,merit_before=merit,
                merit_after=merit,velocity_change=0.,pressure_change=0.,**metrics))
            break
        jac=s.viscous.full_matrix+s.convection.hybrid_jacobian(s.state,s.face_mean.matrix)
        matrix=sparse.bmat([[space.matrix(jac),-s.B.T],[s.B,None]],format='lil')
        rhs=-np.r_[momentum,mass]
        if s.gauge:
            matrix[size,:]=0.;matrix[size,size]=1.;rhs[size]=-gauge
        options={"allow_inexact": True} if allow_inexact_linear_steps else {}
        direction,linear_residual=s._coupled_linear(matrix.tocsc(),rhs,**options)
        old=s.state.copy();old_p=s.p.copy();accepted=False;trials=[];du=dp=float('inf')
        for backtracks in range(max_backtracks+1):
            step=2.**(-backtracks)
            trial=old+space.expand(step*direction[:size])
            trial_p=old_p+step*direction[size:]
            if not np.isfinite(trial).all() or not np.isfinite(trial_p).all():
                trials.append(dict(step_length=step,merit=None,accepted=False,reason='nonfinite_trial'))
                continue
            candidate=evaluate(trial,trial_p)
            du=float(np.linalg.norm(trial-old)/max(np.linalg.norm(trial),1.))
            dp=float(np.linalg.norm(trial_p-old_p)/max(np.linalg.norm(trial_p),1.))
            already_solved=max(candidate[4].values())<tolerance and max(du,dp)<tolerance
            decreased=np.isfinite(candidate[3]) and candidate[3]<=(1-1e-4*step)*merit
            trials.append(dict(step_length=step,merit=candidate[3],accepted=bool(decreased or already_solved)))
            if decreased or already_solved:
                s.state,s.p=trial,trial_p;current=candidate;accepted=True
                break
        row=dict(iteration=iteration+1,accepted=accepted,step_length=step if accepted else 0.,
            backtracks=backtracks,merit_before=merit,merit_after=current[3],
            true_linear_relative_residual=linear_residual,
            velocity_change=du if accepted else 0.,pressure_change=dp if accepted else 0.,
            trials=trials,**current[4])
        s.newton_history.append(row)
        if callback:callback(row)
        if not accepted:
            s.newton_failure='line_search_failed';break
        s.nonlinear_converged=bool(max(current[4].values())<tolerance and max(du,dp)<tolerance)
        if s.nonlinear_converged:break
    if not s.nonlinear_converged and s.newton_failure is None:s.newton_failure='iteration_limit'
    result=s.result()
    result.update(newton_history=list(s.newton_history),newton_failure=s.newton_failure,
                  nonlinear_method='coupled_newton',
                  allow_inexact_linear_steps=bool(allow_inexact_linear_steps))
    return result
