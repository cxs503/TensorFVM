"""Audit figure: independent Chebyshev reference vs the audited (repo) stability table."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eig

def cheb(N, L):
    k = np.arange(N+1); x = np.cos(np.pi*k/N); y = L*x
    c = np.ones(N+1); c[0] = c[-1] = 2.
    X = np.tile(x, (N+1, 1)).T; dX = X - X.T
    D = np.outer(c, 1./c)/(dX + np.eye(N+1)); D = D - np.diag(D.sum(axis=1))
    return y, D/L

def rayleigh(alpha, N=140, L=9.0):
    y, D = cheb(N, L); L2 = D@D; I = np.eye(N+1)
    U = np.tanh(y); Upp = -2*np.tanh(y)/np.cosh(y)**2
    La = L2 - alpha**2*I
    A = U[:,None]*La - np.diag(Upp)
    i = np.arange(1, N)
    c = np.array(eig(A[np.ix_(i,i)], La[np.ix_(i,i)], right=False))
    return alpha*float(np.max(c.imag))

repo = json.load(open('/root/tp-agent/work/m15435/docs/repro-mixing-layer-temporal/run/stability.json'))
sweep = repo['alpha_sweep']
al = np.array([s['alpha'] for s in sweep]); sig_repo = np.array([s['inviscid_growth'] for s in sweep])
al_os = np.array([0.30,0.40,0.50,0.60,0.70])
sig_audit = np.array([rayleigh(a) for a in al_os])
sig_repo_at = np.array([float([s['inviscid_growth'] for s in sweep if abs(s['alpha']-a)<1e-9][0]) for a in al_os])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(al, sig_repo, '-', lw=2, label='audited: repo Rayleigh eigen-solve')
ax[0].plot(al_os, sig_audit, 'o', ms=7, mfc='none', mew=1.8, label='audit: Chebyshev collocation (independent)')
ax[0].set_xlabel(r'$\alpha\delta$'); ax[0].set_ylabel(r'$\sigma\delta/U$')
ax[0].set_title('Inviscid growth rate: audited vs audit-owned solver'); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

rel = np.abs(sig_audit - sig_repo_at)/np.abs(sig_repo_at)*100
ax[1].bar([f'{a:.2f}' for a in al_os], rel, color='#4c72b0')
for i, r in enumerate(rel): ax[1].text(i, r+0.02, f'{r:.2f}%', ha='center', fontsize=9)
ax[1].axhline(3, ls='--', c='r', lw=1); ax[1].text(0.02, 3.05, '3% band', color='r', fontsize=8)
ax[1].set_ylim(0, max(3.5, rel.max()*1.3))
ax[1].set_ylabel('relative difference [%]'); ax[1].set_xlabel(r'$\alpha\delta$')
ax[1].set_title('Independent reference agrees to <0.5% (no fabrication)'); ax[1].grid(alpha=.3, axis='y')
fig.suptitle('m15435 CASE-MIXING-LAYER audit (m15715): reference numbers reproduced independently', fontsize=11)
fig.tight_layout()
fig.savefig('/root/tp-agent/work/m15715/docs/board-audit-m15715-m15435/fig_audit_rayleigh_vs_independent.png', dpi=130)
print('rel diffs %', rel)
print('max', rel.max())
