"""Independent Rayleigh/OS check for U=tanh(y) via Chebyshev collocation (author: audit).
Different discretisation than the repo's uniform-grid second-order FEM/FD:
Chebyshev spectral collocation on y in [-L,L], Dirichlet at walls,
linearised generalised eigenproblem (U L - U'')phi = c L phi, L = D2 - alpha^2.
No use of tensorfvm code."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.linalg import eig

def cheb(N, L):
    k = np.arange(N+1)
    x = np.cos(np.pi*k/N)          # [-1,1] Gauss-Lobatto
    y = L*x
    c = np.ones(N+1); c[0]=2.; c[-1]=2.
    X = np.tile(x,(N+1,1)).T
    dX = X - X.T
    D = np.outer(c,1./c)/(dX + np.eye(N+1))
    D = D - np.diag(D.sum(axis=1))
    D = D/L
    return y, D

def sig(alpha, Re=None, N=140, L=9.0):
    y,D = cheb(N,L)
    D2 = D@D
    U = np.tanh(y); Upp = -2*np.tanh(y)/np.cosh(y)**2
    i = np.arange(1,N)   # interior (phi=0 at walls)
    if Re is None:
        La = D2 - alpha**2*np.eye(N+1)
        A = U[:,None]*La - np.diag(Upp)
    else:
        La = D2 - alpha**2*np.eye(N+1)
        # OS: (U-c)Lphi - U''phi = (1/(i alpha Re)) L^2 phi
        A = U[:,None]*La - np.diag(Upp)
        B = La + (1.0/(1j*alpha*Re))*(La@La)
        c = eig(A[np.ix_(i,i)], B[np.ix_(i,i)], right=False)
        # temporal: omega = alpha*c ; growth = Im(omega) = alpha*Im(c)
        return alpha*max(cc.imag for cc in c), alpha*max(cc.imag for cc in c)
    B = La
    c = eig(A[np.ix_(i,i)], B[np.ix_(i,i)], right=False)
    return alpha*max(cc.imag for cc in c), max(c, key=lambda z: z.imag)

for a in (0.3,0.5,0.7):
    s,cm = sig(a)
    print('inviscid alpha=%.2f sigma=%.5f  c=%.5f%+.5fj' % (a,s,cm.real,cm.imag))
s,cm = sig(0.5, Re=500.0)
print('OS Re=500 alpha=0.5 sigma=%.5f c=%.5f%+.5fj' % (s,cm.real,cm.imag))
best=max(((a,)+sig(a) for a in np.arange(0.20,0.81,0.01)), key=lambda t:t[1])
print('inviscid max sigma=%.5f at alpha=%.2f  (amplification alpha*sigma=%.5f)'%(best[1],best[0],best[0]*best[1]))
bestA=max(((a,a*sig(a)[0]) for a in np.arange(0.20,0.81,0.01)), key=lambda t:t[1])
print('inviscid max alpha*sigma=%.5f at alpha=%.2f'%bestA)
