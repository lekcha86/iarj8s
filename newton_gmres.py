"""
Newton-GMRES: plain (order-2) Newton's method with each linear system solved
INEXACTLY via GMRES+ILU, using an Eisenstat-Walker-style adaptive forcing
term. This is a distinct comparator from I-ARJ8S-GMRES: it tests what
happens when you put an inexact Krylov solver under classical Newton, with
NO high-order structure at all, matching the base ARJ8S paper's own
"Newton-Krylov (ILUTP)" row in their Table 13 -- a baseline we had not
included until now (we had only tested high-order methods, ARJ8S and
I-ARJ8S, with GMRES, never plain Newton with GMRES as its own comparator).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def newton_gmres(F, J, x0, tol=1e-9, max_iter=30, eta_max=0.1, restart=50, gmres_maxiter=200):
    """Newton's method, each step solved inexactly via GMRES+ILU with an
    Eisenstat-Walker Choice-1-style adaptive forcing term:
        eta_k = min(eta_max, ||F(x_k)|| / ||F(x_{k-1})||)
    (eta_0 = eta_max on the first step, matching the base paper's eta0=0.1)."""
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
        precond = spla.LinearOperator(Jx.shape, ilu.solve)
        s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                              restart=restart, maxiter=gmres_maxiter)
        x_new = x + s
        resid_new = np.linalg.norm(F(x_new))
        resid_old = hist[-1]
        x = x_new
        hist.append(resid_new)
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if resid_new < tol:
            return x, hist, 'converged'
        eta = min(eta_max, resid_new / resid_old) if resid_old > 0 else eta_max
    return x, hist, 'maxiter'


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from bratu_test import build_bratu
    from bratu3d_test import build_bratu_3d, newton_sparse

    print("=== 2D Bratu: Newton-GMRES vs direct (N=255, 65,025 unknowns) ===")
    F, J, d = build_bratu(255, 3.51)
    x0 = np.zeros(d)
    t0 = time.time()
    x, hist, status = newton_gmres(F, J, x0)
    t_gmres = time.time() - t0
    print(f"Newton-GMRES: {status}, iters={len(hist)-1}, time={t_gmres:.2f}s, resid={hist[-1]:.3e}")

    print("\n=== 3D Bratu: Newton-GMRES vs direct (N=20, 8,000 unknowns) ===")
    F3, J3, d3 = build_bratu_3d(20, 3.51)
    x0_3 = np.zeros(d3)
    t0 = time.time()
    xN, histN = newton_sparse(F3, J3, x0_3)
    t_direct = time.time() - t0
    print(f"Newton (direct): iters={len(histN)-1}, time={t_direct:.2f}s, resid={histN[-1]:.3e}")
    t0 = time.time()
    x3, hist3, status3 = newton_gmres(F3, J3, x0_3)
    t_gmres3 = time.time() - t0
    print(f"Newton-GMRES:    {status3}, iters={len(hist3)-1}, time={t_gmres3:.2f}s, resid={hist3[-1]:.3e}")
