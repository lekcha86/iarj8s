"""
2D Bratu problem: -Delta u - lambda*e^u = 0 on (0,1)^2, u=0 on boundary.
Same problem family as the base ARJ8S paper's Section 8, built independently
here with scipy sparse matrices, to genuinely test the GMRES/Krylov extension
of I-ARJ8S on a large, moderately ill-conditioned sparse system.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from iarj8s_inertial import iarj8s


def build_bratu(N, lam):
    """N interior points per side -> d = N*N unknowns."""
    h = 1.0 / (N + 1)
    e = np.ones(N)
    T = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format='csr')
    I = sp.identity(N, format='csr')
    L = (sp.kron(I, T) + sp.kron(T, I)) / h**2  # discrete Laplacian, d x d
    L = L.tocsr()
    d = N * N

    def F(u):
        return L @ u + lam * np.exp(u)

    def J(u):
        Jm = L + sp.diags(lam * np.exp(u))
        return Jm.tocsc()

    return F, J, d


def newton_sparse(F, J, x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    hist_resid = [np.linalg.norm(F(x))]
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        s = spla.spsolve(Jx, -Fx)
        x = x + s
        hist_resid.append(np.linalg.norm(F(x)))
        if hist_resid[-1] < tol:
            break
    return x, hist_resid


def arj8s_sparse(F, J, x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    hist_resid = [np.linalg.norm(F(x))]
    for k in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        s1 = spla.spsolve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        alpha = 1.0 - 2.0 * (Fk @ Fz1) / (Fk @ Fk)
        z_prev, Fz_prev = z1, Fz1
        for _ in range(3):
            s_i = spla.spsolve(alpha * Jk, -Fz_prev)
            z_i = z_prev + s_i
            Fz_i = F(z_i)
            z_prev, Fz_prev = z_i, Fz_i
        x = z_prev
        hist_resid.append(np.linalg.norm(Fz_prev))
        if hist_resid[-1] < tol:
            break
    return x, hist_resid


if __name__ == "__main__":
    N = 63  # 63x63 = 3969 unknowns
    lam = 3.51  # regular regime, same value as base paper's Table 13/15
    F, J, d = build_bratu(N, lam)
    print(f"Bratu problem: {d} unknowns, lambda={lam}")

    x0 = np.zeros(d)

    t0 = time.time()
    xN, histN = newton_sparse(F, J, x0)
    tN = time.time() - t0
    print(f"\nNewton (direct sparse LU): iters={len(histN)-1}, time={tN:.3f}s, final resid={histN[-1]:.3e}")

    t0 = time.time()
    xA, histA = arj8s_sparse(F, J, x0)
    tA = time.time() - t0
    print(f"ARJ8S (direct sparse LU):  iters={len(histA)-1}, time={tA:.3f}s, final resid={histA[-1]:.3e}")

    t0 = time.time()
    xI, histI = iarj8s(F, J, x0, tol=1e-10, max_iter=50, use_gmres=True, gmres_tol_base=0.1)
    tI = time.time() - t0
    residI = np.linalg.norm(F(xI))
    print(f"I-ARJ8S (GMRES+ILU):      iters={len(histI)-1}, time={tI:.3f}s, final resid={residI:.3e}")

    print(f"\nAgreement check: ||x_Newton - x_ARJ8S|| = {np.linalg.norm(xN - xA):.3e}")
    print(f"Agreement check: ||x_Newton - x_I-ARJ8S|| = {np.linalg.norm(xN - xI):.3e}")
