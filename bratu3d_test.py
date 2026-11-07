"""
3D Bratu problem: -Delta u - lambda*e^u = 0 on (0,1)^3, u=0 on boundary.
7-point stencil. Direct sparse LU fill-in in 3D scales much worse than 2D,
so this is the genuine regime where GMRES+ILU should show a real advantage.
"""
import time
import resource
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from iarj8s_inertial import iarj8s


def build_bratu_3d(N, lam):
    h = 1.0 / (N + 1)
    e = np.ones(N)
    T1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format='csr')
    I1 = sp.identity(N, format='csr')
    L = (sp.kron(sp.kron(I1, I1), T1) +
         sp.kron(sp.kron(I1, T1), I1) +
         sp.kron(sp.kron(T1, I1), I1)) / h**2
    L = L.tocsr()
    d = N**3

    def F(u):
        return L @ u + lam * np.exp(u)

    def J(u):
        return (L + sp.diags(lam * np.exp(u))).tocsc()

    return F, J, d


def newton_sparse(F, J, x0, tol=1e-9, max_iter=30):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        s = spla.spsolve(Jx, -Fx)
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if hist[-1] < tol:
            break
    return x, hist


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_case(N, lam=3.51):
    F, J, d = build_bratu_3d(N, lam)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({N}^3={d} unknowns), lambda={lam} ===")

    t0 = time.time()
    xN, histN = newton_sparse(F, J, x0, tol=1e-9, max_iter=20)
    tN = time.time() - t0
    memN = peak_mem_mb()
    print(f"Newton (direct sparse LU):  iters={len(histN)-1:2d}, time={tN:8.3f}s, "
          f"resid={histN[-1]:.3e}, peak_mem={memN:.0f}MB")

    t0 = time.time()
    xI, histI = iarj8s(F, J, x0, tol=1e-9, max_iter=30, use_gmres=True,
                        gmres_tol_base=0.1, gmres_tol_stage1=1e-6)
    tI = time.time() - t0
    residI = np.linalg.norm(F(xI))
    memI = peak_mem_mb()
    print(f"I-ARJ8S (GMRES+ILU):        iters={len(histI)-1:2d}, time={tI:8.3f}s, "
          f"resid={residI:.3e}, peak_mem={memI:.0f}MB")

    print(f"Agreement ||x_Newton - x_I-ARJ8S|| = {np.linalg.norm(xN - xI):.3e}")
    print(f"Speedup (direct/GMRES): {tN/tI:.2f}x")
    return dict(N=N, d=d, tN=tN, tI=tI, memN=memN, memI=memI)


if __name__ == "__main__":
    import sys
    Ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [10, 15]
    for N in Ns:
        run_case(N)
