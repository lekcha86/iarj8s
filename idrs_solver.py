"""
IDR(s): Sonneveld & van Gijzen, "IDR(s): A Family of Simple and Fast Algorithms
for Solving Large Nonsymmetric Systems of Linear Equations," SIAM J. Sci.
Comput. 31(2) (2008) 1035-1062. Not available in scipy, so implemented here
directly from the paper's Algorithm (the standard "IDR(s)-biortho" variant),
and verified against scipy.sparse.linalg.spsolve on a test system before use.
"""
import numpy as np
import scipy.sparse as sp


def idrs(A, b, x0=None, s=4, tol=1e-8, maxiter=1000, M=None):
    """Solve A x = b with IDR(s). A: (n,n) sparse/dense/LinearOperator with a
    matvec; M: optional preconditioner object with a .solve(v) or .matvec(v)
    method (right-preconditioning). Returns (x, info) with info=0 on success."""
    n = b.shape[0]
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()

    def matvec(v):
        return A @ v

    def precond(v):
        if M is None:
            return v
        if hasattr(M, 'solve'):
            return M.solve(v)
        return M @ v

    r = b - matvec(x)
    normb = np.linalg.norm(b)
    if normb == 0:
        normb = 1.0
    if np.linalg.norm(r) / normb < tol:
        return x, 0

    rng = np.random.default_rng(0)
    P = rng.standard_normal((n, s))
    P, _ = np.linalg.qr(P)

    G = np.zeros((n, s))
    U = np.zeros((n, s))
    Mmat = np.eye(s)
    om = 1.0

    it = 0
    while it < maxiter:
        f = P.T @ r
        for i in range(s):
            c = np.linalg.solve(Mmat[i:, i:], f[i:])
            v = r - G[:, i:] @ c
            v_p = precond(v)
            u_i = U[:, i:] @ c + om * v_p
            g_i = matvec(u_i)

            for j in range(i):
                alpha = (P[:, j] @ g_i) / Mmat[j, j]
                g_i = g_i - alpha * G[:, j]
                u_i = u_i - alpha * U[:, j]

            U[:, i] = u_i
            G[:, i] = g_i
            Mmat[i:, i] = P[:, i:].T @ G[:, i]

            beta = f[i] / Mmat[i, i]
            r = r - beta * G[:, i]
            x = x + beta * U[:, i]
            it += 1

            res = np.linalg.norm(r) / normb
            if res < tol or it >= maxiter:
                return x, (0 if res < tol else 1)

            if i < s - 1:
                f[i + 1:] = f[i + 1:] - beta * Mmat[i + 1:, i]

        if it >= maxiter:
            break
        v = precond(r)
        t = matvec(v)
        om = (t @ r) / (t @ t) if (t @ t) != 0 else 1.0
        r = r - om * t
        x = x + om * v
        it += 1
        res = np.linalg.norm(r) / normb
        if res < tol:
            return x, 0

    return x, 1


if __name__ == "__main__":
    # Self-test: verify against a direct sparse solve before trusting this
    # in the nonlinear solver.
    import scipy.sparse.linalg as spla
    rng = np.random.default_rng(0)
    n = 200
    A = sp.random(n, n, density=0.05, random_state=1) + sp.identity(n) * 10
    A = A.tocsr()
    x_true = rng.standard_normal(n)
    b = A @ x_true
    x_idrs, info = idrs(A, b, s=4, tol=1e-10, maxiter=500)
    x_direct = spla.spsolve(A, b)
    print("IDR(s) vs direct solve, ||diff|| =", np.linalg.norm(x_idrs - x_direct))
    print("IDR(s) vs true x,      ||diff|| =", np.linalg.norm(x_idrs - x_true))
    print("info =", info)
