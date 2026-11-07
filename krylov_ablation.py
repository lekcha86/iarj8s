"""
Ablation: I-ARJ8S's linear-solver backend (GMRES vs BiCGSTAB vs TFQMR vs IDR(s))
and preconditioner (ILU vs AMG) on the 3D Bratu problem, at a size where the
project already found a genuine GMRES+ILU advantage over direct factorization.
This directly answers the Limitations paragraph's open question: is any
Krylov method/preconditioner combination clearly better, or does "no single
combination dominates" (Ghai/Lu/Jiao 2019) hold here too?
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
from idrs_solver import idrs


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


def _merit(Fv):
    return float(Fv @ Fv)


def _nonmonotone_ls(F, base_pt, Fbase, direction, window, gamma=1e-4, beta=0.5,
                     xi_min=1e-10, max_bt=40):
    f_base = _merit(Fbase)
    M_k = max(window) if window else f_base
    xi = 1.0
    full_trial, full_Ftrial = None, None
    for i in range(max_bt):
        trial = base_pt + xi * direction
        Ftrial = F(trial)
        if i == 0:
            full_trial, full_Ftrial = trial, Ftrial
        f_trial = _merit(Ftrial)
        if f_trial <= M_k - 2.0 * gamma * xi * f_base:
            return trial, Ftrial
        if xi < xi_min:
            break
        xi *= beta
    if np.all(np.isfinite(full_Ftrial)) and np.all(np.isfinite(full_trial)):
        return full_trial, full_Ftrial
    return trial, Ftrial


def _solve_linear(Jk, rhs, backend, precond, tol):
    if backend == 'direct':
        return spla.spsolve(Jk, rhs)
    if backend == 'gmres':
        x, info = spla.gmres(Jk, rhs, M=precond, rtol=tol, atol=0.0, restart=50, maxiter=200)
    elif backend == 'bicgstab':
        x, info = spla.bicgstab(Jk, rhs, M=precond, rtol=tol, atol=0.0, maxiter=200)
    elif backend == 'tfqmr':
        x, info = spla.tfqmr(Jk, rhs, M=precond, rtol=tol, atol=0.0, maxiter=200)
    elif backend == 'idrs':
        x, info = idrs(Jk, rhs, s=4, tol=tol, maxiter=200, M=precond)
    else:
        raise ValueError(backend)
    return x


def _stage_step(F, Jk, base_pt, Fbase, backend, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        direction = _solve_linear(Jk, -Fbase, backend, precond, eta)
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = _solve_linear(Jk, -Fbase, 'direct', None, 0.0)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def make_precond(Jy, kind):
    if kind == 'ilu':
        ilu = spla.spilu(sp.csc_matrix(Jy), drop_tol=1e-5, fill_factor=15)
        return spla.LinearOperator(Jy.shape, ilu.solve)
    if kind == 'amg':
        ml = pyamg.smoothed_aggregation_solver(Jy.tocsr())
        return spla.LinearOperator(Jy.shape, ml.aspreconditioner().matvec)
    raise ValueError(kind)


def iarj8s(F, J, x0, backend='gmres', precond_kind='ilu', tol=1e-9, max_iter=30,
           theta_bar=0.9, window_size=5, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0

    for _ in range(max_iter):
        Fx = F(x)
        window.append(_merit(Fx))
        if len(window) > window_size:
            window.pop(0)

        t_k = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        theta_k = min(theta_bar, (t_prev - 1.0) / t_k)
        t_prev = t_k
        s_k = x - x_prev
        y_trial = x + theta_k * s_k
        Fy_trial = F(y_trial)
        if _merit(Fy_trial) <= _merit(Fx):
            y, Fy = y_trial, Fy_trial
        else:
            y, Fy = x, Fx

        Jy = J(y)
        precond = make_precond(Jy, precond_kind) if backend != 'direct' else None

        gmres_tol1 = min(gmres_tol_stage1, np.linalg.norm(Fy))
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, backend, precond, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step(F, alpha * Jy, z_prev, Fz_prev, backend, precond,
                                           gmres_tol_j, window)
            window.append(_merit(Fz_prev))
            if len(window) > window_size:
                window.pop(0)
            m = _merit(Fz_prev)
            if m < best_merit:
                best_pt, best_F, best_merit = z_prev, Fz_prev, m

        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    F, J, d = build_bratu_3d(N, 3.51)
    x0 = np.zeros(d)
    print(f"3D Bratu N={N} ({d:,} unknowns)\n")

    configs = [
        ('gmres', 'ilu'), ('gmres', 'amg'),
        ('bicgstab', 'ilu'), ('bicgstab', 'amg'),
        ('tfqmr', 'ilu'), ('tfqmr', 'amg'),
        ('idrs', 'ilu'), ('idrs', 'amg'),
    ]
    for backend, prec in configs:
        t0 = time.time()
        x, hist, status = iarj8s(F, J, x0, backend=backend, precond_kind=prec)
        elapsed = time.time() - t0
        print(f"{backend:>10} + {prec:<4}: {status:>10}, iters={len(hist)-1:2d}, "
              f"time={elapsed:7.3f}s, resid={hist[-1]:.3e}")
