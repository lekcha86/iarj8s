"""
Architectural efficiency ablation, mirroring the base ARJ8S paper's Table 14
("Without Decomposition" vs "With Decomposition"): does reusing a single
ILU preconditioner across all 4 Super-Ostrowski stages (I-ARJ8S-GMRES's
actual design) actually matter, versus rebuilding a fresh ILU preconditioner
for every stage's own matrix (Jy for stage 1, alpha*Jy for stages 2-4)?

The base paper's own version of this ablation used DIRECT factorizations
(a full sparse LU per sub-step, vs one LU + 3 back-substitutions); the
GMRES-based analogue here is: one ILU preconditioner per outer iteration,
reused across 4 GMRES solves, vs building a fresh ILU preconditioner
before every one of the 4 GMRES solves.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
sys.path.insert(0, '.')
from bratu3d_test import build_bratu_3d


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


def _build_precond(Jmat):
    ilu = spla.spilu(sp.csc_matrix(Jmat), drop_tol=1e-5, fill_factor=15)
    return spla.LinearOperator(Jmat.shape, ilu.solve)


def _solve_linear(Jk, rhs, precond, gmres_tol):
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0, restart=50, maxiter=200)
    return x


def _stage_step(F, Jk, base_pt, Fbase, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        direction = _solve_linear(Jk, -Fbase, precond, eta)
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = spla.spsolve(Jk, -Fbase)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres(F, J, x0, reuse_precond, tol=1e-9, max_iter=30,
                  theta_bar=0.9, window_size=5, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
    """reuse_precond=True: build ILU once per outer iteration, reuse for all
    4 stages (I-ARJ8S-GMRES's actual design). reuse_precond=False: build a
    fresh ILU preconditioner for each stage's own matrix."""
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0
    precond_builds = 0

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
        shared_precond = _build_precond(Jy) if reuse_precond else None
        if reuse_precond:
            precond_builds += 1

        gmres_tol1 = min(gmres_tol_stage1, np.linalg.norm(Fy))
        precond1 = shared_precond if reuse_precond else _build_precond(Jy)
        if not reuse_precond:
            precond_builds += 1
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, precond1, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            Jstage = alpha * Jy
            precond_j = shared_precond if reuse_precond else _build_precond(Jstage)
            if not reuse_precond:
                precond_builds += 1
            z_prev, Fz_prev = _stage_step(F, Jstage, z_prev, Fz_prev, precond_j,
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
            return x, hist, 'NaN', precond_builds
        if hist[-1] < tol:
            return x, hist, 'converged', precond_builds
    return x, hist, 'maxiter', precond_builds


if __name__ == "__main__":
    for N in [20, 30]:
        F, J, d = build_bratu_3d(N, 3.51)
        x0 = np.zeros(d)
        print(f"\n=== 3D Bratu N={N} ({d:,} unknowns) ===")

        t0 = time.time()
        x_r, hist_r, status_r, builds_r = iarj8s_gmres(F, J, x0, reuse_precond=True)
        t_r = time.time() - t0
        print(f"  WITH reuse (proposed):    {status_r}, iters={len(hist_r)-1}, "
              f"precond_builds={builds_r}, time={t_r:.3f}s, resid={hist_r[-1]:.3e}")

        t0 = time.time()
        x_n, hist_n, status_n, builds_n = iarj8s_gmres(F, J, x0, reuse_precond=False)
        t_n = time.time() - t0
        print(f"  WITHOUT reuse (rebuild):  {status_n}, iters={len(hist_n)-1}, "
              f"precond_builds={builds_n}, time={t_n:.3f}s, resid={hist_n[-1]:.3e}")
        print(f"  Speedup from reuse: {t_n/t_r:.2f}x")
