"""
Combines this project's two established strengths in one test: GMRES
matters at 3D scale (Section 7.6), and I-ARJ8S's globalization uniquely
survives hard, boundary-respecting initial guesses near bifurcation in 2D
(Section 7.7). We had never tested the 3D analogue: does I-ARJ8S-GMRES
survive hard 3D Bratu cases -- at a scale where GMRES already matters --
where Newton-GMRES, Newton-direct, and ARJ8S-direct all fail?
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
sys.path.insert(0, '.')
from bratu3d_test import build_bratu_3d
from newton_gmres import newton_gmres


def newton_direct(F, J, x0, tol=1e-9, max_iter=25):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Fx = F(x)
        s = spla.spsolve(J(x), -Fx)
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def arj8s_direct(F, J, x0, tol=1e-9, max_iter=25):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        s1 = spla.spsolve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        denom = Fk @ Fk
        if denom == 0 or not np.isfinite(denom):
            return x, hist, 'NaN'
        alpha = 1.0 - 2.0 * (Fk @ Fz1) / denom
        zp, Fzp = z1, Fz1
        for _ in range(3):
            si = spla.spsolve(alpha * Jk, -Fzp)
            zi = zp + si
            Fzp = F(zi)
            zp = zi
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def newton_gmres_wrap(F, J, x0, tol=1e-9, max_iter=25):
    return newton_gmres(F, J, x0, tol=tol, max_iter=max_iter)


# --- I-ARJ8S-GMRES, current fixed algorithm ------------------------------
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


def iarj8s_gmres(F, J, x0, tol=1e-9, max_iter=25, theta_bar=0.9, window_size=5,
                  gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
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
        ilu = spla.spilu(sp.csc_matrix(Jy), drop_tol=1e-5, fill_factor=15)
        precond = spla.LinearOperator(Jy.shape, ilu.solve)

        gmres_tol1 = min(gmres_tol_stage1, np.linalg.norm(Fy))
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, precond, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step(F, alpha * Jy, z_prev, Fz_prev, precond,
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


def hard_init_3d(N, c):
    h = 1.0 / (N + 1)
    xs = np.linspace(h, 1 - h, N)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    return (c * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)).flatten()


if __name__ == "__main__":
    N = 20  # 8,000 unknowns -- already established as a size where GMRES matters (Sec 7.6)
    methods = [
        ('Newton-direct', newton_direct),
        ('Newton-GMRES', newton_gmres_wrap),
        ('ARJ8S-direct', arj8s_direct),
        ('I-ARJ8S-GMRES', iarj8s_gmres),
    ]

    for lam in [5.5, 6.0, 6.3, 6.5]:
        for c in [4, 6, 8]:
            x0 = hard_init_3d(N, c)
            F, J, d = build_bratu_3d(N, lam)
            print(f"\n--- lambda={lam}, c={c} (N={N}, d={d}) ---")
            for name, fn in methods:
                t0 = time.time()
                try:
                    x, hist, status = fn(F, J, x0)
                    elapsed = time.time() - t0
                    print(f"  {name:>14}: {status:>10}, iters={len(hist)-1:2d}, "
                          f"time={elapsed:6.2f}s, resid={hist[-1]:.2e}")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {name:>14}: {'ERROR':>10} ({type(e).__name__}), time={elapsed:6.2f}s")
