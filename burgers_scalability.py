"""
Base paper's Table 8 scalability test: 1D viscous Burgers equation,
kappa=0.005, "systems with up to 20,000 unknowns, solved in 20 time steps."
Our existing burgers_test.py uses DENSE matrices, fine at M=101 (99
unknowns) but infeasible at 20,000 (a dense 20,000x20,000 matrix alone
would need ~3.2GB). The Burgers Jacobian is actually tridiagonal, so this
rebuilds the same problem with scipy sparse matrices.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_burgers_sparse(M, N, kappa, mu=5.0, eta=4.0, t_final=1.0):
    h = 1.0 / (M - 1)
    k = t_final / (N - 1)
    x = np.linspace(0, 1, M)
    phi0_full = 2 * kappa * eta * np.pi * np.sin(np.pi * x) / (mu + eta * np.cos(np.pi * x))
    n_int = M - 2

    def make_step(phi_prev_full):
        def F(u_int):
            u = np.empty(M)
            u[0] = 0.0
            u[-1] = 0.0
            u[1:-1] = u_int
            ui = u[1:-1]
            uim1 = u[0:-2]
            uip1 = u[2:]
            ut = (ui - phi_prev_full[1:-1]) / k
            ux = (uip1 - uim1) / (2 * h)
            uxx = (uip1 - 2 * ui + uim1) / h**2
            return ut + ui * ux - kappa * uxx

        def J(u_int):
            u = np.empty(M)
            u[0] = 0.0
            u[-1] = 0.0
            u[1:-1] = u_int
            ui = u[1:-1]
            uim1 = u[0:-2]
            uip1 = u[2:]
            main = 1.0 / k + (uip1 - uim1) / (2 * h) + 2 * kappa / h**2
            lower = -ui[1:] / (2 * h) - kappa / h**2   # sub-diagonal, len n_int-1
            upper = ui[:-1] / (2 * h) - kappa / h**2   # super-diagonal, len n_int-1
            Jm = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csc')
            return Jm

        return F, J

    return phi0_full, make_step, n_int, M, N, h, k


def newton_sparse(F, J, x0, tol=1e-8, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Fx = F(x)
        s = spla.spsolve(J(x), -Fx)
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if hist[-1] < tol:
            break
    return x, hist


def arj8s_sparse(F, J, x0, tol=1e-8, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        s1 = spla.spsolve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        alpha = 1.0 - 2.0 * (Fk @ Fz1) / (Fk @ Fk)
        zp, Fzp = z1, Fz1
        for _ in range(3):
            si = spla.spsolve(alpha * Jk, -Fzp)
            zi = zp + si
            Fzp = F(zi)
            zp = zi
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if hist[-1] < tol:
            break
    return x, hist


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


def iarj8s_sparse(F, J, x0, tol=1e-8, max_iter=50, theta_bar=0.9, window_size=5):
    """Exact-solve I-ARJ8S (current, fixed algorithm), sparse direct solves --
    at this problem size (tridiagonal Jacobian) direct sparse solves are
    already fast, so GMRES is not needed here (consistent with this
    project's own 2D finding that GMRES is not always the right tool)."""
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
        s1 = spla.spsolve(Jy, -Fy)
        z1 = y + s1
        Fz1 = F(z1)
        z_prev, Fz_prev = _nonmonotone_ls(F, y, Fy, s1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            si = spla.spsolve(alpha * Jy, -Fz_prev)
            z_i, Fz_i = _nonmonotone_ls(F, z_prev, Fz_prev, si, window)
            window.append(_merit(Fz_i))
            if len(window) > window_size:
                window.pop(0)
            m = _merit(Fz_i)
            if m < best_merit:
                best_pt, best_F, best_merit = z_i, Fz_i, m
            z_prev, Fz_prev = z_i, Fz_i

        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if hist[-1] < tol:
            break
    return x, hist


def run_time_stepping(method_fn, M, N, kappa, tol=1e-8, max_iter=50):
    phi_full, make_step, n_int, M, N, h, k = build_burgers_sparse(M, N, kappa)
    u_int = phi_full[1:-1].copy()
    total_iters = 0
    t0 = time.time()
    final_resid = 0.0
    for j in range(1, N):
        F, J = make_step(phi_full)
        x, hist = method_fn(F, J, u_int, tol=tol, max_iter=max_iter)
        total_iters += len(hist) - 1
        u_int = x
        phi_full = phi_full.copy()
        phi_full[1:-1] = u_int
        final_resid = np.linalg.norm(F(x))
    elapsed = time.time() - t0
    return total_iters, elapsed, final_resid


if __name__ == "__main__":
    # Paper's Table 8: kappa=0.005, up to 20,000 unknowns, 20 time steps.
    M = 20002  # interior unknowns = M-2 = 20,000
    N = 21     # 20 time steps
    kappa = 0.005

    print(f"Burgers scalability test: {M-2} unknowns, {N-1} time steps, kappa={kappa}\n")

    for name, fn in [('Newton', newton_sparse), ('ARJ8S', arj8s_sparse), ('I-ARJ8S', iarj8s_sparse)]:
        t0 = time.time()
        total_iters, elapsed, final_resid = run_time_stepping(fn, M, N, kappa, tol=1e-8)
        print(f"{name:>10}: total_iters={total_iters}, avg_iter/step={total_iters/(N-1):.2f}, "
              f"time={elapsed:.3f}s, final_resid={final_resid:.3e}")
