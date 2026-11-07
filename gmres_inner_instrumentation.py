"""
Instruments GMRES's own inner iteration count (not just outer Newton/I-ARJ8S
iterations) for both Newton-GMRES and I-ARJ8S-GMRES, to investigate the
non-monotonic wall-clock pattern found in Section 7.10: I-ARJ8S-GMRES's own
time was not smooth in N (12.29s and 12.81s at 27k/64k unknowns, nearly
identical despite N more than doubling, then jumping to 155.07s at 125k) --
we suspected GMRES's own inner-iteration count as the most likely cause but
had not measured it directly. This does that.
"""
import time
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class GMRESCounter:
    """Counts inner (Krylov) GMRES iterations via the pr_norm callback,
    which scipy calls once per inner iteration (not just per restart)."""
    def __init__(self):
        self.n_iter = 0

    def __call__(self, rk=None):
        self.n_iter += 1


def newton_gmres_instrumented(F, J, x0, tol=1e-9, max_iter=30, eta_max=0.1,
                               restart=50, gmres_maxiter=200):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    total_inner = 0
    inner_per_outer = []
    for _ in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
        precond = spla.LinearOperator(Jx.shape, ilu.solve)
        counter = GMRESCounter()
        s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                              restart=restart, maxiter=gmres_maxiter,
                              callback=counter, callback_type='pr_norm')
        total_inner += counter.n_iter
        inner_per_outer.append(counter.n_iter)
        x_new = x + s
        resid_new = np.linalg.norm(F(x_new))
        resid_old = hist[-1]
        x = x_new
        hist.append(resid_new)
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN', total_inner, inner_per_outer
        if resid_new < tol:
            return x, hist, 'converged', total_inner, inner_per_outer
        eta = min(eta_max, resid_new / resid_old) if resid_old > 0 else eta_max
    return x, hist, 'maxiter', total_inner, inner_per_outer


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


def _solve_linear_instrumented(Jk, rhs, precond, gmres_tol, inner_counter):
    counter = GMRESCounter()
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0,
                          restart=50, maxiter=200,
                          callback=counter, callback_type='pr_norm')
    inner_counter[0] += counter.n_iter
    return x


def _stage_step_instrumented(F, Jk, base_pt, Fbase, precond, eta0, window, inner_counter,
                              eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        direction = _solve_linear_instrumented(Jk, -Fbase, precond, eta, inner_counter)
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = spla.spsolve(Jk, -Fbase)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres_instrumented(F, J, x0, tol=1e-9, max_iter=30, theta_bar=0.9,
                               window_size=5, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0
    inner_counter = [0]  # mutable box so nested calls can accumulate
    inner_per_outer = []

    for _ in range(max_iter):
        outer_start_inner = inner_counter[0]
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
        z_prev, Fz_prev = _stage_step_instrumented(F, Jy, y, Fy, precond, gmres_tol1,
                                                    window, inner_counter)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step_instrumented(F, alpha * Jy, z_prev, Fz_prev,
                                                        precond, gmres_tol_j, window,
                                                        inner_counter)
            window.append(_merit(Fz_prev))
            if len(window) > window_size:
                window.pop(0)
            m = _merit(Fz_prev)
            if m < best_merit:
                best_pt, best_F, best_merit = z_prev, Fz_prev, m

        inner_per_outer.append(inner_counter[0] - outer_start_inner)
        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN', inner_counter[0], inner_per_outer
        if hist[-1] < tol:
            return x, hist, 'converged', inner_counter[0], inner_per_outer
    return x, hist, 'maxiter', inner_counter[0], inner_per_outer


def build_bratu_3d(N, lam):
    """3D Bratu problem: -Delta u - lambda*e^u = 0 on (0,1)^3, u=0 on
    boundary, 7-point stencil. Included inline so this script has no
    dependency on other project files."""
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


if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [20, 30, 40]
    for N in Ns:
        F, J, d = build_bratu_3d(N, 3.51)
        x0 = np.zeros(d)
        print(f"\n=== 3D Bratu N={N} ({d:,} unknowns) ===")

        t0 = time.time()
        xN, histN, statusN, totalN, perN = newton_gmres_instrumented(F, J, x0)
        tN = time.time() - t0
        print(f"Newton-GMRES:  {statusN}, outer={len(histN)-1}, "
              f"total_inner_GMRES_iters={totalN}, inner_per_outer={perN}, time={tN:.2f}s")

        t0 = time.time()
        xI, histI, statusI, totalI, perI = iarj8s_gmres_instrumented(F, J, x0)
        tI = time.time() - t0
        print(f"I-ARJ8S-GMRES: {statusI}, outer={len(histI)-1}, "
              f"total_inner_GMRES_iters={totalI}, inner_per_outer={perI}, time={tI:.2f}s")
