"""
Standalone, self-contained script combining two things a reviewer asked us
to strengthen: (1) push the 3D Bratu Newton-GMRES vs I-ARJ8S-GMRES
comparison to larger scale than the 125,000 unknowns tested so far
(targeting the 250,000-500,000+ range), and (2) instrument GMRES's own
INNER iteration count, not just outer Newton/I-ARJ8S iterations, to
investigate why I-ARJ8S-GMRES's wall-clock time was non-monotonic in N.

In smaller-scale sandbox testing (N up to 40, 64,000 unknowns) we found a
striking pattern: I-ARJ8S-GMRES's total inner GMRES iterations are
dominated by its SECOND outer iteration specifically, which needs far more
inner iterations than its first (e.g. at N=40: 119 then 325 -- almost 3x
more on the second outer step), while Newton-GMRES's inner-iteration count
per outer step stays flat and modest (16-31 throughout). This script lets
you check whether that pattern continues, worsens, or changes at larger
scale, which is likely the mechanism behind the non-monotonic wall-clock
finding already in the paper.

Requires: numpy, scipy (pip install numpy scipy)

Usage:
    python3 newton_gmres_instrumented_scaling.py             # default: N=50 63 79
    python3 newton_gmres_instrumented_scaling.py 50 63 79 100 # custom N values
                                                                 (unknowns = N^3;
                                                                  N=63 -> 250,047;
                                                                  N=79 -> 493,039;
                                                                  N=100 -> 1,000,000)

Notes:
  - Only the two GMRES-based methods (Newton-GMRES, I-ARJ8S-GMRES) are run
    by default at these larger sizes -- direct-solve Newton/ARJ8S would be
    extremely slow and memory-hungry at N=79+ in 3D (this project's own
    findings already show direct methods losing badly to GMRES well before
    this scale). Pass --include-direct to also run them if you have the
    patience/memory.
  - Results (including the per-outer-iteration inner GMRES counts) are
    written incrementally to
    newton_gmres_instrumented_scaling_results.csv as each (N, method) pair
    finishes.
"""
import sys
import time
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LAMBDA = 3.51
TOL = 1e-9
MAX_ITER = 30


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


class GMRESCounter:
    def __init__(self):
        self.n_iter = 0

    def __call__(self, rk=None):
        self.n_iter += 1


def newton_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        s = spla.spsolve(J(x), -F(x))
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN', None
        if hist[-1] < tol:
            return x, hist, 'converged', None
    return x, hist, 'maxiter', None


def arj8s_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jk, Fk = J(x), F(x)
        s1 = spla.spsolve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        denom = Fk @ Fk
        if denom == 0 or not np.isfinite(denom):
            return x, hist, 'NaN', None
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
            return x, hist, 'NaN', None
        if hist[-1] < tol:
            return x, hist, 'converged', None
    return x, hist, 'maxiter', None


def newton_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, eta_max=0.1,
                  restart=50, gmres_maxiter=200):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    inner_per_outer = []
    for _ in range(max_iter):
        Fx, Jx = F(x), J(x)
        ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
        precond = spla.LinearOperator(Jx.shape, ilu.solve)
        counter = GMRESCounter()
        s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                              restart=restart, maxiter=gmres_maxiter,
                              callback=counter, callback_type='pr_norm')
        inner_per_outer.append(counter.n_iter)
        x_new = x + s
        resid_new = np.linalg.norm(F(x_new))
        resid_old = hist[-1]
        x = x_new
        hist.append(resid_new)
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN', inner_per_outer
        if resid_new < tol:
            return x, hist, 'converged', inner_per_outer
        eta = min(eta_max, resid_new / resid_old) if resid_old > 0 else eta_max
    return x, hist, 'maxiter', inner_per_outer


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


def _solve_linear_instr(Jk, rhs, precond, gmres_tol, box):
    counter = GMRESCounter()
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0,
                          restart=50, maxiter=200,
                          callback=counter, callback_type='pr_norm')
    box[0] += counter.n_iter
    return x


def _stage_step_instr(F, Jk, base_pt, Fbase, precond, eta0, window, box,
                       eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        direction = _solve_linear_instr(Jk, -Fbase, precond, eta, box)
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = spla.spsolve(Jk, -Fbase)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, theta_bar=0.9,
                  window_size=5, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0
    box = [0]
    inner_per_outer = []

    for _ in range(max_iter):
        start = box[0]
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
        z_prev, Fz_prev = _stage_step_instr(F, Jy, y, Fy, precond, gmres_tol1, window, box)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step_instr(F, alpha * Jy, z_prev, Fz_prev,
                                                 precond, gmres_tol_j, window, box)
            window.append(_merit(Fz_prev))
            if len(window) > window_size:
                window.pop(0)
            m = _merit(Fz_prev)
            if m < best_merit:
                best_pt, best_F, best_merit = z_prev, Fz_prev, m

        inner_per_outer.append(box[0] - start)
        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN', inner_per_outer
        if hist[-1] < tol:
            return x, hist, 'converged', inner_per_outer
    return x, hist, 'maxiter', inner_per_outer


def run_size(N, include_direct, writer, csvfile):
    F, J, d = build_bratu_3d(N, LAMBDA)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({d:,} unknowns) ===", flush=True)

    methods = [('Newton-GMRES', newton_gmres), ('I-ARJ8S-GMRES', iarj8s_gmres)]
    if include_direct:
        methods = [('Newton', newton_direct), ('ARJ8S', arj8s_direct)] + methods

    for name, fn in methods:
        t0 = time.time()
        result = fn(F, J, x0)
        elapsed = time.time() - t0
        x, hist, status, inner = result
        total_inner = sum(inner) if inner else None
        print(f"{name:>14}: {status:>10}, outer={len(hist)-1:2d}, "
              f"time={elapsed:9.2f}s, resid={hist[-1]:.3e}, "
              f"inner_GMRES_total={total_inner}, inner_per_outer={inner}", flush=True)
        writer.writerow(dict(N=N, d=d, method=name, status=status,
                              outer_iters=len(hist) - 1, time=round(elapsed, 3),
                              resid=hist[-1], inner_total=total_inner,
                              inner_per_outer=inner))
        csvfile.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sizes', nargs='*', type=int, default=[50, 63, 79])
    parser.add_argument('--include-direct', action='store_true',
                         help="also run direct-solve Newton/ARJ8S (slow at this scale)")
    args = parser.parse_args()

    fieldnames = ['N', 'd', 'method', 'status', 'outer_iters', 'time', 'resid',
                  'inner_total', 'inner_per_outer']
    with open('newton_gmres_instrumented_scaling_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for N in args.sizes:
            run_size(N, args.include_direct, writer, csvfile)

    print("\nAll done. Full results in newton_gmres_instrumented_scaling_results.csv")


if __name__ == "__main__":
    main()
