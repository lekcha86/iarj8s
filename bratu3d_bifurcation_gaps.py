"""
Fills the 6 untested cells in the paper's 3D bifurcation region map
(Table 15): (lambda,c) = (5.5,4.5), (5.7,5), (5.8,5), (5.9,4.5), (5.9,5),
(6.0,4.5). All other cells in that lambda in [5.5,6.0] x c in {4,4.5,5}
grid were already tested; these six are the only gaps.

Tests all four methods at each point, exactly matching the rest of
Table 15's methodology: 3D Bratu, N=20 (8,000 unknowns), boundary-respecting
hard initial guess u0 = c*sin(pi x)*sin(pi y)*sin(pi z), max 25 iterations,
tol 1e-9.

Requires: numpy, scipy (pip install numpy scipy)

Usage:
    python3 bratu3d_bifurcation_gaps.py
        (runs all 6 missing points automatically -- no arguments needed)

    python3 bratu3d_bifurcation_gaps.py 5.5 4.5
        (runs just one custom (lambda, c) point instead)

Results are written incrementally to
bratu3d_bifurcation_gaps_results.csv as each point finishes, so a crash
partway through does not lose earlier results.
"""
import sys
import time
import csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 20
TOL = 1e-9
MAX_ITER = 25

# The 6 gaps in Table 15, as (lambda, c) pairs.
MISSING_POINTS = [
    (5.5, 4.5),
    (5.7, 5.0),
    (5.8, 5.0),
    (5.9, 4.5),
    (5.9, 5.0),
    (6.0, 4.5),
]


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


def hard_init_3d(N, c):
    h = 1.0 / (N + 1)
    xs = np.linspace(h, 1 - h, N)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    return (c * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)).flatten()


def newton_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            s = spla.spsolve(J(x), -F(x))
        except (RuntimeError, ValueError):
            return x, hist, 'NaN'
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def arj8s_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jk, Fk = J(x), F(x)
        try:
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
        except (RuntimeError, ValueError):
            return x, hist, 'NaN'
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def newton_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, eta_max=0.1, restart=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    for _ in range(max_iter):
        Fx, Jx = F(x), J(x)
        try:
            ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
            precond = spla.LinearOperator(Jx.shape, ilu.solve)
            s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                                  restart=restart, maxiter=200)
        except (RuntimeError, ValueError, FloatingPointError):
            return x, hist, 'NaN'
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
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0,
                          restart=50, maxiter=200)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError('non-finite GMRES solution')
    return x


def _stage_step(F, Jk, base_pt, Fbase, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        try:
            direction = _solve_linear(Jk, -Fbase, precond, eta)
        except (RuntimeError, ValueError, FloatingPointError):
            eta = max(eta / eta_shrink, eta_min)
            continue
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    try:
        direction = spla.spsolve(Jk, -Fbase)
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError('non-finite direct solution')
    except (RuntimeError, ValueError, FloatingPointError):
        return base_pt, Fbase * np.nan
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, theta_bar=0.9,
                  window_size=5, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
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
        try:
            ilu = spla.spilu(sp.csc_matrix(Jy), drop_tol=1e-5, fill_factor=15)
            precond = spla.LinearOperator(Jy.shape, ilu.solve)
        except (RuntimeError, ValueError):
            return x, hist, 'NaN'

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


METHODS = [
    ('Newton-direct', newton_direct),
    ('ARJ8S-direct', arj8s_direct),
    ('Newton-GMRES', newton_gmres),
    ('I-ARJ8S-GMRES', iarj8s_gmres),
]


def run_point(lam, c, writer, csvfile):
    F, J, d = build_bratu_3d(N, lam)
    x0 = hard_init_3d(N, c)
    print(f"\n=== lambda={lam}, c={c} ===", flush=True)

    for name, fn in METHODS:
        t0 = time.time()
        x, hist, status = fn(F, J, x0)
        elapsed = time.time() - t0
        print(f"  {name:>14}: {status:>10}, iters={len(hist)-1:2d}, "
              f"time={elapsed:6.2f}s, resid={hist[-1]:.3e}", flush=True)
        writer.writerow(dict(lam=lam, c=c, method=name, status=status,
                              iters=len(hist) - 1, time=round(elapsed, 3),
                              resid=hist[-1]))
        csvfile.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) >= 2:
        points = [(float(argv[0]), float(argv[1]))]
    else:
        points = MISSING_POINTS

    fieldnames = ['lam', 'c', 'method', 'status', 'iters', 'time', 'resid']
    with open('bratu3d_bifurcation_gaps_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for lam, c in points:
            run_point(lam, c, writer, csvfile)

    print("\nAll done. Full results in bratu3d_bifurcation_gaps_results.csv")


# Runs the 6 missing Table-15 points automatically. Uses sys.argv only if
# it looks like real CLI args (not a notebook kernel's own arguments).
if len(sys.argv) >= 3:
    try:
        float(sys.argv[1])
        float(sys.argv[2])
        main(sys.argv[1:])
    except ValueError:
        main([])
else:
    main([])
