"""
Standalone, self-contained script: does Newton-GMRES eventually lose to
I-ARJ8S-GMRES at larger scale, the way exact-solve Newton/ARJ8S do?

In the sandbox that built this project, testing this at N=40 (64,000
unknowns) timed out after ~280 seconds. This script is exactly that test,
handed off to run on your own machine, at whatever sizes you give it.

Includes four methods per size:
  - Newton (direct sparse LU)
  - Newton-GMRES (plain Newton, each step solved inexactly via GMRES+ILU,
    Eisenstat-Walker-style adaptive forcing term, eta_max=0.1 matching the
    base ARJ8S paper's eta0)
  - ARJ8S (direct sparse LU, exact solves)
  - I-ARJ8S-GMRES (current/fixed algorithm: safeguarded inertial step +
    non-monotone line search + GMRES/ILU with the asymmetric per-stage
    forcing-term schedule)

Requires: numpy, scipy (pip install numpy scipy)

Usage:
    python3 newton_gmres_scaling.py                # default: N=20 30 40 50
    python3 newton_gmres_scaling.py 20 30 40 50 60  # custom N values (N^3 unknowns)

Notes:
  - Direct-solve methods (Newton, ARJ8S) will eventually become the
    bottleneck as N grows (this is the whole point of the comparison --
    see the project's existing findings that direct fill-in in 3D scales
    much worse than Krylov methods). If a direct solve becomes
    impractically slow at your largest N, you can still get useful data
    by only running the two GMRES-based methods; see the --skip-direct-above
    flag.
  - Results are written incrementally to newton_gmres_scaling_results.csv
    as each (N, method) pair finishes, so a crash or Ctrl-C partway through
    does not lose earlier data.
"""
import sys
import time
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
LAMBDA = 3.51
TOL = 1e-9
MAX_ITER = 30
# ---------------------------------------------------------------------------


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


# --- Newton (direct) ---------------------------------------------------
def newton_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
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


# --- Newton-GMRES (Eisenstat-Walker forcing term) -----------------------
def newton_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, eta_max=0.1,
                  restart=50, gmres_maxiter=200):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    for _ in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
        precond = spla.LinearOperator(Jx.shape, ilu.solve)
        s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                              restart=restart, maxiter=gmres_maxiter)
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


# --- ARJ8S (direct) -------------------------------------------------------
def arj8s_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
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


# --- I-ARJ8S-GMRES (current, fixed algorithm) ----------------------------
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


def _solve_linear(Jk, rhs, use_gmres, precond, gmres_tol):
    if not use_gmres:
        return spla.spsolve(Jk, rhs)
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0, restart=50, maxiter=200)
    return x


def _stage_step(F, Jk, base_pt, Fbase, use_gmres, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        direction = _solve_linear(Jk, -Fbase, use_gmres, precond, eta)
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = _solve_linear(Jk, -Fbase, False, None, 0.0)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres(F, J, x0, tol=TOL, max_iter=MAX_ITER, theta_bar=0.9, window_size=5,
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
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, True, precond, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step(F, alpha * Jy, z_prev, Fz_prev, True, precond,
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
    ('Newton', newton_direct, {}, False),
    ('Newton-GMRES', newton_gmres, {}, True),
    ('ARJ8S', arj8s_direct, {}, False),
    ('I-ARJ8S-GMRES', iarj8s_gmres, {}, True),
]


def run_size(N, skip_direct, writer, csvfile):
    F, J, d = build_bratu_3d(N, LAMBDA)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({d:,} unknowns) ===", flush=True)

    for name, fn, kwargs, is_gmres_based in METHODS:
        if skip_direct and not is_gmres_based:
            print(f"{name:>14}: skipped (--skip-direct-above)", flush=True)
            continue
        t0 = time.time()
        x, hist, status = fn(F, J, x0, **kwargs)
        elapsed = time.time() - t0
        print(f"{name:>14}: {status:>10}, iters={len(hist)-1:2d}, "
              f"time={elapsed:8.3f}s, resid={hist[-1]:.3e}", flush=True)
        writer.writerow(dict(N=N, d=d, method=name, status=status,
                              iters=len(hist) - 1, time=round(elapsed, 3),
                              resid=hist[-1]))
        csvfile.flush()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('sizes', nargs='*', type=int, default=[79],
                         help="N values to test (unknowns = N^3)")
    parser.add_argument('--skip-direct-above', type=int, default=None,
                         help="skip Newton/ARJ8S direct solves for N above this "
                              "(they will dominate wall-clock time at large N)")
    args = parser.parse_args(argv)

    fieldnames = ['N', 'd', 'method', 'status', 'iters', 'time', 'resid']
    with open('newton_gmres_scaling_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for N in args.sizes:
            skip_direct = args.skip_direct_above is not None and N > args.skip_direct_above
            run_size(N, skip_direct, writer, csvfile)

    print("\nAll done. Full results in newton_gmres_scaling_results.csv")


# Call main with an empty list to use the default size (N=79) and prevent
# argparse from trying to parse Colab/Jupyter's own internal arguments.
main([])
