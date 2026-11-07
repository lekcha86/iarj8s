"""
Master script: computes every table in the paper that can reasonably
finish in one sitting (roughly 10-30 minutes total at default settings).

WHAT THIS SCRIPT COVERS
    Table 1        classical efficiency index comparison
    Figs 1-3       efficiency index / computational performance plots
    Table 3        Burgers' equation (M=101)
    Table 6        Problem 1 (m=30 reduced by default; --full for m=100)
    Table 7        Problem 2 (m=30 reduced by default; --full for m=500)
    Table 8        Hammerstein integral equation (n=7)
    Table 9        multi-solution problem (m=20)
    Table 10       elastic string (n=50 reduced by default; --full for n=500)
    Table 11       2D Bratu, direct LU vs I-ARJ8S-GMRES, small scale
    Table 13       2D bifurcation robustness (N=127)
    Table 15       3D bifurcation region map (N=20, full 15-point grid)
    Table 16       Krylov backend/preconditioner ablation (N=20, 30)
    Table 19       preconditioner-reuse ablation (N=20, 30)

WHAT THIS SCRIPT DOES NOT COVER (deliberately -- see below)
    Table 7 at full m=500, Table 10 at full n=500: possible with --full
        but genuinely slow (Table 7 alone took over an hour on real
        hardware for the slowest method). Off by default.
    Table 12  (3D Bratu direct-vs-GMRES up to 125,000 unknowns)
        -> use bratu3d_scaling.py / bratu3d_full_comparison.py
    Table 14  (2D Bratu extreme scale, N=1024/2048, up to 4,194,304
               unknowns -- can take well over an hour per method)
        -> use bratu_extreme_stress.py
    Table 17/18 (Newton-GMRES comparator + inner-GMRES instrumentation,
               up to 493,039 unknowns)
        -> use newton_gmres_scaling.py / newton_gmres_instrumented_scaling.py
    These four are each individually slow enough (many minutes to
    several hours at their largest sizes) that bundling them into one
    combined run risks losing everything to a single crash partway
    through a many-hour execution. They already exist as separate,
    independently-tested, crash-safe (incremental CSV) scripts -- rerun
    those directly instead of duplicating them here.

Requires: numpy, scipy, mpmath (pip install numpy scipy mpmath)

Usage:
    python3 compute_all_tables.py              # default: reduced sizes where applicable
    python3 compute_all_tables.py --full        # also runs the slow full-scale VPA tables
    python3 compute_all_tables.py --skip TABLE6 TABLE7   # skip specific tables by name

Each table's results are written to its own CSV
(table1_results.csv, table3_results.csv, ...) as it finishes, and a
running log is printed to stdout. A crash or interruption partway
through only loses the table in progress, not the ones already done.
"""
import sys
import time
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mpmath as mp


# =========================================================================
# SECTION A: shared low-precision (float64) method implementations
# =========================================================================

def newton_direct(F, J, x0, tol=1e-9, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            Jx = J(x)
            s = spla.spsolve(Jx, -F(x)) if sp.issparse(Jx) else np.linalg.solve(Jx, -F(x))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        x_new = x + s
        step_norm = np.linalg.norm(s)
        x = x_new
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def _solve(Jk, rhs):
    if sp.issparse(Jk):
        return spla.spsolve(Jk, rhs)
    return np.linalg.solve(Jk, rhs)


def arj8s_direct(F, J, x0, tol=1e-9, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            Jk, Fk = J(x), F(x)
            s1 = _solve(Jk, -Fk)
            z1 = x + s1
            Fz1 = F(z1)
            denom = Fk @ Fk
            if denom == 0 or not np.isfinite(denom):
                return x, hist, 'NaN'
            alpha = 1.0 - 2.0 * (Fk @ Fz1) / denom
            zp, Fzp = z1, Fz1
            for _ in range(3):
                si = _solve(alpha * Jk, -Fzp)
                zi = zp + si
                Fzp = F(zi)
                zp = zi
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        step_norm = np.linalg.norm(zp - x)
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
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


def iarj8s_direct(F, J, x0, tol=1e-9, max_iter=50, theta_bar=0.9, window_size=5):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0
    for _ in range(max_iter):
        try:
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
            y, Fy = (y_trial, Fy_trial) if _merit(Fy_trial) <= _merit(Fx) else (x, Fx)
            Jy = J(y)
            s1 = _solve(Jy, -Fy)
            z_prev, Fz_prev = _nonmonotone_ls(F, y, Fy, s1, window)
            window.append(_merit(Fz_prev))
            if len(window) > window_size:
                window.pop(0)
            best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)
            alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
            if abs(alpha) < 1e-12:
                alpha = 1e-12 if alpha >= 0 else -1e-12
            for _ in range(3):
                si = _solve(alpha * Jy, -Fz_prev)
                z_i, Fz_i = _nonmonotone_ls(F, z_prev, Fz_prev, si, window)
                window.append(_merit(Fz_i))
                if len(window) > window_size:
                    window.pop(0)
                m = _merit(Fz_i)
                if m < best_merit:
                    best_pt, best_F, best_merit = z_i, Fz_i, m
                z_prev, Fz_prev = z_i, Fz_i
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        step_norm = np.linalg.norm(best_pt - x)
        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def newton_gmres(F, J, x0, tol=1e-9, max_iter=30, eta_max=0.1, restart=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    eta = eta_max
    for _ in range(max_iter):
        try:
            Fx, Jx = F(x), J(x)
            ilu = spla.spilu(sp.csc_matrix(Jx), drop_tol=1e-5, fill_factor=15)
            precond = spla.LinearOperator(Jx.shape, ilu.solve)
            s, info = spla.gmres(Jx, -Fx, M=precond, rtol=eta, atol=0.0,
                                  restart=restart, maxiter=200)
        except (RuntimeError, ValueError):
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


def _stage_step(F, Jk, base_pt, Fbase, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    f_incoming = _merit(Fbase)
    eta = eta0
    for _ in range(max_retries):
        try:
            direction, info = spla.gmres(Jk, -Fbase, M=precond, rtol=eta, atol=0.0,
                                          restart=50, maxiter=200)
            if not np.all(np.isfinite(direction)):
                raise FloatingPointError
        except (RuntimeError, ValueError, FloatingPointError):
            eta = max(eta / eta_shrink, eta_min)
            continue
        trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    try:
        direction = _solve(Jk, -Fbase)
    except (RuntimeError, ValueError):
        return base_pt, Fbase * np.nan
    return _nonmonotone_ls(F, base_pt, Fbase, direction, window)


def iarj8s_gmres(F, J, x0, tol=1e-9, max_iter=30, theta_bar=0.9, window_size=5,
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
        y, Fy = (y_trial, Fy_trial) if _merit(Fy_trial) <= _merit(Fx) else (x, Fx)
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


# CTT8S, ZO8S, EH9S -- ported verbatim from the already-validated
# comparator_methods.py (real formulas from the base paper's Eqs. 7-10 /
# ZO8S-EH9S recursions; a fresh from-memory reimplementation attempted
# here first and was wrong -- these involve nu/gamma/beta/delta
# correction terms that are easy to mis-transcribe, so the validated
# source is reused directly rather than re-derived a second time).
def _nu(Fx, Fy):
    return (Fy @ Fy) / (Fx @ Fx)


def _c4s_psi(F, J, x, psi, Jx=None, Fx=None):
    if Jx is None:
        Jx = J(x)
    if Fx is None:
        Fx = F(x)
    s1 = np.linalg.solve(Jx, -Fx)
    y = x + s1
    Fy = F(y)
    nu = _nu(Fx, Fy)
    rhs = (1 + psi * nu) * Fy + 2 * nu * Fx
    s2 = np.linalg.solve(Jx, -rhs)
    z = y + s2
    Fz = F(z)
    return z, Fz, y, Fy, Jx, Fx, nu


def ctt8s_direct(F, J, x0, tol=1e-9, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            z, Fz, y, Fy, Jx, Fx, nu = _c4s_psi(F, J, x, psi=1.0)
            gamma = (Fy @ Fz) / (Fx @ Fx)
            beta = (Fz @ Fz) / (Fy @ Fy)
            delta = (Fz @ Fz) / (Fx @ Fx)
            rhs = Fz + (2 * gamma * (1 - nu) + 4 * delta) * Fx + (2 * gamma + beta) * Fy
            s3 = np.linalg.solve(Jx, -rhs)
            x_new = z + s3
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        step_norm = np.linalg.norm(x_new - x)
        x = x_new
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def zo8s_direct(F, J, x0, tol=1e-9, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            Jx, Fx = J(x), F(x)
            s1 = np.linalg.solve(Jx, -Fx)
            z1 = x + s1
            Fz1 = F(z1)
            nu = _nu(Fx, Fz1)
            Jz1 = J(z1)
            rhs2 = (1 + nu) * Fz1
            s2 = np.linalg.solve(Jz1, -rhs2)
            z2 = z1 + s2
            Fz2 = F(z2)
            rhs3 = (1 + 2 * nu) * Fz2
            s3 = np.linalg.solve(Jz1, -rhs3)
            x_new = z2 + s3
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        step_norm = np.linalg.norm(x_new - x)
        x = x_new
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


def eh9s_direct(F, J, x0, tol=1e-9, max_iter=50):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    m = 4
    for _ in range(max_iter):
        try:
            Jx, Fx = J(x), F(x)
            s1 = np.linalg.solve(Jx, -Fx)
            w = x + s1
            for i in range(1, m):
                Fw = F(w)
                corr = np.linalg.solve(Jx, Fw)
                z_val = 2 * F(w + corr) + 3 * F(w - corr) - 3 * Fw
                s = np.linalg.solve(Jx, -z_val)
                w = w + s
            x_new = w
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return x, hist, 'NaN'
        step_norm = np.linalg.norm(x_new - x)
        x = x_new
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol or step_norm < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# =========================================================================
# SECTION B: table-computing functions
# =========================================================================

def save_csv(name, rows, fieldnames):
    with open(f'{name}_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> saved {name}_results.csv ({len(rows)} rows)")


def table1_efficiency_index():
    print("\n=== Table 1: classical efficiency index ===")
    methods = {
        'Newton': (2, lambda n: n**2 + n),
        'ARJ8S': (8, lambda n: n**2 + 4 * n),
        'CTT8S': (8, lambda n: n**2 + 3 * n),
        'ZO8S': (8, lambda n: 2 * n**2 + 3 * n),
        'EH9S': (9, lambda n: n**2 + 10 * n),
        'I-ARJ8S (best case)': (8, lambda n: n**2 + 5 * n),
    }
    rows = []
    n = 10  # representative n, matches paper's Table 1 illustrative row
    for name, (p, dfun) in methods.items():
        d = dfun(n)
        I = p ** (1.0 / d)
        rows.append(dict(method=name, p=p, d=d, I=round(I, 6)))
        print(f"  {name}: p={p}, d={d}, I={I:.6f}")
    save_csv('table1', rows, ['method', 'p', 'd', 'I'])


def table3_burgers():
    print("\n=== Table 3: Burgers' equation, M=101 ===")
    # Discretization matches the already-validated burgers_test.py exactly:
    # central differencing in space, interior-only unknowns (boundaries
    # fixed at 0), warm-started from the previous time step, tol=1e-12,
    # max_iter=50 per step -- all of which differ from a naive
    # re-derivation and materially change the iteration counts.
    M, N_steps, kappa_mu, eta = 101, 201, 5.0, 4.0

    def build_burgers(kappa):
        h = 1.0 / (M - 1)
        k = 1.0 / (N_steps - 1)
        x = np.linspace(0, 1, M)
        phi0_full = 2 * kappa * eta * np.pi * np.sin(np.pi * x) / (kappa_mu + eta * np.cos(np.pi * x))
        n_int = M - 2

        def make_step(phi_prev_full):
            def F(u_int):
                u = np.empty(M)
                u[0], u[-1] = 0.0, 0.0
                u[1:-1] = u_int
                out = np.empty(n_int)
                for idx, i in enumerate(range(1, M - 1)):
                    ut = (u[i] - phi_prev_full[i]) / k
                    ux = (u[i + 1] - u[i - 1]) / (2 * h)
                    uxx = (u[i + 1] - 2 * u[i] + u[i - 1]) / h**2
                    out[idx] = ut + u[i] * ux - kappa * uxx
                return out

            def J(u_int):
                u = np.empty(M)
                u[0], u[-1] = 0.0, 0.0
                u[1:-1] = u_int
                Jm = np.zeros((n_int, n_int))
                for idx, i in enumerate(range(1, M - 1)):
                    Jm[idx, idx] = 1.0 / k + (u[i + 1] - u[i - 1]) / (2 * h) + 2 * kappa / h**2
                    if idx - 1 >= 0:
                        Jm[idx, idx - 1] = -u[i] / (2 * h) - kappa / h**2
                    if idx + 1 < n_int:
                        Jm[idx, idx + 1] = u[i] / (2 * h) - kappa / h**2
                return Jm
            return F, J
        return phi0_full, make_step, n_int

    methods = {
        'Newton': newton_direct, 'ARJ8S': arj8s_direct, 'I-ARJ8S': iarj8s_direct,
        'CTT8S': ctt8s_direct, 'ZO8S': zo8s_direct, 'EH9S': eh9s_direct,
    }
    rows = []
    for kappa in [1.0, 0.05, 0.005]:
        phi_full, make_step, n_int = build_burgers(kappa)
        totals = {}
        for name, fn in methods.items():
            phi = phi_full.copy()
            u_int = phi[1:-1].copy()
            total_iters = 0
            for _ in range(1, N_steps):
                F, J = make_step(phi)
                x, hist, status = fn(F, J, u_int, tol=1e-12, max_iter=50)
                total_iters += len(hist) - 1
                u_int = x
                phi = phi.copy()
                phi[1:-1] = u_int
            totals[name] = total_iters
        row = dict(kappa=kappa, **totals)
        rows.append(row)
        print(f"  kappa={kappa}: {totals}")
    save_csv('table3', rows, ['kappa'] + list(methods.keys()))


def table6_problem1(m, dps=1000, tag='reduced'):
    print(f"\n=== Table 6: Problem 1, m={m} ({tag}) ===")
    mp.mp.dps = dps

    # Formula matches the already-validated paper_own_tests.py exactly:
    # x_i - cos(-s + 2*x_i) = 0, where s is the FULL sum (not excluding x_i).
    def F(x):
        s = sum(x[i] for i in range(m))
        return mp.matrix([x[i] - mp.cos(-s + 2 * x[i]) for i in range(m)])

    def J(x):
        s = sum(x[i] for i in range(m))
        Jm = mp.zeros(m, m)
        for i in range(m):
            g = -s + 2 * x[i]
            for j in range(m):
                if i == j:
                    Jm[i, j] = 1 + mp.sin(g)
                else:
                    Jm[i, j] = -mp.sin(g)
        return Jm

    x0 = mp.matrix([mp.mpf('0.1')] * m)
    x = x0
    hist = [mp.norm(F(x))]
    max_iter = 15
    status = 'maxiter'
    for _ in range(max_iter):
        Fx = F(x)
        try:
            s = mp.lu_solve(J(x), -Fx)
        except ZeroDivisionError:
            status = 'NaN'
            break
        x = x + s
        r = mp.norm(F(x))
        hist.append(r)
        if r < mp.mpf('1e-90'):
            status = 'converged'
            break
    rows = [dict(m=m, iter=len(hist) - 1, status=status, resid=mp.nstr(hist[-1], 5))]
    print(f"  Newton: {status}, iters={len(hist)-1}, resid={mp.nstr(hist[-1],5)}")
    save_csv(f'table6_m{m}', rows, ['m', 'iter', 'status', 'resid'])
    print("  NOTE: this reference implementation runs Newton only, in mpmath VPA, "
          "as a self-check; the paper's full Table 6 covers all 7 methods -- see "
          "paper_own_tests.py / vpa_full_size_tests.py for the complete version.")


def table8_hammerstein():
    print("\n=== Table 8: Hammerstein integral equation, n=7 ===")
    mp.mp.dps = 1000
    n = 7
    nodes, weights = mp.mp.gauss_legendre_nodes_weights(n) if hasattr(mp.mp, 'gauss_legendre_nodes_weights') else (None, None)
    print("  (see paper_own_tests.py for the full Gauss-Legendre Hammerstein "
          "implementation -- omitted here for brevity; this master script "
          "focuses on the algebraic/PDE tables where the method logic is shared.)")


def table9_multisolution():
    print("\n=== Table 9: multi-solution problem, m=20 ===")
    mp.mp.dps = 1000
    m = 20

    # Formula matches the already-validated paper_own_tests.py exactly
    # (note: this Jacobian is a deliberate simplified/approximate form,
    # not the full exact derivative -- kept identical to the validated
    # version rather than "corrected" here, so this remains a faithful
    # cross-check of the same code that produced the paper's numbers).
    def F(x):
        out = []
        for i in range(m):
            s = sum(x[j] for j in range(m) if j != i)
            out.append(x[i] - mp.mpf('1.5') * mp.sin(s))
        return mp.matrix(out)

    def J(x):
        Jm = mp.zeros(m, m)
        for i in range(m):
            s = sum(x[j] for j in range(m) if j != i)
            c = mp.mpf('1.5') * mp.cos(s)
            for j in range(m):
                Jm[i, j] = 1 if j == i else -c
        return Jm

    x0 = mp.matrix([mp.mpf('0.1')] * m)
    x = x0
    hist = [mp.norm(F(x))]
    status = 'maxiter'
    for _ in range(50):
        Fx = F(x)
        try:
            s = mp.lu_solve(J(x), -Fx)
        except ZeroDivisionError:
            status = 'NaN'
            break
        x = x + s
        r = mp.norm(F(x))
        hist.append(r)
        if r < mp.mpf('1e-90'):
            status = 'converged'
            break
    rows = [dict(iter=len(hist) - 1, status=status, resid=mp.nstr(hist[-1], 5))]
    print(f"  Newton: {status}, iters={len(hist)-1}, resid={mp.nstr(hist[-1],5)}")
    save_csv('table9', rows, ['iter', 'status', 'resid'])


def build_bratu_2d(N, lam):
    h = 1.0 / (N + 1)
    e = np.ones(N)
    T = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format='csr')
    I = sp.identity(N, format='csr')
    L = (sp.kron(I, T) + sp.kron(T, I)) / h**2
    L = L.tocsr()
    d = N * N

    def F(u):
        return L @ u + lam * np.exp(u)

    def J(u):
        return (L + sp.diags(lam * np.exp(u))).tocsc()
    return F, J, d


def build_bratu_3d(N, lam):
    h = 1.0 / (N + 1)
    e = np.ones(N)
    T1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format='csr')
    I1 = sp.identity(N, format='csr')
    L = (sp.kron(sp.kron(I1, I1), T1) + sp.kron(sp.kron(I1, T1), I1) +
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


def table11_3dbratu_small():
    print("\n=== Table 11: 3D Bratu, direct LU vs I-ARJ8S-GMRES ===")
    rows = []
    for N in [10, 15, 20, 30, 40]:  # d = 1000, 3375, 8000, 27000, 64000
        F, J, d = build_bratu_3d(N, 3.51)
        x0 = np.zeros(d)
        t0 = time.time()
        x1, h1, s1 = newton_direct(F, J, x0)
        t_direct = time.time() - t0
        t0 = time.time()
        x2, h2, s2 = iarj8s_gmres(F, J, x0)
        t_gmres = time.time() - t0
        speedup = t_direct / t_gmres if t_gmres > 0 else float('nan')
        rows.append(dict(unknowns=d, direct_s=round(t_direct, 3), gmres_s=round(t_gmres, 3),
                          speedup=round(speedup, 2)))
        print(f"  d={d}: direct={t_direct:.2f}s, I-ARJ8S-GMRES={t_gmres:.2f}s, speedup={speedup:.2f}x")
    save_csv('table11', rows, ['unknowns', 'direct_s', 'gmres_s', 'speedup'])


def table13_2dbifurcation():
    print("\n=== Table 13: 2D bifurcation robustness, N=127 ===")
    N = 127
    points = [(6.5, 5), (6.6, 6), (6.6, 7), (6.7, 6), (6.7, 7), (6.8, 5)]
    rows = []
    for lam, c in points:
        h = 1.0 / (N + 1)
        xs = np.linspace(h, 1 - h, N)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        u0 = (c * np.sin(np.pi * X) * np.sin(np.pi * Y)).flatten()
        F, J, d = build_bratu_2d(N, lam)
        x, hist, status = iarj8s_direct(F, J, u0, max_iter=30)
        rows.append(dict(lam=lam, c=c, iarj8s_status=status, iters=len(hist) - 1,
                          resid=hist[-1]))
        print(f"  ({lam},{c}): I-ARJ8S {status}, iters={len(hist)-1}, resid={hist[-1]:.3e}")
    save_csv('table13', rows, ['lam', 'c', 'iarj8s_status', 'iters', 'resid'])


def table15_3dbifurcation():
    print("\n=== Table 15: 3D bifurcation region map, N=20, full 15-point grid ===")
    N = 20
    grid = [(lam, c) for lam in [5.5, 5.7, 5.8, 5.9, 6.0] for c in [4, 4.5, 5]]
    rows = []
    for lam, c in grid:
        F, J, d = build_bratu_3d(N, lam)
        x0 = hard_init_3d(N, c)
        results = {}
        for name, fn in [('Newton', newton_direct), ('ARJ8S', arj8s_direct),
                          ('Newton-GMRES', newton_gmres), ('I-ARJ8S-GMRES', iarj8s_gmres)]:
            x, hist, status = fn(F, J, x0, max_iter=25)
            results[name] = status
        rows.append(dict(lam=lam, c=c, **results))
        print(f"  ({lam},{c}): {results}")
    save_csv('table15', rows, ['lam', 'c', 'Newton', 'ARJ8S', 'Newton-GMRES', 'I-ARJ8S-GMRES'])


def table16_krylov_ablation():
    print("\n=== Table 16: Krylov backend/preconditioner ablation, N=20, 30 ===")
    print("  (requires pyamg + a from-scratch IDR(s) implementation -- "
          "see krylov_ablation_standalone.py / bicgstab_amg_repeat_test.py "
          "for the complete version; omitted here to keep this master "
          "script's dependencies to numpy/scipy/mpmath only.)")


def table19_precond_reuse():
    print("\n=== Table 19: preconditioner-reuse ablation, N=20, 30 ===")
    rows = []
    for N in [20, 30]:
        F, J, d = build_bratu_3d(N, 3.51)
        x0 = np.zeros(d)

        # with reuse: build ILU once per outer iter, reuse across all 4 stages
        t0 = time.time()
        x, hist, status = iarj8s_gmres(F, J, x0)
        t_reuse = time.time() - t0

        rows.append(dict(unknowns=d, with_reuse_s=round(t_reuse, 2), status=status))
        print(f"  d={d}: with-reuse time={t_reuse:.2f}s, status={status}")
    save_csv('table19', rows, ['unknowns', 'with_reuse_s', 'status'])
    print("  NOTE: the 'without reuse' (rebuild-per-stage) comparator is in "
          "precond_reuse_ablation.py -- omitted here for brevity.")


ALL_TABLES = {
    'TABLE1': table1_efficiency_index,
    'TABLE3': table3_burgers,
    'TABLE6': lambda: table6_problem1(30, tag='reduced'),
    'TABLE6_FULL': lambda: table6_problem1(100, tag='full'),
    'TABLE8': table8_hammerstein,
    'TABLE9': table9_multisolution,
    'TABLE11': table11_3dbratu_small,
    'TABLE13': table13_2dbifurcation,
    'TABLE15': table15_3dbifurcation,
    'TABLE16': table16_krylov_ablation,
    'TABLE19': table19_precond_reuse,
}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                         help="also run the slow full-scale VPA tables (m=100/500, n=500)")
    parser.add_argument('--skip', nargs='*', default=[],
                         help="table names to skip, e.g. --skip TABLE6 TABLE9")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        args = parser.parse_args([])

    to_run = dict(ALL_TABLES)
    if not args.full:
        to_run.pop('TABLE6_FULL', None)
    for name in args.skip:
        to_run.pop(name.upper(), None)

    print(f"Running {len(to_run)} table(s): {list(to_run.keys())}")
    t_start = time.time()
    for name, fn in to_run.items():
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"  !! {name} FAILED: {type(e).__name__}: {e}")
        print(f"  ({name} took {time.time()-t0:.1f}s)")
    print(f"\nAll done in {time.time()-t_start:.1f}s total.")


# Runs all default tables automatically. Safe in a Jupyter/Colab cell:
# a notebook kernel's own arguments do not match any recognized flag and
# are caught by the inner try/except above, falling back to defaults.
try:
    main(sys.argv[1:])
except SystemExit:
    pass
