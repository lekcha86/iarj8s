"""
Standalone, self-contained script investigating the BiCGSTAB+AMG ~15x
cross-machine discrepancy found earlier in this project (Section 7.9 of
the paper): at N=30 (27,000 unknowns), the sandbox measured 11.82s but a
separate machine measured only 0.76s for the identical configuration.

We since found (Section 7.10) that Google Colab's free tier does not
guarantee the same underlying hardware between separate notebook
sessions, and that I-ARJ8S-GMRES's second outer stage specifically can
show large run-to-run instability even in that setting. This script
tests whether BiCGSTAB+AMG shows the SAME kind of instability WITHIN one
continuous script execution (no session restart in between) -- which
would point away from "different hardware between sessions" and toward
something else (e.g. genuine algorithmic/numerical sensitivity, thermal
throttling, or resource contention). GMRES+AMG and IDR(s)+AMG are also
repeated for comparison, since those were previously the "well-behaved"
configurations.

Also repeats the AMG-hierarchy-determinism check from the sandbox
(build the same preconditioner from the same matrix multiple times, and
check it produces an identical multigrid structure every time) -- this
was previously only verified in the sandbox, never on your own machine.

Requires: numpy, scipy, pyamg (pip install numpy scipy pyamg)

Usage:
    python3 bicgstab_amg_repeat_test.py                # default: N=20 30, 5 repeats each
    python3 bicgstab_amg_repeat_test.py --repeats 8 --sizes 20 30 40
    python3 bicgstab_amg_repeat_test.py --sizes 30 --repeats 10   # focus on the exact case that showed the discrepancy

Notes:
  - Results are written incrementally to
    bicgstab_amg_repeat_results.csv as each (N, backend, trial) finishes.
  - If you are running this in Google Colab specifically, running it
    once, restarting the runtime (Runtime > Restart runtime), and
    running it again is a stronger test of the "different hardware per
    session" hypothesis than repeating within one execution -- consider
    doing both.
"""
import sys
import time
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg

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


# --- IDR(s), Sonneveld & van Gijzen 2008, included inline (not in SciPy) ---
def idrs(A, b, x0=None, s=4, tol=1e-8, maxiter=1000, M=None):
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
    if backend == 'gmres':
        x, info = spla.gmres(Jk, rhs, M=precond, rtol=tol, atol=0.0, restart=50, maxiter=200)
    elif backend == 'bicgstab':
        x, info = spla.bicgstab(Jk, rhs, M=precond, rtol=tol, atol=0.0, maxiter=200)
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
    direction = spla.spsolve(Jk, -Fbase)
    trial, Ftrial = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def make_amg_precond(Jy):
    ml = pyamg.smoothed_aggregation_solver(Jy.tocsr())
    return ml, spla.LinearOperator(Jy.shape, ml.aspreconditioner().matvec)


def iarj8s(F, J, x0, backend, tol=TOL, max_iter=MAX_ITER,
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
        _, precond = make_amg_precond(Jy)

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


def check_amg_determinism(N, trials=3):
    F, J, d = build_bratu_3d(N, LAMBDA)
    x0 = np.zeros(d)
    Jy = J(x0)
    print(f"\n--- AMG hierarchy determinism check, N={N} ---", flush=True)
    results = []
    for t in range(trials):
        ml, _ = make_amg_precond(Jy)
        levels = len(ml.levels)
        complexity = ml.operator_complexity()
        sizes = [lvl.A.shape[0] for lvl in ml.levels]
        results.append((levels, complexity, tuple(sizes)))
        print(f"  trial {t}: levels={levels}, operator_complexity={complexity:.6f}, sizes={sizes}",
              flush=True)
    all_same = len(set(results)) == 1
    print(f"  => {'IDENTICAL every trial (deterministic)' if all_same else 'DIFFERED across trials (non-deterministic!)'}",
          flush=True)
    return all_same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='*', type=int, default=[20, 30])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--backends', nargs='*', default=['bicgstab', 'gmres', 'idrs'],
                         help="which backends to test with AMG (bicgstab is the one under investigation)")
    args = parser.parse_args()

    fieldnames = ['N', 'd', 'backend', 'trial', 'status', 'outer_iters', 'time', 'resid']
    with open('bicgstab_amg_repeat_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for N in args.sizes:
            check_amg_determinism(N)

            F, J, d = build_bratu_3d(N, LAMBDA)
            x0 = np.zeros(d)
            print(f"\n=== 3D Bratu N={N} ({d:,} unknowns), {args.repeats} repeats per backend+AMG ===",
                  flush=True)

            for backend in args.backends:
                times = []
                for trial in range(args.repeats):
                    t0 = time.time()
                    x, hist, status = iarj8s(F, J, x0, backend=backend)
                    elapsed = time.time() - t0
                    times.append(elapsed)
                    print(f"  {backend:>10}+AMG trial {trial}: {status:>10}, "
                          f"iters={len(hist)-1}, time={elapsed:7.3f}s, resid={hist[-1]:.2e}",
                          flush=True)
                    writer.writerow(dict(N=N, d=d, backend=backend, trial=trial,
                                          status=status, outer_iters=len(hist) - 1,
                                          time=round(elapsed, 4), resid=hist[-1]))
                    csvfile.flush()
                spread = max(times) / min(times) if min(times) > 0 else float('nan')
                print(f"  {backend:>10}+AMG: min={min(times):.3f}s max={max(times):.3f}s "
                      f"spread(max/min)={spread:.2f}x", flush=True)

    print("\nAll done. Full results in bicgstab_amg_repeat_results.csv")


if __name__ == "__main__":
    main()
