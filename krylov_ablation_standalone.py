"""
Standalone Krylov backend x preconditioner ablation for I-ARJ8S on 3D Bratu.
Self-contained -- only needs numpy, scipy, and pyamg (pip install pyamg).

This is the slow one: TFQMR in particular can take minutes per run once N
grows past ~30, and the whole 4-backend x 2-preconditioner sweep multiplies
that by 8 configurations per size. Meant to be run on your own machine,
beyond what the sandbox that built this project could push it to (we only
managed N=20 and N=30 there before TFQMR alone was eating minutes).

Usage:
    python3 krylov_ablation_standalone.py                # default: N=20 30
    python3 krylov_ablation_standalone.py 20 30 40 50     # custom N values

Notes:
  - Set SKIP_TFQMR_ABOVE below if TFQMR becomes impractically slow at your
    target N -- it was already the clear worst performer at every size we
    tested (both slower AND converging to a worse residual), so skipping it
    at large N loses little information and saves the most time by far.
  - Results are written incrementally to krylov_ablation_results.csv as each
    (N, backend, preconditioner) triple finishes, so Ctrl-C or a crash
    partway through does not lose earlier data.
  - IDR(s) here is a from-scratch implementation (not in SciPy), verified
    against a direct sparse solve before being trusted in the nonlinear
    solver -- see the idrs() self-test at the bottom of this file
    (run `python3 krylov_ablation_standalone.py --selftest` to check it
    on your own machine before trusting the main results).
"""
import sys
import time
import csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pyamg
    HAVE_PYAMG = True
except ImportError:
    HAVE_PYAMG = False
    print("WARNING: pyamg not installed (pip install pyamg). "
          "AMG-preconditioned configurations will be skipped.", file=sys.stderr)

# ---------------------------------------------------------------------------
LAMBDA = 3.51
TOL = 1e-9
MAX_ITER = 30
SKIP_TFQMR_ABOVE = None   # e.g. set to 40 if TFQMR gets too slow for your patience
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


# --- IDR(s): Sonneveld & van Gijzen, SIAM J. Sci. Comput. 31(2) (2008) 1035-1062 ---
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


def selftest():
    rng = np.random.default_rng(0)
    n = 200
    A = sp.random(n, n, density=0.05, random_state=1) + sp.identity(n) * 10
    A = A.tocsr()
    x_true = rng.standard_normal(n)
    b = A @ x_true
    x_idrs, info = idrs(A, b, s=4, tol=1e-10, maxiter=500)
    x_direct = spla.spsolve(A, b)
    print("IDR(s) vs direct solve, ||diff|| =", np.linalg.norm(x_idrs - x_direct))
    print("IDR(s) vs true x,      ||diff|| =", np.linalg.norm(x_idrs - x_true))
    print("info =", info, "(0 = converged to tol)")


# --- I-ARJ8S core (current, fixed algorithm) with pluggable Krylov backend ---
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


def iarj8s(F, J, x0, backend='gmres', precond_kind='ilu', tol=TOL, max_iter=MAX_ITER,
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


def run_size(N, writer, csvfile):
    F, J, d = build_bratu_3d(N, LAMBDA)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({d:,} unknowns) ===", flush=True)

    configs = [('gmres', 'ilu'), ('bicgstab', 'ilu'), ('tfqmr', 'ilu'), ('idrs', 'ilu')]
    if HAVE_PYAMG:
        configs += [('gmres', 'amg'), ('bicgstab', 'amg'), ('tfqmr', 'amg'), ('idrs', 'amg')]

    for backend, prec in configs:
        if backend == 'tfqmr' and SKIP_TFQMR_ABOVE is not None and N > SKIP_TFQMR_ABOVE:
            print(f"{backend:>10} + {prec:<4}: skipped (N > SKIP_TFQMR_ABOVE)", flush=True)
            continue
        t0 = time.time()
        x, hist, status = iarj8s(F, J, x0, backend=backend, precond_kind=prec)
        elapsed = time.time() - t0
        print(f"{backend:>10} + {prec:<4}: {status:>10}, iters={len(hist)-1:2d}, "
              f"time={elapsed:8.3f}s, resid={hist[-1]:.3e}", flush=True)
        writer.writerow(dict(N=N, d=d, backend=backend, precond=prec, status=status,
                              iters=len(hist) - 1, time=elapsed, resid=hist[-1]))
        csvfile.flush()


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    Ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [20, 30]

    fieldnames = ['N', 'd', 'backend', 'precond', 'status', 'iters', 'time', 'resid']
    with open('krylov_ablation_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for N in Ns:
            run_size(N, writer, csvfile)

    print("\nFull results saved to krylov_ablation_results.csv")


if __name__ == "__main__":
    main()
