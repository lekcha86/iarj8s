"""
Standalone, self-contained comparison script: ALL methods used in the paper
(Newton, ARJ8S, CTT8S, ZO8S, EH9S, NLM8, I-ARJ8S) on the 3D Bratu problem,
at sizes large enough to take a long time -- meant to be run on your own
machine, beyond what the sandbox that built this project could handle
(~2.9GB RAM limit there).

I-ARJ8S here is the CURRENT, fixed algorithm (as of this project's round 8/9):
  - safeguarded inertial (FISTA-type) extrapolation
  - non-monotone line search (DF-SANE/Grippo-Lampariello-Lucidi style) with a
    bold-step fallback: if backtracking shrinks to its floor without finding
    a descent step, try the full undamped step instead of a near-zero one,
    rejecting only if it is actually non-finite
  - "best-of-4-stages" tracking: advance from whichever of the 4 Super-Ostrowski
    stages achieved the lowest residual in that outer iteration, not
    unconditionally the last stage's output (this is what fixed the stall
    where stages 2-4, built on the frozen Jacobian, could increase the
    residual once corrections grew large)
  - optional inexact GMRES+ILU linear solves, with an asymmetric forcing-term
    schedule (stage 1 solved much more accurately than stages 2-4)

Requires: numpy, scipy (pip install numpy scipy)

Usage:
    python3 bratu3d_full_comparison.py                  # default sizes: 20 30 40 50
    python3 bratu3d_full_comparison.py 20 30 40 50 60 70 # custom N values (N^3 unknowns)

Notes for running on your own machine:
  - NLM8 needs an explicit dense n x n matrix every iteration (its matrix-
    valued weight). This is fine at small N but becomes impractical fast
    (n=16,129 already needs ~2GB just for one such matrix). NLM8 is
    automatically SKIPPED above NLM8_MAX_N (edit the constant below) --
    this is a genuine architectural limitation of NLM8 at scale, not a bug.
  - CTT8S and EH9S use dense-Jacobian-adjacent formulas too (matrix weights
    or multiple extra function evals) but stay vector-only, so they scale
    like ARJ8S/ZO8S/Newton -- all use sparse LU (spsolve), same as I-ARJ8S's
    exact-solve path.
  - Direct-solve wall-clock times (Newton, ARJ8S, CTT8S, ZO8S, EH9S) vary
    noticeably run-to-run even with no algorithm change at all (~20-30% in
    our own testing) -- this is normal for shared/virtualized environments;
    run the whole sweep in one sitting for the fairest internal comparison.
  - Results are written incrementally to bratu3d_full_comparison.csv as each
    (N, method) pair finishes, so a crash or Ctrl-C partway through does not
    lose earlier data.
"""
import sys
import time
import csv
import resource
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
LAMBDA = 3.51          # regular (non-bifurcation) regime, matches base paper
TOL = 1e-9
MAX_ITER = 30
NLM8_MAX_N = 30         # N^3 unknowns above which NLM8 is skipped (dense O(n^2) memory)
METHODS_TO_SKIP_ABOVE = {}  # e.g. {'NLM8': 30} is already handled via NLM8_MAX_N;
                             # add other method: N-cutoff pairs here if needed
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


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --- Newton -----------------------------------------------------------------
def newton(F, J, x0, tol=TOL, max_iter=MAX_ITER):
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


# --- ARJ8S (Cordero, Rojas-Hiciano, Torregrosa, Appl. Numer. Math. 229 (2026)) ---
def arj8s(F, J, x0, tol=TOL, max_iter=MAX_ITER):
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
            Fzi = F(zi)
            zp, Fzp = zi, Fzi
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- CTT8S (Cordero, Torregrosa, Triguero-Navarro, Appl. Math. Comput. 498, 129401) ---
def ctt8s(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jx = J(x)
        Fx = F(x)
        s1 = spla.spsolve(Jx, -Fx)
        y = x + s1
        Fy = F(y)
        nu = (Fy @ Fy) / (Fx @ Fx)
        rhs = (1 + nu) * Fy + 2 * nu * Fx
        s2 = spla.spsolve(Jx, -rhs)
        z = y + s2
        Fz = F(z)
        gamma = (Fy @ Fz) / (Fx @ Fx)
        beta = (Fz @ Fz) / (Fy @ Fy)
        delta = (Fz @ Fz) / (Fx @ Fx)
        rhs3 = Fz + (2 * gamma * (1 - nu) + 4 * delta) * Fx + (2 * gamma + beta) * Fy
        s3 = spla.spsolve(Jx, -rhs3)
        x = z + s3
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- ZO8S (Zhanlav, Otgondorj, J. Math. Sci. 279 (2024) 866-875) ---
def zo8s(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        s1 = spla.spsolve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        nu = (Fz1 @ Fz1) / (Fk @ Fk)
        Jz1 = J(z1)  # second Jacobian, reused for both remaining stages
        rhs2 = (1 + nu) * Fz1
        s2 = spla.spsolve(Jz1, -rhs2)
        z2 = z1 + s2
        Fz2 = F(z2)
        rhs3 = (1 + 2 * nu) * Fz2
        s3 = spla.spsolve(Jz1, -rhs3)
        x = z2 + s3
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- EH9S (Erfanifar, Hajarian, Numer. Algorithms 97 (2024) 959-984) ---
def eh9s(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    m = 4
    for _ in range(max_iter):
        Jx = J(x)
        Fx = F(x)
        s1 = spla.spsolve(Jx, -Fx)
        w = x + s1
        for i in range(1, m):
            Fw = F(w)
            corr = spla.spsolve(Jx, Fw)
            zval = 2 * F(w + corr) + 3 * F(w - corr) - 3 * Fw
            s = spla.spsolve(Jx, -zval)
            w = w + s
        x = w
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- NLM8 (Sharma & Arora, via Zhanlav/Mijiddorj/Khuder, Hacettepe J. Math. Stat. 52(4) (2023)) ---
def nlm8(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    n = len(x)
    hist = [np.linalg.norm(F(x))]
    I = np.eye(n)
    for _ in range(max_iter):
        Jx = J(x)
        Fx = F(x)
        LU = spla.splu(Jx.tocsc())
        s1 = LU.solve(-Fx)
        y = x + s1
        Fy = F(y)
        Jy = J(y)
        tk = LU.solve(Jy.toarray())  # explicit dense n x n matrix -- the memory bottleneck
        tk2 = tk @ tk
        tau_k = 13 / 4 * I - 7 / 2 * tk + 5 / 4 * tk2
        z = y - tau_k @ LU.solve(Fy)
        Fz = F(z)
        alpha_k = 7 / 2 * I - 4 * tk + 3 / 2 * tk2
        x = z - alpha_k @ LU.solve(Fz)
        hist.append(np.linalg.norm(F(x)))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- I-ARJ8S (current, fixed algorithm -- this project's own method) --------
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


def iarj8s(F, J, x0, tol=TOL, max_iter=MAX_ITER, theta_bar=0.9, window_size=5,
           use_gmres=False, gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
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
        precond = None
        if use_gmres:
            ilu = spla.spilu(sp.csc_matrix(Jy), drop_tol=1e-5, fill_factor=15)
            precond = spla.LinearOperator(Jy.shape, ilu.solve)

        gmres_tol1 = min(gmres_tol_stage1, np.linalg.norm(Fy))
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, use_gmres, precond, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step(F, alpha * Jy, z_prev, Fz_prev, use_gmres, precond,
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


# --- Runner -------------------------------------------------------------
METHODS = [
    ('Newton', newton, {}),
    ('ARJ8S', arj8s, {}),
    ('CTT8S', ctt8s, {}),
    ('ZO8S', zo8s, {}),
    ('EH9S', eh9s, {}),
    ('NLM8', nlm8, {}),
    ('I-ARJ8S-exact', iarj8s, {'use_gmres': False}),
    ('I-ARJ8S-GMRES', iarj8s, {'use_gmres': True}),
]


def run_case(N, lam=LAMBDA):
    F, J, d = build_bratu_3d(N, lam)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({N}^3={d:,} unknowns), lambda={lam} ===", flush=True)

    results = []
    for name, fn, kwargs in METHODS:
        if name == 'NLM8' and N > NLM8_MAX_N:
            print(f"{name:>14}: skipped (N={N} > NLM8_MAX_N={NLM8_MAX_N}, "
                  f"would need a dense {d}x{d} matrix)", flush=True)
            results.append(dict(N=N, d=d, method=name, status='skipped',
                                 iters=None, time=None, resid=None, mem_mb=None))
            continue
        try:
            t0 = time.time()
            x, hist, status = fn(F, J, x0, **kwargs)
            elapsed = time.time() - t0
            mem = peak_mem_mb()
            print(f"{name:>14}: {status:>10}, iters={len(hist)-1:2d}, "
                  f"time={elapsed:8.3f}s, resid={hist[-1]:.3e}, peak_mem={mem:.0f}MB",
                  flush=True)
            results.append(dict(N=N, d=d, method=name, status=status,
                                 iters=len(hist) - 1, time=elapsed,
                                 resid=hist[-1], mem_mb=mem))
        except MemoryError:
            print(f"{name:>14}: MemoryError -- skipped", flush=True)
            results.append(dict(N=N, d=d, method=name, status='MemoryError',
                                 iters=None, time=None, resid=None, mem_mb=None))
    return results


def main():
    Ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [20, 30, 40, 50]
    csv_path = "bratu3d_full_comparison.csv"
    fieldnames = ['N', 'd', 'method', 'status', 'iters', 'time', 'resid', 'mem_mb']
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    all_results = []
    for N in Ns:
        results = run_case(N)
        all_results.extend(results)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for r in results:
                writer.writerow(r)

    print("\n\n=== SUMMARY (time in seconds) ===")
    method_names = [m[0] for m in METHODS]
    header = f"{'N':>6} {'unknowns':>10}" + "".join(f"{m:>16}" for m in method_names)
    print(header)
    for N in Ns:
        row_results = {r['method']: r for r in all_results if r['N'] == N}
        d = N**3
        line = f"{N:>6} {d:>10,}"
        for m in method_names:
            r = row_results.get(m)
            if r is None or r['time'] is None:
                cell = r['status'] if r else 'n/a'
            else:
                cell = f"{r['time']:.2f}s({r['status'][:4]})"
            line += f"{cell:>16}"
        print(line)
    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
