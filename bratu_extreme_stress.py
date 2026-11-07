"""
Reproduces the base ARJ8S paper's own most extreme test (Table 17): 2D
Bratu near the bifurcation threshold (lambda=6.2), at N=1024 (1,048,576
unknowns) and N=2048 (4,194,304 unknowns) -- their own headline claim
("one of the scarce instances in the literature where an iterative scheme
exceeding fourth order solves a more than four-million unknowns problem
under heavily ill-conditioned singular boundaries").

This matches the base paper's own setup as closely as possible: N=1024
and N=2048, lambda=6.2, a physical null initial guess u0=0 (their own
choice for this specific extreme test, "to establish a well-conditioned
baseline at the first iteration").

Requires: numpy, scipy (pip install numpy scipy)

Usage:
    python3 bratu_extreme_stress.py             # runs N=1024, 2048 automatically
    python3 bratu_extreme_stress.py 512 1024     # custom N values instead

Notes:
  - This is a 2D problem, so per this project's own established finding
    (direct sparse LU beats GMRES+ILU in 2D even at hundreds of thousands
    of unknowns, thanks to excellent nested-dissection fill-in), all
    methods here use exact/direct sparse solves, not GMRES -- matching
    both the base paper's own methodology for this test and our own prior
    2D finding.
  - Memory is the practical limit, not CPU: N=2048 means a
    4,194,304 x 4,194,304 sparse Jacobian. Even with excellent 2D
    fill-in, expect multi-GB peak memory and potentially many minutes per
    linear solve at this scale.
  - Known result from a previous run at this exact setup: none of the
    three methods reached tol=1e-9 within 20 iterations at either size,
    plateauing instead around ||F||~1e-6 to 1e-7 -- a real discrepancy
    against the base paper's own published Table 17, which reports both
    their Newton and their globalized ARJ8S-Armijo converging to machine
    precision in 2-4 iterations at this exact (lambda,N,u0). This script
    is unchanged in its numerical logic from that run; it is being
    re-run to confirm the result is reproducible, not because a bug was
    suspected in the mathematics.
  - Results are written incrementally to
    bratu_extreme_stress_results.csv as each (N, method) pair finishes,
    so a crash or interruption partway through does not lose earlier
    results.
"""
import sys
import time
import csv
try:
    import resource
except ImportError:
    resource = None
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LAMBDA = 6.2
TOL = 1e-9
MAX_ITER = 20

# Note from earlier testing: at N=300 (90,000 unknowns), even with u0=0
# (the base paper's own "well-conditioned baseline" choice), all three
# methods plateau near residual ~5e-9, just above tol=1e-9, and this does
# not improve much even with a larger iteration budget. This looks like a
# genuine, mild ill-conditioning effect from lambda=6.2 combined with
# N^2-scaling condition number growth, even without a hard initial guess.
# The same plateau behavior, worse in relative terms, is what this script
# is expected to reproduce at the much larger N=1024/2048 target scale.


def build_bratu(N, lam):
    """N interior points per side -> d = N*N unknowns."""
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


def peak_mem_mb():
    if resource is not None:
        try:
            res = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == 'darwin':
                return res / (1024.0 * 1024.0)
            return res / 1024.0
        except Exception:
            pass
    if psutil is not None:
        try:
            mem = psutil.Process().memory_info()
            peak = getattr(mem, 'peak_wset', None) or getattr(mem, 'rss', 0)
            return peak / (1024.0 * 1024.0)
        except Exception:
            pass
    return 0.0


def newton_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for _ in range(max_iter):
        try:
            Fx = F(x)
            s = spla.spsolve(J(x), -Fx)
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
        try:
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
        except (RuntimeError, ValueError):
            return x, hist, 'NaN'
        x = zp
        hist.append(np.linalg.norm(Fzp))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


# --- I-ARJ8S, exact-solve variant (current, fixed algorithm) ------------
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


def iarj8s_direct(F, J, x0, tol=TOL, max_iter=MAX_ITER, theta_bar=0.9, window_size=5):
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
        except (RuntimeError, ValueError):
            return x, hist, 'NaN'

        x_prev, x = x, best_pt
        hist.append(np.linalg.norm(best_F))
        if not np.all(np.isfinite(x)):
            return x, hist, 'NaN'
        if hist[-1] < tol:
            return x, hist, 'converged'
    return x, hist, 'maxiter'


METHODS = [
    ('Newton', newton_direct),
    ('ARJ8S', arj8s_direct),
    ('I-ARJ8S', iarj8s_direct),
]


def run_size(N, writer, csvfile):
    F, J, d = build_bratu(N, LAMBDA)
    x0 = np.zeros(d)
    print(f"\n=== 2D Bratu N={N} ({d:,} unknowns), lambda={LAMBDA} ===", flush=True)

    for name, fn in METHODS:
        t0 = time.time()
        x, hist, status = fn(F, J, x0)
        elapsed = time.time() - t0
        mem = peak_mem_mb()
        print(f"{name:>10}: {status:>10}, iters={len(hist)-1:2d}, "
              f"time={elapsed:9.2f}s, resid={hist[-1]:.3e}, peak_mem={mem:.0f}MB", flush=True)
        writer.writerow(dict(N=N, d=d, method=name, status=status,
                              iters=len(hist) - 1, time=round(elapsed, 3),
                              resid=hist[-1], mem_mb=round(mem, 1)))
        csvfile.flush()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        Ns = [int(a) for a in argv] if argv else [1024, 2048]
    except ValueError:
        # argv contained something that isn't an integer (e.g. a notebook
        # kernel's own '-f /path/to/kernel.json' argument) -- fall back
        # to the default sizes rather than crash.
        Ns = [1024, 2048]

    fieldnames = ['N', 'd', 'method', 'status', 'iters', 'time', 'resid', 'mem_mb']
    with open('bratu_extreme_stress_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for N in Ns:
            run_size(N, writer, csvfile)

    print("\nAll done. Full results in bratu_extreme_stress_results.csv")


# Runs N=1024, 2048 automatically if no usable arguments are present.
# Safe to run directly in a Jupyter/Colab cell: a notebook kernel's own
# arguments (e.g. '-f /path/to/kernel.json') do not parse as integers,
# so they are detected and ignored below rather than causing a crash;
# real integer arguments (e.g. from a plain terminal) are still honored.
try:
    main(sys.argv[1:])
except SystemExit:
    pass
