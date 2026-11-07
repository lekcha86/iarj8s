"""
Standalone 3D Bratu scaling test: direct sparse LU (Newton) vs I-ARJ8S-GMRES.
Self-contained -- no dependency on other project files, just numpy + scipy.

Usage:
    python3 bratu3d_scaling.py                  # default sizes: 40 50 60 80
    python3 bratu3d_scaling.py 40 60 80 100 120  # custom N values (N^3 unknowns)

Requires: numpy, scipy  (pip install numpy scipy)

Notes for running on your own machine:
  - Memory: 3D Bratu at N=100 (1,000,000 unknowns) needs several GB of RAM for
    the direct sparse LU factorization (3D fill-in is the whole point of this
    experiment -- it's expected to get large). If you hit MemoryError, either
    lower the largest N you pass, or skip the direct-LU comparison for the
    largest sizes (see SKIP_DIRECT_ABOVE below).
  - Time: direct LU time grows much faster than GMRES time as N increases;
    expect the largest direct-LU runs to take minutes. GMRES stays fast.
  - Results are written incrementally to bratu3d_scaling_results.csv as each
    size finishes, so a crash or Ctrl-C at a large N doesn't lose earlier data.
"""
import sys
import time
import csv
import resource
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# If direct LU becomes infeasible (memory/time) above this N, skip it and
# only report GMRES for larger sizes. Set to None to always attempt both.
SKIP_DIRECT_ABOVE = None   # e.g. set to 80 if direct LU starts OOM-ing for you
LAMBDA = 3.51               # regular (non-bifurcation) regime, matches base paper
TOL = 1e-9
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


def newton_direct(F, J, x0, tol=TOL, max_iter=30):
    x = np.array(x0, dtype=float)
    hist = [np.linalg.norm(F(x))]
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        s = spla.spsolve(Jx, -Fx)
        x = x + s
        hist.append(np.linalg.norm(F(x)))
        if hist[-1] < tol:
            break
    return x, hist


# --- I-ARJ8S core (inertial + non-monotone line search + GMRES), self-contained ---

def _merit(Fv):
    return float(Fv @ Fv)


def _nonmonotone_ls(F, base_pt, Fbase, direction, window, gamma=1e-4, beta=0.5,
                     xi_min=1e-10, max_bt=40):
    f_base = _merit(Fbase)
    M_k = max(window) if window else f_base
    xi = 1.0
    for _ in range(max_bt):
        trial = base_pt + xi * direction
        Ftrial = F(trial)
        f_trial = _merit(Ftrial)
        if f_trial <= M_k - 2.0 * gamma * xi * f_base or xi < xi_min:
            return trial, Ftrial, xi
        xi *= beta
    return trial, Ftrial, xi


def _solve_linear(Jk, rhs, use_gmres, precond, gmres_tol):
    if not use_gmres:
        return spla.spsolve(Jk, rhs)
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0, restart=50, maxiter=200)
    return x


def _stage_step(F, Jk, base_pt, Fbase, use_gmres, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    """Adaptive forcing term per stage; see project notes for why this matters:
    a uniform loose tolerance across all 4 stages was found to stall the method
    in hard regimes, traced to stage 1's error propagating into stages 2-4."""
    f_incoming = _merit(Fbase)
    eta = eta0
    for attempt in range(max_retries):
        direction = _solve_linear(Jk, -Fbase, use_gmres, precond, eta)
        trial, Ftrial, xi = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    direction = _solve_linear(Jk, -Fbase, False, None, 0.0)
    trial, Ftrial, xi = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s_gmres(F, J, x0, tol=TOL, max_iter=30,
                  theta_bar=0.9, window_size=5,
                  gmres_tol_base=0.1, gmres_tol_stage1=1e-6):
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist_resid = [np.linalg.norm(F(x))]
    window = []
    t_prev = 1.0

    for k in range(max_iter):
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

        x_prev, x = x, z_prev
        resid = np.linalg.norm(Fz_prev)
        hist_resid.append(resid)
        if resid < tol:
            break

    return x, hist_resid


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_case(N, lam=LAMBDA, skip_direct=False):
    F, J, d = build_bratu_3d(N, lam)
    x0 = np.zeros(d)
    print(f"\n=== 3D Bratu N={N} ({N}^3={d:,} unknowns), lambda={lam} ===", flush=True)

    result = dict(N=N, d=d, tN=None, memN=None, residN=None,
                   tI=None, memI=None, residI=None, agreement=None)

    xN = None
    if not skip_direct:
        try:
            t0 = time.time()
            xN, histN = newton_direct(F, J, x0)
            result['tN'] = time.time() - t0
            result['memN'] = peak_mem_mb()
            result['residN'] = histN[-1]
            print(f"Newton (direct sparse LU):  iters={len(histN)-1:2d}, "
                  f"time={result['tN']:8.3f}s, resid={result['residN']:.3e}, "
                  f"peak_mem={result['memN']:.0f}MB", flush=True)
        except MemoryError:
            print("Newton (direct sparse LU):  MemoryError -- skipped", flush=True)
    else:
        print("Newton (direct sparse LU):  skipped (SKIP_DIRECT_ABOVE)", flush=True)

    t0 = time.time()
    xI, histI = iarj8s_gmres(F, J, x0)
    result['tI'] = time.time() - t0
    result['memI'] = peak_mem_mb()
    result['residI'] = histI[-1]
    print(f"I-ARJ8S (GMRES+ILU):        iters={len(histI)-1:2d}, "
          f"time={result['tI']:8.3f}s, resid={result['residI']:.3e}, "
          f"peak_mem={result['memI']:.0f}MB", flush=True)

    if xN is not None:
        result['agreement'] = float(np.linalg.norm(xN - xI))
        print(f"Agreement ||x_Newton - x_I-ARJ8S|| = {result['agreement']:.3e}", flush=True)
        if result['tI']:
            print(f"Speedup (direct/GMRES): {result['tN']/result['tI']:.2f}x", flush=True)

    return result


def main():
    Ns = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [40, 50, 60, 80]
    results = []
    csv_path = "bratu3d_scaling_results.csv"
    fieldnames = ['N', 'd', 'tN', 'memN', 'residN', 'tI', 'memI', 'residI', 'agreement']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for N in Ns:
        skip = SKIP_DIRECT_ABOVE is not None and N > SKIP_DIRECT_ABOVE
        r = run_case(N, skip_direct=skip)
        results.append(r)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(r)

    print("\n\n=== SUMMARY ===")
    print(f"{'N':>5} {'unknowns':>10} {'direct(s)':>10} {'GMRES(s)':>10} {'speedup':>8}")
    for r in results:
        tN = f"{r['tN']:.3f}" if r['tN'] is not None else "skipped"
        speedup = f"{r['tN']/r['tI']:.2f}x" if (r['tN'] and r['tI']) else "n/a"
        print(f"{r['N']:>5} {r['d']:>10,} {tN:>10} {r['tI']:>10.3f} {speedup:>8}")
    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
