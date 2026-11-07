import time
import resource
import numpy as np
from bratu_test import build_bratu, newton_sparse, arj8s_sparse
from iarj8s_inertial import iarj8s


def peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_case(N, lam, x0_val):
    F, J, d = build_bratu(N, lam)
    x0 = np.full(d, x0_val)
    print(f"\n=== N={N} ({N}x{N}={d} unknowns), lambda={lam}, x0={x0_val} ===")

    t0 = time.time()
    xA, histA = arj8s_sparse(F, J, x0, tol=1e-9, max_iter=40)
    tA = time.time() - t0
    memA = peak_mem_mb()
    print(f"ARJ8S (direct sparse LU):   iters={len(histA)-1:2d}, time={tA:7.3f}s, "
          f"resid={histA[-1]:.3e}, peak_mem={memA:.0f}MB")

    t0 = time.time()
    xI, histI = iarj8s(F, J, x0, tol=1e-9, max_iter=40, use_gmres=True,
                        gmres_tol_base=0.1, gmres_tol_stage1=1e-6)
    tI = time.time() - t0
    residI = np.linalg.norm(F(xI))
    memI = peak_mem_mb()
    print(f"I-ARJ8S (GMRES+ILU):       iters={len(histI)-1:2d}, time={tI:7.3f}s, "
          f"resid={residI:.3e}, peak_mem={memI:.0f}MB")

    print(f"Agreement ||x_ARJ8S - x_I-ARJ8S|| = {np.linalg.norm(xA - xI):.3e}")
    print(f"Speedup (direct time / GMRES time): {tA/tI:.2f}x")
    return dict(N=N, d=d, tA=tA, tI=tI, memA=memA, memI=memI)


if __name__ == "__main__":
    results = []
    for N in [127, 255, 383, 511]:
        results.append(run_case(N, lam=3.51, x0_val=0.0))

    print("\n\n=== SUMMARY ===")
    print(f"{'N':>5} {'unknowns':>10} {'direct(s)':>10} {'GMRES(s)':>10} {'speedup':>8} {'mem_direct':>11} {'mem_gmres':>10}")
    for r in results:
        print(f"{r['N']:>5} {r['d']:>10} {r['tA']:>10.3f} {r['tI']:>10.3f} "
              f"{r['tA']/r['tI']:>7.2f}x {r['memA']:>10.0f}MB {r['memI']:>9.0f}MB")
