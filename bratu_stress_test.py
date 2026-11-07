import numpy as np
from bratu_test import build_bratu, newton_sparse, arj8s_sparse
from iarj8s_inertial import iarj8s

N = 127
lam = 6.2  # near bifurcation, same regime as base paper's stress test (lambda_c ~ 6.81)
F, J, d = build_bratu(N, lam)
x0 = np.full(d, 2.0)  # "spoiled" constant initial guess, farther from solution than zeros

print(f"Bratu near bifurcation: {d} unknowns, lambda={lam}")

xN, histN = newton_sparse(F, J, x0, tol=1e-9, max_iter=40)
print(f"Newton: iters={len(histN)-1}, final resid={histN[-1]:.3e}, has_nan={np.any(np.isnan(xN))}")
print("Newton residual trajectory:", [f"{r:.3e}" for r in histN])

try:
    xA, histA = arj8s_sparse(F, J, x0, tol=1e-9, max_iter=40)
    print(f"Plain ARJ8S (no globalization): iters={len(histA)-1}, "
          f"final resid={histA[-1]:.3e}, has_nan={np.any(np.isnan(xA))}")
    print("ARJ8S residual trajectory:", [f"{r:.3e}" for r in histA])
except Exception as e:
    print(f"Plain ARJ8S (no globalization): FAILED with {type(e).__name__}: {e}")

xI, histI = iarj8s(F, J, x0, tol=1e-9, max_iter=40, use_gmres=True, gmres_tol_base=0.1)
residI = np.linalg.norm(F(xI))
print(f"I-ARJ8S (inertial+nonmonotone LS+GMRES): iters={len(histI)-1}, "
      f"final resid={residI:.3e}, has_nan={np.any(np.isnan(xI))}")

print(f"\n||x_Newton - x_I-ARJ8S|| = {np.linalg.norm(xN - xI):.3e}")
