import numpy as np
from base_methods import newton, arj8s, arj8s_armijo, acoc

# Paper's "Problem 2": x_i^2 * x_{i+1} - 1 = 0 (i=1..m-1), x_m^2 * x_1 - 1 = 0
# Exact root: all-ones vector. Clean, well-conditioned, exactly known solution.
m = 30

def F(x):
    out = np.empty(m)
    out[:-1] = x[:-1]**2 * x[1:] - 1
    out[-1] = x[-1]**2 * x[0] - 1
    return out

def J(x):
    Jm = np.zeros((m, m))
    for i in range(m - 1):
        Jm[i, i] = 2 * x[i] * x[i + 1]
        Jm[i, i + 1] = x[i]**2
    Jm[m - 1, m - 1] = 2 * x[m - 1] * x[0]
    Jm[m - 1, 0] = x[m - 1]**2
    return Jm

x_star = np.ones(m)
x0 = np.full(m, 1.5)  # same style initial guess as paper (1.5,...,1.5)

xN, histN = newton(F, J, x0, tol=1e-14, max_iter=100)
xA, histA = arj8s(F, J, x0, tol=1e-14, max_iter=100)
xG, histG = arj8s_armijo(F, J, x0, tol=1e-14, max_iter=100)

print("Newton: iters=%d, final resid=%.3e, final err=%.3e" %
      (len(histN)-1, np.linalg.norm(F(xN)), np.linalg.norm(xN - x_star)))
print("ACOC Newton:", [f"{v:.3f}" for v in acoc(histN, x_star)])

print("\nARJ8S: iters=%d, final resid=%.3e, final err=%.3e" %
      (len(histA)-1, np.linalg.norm(F(xA)), np.linalg.norm(xA - x_star)))
print("ACOC ARJ8S:", [f"{v:.3f}" for v in acoc(histA, x_star)])

print("\nARJ8S-Armijo: iters=%d, final resid=%.3e, final err=%.3e" %
      (len(histG)-1, np.linalg.norm(F(xG)), np.linalg.norm(xG - x_star)))
print("ACOC ARJ8S-Armijo:", [f"{v:.3f}" for v in acoc(histG, x_star)])
