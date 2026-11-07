import numpy as np
from iarj8s_inertial import iarj8s
from base_methods import arj8s, acoc

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
x0 = np.full(m, 1.5)

xI, histI = iarj8s(F, J, x0, tol=1e-14, max_iter=100)
xA, histA = arj8s(F, J, x0, tol=1e-14, max_iter=100)

print("Plain ARJ8S:  iters=%d, final resid=%.3e" % (len(histA)-1, np.linalg.norm(F(xA))))
print("I-ARJ8S:      iters=%d, final resid=%.3e" % (len(histI)-1, np.linalg.norm(F(xI))))
print("I-ARJ8S final error vs x_star:", np.linalg.norm(xI - x_star))

# Harder start (farther from root) to get multiple ACOC data points at double precision
x0_far = np.full(m, 2.2)
xI2, histI2 = iarj8s(F, J, x0_far, tol=1e-15, max_iter=100)
print("\nFarther start x0=2.2:")
for i, h in enumerate(histI2):
    print(f"  iter {i}: ||F||={np.linalg.norm(F(h)):.3e}, ||err||={np.linalg.norm(h-x_star):.3e}")
print("ACOC I-ARJ8S (double precision, near-machine-eps limited):", acoc(histI2, x_star))
