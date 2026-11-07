import numpy as np
from base_methods import newton, arj8s, arj8s_armijo, acoc

np.set_printoptions(precision=4, suppress=True)

# Test problem: same family as the paper's "Problem 1"
# x_i - cos(-(sum_j x_j) + 2 x_i) = 0, i=1..m
m = 20

def F(x):
    s = np.sum(x)
    return x - np.cos(-(s) + 2 * x)

def J(x):
    s = np.sum(x)
    n = len(x)
    Jm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Jm[i, j] = 1 + np.sin(-s + 2 * x[i])  # dg_i/dx_i = -1+2 = 1
            else:
                Jm[i, j] = -np.sin(-s + 2 * x[i])  # dg_i/dx_j = -1
    return Jm

# high-precision root via long Newton run from a good guess
x0_root = np.full(m, 0.2)
x_star, hist_star = newton(F, J, x0_root, tol=1e-15, max_iter=200)
print("root (first 5 comps):", x_star[:5])
print("||F(x_star)|| =", np.linalg.norm(F(x_star)))

x0 = np.full(m, 0.1)

xN, histN = newton(F, J, x0, tol=1e-14, max_iter=100)
xA, histA = arj8s(F, J, x0, tol=1e-14, max_iter=100)
xG, histG = arj8s_armijo(F, J, x0, tol=1e-14, max_iter=100)

print("\nNewton: iters=%d, final resid=%.3e" % (len(histN)-1, np.linalg.norm(F(xN))))
print("ACOC Newton:", acoc(histN, x_star))

print("\nARJ8S: iters=%d, final resid=%.3e" % (len(histA)-1, np.linalg.norm(F(xA))))
print("ACOC ARJ8S:", acoc(histA, x_star))

print("\nARJ8S-Armijo: iters=%d, final resid=%.3e" % (len(histG)-1, np.linalg.norm(F(xG))))
print("ACOC ARJ8S-Armijo:", acoc(histG, x_star))
