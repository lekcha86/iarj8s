"""
Reference implementations of:
  - Classical Newton
  - ARJ8S (Super-Ostrowski, m=4 steps, Algorithm 1 in Cordero/Rojas-Hiciano/Torregrosa,
    Appl. Numer. Math. 229 (2026) 209-236)
  - ARJ8S-Armijo (Algorithm 2, multi-stage backtracking globalization)

These are built directly from the paper's algorithm statements, independent of the
(fabricated) earlier draft. Used to (a) verify we reproduce order-8 convergence and
(b) serve as the base to extend with inertial + non-monotone line search + GMRES.
"""
import numpy as np


def newton(F, J, x0, tol=1e-13, max_iter=100):
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        s = np.linalg.solve(Jx, -Fx)
        x = x + s
        hist.append(x.copy())
        if np.linalg.norm(x - hist[-2]) < tol or np.linalg.norm(F(x)) < tol:
            break
    return x, hist


def arj8s(F, J, x0, tol=1e-13, max_iter=100):
    """Super-Ostrowski with m=4 steps => order 8, exactly Algorithm 1 of the paper."""
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for k in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        s1 = np.linalg.solve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        alpha = 1.0 - 2.0 * (Fk @ Fz1) / (Fk @ Fk)
        z_prev = z1
        Fz_prev = Fz1
        for i in range(2, 5):  # i = 2,3,4 -> produces z2,z3,z4
            s_i = np.linalg.solve(alpha * Jk, -Fz_prev)
            z_i = z_prev + s_i
            Fz_i = F(z_i)
            z_prev, Fz_prev = z_i, Fz_i
        x_new = z_prev
        hist.append(x_new.copy())
        err_approx = np.linalg.norm(x_new - x)
        err_resid = np.linalg.norm(F(x_new))
        x = x_new
        if err_approx < tol or err_resid < tol:
            break
    return x, hist


def arj8s_armijo(F, J, x0, tol=1e-13, max_iter=100, beta=0.5, gamma=1e-4, L_max=30):
    """Algorithm 2: multi-stage Armijo backtracking globalization of ARJ8S."""
    u = np.array(x0, dtype=float)
    hist = [u.copy()]
    for k in range(max_iter):
        Ju = J(u)
        Fu = F(u)
        # Stage 1
        s1 = np.linalg.solve(Ju, -Fu)
        d1 = 1.0
        for _ in range(L_max):
            if np.linalg.norm(F(u + d1 * s1)) <= (1 - gamma * d1) * np.linalg.norm(Fu):
                break
            d1 *= beta
        y = u + d1 * s1
        Fy = F(y)

        # Stage 2
        denom = (Fu @ Fu)
        alpha_inv = 1.0 / (1.0 - 2.0 * (Fu @ Fy) / denom)
        s2 = -alpha_inv * np.linalg.solve(Ju, Fy)
        d2 = 1.0
        for _ in range(L_max):
            if np.linalg.norm(F(y + d2 * s2)) <= (1 - gamma * d2) * np.linalg.norm(Fy):
                break
            d2 *= beta
        z1 = y + d2 * s2
        Fz1 = F(z1)

        # Stage 3
        s3 = -alpha_inv * np.linalg.solve(Ju, Fz1)
        d3 = 1.0
        for _ in range(L_max):
            if np.linalg.norm(F(z1 + d3 * s3)) <= (1 - gamma * d3) * np.linalg.norm(Fz1):
                break
            d3 *= beta
        z2 = z1 + d3 * s3
        Fz2 = F(z2)

        # Stage 4
        s4 = -alpha_inv * np.linalg.solve(Ju, Fz2)
        d4 = 1.0
        for _ in range(L_max):
            if np.linalg.norm(F(z2 + d4 * s4)) <= (1 - gamma * d4) * np.linalg.norm(Fz2):
                break
            d4 *= beta
        u_new = z2 + d4 * s4

        hist.append(u_new.copy())
        err_approx = np.linalg.norm(u_new - u)
        err_resid = np.linalg.norm(F(u_new))
        u = u_new
        if err_approx < tol or err_resid < tol:
            break
    return u, hist


def acoc(hist, x_star):
    """Approximate computational order of convergence using true errors."""
    errs = [np.linalg.norm(h - x_star) for h in hist]
    orders = []
    for k in range(2, len(errs) - 1):
        if errs[k] > 0 and errs[k - 1] > 0 and errs[k + 1] > 0:
            num = np.log(errs[k + 1] / errs[k])
            den = np.log(errs[k] / errs[k - 1])
            if den != 0:
                orders.append(num / den)
    return orders
