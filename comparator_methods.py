"""
CTT8S, ZO8S, EH9S: the three comparator methods used in the base ARJ8S paper
itself (Cordero, Rojas-Hiciano, Torregrosa, Appl. Numer. Math. 229 (2026)
209-236, Section 2.2-2.3), implemented directly from that paper's stated
formulas (Eqs. 7-10 and the ZO8S/EH9S recursions). Real, published methods,
not invented -- gives us the same comparator set the original authors used.
"""
import numpy as np


def _nu(Fx, Fy):
    return (Fy @ Fy) / (Fx @ Fx)


def c4s_psi(F, J, x, psi, Jx=None, Fx=None):
    """C4S_psi: one step, order 4. Returns (z, Fz, y, Fy, Jx, Fx, nu)."""
    if Jx is None:
        Jx = J(x)
    if Fx is None:
        Fx = F(x)
    s1 = np.linalg.solve(Jx, -Fx)
    y = x + s1
    Fy = F(y)
    nu = _nu(Fx, Fy)
    rhs = (1 + psi * nu) * Fy + 2 * nu * Fx
    s2 = np.linalg.solve(Jx, -rhs)
    z = y + s2
    Fz = F(z)
    return z, Fz, y, Fy, Jx, Fx, nu


def ctt8s(F, J, x0, tol=1e-13, max_iter=100):
    """CTT8S: Eq. (10) of the base paper, built on C4S1 (psi=1)."""
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for k in range(max_iter):
        z, Fz, y, Fy, Jx, Fx, nu = c4s_psi(F, J, x, psi=1.0)
        gamma = (Fy @ Fz) / (Fx @ Fx)
        beta = (Fz @ Fz) / (Fy @ Fy)
        delta = (Fz @ Fz) / (Fx @ Fx)
        rhs = Fz + (2 * gamma * (1 - nu) + 4 * delta) * Fx + (2 * gamma + beta) * Fy
        s3 = np.linalg.solve(Jx, -rhs)
        x_new = z + s3
        hist.append(x_new.copy())
        err_approx = np.linalg.norm(x_new - x)
        err_resid = np.linalg.norm(F(x_new))
        x = x_new
        if err_approx < tol or err_resid < tol:
            break
    return x, hist


def zo8s(F, J, x0, tol=1e-13, max_iter=100):
    """ZO8S: Zhanlav-Otgondorj recursion with m=3 steps, order 8.
    z1 = Newton step (uses F'(x)). z2 = S5S-type step using F'(z1) (a
    SECOND Jacobian, at the Newton point). The final corrector for z3
    REUSES this same F'(z1) (frozen), not a fresh Jacobian at z2 -- this
    matches the base paper's own cost table (S1=1, S2=2: one system with
    F'(x), two with the single 'other' Jacobian F'(z1))."""
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for k in range(max_iter):
        Jx = J(x)
        Fx = F(x)
        s1 = np.linalg.solve(Jx, -Fx)
        z1 = x + s1
        Fz1 = F(z1)
        nu = _nu(Fx, Fz1)
        Jz1 = J(z1)  # second Jacobian, at the Newton point
        rhs2 = (1 + nu) * Fz1
        s2 = np.linalg.solve(Jz1, -rhs2)
        z2 = z1 + s2
        Fz2 = F(z2)
        rhs3 = (1 + 2 * nu) * Fz2
        s3 = np.linalg.solve(Jz1, -rhs3)  # reuse Jz1, not a fresh J(z2)
        x_new = z2 + s3
        hist.append(x_new.copy())
        err_approx = np.linalg.norm(x_new - x)
        err_resid = np.linalg.norm(F(x_new))
        x = x_new
        if err_approx < tol or err_resid < tol:
            break
    return x, hist


def eh9s(F, J, x0, tol=1e-13, max_iter=100):
    """EH9S: Erfanifar-Hajarian frozen-Jacobian scheme, first 4 steps -> order 9.
    w1 = Newton step; then w_{i+1}=w_i - J^{-1} z_m using a matrix-free
    combination of function evaluations at points offset by +-J^{-1}F(w_i)."""
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    m = 4
    for k in range(max_iter):
        Jx = J(x)
        Fx = F(x)
        s1 = np.linalg.solve(Jx, -Fx)
        w = x + s1
        for i in range(1, m):
            Fw = F(w)
            corr = np.linalg.solve(Jx, Fw)  # J^{-1} F(w_i)
            z_val = 2 * F(w + corr) + 3 * F(w - corr) - 3 * Fw
            s = np.linalg.solve(Jx, -z_val)
            w = w + s
        x_new = w
        hist.append(x_new.copy())
        err_approx = np.linalg.norm(x_new - x)
        err_resid = np.linalg.norm(F(x_new))
        x = x_new
        if err_approx < tol or err_resid < tol:
            break
    return x, hist
