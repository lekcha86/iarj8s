"""
NLM8: Sharma & Arora's eighth-order method, as presented in Zhanlav, Mijiddorj,
Khuder, "A family of Newton-type methods with seventh and eighth-order of
convergence for solving systems of nonlinear equations," Hacettepe J. Math.
Stat. 52(4) (2023) 1006-1021, Eqs. (3.1),(3.13),(3.14):

  yk = xk - F'(xk)^{-1} F(xk)
  tk = F'(xk)^{-1} F'(yk)                      (matrix, not scalar)
  tau_k = 13/4 I - 7/2 tk + 5/4 tk^2
  zk = yk - tau_k F'(xk)^{-1} F(yk)
  alpha_k = 7/2 I - 4 tk + 3/2 tk^2
  x_{k+1} = zk - alpha_k F'(xk)^{-1} F(zk)

A genuinely different architecture from ARJ8S: a *matrix*-valued weight built
from tk = F'(xk)^{-1}F'(yk) rather than ARJ8S's scalar weight built from a
Rayleigh-quotient-like ratio of residual norms. Still only ever factorizes
F'(xk) (one Jacobian per outer iteration, like ARJ8S), but needs n extra
linear solves to form the matrix tk.
"""
import numpy as np


def nlm8(F, J, x0, tol=1e-13, max_iter=100):
    x = np.array(x0, dtype=float)
    n = len(x)
    hist = [x.copy()]
    I = np.eye(n)
    for k in range(max_iter):
        Jk = J(x)
        Fk = F(x)
        Jk_inv_action = np.linalg.inv(Jk)  # explicit inverse: needed to form the matrix tk cheaply
        s1 = -Jk_inv_action @ Fk
        y = x + s1
        Fy = F(y)
        Jy = J(y)
        tk = Jk_inv_action @ Jy
        tk2 = tk @ tk
        tau_k = 13 / 4 * I - 7 / 2 * tk + 5 / 4 * tk2
        z = y - tau_k @ (Jk_inv_action @ Fy)
        Fz = F(z)
        alpha_k = 7 / 2 * I - 4 * tk + 3 / 2 * tk2
        x_new = z - alpha_k @ (Jk_inv_action @ Fz)
        hist.append(x_new.copy())
        err_approx = np.linalg.norm(x_new - x)
        err_resid = np.linalg.norm(F(x_new))
        x = x_new
        if err_approx < tol or err_resid < tol:
            break
    return x, hist
