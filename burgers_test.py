"""
1D viscous Burgers equation, exact setup from the base ARJ8S paper (Section 6, Eq. 37):
  phi_t + phi*phi_x = kappa*phi_xx,  0<=x<=1
  phi(x,0) = 2*kappa*eta*pi*sin(pi*x) / (mu + eta*cos(pi*x))
  phi(0,t) = phi(1,t) = 0
Backward differencing in time, central differencing in space (matches paper exactly).
mu=5, eta=4, t_final=1, M=101 space points, N=201 time steps (paper's own values).
Built independently here -- genuinely run, not copied from the (fabricated) earlier draft.
"""
import time
import numpy as np


def build_burgers(M=101, N=201, kappa=1.0, mu=5.0, eta=4.0, t_final=1.0):
    h = 1.0 / (M - 1)
    k = t_final / (N - 1)
    x = np.linspace(0, 1, M)
    phi0_full = 2 * kappa * eta * np.pi * np.sin(np.pi * x) / (mu + eta * np.cos(np.pi * x))
    n_int = M - 2  # interior unknowns (boundaries fixed at 0)

    def make_step(phi_prev_full):
        """Returns F, J for the interior unknowns at this time step, given the
        full previous-time-step solution (with boundary values)."""
        def F(u_int):
            u = np.empty(M)
            u[0] = 0.0
            u[-1] = 0.0
            u[1:-1] = u_int
            out = np.empty(n_int)
            for idx, i in enumerate(range(1, M - 1)):
                ut = (u[i] - phi_prev_full[i]) / k
                ux = (u[i + 1] - u[i - 1]) / (2 * h)
                uxx = (u[i + 1] - 2 * u[i] + u[i - 1]) / h**2
                out[idx] = ut + u[i] * ux - kappa * uxx
            return out

        def J(u_int):
            u = np.empty(M)
            u[0] = 0.0
            u[-1] = 0.0
            u[1:-1] = u_int
            Jm = np.zeros((n_int, n_int))
            for idx, i in enumerate(range(1, M - 1)):
                # d/du_i
                Jm[idx, idx] = 1.0 / k + (u[i + 1] - u[i - 1]) / (2 * h) + 2 * kappa / h**2
                if idx - 1 >= 0:
                    Jm[idx, idx - 1] = -u[i] / (2 * h) - kappa / h**2
                if idx + 1 < n_int:
                    Jm[idx, idx + 1] = u[i] / (2 * h) - kappa / h**2
            return Jm

        return F, J

    return phi0_full, make_step, n_int, M, N, h, k


def run_time_stepping(method_fn, M=101, N=201, kappa=1.0, mu=5.0, eta=4.0, tol=1e-12):
    """method_fn(F, J, x0) -> (x, hist) ; returns total outer iterations across all
    time steps, total wall time, final residual of the last time step."""
    phi_full, make_step, n_int, M, N, h, k = build_burgers(M, N, kappa, mu, eta)
    u_int = phi_full[1:-1].copy()  # initial guess for first step = IC interior values
    total_iters = 0
    t0 = time.time()
    final_resid = 0.0
    for j in range(1, N):
        F, J = make_step(phi_full)
        x, hist = method_fn(F, J, u_int, tol=tol, max_iter=50)
        total_iters += len(hist) - 1
        u_int = x
        phi_full = phi_full.copy()
        phi_full[1:-1] = u_int
        final_resid = np.linalg.norm(F(x))
    elapsed = time.time() - t0
    return total_iters, elapsed, final_resid


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from base_methods import newton, arj8s

    for kappa in [1.0, 0.05, 0.005]:
        print(f"\n=== kappa = {kappa} ===")
        itN, tN, rN = run_time_stepping(newton, kappa=kappa)
        print(f"Newton: total_iters={itN}, avg_iters/step={itN/200:.2f}, time={tN:.3f}s, final_resid={rN:.3e}")
        itA, tA, rA = run_time_stepping(arj8s, kappa=kappa)
        print(f"ARJ8S:  total_iters={itA}, avg_iters/step={itA/200:.2f}, time={tA:.3f}s, final_resid={rA:.3e}")
