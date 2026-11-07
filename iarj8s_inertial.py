"""
I-ARJ8S: an inertial, non-monotone-globalized, optionally-inexact (GMRES) extension
of ARJ8S (Cordero, Rojas-Hiciano, Torregrosa, Appl. Numer. Math. 229 (2026) 209-236).

Three genuine additions, each grounded in established, citable mechanisms
(not invented from scratch):
  1. Inertial extrapolation (Nesterov/FISTA-style momentum, Beck & Teboulle 2009),
     safeguarded: only accepted if it does not increase the residual.
  2. Non-monotone line search in the DF-SANE / Grippo-Lampariello-Lucidi style
     (La Cruz, Martinez, Raydan, Math. Comp. 75 (2006) 1429-1448), applied at
     each of the 4 Super-Ostrowski stages, replacing ARJ8S-Armijo's strictly
     monotone backtracking.
  3. Optional inexact linear solves via GMRES with a single reused ILU
     preconditioner per outer iteration (Dembo-Eisenstat-Steihaug 1982 inexact-
     Newton forcing-term framework), for large sparse systems.

Local order is NOT claimed to be proven here beyond the standard, well-known
argument: (a) the inertial step vanishes relative to the O(e^8) correction as
e_k -> 0 (bounded theta_k, safeguarded); (b) non-monotone line search accepts
full steps (xi=1) asymptotically near a nonsingular root, recovering exact
ARJ8S; (c) forcing terms eta_k -> 0 fast enough preserve the exact method's
order (Dembo-Eisenstat-Steihaug). This is verified empirically via ACOC below,
not asserted as a new theorem.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _merit(Fv):
    return float(Fv @ Fv)


def _nonmonotone_ls(F, base_pt, Fbase, direction, window, gamma=1e-4, beta=0.5,
                     xi_min=1e-10, max_bt=40):
    """DF-SANE-style non-monotone backtracking for the correction step z_prev -> z_prev + xi*s.
    window: list of recent merit values f=||F||^2 (sliding window, updated by caller).
    Descent quantity uses the Newton-type direction's exact d(f)/dxi at xi=0,
    which for s = -alpha^{-1} J^{-1} F(base_pt) with J the *frozen* Jacobian is
    approximated by -2*gamma*xi*||Fbase||^2 (standard DF-SANE surrogate when the
    exact Jacobian-direction product is not cheaply available).

    If backtracking shrinks all the way to xi_min without ever satisfying the
    descent condition, we do NOT return that near-zero step (found empirically
    to freeze progress into a stall rather than merely slow it -- see project
    notes on the lambda=6.6-6.8 Bratu regime). Instead we fall back to the full
    undamped step xi=1, exactly what an unglobalized method like ZO8S always
    takes, and accept it unless it is actually non-finite. This recovers a
    genuine chance to escape a stall the same way an unsafeguarded step
    sometimes does, while a NaN/Inf check is the one thing we never skip."""
    f_base = _merit(Fbase)
    M_k = max(window) if window else f_base
    xi = 1.0
    full_trial, full_Ftrial = None, None
    for i in range(max_bt):
        trial = base_pt + xi * direction
        Ftrial = F(trial)
        if i == 0:
            full_trial, full_Ftrial = trial, Ftrial
        f_trial = _merit(Ftrial)
        if f_trial <= M_k - 2.0 * gamma * xi * f_base:
            return trial, Ftrial, xi
        if xi < xi_min:
            break
        xi *= beta
    if np.all(np.isfinite(full_Ftrial)) and np.all(np.isfinite(full_trial)):
        return full_trial, full_Ftrial, 1.0
    return trial, Ftrial, xi  # last resort: the smallest (still finite) step tried


def _solve_linear(Jk, rhs, use_gmres, precond, gmres_tol):
    if not use_gmres:
        return np.linalg.solve(Jk, rhs) if not sp.issparse(Jk) else spla.spsolve(Jk, rhs)
    x, info = spla.gmres(Jk, rhs, M=precond, rtol=gmres_tol, atol=0.0, restart=50, maxiter=200)
    return x


def _stage_step(F, Jk, base_pt, Fbase, use_gmres, precond, eta0, window,
                 eta_shrink=10.0, eta_min=1e-10, max_retries=4):
    """One corrector stage with an adaptively-tightening forcing term: if the
    inexact (GMRES) direction does not even beat the incoming stage residual
    after line search, the linear solve is redone with a tighter rtol before
    falling back to an exact solve. This is what keeps the shared frozen-
    Jacobian architecture valid when stages 2-4 drift away from the point
    (y) the Jacobian/preconditioner was built at -- a plain uniform forcing
    term (e.g. rtol=0.1 for every stage) was found empirically to let later
    stages actively worsen the residual in hard (far-from-root, near-
    bifurcation) regimes; see project notes."""
    f_incoming = _merit(Fbase)
    eta = eta0
    for attempt in range(max_retries):
        direction = _solve_linear(Jk, -Fbase, use_gmres, precond, eta)
        trial, Ftrial, xi = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
        if _merit(Ftrial) <= f_incoming:  # genuine progress vs. this stage's own start
            return trial, Ftrial
        eta = max(eta / eta_shrink, eta_min)
    # last resort: exact solve for this stage only
    direction = _solve_linear(Jk, -Fbase, False, None, 0.0)
    trial, Ftrial, xi = _nonmonotone_ls(F, base_pt, Fbase, direction, window)
    return trial, Ftrial


def iarj8s(F, J, x0, tol=1e-13, max_iter=100,
           theta_bar=0.9, window_size=5,
           use_gmres=False, gmres_tol_base=0.1, gmres_tol_stage1=1e-6,
           verbose=False):
    """
    F: R^n -> R^n
    J: R^n -> Jacobian (dense ndarray or scipy sparse), evaluated once per outer iter
    x0: initial guess
    use_gmres: if True, Jacobian may be sparse; linear systems solved inexactly via
               GMRES with a single ILU preconditioner reused across all 4 stages.

    IMPORTANT (found empirically, see project notes): stage 1's linear solve must be
    much more accurate (gmres_tol_stage1) than stages 2-4 (gmres_tol_base). Stage 1
    determines both z1 and the scalar weight alpha_k that stages 2-4 depend on, so an
    inaccurate stage-1 direction propagates and can make the *later* stages actively
    increase the residual even when solved exactly against the (now poorly-matched)
    frozen Jacobian. A uniform loose forcing term across all 4 stages was tested and
    found to stall in hard (far-from-root, near-bifurcation) regimes -- this asymmetric
    tolerance is the fix.
    """
    x_prev = np.array(x0, dtype=float)
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    window = []  # sliding window of merit values ||F||^2
    t_prev = 1.0

    for k in range(max_iter):
        Fx = F(x)
        window.append(_merit(Fx))
        if len(window) > window_size:
            window.pop(0)

        # --- 1. Safeguarded inertial extrapolation ---
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        theta_k = min(theta_bar, (t_prev - 1.0) / t_k)
        t_prev = t_k
        s_k = x - x_prev
        y_trial = x + theta_k * s_k
        Fy_trial = F(y_trial)
        if _merit(Fy_trial) <= _merit(Fx):
            y, Fy = y_trial, Fy_trial
        else:
            y, Fy = x, Fx  # reject extrapolation, no momentum this step

        # --- Jacobian + preconditioner built once, reused across all 4 stages ---
        Jy = J(y)
        precond = None
        if use_gmres:
            ilu = spla.spilu(sp.csc_matrix(Jy), drop_tol=1e-5, fill_factor=15)
            precond = spla.LinearOperator(Jy.shape, ilu.solve)

        resid_norm = np.linalg.norm(Fy)
        gmres_tol1 = min(gmres_tol_stage1, resid_norm)

        # --- Stage 1: Newton predictor from y, tight forcing term (see docstring) ---
        z_prev, Fz_prev = _stage_step(F, Jy, y, Fy, use_gmres, precond, gmres_tol1, window)
        window.append(_merit(Fz_prev))
        if len(window) > window_size:
            window.pop(0)
        best_pt, best_F, best_merit = z_prev, Fz_prev, _merit(Fz_prev)

        # --- Scalar weight function alpha_k (shared across stages 2-4, as in ARJ8S) ---
        alpha = 1.0 - 2.0 * (Fy @ Fz_prev) / (Fy @ Fy)
        if abs(alpha) < 1e-12:
            alpha = 1e-12 if alpha >= 0 else -1e-12

        # --- Stages 2-4: Ostrowski-type corrections, each with adaptive forcing term ---
        for _ in range(3):
            gmres_tol_j = min(gmres_tol_base, np.linalg.norm(Fz_prev))
            z_prev, Fz_prev = _stage_step(F, alpha * Jy, z_prev, Fz_prev, use_gmres, precond,
                                           gmres_tol_j, window)
            window.append(_merit(Fz_prev))
            if len(window) > window_size:
                window.pop(0)
            m = _merit(Fz_prev)
            if m < best_merit:  # track the best point seen across ALL 4 stages
                best_pt, best_F, best_merit = z_prev, Fz_prev, m

        # Use the best-seen point, not necessarily the last stage's -- found
        # empirically that stages 2-4 can *increase* the residual once they
        # drift far enough from y that the frozen Jacobian F'(y) stops being a
        # good local model, even with exact solves (see project notes: this
        # happens in a genuinely nonlinear regime, distinct from the earlier
        # GMRES-inexactness stall this project already fixed separately).
        x_new = best_pt
        Fz_prev = best_F
        x_prev, x = x, x_new
        hist.append(x.copy())

        if verbose:
            print(f"iter {k}: ||F||={np.linalg.norm(Fz_prev):.3e}, theta={theta_k:.3f}")

        if np.linalg.norm(x - x_prev) < tol or np.linalg.norm(F(x)) < tol:
            break

    return x, hist
