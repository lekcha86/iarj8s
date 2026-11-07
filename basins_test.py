"""
Basins of attraction for ARJ8S vs I-ARJ8S on the base paper's exact test system
(Section 5 of the base ARJ8S paper): x1^2=1, x2^2=1, roots (+-1,+-1).
Vectorized across the whole mesh with numpy (the system's Jacobian is diagonal,
which makes this tractable without per-point Python loops).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOTS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']


def classify(X1, X2, tol_root=1e-2):
    """-1 = not converged to any root; 0..3 = index into ROOTS."""
    out = np.full(X1.shape, -1, dtype=int)
    for idx, (r1, r2) in enumerate(ROOTS):
        mask = (np.abs(X1 - r1) < tol_root) & (np.abs(X2 - r2) < tol_root)
        out[mask & (out == -1)] = idx
    return out


def arj8s_basins(X1, X2, tol=1e-3, max_iter=80):
    eps = 1e-14
    for _ in range(max_iter):
        F1, F2 = X1**2 - 1, X2**2 - 1
        J1, J2 = 2 * X1, 2 * X2
        s1_1 = -F1 / np.where(np.abs(J1) < eps, eps, J1)
        s1_2 = -F2 / np.where(np.abs(J2) < eps, eps, J2)
        Z1, Z2 = X1 + s1_1, X2 + s1_2
        Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1
        num = F1 * Fz1 + F2 * Fz2
        den = F1**2 + F2**2
        den = np.where(np.abs(den) < eps, eps, den)
        alpha = 1 - 2 * num / den
        alpha = np.where(np.abs(alpha) < eps, eps, alpha)
        for _ in range(3):
            s_1 = -Fz1 / (alpha * np.where(np.abs(J1) < eps, eps, J1))
            s_2 = -Fz2 / (alpha * np.where(np.abs(J2) < eps, eps, J2))
            Z1, Z2 = Z1 + s_1, Z2 + s_2
            Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1
        err = np.sqrt((Z1 - X1)**2 + (Z2 - X2)**2)
        X1, X2 = Z1, Z2
        if np.nanmax(np.abs(X1)) > 1e150 or np.nanmax(np.abs(X2)) > 1e150:
            X1 = np.nan_to_num(X1, nan=1e300, posinf=1e300, neginf=-1e300)
            X2 = np.nan_to_num(X2, nan=1e300, posinf=1e300, neginf=-1e300)
        if np.all((err < tol) | ~np.isfinite(err)):
            break
    return X1, X2


def iarj8s_basins(X1_0, X2_0, tol=1e-3, max_iter=80, theta_bar=0.9, gamma=1e-4,
                   beta=0.5, max_bt=8):
    eps = 1e-14
    X1, X2 = X1_0.copy(), X2_0.copy()
    X1p, X2p = X1.copy(), X2.copy()
    t_prev = np.ones_like(X1)

    def merit(a, b):
        return (a**2 - 1)**2 + (b**2 - 1)**2

    for _ in range(max_iter):
        f_x = merit(X1, X2)
        t_k = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        theta = np.minimum(theta_bar, (t_prev - 1) / t_k)
        t_prev = t_k
        Y1_trial = X1 + theta * (X1 - X1p)
        Y2_trial = X2 + theta * (X2 - X2p)
        f_trial = merit(Y1_trial, Y2_trial)
        use_trial = f_trial <= f_x
        Y1 = np.where(use_trial, Y1_trial, X1)
        Y2 = np.where(use_trial, Y2_trial, X2)
        Fy1, Fy2 = merit_ok = (Y1**2 - 1), (Y2**2 - 1)

        Jy1, Jy2 = 2 * Y1, 2 * Y2
        Jy1s = np.where(np.abs(Jy1) < eps, eps, Jy1)
        Jy2s = np.where(np.abs(Jy2) < eps, eps, Jy2)

        # Stage 1 with non-monotone line search (vectorized backtracking)
        dir1, dir2 = -Fy1 / Jy1s, -Fy2 / Jy2s
        f_base = Fy1**2 + Fy2**2
        xi = np.ones_like(X1)
        Z1, Z2 = Y1 + dir1, Y2 + dir2
        for _ in range(max_bt):
            Fz1c, Fz2c = Z1**2 - 1, Z2**2 - 1
            f_trial_s = Fz1c**2 + Fz2c**2
            accept = f_trial_s <= f_base  # simple sufficient-decrease vs stage start
            xi = np.where(accept, xi, xi * beta)
            Z1 = np.where(accept, Z1, Y1 + xi * dir1)
            Z2 = np.where(accept, Z2, Y2 + xi * dir2)
        Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1

        num = Fy1 * Fz1 + Fy2 * Fz2
        den = Fy1**2 + Fy2**2
        den = np.where(np.abs(den) < eps, eps, den)
        alpha = 1 - 2 * num / den
        alpha = np.where(np.abs(alpha) < eps, eps, alpha)

        for _ in range(3):
            f_stage = Fz1**2 + Fz2**2
            dir1 = -Fz1 / (alpha * Jy1s)
            dir2 = -Fz2 / (alpha * Jy2s)
            xi = np.ones_like(X1)
            Zn1, Zn2 = Z1 + dir1, Z2 + dir2
            for _ in range(max_bt):
                Fn1, Fn2 = Zn1**2 - 1, Zn2**2 - 1
                f_trial_s = Fn1**2 + Fn2**2
                accept = f_trial_s <= f_stage
                xi = np.where(accept, xi, xi * beta)
                Zn1 = np.where(accept, Zn1, Z1 + xi * dir1)
                Zn2 = np.where(accept, Zn2, Z2 + xi * dir2)
            Z1, Z2 = Zn1, Zn2
            Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1

        err = np.sqrt((Z1 - X1)**2 + (Z2 - X2)**2)
        X1p, X2p = X1, X2
        X1, X2 = Z1, Z2
        bign = (np.abs(X1) > 1e150) | (np.abs(X2) > 1e150) | ~np.isfinite(X1) | ~np.isfinite(X2)
        if np.any(bign):
            X1 = np.where(bign, 1e300, X1)
            X2 = np.where(bign, 1e300, X2)
        if np.all((err < tol) | ~np.isfinite(err) | bign):
            break
    return X1, X2


def nlm8_basins(X1, X2, tol=1e-3, max_iter=80):
    """NLM8 (Sharma & Arora, via Zhanlav/Mijiddorj/Khuder 2023) on the same
    diagonal-Jacobian test system; Jacobian being diagonal here lets the
    matrix-valued weight tk collapse to elementwise arrays too."""
    eps = 1e-14
    for _ in range(max_iter):
        F1, F2 = X1**2 - 1, X2**2 - 1
        J1, J2 = 2 * X1, 2 * X2
        J1s = np.where(np.abs(J1) < eps, eps, J1)
        J2s = np.where(np.abs(J2) < eps, eps, J2)
        Y1, Y2 = X1 - F1 / J1s, X2 - F2 / J2s
        Fy1, Fy2 = Y1**2 - 1, Y2**2 - 1
        Jy1, Jy2 = 2 * Y1, 2 * Y2
        t1, t2 = Jy1 / J1s, Jy2 / J2s
        tau1 = 13 / 4 - 7 / 2 * t1 + 5 / 4 * t1**2
        tau2 = 13 / 4 - 7 / 2 * t2 + 5 / 4 * t2**2
        Z1 = Y1 - tau1 * Fy1 / J1s
        Z2 = Y2 - tau2 * Fy2 / J2s
        Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1
        a1 = 7 / 2 - 4 * t1 + 3 / 2 * t1**2
        a2 = 7 / 2 - 4 * t2 + 3 / 2 * t2**2
        Xn1 = Z1 - a1 * Fz1 / J1s
        Xn2 = Z2 - a2 * Fz2 / J2s
        err = np.sqrt((Xn1 - X1)**2 + (Xn2 - X2)**2)
        X1, X2 = Xn1, Xn2
        bign = (np.abs(X1) > 1e150) | (np.abs(X2) > 1e150) | ~np.isfinite(X1) | ~np.isfinite(X2)
        if np.any(bign):
            X1 = np.where(bign, 1e300, X1)
            X2 = np.where(bign, 1e300, X2)
        if np.all((err < tol) | ~np.isfinite(err) | bign):
            break
    return X1, X2


def ctt8s_basins(X1, X2, tol=1e-3, max_iter=80):
    """CTT8S vectorized for the diagonal-Jacobian test system."""
    eps = 1e-14
    for _ in range(max_iter):
        F1, F2 = X1**2 - 1, X2**2 - 1
        J1, J2 = 2 * X1, 2 * X2
        J1s = np.where(np.abs(J1) < eps, eps, J1)
        J2s = np.where(np.abs(J2) < eps, eps, J2)
        Y1, Y2 = X1 - F1 / J1s, X2 - F2 / J2s
        Fy1, Fy2 = Y1**2 - 1, Y2**2 - 1
        nu1 = (Fy1 * Fy1) / np.where(np.abs(F1 * F1) < eps, eps, F1 * F1)
        nu2 = (Fy2 * Fy2) / np.where(np.abs(F2 * F2) < eps, eps, F2 * F2)
        rhs1 = (1 + nu1) * Fy1 + 2 * nu1 * F1
        rhs2 = (1 + nu2) * Fy2 + 2 * nu2 * F2
        Z1 = Y1 - rhs1 / J1s
        Z2 = Y2 - rhs2 / J2s
        Fz1, Fz2 = Z1**2 - 1, Z2**2 - 1
        gam1 = (Fy1 * Fz1) / np.where(np.abs(F1 * F1) < eps, eps, F1 * F1)
        gam2 = (Fy2 * Fz2) / np.where(np.abs(F2 * F2) < eps, eps, F2 * F2)
        bet1 = (Fz1 * Fz1) / np.where(np.abs(Fy1 * Fy1) < eps, eps, Fy1 * Fy1)
        bet2 = (Fz2 * Fz2) / np.where(np.abs(Fy2 * Fy2) < eps, eps, Fy2 * Fy2)
        del1 = (Fz1 * Fz1) / np.where(np.abs(F1 * F1) < eps, eps, F1 * F1)
        del2 = (Fz2 * Fz2) / np.where(np.abs(F2 * F2) < eps, eps, F2 * F2)
        rr1 = Fz1 + (2 * gam1 * (1 - nu1) + 4 * del1) * F1 + (2 * gam1 + bet1) * Fy1
        rr2 = Fz2 + (2 * gam2 * (1 - nu2) + 4 * del2) * F2 + (2 * gam2 + bet2) * Fy2
        Xn1 = Z1 - rr1 / J1s
        Xn2 = Z2 - rr2 / J2s
        err = np.sqrt((Xn1 - X1)**2 + (Xn2 - X2)**2)
        X1, X2 = Xn1, Xn2
        bign = (np.abs(X1) > 1e150) | (np.abs(X2) > 1e150) | ~np.isfinite(X1) | ~np.isfinite(X2)
        if np.any(bign):
            X1 = np.where(bign, 1e300, X1)
            X2 = np.where(bign, 1e300, X2)
        if np.all((err < tol) | ~np.isfinite(err) | bign):
            break
    return X1, X2


def zo8s_basins(X1, X2, tol=1e-3, max_iter=80):
    """ZO8S vectorized (diagonal Jacobian; second Jacobian at z1, reused)."""
    eps = 1e-14
    for _ in range(max_iter):
        F1, F2 = X1**2 - 1, X2**2 - 1
        J1, J2 = 2 * X1, 2 * X2
        J1s = np.where(np.abs(J1) < eps, eps, J1)
        J2s = np.where(np.abs(J2) < eps, eps, J2)
        Z1_1, Z1_2 = X1 - F1 / J1s, X2 - F2 / J2s
        Fz1_1, Fz1_2 = Z1_1**2 - 1, Z1_2**2 - 1
        nu1 = (Fz1_1 * Fz1_1) / np.where(np.abs(F1 * F1) < eps, eps, F1 * F1)
        nu2 = (Fz1_2 * Fz1_2) / np.where(np.abs(F2 * F2) < eps, eps, F2 * F2)
        Jz1_1, Jz1_2 = 2 * Z1_1, 2 * Z1_2
        Jz1_1s = np.where(np.abs(Jz1_1) < eps, eps, Jz1_1)
        Jz1_2s = np.where(np.abs(Jz1_2) < eps, eps, Jz1_2)
        rhs2_1 = (1 + nu1) * Fz1_1
        rhs2_2 = (1 + nu2) * Fz1_2
        Z2_1 = Z1_1 - rhs2_1 / Jz1_1s
        Z2_2 = Z1_2 - rhs2_2 / Jz1_2s
        Fz2_1, Fz2_2 = Z2_1**2 - 1, Z2_2**2 - 1
        rhs3_1 = (1 + 2 * nu1) * Fz2_1
        rhs3_2 = (1 + 2 * nu2) * Fz2_2
        Xn1 = Z2_1 - rhs3_1 / Jz1_1s  # reuse Jz1
        Xn2 = Z2_2 - rhs3_2 / Jz1_2s
        err = np.sqrt((Xn1 - X1)**2 + (Xn2 - X2)**2)
        X1, X2 = Xn1, Xn2
        bign = (np.abs(X1) > 1e150) | (np.abs(X2) > 1e150) | ~np.isfinite(X1) | ~np.isfinite(X2)
        if np.any(bign):
            X1 = np.where(bign, 1e300, X1)
            X2 = np.where(bign, 1e300, X2)
        if np.all((err < tol) | ~np.isfinite(err) | bign):
            break
    return X1, X2


def eh9s_basins(X1, X2, tol=1e-3, max_iter=80):
    """EH9S vectorized (diagonal Jacobian, frozen at x_k, 4 sub-steps)."""
    eps = 1e-14
    mstep = 4
    for _ in range(max_iter):
        F1, F2 = X1**2 - 1, X2**2 - 1
        J1, J2 = 2 * X1, 2 * X2
        J1s = np.where(np.abs(J1) < eps, eps, J1)
        J2s = np.where(np.abs(J2) < eps, eps, J2)
        W1, W2 = X1 - F1 / J1s, X2 - F2 / J2s
        for _ in range(1, mstep):
            Fw1, Fw2 = W1**2 - 1, W2**2 - 1
            c1, c2 = Fw1 / J1s, Fw2 / J2s
            Fp1 = (W1 + c1)**2 - 1
            Fp2 = (W2 + c2)**2 - 1
            Fm1 = (W1 - c1)**2 - 1
            Fm2 = (W2 - c2)**2 - 1
            z1 = 2 * Fp1 + 3 * Fm1 - 3 * Fw1
            z2 = 2 * Fp2 + 3 * Fm2 - 3 * Fw2
            W1 = W1 - z1 / J1s
            W2 = W2 - z2 / J2s
        err = np.sqrt((W1 - X1)**2 + (W2 - X2)**2)
        X1, X2 = W1, W2
        bign = (np.abs(X1) > 1e150) | (np.abs(X2) > 1e150) | ~np.isfinite(X1) | ~np.isfinite(X2)
        if np.any(bign):
            X1 = np.where(bign, 1e300, X1)
            X2 = np.where(bign, 1e300, X2)
        if np.all((err < tol) | ~np.isfinite(err) | bign):
            break
    return X1, X2


def make_basin_plot(region, methods, titles, fname, n_mesh=800, tol=1e-3, max_iter=80):
    lo, hi = region
    x = np.linspace(lo, hi, n_mesh)
    y = np.linspace(lo, hi, n_mesh)
    X1_0, X2_0 = np.meshgrid(x, y)

    cmap = ListedColormap(['black'] + COLORS)
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4))
    if len(methods) == 1:
        axes = [axes]

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, (ax, method, title) in enumerate(zip(axes, methods, titles)):
        X1f, X2f = method(X1_0.copy(), X2_0.copy(), tol=tol, max_iter=max_iter)
        classes = classify(X1f, X2f)
        ax.imshow(classes, extent=[lo, hi, lo, hi], origin='lower', cmap=cmap,
                  vmin=-1, vmax=3, interpolation='nearest')
        for (r1, r2) in ROOTS:
            ax.plot(r1, r2, 'w*', markersize=9, markeredgecolor='black',
                     markeredgewidth=0.6)
        # Method name + panel letter, not a redundant duplicate of the
        # LaTeX \caption -- journal convention for multi-panel figures.
        ax.set_title(f'{panel_labels[i]} {title}', fontsize=13, pad=6)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.tick_params(direction='out')
        n_black = np.sum(classes == -1)
        print(f"{title} on {region}: non-convergent points = {n_black}/{n_mesh*n_mesh} "
              f"({100*n_black/(n_mesh*n_mesh):.2f}%)")

    plt.tight_layout(pad=0.6)
    plt.savefig(fname, dpi=400)
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    make_basin_plot((-1.5, 1.5), [arj8s_basins, iarj8s_basins, nlm8_basins],
                     ['ARJ8S', 'I-ARJ8S', 'NLM8'], 'basins_near.png', n_mesh=800)
    make_basin_plot((-30, 30), [arj8s_basins, iarj8s_basins, nlm8_basins],
                     ['ARJ8S', 'I-ARJ8S', 'NLM8'], 'basins_far.png', n_mesh=800)
    make_basin_plot((-1.5, 1.5), [ctt8s_basins, zo8s_basins, eh9s_basins],
                     ['CTT8S', 'ZO8S', 'EH9S'], 'basins_near2.png', n_mesh=800)
    make_basin_plot((-30, 30), [ctt8s_basins, zo8s_basins, eh9s_basins],
                     ['CTT8S', 'ZO8S', 'EH9S'], 'basins_far2.png', n_mesh=800)
