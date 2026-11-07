"""High-precision check that I-ARJ8S recovers order 8 once safeguards stop
firing (i.e. inertial step accepted trivially / line search accepts xi=1).
Uses mpmath for the linear solves so we can see genuine ACOC=8 despite the
insanely fast error shrinkage."""
import mpmath as mp

mp.mp.dps = 800
m = 15

def F(x):
    out = [x[i]**2 * x[i + 1] - 1 for i in range(m - 1)]
    out.append(x[m - 1]**2 * x[0] - 1)
    return mp.matrix(out)

def Jrow(x):
    Jm = mp.zeros(m, m)
    for i in range(m - 1):
        Jm[i, i] = 2 * x[i] * x[i + 1]
        Jm[i, i + 1] = x[i]**2
    Jm[m - 1, m - 1] = 2 * x[m - 1] * x[0]
    Jm[m - 1, 0] = x[m - 1]**2
    return Jm

def norm(v):
    return mp.sqrt(sum(vi**2 for vi in v))

x_star = mp.matrix([mp.mpf(1) for _ in range(m)])


def merit(Fv):
    return sum(fi * fi for fi in Fv)


def nonmonotone_ls(base_pt, Fbase, direction, window, gamma=mp.mpf('1e-4'), beta=mp.mpf('0.5')):
    f_base = merit(Fbase)
    M_k = max(window) if window else f_base
    xi = mp.mpf(1)
    for _ in range(60):
        trial = base_pt + xi * direction
        Ftrial = F(trial)
        f_trial = merit(Ftrial)
        if f_trial <= M_k - 2 * gamma * xi * f_base or xi < mp.mpf('1e-30'):
            return trial, Ftrial
        xi *= beta
    return trial, Ftrial


def run_iarj8s(x0, iters, theta_bar=mp.mpf('0.9'), window_size=5):
    x_prev = mp.matrix(x0)
    x = mp.matrix(x0)
    hist = [x]
    window = []
    t_prev = mp.mpf(1)
    for k in range(iters):
        Fx = F(x)
        fx = merit(Fx)
        window.append(fx)
        if len(window) > window_size:
            window.pop(0)
        if fx == 0:
            break

        t_k = (1 + mp.sqrt(1 + 4 * t_prev**2)) / 2
        theta_k = min(theta_bar, (t_prev - 1) / t_k)
        t_prev = t_k
        s_k = x - x_prev
        y_trial = x + theta_k * s_k
        Fy_trial = F(y_trial)
        if merit(Fy_trial) <= fx:
            y, Fy = y_trial, Fy_trial
        else:
            y, Fy = x, Fx

        Jy = Jrow(y)
        s1 = mp.lu_solve(Jy, -Fy)
        z1, Fz1 = nonmonotone_ls(y, Fy, s1, window)
        window.append(merit(Fz1))
        if len(window) > window_size:
            window.pop(0)

        den = sum(Fy[i] * Fy[i] for i in range(m))
        alpha = 1 - 2 * sum(Fy[i] * Fz1[i] for i in range(m)) / den

        z_prev, Fz_prev = z1, Fz1
        for _ in range(3):
            s_j = mp.lu_solve(alpha * Jy, -Fz_prev)
            z_i, Fz_i = nonmonotone_ls(z_prev, Fz_prev, s_j, window)
            window.append(merit(Fz_i))
            if len(window) > window_size:
                window.pop(0)
            z_prev, Fz_prev = z_i, Fz_i

        x_prev, x = x, z_prev
        hist.append(x)
    return hist


x0 = [mp.mpf('2.2') for _ in range(m)]
hist = run_iarj8s(x0, 6)

errs = [norm(h - x_star) for h in hist]
print("I-ARJ8S true errors:")
for i, e in enumerate(errs):
    print(f"  iter {i}: {mp.nstr(e, 6)}")

orders = []
for k in range(1, len(errs) - 1):
    if errs[k] > 0 and errs[k - 1] > 0 and errs[k + 1] > 0:
        num = mp.log(errs[k + 1] / errs[k])
        den = mp.log(errs[k] / errs[k - 1])
        if den != 0:
            orders.append(num / den)
print("I-ARJ8S ACOC:", [mp.nstr(o, 6) for o in orders])
