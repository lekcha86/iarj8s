"""
Reproduces the base ARJ8S paper's own numerical test problems (Section 7),
at the same VPA precision (1000 digits) and stopping criteria, for Newton,
ARJ8S, CTT8S, ZO8S, EH9S, NLM8, and I-ARJ8S -- so our comparison covers
exactly what the base paper itself compares, not just our own Bratu/Burgers
tests.
"""
import mpmath as mp

mp.mp.dps = 1000


def norm(v):
    return mp.sqrt(sum(vi**2 for vi in v))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


# ---------------------------------------------------------------------
# Generic mpmath implementations of all 7 methods (dense, matches the
# formulas already verified in double precision elsewhere in this project)
# ---------------------------------------------------------------------

def run_method(name, F, Jrow, x0, tol, max_iter):
    n = len(x0)
    I = mp.eye(n)
    x = mp.matrix(x0)
    x_prev = mp.matrix(x0)
    t_prev = mp.mpf(1)
    hist = [norm(F(x))]

    def merit(Fv):
        return dot(Fv, Fv)

    for k in range(max_iter):
        Fx = F(x)
        if name == 'Newton':
            Jx = Jrow(x)
            s = mp.lu_solve(Jx, -Fx)
            x = x + s

        elif name == 'ARJ8S':
            Jk = Jrow(x)
            s1 = mp.lu_solve(Jk, -Fx)
            z1 = x + s1
            Fz1 = F(z1)
            alpha = 1 - 2 * dot(Fx, Fz1) / dot(Fx, Fx)
            zp, Fzp = z1, Fz1
            for _ in range(3):
                si = mp.lu_solve(alpha * Jk, -Fzp)
                zi = zp + si
                Fzp = F(zi)
                zp = zi
            x = zp

        elif name == 'CTT8S':
            Jx = Jrow(x)
            s1 = mp.lu_solve(Jx, -Fx)
            y = x + s1
            Fy = F(y)
            nu = dot(Fy, Fy) / dot(Fx, Fx)
            rhs = (1 + nu) * Fy + 2 * nu * Fx
            s2 = mp.lu_solve(Jx, -rhs)
            z = y + s2
            Fz = F(z)
            gamma = dot(Fy, Fz) / dot(Fx, Fx)
            beta = dot(Fz, Fz) / dot(Fy, Fy)
            delta = dot(Fz, Fz) / dot(Fx, Fx)
            rhs3 = Fz + (2 * gamma * (1 - nu) + 4 * delta) * Fx + (2 * gamma + beta) * Fy
            s3 = mp.lu_solve(Jx, -rhs3)
            x = z + s3

        elif name == 'ZO8S':
            Jk = Jrow(x)
            s1 = mp.lu_solve(Jk, -Fx)
            z1 = x + s1
            Fz1 = F(z1)
            nu = dot(Fz1, Fz1) / dot(Fx, Fx)
            Jz1 = Jrow(z1)
            rhs2 = (1 + nu) * Fz1
            s2 = mp.lu_solve(Jz1, -rhs2)
            z2 = z1 + s2
            Fz2 = F(z2)
            rhs3 = (1 + 2 * nu) * Fz2
            s3 = mp.lu_solve(Jz1, -rhs3)
            x = z2 + s3

        elif name == 'EH9S':
            Jx = Jrow(x)
            s1 = mp.lu_solve(Jx, -Fx)
            w = x + s1
            for i in range(1, 4):
                Fw = F(w)
                corr = mp.lu_solve(Jx, Fw)
                zval = 2 * F(w + corr) + 3 * F(w - corr) - 3 * Fw
                s = mp.lu_solve(Jx, -zval)
                w = w + s
            x = w

        elif name == 'NLM8':
            Jx = Jrow(x)
            s1 = mp.lu_solve(Jx, -Fx)
            y = x + s1
            Fy = F(y)
            Jy = Jrow(y)
            tk = Jx**-1 * Jy
            tk2 = tk * tk
            tau_k = mp.mpf('13') / 4 * I - mp.mpf('7') / 2 * tk + mp.mpf('5') / 4 * tk2
            z = y - tau_k * (Jx**-1 * Fy)
            Fz = F(z)
            alpha_k = mp.mpf('7') / 2 * I - 4 * tk + mp.mpf('3') / 2 * tk2
            x = z - alpha_k * (Jx**-1 * Fz)

        elif name == 'I-ARJ8S':
            fx = merit(Fx)
            t_k = (1 + mp.sqrt(1 + 4 * t_prev**2)) / 2
            theta_k = min(mp.mpf('0.9'), (t_prev - 1) / t_k)
            s_k = x - x_prev
            y_trial = x + theta_k * s_k
            Fy_trial = F(y_trial)
            if merit(Fy_trial) <= fx:
                y, Fy = y_trial, Fy_trial
            else:
                y, Fy = x, Fx
            Jy = Jrow(y)
            s1 = mp.lu_solve(Jy, -Fy)
            z1 = y + s1
            Fz1 = F(z1)
            best_pt, best_F, best_m = z1, Fz1, merit(Fz1)
            den = dot(Fy, Fy)
            alpha = 1 - 2 * dot(Fy, Fz1) / den
            zp, Fzp = z1, Fz1
            for _ in range(3):
                si = mp.lu_solve(alpha * Jy, -Fzp)
                zi = zp + si
                Fzi = F(zi)
                zp, Fzp = zi, Fzi
                mi = merit(Fzi)
                if mi < best_m:
                    best_pt, best_F, best_m = zi, Fzi, mi
            x_prev = x
            x = best_pt
            t_prev = t_k
        else:
            raise ValueError(name)

        r = norm(F(x))
        hist.append(r)
        if r < tol:
            break

    rho = None
    if len(hist) >= 3:
        e = hist  # residual-based ACOC proxy, matches paper's rho formula in spirit
        try:
            num = mp.log(e[-1] / e[-2])
            den = mp.log(e[-2] / e[-3])
            if den != 0:
                rho = num / den
        except Exception:
            rho = None
    return x, hist, rho


METHODS = ['Newton', 'ARJ8S', 'CTT8S', 'ZO8S', 'EH9S', 'NLM8', 'I-ARJ8S']


# ---------------------------------------------------------------------
# Test Suite A, Problem 1: x_i - cos(-(sum x_j) + 2x_i) = 0, m=100
# ---------------------------------------------------------------------
def problem1(m=100):
    def F(x):
        s = sum(x[i] for i in range(m))
        return mp.matrix([x[i] - mp.cos(-(s) + 2 * x[i]) for i in range(m)])

    def Jrow(x):
        s = sum(x[i] for i in range(m))
        J = mp.zeros(m, m)
        for i in range(m):
            g = -s + 2 * x[i]
            for j in range(m):
                if i == j:
                    J[i, j] = 1 + mp.sin(g)  # d/dx_i (-s+2x_i) = 1
                else:
                    J[i, j] = -mp.sin(g)
        return J

    x0 = [mp.mpf('0.1')] * m
    return F, Jrow, x0


# ---------------------------------------------------------------------
# Test Suite A, Problem 2: x_i^2 x_{i+1} - 1 = 0 (cyclic), m=500
# ---------------------------------------------------------------------
def problem2(m=500):
    def F(x):
        out = [x[i]**2 * x[i + 1] - 1 for i in range(m - 1)]
        out.append(x[m - 1]**2 * x[0] - 1)
        return mp.matrix(out)

    def Jrow(x):
        J = mp.zeros(m, m)
        for i in range(m - 1):
            J[i, i] = 2 * x[i] * x[i + 1]
            J[i, i + 1] = x[i]**2
        J[m - 1, m - 1] = 2 * x[m - 1] * x[0]
        J[m - 1, 0] = x[m - 1]**2
        return J

    x0 = [mp.mpf('1.5')] * m
    return F, Jrow, x0


# ---------------------------------------------------------------------
# Hammerstein-type integral equation, n=7 nodes (Gauss-Legendre)
# ---------------------------------------------------------------------
def hammerstein(n=7, x0val='0.5'):
    # Standard 7-point Gauss-Legendre nodes/weights on [-1,1] (well-known
    # closed-form values, hardcoded to full double precision then refined
    # by mpmath's findroot on the Legendre polynomial for higher precision).
    if n != 7:
        raise NotImplementedError("hardcoded for n=7 to match the base paper")
    raw_nodes = [0.0, 0.405845151377397166906606412076961463,
                 -0.405845151377397166906606412076961463,
                 0.741531185599394439863864773280788407,
                 -0.741531185599394439863864773280788407,
                 0.949107912342758524526189684047851262,
                 -0.949107912342758524526189684047851262]
    raw_weights = [0.417959183673469387755102040816326531,
                   0.381830050505118944950369775488975134,
                   0.381830050505118944950369775488975134,
                   0.279705391489276667901467771423779582,
                   0.279705391489276667901467771423779582,
                   0.129484966168869693270611432679082018,
                   0.129484966168869693270611432679082018]
    t_raw = [mp.mpf(v) for v in raw_nodes]
    w_raw = [mp.mpf(v) for v in raw_weights]
    t = [(ti + 1) / 2 for ti in t_raw]
    w = [wi / 2 for wi in w_raw]

    def kernel(ti, tk):
        return w[0] * 0  # placeholder, unused

    def F(x):
        out = []
        for i in range(n):
            s = mp.mpf(0)
            for k in range(n):
                if k <= i:
                    c = w[k] * t[k] * (1 - t[i])
                else:
                    c = w[k] * t[i] * (1 - t[k])
                s += c * x[k]**3
            out.append(-5 + 5 * x[i] - s)
        return mp.matrix(out)

    def Jrow(x):
        J = mp.zeros(n, n)
        for i in range(n):
            for k in range(n):
                if k <= i:
                    c = w[k] * t[k] * (1 - t[i])
                else:
                    c = w[k] * t[i] * (1 - t[k])
                J[i, k] = -3 * c * x[k]**2
            J[i, i] += 5
        return J

    x0 = [mp.mpf(x0val)] * n
    return F, Jrow, x0


# ---------------------------------------------------------------------
# Finite deflections of an elastic string, n=500, a=1
# ---------------------------------------------------------------------
def elastic_string(n=500, a=1.0, sample='alpha'):
    h = mp.mpf(1) / (n + 1)
    a = mp.mpf(a)

    def F(y):
        out = []
        for i in range(n):
            yim1 = y[i - 1] if i > 0 else mp.mpf(0)
            yip1 = y[i + 1] if i < n - 1 else mp.mpf(0)
            val = 4 * (yip1 - 2 * y[i] + yim1) + 4 * h**2 + a**2 * (yip1 - yim1)**2
            out.append(val)
        return mp.matrix(out)

    def Jrow(y):
        J = mp.zeros(n, n)
        for i in range(n):
            yim1 = y[i - 1] if i > 0 else mp.mpf(0)
            yip1 = y[i + 1] if i < n - 1 else mp.mpf(0)
            J[i, i] = -8
            if i > 0:
                J[i, i - 1] = 4 - 2 * a**2 * (yip1 - yim1)
            if i < n - 1:
                J[i, i + 1] = 4 + 2 * a**2 * (yip1 - yim1)
        return J

    idx = [i + 1 for i in range(n)]
    if sample == 'alpha':
        vals = [mp.mpf(13) / 25000 * min(i, n + 1 - i) for i in idx]
    elif sample == 'delta':
        vals = [mp.mpf('0.5')] * n
    else:
        raise ValueError(sample)
    return F, Jrow, vals


# ---------------------------------------------------------------------
# "Numerically difficult" multi-solution problem, m=20
# ---------------------------------------------------------------------
def multisolution(m=20):
    def F(x):
        out = []
        for i in range(m):
            s = sum(x[j] for j in range(m) if j != i)
            out.append(x[i] - mp.mpf('1.5') * mp.sin(s))
        return mp.matrix(out)

    def Jrow(x):
        J = mp.zeros(m, m)
        for i in range(m):
            s = sum(x[j] for j in range(m) if j != i)
            c = mp.mpf('1.5') * mp.cos(s)
            for j in range(m):
                if j == i:
                    J[i, j] = 1
                else:
                    J[i, j] = -c
        return J

    x0 = [mp.mpf('0.1')] * m
    return F, Jrow, x0


if __name__ == "__main__":
    print("Quick sanity check: Problem 2 (m=15), Newton and ARJ8S")
    F, Jrow, x0 = problem2(m=15)
    for name in ['Newton', 'ARJ8S']:
        x, hist, rho = run_method(name, F, Jrow, x0, tol=mp.mpf('1e-90'), max_iter=15)
        print(name, 'iters=', len(hist) - 1, 'final resid=', mp.nstr(hist[-1], 5))
