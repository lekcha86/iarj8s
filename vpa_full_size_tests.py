"""
Standalone, self-contained script: reproduces the base ARJ8S paper's own
Section 7 benchmark problems at their ORIGINAL full sizes (Problem 1 m=100,
Problem 2 m=500, elastic string n=500), at the same VPA precision (1000
digits) they used, for all 7 methods (Newton, ARJ8S, CTT8S, ZO8S, EH9S,
NLM8, I-ARJ8S).

This is the slow one. In the sandbox that built this project, a single
Newton iteration on Problem 1 at m=100 alone took ~9 seconds; the other six
methods each need 3-7 dense 1000-digit linear solves per iteration, and
Problem 2 / elastic string at their full n=500 size are roughly (500/30)^3
~= 4600x more expensive per linear solve than the m=30 reduced versions
used in the paper draft. Expect this to run for a long time (likely hours
for the full sweep) -- that is exactly why it was not run to completion in
the sandbox and is being handed off here.

Requires: mpmath (pip install mpmath)

Usage:
    python3 vpa_full_size_tests.py                    # runs everything
    python3 vpa_full_size_tests.py problem1            # just Problem 1
    python3 vpa_full_size_tests.py problem2 elastic     # a subset
    python3 vpa_full_size_tests.py --methods Newton ARJ8S I-ARJ8S problem1
    python3 vpa_full_size_tests.py --dps 500 problem1   # lower precision, faster

Available problem keys: problem1, problem2, hammerstein, multisolution, elastic

Notes:
  - Results are written incrementally to vpa_full_size_results.csv, one row
    per (problem, method) as soon as it finishes, so a crash or Ctrl-C
    partway through does not lose earlier results.
  - hammerstein and multisolution are already fast at their native sizes
    (n=7, m=20) -- included for completeness/consistency checking, not
    because they need a bigger machine.
  - If a method doesn't converge within max_iter, its row is still written
    (status='no_conv') rather than skipped, so you see genuine negative
    results the same way the paper draft does.
  - NLM8 is the one to watch for cost: it needs an explicit dense matrix
    inverse (Jx**-1) and two dense n x n matrix products every iteration,
    at 1000-digit precision. At n=500 this is extremely slow (likely the
    single slowest thing this script does) -- consider dropping it from
    --methods for Problem 2 / elastic string if you want the rest to finish
    sooner, and running it separately, or lowering --dps.
"""
import sys
import time
import csv
import argparse
import mpmath as mp


def norm(v):
    return mp.sqrt(sum(vi**2 for vi in v))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def run_method(name, F, Jrow, x0, tol, max_iter, log_prefix=""):
    n = len(x0)
    I = mp.eye(n)
    x = mp.matrix(x0)
    x_prev = mp.matrix(x0)
    t_prev = mp.mpf(1)
    hist = [norm(F(x))]

    def merit(Fv):
        return dot(Fv, Fv)

    for k in range(max_iter):
        t0 = time.time()
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
        elapsed = time.time() - t0
        print(f"{log_prefix}  iter {k+1}: ||F||={mp.nstr(r, 4)}  ({elapsed:.1f}s)", flush=True)
        if r < tol:
            break

    rho = None
    if len(hist) >= 3:
        e = hist
        try:
            num = mp.log(e[-1] / e[-2])
            den = mp.log(e[-2] / e[-3])
            if den != 0:
                rho = num / den
        except Exception:
            rho = None
    status = 'converged' if hist[-1] < tol else 'no_conv'
    return x, hist, rho, status


METHODS_ALL = ['Newton', 'ARJ8S', 'CTT8S', 'ZO8S', 'EH9S', 'NLM8', 'I-ARJ8S']


# ---------------------------------------------------------------------
# Problem 1: x_i - cos(-(sum x_j) + 2x_i) = 0, paper's m=100
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
                J[i, j] = (1 + mp.sin(g)) if i == j else -mp.sin(g)
        return J

    x0 = [mp.mpf('0.1')] * m
    return F, Jrow, x0


# ---------------------------------------------------------------------
# Problem 2: x_i^2 x_{i+1} - 1 = 0 (cyclic), paper's m=500
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
# Hammerstein integral equation, n=7 (already native size, fast)
# ---------------------------------------------------------------------
def hammerstein(n=7, x0val='0.5'):
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
    t = [(mp.mpf(v) + 1) / 2 for v in raw_nodes]
    w = [mp.mpf(v) / 2 for v in raw_weights]

    def F(x):
        out = []
        for i in range(n):
            s = mp.mpf(0)
            for k in range(n):
                c = w[k] * t[k] * (1 - t[i]) if k <= i else w[k] * t[i] * (1 - t[k])
                s += c * x[k]**3
            out.append(-5 + 5 * x[i] - s)
        return mp.matrix(out)

    def Jrow(x):
        J = mp.zeros(n, n)
        for i in range(n):
            for k in range(n):
                c = w[k] * t[k] * (1 - t[i]) if k <= i else w[k] * t[i] * (1 - t[k])
                J[i, k] = -3 * c * x[k]**2
            J[i, i] += 5
        return J

    x0 = [mp.mpf(x0val)] * n
    return F, Jrow, x0


# ---------------------------------------------------------------------
# Elastic string, paper's n=500
# ---------------------------------------------------------------------
def elastic_string(n=500, a=1.0, sample='alpha'):
    h = mp.mpf(1) / (n + 1)
    a = mp.mpf(a)

    def F(y):
        out = []
        for i in range(n):
            yim1 = y[i - 1] if i > 0 else mp.mpf(0)
            yip1 = y[i + 1] if i < n - 1 else mp.mpf(0)
            out.append(4 * (yip1 - 2 * y[i] + yim1) + 4 * h**2 + a**2 * (yip1 - yim1)**2)
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
# Multi-solution problem, m=20 (native size, fast)
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
                J[i, j] = 1 if j == i else -c
        return J

    x0 = [mp.mpf('0.1')] * m
    return F, Jrow, x0


PROBLEMS = {
    'problem1': lambda: (problem1(m=100), mp.mpf('1e-90'), 15),
    'problem2': lambda: (problem2(m=500), mp.mpf('1e-90'), 15),
    'hammerstein': lambda: (hammerstein(n=7), mp.mpf('1e-90'), 15),
    'multisolution': lambda: (multisolution(m=20), mp.mpf('1e-90'), 50),
    'elastic': lambda: (elastic_string(n=500, sample='alpha'), mp.mpf('1e-40'), 15),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='*', default=list(PROBLEMS.keys()),
                         help=f"which problems to run: {list(PROBLEMS.keys())}")
    parser.add_argument('--methods', nargs='*', default=METHODS_ALL,
                         help=f"which methods to run: {METHODS_ALL}")
    parser.add_argument('--dps', type=int, default=1000,
                         help="decimal digits of precision (paper uses 1000)")
    args = parser.parse_args()

    mp.mp.dps = args.dps

    csv_path = "vpa_full_size_results.csv"
    fieldnames = ['problem', 'method', 'status', 'iters', 'rho', 'resid', 'time']
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for pkey in args.problems:
        if pkey not in PROBLEMS:
            print(f"Unknown problem '{pkey}', skipping. Available: {list(PROBLEMS.keys())}")
            continue
        (F, Jrow, x0), tol, max_iter = PROBLEMS[pkey]()
        n = len(x0)
        print(f"\n{'='*70}\n{pkey} (n={n}), dps={mp.mp.dps}\n{'='*70}", flush=True)

        for name in args.methods:
            if name not in METHODS_ALL:
                print(f"Unknown method '{name}', skipping.")
                continue
            print(f"\n--- {pkey} / {name} ---", flush=True)
            t0 = time.time()
            x, hist, rho, status = run_method(name, F, Jrow, x0, tol, max_iter,
                                               log_prefix=f"[{pkey}/{name}]")
            elapsed = time.time() - t0
            row = dict(problem=pkey, method=name, status=status,
                       iters=len(hist) - 1,
                       rho=(mp.nstr(rho, 6) if rho else ''),
                       resid=mp.nstr(hist[-1], 6),
                       time=round(elapsed, 2))
            print(f"{pkey}/{name}: {status}, iters={row['iters']}, "
                  f"rho={row['rho']}, resid={row['resid']}, time={elapsed:.1f}s", flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print(f"\nAll done. Full results in {csv_path}")


if __name__ == "__main__":
    main()
