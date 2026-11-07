import mpmath as mp

mp.mp.dps = 800  # matches paper's VPA-style high precision

m = 15

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

def norm(v):
    return mp.sqrt(sum(vi**2 for vi in v))

x_star = mp.matrix([mp.mpf(1) for _ in range(m)])

def run_arj8s(x0, iters):
    x = mp.matrix(x0)
    hist = [x]
    for k in range(iters):
        Jk = Jrow(x)
        Fk = F(x)
        den = sum(Fk[i] * Fk[i] for i in range(m))
        if den == 0:
            break  # hit precision floor, residual underflowed to exact 0
        s1 = mp.lu_solve(Jk, -Fk)
        z1 = x + s1
        Fz1 = F(z1)
        num = sum(Fk[i] * Fz1[i] for i in range(m))
        alpha = 1 - 2 * num / den
        z_prev, Fz_prev = z1, Fz1
        for i in range(2, 5):
            s_i = mp.lu_solve(alpha * Jk, -Fz_prev)
            z_i = z_prev + s_i
            Fz_i = F(z_i)
            z_prev, Fz_prev = z_i, Fz_i
        x = z_prev
        hist.append(x)
    return hist

def run_newton(x0, iters):
    x = mp.matrix(x0)
    hist = [x]
    for k in range(iters):
        Jk = Jrow(x)
        Fk = F(x)
        s = mp.lu_solve(Jk, -Fk)
        x = x + s
        hist.append(x)
    return hist

x0 = [mp.mpf('3.0') for _ in range(m)]

histA = run_arj8s(x0, 6)
histN = run_newton(x0, 12)

def acoc_mp(hist, x_star):
    errs = [norm(h - x_star) for h in hist]
    orders = []
    for k in range(1, len(errs) - 1):
        if errs[k] > 0 and errs[k-1] > 0 and errs[k+1] > 0:
            num = mp.log(errs[k+1] / errs[k])
            den = mp.log(errs[k] / errs[k-1])
            if den != 0:
                orders.append(num / den)
    return errs, orders

errsA, ordersA = acoc_mp(histA, x_star)
errsN, ordersN = acoc_mp(histN, x_star)

print("ARJ8S true errors per iter:")
for i, e in enumerate(errsA):
    print(f"  iter {i}: {mp.nstr(e, 6)}")
print("ARJ8S ACOC:", [mp.nstr(o, 6) for o in ordersA])

print("\nNewton true errors per iter:")
for i, e in enumerate(errsN):
    print(f"  iter {i}: {mp.nstr(e, 6)}")
print("Newton ACOC:", [mp.nstr(o, 6) for o in ordersN])
