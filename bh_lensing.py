import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

M = 1.0
bc = 3 * np.sqrt(3) * M
folder = Path(__file__).resolve().parent


def rk4(u, v, h, m):
    k1u = v
    k1v = 3 * m * u * u - u
    u2 = u + 0.5 * h * k1u
    k2u = v + 0.5 * h * k1v
    k2v = 3 * m * u2 * u2 - u2
    u3 = u + 0.5 * h * k2u
    k3u = v + 0.5 * h * k2v
    k3v = 3 * m * u3 * u3 - u3
    u4 = u + h * k3u
    k4u = v + h * k3v
    k4v = 3 * m * u4 * u4 - u4
    un = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
    vn = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return un, vn


def euler(u, v, h, m):
    return u + h * v, v + h * (3 * m * u * u - u)


def step(u, v, h, m, method):
    if method == "rk4":
        return rk4(u, v, h, m)
    return euler(u, v, h, m)


def trace(b, h=0.002, method="rk4", r0=1000.0, rmax=2000.0, m=M, loc=True,
          maxphi=40 * np.pi, keep=False):
    u = 1 / r0
    v = np.sqrt(1 / b**2 - u**2 + 2 * m * u**3)
    phi = np.arcsin(b / r0)
    if m > 0:
        uh = 1 / (2 * m)
    else:
        uh = 1e30
    ue = 1 / rmax
    c0 = 1 / b**2
    drift = 0.0
    pu = [u]
    pp = [phi]
    status = 0
    while phi < maxphi:
        un, vn = step(u, v, h, m, method)
        d = abs(vn**2 + un**2 - 2 * m * un**3 - c0) * b**2
        if d > drift:
            drift = d
        if un >= uh:
            status = 1
            target = uh
        elif vn < 0 and un <= ue:
            status = 2
            target = ue
        if status != 0:
            if loc:
                lo = 0.0
                hi = 1.0
                for i in range(50):
                    mid = (lo + hi) / 2
                    ut, vt = step(u, v, h * mid, m, method)
                    if status == 1:
                        if ut < target:
                            lo = mid
                        else:
                            hi = mid
                    else:
                        if ut > target:
                            lo = mid
                        else:
                            hi = mid
                phi = phi + (lo + hi) / 2 * h
            else:
                phi = phi + h
            if keep:
                pu.append(target)
                pp.append(phi)
            break
        u = un
        v = vn
        phi = phi + h
        if keep:
            pu.append(u)
            pp.append(phi)
    return status, phi, drift, np.array(pu), np.array(pp)


def defl(b, **kw):
    rmax = kw.get("rmax", 2000.0)
    s, phi, d, pu, pp = trace(b, **kw)
    if s != 2:
        return np.nan
    return phi - np.pi + np.arcsin(b / rmax)


def find_bc(lo=4.0, hi=7.0, n=60, **kw):
    for i in range(n):
        mid = (lo + hi) / 2
        s = trace(mid, **kw)[0]
        if s == 1:
            lo = mid
        elif s == 2:
            hi = mid
        else:
            return (lo + hi) / 2, hi - lo, "unresolved"
    return (lo + hi) / 2, hi - lo, "ok"


def umax(b, m=M):
    lo = 0.0
    hi = 1 / (3 * m)
    for i in range(100):
        mid = (lo + hi) / 2
        p = 1 / b**2 - mid**2 + 2 * m * mid**3
        if p > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact(b, n=4000):
    um = umax(b)
    s = np.linspace(0, 1, n + 1)
    u = um * (1 - s**2)
    p = 1 / b**2 - u**2 + 2 * u**3
    g = np.zeros(n + 1)
    g[1:] = 2 * um * s[1:] / np.sqrt(p[1:])
    dp = abs(-2 * um + 6 * um**2)
    g[0] = 2 * np.sqrt(um / dp)
    h = 1 / n
    I = h / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum())
    return 2 * I - np.pi


def series(b):
    return 4 / b + 15 * np.pi / 4 / b**2 + 128 / 3 / b**3


def slope(x, y):
    return np.polyfit(np.log10(x), np.log10(y), 1)[0]


def fig_potential():
    r = np.linspace(2.05, 12, 500)
    V = (1 - 2 / r) / r**2
    plt.figure(figsize=(7, 4.5))
    plt.plot(r, V, label="V(r) = (1-2M/r)/r^2")
    plt.axhline(1 / 36, color="tab:blue", ls="--", label="1/b^2, b = 6M (turns back, escapes)")
    plt.axhline(1 / 20.25, color="tab:red", ls="--", label="1/b^2, b = 4.5M (goes over, captured)")
    plt.axhline(1 / 27, color="gray", ls=":", label="peak = 1/(27M^2), b = b_c")
    plt.axvline(3, color="gray", ls=":")
    plt.text(3.1, 0.005, "r = 3M")
    plt.xlabel("r / M")
    plt.ylabel("V(r)  (1/M^2)")
    plt.title("Effective potential barrier for light")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "fig_potential.png", dpi=170)
    plt.close()


def fig_paths():
    bs = [2.0, 4.0, 5.0, 5.15, bc * (1 - 1e-6), bc * (1 + 1e-6), 5.4, 5.8, 7.0, 10.0]
    plt.figure(figsize=(8, 8), facecolor="#0b1330")
    ax = plt.gca()
    ax.set_facecolor("#0b1330")
    for b in bs:
        s, phi, d, pu, pp = trace(b, h=0.002, r0=200.0, rmax=400.0, keep=True)
        r = 1 / pu
        x = -r * np.cos(pp)
        y = r * np.sin(pp)
        col = "#ff5c5c" if s == 1 else "#5cc8ff"
        ax.plot(x, y, color=col, lw=1.2)
        ax.plot(x, -y, color=col, lw=1.2)
    ax.add_patch(plt.Circle((0, 0), 2, color="black", zorder=5))
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, ec="white", lw=0.8, zorder=6))
    ax.add_patch(plt.Circle((0, 0), 3, fill=False, ec="#ffd166", ls="--", lw=1.2, zorder=6))
    ax.plot([], [], color="#ff5c5c", label="captured")
    ax.plot([], [], color="#5cc8ff", label="escaped")
    ax.plot([], [], color="#ffd166", ls="--", label="photon sphere r=3M")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / M", color="white")
    ax.set_ylabel("y / M", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Light rays around a Schwarzschild black hole", color="white")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(folder / "fig_paths.png", dpi=170, facecolor="#0b1330")
    plt.close()


print("critical impact parameter 3*sqrt(3)M =", bc)

print("\n--- 1. capture or escape ---")
for b in [1.0, 2.0, 3.0, 4.0, 5.0, 5.19, 5.1961, 5.1962, 5.2, 5.5, 8.0, 20.0]:
    s = trace(b)[0]
    print(b, {0: "unresolved", 1: "captured", 2: "escaped"}[s])

print("\n--- 2. finding bc by bisection on b ---")
print("method h  bc_found  error  last_bracket_width")
for method in ["rk4", "euler"]:
    if method == "rk4":
        hl = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    else:
        hl = [0.01, 0.005, 0.002, 0.001, 0.0005]
    for h in hl:
        x, w, st = find_bc(h=h, method=method)
        print(method, h, x, x - bc, w, st)
print("rk4 h=0.002 different start / escape radius")
for r0, rm in [(50.0, 100.0), (200.0, 400.0), (1000.0, 2000.0), (5000.0, 10000.0)]:
    x, w, st = find_bc(h=0.002, r0=r0, rmax=rm)
    print(r0, rm, x, x - bc)

print("\n--- 3. deflection angle ---")
print("b  numerical  exact  num-exact  4M/b  rel.err(4M/b)  3-term series")
bl = [5.3, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0]
for b in bl:
    a = defl(b)
    e = exact(b)
    print(b, a, e, a - e, 4 / b, (4 / b - e) / e, series(b))

print("\nexact() check with n=1000 and n=4000 at b=8:", exact(8.0, 1000), exact(8.0, 4000))

print("\n--- 4. convergence of rk4 and euler ---")
hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
hse = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625])
conv = {}
for b in [8.0, 6.0]:
    ex = exact(b)
    ref = defl(b, h=0.0005)
    print("b =", b, " finite domain gap |ref-exact| =", abs(ref - ex))
    e1 = np.array([abs(defl(b, h=h) - ref) for h in hs])
    e2 = np.array([abs(defl(b, h=h, loc=False) - ref) for h in hs])
    refe = defl(b, h=0.0005)
    e3 = np.array([abs(defl(b, h=h, method="euler") - refe) for h in hse])
    print(" h  err_rk4  err_rk4_noloc")
    for i in range(len(hs)):
        print(" ", hs[i], e1[i], e2[i])
    print(" h  err_euler")
    for i in range(len(hse)):
        print(" ", hse[i], e3[i])
    print(" slope rk4 (h 0.2..0.0125):", slope(hs[:5], e1[:5]))
    print(" slope rk4 no localisation:", slope(hs[:5], e2[:5]))
    print(" slope euler:", slope(hse, e3))
    conv[b] = (e1, e2, e3)

print("\ncost for the same accuracy about 1e-6 at b=8 (derivative calls)")
for method, hh in [("rk4", 0.1), ("rk4", 0.05), ("euler", 0.0005), ("euler", 0.00025)]:
    a = defl(8.0, h=hh, method=method)
    ref = defl(8.0, h=0.0005)
    steps = np.pi / hh
    calls = steps * (4 if method == "rk4" else 1)
    print(method, hh, abs(a - ref), "approx derivative calls", int(calls))

print("\nfirst integral drift (v^2+u^2-2Mu^3 - 1/b^2)*b^2, b=8")
for h in [0.1, 0.05, 0.025, 0.0125]:
    print(h, trace(8.0, h=h)[2])

print("\n--- 5. near critical ---")
eps = 10.0 ** -np.arange(1, 11)
print("eps  alpha  alpha/pi  status")
al = []
for e in eps:
    s, phi, d, pu, pp = trace(bc * (1 + e), h=0.001)
    a = phi - np.pi + np.arcsin(bc * (1 + e) / 2000.0)
    al.append(a)
    print(e, a, a / np.pi, s)
al = np.array(al)
k = (eps <= 1e-3) & (eps >= 1e-8)
fit = np.polyfit(np.log(eps[k]), al[k], 1)
bozza = np.log(216 * (7 - 4 * np.sqrt(3))) - np.pi
print("fit slope", fit[0], "(theory -1)   fit intercept", fit[1], "(theory", bozza, ")")

print("\nsame eps, different h : status")
for e in [1e-3, 1e-6, 1e-9, 1e-11, 1e-13]:
    row = []
    for h in [0.05, 0.01, 0.002]:
        row.append(trace(bc * (1 + e), h=h)[0])
    row2 = []
    for h in [0.05, 0.01, 0.002]:
        row2.append(trace(bc * (1 - e), h=h)[0])
    print(e, "above bc:", row, "below bc:", row2)

print("\nangle budget test at b = bc(1+1e-6), 0 unresolved 1 captured 2 escaped")
for mp in [2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi, 6 * np.pi]:
    print(mp / np.pi, "pi", trace(bc * (1 + 1e-6), maxphi=mp)[0])

print("\nfloating point value of bc:", trace(bc)[0], trace(np.nextafter(bc, 0))[0], trace(np.nextafter(bc, 10))[0])

print("\n--- 6. finite start and end radius, b = 8 ---")
ex = exact(8.0)
rl = np.array([20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0])
er = []
for r in rl:
    a = defl(8.0, h=0.002, r0=r, rmax=2 * r)
    er.append(abs(a - ex))
    print(r, a, a - ex)
er = np.array(er)
print("slope of |error| vs r0 (r0>=50):", slope(rl[1:6], er[1:6]))
print("different b with r0=200 rmax=400:")
for b in [8.0, 15.0, 30.0, 50.0]:
    print(b, defl(b, r0=200.0, rmax=400.0) - exact(b))

print("\n--- 7. other checks ---")
print("no mass (M=0) deflection at b=8, should be 0:", defl(8.0, m=0.0))
print("repeat run same:", defl(8.0), defl(8.0), defl(8.0) == defl(8.0))

fig_potential()
fig_paths()

bb = np.concatenate((bc * (1 + 10.0 ** np.linspace(-6, -1, 40)), np.linspace(6.0, 30.0, 60)))
bb = np.unique(bb)
an = np.array([defl(b) for b in bb])
ae = np.array([exact(b) if b > 5.5 else np.nan for b in bb])
plt.figure(figsize=(8, 5))
plt.semilogy(bb, an, "o", ms=3, label="RK4 numerical")
plt.semilogy(bb, 4 / bb, "--", label="4M/b")
plt.semilogy(bb, series(bb), ":", label="3-term series")
plt.axvline(bc, color="r", ls=":", label="b_c")
plt.xlabel("b / M")
plt.ylabel("deflection angle (rad)")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Light bending vs impact parameter")
plt.tight_layout()
plt.savefig(folder / "fig_deflection.png", dpi=170)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for b, c in [(8.0, "C0"), (6.0, "C1")]:
    e1, e2, e3 = conv[b]
    ax[0].loglog(hs, e1, "o-", color=c, label="RK4 b=%g" % b)
    ax[0].loglog(hs, e2, "s--", color=c, alpha=0.6, label="RK4 no localisation b=%g" % b)
    ax[0].loglog(hse, e3, "^:", color=c, alpha=0.9, label="Euler b=%g" % b)
ax[0].set_xlabel("step h (rad)")
ax[0].set_ylabel("|error in deflection|")
ax[0].set_title("Error vs step size")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3, which="both")
ax[1].loglog(rl, er, "o-")
ax[1].set_xlabel("start radius r0 / M  (escape radius = 2 r0)")
ax[1].set_ylabel("|numerical - exact|")
ax[1].set_title("Finite start / escape radius")
ax[1].grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(folder / "fig_convergence.png", dpi=170)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(np.log(eps), al, "o", label="numerical")
xx = np.linspace(np.log(eps[-1]), np.log(eps[0]), 50)
plt.plot(xx, -xx + bozza, "--", label="-ln(b/bc-1) + ln[216(7-4sqrt3)] - pi")
plt.xlabel("ln(b/bc - 1)")
plt.ylabel("deflection angle (rad)")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Deflection near the critical impact parameter")
plt.tight_layout()
plt.savefig(folder / "fig_near_critical.png", dpi=170)
plt.close()
