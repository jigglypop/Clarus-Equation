import argparse
import math
from dataclasses import dataclass


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def logspace(a_min: float, a_max: float, n: int) -> list[float]:
    if a_min <= 0.0 or a_max <= 0.0:
        raise ValueError("a_min and a_max must be > 0")
    if n <= 1:
        return [a_min]
    la = math.log(a_min)
    lb = math.log(a_max)
    step = (lb - la) / (n - 1)
    return [math.exp(la + i * step) for i in range(n)]


def simpson(y: list[float], x: list[float]) -> float:
    n = len(x)
    if n != len(y):
        raise ValueError("x and y length mismatch")
    if n < 2:
        return 0.0
    if n == 2:
        return 0.5 * (x[1] - x[0]) * (y[0] + y[1])
    if n % 2 == 0:
        n -= 1
        x = x[:n]
        y = y[:n]
        if n < 2:
            return 0.0
    h = (x[-1] - x[0]) / (n - 1)
    s = y[0] + y[-1]
    s_odd = 0.0
    s_even = 0.0
    for i in range(1, n - 1):
        if i % 2 == 1:
            s_odd += y[i]
        else:
            s_even += y[i]
    return (h / 3.0) * (s + 4.0 * s_odd + 2.0 * s_even)


@dataclass(frozen=True)
class Background:
    omega_m0: float
    omega_l0: float

    def e_of_a(self, a: float) -> float:
        return math.sqrt(self.omega_m0 * a ** (-3.0) + self.omega_l0)

    def dlnh_dln_a(self, a: float) -> float:
        e2 = self.omega_m0 * a ** (-3.0) + self.omega_l0
        num = -3.0 * self.omega_m0 * a ** (-3.0)
        return 0.5 * num / e2

    def omega_m_of_a(self, a: float) -> float:
        e2 = self.omega_m0 * a ** (-3.0) + self.omega_l0
        return (self.omega_m0 * a ** (-3.0)) / e2

    def omega_l_of_a(self, a: float) -> float:
        e2 = self.omega_m0 * a ** (-3.0) + self.omega_l0
        return self.omega_l0 / e2


def compute_s_of_a(bg: Background, a_grid: list[float]) -> list[float]:
    omegals = [bg.omega_l_of_a(a) for a in a_grid]
    denom = simpson(omegals, a_grid)
    if denom <= 0.0:
        return [0.0 for _ in a_grid]
    out = []
    for i in range(len(a_grid)):
        num = simpson(omegals[: i + 1], a_grid[: i + 1])
        out.append(clamp01(num / denom))
    return out

def compute_s_of_a_ratio(bg: Background, a_grid: list[float]) -> list[float]:
    omega_l0 = bg.omega_l_of_a(1.0)
    if omega_l0 <= 0.0:
        return [0.0 for _ in a_grid]
    out = []
    for a in a_grid:
        out.append(clamp01(bg.omega_l_of_a(a) / omega_l0))
    return out


def solve_growth(bg: Background, a_grid: list[float], mu_of_a: list[float]) -> tuple[list[float], list[float]]:
    if len(a_grid) != len(mu_of_a):
        raise ValueError("a_grid and mu_of_a length mismatch")
    if len(a_grid) < 3:
        raise ValueError("need at least 3 points")

    d = [0.0 for _ in a_grid]
    dp = [0.0 for _ in a_grid]

    a0 = a_grid[0]
    d[0] = a0
    dp[0] = a0

    lna = [math.log(a) for a in a_grid]
    dln = (lna[-1] - lna[0]) / (len(lna) - 1)

    def mu_at_ln_a(x: float) -> float:
        if x <= lna[0]:
            return mu_of_a[0]
        if x >= lna[-1]:
            return mu_of_a[-1]
        lo = 0
        hi = len(lna) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if lna[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0 = lna[lo]
        x1 = lna[hi]
        w = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
        return (1.0 - w) * mu_of_a[lo] + w * mu_of_a[hi]

    def rhs(x: float, d_val: float, dp_val: float) -> tuple[float, float]:
        a = math.exp(x)
        om = bg.omega_m_of_a(a)
        mu = mu_at_ln_a(x)
        term = 2.0 + bg.dlnh_dln_a(a)
        dd = dp_val
        ddp = -(term * dp_val) + 1.5 * om * mu * d_val
        return dd, ddp

    for i in range(len(a_grid) - 1):
        x = lna[i]
        k1 = rhs(x, d[i], dp[i])
        k2 = rhs(x + 0.5 * dln, d[i] + 0.5 * dln * k1[0], dp[i] + 0.5 * dln * k1[1])
        k3 = rhs(x + 0.5 * dln, d[i] + 0.5 * dln * k2[0], dp[i] + 0.5 * dln * k2[1])
        k4 = rhs(x + dln, d[i] + dln * k3[0], dp[i] + dln * k3[1])

        d[i + 1] = d[i] + (dln / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        dp[i + 1] = dp[i] + (dln / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])

    d1 = d[-1]
    if d1 == 0.0:
        d1 = 1.0
    dn = [v / d1 for v in d]
    fn = []
    for i in range(len(a_grid)):
        if dn[i] <= 0.0:
            fn.append(0.0)
        else:
            fn.append((dp[i] / d1) / dn[i])
    return dn, fn


def luminosity_distance_mpc(bg: Background, h0_km_s_mpc: float, z: float, n: int) -> float:
    if z <= 0.0:
        return 0.0
    c_km_s = 299792.458
    z_grid = linspace(0.0, z, n)
    integrand = []
    for zz in z_grid:
        a = 1.0 / (1.0 + zz)
        e = bg.e_of_a(a)
        integrand.append(1.0 / e)
    chi = simpson(integrand, z_grid)
    return (c_km_s / h0_km_s_mpc) * (1.0 + z) * chi


def h0_t0(bg: Background, a_min: float, n: int) -> float:
    if a_min <= 0.0:
        raise ValueError("a_min must be > 0")
    ln_a_grid = linspace(math.log(a_min), 0.0, n)
    integrand = []
    for ln_a in ln_a_grid:
        a = math.exp(ln_a)
        e = bg.e_of_a(a)
        integrand.append(1.0 / e)
    return simpson(integrand, ln_a_grid)


def main() -> int:
    p = argparse.ArgumentParser(prog="cosmology")
    p.add_argument("--model", choices=["epsilon", "calibrate"], default="epsilon")
    p.add_argument("--epsilon", type=float, default=1.0 / math.e)
    p.add_argument("--omega-lambda", type=float, default=0.685)
    p.add_argument("--omega-m", type=float, default=0.315)
    p.add_argument("--mu", choices=["lcdm", "sfe"], default="sfe")
    p.add_argument("--sdef", choices=["ratio", "cumulative"], default="ratio")
    p.add_argument("--epsilon-grav", type=float, default=0.0)
    p.add_argument("--sigma8-0", type=float, default=0.811)
    p.add_argument("--h0", type=float, default=67.4)
    p.add_argument("--zmax", type=float, default=2.0)
    p.add_argument("--nz", type=int, default=11)
    p.add_argument("--z-list", type=str, default="")
    p.add_argument("--na", type=int, default=2001)
    p.add_argument("--print-h0t0", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--compare-fsigma8", action="store_true")
    args = p.parse_args()

    if args.model == "epsilon":
        eps = args.epsilon
        omega_l0 = 0.5 * (1.0 + eps)
        omega_m0 = 0.5 * (1.0 - eps)
    else:
        omega_l0 = args.omega_lambda
        omega_m0 = args.omega_m

    s = omega_l0 + omega_m0
    if s <= 0.0:
        raise SystemExit("invalid density parameters")
    omega_l0 /= s
    omega_m0 /= s

    bg = Background(omega_m0=omega_m0, omega_l0=omega_l0)

    a_grid = logspace(1.0e-3, 1.0, args.na)
    if args.sdef == "ratio":
        s_grid = compute_s_of_a_ratio(bg, a_grid)
    else:
        s_grid = compute_s_of_a(bg, a_grid)

    if args.mu == "lcdm":
        mu_grid = [1.0 for _ in a_grid]
    else:
        eg = args.epsilon_grav
        mu_grid = [1.0 - eg * ss for ss in s_grid]

    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)

    if args.z_list.strip():
        z_grid = []
        for part in args.z_list.split(","):
            s_part = part.strip()
            if not s_part:
                continue
            z_grid.append(float(s_part))
        if not z_grid:
            z_grid = linspace(0.0, args.zmax, args.nz)
    else:
        z_grid = linspace(0.0, args.zmax, args.nz)
    print("model", args.model)
    print("omega_m0", f"{omega_m0:.9f}")
    print("omega_lambda0", f"{omega_l0:.9f}")
    print("mu", args.mu)
    print("sdef", args.sdef)
    print("epsilon_grav", f"{args.epsilon_grav:.9f}")
    print("sigma8_0", f"{args.sigma8_0:.9f}")
    print("h0", f"{args.h0:.6f}")
    if args.print_h0t0:
        print("h0_t0", f"{h0_t0(bg, a_min=1.0e-6, n=20001):.9f}")
    print("")
    if args.extended:
        print("z,E(z),D_L_Mpc,Omega_m(a),Omega_Lambda(a),S(a),mu(a),D(a),f(z),sigma8(z),f_sigma8(z)")
    else:
        print("z,E(z),D_L_Mpc,D(a),f(z),sigma8(z),f_sigma8(z)")

    d1 = d_norm[-1]
    if d1 == 0.0:
        d1 = 1.0

    for z in z_grid:
        a = 1.0 / (1.0 + z)
        idx = int(round((math.log(a) - math.log(a_grid[0])) / (math.log(a_grid[-1]) - math.log(a_grid[0])) * (len(a_grid) - 1)))
        if idx < 0:
            idx = 0
        if idx >= len(a_grid):
            idx = len(a_grid) - 1

        ez = bg.e_of_a(a)
        dl = luminosity_distance_mpc(bg, args.h0, z, n=2001)
        om = bg.omega_m_of_a(a)
        ol = bg.omega_l_of_a(a)
        ss = s_grid[idx]
        muu = mu_grid[idx]
        d = d_norm[idx]
        fz = f_ln[idx]
        s8 = args.sigma8_0 * d
        fs8 = fz * s8
        if args.extended:
            print(
                f"{z:.6f},"
                f"{ez:.9f},"
                f"{dl:.6f},"
                f"{om:.9f},"
                f"{ol:.9f},"
                f"{ss:.9f},"
                f"{muu:.9f},"
                f"{d:.9f},"
                f"{fz:.9f},"
                f"{s8:.9f},"
                f"{fs8:.9f}"
            )
        else:
            print(
                f"{z:.6f},"
                f"{ez:.9f},"
                f"{dl:.6f},"
                f"{d:.9f},"
                f"{fz:.9f},"
                f"{s8:.9f},"
                f"{fs8:.9f}"
            )

    if args.compare_fsigma8:
        targets = {0.32: 0.438, 0.57: 0.447, 0.70: 0.442}
        print("")
        print("fsigma8_compare(z,pred,target,delta,delta_percent)")
        for zt in [0.32, 0.57, 0.70]:
            a = 1.0 / (1.0 + zt)
            idx = int(round((math.log(a) - math.log(a_grid[0])) / (math.log(a_grid[-1]) - math.log(a_grid[0])) * (len(a_grid) - 1)))
            if idx < 0:
                idx = 0
            if idx >= len(a_grid):
                idx = len(a_grid) - 1
            pred = f_ln[idx] * (args.sigma8_0 * d_norm[idx])
            tgt = targets[zt]
            delta = pred - tgt
            pct = (delta / tgt) * 100.0 if tgt != 0.0 else 0.0
            print(f"{zt:.2f},{pred:.6f},{tgt:.6f},{delta:.6f},{pct:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


