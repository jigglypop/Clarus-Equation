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


def interp_linear(x_grid: list[float], y_grid: list[float], x: float) -> float:
    if len(x_grid) != len(y_grid):
        raise ValueError("x_grid and y_grid length mismatch")
    if not x_grid:
        raise ValueError("empty grid")
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    lo = 0
    hi = len(x_grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    if x1 == x0:
        return y_grid[lo]
    w = (x - x0) / (x1 - x0)
    return (1.0 - w) * y_grid[lo] + w * y_grid[hi]


def parse_fsigma8_triplets(spec: str) -> list[tuple[float, float, float]]:
    """
    Parse "z:fs8:sigma,z:fs8:sigma,..." into a list of triples.
    sigma can be 0 to indicate "unknown/ignored".
    """
    out: list[tuple[float, float, float]] = []
    s = spec.strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        fields = [f.strip() for f in p.split(":")]
        if len(fields) != 3:
            raise ValueError(f"invalid triplet '{p}': expected z:fs8:sigma")
        z = float(fields[0])
        fs8 = float(fields[1])
        sig = float(fields[2])
        if z < 0.0:
            raise ValueError("z must be >= 0")
        if sig < 0.0:
            raise ValueError("sigma must be >= 0")
        out.append((z, fs8, sig))
    out.sort(key=lambda t: t[0])
    return out


def is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


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


def make_mu_grid(s_grid: list[float], epsilon_grav: float) -> list[float]:
    return [1.0 - epsilon_grav * ss for ss in s_grid]


def predict_fsigma8_at_z(
    bg: Background,
    ln_a_grid: list[float],
    a_grid: list[float],
    s_grid: list[float],
    epsilon_grav: float,
    sigma8_0: float,
    z: float,
) -> float:
    mu_grid = make_mu_grid(s_grid, epsilon_grav)
    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)
    a = 1.0 / (1.0 + z)
    ln_a = math.log(a)
    d = interp_linear(ln_a_grid, d_norm, ln_a)
    fz = interp_linear(ln_a_grid, f_ln, ln_a)
    return fz * (sigma8_0 * d)


def calibrate_epsilon_grav_bisect(
    bg: Background,
    ln_a_grid: list[float],
    a_grid: list[float],
    s_grid: list[float],
    sigma8_0: float,
    z_cal: float,
    fs8_target: float,
    eps_min: float,
    eps_max: float,
    max_iter: int = 80,
    tol_abs: float = 1.0e-6,
) -> float:
    if eps_max <= eps_min:
        raise ValueError("invalid epsilon_grav bracket")
    if not (z_cal >= 0.0):
        raise ValueError("z_cal must be >= 0")

    def f(epsg: float) -> float:
        return predict_fsigma8_at_z(bg, ln_a_grid, a_grid, s_grid, epsg, sigma8_0, z_cal) - fs8_target

    f_lo = f(eps_min)
    f_hi = f(eps_max)
    if f_lo == 0.0:
        return eps_min
    if f_hi == 0.0:
        return eps_max
    if f_lo * f_hi > 0.0:
        raise ValueError("calibration target not bracketed by eps_min/eps_max")

    lo = eps_min
    hi = eps_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol_abs:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def main() -> int:
    p = argparse.ArgumentParser(prog="cosmology")
    p.add_argument("--model", choices=["epsilon", "calibrate"], default="epsilon")
    p.add_argument("--epsilon", type=float, default=1.0 / math.e)
    p.add_argument("--omega-lambda", type=float, default=0.685)
    p.add_argument("--omega-m", type=float, default=0.315)
    p.add_argument("--mu", choices=["lcdm", "sfe"], default="sfe")
    p.add_argument("--sdef", choices=["ratio", "cumulative"], default="ratio")
    p.add_argument("--epsilon-grav", type=float, default=0.0)
    p.add_argument("--calibrate-epsilon-grav", action="store_true")
    p.add_argument("--cal-z", type=float, default=float("nan"))
    p.add_argument("--cal-fsigma8", type=float, default=float("nan"))
    p.add_argument("--cal-pick", choices=["first", "last"], default="first")
    p.add_argument("--eps-min", type=float, default=-1.0)
    p.add_argument("--eps-max", type=float, default=1.0)
    p.add_argument("--fsigma8-data", type=str, default="")
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
    ln_a_grid = [math.log(a) for a in a_grid]
    if args.sdef == "ratio":
        s_grid = compute_s_of_a_ratio(bg, a_grid)
    else:
        s_grid = compute_s_of_a(bg, a_grid)

    cal_z_used = float("nan")
    cal_fsigma8_used = float("nan")
    cal_source = ""
    cal_z_source = ""

    epsilon_grav = args.epsilon_grav
    if args.calibrate_epsilon_grav:
        if args.mu != "sfe":
            raise SystemExit("--calibrate-epsilon-grav requires --mu sfe")
        triplets = parse_fsigma8_triplets(args.fsigma8_data)
        if math.isfinite(args.cal_z):
            cal_z_used = args.cal_z
            cal_z_source = "explicit"
        else:
            if not triplets:
                raise SystemExit("--calibrate-epsilon-grav requires --cal-z or non-empty --fsigma8-data")
            if args.cal_pick == "last":
                cal_z_used = triplets[-1][0]
            else:
                cal_z_used = triplets[0][0]
            cal_z_source = "fsigma8_data"

        cal_fsigma8 = args.cal_fsigma8
        cal_source = "explicit"
        if not math.isfinite(cal_fsigma8):
            z_tol = 5.0e-7
            for (zt, fs8, _sig) in triplets:
                if is_close(zt, cal_z_used, z_tol):
                    cal_fsigma8 = fs8
                    cal_source = "fsigma8_data"
                    break
        if not math.isfinite(cal_fsigma8):
            raise SystemExit("--calibrate-epsilon-grav requires --cal-fsigma8 or matching point in --fsigma8-data")
        cal_fsigma8_used = cal_fsigma8
        epsilon_grav = calibrate_epsilon_grav_bisect(
            bg=bg,
            ln_a_grid=ln_a_grid,
            a_grid=a_grid,
            s_grid=s_grid,
            sigma8_0=args.sigma8_0,
            z_cal=cal_z_used,
            fs8_target=cal_fsigma8,
            eps_min=args.eps_min,
            eps_max=args.eps_max,
        )

    mu_grid_lcdm = [1.0 for _ in a_grid]
    d_norm_lcdm, f_ln_lcdm = solve_growth(bg, a_grid, mu_grid_lcdm)

    if args.mu == "lcdm":
        mu_grid = mu_grid_lcdm
        d_norm = d_norm_lcdm
        f_ln = f_ln_lcdm
    else:
        mu_grid = make_mu_grid(s_grid, epsilon_grav)
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
    print("epsilon_grav", f"{epsilon_grav:.9f}")
    if args.calibrate_epsilon_grav:
        print("cal_z", f"{cal_z_used:.6f}")
        print("cal_fsigma8", f"{cal_fsigma8_used:.9f}")
        print("cal_z_source", cal_z_source)
        print("cal_source", cal_source)
    print("sigma8_0", f"{args.sigma8_0:.9f}")
    print("h0", f"{args.h0:.6f}")
    if args.print_h0t0:
        print("h0_t0", f"{h0_t0(bg, a_min=1.0e-6, n=20001):.9f}")
    print("")
    if args.extended:
        print("z,E(z),D_L_Mpc,Omega_m(a),Omega_Lambda(a),S(a),mu(a),D(a),f(z),sigma8(z),f_sigma8(z)")
    else:
        print("z,E(z),D_L_Mpc,D(a),f(z),sigma8(z),f_sigma8(z)")

    for z in z_grid:
        a = 1.0 / (1.0 + z)
        ln_a = math.log(a)

        ez = bg.e_of_a(a)
        dl = luminosity_distance_mpc(bg, args.h0, z, n=2001)
        om = bg.omega_m_of_a(a)
        ol = bg.omega_l_of_a(a)
        ss = interp_linear(ln_a_grid, s_grid, ln_a)
        muu = 1.0 if args.mu == "lcdm" else (1.0 - epsilon_grav * ss)
        d = interp_linear(ln_a_grid, d_norm, ln_a)
        fz = interp_linear(ln_a_grid, f_ln, ln_a)
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
        triplets = parse_fsigma8_triplets(args.fsigma8_data)
        legacy = False
        if not triplets:
            # Legacy illustrative values (no uncertainties). Prefer --fsigma8-data.
            legacy = True
            triplets = [(0.32, 0.438, 0.0), (0.57, 0.447, 0.0), (0.70, 0.442, 0.0)]
        print("")
        if legacy:
            print("fsigma8_compare_mode legacy")
        else:
            print("fsigma8_compare_mode data")
        print("fsigma8_compare(z,is_cal,pred,target,sigma,delta,delta_over_sigma,delta_percent)")
        chi2_all = 0.0
        n_all = 0
        chi2_holdout = 0.0
        n_holdout = 0
        chi2_lcdm_all = 0.0
        n_lcdm_all = 0
        chi2_lcdm_holdout = 0.0
        n_lcdm_holdout = 0
        n_cal_points = 0
        z_cal = cal_z_used if args.calibrate_epsilon_grav else float("nan")
        z_tol = 5.0e-7  # printing uses 6 decimals; treat same-point within this tolerance
        for (zt, tgt, sig) in triplets:
            is_cal = args.calibrate_epsilon_grav and math.isfinite(z_cal) and is_close(zt, z_cal, z_tol)
            if is_cal:
                n_cal_points += 1

            # Model prediction (may be lcdm or sfe depending on --mu).
            a = 1.0 / (1.0 + zt)
            ln_a = math.log(a)
            d_m = interp_linear(ln_a_grid, d_norm, ln_a)
            f_m = interp_linear(ln_a_grid, f_ln, ln_a)
            pred = f_m * (args.sigma8_0 * d_m)

            # Baseline prediction (always lcdm mu=1).
            d_b = interp_linear(ln_a_grid, d_norm_lcdm, ln_a)
            f_b = interp_linear(ln_a_grid, f_ln_lcdm, ln_a)
            pred_lcdm = f_b * (args.sigma8_0 * d_b)

            delta = pred - tgt
            pct = (delta / tgt) * 100.0 if tgt != 0.0 else 0.0
            if sig > 0.0:
                d_over_s = delta / sig
                chi2_all += d_over_s * d_over_s
                n_all += 1
                if not is_cal:
                    chi2_holdout += d_over_s * d_over_s
                    n_holdout += 1

                d_over_s_lcdm = (pred_lcdm - tgt) / sig
                chi2_lcdm_all += d_over_s_lcdm * d_over_s_lcdm
                n_lcdm_all += 1
                if not is_cal:
                    chi2_lcdm_holdout += d_over_s_lcdm * d_over_s_lcdm
                    n_lcdm_holdout += 1
                print(f"{zt:.6f},{1 if is_cal else 0:d},{pred:.9f},{tgt:.9f},{sig:.9f},{delta:.9f},{d_over_s:.6f},{pct:.3f}")
            else:
                print(f"{zt:.6f},{1 if is_cal else 0:d},{pred:.9f},{tgt:.9f},{sig:.9f},{delta:.9f},,{pct:.3f}")
        if args.calibrate_epsilon_grav and n_cal_points == 0:
            print("")
            print("fsigma8_note calibration_z_not_in_data")
        if n_all > 0:
            print("")
            dof_all = n_all
            if dof_all <= 0:
                dof_all = 1
            print("fsigma8_chi2_all", f"{chi2_all:.6f}")
            print("fsigma8_n_all", f"{n_all:d}")
            print("fsigma8_dof_all", f"{dof_all:d}")
            print("fsigma8_chi2_red_all", f"{(chi2_all / dof_all):.6f}")
            if args.calibrate_epsilon_grav and n_holdout > 0:
                dof_h = n_holdout
                if dof_h <= 0:
                    dof_h = 1
                print("fsigma8_chi2_holdout", f"{chi2_holdout:.6f}")
                print("fsigma8_n_holdout", f"{n_holdout:d}")
                print("fsigma8_dof_holdout", f"{dof_h:d}")
                print("fsigma8_chi2_red_holdout", f"{(chi2_holdout / dof_h):.6f}")

            # Baseline (lcdm mu=1) chi2 and delta-chi2 for easy comparison.
            if n_lcdm_all > 0:
                print("")
                print("fsigma8_baseline mu=1")
                print("fsigma8_chi2_lcdm_all", f"{chi2_lcdm_all:.6f}")
                print("fsigma8_n_lcdm_all", f"{n_lcdm_all:d}")
                print("fsigma8_dof_lcdm_all", f"{dof_all:d}")
                print("fsigma8_chi2_red_lcdm_all", f"{(chi2_lcdm_all / dof_all):.6f}")
                print("fsigma8_delta_chi2_all", f"{(chi2_all - chi2_lcdm_all):.6f}")
                print("fsigma8_delta_chi2_red_all", f"{((chi2_all / dof_all) - (chi2_lcdm_all / dof_all)):.6f}")
                if args.calibrate_epsilon_grav and n_lcdm_holdout > 0 and n_holdout > 0:
                    dof_h = n_holdout
                    if dof_h <= 0:
                        dof_h = 1
                    print("fsigma8_chi2_lcdm_holdout", f"{chi2_lcdm_holdout:.6f}")
                    print("fsigma8_n_lcdm_holdout", f"{n_lcdm_holdout:d}")
                    print("fsigma8_dof_lcdm_holdout", f"{dof_h:d}")
                    print("fsigma8_chi2_red_lcdm_holdout", f"{(chi2_lcdm_holdout / dof_h):.6f}")
                    print("fsigma8_delta_chi2_holdout", f"{(chi2_holdout - chi2_lcdm_holdout):.6f}")
                    print("fsigma8_delta_chi2_red_holdout", f"{((chi2_holdout / dof_h) - (chi2_lcdm_holdout / dof_h)):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


