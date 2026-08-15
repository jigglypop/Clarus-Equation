"""Independent numerical evaluation of P-E1 coupling candidates.

This script is a route-lane scratch calculator. It does not import
production modules and does not declare closure.
"""

from __future__ import annotations

from fractions import Fraction
from math import sqrt

R0 = Fraction(9, 2)
LAM0 = Fraction(5, 2)
RHO0 = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
S = Fraction(1, 2)
MU = Fraction(3, 32)


def extinction_area(r: Fraction, lam: Fraction) -> Fraction | None:
    if r <= 0 or lam <= 0:
        return None
    m0 = (1 + r - lam) / r
    if m0 >= 1:
        return Fraction(0)
    bc = (lam - 1) / lam
    if bc <= 0:
        return Fraction(0)
    if bc > 1:
        bc = Fraction(1)
    return (1 / r) * ((lam - 1) * bc - lam * bc * bc / 2)


def quadratic_mass(growth, leak, rho):
    growth = Fraction(growth)
    leak = Fraction(leak)
    rho = Fraction(rho)
    c2 = growth * rho
    c1 = growth * (DELTA - rho) + rho
    c0 = DELTA * (1 + leak - growth)
    disc = c1 * c1 - 4 * c2 * c0
    return c2, c1, c0, disc


def solve_dividing(growth, leak, rho):
    c2, c1, c0, disc = quadratic_mass(growth, leak, rho)
    out = {
        "c2": c2,
        "c1": c1,
        "c0": c0,
        "disc": disc,
        "roots": [],
    }
    if c2 == 0 or disc < 0:
        return out
    sqrt_disc = sqrt(float(disc))
    for sign in (+1.0, -1.0):
        mass = (-float(c1) + sign * sqrt_disc) / (2.0 * float(c2))
        if not (0.0 < mass < 1.0):
            continue
        boundary = float(rho) * mass / (float(DELTA) + float(rho) * mass)
        gain = 1.0 + float(growth) * (1.0 - mass) - float(leak) * (1.0 - boundary)
        tilde = mass * gain
        out["roots"].append(
            {
                "m": mass,
                "b": boundary,
                "g": gain,
                "tilde": tilde,
                "divides": tilde + 1e-15 >= float(THETA),
            }
        )
    return out


def jury_at(mass, boundary, growth, leak, rho):
    j_mm = 1.0 - float(growth) * mass / 2.0
    j_mb = float(leak) * mass / 2.0
    j_bm = float(rho) * (1.0 - boundary)
    j_bb = 1.0 - float(DELTA) - float(rho) * mass
    trace = j_mm + j_bb
    det = j_mm * j_bb - j_mb * j_bm
    j1 = 1.0 - trace + det
    j2 = 1.0 + trace + det
    j3 = 1.0 - det
    disc = (trace / 2.0) ** 2 - det
    if disc >= 0:
        root1 = trace / 2.0 + sqrt(disc)
        root2 = trace / 2.0 - sqrt(disc)
        spectral = max(abs(root1), abs(root2))
    else:
        spectral = sqrt(j_mm * j_bb - j_mb * j_bm + (trace / 2.0) ** 2 * 0 + det)
        # complex pair: modulus^2 = det
        spectral = sqrt(abs(det))
    return {
        "J": ((j_mm, j_mb), (j_bm, j_bb)),
        "trace": trace,
        "det": det,
        "jury": (j1, j2, j3),
        "las_mb": j1 > 0 and j2 > 0 and j3 > 0,
        "spectral": spectral,
    }


def q_step(q):
    copied = q + S * q * (1 - q) * (2 * q - 1) + MU * (1 - 2 * q)
    return Fraction(1, 2) + (copied - Fraction(1, 2))


def predivision(mass, boundary, growth, leak):
    raw = mass * (1 + growth * (1 - mass) - leak * (1 - boundary))
    return max(0.0, float(raw))


def hybrid_step(mass, boundary, q, growth, leak, rho, theta=float(THETA), sigma=1.0):
    tilde = sigma * predivision(mass, boundary, growth, leak)
    divided = tilde >= theta
    next_m = tilde / (2.0 if divided else 1.0)
    next_b = (1.0 - float(DELTA)) * boundary + float(rho) * mass * (1.0 - boundary)
    next_q = float(q_step(Fraction(q).limit_denominator(10_000)))
    return next_m, next_b, next_q, divided, tilde


def iterate(mass, boundary, q, growth, leak, rho, steps=32, theta=float(THETA), sigma=1.0):
    history = []
    m, b, qq = float(mass), float(boundary), float(q)
    divisions = 0
    for _ in range(steps):
        m, b, qq, divided, tilde = hybrid_step(m, b, qq, growth, leak, rho, theta, sigma)
        divisions += int(divided)
        history.append((m, b, qq, divided, tilde))
        if m == 0.0:
            break
    in_r0 = (2 / 5 <= m <= 3 / 5) and (4 / 9 <= b <= 6 / 11)
    in_r1 = (5 / 12 <= m <= 7 / 12) and (5 / 11 <= b <= 7 / 13)
    return {
        "final": (m, b, qq),
        "divisions": divisions,
        "alive": m > 1e-12,
        "in_r0": in_r0,
        "in_r1": in_r1,
        "steps": len(history),
    }


def r0_tilde_range(growth, leak, sigma=1.0, n=41):
    masses = [2 / 5 + (3 / 5 - 2 / 5) * i / (n - 1) for i in range(n)]
    bounds = [4 / 9 + (6 / 11 - 4 / 9) * i / (n - 1) for i in range(n)]
    values = [
        sigma * predivision(m, b, growth, leak) for m in masses for b in bounds
    ]
    return min(values), max(values)


def report_roots(label, growth, leak, rho):
    sol = solve_dividing(growth, leak, rho)
    print(f"{label} disc={float(sol['disc']):.8f} nroots={len(sol['roots'])}")
    for root in sol["roots"]:
        ju = jury_at(root["m"], root["b"], growth, leak, rho)
        print(
            "  m={m:.8f} b={b:.8f} tilde={ti:.8f} div={dv} las={las} "
            "jury=({j1:.6f},{j2:.6f},{j3:.6f}) spec={sp:.6f}".format(
                m=root["m"],
                b=root["b"],
                ti=root["tilde"],
                dv=root["divides"],
                las=ju["las_mb"],
                j1=ju["jury"][0],
                j2=ju["jury"][1],
                j3=ju["jury"][2],
                sp=ju["spectral"],
            )
        )


print("source area", extinction_area(R0, LAM0), float(extinction_area(R0, LAM0)))
print("source roots")
report_roots("SOURCE", R0, LAM0, RHO0)
print("source jury at witness", jury_at(0.5, 0.5, R0, LAM0, RHO0))
print("source R0 tilde", r0_tilde_range(R0, LAM0))

print("\n=== H1 ===")
kappas = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 3), Fraction(2, 5), Fraction(1, 2), 1]
for kappa in kappas:
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(3, 4), "q+")):
        growth = R0 * (1 + kappa * (2 * q - 1))
        report_roots(f"k={kappa} {name} R={growth}", growth, LAM0, RHO0)
        lo, hi = r0_tilde_range(growth, LAM0)
        print(f"  R0 tilde [{lo:.6f},{hi:.6f}] min-theta={lo-float(THETA):.6f}")

print("\nH1 first failed low-q disc")
for i in range(0, 801):
    kappa = i / 2000
    growth = float(R0) * (1 + kappa * (0.5 - 1))
    disc = quadratic_mass(growth, LAM0, RHO0)[3]
    if disc < 0:
        print("k", kappa, "R", growth, "disc", float(disc))
        prev = (i - 1) / 2000
        print("prev k", prev, "R", float(R0) * (1 + prev * -0.5))
        break

print("\n=== H2 ===")
for kappa in (Fraction(1, 4), Fraction(1, 2), 1):
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(3, 4), "q+")):
        rho = RHO0 * (1 + kappa * (2 * q - 1))
        report_roots(f"k={kappa} {name} rho={rho}", R0, LAM0, rho)
        lo, hi = r0_tilde_range(R0, LAM0)
        print(f"  R0 tilde unchanged [{lo:.6f},{hi:.6f}]")

print("\n=== LEAK minus (high-q healthier) ===")
for kappa in (Fraction(1, 4), Fraction(1, 2), 1):
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(3, 4), "q+")):
        leak = LAM0 * (1 - kappa * (2 * q - 1))
        report_roots(f"k={kappa} {name} lam={leak}", R0, leak, RHO0)
        lo, hi = r0_tilde_range(R0, leak)
        print(f"  R0 tilde [{lo:.6f},{hi:.6f}] min-theta={lo-float(THETA):.6f}")
        print("  area@this-q", float(extinction_area(R0, leak)))

print("\n=== LEAK plus (contract-parallel sign) ===")
for kappa in (Fraction(1, 4), Fraction(1, 2), 1):
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(3, 4), "q+")):
        leak = LAM0 * (1 + kappa * (2 * q - 1))
        report_roots(f"k={kappa} {name} lam={leak}", R0, leak, RHO0)
        lo, hi = r0_tilde_range(R0, leak)
        print(f"  R0 tilde [{lo:.6f},{hi:.6f}] min-theta={lo-float(THETA):.6f}")

print("\n=== TWO-DAUGHTER retained-mass sigma=(1+p)/2, p=1-k(1-q) ===")


def sigma_of(kappa, q):
    p = 1 - float(kappa) * (1 - float(q))
    return (1 + p) / 2.0, p


def solve_sigma(kappa, q):
    sig, p = sigma_of(kappa, q)
    # dividing FP: sig * g0 = 2, with same b-nullcline
    # 1 + r(1-m) - lam(1-b) = 2/sig
    # This is equivalent to replacing the '2' target. Algebra:
    # r(1-m) - lam(1-b) = 2/sig - 1
    # Use a 1D search on m in (0,1)
    best = []
    for i in range(1, 4000):
        mass = i / 4000
        boundary = float(RHO0) * mass / (float(DELTA) + float(RHO0) * mass)
        g0 = 1 + float(R0) * (1 - mass) - float(LAM0) * (1 - boundary)
        residual = sig * g0 - 2
        if abs(residual) < 2e-4:
            tilde = sig * mass * g0
            best.append((mass, boundary, g0, tilde, residual, sig, p))
    # refine unique crossings
    return sig, p, best


for kappa in (Fraction(1, 4), Fraction(1, 2), 1):
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(1, 2), "q0"), (Fraction(3, 4), "q+")):
        sig, p, hits = solve_sigma(kappa, q)
        lo, hi = r0_tilde_range(R0, LAM0, sigma=sig)
        print(
            f"k={kappa} {name} p={p:.6f} sig={sig:.6f} "
            f"R0tilde=[{lo:.6f},{hi:.6f}] min-theta={lo-float(THETA):.6f} hits={len(hits)}"
        )
        if hits:
            mass, boundary, g0, tilde, residual, _, _ = hits[len(hits) // 2]
            print(
                f"  approx FP m={mass:.6f} b={boundary:.6f} tilde={tilde:.6f} res={residual:.2e}"
            )

print("\n=== THRESHOLD theta(q)=3/4-(k/4)(2q-1) ===")
for kappa in (Fraction(1, 4), Fraction(1, 2), 1):
    for q, name in ((Fraction(1, 4), "q-"), (Fraction(3, 4), "q+")):
        theta = float(THETA) - float(kappa) / 4 * float(2 * q - 1)
        lo, hi = r0_tilde_range(R0, LAM0)
        print(
            f"k={kappa} {name} theta={theta:.6f} source_witness_tilde=1 "
            f"still_divides={1 >= theta} R0min-theta={lo-theta:.6f}"
        )

print("\n=== T=32 occupancy from R0 corners and center ===")
points = [
    ("center", 0.5, 0.5),
    ("R0sw", 2 / 5, 4 / 9),
    ("R0ne", 3 / 5, 6 / 11),
    ("R0se", 3 / 5, 4 / 9),
    ("R0nw", 2 / 5, 6 / 11),
    ("R1c", 0.5, 6 / 12),
]


def run_pair(tag, growth_fn, leak_fn, rho_fn, theta_fn=None, sigma_fn=None):
    print(f"\n-- {tag} --")
    for kappa in (0, Fraction(1, 4), Fraction(1, 2), 1):
        for pname, m0, b0 in points:
            row = []
            for q in (0.25, 0.75):
                growth = growth_fn(kappa, q)
                leak = leak_fn(kappa, q)
                rho = rho_fn(kappa, q)
                theta = float(THETA) if theta_fn is None else theta_fn(kappa, q)
                sig = 1.0 if sigma_fn is None else sigma_fn(kappa, q)
                out = iterate(m0, b0, q, growth, leak, rho, 32, theta, sig)
                row.append(
                    "q={q} alive={al} inR0={r0} inR1={r1} div={dv} "
                    "final=({m:.4f},{b:.4f},{qq:.4f})".format(
                        q=q,
                        al=out["alive"],
                        r0=out["in_r0"],
                        r1=out["in_r1"],
                        dv=out["divisions"],
                        m=out["final"][0],
                        b=out["final"][1],
                        qq=out["final"][2],
                    )
                )
            differ = row[0].split("alive=")[1][:4] != row[1].split("alive=")[1][:4] or (
                "inR0=True" in row[0]
            ) != ("inR0=True" in row[1])
            print(f"k={kappa} {pname} differ? {differ}")
            for line in row:
                print("   ", line)


def h1_g(k, q):
    return float(R0) * (1 + float(k) * (2 * q - 1))


def h2_rho(k, q):
    return float(RHO0) * (1 + float(k) * (2 * q - 1))


def leak_m(k, q):
    return float(LAM0) * (1 - float(k) * (2 * q - 1))


def leak_p(k, q):
    return float(LAM0) * (1 + float(k) * (2 * q - 1))


def const_r(k, q):
    return float(R0)


def const_l(k, q):
    return float(LAM0)


def const_rho(k, q):
    return float(RHO0)


def th_fn(k, q):
    return float(THETA) - float(k) / 4 * (2 * q - 1)


def sig_fn(k, q):
    return sigma_of(k, q)[0]


run_pair("H1", h1_g, const_l, const_rho)
run_pair("H2", const_r, const_l, h2_rho)
run_pair("LEAKm", const_r, leak_m, const_rho)
run_pair("LEAKp", const_r, leak_p, const_rho)
run_pair("THETA", const_r, const_l, const_rho, theta_fn=th_fn)
run_pair("DAU", const_r, const_l, const_rho, sigma_fn=sig_fn)

print("\n=== q-map check (uncoupled) ===")
q = Fraction(1, 3)
for _ in range(8):
    q = q_step(q)
    print(float(q), end=" ")
print()
q = Fraction(2, 3)
for _ in range(8):
    q = q_step(q)
    print(float(q), end=" ")
print()

print("\n=== parameter-box probe for H1 at k=1/4 ===")
# theorem 7 box
box = {
    "r": (4.49, 4.51),
    "lam": (2.49, 2.51),
    "rho": (0.199, 0.201),
}
kappa = 0.25
for r in box["r"]:
    for lam in box["lam"]:
        for rho in box["rho"]:
            for q, name in ((0.25, "q-"), (0.75, "q+")):
                growth = r * (1 + kappa * (2 * q - 1))
                sol = solve_dividing(growth, lam, rho)
                ok = False
                las = False
                if sol["roots"]:
                    root = sol["roots"][0]
                    ju = jury_at(root["m"], root["b"], growth, lam, rho)
                    ok = root["divides"]
                    las = ju["las_mb"]
                print(
                    f"r={r} lam={lam} rho={rho} {name} n={len(sol['roots'])} "
                    f"div={ok} las={las} disc={float(sol['disc']):.5f}"
                )
