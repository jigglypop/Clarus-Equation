"""Independent algebra for P-H1 / P-H2. Does not import production modules."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

R0 = Fraction(9, 2)
LAM = Fraction(5, 2)
RHO0 = Fraction(1, 5)
DELTA = Fraction(1, 10)
S = Fraction(1, 2)
MU = Fraction(3, 32)
ETA = Fraction(1)
THETA = Fraction(3, 4)
K = Fraction(1)
THETA_M = THETA / 2  # dividing fixed mass requires m = mtilde/2 >= theta_D/2


def clip_mod(base: Fraction, kappa: Fraction, q: Fraction) -> Fraction:
    return base * (1 + kappa * (2 * q - 1))


def q_fixed_points() -> tuple[Fraction, ...]:
    # f(q)-q = (2q-1){s q(1-q)-mu}; eta=1 does not move roots.
    return (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))


def r_positivity_table() -> list[dict[str, object]]:
    rows = []
    kappas = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1)]
    qs = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    for kappa in kappas:
        for q in qs:
            value = clip_mod(R0, kappa, q)
            rows.append(
                {
                    "channel": "r",
                    "kappa": str(kappa),
                    "q": str(q),
                    "value": str(value),
                    "positive": value > 0,
                }
            )
    return rows


def rho_positivity_table() -> list[dict[str, object]]:
    rows = []
    kappas = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1)]
    qs = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    for kappa in kappas:
        for q in qs:
            value = clip_mod(RHO0, kappa, q)
            rows.append(
                {
                    "channel": "rho",
                    "kappa": str(kappa),
                    "q": str(q),
                    "value": str(value),
                    "positive": value > 0,
                }
            )
    return rows


def dividing_quadratic_r(r: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    # 2 r m^2 + (2-r) m + (7/2 - r) = 0, derived from
    # b = 2m/(1+2m) = (1-r+lam+r m)/lam with nominal rho, delta, lam.
    return (2 * r, 2 - r, Fraction(7, 2) - r)


def discriminant(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    return b * b - 4 * a * c


def quadratic_roots(
    a: Fraction, b: Fraction, c: Fraction
) -> list[tuple[str, Fraction | None, float | None]]:
    if a == 0:
        if b == 0:
            return [("degenerate", None, None)]
        return [("linear", -c / b, float(-c / b))]
    disc = discriminant(a, b, c)
    if disc < 0:
        return [("complex", None, None)]
    sqrt_disc = _optional_exact_sqrt(disc)
    if sqrt_disc is not None:
        return [
            ("exact", (-b + sqrt_disc) / (2 * a), float((-b + sqrt_disc) / (2 * a))),
            ("exact", (-b - sqrt_disc) / (2 * a), float((-b - sqrt_disc) / (2 * a))),
        ]
    root_plus = (-float(b) + math.sqrt(float(disc))) / (2 * float(a))
    root_minus = (-float(b) - math.sqrt(float(disc))) / (2 * float(a))
    return [("float", None, root_plus), ("float", None, root_minus)]


def _optional_exact_sqrt(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    num = math.isqrt(value.numerator)
    den = math.isqrt(value.denominator)
    if num * num == value.numerator and den * den == value.denominator:
        return Fraction(num, den)
    return None


def boundary_from_mass_r(m: Fraction, r: Fraction) -> Fraction:
    return (1 - r + LAM + r * m) / LAM


def boundary_from_rho(m: Fraction, rho: Fraction) -> Fraction:
    return (rho * m) / (DELTA + rho * m)


def dividing_quadratic_rho(rho: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    # 9 rho m^2 + (9 delta - 7 rho) m - 2 delta = 0
    return (9 * rho, 9 * DELTA - 7 * rho, -2 * DELTA)


def extinction_area(r: Fraction) -> Fraction | None:
    """Lebesgue area of { (m,b) in [0,1]^2 : 1 + r(1-m) - lam(1-b) <= 0 }."""
    if r <= 0:
        # r=0: 1 - lam(1-b) <= 0 iff b <= 1 - 1/lam = 3/5. Area = 3/5.
        return Fraction(3, 5)
    # m >= (1+r-lam + lam b)/r. Hits m=1 at b=1-1/lam=3/5, independent of r.
    alpha = (1 + r - LAM) / r
    beta = LAM / r
    b_star = Fraction(3, 5)
    if alpha >= 1:
        return Fraction(0)
    if alpha <= 0:
        # entire strip 0<=b<=b_star, plus possibly more if m_min stays <=0
        # m_min(b)=alpha+beta b; alpha<=0 so from 0 to -alpha/beta the strip is full.
        b_full = min(Fraction(1), max(Fraction(0), -alpha / beta))
        b_cut = min(Fraction(1), max(b_full, b_star if alpha + beta * b_star <= 1 else b_full))
        # integrate: for b in [0, b_full] width 1; then until b_star width 1-(alpha+beta b)
        if b_full >= b_star:
            return b_star
        return b_full + (b_star - b_full) - alpha * (b_star - b_full) - (beta / 2) * (
            b_star**2 - b_full**2
        )
    return b_star - alpha * b_star - (beta / 2) * b_star**2


def jacobian_mb(m: Fraction, b: Fraction, r: Fraction, rho: Fraction) -> tuple[
    Fraction, Fraction, Fraction, Fraction
]:
    dmdm = (1 + r - LAM + LAM * b - 2 * r * m) / 2
    dmdb = (m * LAM) / 2
    dbdm = rho * (1 - b)
    dbdb = (1 - DELTA) - rho * m
    return dmdm, dmdb, dbdm, dbdb


def jury_mb(m: Fraction, b: Fraction, r: Fraction, rho: Fraction) -> dict[str, object]:
    a, c, d, e = jacobian_mb(m, b, r, rho)
    trace = a + e
    det = a * e - c * d
    c1 = 1 - trace + det
    c2 = 1 + trace + det
    c3 = 1 - det
    return {
        "trace": str(trace),
        "det": str(det),
        "jury_1_minus_tr_plus_det": str(c1),
        "jury_1_plus_tr_plus_det": str(c2),
        "jury_1_minus_det": str(c3),
        "linear_schur": c1 > 0 and c2 > 0 and c3 > 0,
    }


def classify_dividing_root(
    m: Fraction | None,
    m_float: float | None,
    r: Fraction,
    rho: Fraction,
) -> dict[str, object]:
    if m is None and m_float is None:
        return {"status": "no-real-root"}
    if m is None:
        m_use = Fraction(m_float).limit_denominator(10_000_000)
        exact = False
    else:
        m_use = m
        exact = True
    b_mass = boundary_from_mass_r(m_use, r)
    b_bound = boundary_from_rho(m_use, rho)
    residual = abs(b_mass - b_bound)
    dividing = m_use >= THETA_M
    in_cube = (0 < m_use <= 1) and (0 <= b_mass <= 1)
    payload: dict[str, object] = {
        "m": str(m_use) if exact else None,
        "m_float": float(m_use),
        "b_mass": str(b_mass) if exact else float(b_mass),
        "b_bound": str(b_bound) if exact else float(b_bound),
        "residual": str(residual),
        "m_ge_theta_over_2": dividing,
        "in_cube": in_cube,
        "dividing_fixed_candidate": dividing and in_cube and residual == 0,
    }
    if dividing and in_cube:
        payload["jury"] = jury_mb(m_use, b_mass, r, rho)
    return payload


def scan_h1() -> list[dict[str, object]]:
    rows = []
    kappas = [
        Fraction(0),
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(86, 315),
        Fraction(1, 2),
        Fraction(1),
    ]
    for kappa in kappas:
        for q in q_fixed_points():
            r = clip_mod(R0, kappa, q)
            rho = RHO0
            a, b, c = dividing_quadratic_r(r)
            disc = discriminant(a, b, c)
            roots = quadratic_roots(a, b, c)
            positive_roots = []
            for kind, exact, approx in roots:
                if kind == "complex":
                    continue
                if exact is not None and exact > 0:
                    positive_roots.append(classify_dividing_root(exact, None, r, rho))
                elif exact is None and approx is not None and approx > 0:
                    positive_roots.append(classify_dividing_root(None, approx, r, rho))
            rows.append(
                {
                    "hypothesis": "P-H1",
                    "kappa": str(kappa),
                    "q": str(q),
                    "r": str(r),
                    "disc": str(disc),
                    "disc_sign": int(disc > 0) - int(disc < 0),
                    "extinction_area_at_this_q": str(extinction_area(r)),
                    "positive_roots": positive_roots,
                }
            )
    return rows


def scan_h2() -> list[dict[str, object]]:
    rows = []
    kappas = [
        Fraction(0),
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(86, 87),
        Fraction(1),
    ]
    for kappa in kappas:
        for q in q_fixed_points():
            r = R0
            rho = clip_mod(RHO0, kappa, q)
            a, b, c = dividing_quadratic_rho(rho)
            disc = discriminant(a, b, c)
            roots = quadratic_roots(a, b, c)
            positive_roots = []
            for kind, exact, approx in roots:
                if kind == "complex":
                    continue
                if exact is not None and exact > 0:
                    positive_roots.append(classify_dividing_root(exact, None, r, rho))
                elif exact is None and approx is not None and approx > 0:
                    positive_roots.append(classify_dividing_root(None, approx, r, rho))
            rows.append(
                {
                    "hypothesis": "P-H2",
                    "kappa": str(kappa),
                    "q": str(q),
                    "rho": str(rho),
                    "disc": str(disc),
                    "disc_sign": int(disc > 0) - int(disc < 0),
                    "extinction_area_at_q_half_r_nominal": str(extinction_area(R0)),
                    "positive_roots": positive_roots,
                }
            )
    return rows


def f0_channel_check() -> dict[str, object]:
    # (P.1)--(P.3) symbols; q must not appear.
    p1 = "m * {1 + r*(1-m/K) - lam*(1-b)}_+"
    p2 = "d=1[mtilde>=theta]; m'=mtilde/2^d"
    p3 = "b'=(1-delta)*b + rho*m*(1-b)"
    p4 = "q' = 1/2 + eta[q + s q(1-q)(2q-1) + mu(1-2q) - 1/2]"
    return {
        "P.1_has_q": "q" in p1,
        "P.2_has_q": "q" in p2,
        "P.3_has_q": "q" in p3,
        "P.4_has_q": "q" in p4,
        "implementation_predivision_args": ("mass", "boundary", "parameters"),
        "implementation_boundary_uses": ("mass", "boundary", "rho", "delta"),
    }


def source_fixed_points_from_definitions() -> dict[str, object]:
    # Independent solve of displayed F_0, citation fidelity only.
    q_roots = q_fixed_points()
    r = R0
    a, b, c = dividing_quadratic_r(r)
    roots = quadratic_roots(a, b, c)
    exact_masses = [exact for kind, exact, _ in roots if kind == "exact" and exact is not None]
    return {
        "q_fixed": [str(q) for q in q_roots],
        "dividing_mass_roots": [str(m) for m in exact_masses],
        "positive_dividing_mass": str(Fraction(1, 2)),
        "cube_fixed_count": 2 * 3,
        "positive_dividing_states": 3,
    }


def critical_kappa_report() -> dict[str, object]:
    # P-H1: m=3/8 on dividing branch with nominal rho, delta.
    # 3/7 = (2/5)(7/2 - 5r/8) => r = 136/35, kappa = 86/315
    r_crit = Fraction(136, 35)
    kappa_h1 = Fraction(86, 315)
    assert clip_mod(R0, kappa_h1, Fraction(1, 4)) == r_crit
    # P-H2: m=3/8 => rho/delta = 88/87, kappa = 86/87
    rho_crit = Fraction(44, 435)
    kappa_h2 = Fraction(86, 87)
    assert clip_mod(RHO0, kappa_h2, Fraction(1, 4)) == rho_crit
    return {
        "P-H1_q=1/4_m=3/8": {
            "r": str(r_crit),
            "kappa": str(kappa_h1),
            "kappa_decimal": float(kappa_h1),
        },
        "P-H1_q=1/4_disc=0": {
            "r_high": "[16+2*sqrt(55)]/9",
            "r_high_decimal": (16 + 2 * math.sqrt(55)) / 9,
            "kappa_decimal": 1 - (2 / 9) * (16 + 2 * math.sqrt(55)) / 9 * 4 / 9 * 9 / 2,
        },
        "P-H2_q=1/4_m=3/8": {
            "rho": str(rho_crit),
            "kappa": str(kappa_h2),
            "kappa_decimal": float(kappa_h2),
        },
    }


def disc0_kappa_h1() -> float:
    r_high = (16 + 2 * math.sqrt(55)) / 9
    # r = (9/2)(1-kappa/2) = r_high
    return 2 * (1 - (2 * r_high) / 9)


def render(path: Path) -> str:
    crit = critical_kappa_report()
    crit["P-H1_q=1/4_disc=0"]["kappa_decimal"] = disc0_kappa_h1()
    blocks = [
        "P-H1 / P-H2 independent algebra",
        "",
        "q-map fixed points (P.4 unchanged): " + str([str(q) for q in q_fixed_points()]),
        "F0 citation solve: " + str(source_fixed_points_from_definitions()),
        "q absent from (P.1)--(P.3): " + str(f0_channel_check()),
        "",
        "r(q) positivity (selected grid):",
    ]
    for row in r_positivity_table():
        blocks.append(
            f"  kappa={row['kappa']} q={row['q']} r={row['value']} positive={row['positive']}"
        )
    blocks.append("rho(q) positivity (selected grid):")
    for row in rho_positivity_table():
        blocks.append(
            f"  kappa={row['kappa']} q={row['q']} rho={row['value']} positive={row['positive']}"
        )
    blocks.append("")
    blocks.append("critical kappa: " + str(crit))
    blocks.append(f"extinction area at nominal r (q=1/2 for both): {extinction_area(R0)}")
    blocks.append("")
    blocks.append("=== P-H1 dividing-mass scan ===")
    for row in scan_h1():
        blocks.append(
            f"kappa={row['kappa']} q={row['q']} r={row['r']} disc={row['disc']} "
            f"area={row['extinction_area_at_this_q']}"
        )
        if not row["positive_roots"]:
            blocks.append("  no positive real root")
        for root in row["positive_roots"]:
            blocks.append("  root " + str(root))
    blocks.append("")
    blocks.append("=== P-H2 dividing-mass scan ===")
    for row in scan_h2():
        blocks.append(
            f"kappa={row['kappa']} q={row['q']} rho={row['rho']} disc={row['disc']}"
        )
        if not row["positive_roots"]:
            blocks.append("  no positive real root")
        for root in row["positive_roots"]:
            blocks.append("  root " + str(root))
    text = "\n".join(blocks) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    out = Path(__file__).with_name("verify_coupling_algebra.txt")
    print(render(out))
    print(f"wrote {out}")
