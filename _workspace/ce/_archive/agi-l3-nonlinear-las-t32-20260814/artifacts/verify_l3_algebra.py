"""Independent algebra for N-E1 / N-E2 / N-E3. Does not import production.

Exact Fraction is used only for one-step identities and Jacobian boxes.
T=32 orbits use inflated float intervals: unbounded Fraction iteration of
the quadratic map explodes bit-height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


R0 = Fraction(9, 2)
LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
S = Fraction(1, 2)
MU = Fraction(3, 32)
THETA = Fraction(3, 4)
KAPPA = Fraction(1, 4)
DISC = 18601

R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))


def growth(q: Fraction) -> Fraction:
    return R0 * (1 + KAPPA * (2 * q - 1))


def q_next(q: Fraction) -> Fraction:
    return q + S * q * (1 - q) * (2 * q - 1) + MU * (1 - 2 * q)


def predivision(m: Fraction, b: Fraction, r: Fraction) -> Fraction:
    raw = m * (1 + r * (1 - m) - LAM * (1 - b))
    return raw if raw > 0 else Fraction(0)


def step(m: Fraction, b: Fraction, q: Fraction):
    r = growth(q)
    mt = predivision(m, b, r)
    m2 = mt / 2 if mt >= THETA else mt
    b2 = (1 - DELTA) * b + RHO * m * (1 - b)
    return m2, b2, q_next(q)


def f_prime(q: Fraction) -> Fraction:
    return -3 * q * q + 3 * q + Fraction(5, 16)


def in_r0_f(m: float, b: float) -> bool:
    return 0.4 <= m <= 0.6 and float(R0_B[0]) <= b <= float(R0_B[1])


@dataclass(frozen=True)
class Qs:
    a: Fraction
    b: Fraction

    def __add__(self, other):
        o = _qs(other)
        return Qs(self.a + o.a, self.b + o.b)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        o = _qs(other)
        return Qs(self.a - o.a, self.b - o.b)

    def __rsub__(self, other):
        return _qs(other) - self

    def __neg__(self):
        return Qs(-self.a, -self.b)

    def __mul__(self, other):
        o = _qs(other)
        return Qs(self.a * o.a + self.b * o.b * DISC, self.a * o.b + self.b * o.a)

    def __rmul__(self, other):
        return self * other

    def inv(self):
        n = self.a * self.a - DISC * self.b * self.b
        return Qs(self.a / n, -self.b / n)

    def __truediv__(self, other):
        return self * _qs(other).inv()

    def __eq__(self, other):
        o = _qs(other)
        return self.a == o.a and self.b == o.b

    def sign(self) -> int:
        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.b > 0:
            if self.a >= 0:
                return 1
            return 1 if Fraction(DISC) > (self.a / self.b) ** 2 else -1
        if self.a <= 0:
            return -1
        return 1 if self.a * self.a > Fraction(DISC) * self.b * self.b else -1

    def __gt__(self, other):
        return (self - _qs(other)).sign() > 0

    def __ge__(self, other):
        return (self - _qs(other)).sign() >= 0

    def as_float(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(DISC)


def _qs(value) -> Qs:
    if isinstance(value, Qs):
        return value
    return Qs(Fraction(value), Fraction(0))


def step_qs(m: Qs, b: Qs, q: Fraction):
    r = growth(q)
    raw = m * (1 + r * (1 - m) - LAM * (1 - b))
    mt = raw if raw > 0 else _qs(0)
    m2 = mt / 2 if mt >= THETA else mt
    b2 = (1 - DELTA) * b + RHO * m * (1 - b)
    return m2, b2, q_next(q)


def sqrt_bounds(n: int) -> tuple[Fraction, Fraction]:
    a = math.isqrt(n)
    x = Fraction(2 * a + 1, 2)
    x = (x + n / x) / 2
    x = (x + n / x) / 2
    x = x.limit_denominator(10**9)
    if x * x <= n:
        lo = x
        hi = (Fraction(n) / x).limit_denominator(10**9)
        if hi * hi < n:
            hi = Fraction(a + 1)
    else:
        hi = x
        lo = (Fraction(n) / x).limit_denominator(10**9)
        if lo * lo > n:
            lo = Fraction(a)
    if lo * lo > n:
        lo = Fraction(a)
    if hi * hi < n:
        hi = Fraction(a + 1)
    return lo, hi


@dataclass(frozen=True)
class Ival:
    lo: Fraction
    hi: Fraction

    def __add__(self, other):
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        return Ival(self.lo + o.lo, self.hi + o.hi)

    def __sub__(self, other):
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        return Ival(self.lo - o.hi, self.hi - o.lo)

    def __mul__(self, other):
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        xs = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Ival(min(xs), max(xs))

    def __truediv__(self, other):
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        xs = (self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi)
        return Ival(min(xs), max(xs))

    def absbound(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))


def iv(value) -> Ival:
    if isinstance(value, Ival):
        return value
    v = Fraction(value)
    return Ival(v, v)


def box_rows(m_iv: Ival, b_iv: Ival, q_iv: Ival, w: Fraction, u: Fraction):
    """Weighted infinity-norm row sums of DF on a box, plus mtilde lower bound."""
    r_iv = iv(R0) * (iv(1) + iv(KAPPA) * (iv(2) * q_iv - 1))
    g_iv = iv(1) + r_iv * (iv(1) - m_iv) - iv(LAM) * (iv(1) - b_iv)
    j11 = g_iv / 2 - r_iv * m_iv / 2
    j12 = iv(LAM) * m_iv / 2
    j13 = (m_iv / 2) * (iv(1) - m_iv) * Fraction(9, 4)
    j21 = iv(RHO) * (iv(1) - b_iv)
    j22 = iv(1 - DELTA) - iv(RHO) * m_iv
    fp_vals = [f_prime(q_iv.lo), f_prime(q_iv.hi)]
    if q_iv.lo <= Fraction(1, 2) <= q_iv.hi:
        fp_vals.append(f_prime(Fraction(1, 2)))
    j33 = Ival(min(fp_vals), max(fp_vals))
    row1 = j11.absbound() + j12.absbound() / w + j13.absbound() / u
    row2 = w * j21.absbound() + j22.absbound()
    row3 = j33.absbound()
    mt_lo = (m_iv * g_iv).lo
    return row1, row2, row3, max(row1, row2, row3), mt_lo


def certify_minus(nu: Fraction, w: Fraction, u: Fraction) -> dict:
    m0, b0, q0 = Fraction(7, 18), Fraction(7, 16), Fraction(1, 4)
    m_iv = Ival(m0 - nu, m0 + nu)
    b_iv = Ival(b0 - nu, b0 + nu)
    q_iv = Ival(q0 - nu, q0 + nu)
    unit = 0 < m_iv.lo and m_iv.hi < 1 and 0 < b_iv.lo and b_iv.hi < 1
    unit = unit and 0 < q_iv.lo and q_iv.hi < 1
    row1, row2, row3, lip, mt_lo = box_rows(m_iv, b_iv, q_iv, w, u)
    return {
        "nu": nu,
        "w": w,
        "u": u,
        "unit": unit,
        "mt_lo": mt_lo,
        "div": mt_lo >= THETA,
        "lip": lip,
        "ok": unit and mt_lo >= THETA and lip < 1,
        "rows": (row1, row2, row3),
    }


def certify_plus(nu: Fraction, w: Fraction, u: Fraction) -> dict:
    m_star = Qs(Fraction(49, 324), Fraction(1, 324))
    b_star = Qs(Fraction(-51, 160), Fraction(1, 160))
    s_lo, s_hi = sqrt_bounds(DISC)
    m_lo = m_star.a + m_star.b * s_lo
    m_hi = m_star.a + m_star.b * s_hi
    b_lo = b_star.a + b_star.b * s_lo
    b_hi = b_star.a + b_star.b * s_hi
    m_iv = Ival(m_lo - nu, m_hi + nu)
    b_iv = Ival(b_lo - nu, b_hi + nu)
    q_iv = Ival(Fraction(3, 4) - nu, Fraction(3, 4) + nu)
    unit = 0 < m_iv.lo and m_iv.hi < 1 and 0 < b_iv.lo and b_iv.hi < 1
    unit = unit and 0 < q_iv.lo and q_iv.hi < 1
    row1, row2, row3, lip, mt_lo = box_rows(m_iv, b_iv, q_iv, w, u)
    return {
        "nu": nu,
        "w": w,
        "u": u,
        "unit": unit,
        "mt_lo": mt_lo,
        "div": mt_lo >= THETA,
        "lip": lip,
        "ok": unit and mt_lo >= THETA and lip < 1,
        "rows": (row1, row2, row3),
    }


def jury_mb(m: Fraction, b: Fraction, r: Fraction):
    g = 1 + r * (1 - m) - LAM * (1 - b)
    j11 = g / 2 - r * m / 2
    j12 = LAM * m / 2
    j21 = RHO * (1 - b)
    j22 = 1 - DELTA - RHO * m
    tr = j11 + j22
    det = j11 * j22 - j12 * j21
    return 1 - tr + det, 1 + tr + det, 1 - det


def extinction_area(r: Fraction) -> Fraction:
    alpha = (1 + r - LAM) / r
    beta = LAM / r
    b_star = Fraction(3, 5)
    return b_star - alpha * b_star - (beta / 2) * b_star**2


def registered_grid():
    nodes_m = [R0_M[0] + Fraction(i, 4) * (R0_M[1] - R0_M[0]) for i in range(5)]
    nodes_b = [R0_B[0] + Fraction(j, 4) * (R0_B[1] - R0_B[0]) for j in range(5)]
    return [(m, b) for m in nodes_m for b in nodes_b]


def step_float(m: float, b: float, q: float):
    r = float(R0) * (1.0 + float(KAPPA) * (2.0 * q - 1.0))
    raw = m * (1.0 + r * (1.0 - m) - float(LAM) * (1.0 - b))
    mt = raw if raw > 0.0 else 0.0
    m2 = mt / 2.0 if mt >= float(THETA) else mt
    b2 = (1.0 - float(DELTA)) * b + float(RHO) * m * (1.0 - b)
    q2 = q + float(S) * q * (1.0 - q) * (2.0 * q - 1.0) + float(MU) * (1.0 - 2.0 * q)
    return m2, b2, q2, mt


def iterate_float(m: float, b: float, q: float, ticks: int):
    divs = 0
    for _ in range(ticks):
        m, b, q, mt = step_float(m, b, q)
        if mt >= float(THETA):
            divs += 1
    return m, b, q, divs


def step_fint(mlo, mhi, blo, bhi, q: float):
    """Axis-aligned float hull of one step at frozen q, inflated."""
    r = float(R0) * (1.0 + float(KAPPA) * (2.0 * q - 1.0))
    lam = float(LAM)
    # gain = 1 + r(1-m) - lam(1-b); increasing in b, decreasing in m
    g_lo = 1.0 + r * (1.0 - mhi) - lam * (1.0 - blo)
    g_hi = 1.0 + r * (1.0 - mlo) - lam * (1.0 - bhi)
    raws = [mlo * g_lo, mlo * g_hi, mhi * g_lo, mhi * g_hi]
    raw_lo, raw_hi = min(raws), max(raws)
    mt_lo = max(0.0, raw_lo)
    mt_hi = max(0.0, raw_hi)
    if mt_lo >= float(THETA):
        nlo, nhi = mt_lo / 2.0, mt_hi / 2.0
        branch = "div"
    elif mt_hi < float(THETA):
        nlo, nhi = mt_lo, mt_hi
        branch = "nodiv"
    else:
        nlo = min(mt_lo, float(THETA) / 2.0)
        nhi = max(mt_hi, float(THETA) / 2.0)
        branch = "mixed"
    # b' = 0.9 b + 0.2 m (1-b); increasing in m and in b on the cube
    blo2 = 0.9 * blo + 0.2 * mlo * (1.0 - blo)
    bhi2 = 0.9 * bhi + 0.2 * mhi * (1.0 - bhi)
    if blo2 > bhi2:
        blo2, bhi2 = bhi2, blo2
    pad = 1e-12 * (1.0 + abs(nlo) + abs(nhi) + abs(blo2) + abs(bhi2)) + 1e-14
    return nlo - pad, nhi + pad, blo2 - pad, bhi2 + pad, branch, mt_lo, mt_hi


def enclose_t32_float(mlo, mhi, blo, bhi, q0: float, splits: int = 4):
    """Split the rectangle, inflate-float iterate 32 steps, return hull."""
    ms = [mlo + (mhi - mlo) * i / splits for i in range(splits + 1)]
    bs = [blo + (bhi - blo) * j / splits for j in range(splits + 1)]
    hull_m = [math.inf, -math.inf]
    hull_b = [math.inf, -math.inf]
    mixed = 0
    ncells = 0
    inverted = 0
    for i in range(splits):
        for j in range(splits):
            a, c = ms[i], ms[i + 1]
            d, e = bs[j], bs[j + 1]
            ncells += 1
            for _t in range(32):
                a, c, d, e, br, _mtlo, _mthi = step_fint(a, c, d, e, q0)
                if br == "mixed":
                    mixed += 1
                if a > c:
                    a, c = c, a
                    inverted += 1
                if d > e:
                    d, e = e, d
                    inverted += 1
            hull_m[0] = min(hull_m[0], a)
            hull_m[1] = max(hull_m[1], c)
            hull_b[0] = min(hull_b[0], d)
            hull_b[1] = max(hull_b[1], e)
    inside = (
        float(R0_M[0]) <= hull_m[0]
        and hull_m[1] <= float(R0_M[1])
        and float(R0_B[0]) <= hull_b[0]
        and hull_b[1] <= float(R0_B[1])
    )
    outside = (
        hull_m[1] < float(R0_M[0])
        or hull_m[0] > float(R0_M[1])
        or hull_b[1] < float(R0_B[0])
        or hull_b[0] > float(R0_B[1])
    )
    return {
        "hull_m": hull_m,
        "hull_b": hull_b,
        "ncells": ncells,
        "mixed": mixed,
        "inverted": inverted,
        "inside": inside,
        "outside": outside,
        "undecided": not inside and not outside,
    }


def render(path: Path) -> str:
    lines: list[str] = []
    p = lines.append
    p("L3 independent algebra (N-E1 / N-E2 / N-E3)")
    p("no production import; T=32 uses inflated float intervals")
    p("")

    r_minus = growth(Fraction(1, 4))
    r_plus = growth(Fraction(3, 4))
    r_half = growth(Fraction(1, 2))
    p(f"r(1/4)={r_minus} r(1/2)={r_half} r(3/4)={r_plus}")
    p(f"r(1/2)==r0: {r_half == R0}")

    zm, zb, zq = Fraction(7, 18), Fraction(7, 16), Fraction(1, 4)
    fm, fb, fq = step(zm, zb, zq)
    p(f"F(Z_-)=({fm},{fb},{fq}) fixed={fm==zm and fb==zb and fq==zq}")
    p(f"mtilde(Z_-)={predivision(zm, zb, r_minus)} dividing={predivision(zm,zb,r_minus)>=THETA}")

    a, bcoef, c = 2 * r_plus, 2 - r_plus, Fraction(7, 2) - r_plus
    disc = bcoef * bcoef - 4 * a * c
    p(f"plus quad ({a})m^2+({bcoef})m+({c})=0 disc={disc}")
    p(f"disc==18601/256: {disc == Fraction(18601, 256)}")

    mp = Qs(Fraction(49, 324), Fraction(1, 324))
    bp = Qs(Fraction(-51, 160), Fraction(1, 160))
    p(f"Z_+ m=(49+sqrt(18601))/324 ~ {mp.as_float()}")
    p(f"Z_+ b=(sqrt(18601)-51)/160 ~ {bp.as_float()}")
    p(f"b==2m/(1+2m): {(2 * mp) / (1 + 2 * mp) == bp}")
    fpm, fpb, fpq = step_qs(mp, bp, Fraction(3, 4))
    p(f"F(Z_+)==Z_+: {fpm == mp and fpb == bp and fpq == Fraction(3, 4)}")
    mt_p = mp * (1 + r_plus * (1 - mp) - LAM * (1 - bp))
    p(f"mtilde(Z_+)~{mt_p.as_float()} dividing={mt_p >= THETA}")

    c1, c2, c3 = jury_mb(zm, zb, r_minus)
    p(f"Jury(Z_-) {c1} {c2} {c3}")
    p(
        f"pred match {c1==Fraction(469,5760)} {c2==Fraction(12641,5760)} {c3==Fraction(331,384)}"
    )
    g_plus = 1 + r_plus * (1 - mp) - LAM * (1 - bp)
    j11 = g_plus / 2 - r_plus * mp / 2
    j12 = LAM * mp / 2
    j21 = RHO * (1 - bp)
    j22 = (1 - DELTA) - RHO * mp
    tr = j11 + j22
    det = j11 * j22 - j12 * j21
    c1p, c2p, c3p = 1 - tr + det, 1 + tr + det, 1 - det
    p(f"Jury(Z_+) pos {c1p>0} {c2p>0} {c3p>0} ~({c1p.as_float()},{c2p.as_float()},{c3p.as_float()})")
    p(f"f'(1/4)={f_prime(Fraction(1,4))} f'(3/4)={f_prime(Fraction(3,4))}")

    p("")
    p("=== N-E3 ===")
    area = extinction_area(r_half)
    p(f"area={area} ==1/10={area==Fraction(1,10)} >=1/20={area>=Fraction(1,20)}")
    # explicit integral pieces
    p("integral: int_0^{3/5} (1/3 - 5b/9) db = 1/5 - 1/10 = 1/10")

    p("")
    p("=== N-E1 cubes ===")
    weights = [
        (Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(2)),
        (Fraction(5, 4), Fraction(2)),
        (Fraction(5, 3), Fraction(2)),
        (Fraction(1), Fraction(3)),
        (Fraction(2), Fraction(2)),
        (Fraction(2), Fraction(3)),
        (Fraction(3, 2), Fraction(2)),
        (Fraction(8, 5), Fraction(5, 2)),
        (Fraction(9, 5), Fraction(3)),
    ]
    nus = [
        Fraction(1, 80),
        Fraction(1, 120),
        Fraction(1, 200),
        Fraction(1, 300),
        Fraction(1, 400),
        Fraction(1, 600),
        Fraction(1, 800),
    ]
    best_m = None
    for w, u in weights:
        for nu in nus:
            cert = certify_minus(nu, w, u)
            if cert["ok"]:
                p(
                    f"Z_- OK nu={nu} w={w} u={u} lip={cert['lip']} mt_lo={cert['mt_lo']}"
                )
                if best_m is None:
                    best_m = cert
                break
        else:
            # report the tightest failed attempt at smallest nu
            cert = certify_minus(nus[-1], w, u)
            p(
                f"Z_- fail w={w} u={u} nu={nus[-1]} div={cert['div']} "
                f"lip={cert['lip']} mt_lo={cert['mt_lo']}"
            )
    best_p = None
    for w, u in weights:
        for nu in nus:
            cert = certify_plus(nu, w, u)
            if cert["ok"]:
                p(
                    f"Z_+ OK nu={nu} w={w} u={u} lip={cert['lip']} mt_lo={cert['mt_lo']}"
                )
                if best_p is None:
                    best_p = cert
                break
        else:
            cert = certify_plus(nus[-1], w, u)
            p(
                f"Z_+ fail w={w} u={u} nu={nus[-1]} div={cert['div']} "
                f"lip={cert['lip']} mt_lo={cert['mt_lo']}"
            )
    p(f"certified minus={best_m is not None} plus={best_p is not None}")

    p("Z_+ 2D slice (q frozen at 3/4, not the cube claim)")
    m_star = Qs(Fraction(49, 324), Fraction(1, 324))
    b_star = Qs(Fraction(-51, 160), Fraction(1, 160))
    s_lo, s_hi = sqrt_bounds(DISC)
    m_lo = m_star.a + m_star.b * s_lo
    m_hi = m_star.a + m_star.b * s_hi
    b_lo = b_star.a + b_star.b * s_lo
    b_hi = b_star.a + b_star.b * s_hi
    for w in (Fraction(2), Fraction(3, 2), Fraction(8, 5)):
        for nu in (Fraction(1, 80), Fraction(1, 200), Fraction(1, 400)):
            m_iv = Ival(m_lo - nu, m_hi + nu)
            b_iv = Ival(b_lo - nu, b_hi + nu)
            q_iv = Ival(Fraction(3, 4), Fraction(3, 4))
            _r1, _r2, _r3, lip, mt_lo = box_rows(m_iv, b_iv, q_iv, w, Fraction(1))
            ok = mt_lo >= THETA and lip < 1
            p(f"  2D w={w} nu={nu} lip={lip} mt_lo={mt_lo} ok={ok}")
            if ok:
                break

    p("")
    p("=== N-E1 float probes (hunt) ===")
    grew = 0
    nprobe = 0
    for s in (0.05, 0.02, 0.01, 0.005):
        for h, k, pv in (
            (s, 0, 0),
            (-s, 0, 0),
            (0, s, 0),
            (0, -s, 0),
            (0, 0, s),
            (0, 0, -s),
            (s, s, 0),
            (s, 0, s),
            (-s, -s, 0),
        ):
            m = float(zm) + h
            b = float(zb) + k
            q = float(zq) + pv
            if not (0 <= m <= 1 and 0 <= b <= 1 and 0 <= q <= 1):
                continue
            nprobe += 1
            mt, bt, qt, _d = iterate_float(m, b, q, 64)
            d0 = max(abs(h), abs(k), abs(pv))
            dT = max(abs(mt - float(zm)), abs(bt - float(zb)), abs(qt - float(zq)))
            if dT > 2 * d0 and dT > 0.02:
                grew += 1
                p(f"GREW h={h} k={k} p={pv} dT={dT} end=({mt},{bt},{qt})")
    p(f"Z_- grew={grew}/{nprobe}")

    p("")
    p("=== N-E2 float grid T=32 (preregistered G) ===")
    occ = divs = both_in = both_out = low_in = high_in = 0
    for m, b in registered_grid():
        mlo, blo, _q, dlo = iterate_float(float(m), float(b), 0.25, 32)
        mhi, bhi, _q, dhi = iterate_float(float(m), float(b), 0.75, 32)
        ilo, ihi = in_r0_f(mlo, blo), in_r0_f(mhi, bhi)
        occ += int(ilo != ihi)
        divs += int(dlo != dhi)
        both_in += int(ilo and ihi)
        both_out += int((not ilo) and (not ihi))
        low_in += int(ilo)
        high_in += int(ihi)
        p(
            f"({m},{b}) lo_in={ilo} hi_in={ihi} div=({dlo},{dhi}) "
            f"lo=({mlo:.6f},{blo:.6f}) hi=({mhi:.6f},{bhi:.6f})"
        )
    p(
        f"|G|=25 occ_split={occ} div_split={divs} both_in={both_in} "
        f"both_out={both_out} low_in={low_in} high_in={high_in}"
    )

    p("")
    p("=== N-E2 B_c corners+center float T=32 ===")
    pts = [
        (BC_M[0], BC_B[0]),
        (BC_M[0], BC_B[1]),
        (BC_M[1], BC_B[0]),
        (BC_M[1], BC_B[1]),
        ((BC_M[0] + BC_M[1]) / 2, (BC_B[0] + BC_B[1]) / 2),
        (Fraction(1, 2), Fraction(49, 99)),
    ]
    for m, b in pts:
        mlo, blo, _q, dlo = iterate_float(float(m), float(b), 0.25, 32)
        mhi, bhi, _q, dhi = iterate_float(float(m), float(b), 0.75, 32)
        p(
            f"({m},{b}) lo_in={in_r0_f(mlo,blo)} hi_in={in_r0_f(mhi,bhi)} "
            f"div=({dlo},{dhi}) lo=({mlo:.6f},{blo:.6f}) hi=({mhi:.6f},{bhi:.6f})"
        )

    p("")
    p("=== N-E2 inflated-float enclosure T=32 ===")
    for name, mm, bb, splits in (
        ("Bc", BC_M, BC_B, 6),
        ("R0", R0_M, R0_B, 8),
    ):
        for q0 in (0.25, 0.75):
            enc = enclose_t32_float(float(mm[0]), float(mm[1]), float(bb[0]), float(bb[1]), q0, splits)
            p(
                f"{name} q0={q0} cells={enc['ncells']} mixed={enc['mixed']} "
                f"inv={enc['inverted']} "
                f"inside={enc['inside']} outside={enc['outside']} "
                f"undecided={enc['undecided']} "
                f"hull_m={enc['hull_m']} hull_b={enc['hull_b']}"
            )

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    out = Path(__file__).with_name("verify_l3_algebra.txt")
    print(render(out))
    print(f"wrote {out}")
