"""Independent O-E1 / O-E2 / O-H1 enclosure. No production import.

Algebra copied from the predecessor verify_l3_algebra.py / (P.1)--(P.4)
with r(q) = r0 (1 + kappa (2q-1)), kappa = 1/4. q0 in {1/4, 3/4} is a
fixed point of the label map, so each pair is a planar hybrid map.

Branch rule: mtilde >= 3/4 divides (m <- mtilde/2); else no divide.
A mixed box is split, never wrapped into one hull.

Outward fixed-denominator rounding keeps bit-height bounded and is an
outer enclosure (not an exact image).
"""

from __future__ import annotations

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

R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))
CENTER = (Fraction(1, 2), Fraction(49, 99))

# Preregistered open boxes (closures used for enclosure; open U is subset).
U_BOXES = {
    "U0": (BC_M, BC_B),
    "U1": (
        (Fraction(43, 90), Fraction(47, 90)),
        (Fraction(431, 891), Fraction(451, 891)),
    ),
    "U2": (
        (Fraction(133, 270), Fraction(137, 270)),
        (Fraction(1313, 2673), Fraction(1333, 2673)),
    ),
    "U3": (
        (Fraction(403, 810), Fraction(407, 810)),
        (Fraction(3959, 8019), Fraction(3979, 8019)),
    ),
}

# Cited N-E1 contracting (m,b) slices at frozen q. Not re-proved.
QM_M = (Fraction(7, 18) - Fraction(1, 200), Fraction(7, 18) + Fraction(1, 200))
QM_B = (Fraction(7, 16) - Fraction(1, 200), Fraction(7, 16) + Fraction(1, 200))

SCALE = 10**18


def growth(q: Fraction) -> Fraction:
    return R0 * (1 + KAPPA * (2 * q - 1))


def q_next(q: Fraction) -> Fraction:
    return q + S * q * (1 - q) * (2 * q - 1) + MU * (1 - 2 * q)


def outward_lo(x: Fraction) -> Fraction:
    k = (x.numerator * SCALE) // x.denominator
    return Fraction(k, SCALE)


def outward_hi(x: Fraction) -> Fraction:
    k = (x.numerator * SCALE + x.denominator - 1) // x.denominator
    return Fraction(k, SCALE)


@dataclass(frozen=True)
class Box:
    mlo: Fraction
    mhi: Fraction
    blo: Fraction
    bhi: Fraction

    def empty(self) -> bool:
        return self.mlo > self.mhi or self.blo > self.bhi

    def width(self) -> Fraction:
        return max(self.mhi - self.mlo, self.bhi - self.blo)

    def outward(self) -> "Box":
        return Box(
            outward_lo(self.mlo),
            outward_hi(self.mhi),
            outward_lo(self.blo),
            outward_hi(self.bhi),
        )

    def quad_split(self) -> tuple["Box", ...]:
        mm = (self.mlo + self.mhi) / 2
        bm = (self.blo + self.bhi) / 2
        return (
            Box(self.mlo, mm, self.blo, bm),
            Box(mm, self.mhi, self.blo, bm),
            Box(self.mlo, mm, bm, self.bhi),
            Box(mm, self.mhi, bm, self.bhi),
        )


def raw_range(box: Box, r: Fraction) -> tuple[Fraction, Fraction]:
    """Exact coordinate range of m * (1 + r(1-m) - lam(1-b)) on a closed box.

    raw = (1+r-lam) m - r m^2 + lam m b.  No interior critical point for m>0
    (d raw / d b = lam m > 0).  On each b-edge the m-critical point is checked.
    """

    alpha = 1 + r - LAM
    vals: list[Fraction] = []
    for b in (box.blo, box.bhi):
        a = alpha + LAM * b
        for m in (box.mlo, box.mhi):
            vals.append(a * m - r * m * m)
        mcrit = a / (2 * r)
        if box.mlo <= mcrit <= box.mhi:
            vals.append(a * mcrit - r * mcrit * mcrit)
    return min(vals), max(vals)


def bnext_range(box: Box) -> tuple[Fraction, Fraction]:
    """b' = (1-delta)b + rho m (1-b).  Increasing in m and in b on the cube."""

    def bp(m: Fraction, b: Fraction) -> Fraction:
        return (1 - DELTA) * b + RHO * m * (1 - b)

    corners = (
        bp(box.mlo, box.blo),
        bp(box.mlo, box.bhi),
        bp(box.mhi, box.blo),
        bp(box.mhi, box.bhi),
    )
    return min(corners), max(corners)


def psi_bounds(mlo: Fraction, mhi: Fraction, r: Fraction) -> tuple[Fraction, Fraction]:
    """Outer bounds on the division curve b = psi(m) for m in [mlo, mhi]."""

    alpha = 1 + r - LAM
    lo = (THETA / mhi + r * mlo - alpha) / LAM
    hi = (THETA / mlo + r * mhi - alpha) / LAM
    return lo, hi


def classify(box: Box, r: Fraction) -> str:
    if box.mhi <= 0:
        return "nodiv"
    mt_lo, mt_hi = raw_range(box, r)
    if mt_lo >= THETA:
        return "div"
    if mt_hi < THETA:
        return "nodiv"
    return "mixed"


def image_single_branch(box: Box, r: Fraction, branch: str) -> Box:
    mt_lo, mt_hi = raw_range(box, r)
    if branch == "div":
        nlo, nhi = mt_lo / 2, mt_hi / 2
    elif branch == "nodiv":
        nlo, nhi = max(Fraction(0), mt_lo), max(Fraction(0), mt_hi)
    else:
        raise ValueError("image_single_branch forbids mixed")
    blo2, bhi2 = bnext_range(box)
    return Box(nlo, nhi, blo2, bhi2).outward()


def subset(inner: Box, outer_m, outer_b) -> bool:
    return (
        outer_m[0] <= inner.mlo
        and inner.mhi <= outer_m[1]
        and outer_b[0] <= inner.blo
        and inner.bhi <= outer_b[1]
    )


def disjoint_r0(box: Box) -> bool:
    return (
        box.mhi < R0_M[0]
        or box.mlo > R0_M[1]
        or box.bhi < R0_B[0]
        or box.blo > R0_B[1]
    )


def inside_r0(box: Box) -> bool:
    return subset(box, R0_M, R0_B)


def occupancy_of(box: Box) -> str:
    if inside_r0(box):
        return "in"
    if disjoint_r0(box):
        return "out"
    return "straddle"


def sqrt_bounds(n: int) -> tuple[Fraction, Fraction]:
    a = int(n**0.5)
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


def qplus_outer() -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """Cited N-E1 outer (m,b) box around Z_+ (nu=1/200).  Not a new LAS proof."""

    nu = Fraction(1, 200)
    s_lo, s_hi = sqrt_bounds(18601)
    m_lo = Fraction(49, 324) + s_lo / 324 - nu
    m_hi = Fraction(49, 324) + s_hi / 324 + nu
    b_lo = Fraction(-51, 160) + s_lo / 160 - nu
    b_hi = Fraction(-51, 160) + s_hi / 160 + nu
    return (m_lo, m_hi), (b_lo, b_hi)


def try_enclose(domain: Box, r: Fraction, ticks: int = 32):
    """Single-domain enclosure.  Mixed or fat hull => need_split.  No wrap."""

    h = domain
    divs = 0
    entered_qm = False
    entered_qp = False
    qp_m, qp_b = qplus_outer()
    r_minus = growth(Fraction(1, 4))
    r_plus = growth(Fraction(3, 4))
    for t in range(ticks):
        br = classify(h, r)
        if br == "mixed":
            return {
                "status": "need_split",
                "reason": "mixed",
                "t": t,
                "box": h,
                "psi": psi_bounds(h.mlo, h.mhi, r),
                "raw": raw_range(h, r),
            }
        if br == "div":
            divs += 1
        h = image_single_branch(h, r, br)
        # Q_- is a cited F_{1/4}-invariant slice.  Do not use it at q=3/4.
        if r == r_minus and subset(h, QM_M, QM_B):
            entered_qm = True
            rem = ticks - (t + 1)
            mt_lo, _ = raw_range(Box(QM_M[0], QM_M[1], QM_B[0], QM_B[1]), r)
            if mt_lo >= THETA:
                divs += rem
            return {
                "status": "ok",
                "occ": "out",
                "divs": (divs, divs),
                "t_entry": t + 1,
                "target": "Q-",
                "final": h,
                "entered_qm": True,
                "entered_qp": False,
            }
        # Q_+ outer AABB is not the cited invariant ball.  Record only.
        if r == r_plus and subset(h, qp_m, qp_b):
            entered_qp = True
    occ = occupancy_of(h)
    if occ == "straddle":
        return {
            "status": "need_split",
            "reason": "straddle",
            "t": ticks,
            "box": h,
            "divs": (divs, divs),
            "entered_qm": entered_qm,
            "entered_qp": entered_qp,
        }
    return {
        "status": "ok",
        "occ": occ,
        "divs": (divs, divs),
        "t_entry": None,
        "target": None,
        "final": h,
        "entered_qm": entered_qm,
        "entered_qp": entered_qp,
    }


def prove_box(name: str, domain: Box, r: Fraction, max_cells: int, min_width: Fraction):
    """Adaptive domain split.  Returns a decision or the blocking cut."""

    work = [domain]
    seen = 0
    splits = 0
    occs: set[str] = set()
    div_lo, div_hi = 32, 0
    mixed_cuts = []
    straddle_cuts = []
    entries = []
    while work:
        box = work.pop()
        seen += 1
        if seen > max_cells:
            return {
                "name": name,
                "decision": "undecided",
                "why": "max_cells",
                "seen": seen,
                "splits": splits,
                "occs": occs,
                "divs": (div_lo, div_hi),
                "mixed_cuts": mixed_cuts[:5],
                "straddle_cuts": straddle_cuts[:5],
            }
        res = try_enclose(box, r)
        if res["status"] == "ok":
            occs.add(res["occ"])
            d0, d1 = res["divs"]
            div_lo = min(div_lo, d0)
            div_hi = max(div_hi, d1)
            if res.get("t_entry") is not None:
                entries.append((res["target"], res["t_entry"]))
            continue
        if box.width() <= min_width:
            if res["reason"] == "mixed":
                mixed_cuts.append((box, res.get("t"), res.get("raw"), res.get("psi")))
            else:
                straddle_cuts.append((box, res.get("t"), res.get("box")))
            continue
        work.extend(box.quad_split())
        splits += 1
    if mixed_cuts or straddle_cuts:
        return {
            "name": name,
            "decision": "undecided",
            "why": "cut",
            "seen": seen,
            "splits": splits,
            "occs": occs,
            "divs": (div_lo, div_hi),
            "mixed_cuts": mixed_cuts[:8],
            "straddle_cuts": straddle_cuts[:8],
            "n_mixed": len(mixed_cuts),
            "n_straddle": len(straddle_cuts),
        }
    if len(occs) == 1:
        return {
            "name": name,
            "decision": "proved",
            "occ": next(iter(occs)),
            "seen": seen,
            "splits": splits,
            "divs": (div_lo, div_hi),
            "entries": entries[:8],
        }
    return {
        "name": name,
        "decision": "mixed_occ",
        "occs": occs,
        "seen": seen,
        "splits": splits,
        "divs": (div_lo, div_hi),
    }


def point_box(m: Fraction, b: Fraction, pad: Fraction) -> Box:
    return Box(m - pad, m + pad, b - pad, b + pad)


def first_step_report(box: Box, r: Fraction) -> dict:
    mt = raw_range(box, r)
    br = classify(box, r)
    psi = psi_bounds(box.mlo, box.mhi, r)
    return {"raw": mt, "branch": br, "psi": psi}


def subset_frac(inner_m, inner_b, outer_m, outer_b) -> bool:
    return (
        outer_m[0] <= inner_m[0]
        and inner_m[1] <= outer_m[1]
        and outer_b[0] <= inner_b[0]
        and inner_b[1] <= outer_b[1]
    )


def render(path: Path) -> str:
    lines: list[str] = []
    p = lines.append
    p("O-E2 open-set independent enclosure")
    p("preregister: artifacts/o_e2_preregister.md (written first)")
    p("no production import; no global continuity")
    p("")

    r_lo = growth(Fraction(1, 4))
    r_hi = growth(Fraction(3, 4))
    p(f"r(1/4)={r_lo} r(3/4)={r_hi}")
    p(f"q_next(1/4)={q_next(Fraction(1, 4))} q_next(3/4)={q_next(Fraction(3, 4))}")
    p(f"q frozen: {q_next(Fraction(1, 4)) == Fraction(1, 4) and q_next(Fraction(3, 4)) == Fraction(3, 4)}")

    qp_m, qp_b = qplus_outer()
    qminus_out = QM_M[1] < R0_M[0] or QM_B[1] < R0_B[0]
    qplus_in = subset(Box(qp_m[0], qp_m[1], qp_b[0], qp_b[1]), R0_M, R0_B)
    p(f"Q_- disjoint from R0 (m-hi={QM_M[1]} < 2/5): {QM_M[1] < R0_M[0]}")
    p(f"Q_+ outer subset R0: {qplus_in}")
    p(f"Q_+ outer m={qp_m} b={qp_b}")
    mt_qm = raw_range(Box(QM_M[0], QM_M[1], QM_B[0], QM_B[1]), r_lo)
    mt_qp = raw_range(Box(qp_m[0], qp_m[1], qp_b[0], qp_b[1]), r_hi)
    p(f"mtilde on Q_- at r(1/4): {mt_qm} div={mt_qm[0] >= THETA}")
    p(f"mtilde on Q_+ outer at r(3/4): {mt_qp} div={mt_qp[0] >= THETA}")
    p("")

    p("=== geometry (no trajectories) ===")
    for name, (mm, bb) in U_BOXES.items():
        in_bc = subset_frac(mm, bb, BC_M, BC_B)
        in_r0 = subset_frac(mm, bb, R0_M, R0_B)
        p(f"{name} subset Bc={in_bc} subset R0={in_r0} m={mm} b={bb}")
    p("")

    p("=== first-step branch on closures (exact raw range) ===")
    for name, (mm, bb) in U_BOXES.items():
        box = Box(mm[0], mm[1], bb[0], bb[1])
        for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
            rep = first_step_report(box, r)
            p(
                f"{name} {label} branch={rep['branch']} "
                f"raw=[{rep['raw'][0]}, {rep['raw'][1]}] "
                f"psi=[{rep['psi'][0]}, {rep['psi'][1]}]"
            )
    p("")

    # Center point: tiny pad, both labels.
    p("=== center pad enclosure (counterexample hunt) ===")
    for pad in (Fraction(1, 10**9), Fraction(1, 10**6), Fraction(1, 10**4)):
        pb = point_box(CENTER[0], CENTER[1], pad)
        for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
            res = try_enclose(pb, r)
            p(
                f"center pad={pad} {label} status={res['status']} "
                f"{ {k: res[k] for k in res if k in ('occ','divs','t_entry','target','reason','t')} }"
            )
    p("")

    # Interior sample of U3 (still a point enclosure, not an open-set proof).
    p("=== U3 corner+center pads (witness only) ===")
    u3m, u3b = U_BOXES["U3"]
    samples = [
        CENTER,
        ((u3m[0] + u3m[1]) / 2, (u3b[0] + u3b[1]) / 2),
        (u3m[0] + (u3m[1] - u3m[0]) / 10, u3b[0] + (u3b[1] - u3b[0]) / 10),
        (u3m[1] - (u3m[1] - u3m[0]) / 10, u3b[1] - (u3b[1] - u3b[0]) / 10),
    ]
    for m, b in samples:
        for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
            res = try_enclose(point_box(m, b, Fraction(1, 10**8)), r)
            p(
                f"pt({m},{b}) {label} status={res['status']} "
                f"{ {k: res[k] for k in res if k in ('occ','divs','t_entry','target','reason')} }"
            )
    p("")

    p("=== adaptive enclosure of registered closures ===")
    budgets = {
        "U3": (4000, Fraction(1, 10**7)),
        "U2": (4000, Fraction(1, 10**7)),
        "U1": (6000, Fraction(1, 10**6)),
        "U0": (6000, Fraction(1, 10**6)),
    }
    results = {}
    for name in ("U3", "U2", "U1", "U0"):
        mm, bb = U_BOXES[name]
        domain = Box(mm[0], mm[1], bb[0], bb[1])
        max_cells, min_w = budgets[name]
        for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
            key = f"{name}:{label}"
            res = prove_box(key, domain, r, max_cells, min_w)
            results[key] = res
            p(
                f"{key} decision={res['decision']} "
                f"occ={res.get('occ')} occs={res.get('occs')} "
                f"divs={res.get('divs')} seen={res.get('seen')} "
                f"splits={res.get('splits')} why={res.get('why')} "
                f"n_mixed={res.get('n_mixed')} n_straddle={res.get('n_straddle')}"
            )
            if res.get("mixed_cuts"):
                box, t, raw, psi = res["mixed_cuts"][0]
                p(
                    f"  first mixed cut t={t} raw={raw} psi={psi} "
                    f"m=[{box.mlo},{box.mhi}] b=[{box.blo},{box.bhi}]"
                )
            if res.get("straddle_cuts"):
                dbox, t, h = res["straddle_cuts"][0]
                p(
                    f"  first straddle t={t} domain_w={dbox.width()} "
                    f"hull_m=[{h.mlo},{h.mhi}] hull_b=[{h.blo},{h.bhi}]"
                )
            if res.get("entries"):
                p(f"  entries={res['entries'][:6]}")
    p("")

    p("=== pair summary ===")
    for name in ("U3", "U2", "U1", "U0"):
        a = results[f"{name}:q=1/4"]
        b = results[f"{name}:q=3/4"]
        if a["decision"] == "proved" and b["decision"] == "proved":
            occ_split = a.get("occ") != b.get("occ")
            count_split = a.get("divs")[0] == a.get("divs")[1] and b.get("divs")[0] == b.get("divs")[1] and a.get("divs")[0] != b.get("divs")[0]
            p(
                f"{name} BOTH PROVED occ=({a.get('occ')},{b.get('occ')}) "
                f"occ_split={occ_split} count_split={count_split} "
                f"divs=({a.get('divs')},{b.get('divs')})"
            )
        else:
            p(
                f"{name} NOT CLOSED lo={a['decision']}/{a.get('why')} "
                f"hi={b['decision']}/{b.get('why')}"
            )

    p("")
    p("=== T=32 hull without Q- shortcut (exact inclusions) ===")

    def trace_full(box: Box, r: Fraction, ticks: int = 32):
        h = box
        mixed = False
        divs = 0
        for _t in range(ticks):
            br = classify(h, r)
            if br == "mixed":
                mixed = True
                break
            if br == "div":
                divs += 1
            h = image_single_branch(h, r, br)
        return mixed, divs, h

    for name in ("U0", "U1", "U2", "U3"):
        mm, bb = U_BOXES[name]
        domain = Box(mm[0], mm[1], bb[0], bb[1])
        for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
            mixed, divs, h = trace_full(domain, r)
            p(
                f"{name} {label} mixed={mixed} divs={divs} "
                f"in_R0={inside_r0(h)} disjoint_R0={disjoint_r0(h)} "
                f"in_Q-={subset(h, QM_M, QM_B)} "
                f"m=[{h.mlo},{h.mhi}] b=[{h.blo},{h.bhi}]"
            )
            if label == "q=1/4":
                p(f"  mhi < 2/5: {h.mhi < R0_M[0]} ({h.mhi} < {R0_M[0]})")
            else:
                p(
                    f"  subset R0: {inside_r0(h)} "
                    f"mlo>=2/5={h.mlo >= R0_M[0]} mhi<=3/5={h.mhi <= R0_M[1]} "
                    f"blo>=4/9={h.blo >= R0_B[0]} bhi<=6/11={h.bhi <= R0_B[1]}"
                )

    p("")
    p("=== independent raw_range check (U0, both r) ===")
    u0 = Box(BC_M[0], BC_M[1], BC_B[0], BC_B[1])
    for label, r in (("q=1/4", r_lo), ("q=3/4", r_hi)):
        lo, hi = raw_range(u0, r)
        samples = []
        for i in range(11):
            for j in range(11):
                m = u0.mlo + (u0.mhi - u0.mlo) * i / 10
                b = u0.blo + (u0.bhi - u0.blo) * j / 10
                samples.append(m * (1 + r * (1 - m) - LAM * (1 - b)))
        slo, shi = min(samples), max(samples)
        p(
            f"U0 {label} exact=[{lo},{hi}] grid11=[{slo},{shi}] "
            f"grid_inside={lo <= slo and shi <= hi} above_theta={lo >= THETA}"
        )

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    out = Path(__file__).with_name("verify_o_e2_open_set.txt")
    print(render(out))
    print(f"wrote {out}")
