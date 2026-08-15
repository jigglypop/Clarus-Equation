"""Independent L7 algebra. No production import.

Preregister: artifacts/l7_preregister.md (written first; no occupancy).
This script checks only the new L7 reductions:
  - body indexing (S,A)=(L,R) and W=I drives
  - named-I loop I = o^A(beta), u^A = I * u_I(e^gamma)
  - frozen-sigma feedforward u^A = sigma * u_I(e^gamma)
  - sigma-overwrite sigma <- o^A(beta) is the same gamma gate
  - u=0 one-step extinction on closed B_c
It does not re-prove N-E1, N-E3, O-E1, L4-E1--E3, L5-E1--E3, or
L6-E1--E3 occupancy or one-step maps.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
R0_M = (Fraction(2, 5), Fraction(3, 5))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))
T = 32

E1 = (Fraction(1), Fraction(0))
E2 = (Fraction(0), Fraction(1))
W_I = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))

PHI1 = (E1, E2, E2)
PHI2 = (E1, E1, E2)


def matvec(weight, flux):
    return (
        weight[0][0] * flux[0] + weight[0][1] * flux[1],
        weight[1][0] * flux[0] + weight[1][1] * flux[1],
    )


def mtilde_full(mass, boundary, drive, growth=Fraction(81, 16)):
    raw = mass * (
        1 + drive * growth * (1 - mass) - LAM * (1 - boundary)
    )
    return raw if raw > 0 else Fraction(0)


def lines():
    out = []

    u_e1 = matvec(W_I, E1)
    u_e2 = matvec(W_I, E2)
    out.append(f"I e1 = {u_e1}")
    out.append(f"I e2 = {u_e2}")
    assert u_e1 == (Fraction(1), Fraction(0))
    assert u_e2 == (Fraction(0), Fraction(1))

    u_s_e1, u_a_e1 = u_e1
    u_s_e2, u_a_e2 = u_e2
    out.append(f"sensor u_I(e1) = {u_s_e1}")
    out.append(f"sensor u_I(e2) = {u_s_e2}")
    out.append(f"action u_I(e1) = {u_a_e1}")
    out.append(f"action u_I(e2) = {u_a_e2}")
    assert u_s_e1 == 1 and u_s_e2 == 0
    assert u_a_e1 == 0 and u_a_e2 == 1

    assert PHI1[0] == PHI2[0] == E1
    assert PHI1[2] == PHI2[2] == E2
    assert PHI1[1] == E2 and PHI2[1] == E1
    out.append("tasks share alpha=e1 and gamma=e2; differ only in beta")

    # Cited bits, not scored here:
    # alpha = e1 => sensor u = 1 => o^S = 1 by O-E1 / L5.
    sigma_after_alpha = 1
    out.append(f"cited sigma after alpha=e1 = {sigma_after_alpha}")

    # Epoch beta still uses the L5 gate.
    u_a_beta_phi1 = sigma_after_alpha * u_a_e2
    u_a_beta_phi2 = sigma_after_alpha * u_a_e1
    out.append(f"loop / overwrite u^A(beta, phi1) = {u_a_beta_phi1}")
    out.append(f"loop / overwrite u^A(beta, phi2) = {u_a_beta_phi2}")
    assert u_a_beta_phi1 == 1
    assert u_a_beta_phi2 == 0

    # Cited occupancy after beta, not a new hull:
    # u=1 on U0 x {3/4} => o^A = 1 by O-E1.
    # u=0 on B_c => o^A = 0 by one-step extinction (checked below).
    o_a_beta_phi1 = 1
    o_a_beta_phi2 = 0
    named_I_phi1 = o_a_beta_phi1
    named_I_phi2 = o_a_beta_phi2
    out.append(f"cited I(phi1) = o^A(beta) = {named_I_phi1}")
    out.append(f"cited I(phi2) = o^A(beta) = {named_I_phi2}")

    u_a_gamma_loop_phi1 = named_I_phi1 * u_a_e2
    u_a_gamma_loop_phi2 = named_I_phi2 * u_a_e2
    out.append(f"loop u^A(gamma, phi1) = {u_a_gamma_loop_phi1}")
    out.append(f"loop u^A(gamma, phi2) = {u_a_gamma_loop_phi2}")
    assert u_a_gamma_loop_phi1 == 1
    assert u_a_gamma_loop_phi2 == 0
    assert u_a_gamma_loop_phi1 != u_a_gamma_loop_phi2

    # Feedforward freezes sigma and ignores I.
    u_a_gamma_ff_phi1 = sigma_after_alpha * u_a_e2
    u_a_gamma_ff_phi2 = sigma_after_alpha * u_a_e2
    out.append(f"feedforward u^A(gamma, phi1) = {u_a_gamma_ff_phi1}")
    out.append(f"feedforward u^A(gamma, phi2) = {u_a_gamma_ff_phi2}")
    assert u_a_gamma_ff_phi1 == u_a_gamma_ff_phi2 == 1

    # H1: overwrite sigma <- o^A(beta), then L5 gate at gamma.
    sigma_ow_phi1 = o_a_beta_phi1
    sigma_ow_phi2 = o_a_beta_phi2
    u_a_gamma_ow_phi1 = sigma_ow_phi1 * u_a_e2
    u_a_gamma_ow_phi2 = sigma_ow_phi2 * u_a_e2
    out.append(f"overwrite u^A(gamma, phi1) = {u_a_gamma_ow_phi1}")
    out.append(f"overwrite u^A(gamma, phi2) = {u_a_gamma_ow_phi2}")
    assert u_a_gamma_ow_phi1 == u_a_gamma_loop_phi1
    assert u_a_gamma_ow_phi2 == u_a_gamma_loop_phi2
    out.append("overwrite gamma gate equals named-I gamma gate on both tasks")

    b_hi = BC_B[1]
    crit = 1 - 1 / LAM
    gap = crit - b_hi
    factor_at_bhi = 1 - LAM * (1 - b_hi)
    out.append(f"B_c b_hi = {b_hi}")
    out.append(f"u=0 sign change at b = 1 - 1/lambda = {crit}")
    out.append(f"crit - b_hi = {gap} > 0 : {gap > 0}")
    out.append(f"1 - lambda(1-b_hi) = {factor_at_bhi}")
    assert factor_at_bhi == Fraction(-53, 297)
    assert gap == Fraction(106, 1485)
    assert factor_at_bhi < 0
    assert gap > 0
    assert BC_M[0] > 0

    raw = BC_M[0] * (1 - LAM * (1 - b_hi))
    m1 = Fraction(0) if raw <= 0 else raw
    out.append(f"worst-case mtilde(u=0) on closed B_c = {m1}")
    assert m1 == 0
    assert m1 < THETA

    for drive in (Fraction(0), Fraction(1)):
        for boundary in (Fraction(0), b_hi, Fraction(1)):
            assert mtilde_full(Fraction(0), boundary, drive) == 0
    out.append("m=0 absorbs u in {0,1} at every tested b")

    mass = Fraction(0)
    boundary = b_hi
    for _ in range(T):
        mass = mtilde_full(mass, boundary, Fraction(1))
        boundary = (1 - DELTA) * boundary + RHO * mass * (1 - boundary)
    out.append(f"after T=32 from m=0 at u=1: m={mass}")
    assert mass == 0
    assert not (R0_M[0] <= mass <= R0_M[1])

    out.append("operators: loop gamma drives differ; feedforward gamma drives coincide")
    out.append("H1: overwrite uses the same two cubes and the same named slot")
    return out


def main() -> None:
    text = "\n".join(lines()) + "\n"
    out_path = Path(__file__).with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
