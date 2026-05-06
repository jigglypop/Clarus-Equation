"""Audit the phase-area coefficient pi^2/2 used in horizon entropy lift.

The goal is to check whether the coefficient can be tied to an existing CE
phase-space rule rather than being a free numerical fit.

Two candidate constructions are compared:
    1. half-cycle ordered phase area: int_0^pi theta dtheta = pi^2/2
    2. adjoint-normalized full phase square: (2pi)^2 / (d^2 - 1)

They coincide exactly only for d=3.
"""

from __future__ import annotations

import math


def adjoint_phase_measure(d: int) -> float:
    return (2.0 * math.pi) ** 2 / (d * d - 1.0)


def rel_error(pred: float, ref: float) -> float:
    return 100.0 * (pred / ref - 1.0)


def main() -> int:
    half_cycle_area = 0.5 * math.pi * math.pi

    print("# Phase-Area Principle Gate")
    print()
    print("## Two constructions")
    print()
    print(f"half-cycle ordered area = integral_0^pi theta dtheta = {half_cycle_area:.8f}")
    print("adjoint-normalized phase square = (2pi)^2/(d^2-1)")
    print()

    print("| d | d^2-1 | (2pi)^2/(d^2-1) | error vs pi^2/2 |")
    print("|---:|---:|---:|---:|")
    for d in range(2, 8):
        measure = adjoint_phase_measure(d)
        print(f"| {d} | {d*d-1} | {measure:.8f} | {rel_error(measure, half_cycle_area):+.3f}% |")
    print()

    d_solution = math.sqrt(1.0 + (2.0 * math.pi) ** 2 / half_cycle_area)
    print("## Equality condition")
    print()
    print("(2pi)^2/(d^2-1) = pi^2/2")
    print(f"d = sqrt(1 + 8) = {d_solution:.8f}")
    print()

    print("## Interpretation")
    print()
    print("For d=3, the SU(d) adjoint count d^2-1 equals 8.")
    print("The full phase square (2pi)^2 divided by this count equals the half-cycle area pi^2/2.")
    print("Thus the horizon lift coefficient can be read as a 3D adjoint-normalized phase area.")
    print()

    print("## Verdict")
    print()
    print("The coefficient pi^2/2 is not arbitrary inside the CE grammar.")
    print("It is conditionally tied to d=3 through equality of half-cycle area and adjoint-normalized phase measure.")
    print("This supports the phase-area lift, but the horizon entropy readout principle still remains a Bridge assumption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
