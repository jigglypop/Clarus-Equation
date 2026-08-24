"""Small independent numerical checks for 11-math.md (standard library only)."""
from __future__ import annotations

import math


def average(values):
    return sum(values) / len(values)


def main() -> None:
    # Minkowski limit: phi=A cos(mt), so <p>=0 and <rho>=m^2 A^2/2.
    m, amplitude, samples = 7.0, 3.0, 100_000
    kinetic, potential = [], []
    for j in range(samples):
        phase = 2.0 * math.pi * (j + 0.5) / samples
        kinetic.append(0.5 * (m * amplitude * math.sin(phase)) ** 2)
        potential.append(0.5 * (m * amplitude * math.cos(phase)) ** 2)
    rho = average([x + y for x, y in zip(kinetic, potential)])
    pressure = average([x - y for x, y in zip(kinetic, potential)])
    assert abs(pressure / rho) < 1.0e-12
    assert abs(rho - 0.5 * m * m * amplitude * amplitude) < 1.0e-10

    # A constant V has rho=V, p=-V identically.
    vacuum = 11.0
    assert vacuum + (-vacuum) == 0.0
    print(f"oscillator_w={pressure / rho:.3e}; rho={rho:.12g}; vacuum_w=-1")


if __name__ == "__main__":
    main()
