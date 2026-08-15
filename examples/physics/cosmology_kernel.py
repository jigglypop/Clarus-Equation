"""Dimensionless flat-FLRW background kernel.

The kernel contains only background identities.  It does not select density
parameters or fit observations.  All densities are present-day fractions and
``E(a) = H(a) / H0`` is dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _require_scale_factor(a: float) -> None:
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("scale factor a must be finite and > 0")


def cpl_w(a: float, w0: float = -1.0, wa: float = 0.0) -> float:
    """Return the CPL equation of state ``w(a) = w0 + wa (1 - a)``."""

    _require_scale_factor(a)
    return w0 + wa * (1.0 - a)


def cpl_density_scale(a: float, w0: float = -1.0, wa: float = 0.0) -> float:
    """Return ``rho_de(a) / rho_de0`` for the CPL equation of state."""

    _require_scale_factor(a)
    power = -3.0 * (1.0 + w0 + wa)
    return a**power * math.exp(3.0 * wa * (a - 1.0))


@dataclass(frozen=True)
class FlatFLRW:
    """Radiation + pressureless matter + CPL dark-energy background.

    No curvature term is included.  A physically normalized flat model has
    ``omega_r0 + omega_m0 + omega_de0 == 1``; the identity methods themselves
    remain useful for unnormalised intermediate calculations and therefore do
    not silently renormalize their inputs.
    """

    omega_m0: float
    omega_de0: float
    omega_r0: float = 0.0
    w0: float = -1.0
    wa: float = 0.0

    @property
    def density_sum0(self) -> float:
        return self.omega_r0 + self.omega_m0 + self.omega_de0

    def e2_of_a(self, a: float) -> float:
        """Return exact dimensionless ``E(a)^2`` for the configured fluids."""

        _require_scale_factor(a)
        return (
            self.omega_r0 * a ** (-4.0)
            + self.omega_m0 * a ** (-3.0)
            + self.omega_de0 * cpl_density_scale(a, self.w0, self.wa)
        )

    def e_of_a(self, a: float) -> float:
        e2 = self.e2_of_a(a)
        if e2 <= 0.0:
            raise ValueError("E(a)^2 must be positive")
        return math.sqrt(e2)

    def omega_r_of_a(self, a: float) -> float:
        return self.omega_r0 * a ** (-4.0) / self.e2_of_a(a)

    def omega_m_of_a(self, a: float) -> float:
        return self.omega_m0 * a ** (-3.0) / self.e2_of_a(a)

    def omega_de_of_a(self, a: float) -> float:
        return self.omega_de0 * cpl_density_scale(a, self.w0, self.wa) / self.e2_of_a(a)

    def dlnh_dln_a(self, a: float) -> float:
        """Return ``d ln(H) / d ln(a)`` without numerical differencing."""

        omega_r = self.omega_r_of_a(a)
        omega_m = self.omega_m_of_a(a)
        omega_de = self.omega_de_of_a(a)
        return 0.5 * (
            -4.0 * omega_r
            - 3.0 * omega_m
            - 3.0 * (1.0 + cpl_w(a, self.w0, self.wa)) * omega_de
        )

    def ricci_over_h2(self, a: float) -> float:
        """Return the exact flat-FLRW Ricci scalar ratio ``R / H^2``.

        For a cosmological constant (``w0=-1, wa=0``), this reduces to
        ``12 - 9 Omega_m(a) - 12 Omega_r(a)``.
        """

        return 6.0 * (2.0 + self.dlnh_dln_a(a))
