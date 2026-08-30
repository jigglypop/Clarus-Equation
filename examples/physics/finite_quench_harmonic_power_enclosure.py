"""Exact conditional aggregation of supplied convergence transfer envelopes.

This module implements the nonnegative integration step between an all-k
harmonic convergence transfer and its angular auto-spectrum.  It deliberately
does not manufacture the missing physics: every transfer and primordial-power
interval must hold on an entire log-k cell, rather than merely at a grid node.

The adopted dimensionless convention is

    C_ell^{kappa kappa}
        = 4 pi integral d ln(k) mathcal P_R(k) |Delta_ell^kappa(k)|^2.

The exact receipt encloses C_ell / (4 pi), so the transcendental factor remains
symbolic.  A supplied opaque low/high tail contract can assemble a conditional
full-range interval, but it cannot by itself prove that the tail numbers bound
the exterior integrals.  Full-spectrum proof remains false until a
reconstructible majorant-and-remainder receipt is verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Integral
from typing import Iterable


def _exact_fraction(value: object, name: str) -> Fraction:
    """Freeze one finite real scalar as an exact rational."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar, not bool")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    if isinstance(value, (Fraction, Integral)):
        return Fraction(value)
    raise ValueError(f"{name} must be an int, Fraction, or finite float")


def _ordered_interval(
    value: object,
    name: str,
) -> tuple[Fraction, Fraction]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    lower = _exact_fraction(value[0], f"{name} lower endpoint")
    upper = _exact_fraction(value[1], f"{name} upper endpoint")
    if lower > upper:
        raise ValueError(f"{name} endpoints are reversed")
    return lower, upper


def _square_interval(
    interval: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    lower, upper = interval
    squared_upper = max(lower * lower, upper * upper)
    squared_lower = (
        Fraction(0)
        if lower <= 0 <= upper
        else min(lower * lower, upper * upper)
    )
    return squared_lower, squared_upper


@dataclass(frozen=True)
class ConvergenceHarmonicLogKCellEnvelope:
    """Bin-wide supplied envelopes on one dimensionless ln(k/k_pivot) cell."""

    log_k_over_pivot_interval: tuple[Fraction, Fraction]
    primordial_curvature_power_interval: tuple[Fraction, Fraction]
    convergence_transfer_real_interval: tuple[Fraction, Fraction]
    convergence_transfer_imaginary_interval: tuple[Fraction, Fraction]

    @classmethod
    def freeze(
        cls,
        *,
        log_k_over_pivot_interval: object,
        primordial_curvature_power_interval: object,
        convergence_transfer_real_interval: object,
        convergence_transfer_imaginary_interval: object = (0, 0),
    ) -> "ConvergenceHarmonicLogKCellEnvelope":
        log_cell = _ordered_interval(
            log_k_over_pivot_interval,
            "log-k cell",
        )
        if log_cell[0] == log_cell[1]:
            raise ValueError("log-k cell must have positive width")
        primordial = _ordered_interval(
            primordial_curvature_power_interval,
            "primordial curvature power interval",
        )
        if primordial[0] < 0:
            raise ValueError(
                "primordial curvature auto-power must be nonnegative"
            )
        return cls(
            log_k_over_pivot_interval=log_cell,
            primordial_curvature_power_interval=primordial,
            convergence_transfer_real_interval=_ordered_interval(
                convergence_transfer_real_interval,
                "convergence transfer real interval",
            ),
            convergence_transfer_imaginary_interval=_ordered_interval(
                convergence_transfer_imaginary_interval,
                "convergence transfer imaginary interval",
            ),
        )


@dataclass(frozen=True)
class ConvergenceHarmonicExteriorTailBoundCertificate:
    """Opaque supplied claim for both exterior reduced-integral tails.

    This legacy public name is retained for compatibility.  The freeze method
    checks types, signs, labels, and provenance metadata only; it does not
    reconstruct either exterior integral proof.
    """

    ell: int
    band_log_k_over_pivot_interval: tuple[Fraction, Fraction]
    low_k_reduced_tail_upper_bound: Fraction
    high_k_reduced_tail_upper_bound: Fraction
    proof_reference: str
    dimensionless_log_k_exterior_domains_locked: bool
    nonnegative_reduced_integrand_on_exterior_domains_proven: bool
    low_k_exterior_reduced_integral_upper_bound_certified: bool
    high_k_exterior_reduced_integral_upper_bound_certified: bool
    opaque_external_tail_claim_only: bool
    role: str = (
        "SUPPLIED_OPAQUE_EXTERNAL_TAIL_BOUND_CLAIM_NOT_A_RECONSTRUCTIBLE_"
        "EXTERIOR_INTEGRAL_PROOF"
    )

    @classmethod
    def freeze(
        cls,
        *,
        ell: object,
        band_log_k_over_pivot_interval: object,
        low_k_reduced_tail_upper_bound: object,
        high_k_reduced_tail_upper_bound: object,
        proof_reference: object,
        exterior_integral_bounds_certified: bool,
    ) -> "ConvergenceHarmonicExteriorTailBoundCertificate":
        if isinstance(ell, bool) or not isinstance(ell, Integral):
            raise ValueError("tail certificate ell must be an integer")
        harmonic = int(ell)
        if harmonic < 2:
            raise ValueError("tail certificate requires ell >= 2")
        band = _ordered_interval(
            band_log_k_over_pivot_interval,
            "tail certificate band",
        )
        if band[0] == band[1]:
            raise ValueError("tail certificate band must have positive width")
        low_tail = _exact_fraction(
            low_k_reduced_tail_upper_bound,
            "low-k reduced tail upper bound",
        )
        high_tail = _exact_fraction(
            high_k_reduced_tail_upper_bound,
            "high-k reduced tail upper bound",
        )
        if low_tail < 0 or high_tail < 0:
            raise ValueError("supplied reduced tail bounds must be nonnegative")
        if exterior_integral_bounds_certified is not True:
            raise ValueError(
                "both exterior reduced-integral bounds must be explicitly supplied"
            )
        if not isinstance(proof_reference, str) or not proof_reference.strip():
            raise ValueError("tail certificate requires a nonempty proof reference")
        return cls(
            ell=harmonic,
            band_log_k_over_pivot_interval=band,
            low_k_reduced_tail_upper_bound=low_tail,
            high_k_reduced_tail_upper_bound=high_tail,
            proof_reference=proof_reference.strip(),
            dimensionless_log_k_exterior_domains_locked=False,
            nonnegative_reduced_integrand_on_exterior_domains_proven=False,
            low_k_exterior_reduced_integral_upper_bound_certified=False,
            high_k_exterior_reduced_integral_upper_bound_certified=False,
            opaque_external_tail_claim_only=True,
        )


@dataclass(frozen=True)
class ConvergenceHarmonicPowerEnclosureReceipt:
    """Conditional exact enclosure for a supplied harmonic auto-spectrum."""

    ell: int
    log_k_over_pivot_partition: tuple[tuple[Fraction, Fraction], ...]
    band_log_k_over_pivot_interval: tuple[Fraction, Fraction]
    primordial_curvature_power_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    convergence_transfer_real_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    convergence_transfer_imaginary_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    convergence_transfer_modulus_squared_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    reduced_integrand_cell_intervals: (
        tuple[tuple[Fraction, Fraction], ...]
    )
    reduced_band_angular_power_interval: tuple[Fraction, Fraction]
    exterior_tail_bound_certificate: (
        ConvergenceHarmonicExteriorTailBoundCertificate | None
    ) = field(repr=False)
    certified_low_k_reduced_tail_upper_bound: Fraction | None
    certified_high_k_reduced_tail_upper_bound: Fraction | None
    certified_low_k_tail_domain_upper_endpoint: Fraction | None
    certified_high_k_tail_domain_lower_endpoint: Fraction | None
    reduced_full_angular_power_interval: (
        tuple[Fraction, Fraction] | None
    )
    supplied_low_k_reduced_tail_upper_bound: Fraction | None
    supplied_high_k_reduced_tail_upper_bound: Fraction | None
    supplied_low_k_tail_domain_upper_endpoint: Fraction | None
    supplied_high_k_tail_domain_lower_endpoint: Fraction | None
    conditional_reduced_full_angular_power_interval: (
        tuple[Fraction, Fraction] | None
    )
    dimensionless_log_k_coordinate_used: bool
    dimensionless_primordial_curvature_power_used: bool
    dimensionless_convergence_transfer_used: bool
    curvature_fourier_covariance_convention_adopted: bool
    four_pi_d_log_k_angular_power_identity_adopted: bool
    four_pi_factor_kept_symbolic: bool
    input_is_supplied_convergence_transfer_contract: bool
    binwide_not_nodewise_envelopes_required: bool
    exact_rational_nonnegative_cell_sum_proven: bool
    band_limited_angular_auto_power_enclosed: bool
    opaque_exterior_tail_contract_accepted: bool
    reconstructible_exterior_tail_proof_verified: bool
    both_exterior_tail_integral_upper_bounds_certified: bool
    certified_tail_domains_locked_to_partition_exterior: bool
    full_angular_auto_power_enclosed: bool
    lensing_potential_to_convergence_factor_derived: bool = False
    weyl_transfer_from_ce_dynamics_derived: bool = False
    all_k_einstein_boltzmann_transfer_derived: bool = False
    primordial_curvature_spectrum_derived: bool = False
    source_redshift_distribution_derived: bool = False
    post_born_or_relativistic_corrections_enclosed: bool = False
    covariance_or_likelihood_enclosed: bool = False
    role: str = (
        "CONDITIONAL_SUPPLIED_BINWIDE_LOG_K_HARMONIC_CONVERGENCE_AUTO_"
        "POWER_ENCLOSURE_NOT_ALL_K_DYNAMICS_PRIMORDIAL_SOURCE_POST_BORN_"
        "COVARIANCE_OR_LIKELIHOOD_PROOF"
    )


def enclose_convergence_harmonic_auto_power(
    *,
    ell: object,
    cells: Iterable[ConvergenceHarmonicLogKCellEnvelope],
    cellwide_envelopes_certified: bool,
    exterior_tail_certificate: (
        ConvergenceHarmonicExteriorTailBoundCertificate | None
    ) = None,
) -> ConvergenceHarmonicPowerEnclosureReceipt:
    """Enclose C_ell/(4*pi) from certified cell-wide supplied inputs.

    ``cellwide_envelopes_certified`` is an explicit semantic contract: each
    supplied interval must contain its function for every k in that cell.
    Grid-node samples alone do not satisfy this contract.

    The legacy typed exterior-tail object binds supplied tail numbers to this
    exact ell and finite partition.  It can produce a clearly labelled
    conditional interval, but opaque numbers plus a proof-reference string are
    deliberately not accepted as evidence for a full-spectrum enclosure.
    """

    if isinstance(ell, bool) or not isinstance(ell, Integral):
        raise ValueError("ell must be an integer")
    harmonic = int(ell)
    if harmonic < 2:
        raise ValueError("lensing convergence auto-power requires ell >= 2")
    if cellwide_envelopes_certified is not True:
        raise ValueError(
            "bin-wide envelopes must be certified; grid nodes are insufficient"
        )

    supplied_cells = tuple(cells)
    if not supplied_cells:
        raise ValueError("at least one log-k cell is required")
    if not all(
        isinstance(cell, ConvergenceHarmonicLogKCellEnvelope)
        for cell in supplied_cells
    ):
        raise ValueError("every cell must be a frozen harmonic envelope")
    frozen_cells = tuple(
        ConvergenceHarmonicLogKCellEnvelope.freeze(
            log_k_over_pivot_interval=(
                cell.log_k_over_pivot_interval
            ),
            primordial_curvature_power_interval=(
                cell.primordial_curvature_power_interval
            ),
            convergence_transfer_real_interval=(
                cell.convergence_transfer_real_interval
            ),
            convergence_transfer_imaginary_interval=(
                cell.convergence_transfer_imaginary_interval
            ),
        )
        for cell in supplied_cells
    )

    for previous, current in zip(frozen_cells, frozen_cells[1:]):
        if (
            previous.log_k_over_pivot_interval[1]
            != current.log_k_over_pivot_interval[0]
        ):
            raise ValueError("log-k cells must form a contiguous partition")

    modulus_intervals: list[tuple[Fraction, Fraction]] = []
    integrand_intervals: list[tuple[Fraction, Fraction]] = []
    reduced_lower = Fraction(0)
    reduced_upper = Fraction(0)

    for cell in frozen_cells:
        log_lower, log_upper = cell.log_k_over_pivot_interval
        if log_lower >= log_upper:
            raise ValueError("every frozen log-k cell must have positive width")
        power_lower, power_upper = (
            cell.primordial_curvature_power_interval
        )
        if power_lower < 0 or power_lower > power_upper:
            raise ValueError("frozen primordial auto-power interval is invalid")

        real_squared = _square_interval(
            cell.convergence_transfer_real_interval
        )
        imaginary_squared = _square_interval(
            cell.convergence_transfer_imaginary_interval
        )
        modulus_squared = (
            real_squared[0] + imaginary_squared[0],
            real_squared[1] + imaginary_squared[1],
        )
        integrand = (
            power_lower * modulus_squared[0],
            power_upper * modulus_squared[1],
        )
        width = log_upper - log_lower
        reduced_lower += width * integrand[0]
        reduced_upper += width * integrand[1]
        modulus_intervals.append(modulus_squared)
        integrand_intervals.append(integrand)

    band = (
        frozen_cells[0].log_k_over_pivot_interval[0],
        frozen_cells[-1].log_k_over_pivot_interval[1],
    )
    if exterior_tail_certificate is None:
        certificate = None
        low_tail = None
        high_tail = None
    else:
        if not isinstance(
            exterior_tail_certificate,
            ConvergenceHarmonicExteriorTailBoundCertificate,
        ):
            raise ValueError("exterior tails require a frozen certificate")
        certificate = exterior_tail_certificate
        if not certificate.opaque_external_tail_claim_only or any(
            (
                certificate.dimensionless_log_k_exterior_domains_locked,
                certificate
                .nonnegative_reduced_integrand_on_exterior_domains_proven,
                certificate
                .low_k_exterior_reduced_integral_upper_bound_certified,
                certificate
                .high_k_exterior_reduced_integral_upper_bound_certified,
            )
        ):
            raise ValueError("opaque exterior-tail contract boundary is falsified")
        if certificate.ell != harmonic or (
            certificate.band_log_k_over_pivot_interval != band
        ):
            raise ValueError("exterior-tail certificate provenance mismatch")
        if not certificate.proof_reference.strip():
            raise ValueError("exterior-tail certificate proof reference is empty")
        low_tail = certificate.low_k_reduced_tail_upper_bound
        high_tail = certificate.high_k_reduced_tail_upper_bound
        if low_tail < 0 or high_tail < 0:
            raise ValueError("supplied reduced tail bounds must be nonnegative")
    opaque_tail_contract_accepted = certificate is not None
    conditional_full_interval = (
        (
            reduced_lower,
            reduced_upper + low_tail + high_tail,
        )
        if (
            opaque_tail_contract_accepted
            and low_tail is not None
            and high_tail is not None
        )
        else None
    )

    return ConvergenceHarmonicPowerEnclosureReceipt(
        ell=harmonic,
        log_k_over_pivot_partition=tuple(
            cell.log_k_over_pivot_interval for cell in frozen_cells
        ),
        band_log_k_over_pivot_interval=band,
        primordial_curvature_power_cell_intervals=tuple(
            cell.primordial_curvature_power_interval
            for cell in frozen_cells
        ),
        convergence_transfer_real_cell_intervals=tuple(
            cell.convergence_transfer_real_interval for cell in frozen_cells
        ),
        convergence_transfer_imaginary_cell_intervals=tuple(
            cell.convergence_transfer_imaginary_interval
            for cell in frozen_cells
        ),
        convergence_transfer_modulus_squared_cell_intervals=tuple(
            modulus_intervals
        ),
        reduced_integrand_cell_intervals=tuple(integrand_intervals),
        reduced_band_angular_power_interval=(reduced_lower, reduced_upper),
        exterior_tail_bound_certificate=certificate,
        certified_low_k_reduced_tail_upper_bound=None,
        certified_high_k_reduced_tail_upper_bound=None,
        certified_low_k_tail_domain_upper_endpoint=None,
        certified_high_k_tail_domain_lower_endpoint=None,
        reduced_full_angular_power_interval=None,
        supplied_low_k_reduced_tail_upper_bound=low_tail,
        supplied_high_k_reduced_tail_upper_bound=high_tail,
        supplied_low_k_tail_domain_upper_endpoint=(
            band[0]
            if opaque_tail_contract_accepted
            else None
        ),
        supplied_high_k_tail_domain_lower_endpoint=(
            band[1]
            if opaque_tail_contract_accepted
            else None
        ),
        conditional_reduced_full_angular_power_interval=(
            conditional_full_interval
        ),
        dimensionless_log_k_coordinate_used=True,
        dimensionless_primordial_curvature_power_used=True,
        dimensionless_convergence_transfer_used=True,
        curvature_fourier_covariance_convention_adopted=True,
        four_pi_d_log_k_angular_power_identity_adopted=True,
        four_pi_factor_kept_symbolic=True,
        input_is_supplied_convergence_transfer_contract=True,
        binwide_not_nodewise_envelopes_required=True,
        exact_rational_nonnegative_cell_sum_proven=True,
        band_limited_angular_auto_power_enclosed=True,
        opaque_exterior_tail_contract_accepted=(
            opaque_tail_contract_accepted
        ),
        reconstructible_exterior_tail_proof_verified=False,
        both_exterior_tail_integral_upper_bounds_certified=False,
        certified_tail_domains_locked_to_partition_exterior=False,
        full_angular_auto_power_enclosed=False,
    )
