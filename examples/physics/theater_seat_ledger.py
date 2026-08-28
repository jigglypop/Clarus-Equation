"""Conditional energy ledger for the zero-dimensional theater analogy.

Before an opening surface, a seat spectrum contains only dimensionless counts,
occupancies, and relative energy weights.  A physical FLRW energy density is
created only after a number-density scale and a reference energy are supplied.

The module proves bookkeeping identities.  It deliberately does not infer a
microscopic pressure, interaction current, or dark-sector identity from the
seat ledger alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SeatType:
    """One pregeometric alternative with a dimensionless energy weight.

    ``degeneracy`` is the expected number of available seats of this type per
    reference event.  ``occupancy`` is an expected fraction in ``[0, 1]`` and
    ``relative_energy`` is energy per occupied seat divided by a separately
    supplied reference energy.
    """

    label: str
    degeneracy: float
    occupancy: float
    relative_energy: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("seat label must be non-empty")
        for name, value in (
            ("degeneracy", self.degeneracy),
            ("occupancy", self.occupancy),
            ("relative_energy", self.relative_energy),
        ):
            _require_finite(name, value)
        if self.degeneracy < 0.0:
            raise ValueError("degeneracy must be non-negative")
        if not 0.0 <= self.occupancy <= 1.0:
            raise ValueError("occupancy must lie in [0, 1]")
        if self.relative_energy < 0.0:
            raise ValueError("relative_energy must be non-negative")

    @property
    def expected_occupied_seats(self) -> float:
        return self.degeneracy * self.occupancy

    @property
    def dimensionless_energy_weight(self) -> float:
        return self.expected_occupied_seats * self.relative_energy

    def count_energy_degenerate_copy(self, *, count_rescaling: float) -> SeatType:
        """Return an exactly energy-weight-degenerate seat description."""

        _require_finite("count_rescaling", count_rescaling)
        if count_rescaling <= 0.0:
            raise ValueError("count_rescaling must be positive")
        return SeatType(
            label=self.label,
            degeneracy=self.degeneracy * count_rescaling,
            occupancy=self.occupancy,
            relative_energy=self.relative_energy / count_rescaling,
        )


@dataclass(frozen=True)
class PregeometricSeatSpectrum:
    """Dimensionless alternative data with no physical-volume density yet."""

    seats: tuple[SeatType, ...]

    def __post_init__(self) -> None:
        if not self.seats:
            raise ValueError("a seat spectrum must contain at least one seat type")
        labels = tuple(seat.label for seat in self.seats)
        if len(set(labels)) != len(labels):
            raise ValueError("seat labels must be unique")

    @property
    def total_dimensionless_energy_weight(self) -> float:
        return sum(seat.dimensionless_energy_weight for seat in self.seats)

    def normalized_energy_fractions(self) -> tuple[tuple[str, float], ...]:
        total = self.total_dimensionless_energy_weight
        if total <= 0.0:
            raise ZeroDivisionError("the spectrum has zero occupied energy weight")
        return tuple(
            (seat.label, seat.dimensionless_energy_weight / total)
            for seat in self.seats
        )


@dataclass(frozen=True)
class OpeningScale:
    """Dimensional matching data supplied on an FLRW opening surface.

    The product ``event_number_density * reference_event_energy`` has energy-
    density units.  The unit system is chosen by the caller.
    """

    event_number_density: float
    reference_event_energy: float

    def __post_init__(self) -> None:
        _require_finite("event_number_density", self.event_number_density)
        _require_finite("reference_event_energy", self.reference_event_energy)
        if self.event_number_density <= 0.0 or self.reference_event_energy <= 0.0:
            raise ValueError("opening scales must be positive")

    @property
    def energy_density_scale(self) -> float:
        return self.event_number_density * self.reference_event_energy


@dataclass(frozen=True)
class OpenedSeatDensity:
    label: str
    energy_density: float


def open_spectrum(
    spectrum: PregeometricSeatSpectrum,
    scale: OpeningScale,
) -> tuple[OpenedSeatDensity, ...]:
    """Map dimensionless seat weights to densities on a declared surface."""

    return tuple(
        OpenedSeatDensity(
            label=seat.label,
            energy_density=(
                scale.energy_density_scale * seat.dimensionless_energy_weight
            ),
        )
        for seat in spectrum.seats
    )


@dataclass(frozen=True)
class SeatScalingAudit:
    """FLRW scaling identity for one component in a fixed comoving patch.

    Let ``N`` be occupied-seat number and ``epsilon`` energy per occupied seat.
    The two logarithmic derivatives are dimensionless.  The current input is
    the dimensionless ratio ``Q/(H*rho)`` in the convention

        rho_dot + 3 H (rho + p) = Q.

    The ledger fixes an effective background equation of state.  It fixes the
    intrinsic physical ``p/rho`` only after the interaction current is given.
    """

    d_log_number_d_log_a: float
    d_log_energy_d_log_a: float
    source_over_hubble_density: float = 0.0
    role: str = "CONDITIONAL_LEDGER_IDENTITY_NOT_DARK_SECTOR_IDENTITY"

    def __post_init__(self) -> None:
        for name, value in (
            ("d_log_number_d_log_a", self.d_log_number_d_log_a),
            ("d_log_energy_d_log_a", self.d_log_energy_d_log_a),
            ("source_over_hubble_density", self.source_over_hubble_density),
        ):
            _require_finite(name, value)

    @property
    def comoving_energy_exponent(self) -> float:
        return self.d_log_number_d_log_a + self.d_log_energy_d_log_a

    @property
    def density_exponent(self) -> float:
        """Return ``d log(rho) / d log(a)`` for ``V proportional to a**3``."""

        return self.comoving_energy_exponent - 3.0

    @property
    def effective_w(self) -> float:
        """Return the background-scaling definition of ``w_eff``."""

        return -self.comoving_energy_exponent / 3.0

    @property
    def intrinsic_w_given_current(self) -> float:
        """Return physical ``p/rho`` after ``Q/(H*rho)`` is declared."""

        return self.effective_w + self.source_over_hubble_density / 3.0

    @property
    def separately_conserved(self) -> bool:
        return self.source_over_hubble_density == 0.0


def source_ratio_from_number_current(
    *,
    number_creation_over_hubble_number: float,
    energy_drift_over_hubble_energy: float,
    intrinsic_w: float,
) -> float:
    """Return ``Q/(H*rho)`` from the homogeneous covariant current identity.

    With ``J^mu=n u^mu`` and ``nabla_mu J^mu=Psi``, the first input is
    ``Psi/(H*n)``.  The second is ``u.grad(epsilon)/(H*epsilon)``.  For
    ``rho=n*epsilon`` the continuity equation gives their sum plus ``3*w``.
    """

    values = (
        number_creation_over_hubble_number,
        energy_drift_over_hubble_energy,
        intrinsic_w,
    )
    for name, value in zip(
        ("number_creation", "energy_drift", "intrinsic_w"), values
    ):
        _require_finite(name, value)
    return (
        number_creation_over_hubble_number
        + energy_drift_over_hubble_energy
        + 3.0 * intrinsic_w
    )


@dataclass(frozen=True)
class TransferChannel:
    label: str
    energy_transfer_rate: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("transfer label must be non-empty")
        _require_finite("energy_transfer_rate", self.energy_transfer_rate)


def total_transfer_residual(channels: Iterable[TransferChannel]) -> float:
    """Return ``sum_i Q_i``; a closed total stress tensor requires zero."""

    return sum(channel.energy_transfer_rate for channel in channels)


def reservoir_closure_rate(destination_rates: Iterable[float]) -> float:
    """Return the unique reservoir rate that closes a declared scalar ledger."""

    total = 0.0
    for rate in destination_rates:
        _require_finite("destination_rate", rate)
        total += rate
    return -total


@dataclass(frozen=True)
class UniformOpeningEnergyRequirement:
    """Idealized uniform-H energy requirement for one acoustic-length ratio."""

    length_ratio: float
    hubble_ratio: float
    extra_density_over_baseline: float
    extra_fraction_of_total: float
    role: str = "UNIFORM_H_DIAGNOSTIC_NOT_TIME_LOCALIZED_EARLY_SECTOR"


def uniform_opening_energy_requirement(
    length_ratio: float,
) -> UniformOpeningEnergyRequirement:
    """Translate ``r_new/r_base`` into two explicitly different fractions.

    Assumptions: the sound speed and integration endpoints are unchanged and
    ``H_new/H_base`` is constant over the entire acoustic integral.  Under
    these assumptions ``r`` scales as ``1/H`` and density as ``H**2``.
    """

    _require_finite("length_ratio", length_ratio)
    if not 0.0 < length_ratio <= 1.0:
        raise ValueError("a positive opening component requires 0 < ratio <= 1")
    hubble_ratio = 1.0 / length_ratio
    extra_over_baseline = hubble_ratio * hubble_ratio - 1.0
    extra_fraction = extra_over_baseline / (1.0 + extra_over_baseline)
    return UniformOpeningEnergyRequirement(
        length_ratio=length_ratio,
        hubble_ratio=hubble_ratio,
        extra_density_over_baseline=extra_over_baseline,
        extra_fraction_of_total=extra_fraction,
    )
