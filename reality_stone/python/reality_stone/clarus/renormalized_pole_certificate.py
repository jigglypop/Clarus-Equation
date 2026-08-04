"""Fail-closed numerical controls for a renormalized scalar pole.

The registered CE value ``29.64757 MeV`` is an inverse-correlation scale
ansatz.  It is not promoted here to a propagator pole.  Higher control stages
require a renormalized-action manifest, counterterm provenance, scalar
quadratic-kernel replicas, and dispersion samples.  Every stage is recomputed
from those typed inputs; callers cannot supply pass/fail booleans.

Even a complete numerical control remains short of a physical CE particle.
Kallen--Lehmann spectral positivity, an asymptotic state, LSZ reduction, and
the CE operator identity are deliberately outside this certificate and remain
locked false.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any


CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV = 29.64757
_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class RenormalizedPoleStage(str, Enum):
    """Monotone control stages; none is a physical-LSZ claim."""

    REGISTERED_SCALE = "REGISTERED_SCALE"
    RENORMALIZED_KERNEL_CONTROL = "RENORMALIZED_KERNEL_CONTROL"
    ISOLATED_SIMPLE_POLE_CONTROL = "ISOLATED_SIMPLE_POLE_CONTROL"
    POSITIVE_RESIDUE_CONTROL = "POSITIVE_RESIDUE_CONTROL"
    DISPERSION_CONTROL = "DISPERSION_CONTROL"


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: str, *, name: str) -> str:
    normalized = _nonempty_string(value, name=name).lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in _HEX_DIGITS for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return normalized


def _finite_tuple(
    values: Sequence[Real],
    *,
    name: str,
    minimum_length: int,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a numerical sequence")
    result = tuple(
        _finite_real(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if len(result) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} values")
    return result


def _strictly_increasing(values: tuple[float, ...], *, name: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class RenormalizedKernelManifest:
    """Provenance for one renormalized scalar quadratic operator."""

    schema_version: str
    field_id: str
    action_sha256: str
    counterterm_sha256: str
    background_sha256: str
    renormalization_scheme: str
    kernel_convention: str = "Gamma_R^(2)(s), residue=1/(d Gamma_R^(2)/ds)"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _nonempty_string(self.schema_version, name="schema_version"),
        )
        object.__setattr__(
            self,
            "field_id",
            _nonempty_string(self.field_id, name="field_id"),
        )
        object.__setattr__(
            self,
            "action_sha256",
            _sha256(self.action_sha256, name="action_sha256"),
        )
        object.__setattr__(
            self,
            "counterterm_sha256",
            _sha256(self.counterterm_sha256, name="counterterm_sha256"),
        )
        object.__setattr__(
            self,
            "background_sha256",
            _sha256(self.background_sha256, name="background_sha256"),
        )
        object.__setattr__(
            self,
            "renormalization_scheme",
            _nonempty_string(
                self.renormalization_scheme,
                name="renormalization_scheme",
            ),
        )
        object.__setattr__(
            self,
            "kernel_convention",
            _nonempty_string(self.kernel_convention, name="kernel_convention"),
        )


@dataclass(frozen=True)
class ScalarKernelReplica:
    """One gauge/scale holdout of the scalar inverse two-point kernel.

    ``invariant_s_mev2`` is the real Minkowski invariant ``s``.  The real and
    imaginary kernel arrays are sampled on that same strictly increasing grid.
    ``first_cut_s_mev2`` is numerical threshold data, not a boolean assertion.
    """

    replica_id: str
    field_id: str
    action_sha256: str
    counterterm_sha256: str
    background_sha256: str
    gauge_parameter: float
    renormalization_scale_mev: float
    invariant_s_mev2: tuple[float, ...]
    kernel_real_mev2: tuple[float, ...]
    kernel_imag_mev2: tuple[float, ...]
    first_cut_s_mev2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replica_id",
            _nonempty_string(self.replica_id, name="replica_id"),
        )
        object.__setattr__(
            self,
            "field_id",
            _nonempty_string(self.field_id, name="field_id"),
        )
        for field_name in (
            "action_sha256",
            "counterterm_sha256",
            "background_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "gauge_parameter",
            _finite_real(self.gauge_parameter, name="gauge_parameter"),
        )
        object.__setattr__(
            self,
            "renormalization_scale_mev",
            _positive(
                self.renormalization_scale_mev,
                name="renormalization_scale_mev",
            ),
        )
        invariant = _finite_tuple(
            self.invariant_s_mev2,
            name="invariant_s_mev2",
            minimum_length=3,
        )
        real = _finite_tuple(
            self.kernel_real_mev2,
            name="kernel_real_mev2",
            minimum_length=3,
        )
        imaginary = _finite_tuple(
            self.kernel_imag_mev2,
            name="kernel_imag_mev2",
            minimum_length=3,
        )
        if len({len(invariant), len(real), len(imaginary)}) != 1:
            raise ValueError("kernel sample arrays must have matching lengths")
        _strictly_increasing(invariant, name="invariant_s_mev2")
        object.__setattr__(self, "invariant_s_mev2", invariant)
        object.__setattr__(self, "kernel_real_mev2", real)
        object.__setattr__(self, "kernel_imag_mev2", imaginary)
        object.__setattr__(
            self,
            "first_cut_s_mev2",
            _finite_real(self.first_cut_s_mev2, name="first_cut_s_mev2"),
        )


@dataclass(frozen=True)
class ScalarDispersionReplica:
    """On-shell energy samples tied to one kernel replica."""

    replica_id: str
    field_id: str
    action_sha256: str
    counterterm_sha256: str
    background_sha256: str
    gauge_parameter: float
    renormalization_scale_mev: float
    momentum_mev: tuple[float, ...]
    energy_mev: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replica_id",
            _nonempty_string(self.replica_id, name="replica_id"),
        )
        object.__setattr__(
            self,
            "field_id",
            _nonempty_string(self.field_id, name="field_id"),
        )
        for field_name in (
            "action_sha256",
            "counterterm_sha256",
            "background_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "gauge_parameter",
            _finite_real(self.gauge_parameter, name="gauge_parameter"),
        )
        object.__setattr__(
            self,
            "renormalization_scale_mev",
            _positive(
                self.renormalization_scale_mev,
                name="renormalization_scale_mev",
            ),
        )
        momentum = _finite_tuple(
            self.momentum_mev,
            name="momentum_mev",
            minimum_length=2,
        )
        energy = _finite_tuple(
            self.energy_mev,
            name="energy_mev",
            minimum_length=2,
        )
        if len(momentum) != len(energy):
            raise ValueError("momentum_mev and energy_mev must have matching lengths")
        if any(value < 0.0 for value in momentum):
            raise ValueError("momentum_mev must be nonnegative")
        if any(value < 0.0 for value in energy):
            raise ValueError("energy_mev must be nonnegative")
        _strictly_increasing(momentum, name="momentum_mev")
        object.__setattr__(self, "momentum_mev", momentum)
        object.__setattr__(self, "energy_mev", energy)


@dataclass(frozen=True)
class PoleCertificateTolerances:
    """Predeclared numerical thresholds used by every replica."""

    kernel_zero_abs_tolerance_mev2: float = 1.0e-10
    minimum_abs_kernel_derivative: float = 1.0e-8
    maximum_imaginary_kernel_abs_mev2: float = 1.0e-10
    minimum_cut_gap_mev2: float = 1.0e-6
    maximum_registered_mass_relative_error: float = 1.0e-8
    maximum_pole_relative_drift: float = 1.0e-8
    maximum_residue_relative_drift: float = 1.0e-8
    maximum_dispersion_relative_residual: float = 1.0e-8
    momentum_zero_abs_tolerance_mev: float = 1.0e-10

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), name=field_name),
            )


@dataclass(frozen=True)
class KernelPoleDiagnostic:
    """Quantities recomputed from one scalar-kernel replica."""

    replica_id: str
    real_root_candidate_count: int
    pole_s_mev2: float | None
    pole_mass_mev: float | None
    kernel_real_residual_mev2: float | None
    kernel_derivative: float | None
    pole_residue: float | None
    kernel_imaginary_at_pole_mev2: float | None
    cut_gap_mev2: float | None
    unique_interior_real_root: bool
    kernel_zero_residual_pass: bool
    simple_root: bool
    imaginary_part_pass: bool
    positive_mass_squared: bool
    cut_separated: bool
    registered_mass_match: bool
    positive_residue: bool


@dataclass(frozen=True)
class DispersionDiagnostic:
    """Residuals of ``E^2=m_pole^2+p^2`` for one replica."""

    replica_id: str
    provenance_matches_kernel_manifest: bool
    rest_sample_present: bool
    nonzero_momentum_sample_present: bool
    maximum_absolute_residual_mev2: float | None
    maximum_relative_residual: float | None
    dispersion_pass: bool


@dataclass(frozen=True)
class RenormalizedPoleCertificate:
    """Serializable monotone result with physical claims locked false."""

    schema_version: str
    registered_inverse_correlation_scale_mev: float
    maximum_supported_stage: RenormalizedPoleStage
    manifest: RenormalizedKernelManifest | None
    kernel_replica_count: int
    dispersion_replica_count: int
    kernel_manifest_consistent: bool
    gauge_holdout_present: bool
    scale_holdout_present: bool
    maximum_gauge_pole_relative_drift: float | None
    maximum_scale_pole_relative_drift: float | None
    maximum_global_pole_relative_spread: float | None
    maximum_gauge_residue_relative_drift: float | None
    maximum_scale_residue_relative_drift: float | None
    maximum_global_residue_relative_spread: float | None
    kernel_diagnostics: tuple[KernelPoleDiagnostic, ...]
    dispersion_diagnostics: tuple[DispersionDiagnostic, ...]
    renormalized_kernel_control_pass: bool
    isolated_simple_pole_control_pass: bool
    positive_residue_control_pass: bool
    dispersion_control_pass: bool
    spectral_positivity_derived: bool
    asymptotic_state_derived: bool
    physical_lsz_particle_derived: bool
    ce_field_identity_derived: bool
    first_blocker: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["maximum_supported_stage"] = self.maximum_supported_stage.value
        return payload


@dataclass(frozen=True)
class _RootCandidate:
    pole_s_mev2: float
    real_residual_mev2: float
    derivative: float
    imaginary_at_root_mev2: float


def _root_candidates(
    replica: ScalarKernelReplica,
    tolerances: PoleCertificateTolerances,
) -> tuple[_RootCandidate, ...]:
    invariant = replica.invariant_s_mev2
    real = replica.kernel_real_mev2
    imaginary = replica.kernel_imag_mev2
    zero_tolerance = tolerances.kernel_zero_abs_tolerance_mev2
    candidates: list[_RootCandidate] = []

    for index in range(1, len(invariant) - 1):
        if abs(real[index]) <= zero_tolerance:
            derivative = (real[index + 1] - real[index - 1]) / (
                invariant[index + 1] - invariant[index - 1]
            )
            candidates.append(
                _RootCandidate(
                    pole_s_mev2=invariant[index],
                    real_residual_mev2=abs(real[index]),
                    derivative=derivative,
                    imaginary_at_root_mev2=imaginary[index],
                )
            )

    for index in range(len(invariant) - 1):
        left_real = real[index]
        right_real = real[index + 1]
        if abs(left_real) <= zero_tolerance or abs(right_real) <= zero_tolerance:
            continue
        if left_real * right_real >= 0.0:
            continue
        interval = invariant[index + 1] - invariant[index]
        derivative = (right_real - left_real) / interval
        fraction = -left_real / (right_real - left_real)
        pole_s = invariant[index] + fraction * interval
        imaginary_at_root = imaginary[index] + fraction * (imaginary[index + 1] - imaginary[index])
        candidates.append(
            _RootCandidate(
                pole_s_mev2=pole_s,
                real_residual_mev2=0.0,
                derivative=derivative,
                imaginary_at_root_mev2=imaginary_at_root,
            )
        )
    return tuple(candidates)


def _kernel_diagnostic(
    replica: ScalarKernelReplica,
    *,
    registered_mass_mev: float,
    tolerances: PoleCertificateTolerances,
) -> KernelPoleDiagnostic:
    candidates = _root_candidates(replica, tolerances)
    unique = len(candidates) == 1
    if not unique:
        return KernelPoleDiagnostic(
            replica_id=replica.replica_id,
            real_root_candidate_count=len(candidates),
            pole_s_mev2=None,
            pole_mass_mev=None,
            kernel_real_residual_mev2=None,
            kernel_derivative=None,
            pole_residue=None,
            kernel_imaginary_at_pole_mev2=None,
            cut_gap_mev2=None,
            unique_interior_real_root=False,
            kernel_zero_residual_pass=False,
            simple_root=False,
            imaginary_part_pass=False,
            positive_mass_squared=False,
            cut_separated=False,
            registered_mass_match=False,
            positive_residue=False,
        )

    candidate = candidates[0]
    simple = abs(candidate.derivative) >= tolerances.minimum_abs_kernel_derivative
    residue = 1.0 / candidate.derivative if simple else None
    positive_mass_squared = candidate.pole_s_mev2 > 0.0
    pole_mass = math.sqrt(candidate.pole_s_mev2) if positive_mass_squared else None
    imaginary_pass = (
        abs(candidate.imaginary_at_root_mev2) <= tolerances.maximum_imaginary_kernel_abs_mev2
    )
    cut_gap = replica.first_cut_s_mev2 - candidate.pole_s_mev2
    cut_separated = cut_gap >= tolerances.minimum_cut_gap_mev2
    zero_pass = candidate.real_residual_mev2 <= tolerances.kernel_zero_abs_tolerance_mev2
    if pole_mass is None:
        registered_match = False
    else:
        registered_error = abs(pole_mass - registered_mass_mev) / registered_mass_mev
        registered_match = registered_error <= tolerances.maximum_registered_mass_relative_error
    return KernelPoleDiagnostic(
        replica_id=replica.replica_id,
        real_root_candidate_count=1,
        pole_s_mev2=candidate.pole_s_mev2,
        pole_mass_mev=pole_mass,
        kernel_real_residual_mev2=candidate.real_residual_mev2,
        kernel_derivative=candidate.derivative,
        pole_residue=residue,
        kernel_imaginary_at_pole_mev2=candidate.imaginary_at_root_mev2,
        cut_gap_mev2=cut_gap,
        unique_interior_real_root=True,
        kernel_zero_residual_pass=zero_pass,
        simple_root=simple,
        imaginary_part_pass=imaginary_pass,
        positive_mass_squared=positive_mass_squared,
        cut_separated=cut_separated,
        registered_mass_match=registered_match,
        positive_residue=residue is not None and residue > 0.0,
    )


def _relative_drift(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _holdout_pairs(
    replicas: tuple[ScalarKernelReplica, ...],
    *,
    same_attribute: str,
    varied_attribute: str,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(replicas):
        for right in replicas[left_index + 1 :]:
            if getattr(left, same_attribute) != getattr(right, same_attribute):
                continue
            if getattr(left, varied_attribute) == getattr(right, varied_attribute):
                continue
            pairs.append((left.replica_id, right.replica_id))
    return tuple(pairs)


def _maximum_pair_drift(
    diagnostics: dict[str, KernelPoleDiagnostic],
    pairs: tuple[tuple[str, str], ...],
    *,
    attribute: str,
) -> float | None:
    drifts: list[float] = []
    for left_id, right_id in pairs:
        left_value = getattr(diagnostics[left_id], attribute)
        right_value = getattr(diagnostics[right_id], attribute)
        if left_value is None or right_value is None:
            continue
        drifts.append(_relative_drift(left_value, right_value))
    return max(drifts) if drifts else None


def _maximum_global_spread(
    diagnostics: tuple[KernelPoleDiagnostic, ...],
    *,
    attribute: str,
) -> float | None:
    values = [
        value for diagnostic in diagnostics if (value := getattr(diagnostic, attribute)) is not None
    ]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return max(
        _relative_drift(left, right)
        for left_index, left in enumerate(values)
        for right in values[left_index + 1 :]
    )


def _replica_matches_manifest(
    replica: ScalarKernelReplica | ScalarDispersionReplica,
    manifest: RenormalizedKernelManifest,
) -> bool:
    return (
        replica.field_id == manifest.field_id
        and replica.action_sha256 == manifest.action_sha256
        and replica.counterterm_sha256 == manifest.counterterm_sha256
        and replica.background_sha256 == manifest.background_sha256
    )


def _dispersion_diagnostic(
    dispersion: ScalarDispersionReplica,
    *,
    kernel: ScalarKernelReplica | None,
    pole: KernelPoleDiagnostic | None,
    manifest: RenormalizedKernelManifest | None,
    registered_mass_mev: float,
    tolerances: PoleCertificateTolerances,
) -> DispersionDiagnostic:
    provenance_match = (
        manifest is not None
        and kernel is not None
        and _replica_matches_manifest(dispersion, manifest)
        and dispersion.gauge_parameter == kernel.gauge_parameter
        and dispersion.renormalization_scale_mev == kernel.renormalization_scale_mev
    )
    rest_present = dispersion.momentum_mev[0] <= tolerances.momentum_zero_abs_tolerance_mev
    nonzero_present = any(
        momentum > tolerances.momentum_zero_abs_tolerance_mev
        for momentum in dispersion.momentum_mev
    )
    if pole is None or pole.pole_s_mev2 is None:
        maximum_absolute = None
        maximum_relative = None
    else:
        absolute_residuals: list[float] = []
        relative_residuals: list[float] = []
        normalization_floor = registered_mass_mev**2
        for momentum, energy in zip(
            dispersion.momentum_mev,
            dispersion.energy_mev,
        ):
            expected = pole.pole_s_mev2 + momentum**2
            observed = energy**2
            residual = abs(observed - expected)
            absolute_residuals.append(residual)
            relative_residuals.append(residual / max(abs(expected), normalization_floor, 1.0e-300))
        maximum_absolute = max(absolute_residuals)
        maximum_relative = max(relative_residuals)
    dispersion_pass = (
        provenance_match
        and rest_present
        and nonzero_present
        and maximum_relative is not None
        and maximum_relative <= tolerances.maximum_dispersion_relative_residual
    )
    return DispersionDiagnostic(
        replica_id=dispersion.replica_id,
        provenance_matches_kernel_manifest=provenance_match,
        rest_sample_present=rest_present,
        nonzero_momentum_sample_present=nonzero_present,
        maximum_absolute_residual_mev2=maximum_absolute,
        maximum_relative_residual=maximum_relative,
        dispersion_pass=dispersion_pass,
    )


def _validated_kernel_replicas(
    replicas: Sequence[ScalarKernelReplica],
) -> tuple[ScalarKernelReplica, ...]:
    if isinstance(replicas, (str, bytes)) or not isinstance(replicas, Sequence):
        raise ValueError("kernel_replicas must be a sequence")
    result = tuple(replicas)
    if any(not isinstance(replica, ScalarKernelReplica) for replica in result):
        raise ValueError("kernel_replicas must contain ScalarKernelReplica values")
    return result


def _validated_dispersion_replicas(
    replicas: Sequence[ScalarDispersionReplica],
) -> tuple[ScalarDispersionReplica, ...]:
    if isinstance(replicas, (str, bytes)) or not isinstance(replicas, Sequence):
        raise ValueError("dispersion_replicas must be a sequence")
    result = tuple(replicas)
    if any(not isinstance(replica, ScalarDispersionReplica) for replica in result):
        raise ValueError("dispersion_replicas must contain ScalarDispersionReplica values")
    return result


def renormalized_pole_certificate(
    *,
    registered_inverse_correlation_scale_mev: Real = (CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV),
    manifest: RenormalizedKernelManifest | None = None,
    kernel_replicas: Sequence[ScalarKernelReplica] = (),
    dispersion_replicas: Sequence[ScalarDispersionReplica] = (),
    tolerances: PoleCertificateTolerances | None = None,
) -> RenormalizedPoleCertificate:
    """Compute the highest supported scalar-pole control stage.

    Missing evidence returns a lower stage.  Malformed numerical inputs raise
    ``ValueError``.  Provenance disagreements are scientific gate failures and
    therefore return a certificate stopped at the preceding stage.
    """

    registered_mass = _positive(
        registered_inverse_correlation_scale_mev,
        name="registered_inverse_correlation_scale_mev",
    )
    if manifest is not None and not isinstance(manifest, RenormalizedKernelManifest):
        raise ValueError("manifest must be a RenormalizedKernelManifest or None")
    thresholds = tolerances or PoleCertificateTolerances()
    if not isinstance(thresholds, PoleCertificateTolerances):
        raise ValueError("tolerances must be a PoleCertificateTolerances value")
    kernels = _validated_kernel_replicas(kernel_replicas)
    dispersions = _validated_dispersion_replicas(dispersion_replicas)

    kernel_ids = tuple(replica.replica_id for replica in kernels)
    unique_kernel_ids = len(set(kernel_ids)) == len(kernel_ids)
    unique_kernel_coordinates = len(
        {(replica.gauge_parameter, replica.renormalization_scale_mev) for replica in kernels}
    ) == len(kernels)
    kernel_manifest_consistent = bool(
        manifest is not None
        and kernels
        and unique_kernel_ids
        and unique_kernel_coordinates
        and all(_replica_matches_manifest(replica, manifest) for replica in kernels)
    )
    kernel_control = kernel_manifest_consistent

    kernel_diagnostics = tuple(
        _kernel_diagnostic(
            replica,
            registered_mass_mev=registered_mass,
            tolerances=thresholds,
        )
        for replica in kernels
    )
    diagnostic_by_id = {diagnostic.replica_id: diagnostic for diagnostic in kernel_diagnostics}
    kernel_by_id = {replica.replica_id: replica for replica in kernels}

    gauge_pairs = _holdout_pairs(
        kernels,
        same_attribute="renormalization_scale_mev",
        varied_attribute="gauge_parameter",
    )
    scale_pairs = _holdout_pairs(
        kernels,
        same_attribute="gauge_parameter",
        varied_attribute="renormalization_scale_mev",
    )
    gauge_holdout_present = bool(gauge_pairs)
    scale_holdout_present = bool(scale_pairs)
    maximum_gauge_pole_drift = _maximum_pair_drift(
        diagnostic_by_id,
        gauge_pairs,
        attribute="pole_s_mev2",
    )
    maximum_scale_pole_drift = _maximum_pair_drift(
        diagnostic_by_id,
        scale_pairs,
        attribute="pole_s_mev2",
    )
    maximum_global_pole_spread = _maximum_global_spread(
        kernel_diagnostics,
        attribute="pole_s_mev2",
    )
    maximum_gauge_residue_drift = _maximum_pair_drift(
        diagnostic_by_id,
        gauge_pairs,
        attribute="pole_residue",
    )
    maximum_scale_residue_drift = _maximum_pair_drift(
        diagnostic_by_id,
        scale_pairs,
        attribute="pole_residue",
    )
    maximum_global_residue_spread = _maximum_global_spread(
        kernel_diagnostics,
        attribute="pole_residue",
    )

    local_isolated_poles = bool(kernel_diagnostics) and all(
        diagnostic.unique_interior_real_root
        and diagnostic.kernel_zero_residual_pass
        and diagnostic.simple_root
        and diagnostic.imaginary_part_pass
        and diagnostic.positive_mass_squared
        and diagnostic.cut_separated
        and diagnostic.registered_mass_match
        for diagnostic in kernel_diagnostics
    )
    pole_drift_pass = all(
        drift is not None and drift <= thresholds.maximum_pole_relative_drift
        for drift in (
            maximum_gauge_pole_drift,
            maximum_scale_pole_drift,
            maximum_global_pole_spread,
        )
    )
    isolated_pole_control = (
        kernel_control
        and gauge_holdout_present
        and scale_holdout_present
        and local_isolated_poles
        and pole_drift_pass
    )

    positive_local_residues = bool(kernel_diagnostics) and all(
        diagnostic.positive_residue for diagnostic in kernel_diagnostics
    )
    residue_drift_pass = all(
        drift is not None and drift <= thresholds.maximum_residue_relative_drift
        for drift in (
            maximum_gauge_residue_drift,
            maximum_scale_residue_drift,
            maximum_global_residue_spread,
        )
    )
    positive_residue_control = (
        isolated_pole_control and positive_local_residues and residue_drift_pass
    )

    dispersion_ids = tuple(replica.replica_id for replica in dispersions)
    dispersion_mapping_complete = len(set(dispersion_ids)) == len(dispersion_ids) and set(
        dispersion_ids
    ) == set(kernel_ids)
    dispersion_diagnostics = tuple(
        _dispersion_diagnostic(
            dispersion,
            kernel=kernel_by_id.get(dispersion.replica_id),
            pole=diagnostic_by_id.get(dispersion.replica_id),
            manifest=manifest,
            registered_mass_mev=registered_mass,
            tolerances=thresholds,
        )
        for dispersion in dispersions
    )
    dispersion_control = (
        positive_residue_control
        and dispersion_mapping_complete
        and bool(dispersion_diagnostics)
        and all(diagnostic.dispersion_pass for diagnostic in dispersion_diagnostics)
    )

    stage = RenormalizedPoleStage.REGISTERED_SCALE
    if kernel_control:
        stage = RenormalizedPoleStage.RENORMALIZED_KERNEL_CONTROL
    if isolated_pole_control:
        stage = RenormalizedPoleStage.ISOLATED_SIMPLE_POLE_CONTROL
    if positive_residue_control:
        stage = RenormalizedPoleStage.POSITIVE_RESIDUE_CONTROL
    if dispersion_control:
        stage = RenormalizedPoleStage.DISPERSION_CONTROL

    blockers: list[str] = []
    if manifest is None:
        blockers.extend(
            (
                "renormalized action manifest is absent",
                "counterterm manifest is absent",
            )
        )
    if not kernels:
        blockers.append("scalar quadratic-kernel replica data are absent")
    elif not unique_kernel_ids:
        blockers.append("kernel replica identifiers are not unique")
    elif not unique_kernel_coordinates:
        blockers.append("kernel gauge/scale coordinates are not unique")
    elif not kernel_manifest_consistent:
        blockers.append("kernel replica provenance does not match the manifest")

    if kernel_control and not isolated_pole_control:
        if not gauge_holdout_present:
            blockers.append("a gauge-parameter holdout is absent")
        if not scale_holdout_present:
            blockers.append("a renormalization-scale holdout is absent")
        if not local_isolated_poles:
            blockers.append(
                "a replica lacks a stable cut-separated simple pole at the registered scale"
            )
        if not pole_drift_pass:
            blockers.append("pole gauge/scale drift exceeds tolerance")

    if isolated_pole_control and not positive_residue_control:
        if not positive_local_residues:
            blockers.append("a scalar pole residue is non-positive")
        if not residue_drift_pass:
            blockers.append("residue gauge/scale drift exceeds tolerance")

    if positive_residue_control and not dispersion_control:
        if not dispersion_mapping_complete:
            blockers.append("dispersion data do not cover every kernel replica")
        elif not all(
            diagnostic.provenance_matches_kernel_manifest for diagnostic in dispersion_diagnostics
        ):
            blockers.append("dispersion provenance does not match the kernel manifest")
        else:
            blockers.append("E^2=m^2+p^2 dispersion residual exceeds tolerance")

    blockers.extend(
        (
            "Kallen-Lehmann spectral positivity is not derived",
            "an asymptotic state and LSZ reduction are not derived",
            "the scalar control operator is not identified with the CE field",
        )
    )
    return RenormalizedPoleCertificate(
        schema_version="1.0",
        registered_inverse_correlation_scale_mev=registered_mass,
        maximum_supported_stage=stage,
        manifest=manifest,
        kernel_replica_count=len(kernels),
        dispersion_replica_count=len(dispersions),
        kernel_manifest_consistent=kernel_manifest_consistent,
        gauge_holdout_present=gauge_holdout_present,
        scale_holdout_present=scale_holdout_present,
        maximum_gauge_pole_relative_drift=maximum_gauge_pole_drift,
        maximum_scale_pole_relative_drift=maximum_scale_pole_drift,
        maximum_global_pole_relative_spread=maximum_global_pole_spread,
        maximum_gauge_residue_relative_drift=maximum_gauge_residue_drift,
        maximum_scale_residue_relative_drift=maximum_scale_residue_drift,
        maximum_global_residue_relative_spread=maximum_global_residue_spread,
        kernel_diagnostics=kernel_diagnostics,
        dispersion_diagnostics=dispersion_diagnostics,
        renormalized_kernel_control_pass=kernel_control,
        isolated_simple_pole_control_pass=isolated_pole_control,
        positive_residue_control_pass=positive_residue_control,
        dispersion_control_pass=dispersion_control,
        spectral_positivity_derived=False,
        asymptotic_state_derived=False,
        physical_lsz_particle_derived=False,
        ce_field_identity_derived=False,
        first_blocker=blockers[0],
        blockers=tuple(blockers),
    )


def current_ce_renormalized_pole_certificate() -> RenormalizedPoleCertificate:
    """Return the current CE state with only its registered scale present."""

    return renormalized_pole_certificate()
