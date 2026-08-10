"""Strict numeric payloads for the five polarized-D--T evidence roles.

The repository artifact verifier binds bytes to a manifest.  Binding bytes is
not enough on its own: an opaque payload must not be able to assert arbitrary
summary claims.  This module therefore accepts only class-specific raw-data
schemas and derives the summaries used by the polarized-fusion gates.

The input is assumed to have already been decoded by a JSON loader which
rejects duplicate keys and non-finite constants.  All keys, units, numeric
types, physical domains, and cross-artifact links are nevertheless checked
again here.  No function imports :mod:`fusion_polarized_evidence_loop`, which
keeps this module usable by that verifier without a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .fusion_equation_iteration_loop import bosch_hale_dt_cross_section_m2


PAYLOAD_SCHEMA_VERSION = "fusion-polarized-artifact-raw-v1"
DERIVED_CLAIMS_SCHEMA_VERSION = "fusion-polarized-derived-claims-v1"
PYTHON_CANONICAL_JSON_CONVENTION = (
    "python-json-allow_nan_false-ensure_ascii_true-sort_keys_true-compact-v1"
)

POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS = (
    "polarized_dt_state_resolved_operator_v1"
)
REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS = (
    "reactor_rate_polarized_deuterium_source_v1"
)
REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS = (
    "reactor_rate_polarized_tritium_source_v1"
)
BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS = "burn_weighted_dt_retention_v1"
POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS = "polarized_dt_wall_plug_ledger_v1"

KNOWN_EVIDENCE_CLASSES = frozenset(
    (
        POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS,
        REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS,
        BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS,
        POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS,
    )
)

EV_TO_JOULE = 1.602176634e-19
DT_FUSION_ENERGY_MEV = 17.6
REACTION_TEMPERATURE_KEV = 10.0
OPERATOR_GRID_MIN_ENERGY_KEV = 0.5
OPERATOR_GRID_MAX_ENERGY_KEV = 550.0
MIN_OPERATOR_GRID_POINTS = 101
MAX_OPERATOR_GRID_POINTS = 2_001
MAX_SCALAR_OPERATOR_GRID_POINTS = 501
MAX_OPERATOR_LOG_ENERGY_GAP = 0.1

SCALAR_TARGET_RATIO_CONTROL_VARIANT = "scalar_target_ratio_control_v1"
SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT = (
    "six_channel_longitudinal_initial_spin_v1"
)
LONGITUDINAL_CHANNEL_ORDER = (
    "mD=-1,mT=-1/2",
    "mD=-1,mT=+1/2",
    "mD=0,mT=-1/2",
    "mD=0,mT=+1/2",
    "mD=+1,mT=-1/2",
    "mD=+1,mT=+1/2",
)
CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER = "channel_major_energy_minor"
COVARIANCE_FACTOR_REPRESENTATION = (
    "row_mode_factor_l_with_c_equals_l_transpose_l_v1"
)
COVARIANCE_UNCERTAINTY_SCOPE = "nuclear_model_systematic"
COVARIANCE_UNCERTAINTY_LEVEL = "1sigma"
CHANNEL_NORMALIZATION_ABS_TOLERANCE = 1.0e-12
TARGET_WEIGHT_ABS_TOLERANCE = 1.0e-12
TARGET_RATIO_GRID_ABS_TOLERANCE = 1.0e-12
COVARIANCE_NORMALIZATION_ABS_TOLERANCE = 1.0e-14
MAX_COVARIANCE_FACTOR_VALUES = 250_000
MIN_REQUIRED_REACTIVITY_RATIO = 1.01
MAXWELLIAN_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-3
MAX_LOCAL_ADJACENT_RELATIVE_CHANGE = 0.75
MAX_LOCAL_SECOND_DIFFERENCE_RELATIVE = 0.35
LOCAL_REGULARITY_SCALE_FLOOR_FRACTION = 0.05
MAXWELLIAN_COARSENING_STRIDES = (2, 4)
BURN_STATE_OPERATOR_CHANNEL_PROJECTION = (
    (1.0 / 6.0, 1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, -1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, 0.0, -1.0 / 6.0),
    (1.0 / 6.0, 0.0, -1.0 / 6.0),
    (1.0 / 6.0, -1.0 / 4.0, 1.0 / 12.0),
    (1.0 / 6.0, 1.0 / 4.0, 1.0 / 12.0),
)
# This is a floating-point identifiability floor, not an experimental error floor.
MIN_NUMERICALLY_MEANINGFUL_RELATIVE_UNCERTAINTY = 1.0e-12
REACTION_ENERGY_FRAME = "center_of_mass"
REACTION_OBSERVABLE = "angle_integrated_total_cross_section_ratio_to_unpolarized"
BASELINE_CROSS_SECTION_MODEL = "bosch_hale_dt_1992"
LONGITUDINAL_QUANTIZATION_AXIS = "longitudinal_beam_axis"
INITIAL_STATE_BASIS = "diagonal_in_product_spin_projection_basis"

WALL_PLUG_COMPONENT_CATEGORIES = frozenset(
    (
        "deuterium_source_and_polarization",
        "tritium_source_and_polarization",
        "cryogenic_refrigeration",
        "magnetic_field_systems",
        "rf_microwave_and_laser",
        "compression_and_injection",
        "vacuum_and_exhaust",
        "instrumentation_and_controls",
        "fuel_recycle_and_repolarization",
        "tritium_processing_and_safety",
        "balance_of_plant",
    )
)
SOURCE_RATE_UNCERTAINTY_VARIABLES = (
    "observed_output_count",
    "background_count",
    "detection_efficiency",
    "measurement_duration_s",
)
BURN_RETENTION_UNCERTAINTY_VARIABLES = (
    "burn_weighted_dt_polarization_product",
    "burn_weighted_deuteron_tensor_polarization",
)
WALL_PLUG_UNCERTAINTY_VARIABLES = (
    *tuple(sorted(WALL_PLUG_COMPONENT_CATEGORIES)),
    "injected_dt_pair_count",
    "measurement_duration_s",
)
MIN_ENGINEERING_EVENT_COUNT = 1_000_000

ENGINEERING_PRIMARY_PAYLOAD_ROLES = {
    REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS: "measurement",
    REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS: "measurement",
    BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS: "measurement",
    POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS: "component_ledger",
}
ENGINEERING_AUXILIARY_PAYLOAD_ROLES = {
    REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS: frozenset(
        ("event_counts", "calibration", "pipeline")
    ),
    REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS: frozenset(
        ("event_counts", "calibration", "pipeline")
    ),
    BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS: frozenset(
        ("burn_events", "calibration", "estimator")
    ),
    POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS: frozenset(
        ("meter_records", "calibrations", "uncertainty_pipeline")
    ),
}

_DOCUMENT_COMMON_KEYS = frozenset(("schema_version", "evidence_class", "payload_role"))
_PLANT_CONFIGURATION_KEYS = frozenset(
    (
        "configuration_id",
        "fusion_power_w",
        "fusion_power_unit",
        "single_pass_burn_fraction",
        "burn_fraction_unit",
    )
)


class ArtifactPayloadError(ValueError):
    """A raw evidence payload is malformed or physically inconsistent."""


def _canonical_json_value_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError, MemoryError) as exc:
        raise ArtifactPayloadError("value is not finite canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DerivedArtifactClaims:
    """Immutable, content-addressable claims derived from one raw payload.

    ``claims_canonical_json`` is the authoritative immutable representation.
    The :attr:`claims` property returns a fresh dictionary, so callers cannot
    mutate the stored result.  Linked manifest hashes carry role labels to
    make D/T cross-joins unambiguous.
    """

    schema_version: str
    evidence_class: str
    manifest_sha256: str
    plant_configuration_id: str | None
    linked_manifest_sha256s: tuple[tuple[str, str], ...]
    typed_payload_sha256s: tuple[tuple[str, str], ...]
    canonical_json_convention: str
    claims_canonical_json: str
    claims_sha256: str

    @property
    def claims(self) -> dict[str, Any]:
        value = json.loads(self.claims_canonical_json)
        if type(value) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("derived claims representation is not an object")
        return value


def _require_object(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ArtifactPayloadError(f"{name} must be a plain object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: frozenset[str], *, name: str) -> None:
    if any(type(key) is not str for key in value):
        raise ArtifactPayloadError(f"{name} keys must be JSON strings")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactPayloadError(f"{name} keys are invalid; missing={missing}, extra={extra}")


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ArtifactPayloadError(f"{name} must be a finite JSON number (not bool)")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactPayloadError(f"{name} must be finite")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise ArtifactPayloadError(f"{name} must not use negative zero")
    return result


def _positive_number(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result <= 0.0:
        raise ArtifactPayloadError(f"{name} must be positive")
    return result


def _nonnegative_number(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0:
        raise ArtifactPayloadError(f"{name} must be nonnegative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactPayloadError(f"{name} must be a positive integer (not bool)")
    return value


def _finite_derived(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ArtifactPayloadError(f"derived {name} overflowed or became non-finite")
    return 0.0 if value == 0.0 else value


def _require_literal(value: object, expected: str, *, name: str) -> None:
    if type(value) is not str or value != expected:
        raise ArtifactPayloadError(f"{name} must be {expected!r}")


def _manifest_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 71 or not value.startswith("sha256:"):
        raise ArtifactPayloadError(f"{name} must be a lowercase sha256: digest")
    hexadecimal = value[7:]
    if any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ArtifactPayloadError(f"{name} must be a lowercase sha256: digest")
    return value


def _configuration_id(value: object) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in value
        )
    ):
        raise ArtifactPayloadError("plant_configuration.configuration_id is invalid")
    return value


def _plant_configuration(value: object) -> tuple[str, float, float, dict[str, object]]:
    plant = _require_object(value, name="plant_configuration")
    _require_exact_keys(plant, _PLANT_CONFIGURATION_KEYS, name="plant_configuration")
    configuration_id = _configuration_id(plant["configuration_id"])
    fusion_power_w = _positive_number(
        plant["fusion_power_w"], name="plant_configuration.fusion_power_w"
    )
    _require_literal(
        plant["fusion_power_unit"], "W", name="plant_configuration.fusion_power_unit"
    )
    burn_fraction = _positive_number(
        plant["single_pass_burn_fraction"],
        name="plant_configuration.single_pass_burn_fraction",
    )
    if burn_fraction >= 1.0:
        raise ArtifactPayloadError(
            "plant_configuration.single_pass_burn_fraction must be below one"
        )
    _require_literal(
        plant["burn_fraction_unit"], "1", name="plant_configuration.burn_fraction_unit"
    )
    fusion_energy_j = DT_FUSION_ENERGY_MEV * 1.0e6 * EV_TO_JOULE
    required_reaction_rate_s = _finite_derived(
        fusion_power_w / fusion_energy_j,
        name="required_fusion_reaction_rate_s",
    )
    required_species_rate_s = _finite_derived(
        required_reaction_rate_s / burn_fraction,
        name="required_per_species_fuel_rate_s",
    )
    claims: dict[str, object] = {
        "validated_integrated_plant_configuration_id": configuration_id,
        "declared_reference_fusion_power_w": fusion_power_w,
        "declared_single_pass_burn_fraction": burn_fraction,
        "required_fusion_reaction_rate_s": required_reaction_rate_s,
        "required_per_species_fuel_rate_s": required_species_rate_s,
    }
    return configuration_id, fusion_power_w, burn_fraction, claims


def _document_header(
    payload: dict[str, Any], *, evidence_class: str, payload_role: str, extra_keys: frozenset[str]
) -> None:
    _require_exact_keys(payload, _DOCUMENT_COMMON_KEYS | extra_keys, name="payload")
    _require_literal(payload["schema_version"], PAYLOAD_SCHEMA_VERSION, name="schema_version")
    _require_literal(payload["evidence_class"], evidence_class, name="evidence_class")
    _require_literal(payload["payload_role"], payload_role, name="payload_role")


def _canonical_claims_result(
    *,
    evidence_class: str,
    manifest_sha256: str,
    claims: dict[str, object],
    plant_configuration_id: str | None = None,
    linked_manifest_sha256s: tuple[tuple[str, str], ...] = (),
    typed_payload_sha256s: tuple[tuple[str, str], ...] = (),
) -> DerivedArtifactClaims:
    try:
        canonical = json.dumps(
            claims,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - internal invariant
        raise ArtifactPayloadError("derived claims are not canonical JSON") from exc
    claims_sha256 = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return DerivedArtifactClaims(
        schema_version=DERIVED_CLAIMS_SCHEMA_VERSION,
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        plant_configuration_id=plant_configuration_id,
        linked_manifest_sha256s=linked_manifest_sha256s,
        typed_payload_sha256s=typed_payload_sha256s,
        canonical_json_convention=PYTHON_CANONICAL_JSON_CONVENTION,
        claims_canonical_json=canonical,
        claims_sha256=claims_sha256,
    )


def _spin_one_populations(vector: float, tensor: float) -> tuple[float, float, float]:
    populations = (
        (2.0 + tensor + 3.0 * vector) / 6.0,
        (1.0 - tensor) / 3.0,
        (2.0 + tensor - 3.0 * vector) / 6.0,
    )
    tolerance = 2.0e-15
    if any(value < -tolerance or value > 1.0 + tolerance for value in populations):
        raise ArtifactPayloadError("deuteron vector/tensor values do not define a physical spin-1 state")
    return tuple(min(1.0, max(0.0, value)) for value in populations)  # type: ignore[return-value]


def _numeric_list(value: object, *, name: str) -> list[float]:
    if type(value) is not list:
        raise ArtifactPayloadError(f"{name} must be a JSON array")
    return [_finite_number(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _normalized_maxwellian_coefficients(
    energy_kev: np.ndarray[Any, np.dtype[np.float64]],
    *,
    temperature_kev: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    cross_sections = np.asarray(
        [bosch_hale_dt_cross_section_m2(float(value)) for value in energy_kev],
        dtype=float,
    )
    kernel = cross_sections * energy_kev * np.exp(-energy_kev / temperature_kev)
    intervals = np.diff(energy_kev)
    coefficients = np.zeros(len(energy_kev), dtype=float)
    coefficients[:-1] += 0.5 * intervals * kernel[:-1]
    coefficients[1:] += 0.5 * intervals * kernel[1:]
    baseline = math.fsum(float(value) for value in coefficients)
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ArtifactPayloadError("Bosch-Hale Maxwellian kernel integral vanished")
    return coefficients / baseline


def _scaled_euclidean_norm(
    values: np.ndarray[Any, np.dtype[np.float64]],
    *,
    name: str,
) -> float:
    """Return a finite Euclidean norm without avoidable square overflow."""

    flattened = np.ravel(values)
    if flattened.size == 0:
        return 0.0
    scale = float(np.max(np.abs(flattened)))
    if not math.isfinite(scale):
        raise ArtifactPayloadError(f"{name} became non-finite")
    if scale == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = flattened / scale
        squared_sum = math.fsum(float(value) * float(value) for value in normalized)
    return _finite_derived(scale * math.sqrt(squared_sum), name=name)


def _dimensionless_local_regularity(
    samples: np.ndarray[Any, np.dtype[np.float64]],
    log_energy: np.ndarray[Any, np.dtype[np.float64]],
    *,
    name: str,
) -> tuple[float, float]:
    """Measure local jumps and log-energy interpolation defects.

    All trailing sample dimensions are one physical vector.  The norm ratios
    are therefore dimensionless.  A small global scale floor prevents smooth
    zero crossings or negligible resonance tails from receiving an infinite
    relative penalty, while still exposing grid-scale alternating aliases.
    """

    values = np.asarray(samples, dtype=float)
    coordinate = np.asarray(log_energy, dtype=float)
    if values.ndim < 2 or values.shape[0] != len(coordinate):
        raise ArtifactPayloadError(f"{name} regularity shape is invalid")
    if len(coordinate) < 3:
        raise ArtifactPayloadError(f"{name} regularity needs at least three points")
    sample_norms = [
        _scaled_euclidean_norm(values[index], name=f"{name} sample norm")
        for index in range(len(coordinate))
    ]
    global_scale = max(sample_norms)
    if global_scale == 0.0:
        return 0.0, 0.0
    scale_floor = LOCAL_REGULARITY_SCALE_FLOOR_FRACTION * global_scale

    adjacent_residuals: list[float] = []
    for index in range(1, len(coordinate)):
        with np.errstate(over="raise", invalid="raise"):
            difference = values[index] - values[index - 1]
        numerator = _scaled_euclidean_norm(
            difference,
            name=f"{name} adjacent difference",
        )
        denominator = max(sample_norms[index - 1], sample_norms[index], scale_floor)
        adjacent_residuals.append(numerator / denominator)

    second_difference_residuals: list[float] = []
    for index in range(1, len(coordinate) - 1):
        span = float(coordinate[index + 1] - coordinate[index - 1])
        if not math.isfinite(span) or span <= 0.0:
            raise ArtifactPayloadError(f"{name} log-energy coordinate is invalid")
        fraction = float((coordinate[index] - coordinate[index - 1]) / span)
        with np.errstate(over="raise", invalid="raise"):
            interpolated = (
                (1.0 - fraction) * values[index - 1]
                + fraction * values[index + 1]
            )
            defect = values[index] - interpolated
        numerator = _scaled_euclidean_norm(
            defect,
            name=f"{name} local second difference",
        )
        denominator = max(
            sample_norms[index - 1],
            sample_norms[index],
            sample_norms[index + 1],
            scale_floor,
        )
        second_difference_residuals.append(numerator / denominator)

    return max(adjacent_residuals), max(second_difference_residuals)


def _coarsening_indices(point_count: int, *, stride: int, phase: int) -> list[int]:
    """Return one endpoint-complete phase of a fixed-stride nested grid."""

    if stride < 2 or not 0 <= phase < stride:  # pragma: no cover - internal constants
        raise ArtifactPayloadError("invalid Maxwellian coarsening phase")
    return sorted({0, point_count - 1, *range(phase, point_count, stride)})


def _derive_scalar_reaction_operator(
    payload: dict[str, Any], *, manifest_sha256: str
) -> DerivedArtifactClaims:
    keys = frozenset(
        (
            "operator_variant",
            "energy_frame",
            "observable",
            "baseline_cross_section_model",
            "quantization_axis",
            "initial_state_basis",
            "temperature_kev",
            "temperature_unit",
            "target_reactivity_ratio",
            "energy_grid_kev",
            "energy_unit",
            "reactivity_ratio_grid",
            "ratio_unit",
            "reactivity_ratio_covariance",
            "covariance_unit",
            "target_deuteron_vector_polarization",
            "target_deuteron_tensor_polarization",
            "target_triton_vector_polarization",
            "polarization_unit",
        )
    )
    evidence_class = POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS
    _document_header(payload, evidence_class=evidence_class, payload_role="operator_grid", extra_keys=keys)
    _require_literal(
        payload["operator_variant"],
        SCALAR_TARGET_RATIO_CONTROL_VARIANT,
        name="operator_variant",
    )
    _require_literal(payload["energy_frame"], REACTION_ENERGY_FRAME, name="energy_frame")
    _require_literal(payload["observable"], REACTION_OBSERVABLE, name="observable")
    _require_literal(
        payload["baseline_cross_section_model"],
        BASELINE_CROSS_SECTION_MODEL,
        name="baseline_cross_section_model",
    )
    _require_literal(
        payload["quantization_axis"],
        LONGITUDINAL_QUANTIZATION_AXIS,
        name="quantization_axis",
    )
    _require_literal(
        payload["initial_state_basis"],
        INITIAL_STATE_BASIS,
        name="initial_state_basis",
    )
    temperature_kev = _finite_number(payload["temperature_kev"], name="temperature_kev")
    if temperature_kev != REACTION_TEMPERATURE_KEV:
        raise ArtifactPayloadError("temperature_kev must be exactly 10 keV")
    _require_literal(payload["temperature_unit"], "keV", name="temperature_unit")
    target_ratio = _positive_number(
        payload["target_reactivity_ratio"], name="target_reactivity_ratio"
    )
    if target_ratio < MIN_REQUIRED_REACTIVITY_RATIO:
        raise ArtifactPayloadError("target_reactivity_ratio must be at least 1.01")
    _require_literal(payload["energy_unit"], "keV", name="energy_unit")
    _require_literal(payload["ratio_unit"], "1", name="ratio_unit")
    _require_literal(payload["covariance_unit"], "1", name="covariance_unit")
    _require_literal(payload["polarization_unit"], "1", name="polarization_unit")

    energy = _numeric_list(payload["energy_grid_kev"], name="energy_grid_kev")
    ratios = _numeric_list(payload["reactivity_ratio_grid"], name="reactivity_ratio_grid")
    point_count = len(energy)
    if not MIN_OPERATOR_GRID_POINTS <= point_count <= MAX_SCALAR_OPERATOR_GRID_POINTS:
        raise ArtifactPayloadError(
            "scalar energy_grid_kev must contain "
            f"{MIN_OPERATOR_GRID_POINTS}..{MAX_SCALAR_OPERATOR_GRID_POINTS} points"
        )
    if len(ratios) != point_count:
        raise ArtifactPayloadError("reactivity_ratio_grid length must match energy_grid_kev")
    if energy[0] != OPERATOR_GRID_MIN_ENERGY_KEV or energy[-1] != OPERATOR_GRID_MAX_ENERGY_KEV:
        raise ArtifactPayloadError("energy_grid_kev must span exactly 0.5..550 keV")
    if any(right <= left for left, right in zip(energy, energy[1:])):
        raise ArtifactPayloadError("energy_grid_kev must be strictly increasing")
    if max(math.log(right / left) for left, right in zip(energy, energy[1:])) > (
        MAX_OPERATOR_LOG_ENERGY_GAP
    ):
        raise ArtifactPayloadError("energy_grid_kev contains an unresolved logarithmic gap")
    if any(value <= 0.0 for value in ratios):
        raise ArtifactPayloadError("reactivity_ratio_grid values must be positive")

    covariance_rows = payload["reactivity_ratio_covariance"]
    if type(covariance_rows) is not list or len(covariance_rows) != point_count:
        raise ArtifactPayloadError("reactivity_ratio_covariance must be an NxN JSON array")
    covariance_data: list[list[float]] = []
    for row_index, row in enumerate(covariance_rows):
        if type(row) is not list or len(row) != point_count:
            raise ArtifactPayloadError("reactivity_ratio_covariance must be an NxN JSON array")
        covariance_data.append(
            [
                _finite_number(value, name=f"reactivity_ratio_covariance[{row_index}][{column}]")
                for column, value in enumerate(row)
            ]
        )
    covariance = np.asarray(covariance_data, dtype=float)
    if not np.array_equal(covariance, covariance.T):
        raise ArtifactPayloadError("reactivity_ratio_covariance must be symmetric")
    if np.any(np.diag(covariance) < 0.0):
        raise ArtifactPayloadError("reactivity_ratio_covariance diagonal must be nonnegative")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ArtifactPayloadError("reactivity_ratio_covariance eigensystem failed") from exc
    eigen_scale = float(np.max(np.abs(eigenvalues)))
    psd_tolerance = (
        16.0 * np.finfo(float).eps * point_count * eigen_scale
    )
    if float(eigenvalues[0]) < -psd_tolerance:
        raise ArtifactPayloadError("reactivity_ratio_covariance must be positive semidefinite")

    energy_array = np.asarray(energy, dtype=float)
    ratio_array = np.asarray(ratios, dtype=float)
    cross_sections = np.asarray(
        [bosch_hale_dt_cross_section_m2(float(value)) for value in energy_array], dtype=float
    )
    kernel = cross_sections * energy_array * np.exp(-energy_array / temperature_kev)
    intervals = np.diff(energy_array)
    integration_coefficients = np.zeros(point_count, dtype=float)
    integration_coefficients[:-1] += 0.5 * intervals * kernel[:-1]
    integration_coefficients[1:] += 0.5 * intervals * kernel[1:]
    baseline = math.fsum(float(value) for value in integration_coefficients)
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ArtifactPayloadError("Bosch-Hale Maxwellian kernel integral vanished")
    normalized_coefficients = integration_coefficients / baseline
    integrated_ratio = _finite_derived(
        float(normalized_coefficients @ ratio_array),
        name="Maxwellian reactivity ratio",
    )
    variance = float(normalized_coefficients @ covariance @ normalized_coefficients)
    variance_tolerance = (
        16.0
        * np.finfo(float).eps
        * point_count
        * eigen_scale
        * float(normalized_coefficients @ normalized_coefficients)
    )
    if variance < -variance_tolerance:
        raise ArtifactPayloadError("integrated covariance produced a negative variance")
    integrated_std = _finite_derived(
        math.sqrt(max(0.0, variance)),
        name="Maxwellian reactivity-ratio standard deviation",
    )
    lower_3sigma = _finite_derived(
        integrated_ratio - 3.0 * integrated_std,
        name="Maxwellian reactivity-ratio lower 3sigma bound",
    )
    if lower_3sigma < max(target_ratio, MIN_REQUIRED_REACTIVITY_RATIO):
        raise ArtifactPayloadError(
            "Maxwellian reactivity-ratio lower 3sigma bound must preserve the declared one-percent target"
        )

    p_d = _finite_number(
        payload["target_deuteron_vector_polarization"],
        name="target_deuteron_vector_polarization",
    )
    p_zz = _finite_number(
        payload["target_deuteron_tensor_polarization"],
        name="target_deuteron_tensor_polarization",
    )
    p_t = _finite_number(
        payload["target_triton_vector_polarization"],
        name="target_triton_vector_polarization",
    )
    p_plus, p_zero, p_minus = _spin_one_populations(p_d, p_zz)
    if not -1.0 <= p_t <= 1.0:
        raise ArtifactPayloadError("target_triton_vector_polarization must lie in [-1, 1]")

    claims: dict[str, object] = {
        "operator_variant": SCALAR_TARGET_RATIO_CONTROL_VARIANT,
        "operator_representation": "scalar_target_ratio_grid_control",
        "energy_frame": REACTION_ENERGY_FRAME,
        "observable": REACTION_OBSERVABLE,
        "baseline_cross_section_model": BASELINE_CROSS_SECTION_MODEL,
        "quantization_axis": LONGITUDINAL_QUANTIZATION_AXIS,
        "initial_state_basis": INITIAL_STATE_BASIS,
        "temperature_kev": temperature_kev,
        "target_reactivity_ratio": target_ratio,
        "operator_grid_point_count": point_count,
        "operator_grid_min_energy_kev": energy[0],
        "operator_grid_max_energy_kev": energy[-1],
        "operator_energy_grid_sha256": _canonical_json_value_sha256(energy),
        "validated_target_deuteron_vector_polarization": p_d,
        "validated_target_deuteron_tensor_polarization": p_zz,
        "validated_target_triton_vector_polarization": p_t,
        "validated_target_deuteron_mplus_population": p_plus,
        "validated_target_deuteron_mzero_population": p_zero,
        "validated_target_deuteron_mminus_population": p_minus,
        "validated_target_state_maxwellian_reactivity_ratio": integrated_ratio,
        "validated_target_state_maxwellian_reactivity_ratio_std": integrated_std,
        "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma": lower_3sigma,
        # A scalar ratio grid, even with covariance, is not the underlying
        # initial-spin operator.  This stays false until a later schema carries
        # explicit channel/operator components and contracts the requested
        # density matrices locally.
        "target_state_resolved_operator_available": False,
        "machine_readable_energy_grid_available": True,
        "systematic_covariance_available": True,
        "local_numeric_reproduction_pass": True,
        "target_ratio_pass_at_3sigma": lower_3sigma >= target_ratio,
    }
    return _canonical_claims_result(
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        claims=claims,
    )


def _derive_six_channel_reaction_operator(
    payload: dict[str, Any], *, manifest_sha256: str
) -> DerivedArtifactClaims:
    """Contract a longitudinal six-channel initial-spin operator locally.

    The fixed flattening convention is channel-major, energy-minor.  If the
    covariance factor has rows :math:`L_r`, its implied covariance is exactly
    :math:`C=L^T L`; materializing the potentially very large ``6N x 6N``
    matrix is neither necessary nor permitted here.
    """

    keys = frozenset(
        (
            "operator_variant",
            "energy_frame",
            "observable",
            "baseline_cross_section_model",
            "quantization_axis",
            "initial_state_basis",
            "temperature_kev",
            "temperature_unit",
            "target_reactivity_ratio",
            "energy_grid_kev",
            "energy_unit",
            "channel_order",
            "channel_reactivity_ratio_grids",
            "target_channel_weights",
            "target_state_reactivity_ratio_grid",
            "ratio_unit",
            "reactivity_ratio_covariance_factor",
            "target_deuteron_vector_polarization",
            "target_deuteron_tensor_polarization",
            "target_triton_vector_polarization",
            "polarization_unit",
        )
    )
    evidence_class = POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS
    _document_header(
        payload,
        evidence_class=evidence_class,
        payload_role="operator_grid",
        extra_keys=keys,
    )
    _require_literal(
        payload["operator_variant"],
        SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT,
        name="operator_variant",
    )
    _require_literal(payload["energy_frame"], REACTION_ENERGY_FRAME, name="energy_frame")
    _require_literal(payload["observable"], REACTION_OBSERVABLE, name="observable")
    _require_literal(
        payload["baseline_cross_section_model"],
        BASELINE_CROSS_SECTION_MODEL,
        name="baseline_cross_section_model",
    )
    _require_literal(
        payload["quantization_axis"],
        LONGITUDINAL_QUANTIZATION_AXIS,
        name="quantization_axis",
    )
    _require_literal(
        payload["initial_state_basis"],
        INITIAL_STATE_BASIS,
        name="initial_state_basis",
    )
    temperature_kev = _finite_number(payload["temperature_kev"], name="temperature_kev")
    if temperature_kev != REACTION_TEMPERATURE_KEV:
        raise ArtifactPayloadError("temperature_kev must be exactly 10 keV")
    _require_literal(payload["temperature_unit"], "keV", name="temperature_unit")
    target_ratio = _positive_number(
        payload["target_reactivity_ratio"], name="target_reactivity_ratio"
    )
    if target_ratio < MIN_REQUIRED_REACTIVITY_RATIO:
        raise ArtifactPayloadError("target_reactivity_ratio must be at least 1.01")
    _require_literal(payload["energy_unit"], "keV", name="energy_unit")
    _require_literal(payload["ratio_unit"], "1", name="ratio_unit")
    _require_literal(payload["polarization_unit"], "1", name="polarization_unit")

    channel_order = payload["channel_order"]
    if type(channel_order) is not list or channel_order != list(LONGITUDINAL_CHANNEL_ORDER):
        raise ArtifactPayloadError("channel_order does not match the fixed longitudinal order")

    energy_document = payload["energy_grid_kev"]
    if type(energy_document) is not list or not (
        MIN_OPERATOR_GRID_POINTS
        <= len(energy_document)
        <= MAX_OPERATOR_GRID_POINTS
    ):
        raise ArtifactPayloadError(
            f"energy_grid_kev must contain {MIN_OPERATOR_GRID_POINTS}..{MAX_OPERATOR_GRID_POINTS} points"
        )
    energy = _numeric_list(energy_document, name="energy_grid_kev")
    point_count = len(energy)
    if energy[0] != OPERATOR_GRID_MIN_ENERGY_KEV or energy[-1] != OPERATOR_GRID_MAX_ENERGY_KEV:
        raise ArtifactPayloadError("energy_grid_kev must span exactly 0.5..550 keV")
    if any(right <= left for left, right in zip(energy, energy[1:])):
        raise ArtifactPayloadError("energy_grid_kev must be strictly increasing")
    if max(math.log(right / left) for left, right in zip(energy, energy[1:])) > (
        MAX_OPERATOR_LOG_ENERGY_GAP
    ):
        raise ArtifactPayloadError("energy_grid_kev contains an unresolved logarithmic gap")

    channel_rows = payload["channel_reactivity_ratio_grids"]
    channel_count = len(LONGITUDINAL_CHANNEL_ORDER)
    if type(channel_rows) is not list or len(channel_rows) != channel_count:
        raise ArtifactPayloadError(
            "channel_reactivity_ratio_grids must contain exactly six channel rows"
        )
    channel_data: list[list[float]] = []
    for channel_index, row in enumerate(channel_rows):
        if type(row) is not list or len(row) != point_count:
            raise ArtifactPayloadError("each channel ratio row must match energy_grid_kev")
        values = _numeric_list(
            row,
            name=f"channel_reactivity_ratio_grids[{channel_index}]",
        )
        if any(value < 0.0 for value in values):
            raise ArtifactPayloadError("channel reactivity ratios must be nonnegative")
        channel_data.append(values)

    unpolarized_residuals: list[float] = []
    for energy_index in range(point_count):
        channel_mean = _finite_derived(
            math.fsum(channel_data[channel][energy_index] for channel in range(channel_count))
            / channel_count,
            name="unpolarized channel mean",
        )
        unpolarized_residuals.append(abs(channel_mean - 1.0))
    max_unpolarized_residual = max(unpolarized_residuals)
    if max_unpolarized_residual > CHANNEL_NORMALIZATION_ABS_TOLERANCE:
        raise ArtifactPayloadError(
            "six-channel arithmetic mean must preserve the unpolarized ratio one"
        )
    spin_reversal_pairs = ((0, 5), (1, 4), (2, 3))
    max_spin_reversal_residual = max(
        abs(channel_data[left][energy_index] - channel_data[right][energy_index])
        for left, right in spin_reversal_pairs
        for energy_index in range(point_count)
    )
    if any(
        channel_data[left][energy_index] != channel_data[right][energy_index]
        for left, right in spin_reversal_pairs
        for energy_index in range(point_count)
    ):
        raise ArtifactPayloadError(
            "longitudinal total-cross-section channels must preserve simultaneous spin reversal exactly"
        )
    energy_array = np.asarray(energy, dtype=float)
    log_energy = np.log(energy_array)
    operator_energy_samples = np.asarray(channel_data, dtype=float).T
    (
        operator_local_adjacent_relative_change,
        operator_local_second_difference_relative,
    ) = _dimensionless_local_regularity(
        operator_energy_samples,
        log_energy,
        name="six-channel operator",
    )
    if (
        operator_local_adjacent_relative_change
        > MAX_LOCAL_ADJACENT_RELATIVE_CHANGE
    ):
        raise ArtifactPayloadError(
            "six-channel operator violates the local adjacent regularity bound"
        )
    if (
        operator_local_second_difference_relative
        > MAX_LOCAL_SECOND_DIFFERENCE_RELATIVE
    ):
        raise ArtifactPayloadError(
            "six-channel operator violates the local second-difference regularity bound"
        )

    p_d = _finite_number(
        payload["target_deuteron_vector_polarization"],
        name="target_deuteron_vector_polarization",
    )
    p_zz = _finite_number(
        payload["target_deuteron_tensor_polarization"],
        name="target_deuteron_tensor_polarization",
    )
    p_t = _finite_number(
        payload["target_triton_vector_polarization"],
        name="target_triton_vector_polarization",
    )
    p_plus, p_zero, p_minus = _spin_one_populations(p_d, p_zz)
    if not -1.0 <= p_t <= 1.0:
        raise ArtifactPayloadError("target_triton_vector_polarization must lie in [-1, 1]")
    t_minus = (1.0 - p_t) / 2.0
    t_plus = (1.0 + p_t) / 2.0
    target_weights = (
        p_minus * t_minus,
        p_minus * t_plus,
        p_zero * t_minus,
        p_zero * t_plus,
        p_plus * t_minus,
        p_plus * t_plus,
    )
    if abs(math.fsum(target_weights) - 1.0) > TARGET_WEIGHT_ABS_TOLERANCE:
        raise ArtifactPayloadError("derived six-channel target weights do not sum to one")
    declared_weight_document = payload["target_channel_weights"]
    if type(declared_weight_document) is not list or len(declared_weight_document) != channel_count:
        raise ArtifactPayloadError("target_channel_weights must contain exactly six values")
    declared_weights = _numeric_list(
        declared_weight_document,
        name="target_channel_weights",
    )
    if any(value < 0.0 or value > 1.0 for value in declared_weights):
        raise ArtifactPayloadError("target_channel_weights must lie in [0, 1]")
    weight_residual = max(
        abs(declared - derived)
        for declared, derived in zip(declared_weights, target_weights, strict=True)
    )
    if weight_residual > TARGET_WEIGHT_ABS_TOLERANCE:
        raise ArtifactPayloadError(
            "target_channel_weights do not match the physical spin populations"
        )

    derived_target_grid = [
        _finite_derived(
            math.fsum(
                target_weights[channel] * channel_data[channel][energy_index]
                for channel in range(channel_count)
            ),
            name="target-state ratio grid",
        )
        for energy_index in range(point_count)
    ]
    declared_target_grid_document = payload["target_state_reactivity_ratio_grid"]
    if (
        type(declared_target_grid_document) is not list
        or len(declared_target_grid_document) != point_count
    ):
        raise ArtifactPayloadError(
            "target_state_reactivity_ratio_grid length must match energy_grid_kev"
        )
    declared_target_grid = _numeric_list(
        declared_target_grid_document,
        name="target_state_reactivity_ratio_grid",
    )
    if any(value < 0.0 for value in declared_target_grid):
        raise ArtifactPayloadError("target-state reactivity ratios must be nonnegative")
    target_grid_residual = max(
        abs(declared - derived)
        for declared, derived in zip(declared_target_grid, derived_target_grid, strict=True)
    )
    if target_grid_residual > TARGET_RATIO_GRID_ABS_TOLERANCE:
        raise ArtifactPayloadError(
            "target_state_reactivity_ratio_grid was not derived from the six channels"
        )

    factor_document = _require_object(
        payload["reactivity_ratio_covariance_factor"],
        name="reactivity_ratio_covariance_factor",
    )
    _require_exact_keys(
        factor_document,
        frozenset(
            (
                "representation",
                "flatten_order",
                "factor_unit",
                "uncertainty_scope",
                "uncertainty_level",
                "modes",
            )
        ),
        name="reactivity_ratio_covariance_factor",
    )
    _require_literal(
        factor_document["representation"],
        COVARIANCE_FACTOR_REPRESENTATION,
        name="reactivity_ratio_covariance_factor.representation",
    )
    _require_literal(
        factor_document["flatten_order"],
        CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
        name="reactivity_ratio_covariance_factor.flatten_order",
    )
    _require_literal(
        factor_document["factor_unit"],
        "1",
        name="reactivity_ratio_covariance_factor.factor_unit",
    )
    _require_literal(
        factor_document["uncertainty_scope"],
        COVARIANCE_UNCERTAINTY_SCOPE,
        name="reactivity_ratio_covariance_factor.uncertainty_scope",
    )
    _require_literal(
        factor_document["uncertainty_level"],
        COVARIANCE_UNCERTAINTY_LEVEL,
        name="reactivity_ratio_covariance_factor.uncertainty_level",
    )
    mode_rows = factor_document["modes"]
    flattened_count = channel_count * point_count
    if type(mode_rows) is not list or not 1 <= len(mode_rows) <= flattened_count:
        raise ArtifactPayloadError("covariance factor must contain 1..6N mode rows")
    mode_count = len(mode_rows)
    if mode_count * flattened_count > MAX_COVARIANCE_FACTOR_VALUES:
        raise ArtifactPayloadError("covariance factor exceeds the fixed numeric-value ceiling")
    factor_data: list[list[float]] = []
    max_covariance_normalization_residual = 0.0
    max_covariance_spin_reversal_residual = 0.0
    for mode_index, row in enumerate(mode_rows):
        if type(row) is not list or len(row) != flattened_count:
            raise ArtifactPayloadError(
                "each covariance-factor mode must have exactly 6N flattened values"
            )
        values = _numeric_list(
            row,
            name=f"reactivity_ratio_covariance_factor.modes[{mode_index}]",
        )
        factor_data.append(values)
        for energy_index in range(point_count):
            residual = abs(
                math.fsum(
                    values[channel * point_count + energy_index]
                    for channel in range(channel_count)
                )
            )
            max_covariance_normalization_residual = max(
                max_covariance_normalization_residual,
                residual,
            )
            max_covariance_spin_reversal_residual = max(
                max_covariance_spin_reversal_residual,
                *(
                    abs(
                        values[left * point_count + energy_index]
                        - values[right * point_count + energy_index]
                    )
                    for left, right in spin_reversal_pairs
                ),
            )
    if (
        max_covariance_normalization_residual
        > COVARIANCE_NORMALIZATION_ABS_TOLERANCE
    ):
        raise ArtifactPayloadError(
            "each covariance mode must preserve the unpolarized channel normalization"
        )
    if any(
        mode[left * point_count + energy_index]
        != mode[right * point_count + energy_index]
        for mode in factor_data
        for left, right in spin_reversal_pairs
        for energy_index in range(point_count)
    ):
        raise ArtifactPayloadError(
            "each covariance mode must preserve simultaneous spin reversal exactly"
        )

    factor = np.asarray(factor_data, dtype=float)
    factor_frobenius_norm = _scaled_euclidean_norm(
        factor,
        name="covariance factor Frobenius norm",
    )
    factor_energy_samples = np.transpose(
        factor.reshape(mode_count, channel_count, point_count),
        (2, 0, 1),
    )
    (
        covariance_local_adjacent_relative_change,
        covariance_local_second_difference_relative,
    ) = _dimensionless_local_regularity(
        factor_energy_samples,
        log_energy,
        name="covariance factor",
    )
    if (
        covariance_local_adjacent_relative_change
        > MAX_LOCAL_ADJACENT_RELATIVE_CHANGE
    ):
        raise ArtifactPayloadError(
            "covariance factor violates the local adjacent regularity bound"
        )
    if (
        covariance_local_second_difference_relative
        > MAX_LOCAL_SECOND_DIFFERENCE_RELATIVE
    ):
        raise ArtifactPayloadError(
            "covariance factor violates the local second-difference regularity bound"
        )
    normalized_coefficients = _normalized_maxwellian_coefficients(
        energy_array,
        temperature_kev=temperature_kev,
    )
    channel_projection = np.asarray(
        BURN_STATE_OPERATOR_CHANNEL_PROJECTION, dtype=np.float64
    )
    integrated_channel_responses = np.asarray(
        [
            math.fsum(
                float(coefficient) * channel_data[channel][energy_index]
                for energy_index, coefficient in enumerate(normalized_coefficients)
            )
            for channel in range(channel_count)
        ],
        dtype=np.float64,
    )
    integrated_state_response_coefficients = (
        channel_projection.T @ integrated_channel_responses
    )
    coefficient_projected_modes: list[tuple[float, float, float]] = []
    for mode in factor_data:
        integrated_mode_channels = np.asarray(
            [
                math.fsum(
                    float(coefficient)
                    * mode[channel * point_count + energy_index]
                    for energy_index, coefficient in enumerate(
                        normalized_coefficients
                    )
                )
                for channel in range(channel_count)
            ],
            dtype=np.float64,
        )
        projected_mode = channel_projection.T @ integrated_mode_channels
        coefficient_projected_modes.append(
            tuple(float(value) for value in projected_mode)
        )
    coefficient_mode_array = np.asarray(coefficient_projected_modes, dtype=np.float64)
    integrated_state_response_covariance = (
        coefficient_mode_array.T @ coefficient_mode_array
    )
    if not np.all(np.isfinite(integrated_state_response_coefficients)) or not np.all(
        np.isfinite(integrated_state_response_covariance)
    ):
        raise ArtifactPayloadError("integrated operator response coefficients are non-finite")
    integrated_ratio = _finite_derived(
        math.fsum(
            float(coefficient) * ratio
            for coefficient, ratio in zip(
                normalized_coefficients,
                derived_target_grid,
                strict=True,
            )
        ),
        name="Maxwellian reactivity ratio",
    )
    target_contraction_state = np.asarray((1.0, p_d * p_t, p_zz))
    linear_contraction_ratio = _finite_derived(
        float(integrated_state_response_coefficients @ target_contraction_state),
        name="affine burn-state operator contraction ratio",
    )
    linear_contraction_residual = abs(linear_contraction_ratio - integrated_ratio)
    if linear_contraction_residual > TARGET_RATIO_GRID_ABS_TOLERANCE:
        raise ArtifactPayloadError(
            "six-channel operator does not reduce to the declared vector-product/tensor contraction"
        )
    derivative_weights = [
        float(coefficient) * weight
        for weight in target_weights
        for coefficient in normalized_coefficients
    ]
    projected_modes = [
        math.fsum(
            factor_value * derivative_weight
            for factor_value, derivative_weight in zip(
                mode,
                derivative_weights,
                strict=True,
            )
        )
        for mode in factor_data
    ]
    variance = _finite_derived(
        math.fsum(value * value for value in projected_modes),
        name="Maxwellian reactivity-ratio variance",
    )
    integrated_std = _finite_derived(
        math.sqrt(variance),
        name="Maxwellian reactivity-ratio standard deviation",
    )
    numerical_uncertainty_floor = (
        MIN_NUMERICALLY_MEANINGFUL_RELATIVE_UNCERTAINTY
        * max(1.0, abs(integrated_ratio))
    )
    if integrated_std < numerical_uncertainty_floor:
        raise ArtifactPayloadError(
            "covariance factor must give the target contraction nonzero uncertainty "
            "above the numerical-meaning floor"
        )
    if factor_frobenius_norm < numerical_uncertainty_floor:
        raise ArtifactPayloadError(
            "covariance factor norm is below the numerical-meaning floor"
        )
    lower_3sigma = _finite_derived(
        integrated_ratio - 3.0 * integrated_std,
        name="Maxwellian reactivity-ratio lower 3sigma bound",
    )
    if lower_3sigma < max(target_ratio, MIN_REQUIRED_REACTIVITY_RATIO):
        raise ArtifactPayloadError(
            "Maxwellian reactivity-ratio lower 3sigma bound must preserve the declared one-percent target"
        )

    coarsening_phase_summaries: list[dict[str, object]] = []
    for stride in MAXWELLIAN_COARSENING_STRIDES:
        for phase in range(stride):
            phase_indices = _coarsening_indices(
                point_count,
                stride=stride,
                phase=phase,
            )
            phase_energy = energy_array[phase_indices]
            phase_coefficients = _normalized_maxwellian_coefficients(
                phase_energy,
                temperature_kev=temperature_kev,
            )
            phase_ratio = _finite_derived(
                math.fsum(
                    float(coefficient) * derived_target_grid[index]
                    for coefficient, index in zip(
                        phase_coefficients,
                        phase_indices,
                        strict=True,
                    )
                ),
                name="multi-phase coarsened Maxwellian reactivity ratio",
            )
            phase_derivative_weights = [
                float(coefficient) * weight
                for weight in target_weights
                for coefficient in phase_coefficients
            ]
            phase_projected_modes = [
                math.fsum(
                    mode[channel * point_count + energy_index] * derivative_weight
                    for derivative_weight, (channel, energy_index) in zip(
                        phase_derivative_weights,
                        (
                            (channel, energy_index)
                            for channel in range(channel_count)
                            for energy_index in phase_indices
                        ),
                        strict=True,
                    )
                )
                for mode in factor_data
            ]
            phase_variance = _finite_derived(
                math.fsum(value * value for value in phase_projected_modes),
                name="multi-phase coarsened Maxwellian reactivity-ratio variance",
            )
            phase_std = _finite_derived(
                math.sqrt(phase_variance),
                name=(
                    "multi-phase coarsened Maxwellian reactivity-ratio "
                    "standard deviation"
                ),
            )
            phase_lower_3sigma = _finite_derived(
                phase_ratio - 3.0 * phase_std,
                name=(
                    "multi-phase coarsened Maxwellian reactivity-ratio "
                    "lower 3sigma bound"
                ),
            )
            phase_channel_responses = np.asarray(
                [
                    math.fsum(
                        float(coefficient) * channel_data[channel][energy_index]
                        for coefficient, energy_index in zip(
                            phase_coefficients, phase_indices, strict=True
                        )
                    )
                    for channel in range(channel_count)
                ],
                dtype=np.float64,
            )
            phase_state_response_coefficients = (
                channel_projection.T @ phase_channel_responses
            )
            phase_coefficient_modes: list[tuple[float, float, float]] = []
            for mode in factor_data:
                phase_mode_channels = np.asarray(
                    [
                        math.fsum(
                            float(coefficient)
                            * mode[channel * point_count + energy_index]
                            for coefficient, energy_index in zip(
                                phase_coefficients, phase_indices, strict=True
                            )
                        )
                        for channel in range(channel_count)
                    ],
                    dtype=np.float64,
                )
                phase_coefficient_modes.append(
                    tuple(
                        float(value)
                        for value in channel_projection.T @ phase_mode_channels
                    )
                )
            phase_coefficient_mode_array = np.asarray(
                phase_coefficient_modes, dtype=np.float64
            )
            phase_state_response_covariance = (
                phase_coefficient_mode_array.T @ phase_coefficient_mode_array
            )
            if not np.all(np.isfinite(phase_state_response_coefficients)) or not np.all(
                np.isfinite(phase_state_response_covariance)
            ):
                raise ArtifactPayloadError(
                    "coarsened operator response coefficients are non-finite"
                )
            phase_ratio_residual = abs(phase_ratio - integrated_ratio) / integrated_ratio
            phase_std_residual = abs(phase_std - integrated_std) / integrated_std
            phase_max_residual = max(phase_ratio_residual, phase_std_residual)
            if phase_max_residual > MAXWELLIAN_QUADRATURE_RELATIVE_TOLERANCE:
                raise ArtifactPayloadError(
                    "Maxwellian ratio/covariance contraction did not converge under "
                    f"multi-phase stride={stride}, phase={phase} coarsening"
                )
            if phase_lower_3sigma < max(target_ratio, MIN_REQUIRED_REACTIVITY_RATIO):
                raise ArtifactPayloadError(
                    "multi-phase coarsened Maxwellian lower 3sigma bound does not "
                    f"preserve the one-percent target at stride={stride}, phase={phase}"
                )
            coarsening_phase_summaries.append(
                {
                    "stride": stride,
                    "phase": phase,
                    "point_count": len(phase_indices),
                    "reactivity_ratio": phase_ratio,
                    "reactivity_ratio_std": phase_std,
                    "reactivity_ratio_lower_3sigma": phase_lower_3sigma,
                    "ratio_relative_residual": phase_ratio_residual,
                    "std_relative_residual": phase_std_residual,
                    "max_relative_residual": phase_max_residual,
                    "burn_state_operator_response_coefficients": (
                        phase_state_response_coefficients.tolist()
                    ),
                    "burn_state_operator_response_coefficient_covariance": (
                        phase_state_response_covariance.tolist()
                    ),
                }
            )

    reference_coarsening = coarsening_phase_summaries[0]
    coarse_indices = _coarsening_indices(point_count, stride=2, phase=0)
    coarse_ratio = float(reference_coarsening["reactivity_ratio"])
    coarse_std = float(reference_coarsening["reactivity_ratio_std"])
    coarse_lower_3sigma = float(
        reference_coarsening["reactivity_ratio_lower_3sigma"]
    )
    ratio_quadrature_residual = float(
        reference_coarsening["ratio_relative_residual"]
    )
    std_quadrature_residual = float(reference_coarsening["std_relative_residual"])
    max_quadrature_residual = max(
        float(summary["max_relative_residual"])
        for summary in coarsening_phase_summaries
    )
    min_coarsened_lower_3sigma = min(
        float(summary["reactivity_ratio_lower_3sigma"])
        for summary in coarsening_phase_summaries
    )

    # The numeric-value ceiling bounds this direct SVD.  Unlike eigendecomposing
    # L L^T, SVD does not square the condition number or turn Gram roundoff into
    # dozens of false nonzero singular modes for duplicated factor rows.
    try:
        with np.errstate(over="raise", invalid="raise"):
            singular_values = np.linalg.svd(factor, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ArtifactPayloadError("covariance factor rank calculation failed") from exc
    if not np.all(np.isfinite(singular_values)):
        raise ArtifactPayloadError("covariance factor singular values are non-finite")
    largest_singular_value = float(singular_values[0])
    singular_tolerance = (
        max(factor.shape)
        * np.finfo(float).eps
        * largest_singular_value
    )
    factor_rank = int(
        sum(
            bool(float(value) > singular_tolerance)
            for value in singular_values
        )
    )
    if factor_rank <= 0:
        raise ArtifactPayloadError("covariance factor must have nonzero rank")

    claims: dict[str, object] = {
        "operator_variant": SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT,
        "operator_representation": "six_channel_longitudinal_initial_spin_operator",
        "energy_frame": REACTION_ENERGY_FRAME,
        "observable": REACTION_OBSERVABLE,
        "baseline_cross_section_model": BASELINE_CROSS_SECTION_MODEL,
        "quantization_axis": LONGITUDINAL_QUANTIZATION_AXIS,
        "initial_state_basis": INITIAL_STATE_BASIS,
        "operator_channel_order": list(LONGITUDINAL_CHANNEL_ORDER),
        "operator_channel_count": channel_count,
        "operator_grid_point_count": point_count,
        "operator_grid_min_energy_kev": energy[0],
        "operator_grid_max_energy_kev": energy[-1],
        "temperature_kev": temperature_kev,
        "target_reactivity_ratio": target_ratio,
        "validated_target_deuteron_vector_polarization": p_d,
        "validated_target_deuteron_tensor_polarization": p_zz,
        "validated_target_triton_vector_polarization": p_t,
        "validated_target_deuteron_mplus_population": p_plus,
        "validated_target_deuteron_mzero_population": p_zero,
        "validated_target_deuteron_mminus_population": p_minus,
        "validated_target_triton_mplus_population": t_plus,
        "validated_target_triton_mminus_population": t_minus,
        "validated_target_channel_weights": list(target_weights),
        "validated_target_state_maxwellian_reactivity_ratio": integrated_ratio,
        "validated_target_state_maxwellian_reactivity_ratio_std": integrated_std,
        "validated_target_state_maxwellian_reactivity_ratio_lower_3sigma": lower_3sigma,
        "burn_state_operator_contraction_model": (
            "maxwellian_affine_intercept_vector_product_and_deuteron_tensor_v2"
        ),
        "burn_state_operator_contraction_variables": [
            "intercept_one",
            "burn_weighted_dt_polarization_product",
            "burn_weighted_deuteron_tensor_polarization",
        ],
        "burn_state_operator_channel_projection_matrix": [
            list(row) for row in BURN_STATE_OPERATOR_CHANNEL_PROJECTION
        ],
        "burn_state_operator_response_coefficients": (
            integrated_state_response_coefficients.tolist()
        ),
        "burn_state_operator_response_coefficient_covariance": (
            integrated_state_response_covariance.tolist()
        ),
        "burn_state_operator_response_uncertainty_scope": (
            COVARIANCE_UNCERTAINTY_SCOPE
        ),
        "burn_state_operator_response_uncertainty_level": (
            COVARIANCE_UNCERTAINTY_LEVEL
        ),
        "target_state_linear_contraction_max_abs_residual": (
            linear_contraction_residual
        ),
        "unpolarized_channel_mean_max_abs_residual": max_unpolarized_residual,
        "simultaneous_spin_reversal_max_abs_residual": max_spin_reversal_residual,
        "simultaneous_spin_reversal_exact": True,
        "target_channel_weight_max_abs_residual": weight_residual,
        "target_ratio_grid_max_abs_residual": target_grid_residual,
        "operator_local_adjacent_max_relative_change": (
            operator_local_adjacent_relative_change
        ),
        "operator_local_second_difference_max_relative": (
            operator_local_second_difference_relative
        ),
        "covariance_local_adjacent_max_relative_change": (
            covariance_local_adjacent_relative_change
        ),
        "covariance_local_second_difference_max_relative": (
            covariance_local_second_difference_relative
        ),
        "local_adjacent_relative_change_limit": (
            MAX_LOCAL_ADJACENT_RELATIVE_CHANGE
        ),
        "local_second_difference_relative_limit": (
            MAX_LOCAL_SECOND_DIFFERENCE_RELATIVE
        ),
        "local_regularity_scale_floor_fraction": (
            LOCAL_REGULARITY_SCALE_FLOOR_FRACTION
        ),
        "covariance_factor_representation": COVARIANCE_FACTOR_REPRESENTATION,
        "covariance_factor_flatten_order": CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER,
        "covariance_uncertainty_scope": COVARIANCE_UNCERTAINTY_SCOPE,
        "covariance_uncertainty_level": COVARIANCE_UNCERTAINTY_LEVEL,
        "covariance_factor_mode_count": mode_count,
        "covariance_factor_rank": factor_rank,
        "covariance_factor_value_count": mode_count * flattened_count,
        "covariance_factor_frobenius_norm": factor_frobenius_norm,
        "numerically_meaningful_uncertainty_relative_floor": (
            MIN_NUMERICALLY_MEANINGFUL_RELATIVE_UNCERTAINTY
        ),
        "target_state_uncertainty_numerical_floor": numerical_uncertainty_floor,
        "covariance_mode_channel_sum_max_abs_residual": (
            max_covariance_normalization_residual
        ),
        "covariance_mode_spin_reversal_max_abs_residual": (
            max_covariance_spin_reversal_residual
        ),
        "covariance_mode_spin_reversal_exact": True,
        "maxwellian_coefficient_sum": math.fsum(
            float(value) for value in normalized_coefficients
        ),
        "maxwellian_quadrature_coarsening_strides": list(
            MAXWELLIAN_COARSENING_STRIDES
        ),
        "maxwellian_quadrature_phase_count": len(coarsening_phase_summaries),
        "maxwellian_quadrature_phase_summaries": coarsening_phase_summaries,
        "maxwellian_quadrature_coarsened_point_count": len(coarse_indices),
        "coarsened_target_state_maxwellian_reactivity_ratio": coarse_ratio,
        "coarsened_target_state_maxwellian_reactivity_ratio_std": coarse_std,
        "coarsened_target_state_maxwellian_reactivity_ratio_lower_3sigma": (
            coarse_lower_3sigma
        ),
        "maxwellian_ratio_quadrature_relative_residual": ratio_quadrature_residual,
        "maxwellian_std_quadrature_relative_residual": std_quadrature_residual,
        "maxwellian_quadrature_max_relative_residual": max_quadrature_residual,
        "maxwellian_quadrature_min_coarsened_lower_3sigma": (
            min_coarsened_lower_3sigma
        ),
        "maxwellian_quadrature_relative_tolerance": (
            MAXWELLIAN_QUADRATURE_RELATIVE_TOLERANCE
        ),
        "maxwellian_quadrature_convergence_pass": True,
        "coarsened_target_ratio_pass_at_3sigma": True,
        "flattened_derivative_weight_count": flattened_count,
        "target_state_resolved_operator_available": True,
        "machine_readable_energy_grid_available": True,
        "systematic_covariance_available": True,
        "local_numeric_reproduction_pass": True,
        "target_ratio_pass_at_3sigma": lower_3sigma >= target_ratio,
    }
    return _canonical_claims_result(
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        claims=claims,
    )


def _derive_reaction_operator(
    payload: dict[str, Any], *, manifest_sha256: str
) -> DerivedArtifactClaims:
    variant = payload.get("operator_variant")
    if type(variant) is not str:
        raise ArtifactPayloadError("operator_variant must be an exact supported discriminator")
    if variant == SCALAR_TARGET_RATIO_CONTROL_VARIANT:
        return _derive_scalar_reaction_operator(payload, manifest_sha256=manifest_sha256)
    if variant == SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT:
        return _derive_six_channel_reaction_operator(payload, manifest_sha256=manifest_sha256)
    raise ArtifactPayloadError("operator_variant must be an exact supported discriminator")


def _ascii_identifier(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in value
        )
    ):
        raise ArtifactPayloadError(f"{name} is invalid")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactPayloadError(f"{name} must be a nonnegative integer (not bool)")
    return value


def _literal_string_list(
    value: object, expected: tuple[str, ...], *, name: str
) -> tuple[str, ...]:
    if type(value) is not list or tuple(value) != expected:
        raise ArtifactPayloadError(f"{name} must exactly match the required ordered variables")
    if any(type(item) is not str for item in value):
        raise ArtifactPayloadError(f"{name} must contain strings")
    return expected


def _correlation_matrix(
    value: object,
    *,
    size: int,
    name: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if type(value) is not list or len(value) != size:
        raise ArtifactPayloadError(f"{name} must be an exact {size}x{size} matrix")
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if type(row) is not list or len(row) != size:
            raise ArtifactPayloadError(f"{name} must be an exact {size}x{size} matrix")
        rows.append(
            [
                _finite_number(item, name=f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.array_equal(matrix, matrix.T):
        raise ArtifactPayloadError(f"{name} must be exactly symmetric")
    if not np.array_equal(np.diag(matrix), np.ones(size, dtype=np.float64)):
        raise ArtifactPayloadError(f"{name} diagonal must be exactly one")
    if np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise ArtifactPayloadError(f"{name} entries must remain in [-1, 1]")
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * size
        * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    )
    if (
        not np.all(np.isfinite(eigenvalues))
        or float(np.min(eigenvalues)) < -eigen_tolerance
    ):
        raise ArtifactPayloadError(f"{name} must be positive semidefinite")
    return matrix


def _gradient_variance(
    gradient: np.ndarray[Any, np.dtype[np.float64]],
    standard_deviations: np.ndarray[Any, np.dtype[np.float64]],
    correlation: np.ndarray[Any, np.dtype[np.float64]],
    *,
    name: str,
) -> float:
    covariance = correlation * np.outer(standard_deviations, standard_deviations)
    variance = float(gradient @ covariance @ gradient)
    variance = _finite_derived(variance, name=f"{name} variance")
    if variance <= 0.0:
        raise ArtifactPayloadError(f"derived {name} variance must be positive")
    return variance


def _engineering_payload_bundle(
    payload: dict[str, Any],
    *,
    evidence_class: str,
    auxiliary_payloads: dict[str, dict[str, Any]] | None,
    typed_payload_sha256s: dict[str, str] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], tuple[tuple[str, str], ...]]:
    expected_auxiliary_roles = ENGINEERING_AUXILIARY_PAYLOAD_ROLES[evidence_class]
    primary_role = ENGINEERING_PRIMARY_PAYLOAD_ROLES[evidence_class]
    auxiliary = _require_object(auxiliary_payloads, name="auxiliary_payloads")
    _require_exact_keys(auxiliary, expected_auxiliary_roles, name="auxiliary_payloads")
    typed_hashes = _require_object(typed_payload_sha256s, name="typed_payload_sha256s")
    expected_typed_roles = expected_auxiliary_roles | frozenset((primary_role,))
    _require_exact_keys(typed_hashes, expected_typed_roles, name="typed_payload_sha256s")
    verified_typed_hashes = {
        role: _manifest_sha256(digest, name=f"typed_payload_sha256s.{role}")
        for role, digest in typed_hashes.items()
    }
    declared_hashes = _require_object(
        payload.get("auxiliary_payload_sha256s"),
        name="auxiliary_payload_sha256s",
    )
    _require_exact_keys(
        declared_hashes,
        expected_auxiliary_roles,
        name="auxiliary_payload_sha256s",
    )
    verified_declared_hashes = {
        role: _manifest_sha256(digest, name=f"auxiliary_payload_sha256s.{role}")
        for role, digest in declared_hashes.items()
    }
    if any(
        verified_declared_hashes[role] != verified_typed_hashes[role]
        for role in expected_auxiliary_roles
    ):
        raise ArtifactPayloadError(
            "auxiliary_payload_sha256s must exactly match manifest-bound payload bytes"
        )
    verified_auxiliary: dict[str, dict[str, Any]] = {}
    for role in expected_auxiliary_roles:
        document = _require_object(auxiliary[role], name=f"auxiliary_payloads.{role}")
        if document.get("payload_role") != role:
            raise ArtifactPayloadError(f"auxiliary_payloads.{role} payload_role mismatch")
        verified_auxiliary[role] = document
    return (
        verified_auxiliary,
        verified_declared_hashes,
        tuple(sorted(verified_typed_hashes.items())),
    )


def _derive_source_measurement(
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    species: str,
    auxiliary_payloads: dict[str, dict[str, Any]] | None,
    typed_payload_sha256s: dict[str, str] | None,
) -> DerivedArtifactClaims:
    if species == "D":
        evidence_class = REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS
        polarization_keys = frozenset(
            (
                "deuteron_vector_polarization",
                "deuteron_vector_polarization_std",
                "deuteron_tensor_polarization",
                "deuteron_tensor_polarization_std",
            )
        )
    else:
        evidence_class = REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS
        polarization_keys = frozenset(
            ("triton_vector_polarization", "triton_vector_polarization_std")
        )
    _document_header(
        payload,
        evidence_class=evidence_class,
        payload_role="measurement",
        extra_keys=frozenset(
            ("plant_configuration", "measurement", "auxiliary_payload_sha256s")
        ),
    )
    auxiliary, auxiliary_hashes, typed_hash_items = _engineering_payload_bundle(
        payload,
        evidence_class=evidence_class,
        auxiliary_payloads=auxiliary_payloads,
        typed_payload_sha256s=typed_payload_sha256s,
    )
    configuration_id, _, _, common_claims = _plant_configuration(
        payload["plant_configuration"]
    )
    measurement = _require_object(payload["measurement"], name="measurement")
    measurement_keys = frozenset(
        (
            "species",
            "output_count",
            "output_count_unit",
            "output_count_std",
            "measurement_duration_s",
            "duration_unit",
            "polarization_unit",
            "uncertainty_level",
        )
    ) | polarization_keys
    _require_exact_keys(measurement, measurement_keys, name="measurement")
    _require_literal(measurement["species"], species, name="measurement.species")
    output_count = _positive_integer(measurement["output_count"], name="measurement.output_count")
    if output_count < MIN_ENGINEERING_EVENT_COUNT:
        raise ArtifactPayloadError(
            f"measurement.output_count must be at least {MIN_ENGINEERING_EVENT_COUNT}"
        )
    output_count_std = _positive_number(
        measurement["output_count_std"], name="measurement.output_count_std"
    )
    duration_s = _positive_number(
        measurement["measurement_duration_s"], name="measurement.measurement_duration_s"
    )
    _require_literal(
        measurement["output_count_unit"], "particles", name="measurement.output_count_unit"
    )
    _require_literal(measurement["duration_unit"], "s", name="measurement.duration_unit")
    _require_literal(
        measurement["polarization_unit"], "1", name="measurement.polarization_unit"
    )
    _require_literal(
        measurement["uncertainty_level"], "1sigma", name="measurement.uncertainty_level"
    )

    event_counts = auxiliary["event_counts"]
    _document_header(
        event_counts,
        evidence_class=evidence_class,
        payload_role="event_counts",
        extra_keys=frozenset(
            (
                "species",
                "observed_output_count",
                "observed_output_count_unit",
                "observed_output_count_std",
            )
        ),
    )
    _require_literal(event_counts["species"], species, name="event_counts.species")
    _require_literal(
        event_counts["observed_output_count_unit"],
        "particles",
        name="event_counts.observed_output_count_unit",
    )
    event_output_count = _positive_integer(
        event_counts["observed_output_count"],
        name="event_counts.observed_output_count",
    )
    event_output_count_std = _positive_number(
        event_counts["observed_output_count_std"],
        name="event_counts.observed_output_count_std",
    )
    if event_output_count != output_count or event_output_count_std != output_count_std:
        raise ArtifactPayloadError("measurement and event_counts must match exactly")

    calibration = auxiliary["calibration"]
    _document_header(
        calibration,
        evidence_class=evidence_class,
        payload_role="calibration",
        extra_keys=frozenset(
            (
                "species",
                "calibration_id",
                "calibration_model",
                "detection_efficiency",
                "detection_efficiency_std",
                "background_count",
                "background_count_std",
                "background_count_unit",
            )
        ),
    )
    _require_literal(calibration["species"], species, name="calibration.species")
    calibration_id = _ascii_identifier(
        calibration["calibration_id"], name="calibration.calibration_id"
    )
    _require_literal(
        calibration["calibration_model"],
        "absolute_count_efficiency_and_background_v1",
        name="calibration.calibration_model",
    )
    efficiency = _positive_number(
        calibration["detection_efficiency"],
        name="calibration.detection_efficiency",
    )
    if efficiency > 1.0:
        raise ArtifactPayloadError("calibration.detection_efficiency must not exceed one")
    efficiency_std = _positive_number(
        calibration["detection_efficiency_std"],
        name="calibration.detection_efficiency_std",
    )
    if efficiency - 3.0 * efficiency_std <= 0.0:
        raise ArtifactPayloadError(
            "calibration detection efficiency must remain positive at 3sigma"
        )
    background_count = _nonnegative_integer(
        calibration["background_count"], name="calibration.background_count"
    )
    background_count_std = _positive_number(
        calibration["background_count_std"],
        name="calibration.background_count_std",
    )
    _require_literal(
        calibration["background_count_unit"],
        "particles",
        name="calibration.background_count_unit",
    )
    background_count_lower_3sigma = background_count - 3.0 * background_count_std
    if background_count_lower_3sigma < 0.0:
        raise ArtifactPayloadError(
            "calibration background count must remain nonnegative at 3sigma"
        )
    corrected_count = output_count - background_count
    if corrected_count < MIN_ENGINEERING_EVENT_COUNT:
        raise ArtifactPayloadError(
            "background-subtracted output count must be at least "
            f"{MIN_ENGINEERING_EVENT_COUNT}"
        )
    corrected_count_lower_3sigma = (
        output_count
        - 3.0 * output_count_std
        - background_count
        - 3.0 * background_count_std
    )
    if corrected_count_lower_3sigma < MIN_ENGINEERING_EVENT_COUNT:
        raise ArtifactPayloadError(
            "background-subtracted output count must remain above the minimum sample "
            "size at 3sigma"
        )

    pipeline = auxiliary["pipeline"]
    pipeline_extra_keys = frozenset(
        (
            "species",
            "pipeline_id",
            "rate_estimator",
            "measurement_duration_s",
            "measurement_duration_std_s",
            "duration_unit",
            "uncertainty_variables",
            "correlation_matrix",
            "uncertainty_level",
        )
    )
    if species == "D":
        pipeline_extra_keys |= frozenset(
            (
                "polarization_uncertainty_variables",
                "polarization_correlation_matrix",
            )
        )
    _document_header(
        pipeline,
        evidence_class=evidence_class,
        payload_role="pipeline",
        extra_keys=pipeline_extra_keys,
    )
    _require_literal(pipeline["species"], species, name="pipeline.species")
    pipeline_id = _ascii_identifier(pipeline["pipeline_id"], name="pipeline.pipeline_id")
    rate_estimator = (
        "background_subtracted_efficiency_corrected_count_over_duration_v1"
    )
    _require_literal(
        pipeline["rate_estimator"], rate_estimator, name="pipeline.rate_estimator"
    )
    pipeline_duration_s = _positive_number(
        pipeline["measurement_duration_s"],
        name="pipeline.measurement_duration_s",
    )
    duration_std_s = _positive_number(
        pipeline["measurement_duration_std_s"],
        name="pipeline.measurement_duration_std_s",
    )
    if pipeline_duration_s != duration_s:
        raise ArtifactPayloadError("measurement and pipeline duration must match exactly")
    if duration_s - 3.0 * duration_std_s <= 0.0:
        raise ArtifactPayloadError("source measurement duration must remain positive at 3sigma")
    _require_literal(pipeline["duration_unit"], "s", name="pipeline.duration_unit")
    _require_literal(
        pipeline["uncertainty_level"], "1sigma", name="pipeline.uncertainty_level"
    )
    _literal_string_list(
        pipeline["uncertainty_variables"],
        SOURCE_RATE_UNCERTAINTY_VARIABLES,
        name="pipeline.uncertainty_variables",
    )
    correlation = _correlation_matrix(
        pipeline["correlation_matrix"],
        size=len(SOURCE_RATE_UNCERTAINTY_VARIABLES),
        name="pipeline.correlation_matrix",
    )
    polarization_correlation: np.ndarray[Any, np.dtype[np.float64]] | None = None
    if species == "D":
        _literal_string_list(
            pipeline["polarization_uncertainty_variables"],
            (
                "deuteron_vector_polarization",
                "deuteron_tensor_polarization",
            ),
            name="pipeline.polarization_uncertainty_variables",
        )
        polarization_correlation = _correlation_matrix(
            pipeline["polarization_correlation_matrix"],
            size=2,
            name="pipeline.polarization_correlation_matrix",
        )

    source_rate_s = _finite_derived(
        corrected_count / efficiency / duration_s,
        name="source_rate_s",
    )
    gradient = np.asarray(
        (
            1.0 / efficiency / duration_s,
            -1.0 / efficiency / duration_s,
            -source_rate_s / efficiency,
            -source_rate_s / duration_s,
        ),
        dtype=np.float64,
    )
    standard_deviations = np.asarray(
        (output_count_std, background_count_std, efficiency_std, duration_std_s),
        dtype=np.float64,
    )
    source_rate_variance_s2 = _gradient_variance(
        gradient,
        standard_deviations,
        correlation,
        name="source_rate_s",
    )
    source_rate_linearized_std_s = _finite_derived(
        math.sqrt(source_rate_variance_s2), name="source_rate_linearized_std_s"
    )
    efficiency_upper_3sigma = min(1.0, efficiency + 3.0 * efficiency_std)
    duration_upper_3sigma_s = _finite_derived(
        duration_s + 3.0 * duration_std_s,
        name="measurement_duration_upper_3sigma_s",
    )
    source_rate_lower_3sigma_s = _finite_derived(
        corrected_count_lower_3sigma
        / efficiency_upper_3sigma
        / duration_upper_3sigma_s,
        name="source_rate_lower_3sigma_s",
    )
    source_rate_bound_model = (
        "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
    )
    required_rate_s = float(common_claims["required_per_species_fuel_rate_s"])

    claims = dict(common_claims)
    claims.update(
        {
            "species": species,
            "reactor_rate_source_demonstrated": source_rate_lower_3sigma_s >= required_rate_s,
            "source_rate_s": source_rate_s,
            "source_rate_linearized_std_s": source_rate_linearized_std_s,
            "source_rate_lower_3sigma_s": source_rate_lower_3sigma_s,
            "source_rate_3sigma_bound_model": source_rate_bound_model,
            "measurement_duration_s": duration_s,
            "measurement_duration_std_s": duration_std_s,
            "output_count": output_count,
            "output_count_std": output_count_std,
            "background_count": background_count,
            "background_count_std": background_count_std,
            "detection_efficiency": efficiency,
            "detection_efficiency_std": efficiency_std,
            "source_rate_estimator": rate_estimator,
            "source_rate_uncertainty_variables": list(
                SOURCE_RATE_UNCERTAINTY_VARIABLES
            ),
            "source_rate_uncertainty_correlation_matrix": correlation.tolist(),
            "source_rate_uncertainty_gradient": gradient.tolist(),
            "linearized_correlated_uncertainty_diagnostic_available": True,
            "source_event_count_calibration_id": calibration_id,
            "source_rate_pipeline_id": pipeline_id,
            "auxiliary_payload_sha256s": auxiliary_hashes,
        }
    )
    if species == "D":
        p_d = _finite_number(
            measurement["deuteron_vector_polarization"],
            name="measurement.deuteron_vector_polarization",
        )
        p_d_std = _positive_number(
            measurement["deuteron_vector_polarization_std"],
            name="measurement.deuteron_vector_polarization_std",
        )
        p_zz = _finite_number(
            measurement["deuteron_tensor_polarization"],
            name="measurement.deuteron_tensor_polarization",
        )
        p_zz_std = _positive_number(
            measurement["deuteron_tensor_polarization_std"],
            name="measurement.deuteron_tensor_polarization_std",
        )
        if polarization_correlation is None:  # pragma: no cover - branch invariant
            raise ArtifactPayloadError("deuterium polarization correlation is missing")
        polarization_covariance = polarization_correlation * np.outer(
            np.asarray((p_d_std, p_zz_std), dtype=np.float64),
            np.asarray((p_d_std, p_zz_std), dtype=np.float64),
        )
        polarization_covariance_determinant = _finite_derived(
            max(0.0, float(np.linalg.det(polarization_covariance))),
            name="deuterium polarization covariance determinant",
        )
        p_d_low, p_d_high = p_d - 3.0 * p_d_std, p_d + 3.0 * p_d_std
        p_zz_low, p_zz_high = p_zz - 3.0 * p_zz_std, p_zz + 3.0 * p_zz_std
        if not (
            0.0 <= p_d_low <= p_d_high <= 1.0
            and -2.0 <= p_zz_low <= p_zz_high <= 1.0
            and 3.0 * p_d_high <= 2.0 + p_zz_low
        ):
            raise ArtifactPayloadError(
                "deuterium source polarization is not physical over its independent 3sigma box"
            )
        claims.update(
            {
                "deuteron_vector_polarization": p_d,
                "deuteron_vector_polarization_std": p_d_std,
                "deuteron_vector_polarization_lower_3sigma": p_d_low,
                "deuteron_vector_polarization_upper_3sigma": p_d_high,
                "deuteron_tensor_polarization": p_zz,
                "deuteron_tensor_polarization_std": p_zz_std,
                "deuteron_tensor_polarization_lower_3sigma": p_zz_low,
                "deuteron_tensor_polarization_upper_3sigma": p_zz_high,
                "deuteron_polarization_uncertainty_variables": [
                    "deuteron_vector_polarization",
                    "deuteron_tensor_polarization",
                ],
                "deuteron_polarization_uncertainty_correlation_matrix": (
                    polarization_correlation.tolist()
                ),
                "deuteron_polarization_uncertainty_covariance_matrix": (
                    polarization_covariance.tolist()
                ),
                "deuteron_polarization_uncertainty_covariance_determinant": (
                    polarization_covariance_determinant
                ),
            }
        )
    else:
        p_t = _finite_number(
            measurement["triton_vector_polarization"],
            name="measurement.triton_vector_polarization",
        )
        p_t_std = _positive_number(
            measurement["triton_vector_polarization_std"],
            name="measurement.triton_vector_polarization_std",
        )
        p_t_low, p_t_high = p_t - 3.0 * p_t_std, p_t + 3.0 * p_t_std
        if not 0.0 <= p_t_low <= p_t_high <= 1.0:
            raise ArtifactPayloadError(
                "tritium source polarization is not physical over its 3sigma interval"
            )
        claims.update(
            {
                "triton_vector_polarization": p_t,
                "triton_vector_polarization_std": p_t_std,
                "triton_vector_polarization_lower_3sigma": p_t_low,
                "triton_vector_polarization_upper_3sigma": p_t_high,
            }
        )
    return _canonical_claims_result(
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        claims=claims,
        plant_configuration_id=configuration_id,
        typed_payload_sha256s=typed_hash_items,
    )


def _derive_burn_retention(
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    auxiliary_payloads: dict[str, dict[str, Any]] | None,
    typed_payload_sha256s: dict[str, str] | None,
) -> DerivedArtifactClaims:
    evidence_class = BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS
    _document_header(
        payload,
        evidence_class=evidence_class,
        payload_role="measurement",
        extra_keys=frozenset(
            ("plant_configuration", "measurement", "auxiliary_payload_sha256s")
        ),
    )
    auxiliary, auxiliary_hashes, typed_hash_items = _engineering_payload_bundle(
        payload,
        evidence_class=evidence_class,
        auxiliary_payloads=auxiliary_payloads,
        typed_payload_sha256s=typed_payload_sha256s,
    )
    configuration_id, _, _, common_claims = _plant_configuration(
        payload["plant_configuration"]
    )
    measurement = _require_object(payload["measurement"], name="measurement")
    keys = frozenset(
        (
            "burn_weighted_dt_polarization_product",
            "burn_weighted_dt_polarization_product_std",
            "burn_weighted_deuteron_tensor_polarization",
            "burn_weighted_deuteron_tensor_polarization_std",
            "polarization_unit",
            "uncertainty_level",
            "reaction_count",
            "reaction_count_unit",
            "measurement_duration_s",
            "duration_unit",
            "deuterium_source_manifest_sha256",
            "tritium_source_manifest_sha256",
            "reaction_operator_manifest_sha256",
            "quantization_axis",
            "initial_state_basis",
            "deuteron_tensor_polarization_convention",
            "collision_ensemble_temperature_kev",
            "temperature_unit",
            "spin_state_energy_dependence",
        )
    )
    _require_exact_keys(measurement, keys, name="measurement")
    _require_literal(
        measurement["polarization_unit"], "1", name="measurement.polarization_unit"
    )
    _require_literal(
        measurement["uncertainty_level"], "1sigma", name="measurement.uncertainty_level"
    )
    _require_literal(
        measurement["reaction_count_unit"], "reactions", name="measurement.reaction_count_unit"
    )
    _require_literal(measurement["duration_unit"], "s", name="measurement.duration_unit")
    _require_literal(
        measurement["quantization_axis"],
        LONGITUDINAL_QUANTIZATION_AXIS,
        name="measurement.quantization_axis",
    )
    _require_literal(
        measurement["initial_state_basis"],
        INITIAL_STATE_BASIS,
        name="measurement.initial_state_basis",
    )
    tensor_convention = "Pzz=p_plus+p_minus-2*p_zero"
    _require_literal(
        measurement["deuteron_tensor_polarization_convention"],
        tensor_convention,
        name="measurement.deuteron_tensor_polarization_convention",
    )
    collision_temperature_kev = _finite_number(
        measurement["collision_ensemble_temperature_kev"],
        name="measurement.collision_ensemble_temperature_kev",
    )
    if collision_temperature_kev != REACTION_TEMPERATURE_KEV:
        raise ArtifactPayloadError(
            "measurement.collision_ensemble_temperature_kev must be exactly 10 keV"
        )
    _require_literal(
        measurement["temperature_unit"], "keV", name="measurement.temperature_unit"
    )
    spin_state_energy_dependence = (
        "declared_energy_independent_diagonal_spin_state_engineering_control_v1"
    )
    _require_literal(
        measurement["spin_state_energy_dependence"],
        spin_state_energy_dependence,
        name="measurement.spin_state_energy_dependence",
    )
    product = _finite_number(
        measurement["burn_weighted_dt_polarization_product"],
        name="measurement.burn_weighted_dt_polarization_product",
    )
    product_std = _positive_number(
        measurement["burn_weighted_dt_polarization_product_std"],
        name="measurement.burn_weighted_dt_polarization_product_std",
    )
    p_zz = _finite_number(
        measurement["burn_weighted_deuteron_tensor_polarization"],
        name="measurement.burn_weighted_deuteron_tensor_polarization",
    )
    p_zz_std = _positive_number(
        measurement["burn_weighted_deuteron_tensor_polarization_std"],
        name="measurement.burn_weighted_deuteron_tensor_polarization_std",
    )
    reaction_count = _positive_integer(
        measurement["reaction_count"], name="measurement.reaction_count"
    )
    if reaction_count < MIN_ENGINEERING_EVENT_COUNT:
        raise ArtifactPayloadError(
            f"measurement.reaction_count must be at least {MIN_ENGINEERING_EVENT_COUNT}"
        )
    duration_s = _positive_number(
        measurement["measurement_duration_s"], name="measurement.measurement_duration_s"
    )

    burn_events = auxiliary["burn_events"]
    _document_header(
        burn_events,
        evidence_class=evidence_class,
        payload_role="burn_events",
        extra_keys=frozenset(
            (
                "event_dataset_id",
                "reaction_count",
                "reaction_count_unit",
                "measurement_duration_s",
                "measurement_duration_std_s",
                "duration_unit",
            )
        ),
    )
    event_dataset_id = _ascii_identifier(
        burn_events["event_dataset_id"], name="burn_events.event_dataset_id"
    )
    event_reaction_count = _positive_integer(
        burn_events["reaction_count"], name="burn_events.reaction_count"
    )
    event_duration_s = _positive_number(
        burn_events["measurement_duration_s"],
        name="burn_events.measurement_duration_s",
    )
    duration_std_s = _positive_number(
        burn_events["measurement_duration_std_s"],
        name="burn_events.measurement_duration_std_s",
    )
    _require_literal(
        burn_events["reaction_count_unit"],
        "reactions",
        name="burn_events.reaction_count_unit",
    )
    _require_literal(burn_events["duration_unit"], "s", name="burn_events.duration_unit")
    if event_reaction_count != reaction_count or event_duration_s != duration_s:
        raise ArtifactPayloadError("measurement and burn_events must match exactly")
    if duration_s - 3.0 * duration_std_s <= 0.0:
        raise ArtifactPayloadError("burn measurement duration must remain positive at 3sigma")

    calibration = auxiliary["calibration"]
    _document_header(
        calibration,
        evidence_class=evidence_class,
        payload_role="calibration",
        extra_keys=frozenset(
            (
                "calibration_id",
                "calibration_model",
                "product_observable",
                "tensor_observable",
            )
        ),
    )
    calibration_id = _ascii_identifier(
        calibration["calibration_id"], name="calibration.calibration_id"
    )
    _require_literal(
        calibration["calibration_model"],
        "absolute_burn_product_and_tensor_polarimetry_v1",
        name="calibration.calibration_model",
    )
    _require_literal(
        calibration["product_observable"],
        "burn_weighted_dt_vector_polarization_product",
        name="calibration.product_observable",
    )
    _require_literal(
        calibration["tensor_observable"],
        "burn_weighted_deuteron_tensor_polarization",
        name="calibration.tensor_observable",
    )

    estimator = auxiliary["estimator"]
    _document_header(
        estimator,
        evidence_class=evidence_class,
        payload_role="estimator",
        extra_keys=frozenset(
            (
                "estimator_id",
                "observable_semantics",
                "uncertainty_variables",
                "correlation_matrix",
                "uncertainty_level",
                "operator_burn_uncertainty_relationship",
                "quantization_axis",
                "initial_state_basis",
                "deuteron_tensor_polarization_convention",
                "collision_ensemble_temperature_kev",
                "temperature_unit",
                "spin_state_energy_dependence",
            )
        ),
    )
    estimator_id = _ascii_identifier(
        estimator["estimator_id"], name="estimator.estimator_id"
    )
    observable_semantics = (
        "burn_weighted_dt_vector_product_and_deuteron_tensor_joint_estimator_v1"
    )
    _require_literal(
        estimator["observable_semantics"],
        observable_semantics,
        name="estimator.observable_semantics",
    )
    _require_literal(
        estimator["uncertainty_level"], "1sigma", name="estimator.uncertainty_level"
    )
    operator_burn_uncertainty_relationship = (
        "unknown_cross_artifact_correlation_componentwise_3sigma_robust_box_v1"
    )
    _require_literal(
        estimator["operator_burn_uncertainty_relationship"],
        operator_burn_uncertainty_relationship,
        name="estimator.operator_burn_uncertainty_relationship",
    )
    for field_name in (
        "quantization_axis",
        "initial_state_basis",
        "deuteron_tensor_polarization_convention",
        "collision_ensemble_temperature_kev",
        "temperature_unit",
        "spin_state_energy_dependence",
    ):
        if estimator[field_name] != measurement[field_name]:
            raise ArtifactPayloadError(
                f"estimator.{field_name} must match measurement.{field_name} exactly"
            )
    _literal_string_list(
        estimator["uncertainty_variables"],
        BURN_RETENTION_UNCERTAINTY_VARIABLES,
        name="estimator.uncertainty_variables",
    )
    correlation = _correlation_matrix(
        estimator["correlation_matrix"],
        size=len(BURN_RETENTION_UNCERTAINTY_VARIABLES),
        name="estimator.correlation_matrix",
    )
    burn_covariance = correlation * np.outer(
        np.asarray((product_std, p_zz_std), dtype=np.float64),
        np.asarray((product_std, p_zz_std), dtype=np.float64),
    )
    covariance_determinant = _finite_derived(
        max(0.0, float(np.linalg.det(burn_covariance))),
        name="burn retention covariance determinant",
    )
    product_low = _finite_derived(product - 3.0 * product_std, name="burn product lower bound")
    product_high = _finite_derived(product + 3.0 * product_std, name="burn product upper bound")
    p_zz_low = _finite_derived(p_zz - 3.0 * p_zz_std, name="burn pzz lower bound")
    p_zz_high = _finite_derived(p_zz + 3.0 * p_zz_std, name="burn pzz upper bound")
    if not 0.0 <= product_low <= product_high <= 1.0:
        raise ArtifactPayloadError("burn polarization product must remain in [0, 1] at 3sigma")
    if not -2.0 <= p_zz_low <= p_zz_high <= 1.0:
        raise ArtifactPayloadError("burn deuteron tensor polarization is unphysical at 3sigma")
    if product_high > (2.0 + p_zz_low) / 3.0:
        raise ArtifactPayloadError(
            "burn product and tensor polarization violate the spin-1 positivity envelope"
        )
    deuterium_hash = _manifest_sha256(
        measurement["deuterium_source_manifest_sha256"],
        name="measurement.deuterium_source_manifest_sha256",
    )
    tritium_hash = _manifest_sha256(
        measurement["tritium_source_manifest_sha256"],
        name="measurement.tritium_source_manifest_sha256",
    )
    if deuterium_hash == tritium_hash:
        raise ArtifactPayloadError("D and T source manifests must be distinct")
    reaction_operator_hash = _manifest_sha256(
        measurement["reaction_operator_manifest_sha256"],
        name="measurement.reaction_operator_manifest_sha256",
    )
    if reaction_operator_hash in (deuterium_hash, tritium_hash):
        raise ArtifactPayloadError("reaction operator and source manifests must be distinct")
    claims = dict(common_claims)
    claims.update(
        {
            "burn_weighted_dt_polarization_product_measured": True,
            "burn_weighted_dt_polarization_product": product,
            "burn_weighted_dt_polarization_product_std": product_std,
            "burn_weighted_dt_polarization_product_lower_3sigma": product_low,
            "burn_weighted_dt_polarization_product_upper_3sigma": product_high,
            "burn_weighted_deuteron_tensor_polarization": p_zz,
            "burn_weighted_deuteron_tensor_polarization_std": p_zz_std,
            "burn_weighted_deuteron_tensor_polarization_lower_3sigma": p_zz_low,
            "burn_weighted_deuteron_tensor_polarization_upper_3sigma": p_zz_high,
            "measurement_duration_s": duration_s,
            "measurement_duration_std_s": duration_std_s,
            "reaction_count": reaction_count,
            "burn_event_dataset_id": event_dataset_id,
            "burn_calibration_id": calibration_id,
            "burn_estimator_id": estimator_id,
            "burn_observable_semantics": observable_semantics,
            "operator_burn_uncertainty_relationship": (
                operator_burn_uncertainty_relationship
            ),
            "burn_quantization_axis": LONGITUDINAL_QUANTIZATION_AXIS,
            "burn_initial_state_basis": INITIAL_STATE_BASIS,
            "burn_deuteron_tensor_polarization_convention": tensor_convention,
            "burn_collision_ensemble_temperature_kev": collision_temperature_kev,
            "burn_spin_state_energy_dependence": spin_state_energy_dependence,
            "burn_uncertainty_variables": list(BURN_RETENTION_UNCERTAINTY_VARIABLES),
            "burn_uncertainty_correlation_matrix": correlation.tolist(),
            "burn_uncertainty_covariance_matrix": burn_covariance.tolist(),
            "burn_uncertainty_covariance_determinant": covariance_determinant,
            "strict_correlated_uncertainty_propagation_pass": True,
            "auxiliary_payload_sha256s": auxiliary_hashes,
            "deuterium_source_manifest_sha256": deuterium_hash,
            "tritium_source_manifest_sha256": tritium_hash,
            "reaction_operator_manifest_sha256": reaction_operator_hash,
        }
    )
    return _canonical_claims_result(
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        claims=claims,
        plant_configuration_id=configuration_id,
        linked_manifest_sha256s=(
            ("deuterium_source", deuterium_hash),
            ("tritium_source", tritium_hash),
            ("reaction_operator", reaction_operator_hash),
        ),
        typed_payload_sha256s=typed_hash_items,
    )


def _derive_wall_plug_ledger(
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    auxiliary_payloads: dict[str, dict[str, Any]] | None,
    typed_payload_sha256s: dict[str, str] | None,
) -> DerivedArtifactClaims:
    evidence_class = POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS
    _document_header(
        payload,
        evidence_class=evidence_class,
        payload_role="component_ledger",
        extra_keys=frozenset(
            ("plant_configuration", "ledger", "auxiliary_payload_sha256s")
        ),
    )
    auxiliary, auxiliary_hashes, typed_hash_items = _engineering_payload_bundle(
        payload,
        evidence_class=evidence_class,
        auxiliary_payloads=auxiliary_payloads,
        typed_payload_sha256s=typed_payload_sha256s,
    )
    configuration_id, _, _, common_claims = _plant_configuration(
        payload["plant_configuration"]
    )
    ledger = _require_object(payload["ledger"], name="ledger")
    keys = frozenset(
        (
            "uncertainty_model",
            "energy_unit",
            "components",
            "injected_dt_pair_count",
            "pair_count_unit",
            "measurement_duration_s",
            "duration_unit",
            "deuterium_source_manifest_sha256",
            "tritium_source_manifest_sha256",
        )
    )
    _require_exact_keys(ledger, keys, name="ledger")
    _require_literal(
        ledger["uncertainty_model"],
        "manifest_bound_correlated_1sigma",
        name="ledger.uncertainty_model",
    )
    _require_literal(ledger["energy_unit"], "J", name="ledger.energy_unit")
    _require_literal(ledger["pair_count_unit"], "pairs", name="ledger.pair_count_unit")
    _require_literal(ledger["duration_unit"], "s", name="ledger.duration_unit")
    pair_count = _positive_integer(
        ledger["injected_dt_pair_count"], name="ledger.injected_dt_pair_count"
    )
    if pair_count < MIN_ENGINEERING_EVENT_COUNT:
        raise ArtifactPayloadError(
            f"ledger.injected_dt_pair_count must be at least {MIN_ENGINEERING_EVENT_COUNT}"
        )
    duration_s = _positive_number(
        ledger["measurement_duration_s"], name="ledger.measurement_duration_s"
    )
    components = _require_object(ledger["components"], name="ledger.components")
    _require_exact_keys(components, WALL_PLUG_COMPONENT_CATEGORIES, name="ledger.components")
    component_energies: dict[str, float] = {}
    component_stds: dict[str, float] = {}
    for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES):
        component = _require_object(components[category], name=f"ledger.components.{category}")
        _require_exact_keys(
            component, frozenset(("energy_j", "energy_std_j")), name=f"ledger.components.{category}"
        )
        component_energies[category] = _positive_number(
            component["energy_j"], name=f"ledger.components.{category}.energy_j"
        )
        component_stds[category] = _positive_number(
            component["energy_std_j"], name=f"ledger.components.{category}.energy_std_j"
        )
        if component_energies[category] - 3.0 * component_stds[category] <= 0.0:
            raise ArtifactPayloadError(
                f"wall-plug component {category} must remain positive at 3sigma"
            )

    meter_records = auxiliary["meter_records"]
    _document_header(
        meter_records,
        evidence_class=evidence_class,
        payload_role="meter_records",
        extra_keys=frozenset(
            (
                "energy_unit",
                "components",
                "injected_dt_pair_count",
                "injected_dt_pair_count_std",
                "pair_count_unit",
                "measurement_duration_s",
                "measurement_duration_std_s",
                "duration_unit",
            )
        ),
    )
    _require_literal(meter_records["energy_unit"], "J", name="meter_records.energy_unit")
    _require_literal(
        meter_records["pair_count_unit"], "pairs", name="meter_records.pair_count_unit"
    )
    _require_literal(meter_records["duration_unit"], "s", name="meter_records.duration_unit")
    meter_pair_count = _positive_integer(
        meter_records["injected_dt_pair_count"],
        name="meter_records.injected_dt_pair_count",
    )
    pair_count_std = _positive_number(
        meter_records["injected_dt_pair_count_std"],
        name="meter_records.injected_dt_pair_count_std",
    )
    meter_duration_s = _positive_number(
        meter_records["measurement_duration_s"],
        name="meter_records.measurement_duration_s",
    )
    duration_std_s = _positive_number(
        meter_records["measurement_duration_std_s"],
        name="meter_records.measurement_duration_std_s",
    )
    if meter_pair_count != pair_count or meter_duration_s != duration_s:
        raise ArtifactPayloadError("component_ledger and meter_records must match exactly")
    if pair_count - 3.0 * pair_count_std <= 0.0:
        raise ArtifactPayloadError("injected pair count must remain positive at 3sigma")
    if duration_s - 3.0 * duration_std_s <= 0.0:
        raise ArtifactPayloadError("wall-plug duration must remain positive at 3sigma")
    meter_components = _require_object(
        meter_records["components"], name="meter_records.components"
    )
    _require_exact_keys(
        meter_components,
        WALL_PLUG_COMPONENT_CATEGORIES,
        name="meter_records.components",
    )
    meter_ids: list[str] = []
    for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES):
        meter_component = _require_object(
            meter_components[category], name=f"meter_records.components.{category}"
        )
        _require_exact_keys(
            meter_component,
            frozenset(("energy_j", "energy_std_j", "meter_id")),
            name=f"meter_records.components.{category}",
        )
        meter_energy = _positive_number(
            meter_component["energy_j"],
            name=f"meter_records.components.{category}.energy_j",
        )
        meter_std = _positive_number(
            meter_component["energy_std_j"],
            name=f"meter_records.components.{category}.energy_std_j",
        )
        meter_id = _ascii_identifier(
            meter_component["meter_id"],
            name=f"meter_records.components.{category}.meter_id",
        )
        if (
            meter_energy != component_energies[category]
            or meter_std != component_stds[category]
        ):
            raise ArtifactPayloadError(
                f"component_ledger and meter_records disagree for {category}"
            )
        meter_ids.append(meter_id)
    if len(set(meter_ids)) != len(meter_ids):
        raise ArtifactPayloadError("wall-plug energy meter identifiers must be distinct")

    calibrations = auxiliary["calibrations"]
    _document_header(
        calibrations,
        evidence_class=evidence_class,
        payload_role="calibrations",
        extra_keys=frozenset(
            (
                "calibration_model",
                "energy_meter_calibration_ids",
                "pair_counter_calibration_id",
                "clock_calibration_id",
            )
        ),
    )
    _require_literal(
        calibrations["calibration_model"],
        "traceable_absolute_energy_count_and_time_v1",
        name="calibrations.calibration_model",
    )
    energy_calibration_ids = _require_object(
        calibrations["energy_meter_calibration_ids"],
        name="calibrations.energy_meter_calibration_ids",
    )
    _require_exact_keys(
        energy_calibration_ids,
        WALL_PLUG_COMPONENT_CATEGORIES,
        name="calibrations.energy_meter_calibration_ids",
    )
    verified_energy_calibration_ids = {
        category: _ascii_identifier(
            energy_calibration_ids[category],
            name=f"calibrations.energy_meter_calibration_ids.{category}",
        )
        for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    }
    pair_counter_calibration_id = _ascii_identifier(
        calibrations["pair_counter_calibration_id"],
        name="calibrations.pair_counter_calibration_id",
    )
    clock_calibration_id = _ascii_identifier(
        calibrations["clock_calibration_id"],
        name="calibrations.clock_calibration_id",
    )
    all_calibration_ids = tuple(verified_energy_calibration_ids.values()) + (
        pair_counter_calibration_id,
        clock_calibration_id,
    )
    if len(set(all_calibration_ids)) != len(all_calibration_ids):
        raise ArtifactPayloadError("wall-plug calibration identifiers must be distinct")

    uncertainty_pipeline = auxiliary["uncertainty_pipeline"]
    _document_header(
        uncertainty_pipeline,
        evidence_class=evidence_class,
        payload_role="uncertainty_pipeline",
        extra_keys=frozenset(
            (
                "pipeline_id",
                "uncertainty_propagation",
                "uncertainty_variables",
                "correlation_matrix",
                "uncertainty_level",
            )
        ),
    )
    pipeline_id = _ascii_identifier(
        uncertainty_pipeline["pipeline_id"],
        name="uncertainty_pipeline.pipeline_id",
    )
    uncertainty_propagation = "first_order_gradient_covariance_v1"
    _require_literal(
        uncertainty_pipeline["uncertainty_propagation"],
        uncertainty_propagation,
        name="uncertainty_pipeline.uncertainty_propagation",
    )
    _require_literal(
        uncertainty_pipeline["uncertainty_level"],
        "1sigma",
        name="uncertainty_pipeline.uncertainty_level",
    )
    _literal_string_list(
        uncertainty_pipeline["uncertainty_variables"],
        WALL_PLUG_UNCERTAINTY_VARIABLES,
        name="uncertainty_pipeline.uncertainty_variables",
    )
    correlation = _correlation_matrix(
        uncertainty_pipeline["correlation_matrix"],
        size=len(WALL_PLUG_UNCERTAINTY_VARIABLES),
        name="uncertainty_pipeline.correlation_matrix",
    )

    try:
        total_energy_j = math.fsum(component_energies.values())
    except OverflowError as exc:
        raise ArtifactPayloadError("wall-plug component sum overflowed") from exc
    total_energy_j = _finite_derived(total_energy_j, name="wall_plug_total_energy_j")
    standard_deviations = np.asarray(
        (
            *(component_stds[category] for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES)),
            pair_count_std,
            duration_std_s,
        ),
        dtype=np.float64,
    )
    total_energy_gradient = np.asarray(
        (*([1.0] * len(WALL_PLUG_COMPONENT_CATEGORIES)), 0.0, 0.0),
        dtype=np.float64,
    )
    total_energy_variance_j2 = _gradient_variance(
        total_energy_gradient,
        standard_deviations,
        correlation,
        name="wall_plug_total_energy_j",
    )
    total_energy_linearized_std_j = _finite_derived(
        math.sqrt(total_energy_variance_j2),
        name="wall_plug_total_energy_linearized_std_j",
    )
    component_energy_lower_3sigma_j = {
        category: component_energies[category] - 3.0 * component_stds[category]
        for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    }
    component_energy_upper_3sigma_j = {
        category: component_energies[category] + 3.0 * component_stds[category]
        for category in sorted(WALL_PLUG_COMPONENT_CATEGORIES)
    }
    total_energy_lower_3sigma_j = _finite_derived(
        math.fsum(component_energy_lower_3sigma_j.values()),
        name="wall_plug_total_energy_lower_3sigma_j",
    )
    total_energy_upper_3sigma_j = _finite_derived(
        math.fsum(component_energy_upper_3sigma_j.values()),
        name="wall_plug_total_energy_upper_3sigma_j",
    )
    energy_per_pair_ev = _finite_derived(
        total_energy_j / pair_count / EV_TO_JOULE,
        name="wall_plug_energy_per_injected_dt_pair_ev",
    )
    energy_per_pair_gradient = np.asarray(
        (
            *(
                [1.0 / pair_count / EV_TO_JOULE]
                * len(WALL_PLUG_COMPONENT_CATEGORIES)
            ),
            -total_energy_j / (pair_count * pair_count) / EV_TO_JOULE,
            0.0,
        ),
        dtype=np.float64,
    )
    energy_per_pair_variance_ev2 = _gradient_variance(
        energy_per_pair_gradient,
        standard_deviations,
        correlation,
        name="wall_plug_energy_per_injected_dt_pair_ev",
    )
    energy_linearized_std_per_pair_ev = _finite_derived(
        math.sqrt(energy_per_pair_variance_ev2),
        name="wall_plug_energy_linearized_std_per_injected_dt_pair_ev",
    )
    pair_count_lower_3sigma = pair_count - 3.0 * pair_count_std
    pair_count_upper_3sigma = pair_count + 3.0 * pair_count_std
    energy_per_pair_lower_3sigma_ev = _finite_derived(
        total_energy_lower_3sigma_j / pair_count_upper_3sigma / EV_TO_JOULE,
        name="wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev",
    )
    energy_per_pair_upper_3sigma_ev = _finite_derived(
        total_energy_upper_3sigma_j / pair_count_lower_3sigma / EV_TO_JOULE,
        name="wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev",
    )
    energy_robust_at_3sigma = (
        total_energy_lower_3sigma_j > 0.0
        and energy_per_pair_lower_3sigma_ev > 0.0
    )
    if not energy_robust_at_3sigma:
        raise ArtifactPayloadError(
            "wall-plug total and per-pair energy must remain positive at 3sigma"
        )
    pair_flow_rate_s = _finite_derived(
        pair_count / duration_s,
        name="measurement_injected_dt_pair_flow_rate_s",
    )
    pair_flow_gradient = np.asarray(
        (
            *([0.0] * len(WALL_PLUG_COMPONENT_CATEGORIES)),
            1.0 / duration_s,
            -pair_count / (duration_s * duration_s),
        ),
        dtype=np.float64,
    )
    pair_flow_variance_s2 = _gradient_variance(
        pair_flow_gradient,
        standard_deviations,
        correlation,
        name="measurement_injected_dt_pair_flow_rate_s",
    )
    pair_flow_linearized_std_s = _finite_derived(
        math.sqrt(pair_flow_variance_s2),
        name="measurement_injected_dt_pair_flow_rate_linearized_std_s",
    )
    pair_flow_lower_3sigma_s = _finite_derived(
        pair_count_lower_3sigma / (duration_s + 3.0 * duration_std_s),
        name="measurement_injected_dt_pair_flow_rate_lower_3sigma_s",
    )
    pair_flow_upper_3sigma_s = _finite_derived(
        pair_count_upper_3sigma / (duration_s - 3.0 * duration_std_s),
        name="measurement_injected_dt_pair_flow_rate_upper_3sigma_s",
    )
    wall_plug_bound_model = (
        "unknown_correlation_componentwise_3sigma_box_exact_monotone_v1"
    )
    required_rate_s = float(common_claims["required_per_species_fuel_rate_s"])
    deuterium_hash = _manifest_sha256(
        ledger["deuterium_source_manifest_sha256"],
        name="ledger.deuterium_source_manifest_sha256",
    )
    tritium_hash = _manifest_sha256(
        ledger["tritium_source_manifest_sha256"],
        name="ledger.tritium_source_manifest_sha256",
    )
    if deuterium_hash == tritium_hash:
        raise ArtifactPayloadError("D and T source manifests must be distinct")
    claims = dict(common_claims)
    claims.update(
        {
            "complete_wall_plug_energy_per_injected_pair_measured": (
                energy_robust_at_3sigma
            ),
            "complete_wall_plug_uncertainty_flow_and_provenance_available": (
                energy_robust_at_3sigma
            ),
            "wall_plug_component_categories": sorted(WALL_PLUG_COMPONENT_CATEGORIES),
            "wall_plug_component_energies_j": component_energies,
            "wall_plug_component_energy_stds_j": component_stds,
            "wall_plug_component_energy_lower_3sigma_j": (
                component_energy_lower_3sigma_j
            ),
            "wall_plug_component_energy_upper_3sigma_j": (
                component_energy_upper_3sigma_j
            ),
            "wall_plug_total_energy_j": total_energy_j,
            "wall_plug_total_energy_linearized_std_j": (
                total_energy_linearized_std_j
            ),
            "wall_plug_total_energy_lower_3sigma_j": total_energy_lower_3sigma_j,
            "wall_plug_total_energy_upper_3sigma_j": total_energy_upper_3sigma_j,
            "wall_plug_energy_per_injected_dt_pair_ev": energy_per_pair_ev,
            "wall_plug_energy_linearized_std_per_injected_dt_pair_ev": (
                energy_linearized_std_per_pair_ev
            ),
            "wall_plug_energy_per_injected_dt_pair_lower_3sigma_ev": (
                energy_per_pair_lower_3sigma_ev
            ),
            "wall_plug_energy_per_injected_dt_pair_upper_3sigma_ev": (
                energy_per_pair_upper_3sigma_ev
            ),
            "wall_plug_energy_robust_at_3sigma": energy_robust_at_3sigma,
            "measurement_duration_s": duration_s,
            "measurement_duration_std_s": duration_std_s,
            "measurement_injected_dt_pair_count": pair_count,
            "measurement_injected_dt_pair_count_std": pair_count_std,
            "measurement_injected_dt_pair_flow_rate_s": pair_flow_rate_s,
            "measurement_injected_dt_pair_flow_rate_linearized_std_s": (
                pair_flow_linearized_std_s
            ),
            "measurement_injected_dt_pair_flow_rate_lower_3sigma_s": (
                pair_flow_lower_3sigma_s
            ),
            "measurement_injected_dt_pair_flow_rate_upper_3sigma_s": (
                pair_flow_upper_3sigma_s
            ),
            "measurement_pair_flow_meets_required_throughput": (
                pair_flow_lower_3sigma_s >= required_rate_s
            ),
            "uncertainty_model": "manifest_bound_correlated_1sigma",
            "wall_plug_uncertainty_variables": list(WALL_PLUG_UNCERTAINTY_VARIABLES),
            "wall_plug_uncertainty_correlation_matrix": correlation.tolist(),
            "wall_plug_total_energy_gradient": total_energy_gradient.tolist(),
            "wall_plug_energy_per_pair_gradient": energy_per_pair_gradient.tolist(),
            "wall_plug_pair_flow_gradient": pair_flow_gradient.tolist(),
            "wall_plug_uncertainty_propagation": uncertainty_propagation,
            "wall_plug_3sigma_bound_model": wall_plug_bound_model,
            "wall_plug_uncertainty_pipeline_id": pipeline_id,
            "wall_plug_energy_meter_ids": meter_ids,
            "wall_plug_energy_meter_calibration_ids": verified_energy_calibration_ids,
            "wall_plug_pair_counter_calibration_id": pair_counter_calibration_id,
            "wall_plug_clock_calibration_id": clock_calibration_id,
            "linearized_correlated_uncertainty_diagnostic_available": True,
            "auxiliary_payload_sha256s": auxiliary_hashes,
            "deuterium_source_manifest_sha256": deuterium_hash,
            "tritium_source_manifest_sha256": tritium_hash,
        }
    )
    return _canonical_claims_result(
        evidence_class=evidence_class,
        manifest_sha256=manifest_sha256,
        claims=claims,
        plant_configuration_id=configuration_id,
        linked_manifest_sha256s=(
            ("deuterium_source", deuterium_hash),
            ("tritium_source", tritium_hash),
        ),
        typed_payload_sha256s=typed_hash_items,
    )


def derive_artifact_claims(
    evidence_class: str,
    payload: dict[str, Any],
    *,
    manifest_sha256: str,
    auxiliary_payloads: dict[str, dict[str, Any]] | None = None,
    typed_payload_sha256s: dict[str, str] | None = None,
) -> DerivedArtifactClaims:
    """Validate one strict raw payload and derive its immutable summary claims."""

    if type(evidence_class) is not str or evidence_class not in KNOWN_EVIDENCE_CLASSES:
        raise ArtifactPayloadError("unknown evidence_class")
    verified_manifest_sha256 = _manifest_sha256(manifest_sha256, name="manifest_sha256")
    document = _require_object(payload, name="payload")
    try:
        if evidence_class == POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS:
            if auxiliary_payloads not in (None, {}):
                raise ArtifactPayloadError("reaction operator does not accept auxiliary payloads")
            if typed_payload_sha256s is not None:
                typed_hashes = _require_object(
                    typed_payload_sha256s, name="typed_payload_sha256s"
                )
                _require_exact_keys(
                    typed_hashes,
                    frozenset(("operator_grid",)),
                    name="typed_payload_sha256s",
                )
                _manifest_sha256(
                    typed_hashes["operator_grid"],
                    name="typed_payload_sha256s.operator_grid",
                )
            return _derive_reaction_operator(document, manifest_sha256=verified_manifest_sha256)
        if evidence_class == REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS:
            return _derive_source_measurement(
                document,
                manifest_sha256=verified_manifest_sha256,
                species="D",
                auxiliary_payloads=auxiliary_payloads,
                typed_payload_sha256s=typed_payload_sha256s,
            )
        if evidence_class == REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS:
            return _derive_source_measurement(
                document,
                manifest_sha256=verified_manifest_sha256,
                species="T",
                auxiliary_payloads=auxiliary_payloads,
                typed_payload_sha256s=typed_payload_sha256s,
            )
        if evidence_class == BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS:
            return _derive_burn_retention(
                document,
                manifest_sha256=verified_manifest_sha256,
                auxiliary_payloads=auxiliary_payloads,
                typed_payload_sha256s=typed_payload_sha256s,
            )
        return _derive_wall_plug_ledger(
            document,
            manifest_sha256=verified_manifest_sha256,
            auxiliary_payloads=auxiliary_payloads,
            typed_payload_sha256s=typed_payload_sha256s,
        )
    except ArtifactPayloadError:
        raise
    except (ArithmeticError, MemoryError, np.linalg.LinAlgError) as exc:
        raise ArtifactPayloadError("numeric payload derivation failed closed") from exc


__all__ = [
    "ArtifactPayloadError",
    "BASELINE_CROSS_SECTION_MODEL",
    "BURN_WEIGHTED_DT_RETENTION_EVIDENCE_CLASS",
    "CHANNEL_MAJOR_ENERGY_MINOR_FLATTEN_ORDER",
    "CHANNEL_NORMALIZATION_ABS_TOLERANCE",
    "COVARIANCE_FACTOR_REPRESENTATION",
    "COVARIANCE_UNCERTAINTY_LEVEL",
    "COVARIANCE_UNCERTAINTY_SCOPE",
    "DERIVED_CLAIMS_SCHEMA_VERSION",
    "DerivedArtifactClaims",
    "BURN_RETENTION_UNCERTAINTY_VARIABLES",
    "ENGINEERING_AUXILIARY_PAYLOAD_ROLES",
    "ENGINEERING_PRIMARY_PAYLOAD_ROLES",
    "KNOWN_EVIDENCE_CLASSES",
    "LONGITUDINAL_CHANNEL_ORDER",
    "LONGITUDINAL_QUANTIZATION_AXIS",
    "MAX_COVARIANCE_FACTOR_VALUES",
    "MAX_LOCAL_ADJACENT_RELATIVE_CHANGE",
    "MAX_LOCAL_SECOND_DIFFERENCE_RELATIVE",
    "MAX_OPERATOR_GRID_POINTS",
    "MAX_OPERATOR_LOG_ENERGY_GAP",
    "MAX_SCALAR_OPERATOR_GRID_POINTS",
    "MAXWELLIAN_QUADRATURE_RELATIVE_TOLERANCE",
    "MAXWELLIAN_COARSENING_STRIDES",
    "MIN_REQUIRED_REACTIVITY_RATIO",
    "MIN_NUMERICALLY_MEANINGFUL_RELATIVE_UNCERTAINTY",
    "MIN_OPERATOR_GRID_POINTS",
    "PAYLOAD_SCHEMA_VERSION",
    "PYTHON_CANONICAL_JSON_CONVENTION",
    "POLARIZED_DT_STATE_RESOLVED_OPERATOR_EVIDENCE_CLASS",
    "POLARIZED_DT_WALL_PLUG_LEDGER_EVIDENCE_CLASS",
    "REACTOR_RATE_POLARIZED_DEUTERIUM_SOURCE_EVIDENCE_CLASS",
    "REACTOR_RATE_POLARIZED_TRITIUM_SOURCE_EVIDENCE_CLASS",
    "REACTION_ENERGY_FRAME",
    "REACTION_OBSERVABLE",
    "SCALAR_TARGET_RATIO_CONTROL_VARIANT",
    "SIX_CHANNEL_LONGITUDINAL_OPERATOR_VARIANT",
    "SOURCE_RATE_UNCERTAINTY_VARIABLES",
    "INITIAL_STATE_BASIS",
    "WALL_PLUG_COMPONENT_CATEGORIES",
    "WALL_PLUG_UNCERTAINTY_VARIABLES",
    "derive_artifact_claims",
]
