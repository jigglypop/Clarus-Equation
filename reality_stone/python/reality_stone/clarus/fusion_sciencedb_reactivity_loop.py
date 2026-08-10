"""10 keV Maxwellian audit for the pinned ScienceDB D--T cross-section table.

The Han et al. ScienceDB V1 payload contains an unpolarized
``T(d,n)4He-CS.txt`` table with one scalar ``ERR`` column per energy.  This
module reads that table only after the complete six-file payload has passed
the byte-level integrity audit in :mod:`fusion_sciencedb_payload_loop`.

Four deliberately explicit interpolation controls are integrated:

* direct cross section, log(sigma) against log(E);
* direct cross section, linear sigma against linear E;
* Gamow-removed S factor, log(S) against linear E;
* Gamow-removed S factor, linear S against linear E.

For each control, shifting every tabulated point coherently to ``CS-ERR`` or
``CS+ERR`` gives a fully correlated endpoint envelope.  It is not a
covariance-derived confidence interval.  The public payload has neither a
numeric covariance matrix nor an initial-state spin operator, so this audit
can close only an unpolarized baseline sensitivity calculation.  It cannot
certify a sub-one-percent baseline or a polarized/CE reaction branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .fusion_equation_iteration_loop import bosch_hale_dt_cross_section_m2
from .fusion_full_loop import (
    BOSCH_HALE_DT_TEMPERATURE_MAX_KEV,
    BOSCH_HALE_DT_TEMPERATURE_MIN_KEV,
    bosch_hale_dt_reactivity,
)
from .fusion_resonance_loop import DEUTERON_MASS_MEV, TRITON_MASS_MEV
from .fusion_sciencedb_payload_loop import (
    HAN_RMATRIX_DOI,
    HAN_SCIENCEDB_DOI,
    SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY,
    SCIENCEDB_V1_FILE_SPECS,
    SCIENCEDB_VERSION,
    ScienceDBV1PayloadAudit,
    _path_is_link_or_reparse_point,
    audit_sciencedb_v1_payload,
)


DT_CROSS_SECTION_FILENAME = "T(d,n)4He-CS.txt"
DT_CROSS_SECTION_COLUMNS = ("Ed(MeV)", "CS(mb)", "ERR(mb)")
DT_CROSS_SECTION_EXPECTED_ROWS = 54
DT_CROSS_SECTION_EXPECTED_SHA256 = (
    "3cdc7242f7e135832865057bb522f986d839b10afee40e995b36fcd16f9f996e"
)

DEFAULT_TEMPERATURE_KEV = 10.0
DEFAULT_ENERGY_GRID_POINTS = 4_001
INTEGRATION_MIN_ENERGY_KEV = 0.5
INTEGRATION_MAX_ENERGY_KEV = 550.0
DT_GAMOW_B_SQRT_KEV = 34.3827
MILLIBARN_TO_M2 = 1.0e-31
SPEED_OF_LIGHT_M_S = 299_792_458.0
ONE_PERCENT = 0.01
MAX_GRID_REFINEMENT_RELATIVE_RESIDUAL = 1.0e-5

SIGMA_LOG_LOG = "sigma_log_log"
SIGMA_LINEAR = "sigma_linear"
S_FACTOR_LOG_LINEAR = "s_factor_log_linear"
S_FACTOR_LINEAR = "s_factor_linear"


class ScienceDBReactivityIntegrityError(RuntimeError):
    """Raised before table parsing when the pinned payload is not exact."""


@dataclass(frozen=True)
class DTTable:
    """Strictly parsed unpolarized D--T cross-section rows."""

    deuteron_lab_energy_mev: np.ndarray
    cross_section_millibarn: np.ndarray
    pointwise_err_millibarn: np.ndarray


@dataclass(frozen=True)
class ReactivityEnvelope:
    """One interpolation control and its coherent pointwise-error envelope."""

    method: str
    interpolation_statement: str
    central_reactivity_cm3_s: float
    all_points_minus_err_reactivity_cm3_s: float
    all_points_plus_err_reactivity_cm3_s: float
    central_to_bosch_hale_closed_ratio: float
    central_to_bosch_hale_same_kernel_ratio: float
    lower_to_central_minus_one: float
    upper_to_central_minus_one: float
    pointwise_err_shift_is_fully_correlated_endpoint_control: bool
    covariance_confidence_interval_derived: bool


@dataclass(frozen=True)
class ScienceDBDTReactivityAudit:
    """Fail-closed result for the pinned unpolarized 10 keV calculation."""

    schema_version: str
    source_paper_doi: str
    source_dataset_doi: str
    source_dataset_version: str
    payload_audit: ScienceDBV1PayloadAudit
    dt_cross_section_filename: str
    dt_cross_section_expected_sha256: str
    dt_cross_section_runtime_sha256: str
    payload_integrity_verified_before_dt_parse: bool
    dt_table_parsed_from_integrity_verified_raw_bytes: bool
    dt_table_row_count: int
    temperature_kev: float
    energy_grid_points: int
    refined_energy_grid_points: int
    integration_min_energy_kev: float
    integration_max_energy_kev: float
    deuteron_mass_mev: float
    triton_mass_mev: float
    deuteron_lab_to_cm_energy_factor: float
    table_lab_energy_min_mev: float
    table_lab_energy_max_mev: float
    table_cm_energy_min_kev: float
    table_cm_energy_max_kev: float
    integration_grid_inside_table_domain: bool
    bosch_hale_closed_reactivity_cm3_s: float
    bosch_hale_same_kernel_reactivity_cm3_s: float
    bosch_hale_same_kernel_to_closed_ratio: float
    sigma_log_log: ReactivityEnvelope
    sigma_linear: ReactivityEnvelope
    s_factor_log_linear: ReactivityEnvelope
    s_factor_linear: ReactivityEnvelope
    sigma_interpolation_relative_spread: float
    s_factor_interpolation_relative_spread: float
    all_method_central_relative_spread: float
    grid_refinement_max_relative_residual: float
    grid_refinement_tolerance: float
    grid_refinement_gate_pass: bool
    conservative_method_and_err_lower_cm3_s: float
    conservative_method_and_err_upper_cm3_s: float
    interpolation_spread_below_one_percent: bool
    only_pointwise_scalar_err_available: bool
    numeric_covariance_matrix_available: bool
    initial_state_spin_operator_available: bool
    unpolarized_sub_one_percent_certification_gate_pass: bool
    physical_state_resolved_one_percent_branch_gate_pass: bool
    maximum_supported_stage: str
    status: str

    @property
    def interpolation_envelopes(self) -> tuple[ReactivityEnvelope, ...]:
        """Return the four registered interpolation controls in fixed order."""

        return (
            self.sigma_log_log,
            self.sigma_linear,
            self.s_factor_log_linear,
            self.s_factor_linear,
        )


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "reality_stone").is_dir():
            return parent
    return None


def _validated_temperature(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("temperature_kev must be a real number")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_kev must be positive and finite")
    if not (BOSCH_HALE_DT_TEMPERATURE_MIN_KEV <= temperature <= BOSCH_HALE_DT_TEMPERATURE_MAX_KEV):
        raise ValueError("temperature_kev lies outside the Bosch-Hale 0.2--100 keV fit range")
    return temperature


def _validated_grid_points(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("energy_grid_points must be an integer")
    points = int(value)
    if points < 101:
        raise ValueError("energy_grid_points must be at least 101")
    return points


def _safe_relative_parts(value: str) -> tuple[str, ...] | None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        return None
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if (
        windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or posix_path.as_posix() != value
    ):
        return None
    if not posix_path.parts or any(
        part in {"", ".", ".."} or ":" in part for part in posix_path.parts
    ):
        return None
    return tuple(posix_path.parts)


def _dt_file_spec_sha256() -> str:
    matches = [
        spec.expected_sha256
        for spec in SCIENCEDB_V1_FILE_SPECS
        if spec.filename == DT_CROSS_SECTION_FILENAME
    ]
    if matches != [DT_CROSS_SECTION_EXPECTED_SHA256]:
        raise RuntimeError("D-T cross-section manifest contract is inconsistent")
    return matches[0]


def _integrity_verified_dt_bytes(
    *,
    repository_root: Path,
    repository_relative_directory: str,
    payload_audit: ScienceDBV1PayloadAudit,
) -> tuple[bytes, str]:
    """Read D-T bytes only after the aggregate payload integrity gate passes."""

    if not payload_audit.payload_integrity_gate_pass:
        raise ScienceDBReactivityIntegrityError(
            "ScienceDB V1 payload integrity failed before D-T table parsing"
        )
    parts = _safe_relative_parts(repository_relative_directory)
    if parts is None:
        raise ScienceDBReactivityIntegrityError("unsafe ScienceDB payload directory")
    target = repository_root.joinpath(*parts, DT_CROSS_SECTION_FILENAME)
    try:
        resolved_root = repository_root.resolve(strict=True)
        resolved_target = target.resolve(strict=True)
        resolved_target.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ScienceDBReactivityIntegrityError("unsafe D-T table path") from exc
    if _path_is_link_or_reparse_point(target) or not resolved_target.is_file():
        raise ScienceDBReactivityIntegrityError("D-T table must be a regular non-symlink file")
    try:
        raw_bytes = resolved_target.read_bytes()
    except OSError as exc:
        raise ScienceDBReactivityIntegrityError("D-T table could not be read") from exc
    runtime_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if not hmac.compare_digest(runtime_sha256, _dt_file_spec_sha256()):
        raise ScienceDBReactivityIntegrityError(
            "D-T table changed after aggregate payload integrity verification"
        )
    return raw_bytes, runtime_sha256


def _parse_dt_table(raw_bytes: bytes) -> DTTable:
    """Parse the exact three-column table; callers must verify integrity first."""

    try:
        text = raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ScienceDBReactivityIntegrityError("D-T table is not ASCII") from exc
    lines = text.splitlines()
    if not lines or tuple(lines[0].split()) != DT_CROSS_SECTION_COLUMNS:
        raise ScienceDBReactivityIntegrityError("D-T table header contract failed")
    rows: list[tuple[float, float, float]] = []
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != 3:
            raise ScienceDBReactivityIntegrityError("D-T table row width contract failed")
        try:
            row = tuple(float(token) for token in tokens)
        except ValueError as exc:
            raise ScienceDBReactivityIntegrityError("D-T table contains a nonnumeric row") from exc
        if len(row) != 3 or not all(math.isfinite(value) for value in row):
            raise ScienceDBReactivityIntegrityError("D-T table contains a nonfinite row")
        rows.append((row[0], row[1], row[2]))
    if len(rows) != DT_CROSS_SECTION_EXPECTED_ROWS:
        raise ScienceDBReactivityIntegrityError("D-T table row-count contract failed")

    array = np.asarray(rows, dtype=float)
    energies = array[:, 0]
    cross_sections = array[:, 1]
    errors = array[:, 2]
    if np.any(np.diff(energies) <= 0.0) or np.any(energies <= 0.0):
        raise ScienceDBReactivityIntegrityError("D-T lab energies must be strictly increasing")
    if np.any(cross_sections <= 0.0) or np.any(errors < 0.0):
        raise ScienceDBReactivityIntegrityError("D-T CS must be positive and ERR nonnegative")
    if np.any(cross_sections - errors <= 0.0):
        raise ScienceDBReactivityIntegrityError("CS-ERR must stay positive for log controls")
    return DTTable(
        deuteron_lab_energy_mev=energies,
        cross_section_millibarn=cross_sections,
        pointwise_err_millibarn=errors,
    )


def _interpolate_cross_section(
    *,
    method: str,
    table_energies_kev: np.ndarray,
    table_cross_sections_m2: np.ndarray,
    grid_energies_kev: np.ndarray,
) -> np.ndarray:
    if np.any(table_cross_sections_m2 <= 0.0):
        raise ValueError("interpolation inputs must be positive")
    if method == SIGMA_LOG_LOG:
        return np.exp(
            np.interp(
                np.log(grid_energies_kev),
                np.log(table_energies_kev),
                np.log(table_cross_sections_m2),
            )
        )
    if method == SIGMA_LINEAR:
        return np.interp(
            grid_energies_kev,
            table_energies_kev,
            table_cross_sections_m2,
        )

    s_factor = (
        table_cross_sections_m2
        * table_energies_kev
        * np.exp(DT_GAMOW_B_SQRT_KEV / np.sqrt(table_energies_kev))
    )
    if method == S_FACTOR_LOG_LINEAR:
        interpolated_s_factor = np.exp(
            np.interp(grid_energies_kev, table_energies_kev, np.log(s_factor))
        )
    elif method == S_FACTOR_LINEAR:
        interpolated_s_factor = np.interp(
            grid_energies_kev,
            table_energies_kev,
            s_factor,
        )
    else:
        raise ValueError(f"unknown interpolation method: {method}")
    return (
        interpolated_s_factor
        / grid_energies_kev
        * np.exp(-DT_GAMOW_B_SQRT_KEV / np.sqrt(grid_energies_kev))
    )


def _maxwellian_reactivity_cm3_s(
    cross_sections_m2: np.ndarray,
    energies_kev: np.ndarray,
    *,
    temperature_kev: float,
) -> float:
    weights = cross_sections_m2 * energies_kev * np.exp(-energies_kev / temperature_kev)
    integral = float(np.trapezoid(weights, energies_kev))
    reduced_mass_kev = (
        1.0e3 * DEUTERON_MASS_MEV * TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    )
    reactivity_m3_s = (
        math.sqrt(8.0 / (math.pi * reduced_mass_kev))
        * integral
        / temperature_kev**1.5
        * SPEED_OF_LIGHT_M_S
    )
    result = reactivity_m3_s * 1.0e6
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("Maxwellian reactivity integration failed")
    return result


def _interpolation_statement(method: str) -> str:
    statements = {
        SIGMA_LOG_LOG: "linear interpolation of log(sigma) against log(E_cm)",
        SIGMA_LINEAR: "linear interpolation of sigma against E_cm",
        S_FACTOR_LOG_LINEAR: "linear interpolation of log(S) against E_cm",
        S_FACTOR_LINEAR: "linear interpolation of S against E_cm",
    }
    return statements[method]


def _reactivity_envelope(
    *,
    method: str,
    table_energies_kev: np.ndarray,
    central_cross_sections_m2: np.ndarray,
    lower_cross_sections_m2: np.ndarray,
    upper_cross_sections_m2: np.ndarray,
    grid_energies_kev: np.ndarray,
    temperature_kev: float,
    bosch_hale_closed: float,
    bosch_hale_same_kernel: float,
) -> ReactivityEnvelope:
    values = []
    for table_values in (
        central_cross_sections_m2,
        lower_cross_sections_m2,
        upper_cross_sections_m2,
    ):
        interpolated = _interpolate_cross_section(
            method=method,
            table_energies_kev=table_energies_kev,
            table_cross_sections_m2=table_values,
            grid_energies_kev=grid_energies_kev,
        )
        values.append(
            _maxwellian_reactivity_cm3_s(
                interpolated,
                grid_energies_kev,
                temperature_kev=temperature_kev,
            )
        )
    central, lower, upper = values
    if not lower < central < upper:
        raise RuntimeError("coherent pointwise ERR endpoints do not bracket the central value")
    return ReactivityEnvelope(
        method=method,
        interpolation_statement=_interpolation_statement(method),
        central_reactivity_cm3_s=central,
        all_points_minus_err_reactivity_cm3_s=lower,
        all_points_plus_err_reactivity_cm3_s=upper,
        central_to_bosch_hale_closed_ratio=central / bosch_hale_closed,
        central_to_bosch_hale_same_kernel_ratio=central / bosch_hale_same_kernel,
        lower_to_central_minus_one=lower / central - 1.0,
        upper_to_central_minus_one=upper / central - 1.0,
        pointwise_err_shift_is_fully_correlated_endpoint_control=True,
        covariance_confidence_interval_derived=False,
    )


def audit_sciencedb_dt_reactivity(
    *,
    repository_root: str | Path | None = None,
    repository_relative_directory: str = SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY,
    temperature_kev: Real = DEFAULT_TEMPERATURE_KEV,
    energy_grid_points: Integral = DEFAULT_ENERGY_GRID_POINTS,
) -> ScienceDBDTReactivityAudit:
    """Integrate the exact local D-T table after full payload verification.

    An integrity failure raises :class:`ScienceDBReactivityIntegrityError`
    before ``_parse_dt_table`` is called.  Valid inputs return a numerical
    unpolarized audit whose physical one-percent gate remains fail-closed.
    """

    temperature = _validated_temperature(temperature_kev)
    points = _validated_grid_points(energy_grid_points)
    requested_root = Path(repository_root) if repository_root is not None else _repository_root()
    if requested_root is None:
        raise ScienceDBReactivityIntegrityError("repository root could not be located")
    try:
        root = requested_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ScienceDBReactivityIntegrityError("repository root is unavailable") from exc

    payload_audit = audit_sciencedb_v1_payload(
        repository_root=root,
        repository_relative_directory=repository_relative_directory,
    )
    raw_bytes, runtime_sha256 = _integrity_verified_dt_bytes(
        repository_root=root,
        repository_relative_directory=repository_relative_directory,
        payload_audit=payload_audit,
    )
    table = _parse_dt_table(raw_bytes)

    lab_to_cm = TRITON_MASS_MEV / (DEUTERON_MASS_MEV + TRITON_MASS_MEV)
    table_energies_kev = 1.0e3 * lab_to_cm * table.deuteron_lab_energy_mev
    grid_energies_kev = np.geomspace(
        INTEGRATION_MIN_ENERGY_KEV,
        INTEGRATION_MAX_ENERGY_KEV,
        points,
    )
    grid_inside_domain = bool(
        grid_energies_kev[0] >= table_energies_kev[0]
        and grid_energies_kev[-1] <= table_energies_kev[-1]
    )
    if not grid_inside_domain:
        raise ValueError("Maxwellian integration grid would extrapolate beyond the D-T table")

    central_m2 = table.cross_section_millibarn * MILLIBARN_TO_M2
    lower_m2 = (table.cross_section_millibarn - table.pointwise_err_millibarn) * MILLIBARN_TO_M2
    upper_m2 = (table.cross_section_millibarn + table.pointwise_err_millibarn) * MILLIBARN_TO_M2

    _, _, bosch_hale_closed = bosch_hale_dt_reactivity(temperature)
    bosch_hale_cross_sections = np.asarray(
        [bosch_hale_dt_cross_section_m2(float(energy)) for energy in grid_energies_kev]
    )
    bosch_hale_same_kernel = _maxwellian_reactivity_cm3_s(
        bosch_hale_cross_sections,
        grid_energies_kev,
        temperature_kev=temperature,
    )

    envelope_kwargs = {
        "table_energies_kev": table_energies_kev,
        "central_cross_sections_m2": central_m2,
        "lower_cross_sections_m2": lower_m2,
        "upper_cross_sections_m2": upper_m2,
        "grid_energies_kev": grid_energies_kev,
        "temperature_kev": temperature,
        "bosch_hale_closed": bosch_hale_closed,
        "bosch_hale_same_kernel": bosch_hale_same_kernel,
    }
    sigma_log_log = _reactivity_envelope(method=SIGMA_LOG_LOG, **envelope_kwargs)
    sigma_linear = _reactivity_envelope(method=SIGMA_LINEAR, **envelope_kwargs)
    s_factor_log_linear = _reactivity_envelope(
        method=S_FACTOR_LOG_LINEAR,
        **envelope_kwargs,
    )
    s_factor_linear = _reactivity_envelope(method=S_FACTOR_LINEAR, **envelope_kwargs)
    envelopes = (
        sigma_log_log,
        sigma_linear,
        s_factor_log_linear,
        s_factor_linear,
    )
    central_values = [item.central_reactivity_cm3_s for item in envelopes]
    all_method_spread = max(central_values) / min(central_values) - 1.0
    interpolation_below_one_percent = all_method_spread < ONE_PERCENT
    refined_points = 2 * points - 1
    refined_grid_energies_kev = np.geomspace(
        INTEGRATION_MIN_ENERGY_KEV,
        INTEGRATION_MAX_ENERGY_KEV,
        refined_points,
    )
    refined_central_values = []
    for method in (SIGMA_LOG_LOG, SIGMA_LINEAR, S_FACTOR_LOG_LINEAR, S_FACTOR_LINEAR):
        refined_cross_sections = _interpolate_cross_section(
            method=method,
            table_energies_kev=table_energies_kev,
            table_cross_sections_m2=central_m2,
            grid_energies_kev=refined_grid_energies_kev,
        )
        refined_central_values.append(
            _maxwellian_reactivity_cm3_s(
                refined_cross_sections,
                refined_grid_energies_kev,
                temperature_kev=temperature,
            )
        )
    grid_refinement_residual = max(
        abs(coarse - refined) / refined
        for coarse, refined in zip(central_values, refined_central_values, strict=True)
    )
    grid_refinement_pass = (
        grid_refinement_residual < MAX_GRID_REFINEMENT_RELATIVE_RESIDUAL
    )

    numeric_covariance_available = (
        payload_audit.numeric_covariance_matrix_or_correlation_payload_available
    )
    initial_state_spin_available = payload_audit.initial_state_spin_columns_or_operator_available
    unpolarized_precision_gate = (
        payload_audit.payload_integrity_gate_pass
        and grid_inside_domain
        and grid_refinement_pass
        and interpolation_below_one_percent
        and numeric_covariance_available
    )
    physical_gate = unpolarized_precision_gate and initial_state_spin_available

    return ScienceDBDTReactivityAudit(
        schema_version="fusion-sciencedb-dt-reactivity-v1",
        source_paper_doi=HAN_RMATRIX_DOI,
        source_dataset_doi=HAN_SCIENCEDB_DOI,
        source_dataset_version=SCIENCEDB_VERSION,
        payload_audit=payload_audit,
        dt_cross_section_filename=DT_CROSS_SECTION_FILENAME,
        dt_cross_section_expected_sha256=DT_CROSS_SECTION_EXPECTED_SHA256,
        dt_cross_section_runtime_sha256=runtime_sha256,
        payload_integrity_verified_before_dt_parse=True,
        dt_table_parsed_from_integrity_verified_raw_bytes=True,
        dt_table_row_count=len(table.deuteron_lab_energy_mev),
        temperature_kev=temperature,
        energy_grid_points=points,
        refined_energy_grid_points=refined_points,
        integration_min_energy_kev=INTEGRATION_MIN_ENERGY_KEV,
        integration_max_energy_kev=INTEGRATION_MAX_ENERGY_KEV,
        deuteron_mass_mev=DEUTERON_MASS_MEV,
        triton_mass_mev=TRITON_MASS_MEV,
        deuteron_lab_to_cm_energy_factor=lab_to_cm,
        table_lab_energy_min_mev=float(table.deuteron_lab_energy_mev[0]),
        table_lab_energy_max_mev=float(table.deuteron_lab_energy_mev[-1]),
        table_cm_energy_min_kev=float(table_energies_kev[0]),
        table_cm_energy_max_kev=float(table_energies_kev[-1]),
        integration_grid_inside_table_domain=grid_inside_domain,
        bosch_hale_closed_reactivity_cm3_s=bosch_hale_closed,
        bosch_hale_same_kernel_reactivity_cm3_s=bosch_hale_same_kernel,
        bosch_hale_same_kernel_to_closed_ratio=bosch_hale_same_kernel / bosch_hale_closed,
        sigma_log_log=sigma_log_log,
        sigma_linear=sigma_linear,
        s_factor_log_linear=s_factor_log_linear,
        s_factor_linear=s_factor_linear,
        sigma_interpolation_relative_spread=(
            sigma_linear.central_reactivity_cm3_s / sigma_log_log.central_reactivity_cm3_s - 1.0
        ),
        s_factor_interpolation_relative_spread=(
            s_factor_linear.central_reactivity_cm3_s / s_factor_log_linear.central_reactivity_cm3_s
            - 1.0
        ),
        all_method_central_relative_spread=all_method_spread,
        grid_refinement_max_relative_residual=grid_refinement_residual,
        grid_refinement_tolerance=MAX_GRID_REFINEMENT_RELATIVE_RESIDUAL,
        grid_refinement_gate_pass=grid_refinement_pass,
        conservative_method_and_err_lower_cm3_s=min(
            item.all_points_minus_err_reactivity_cm3_s for item in envelopes
        ),
        conservative_method_and_err_upper_cm3_s=max(
            item.all_points_plus_err_reactivity_cm3_s for item in envelopes
        ),
        interpolation_spread_below_one_percent=interpolation_below_one_percent,
        only_pointwise_scalar_err_available=payload_audit.pointwise_scalar_err_columns_only,
        numeric_covariance_matrix_available=numeric_covariance_available,
        initial_state_spin_operator_available=initial_state_spin_available,
        unpolarized_sub_one_percent_certification_gate_pass=unpolarized_precision_gate,
        physical_state_resolved_one_percent_branch_gate_pass=physical_gate,
        maximum_supported_stage="integrity-pinned unpolarized Maxwellian sensitivity control",
        status=(
            "unpolarized 10 keV table integral reproduced; sub-one-percent and state-resolved "
            "claims fail closed on interpolation spread, absent covariance, and absent spin operator"
        ),
    )


def current_sciencedb_dt_reactivity_audit() -> ScienceDBDTReactivityAudit:
    """Return the canonical repository audit at 10 keV on 4,001 grid points."""

    return audit_sciencedb_dt_reactivity()


__all__ = [
    "DEFAULT_ENERGY_GRID_POINTS",
    "DEFAULT_TEMPERATURE_KEV",
    "DT_CROSS_SECTION_EXPECTED_ROWS",
    "DT_CROSS_SECTION_EXPECTED_SHA256",
    "DT_CROSS_SECTION_FILENAME",
    "DT_GAMOW_B_SQRT_KEV",
    "INTEGRATION_MAX_ENERGY_KEV",
    "INTEGRATION_MIN_ENERGY_KEV",
    "MAX_GRID_REFINEMENT_RELATIVE_RESIDUAL",
    "ReactivityEnvelope",
    "S_FACTOR_LINEAR",
    "S_FACTOR_LOG_LINEAR",
    "SIGMA_LINEAR",
    "SIGMA_LOG_LOG",
    "ScienceDBDTReactivityAudit",
    "ScienceDBReactivityIntegrityError",
    "audit_sciencedb_dt_reactivity",
    "current_sciencedb_dt_reactivity_audit",
]
