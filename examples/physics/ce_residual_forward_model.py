"""Minimal CE residual cosmology forward model.

This is the next step after ``cosmology_ratio_audit``: use the CE density
ratios as present-day boundary data, then compute background expansion,
distances, and linear growth in a conservative w0-wa/GR limit.

It is intentionally not a particle dark-matter or detector model.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.physics.cosmology import interp_linear, linspace, logspace, simpson  # noqa: E402
from examples.physics.cosmology_ratio_audit import CE_RATIOS  # noqa: E402


C_KM_S = 299792.458
TCMB_REFERENCE_K = 2.7255
OMEGA_GAMMA_H2_REFERENCE = 2.469e-5
NEUTRINO_RADIATION_FACTOR = 0.22710731766
DEFAULT_N_EFF = 3.044
EARLY_RD_INTEGRATION_POINTS = 10001
MPC_METERS = 3.0856775814913673e22
SPEED_OF_LIGHT_M_S = 299792458.0
NEWTON_G_SI = 6.67430e-11
HYDROGEN_ATOM_MASS_KG = 1.6735575e-27
THOMSON_CROSS_SECTION_M2 = 6.6524587321e-29
RECOMBINATION_REDSHIFT_UNIT = "dimensionless_redshift_z"
RECOMBINATION_XE_CONVENTION = "electrons_per_hydrogen_nucleus"
SUPPORTED_RECOMBINATION_SOLVERS = frozenset({"CLASS", "CAMB", "HyRec"})


@dataclass(frozen=True)
class CEForwardParams:
    omega_b0: float = CE_RATIOS["omega_b"]
    omega_dm0: float = CE_RATIOS["omega_c"]
    omega_lambda0: float = CE_RATIOS["omega_lambda"]
    h0: float = 67.4
    rd_mpc: float = 147.09
    rd_mode: str = "external"
    tcmb_k: float = TCMB_REFERENCE_K
    n_eff: float = DEFAULT_N_EFF
    sigma8_0: float = 0.811
    w0: float = -1.0
    wa: float = 0.0
    gravity_mu_coupling: float = 0.0

    def __post_init__(self) -> None:
        if self.rd_mode not in {"external", "early-universe"}:
            raise ValueError("rd_mode must be 'external' or 'early-universe'")
        if self.h0 <= 0.0:
            raise ValueError("h0 must be positive")
        if self.rd_mpc <= 0.0:
            raise ValueError("rd_mpc must be positive")
        if self.tcmb_k <= 0.0:
            raise ValueError("tcmb_k must be positive")
        if self.n_eff < 0.0:
            raise ValueError("n_eff must be non-negative")

    @property
    def omega_m0(self) -> float:
        return self.omega_b0 + self.omega_dm0

    @property
    def h(self) -> float:
        return self.h0 / 100.0

    @property
    def omega_b_h2(self) -> float:
        return self.omega_b0 * self.h * self.h

    @property
    def omega_m_h2(self) -> float:
        return self.omega_m0 * self.h * self.h

    @property
    def density_norm(self) -> float:
        return self.omega_m0 + self.omega_lambda0

    @property
    def omega_m0_background(self) -> float:
        return self.omega_m0 / self.density_norm

    @property
    def omega_lambda0_background(self) -> float:
        return self.omega_lambda0 / self.density_norm

    @property
    def is_flat(self) -> bool:
        return abs(self.omega_m0 + self.omega_lambda0 - 1.0) < 1.0e-3


@dataclass(frozen=True)
class ForwardCoverage:
    has_density_ratios: bool = True
    has_background_expansion_model: bool = True
    has_growth_model_for_s8: bool = True
    has_particle_dark_matter_model: bool = False
    has_detector_likelihood: bool = False

    @property
    def summary(self) -> str:
        return (
            "background and growth forward model implemented; "
            "particle dark matter and detector likelihood still open"
        )


@dataclass(frozen=True)
class EarlyUniverseSoundHorizon:
    omega_b_h2: float
    omega_m_h2: float
    omega_gamma_h2: float
    omega_r_h2: float
    omega_gamma0: float
    omega_r0: float
    z_drag: float
    a_drag: float
    sound_speed_drag_km_s: float
    rd_mpc: float
    integration_points: int
    status: str


@dataclass(frozen=True)
class SoundHorizonSelection:
    mode: str
    rd_mpc: float
    role: str
    source: str
    note: str
    early_universe: EarlyUniverseSoundHorizon | None


@dataclass(frozen=True)
class RecombinationHistoryMetadata:
    """Provenance and cosmology that produced an external x_e(z) table."""

    solver_family: str
    solver_version: str
    recombination_backend: str
    source_label: str
    redshift_unit: str
    electron_fraction_convention: str
    helium_mass_fraction_y_p: float
    h0_km_s_mpc: float
    omega_b_h2: float
    omega_m_h2: float
    tcmb_k: float
    n_eff: float

    def __post_init__(self) -> None:
        if self.solver_family not in SUPPORTED_RECOMBINATION_SOLVERS:
            raise ValueError(
                "solver_family must be one of CLASS, CAMB, or HyRec"
            )
        for name, value in (
            ("solver_version", self.solver_version),
            ("recombination_backend", self.recombination_backend),
            ("source_label", self.source_label),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.redshift_unit != RECOMBINATION_REDSHIFT_UNIT:
            raise ValueError(
                f"redshift_unit must be {RECOMBINATION_REDSHIFT_UNIT!r}"
            )
        if self.electron_fraction_convention != RECOMBINATION_XE_CONVENTION:
            raise ValueError(
                "electron_fraction_convention must explicitly be "
                f"{RECOMBINATION_XE_CONVENTION!r}"
            )
        if not 0.0 <= self.helium_mass_fraction_y_p < 1.0:
            raise ValueError("helium_mass_fraction_y_p must lie in [0, 1)")
        for name, value in (
            ("h0_km_s_mpc", self.h0_km_s_mpc),
            ("omega_b_h2", self.omega_b_h2),
            ("omega_m_h2", self.omega_m_h2),
            ("tcmb_k", self.tcmb_k),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.n_eff) or self.n_eff < 0.0:
            raise ValueError("n_eff must be finite and non-negative")


@dataclass(frozen=True)
class RecombinationHistory:
    """Validated external recombination history in canonical ascending-z order."""

    redshift: tuple[float, ...]
    electron_fraction: tuple[float, ...]
    metadata: RecombinationHistoryMetadata
    source_sha256: str
    original_grid_order: str
    redshift_column: int
    electron_fraction_column: int
    delimiter: str | None

    def __post_init__(self) -> None:
        if len(self.redshift) != len(self.electron_fraction):
            raise ValueError("redshift and electron_fraction lengths must match")
        if len(self.redshift) < 4:
            raise ValueError("recombination history must contain at least four rows")
        if self.original_grid_order not in {"ascending", "descending"}:
            raise ValueError("original_grid_order must be ascending or descending")
        if (
            self.redshift_column < 0
            or self.electron_fraction_column < 0
            or self.redshift_column == self.electron_fraction_column
        ):
            raise ValueError("history column indices must be distinct and non-negative")
        if self.delimiter == "":
            raise ValueError("history delimiter cannot be empty")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        if abs(self.redshift[0]) > 1.0e-12:
            raise ValueError("redshift grid must include z=0 as its first canonical row")
        for index, (z_value, x_e) in enumerate(
            zip(self.redshift, self.electron_fraction)
        ):
            if not math.isfinite(z_value) or z_value < 0.0:
                raise ValueError(f"redshift row {index} must be finite and non-negative")
            if not math.isfinite(x_e) or not 0.0 <= x_e <= 2.0:
                raise ValueError(
                    f"electron fraction row {index} must lie in [0, 2]"
                )
            if index and z_value <= self.redshift[index - 1]:
                raise ValueError("canonical redshift grid must be strictly increasing")


@dataclass(frozen=True)
class DragOpticalDepthBenchmark:
    """Benchmark result derived from a hashed external recombination history."""

    solver_family: str
    solver_version: str
    recombination_backend: str
    source_label: str
    source_sha256: str
    history_metadata: RecombinationHistoryMetadata
    input_redshift_column: int
    input_electron_fraction_column: int
    input_delimiter: str | None
    convention: str
    tau_drag_target: float
    z_drag: float
    a_drag: float
    rd_mpc: float
    rd_unit: str
    hydrogen_nuclei_today_m3: float
    tau_drag_at_z_max: float
    redshift_grid: tuple[float, ...]
    tau_drag_grid: tuple[float, ...]
    crossing_bracket: tuple[float, float]
    crossing_bracket_width: float
    rd_integration_points: int
    integration_method: str
    status: str


def sha256_hexdigest(payload: bytes) -> str:
    """Return the SHA-256 digest of the exact bytes supplied by a solver export."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes so the source hash is unambiguous")
    return hashlib.sha256(payload).hexdigest()


def recombination_history_file_sha256(path: str | Path) -> str:
    """Hash a solver export before loading it through the checked adapter."""
    return sha256_hexdigest(Path(path).read_bytes())


def parse_recombination_history_bytes(
    payload: bytes,
    *,
    metadata: RecombinationHistoryMetadata,
    expected_sha256: str,
    redshift_column: int,
    electron_fraction_column: int,
    delimiter: str | None = None,
) -> RecombinationHistory:
    """Parse a hashed CLASS/CAMB/HyRec z,x_e table.

    Comment text beginning with ``#`` and blank lines are ignored. Column
    indices are zero-based. Whitespace splitting is used when ``delimiter`` is
    ``None``. The raw-byte SHA-256 must be supplied independently and match
    before any numeric row is accepted.
    """
    actual_sha256 = sha256_hexdigest(payload)
    normalized_expected = expected_sha256.lower()
    if len(normalized_expected) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized_expected
    ):
        raise ValueError("expected_sha256 must be a SHA-256 hex digest")
    if actual_sha256 != normalized_expected:
        raise ValueError(
            "recombination history SHA-256 mismatch: "
            f"expected {normalized_expected}, got {actual_sha256}"
        )
    if (
        not isinstance(redshift_column, int)
        or not isinstance(electron_fraction_column, int)
        or redshift_column < 0
        or electron_fraction_column < 0
    ):
        raise ValueError("column indices must be non-negative integers")
    if redshift_column == electron_fraction_column:
        raise ValueError("redshift and electron-fraction columns must differ")
    if delimiter == "":
        raise ValueError("delimiter cannot be empty")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("recombination history must be UTF-8 text") from exc

    rows: list[tuple[float, float]] = []
    required_column = max(redshift_column, electron_fraction_column)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        fields = content.split() if delimiter is None else content.split(delimiter)
        fields = [field.strip() for field in fields]
        if len(fields) <= required_column:
            raise ValueError(
                f"line {line_number} does not contain requested column "
                f"{required_column}"
            )
        try:
            z_value = float(fields[redshift_column])
            x_e = float(fields[electron_fraction_column])
        except ValueError as exc:
            raise ValueError(
                f"line {line_number} has a non-numeric z or x_e value"
            ) from exc
        rows.append((z_value, x_e))

    if len(rows) < 4:
        raise ValueError("recombination history must contain at least four data rows")
    ascending = all(rows[index][0] < rows[index + 1][0] for index in range(len(rows) - 1))
    descending = all(
        rows[index][0] > rows[index + 1][0] for index in range(len(rows) - 1)
    )
    if not ascending and not descending:
        raise ValueError("redshift grid must be strictly monotonic with no duplicates")
    original_grid_order = "ascending" if ascending else "descending"
    if descending:
        rows.reverse()

    return RecombinationHistory(
        redshift=tuple(row[0] for row in rows),
        electron_fraction=tuple(row[1] for row in rows),
        metadata=metadata,
        source_sha256=actual_sha256,
        original_grid_order=original_grid_order,
        redshift_column=redshift_column,
        electron_fraction_column=electron_fraction_column,
        delimiter=delimiter,
    )


def load_recombination_history_table(
    path: str | Path,
    *,
    metadata: RecombinationHistoryMetadata,
    expected_sha256: str,
    redshift_column: int,
    electron_fraction_column: int,
    delimiter: str | None = None,
) -> RecombinationHistory:
    """Load an immutable-by-hash external recombination table from disk."""
    payload = Path(path).read_bytes()
    return parse_recombination_history_bytes(
        payload,
        metadata=metadata,
        expected_sha256=expected_sha256,
        redshift_column=redshift_column,
        electron_fraction_column=electron_fraction_column,
        delimiter=delimiter,
    )


def photon_density_h2(tcmb_k: float) -> float:
    """Return omega_gamma = Omega_gamma h^2 from the blackbody temperature."""
    if tcmb_k <= 0.0:
        raise ValueError("tcmb_k must be positive")
    return OMEGA_GAMMA_H2_REFERENCE * (tcmb_k / TCMB_REFERENCE_K) ** 4


def radiation_density_h2(tcmb_k: float, n_eff: float) -> float:
    """Return omega_r including standard free-streaming relativistic neutrinos."""
    if n_eff < 0.0:
        raise ValueError("n_eff must be non-negative")
    omega_gamma_h2 = photon_density_h2(tcmb_k)
    return omega_gamma_h2 * (1.0 + NEUTRINO_RADIATION_FACTOR * n_eff)


def early_hubble_rate_s_inverse(z: float, params: CEForwardParams) -> float:
    """Return H(z) in s^-1 for the benchmark radiation+matter+Lambda background."""
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("redshift must be finite and non-negative")
    omega_r0 = radiation_density_h2(params.tcmb_k, params.n_eff) / (params.h**2)
    expansion_squared = (
        omega_r0 * (1.0 + z) ** 4
        + params.omega_m0 * (1.0 + z) ** 3
        + params.omega_lambda0
    )
    h0_s_inverse = params.h0 * 1000.0 / MPC_METERS
    return h0_s_inverse * math.sqrt(expansion_squared)


def hydrogen_nuclei_number_density_today_m3(
    params: CEForwardParams,
    helium_mass_fraction_y_p: float,
) -> float:
    """Return n_H,0 in m^-3 using the declared primordial helium mass fraction."""
    if not 0.0 <= helium_mass_fraction_y_p < 1.0:
        raise ValueError("helium_mass_fraction_y_p must lie in [0, 1)")
    h0_s_inverse = params.h0 * 1000.0 / MPC_METERS
    critical_density_kg_m3 = (
        3.0 * h0_s_inverse * h0_s_inverse / (8.0 * math.pi * NEWTON_G_SI)
    )
    baryon_density_kg_m3 = params.omega_b0 * critical_density_kg_m3
    return (
        (1.0 - helium_mass_fraction_y_p)
        * baryon_density_kg_m3
        / HYDROGEN_ATOM_MASS_KG
    )


def baryon_drag_inertia_ratio(z: float, params: CEForwardParams) -> float:
    """Return R=3 rho_b/(4 rho_gamma) for the declared CE/temperature inputs."""
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("redshift must be finite and non-negative")
    omega_gamma_h2 = photon_density_h2(params.tcmb_k)
    return 3.0 * params.omega_b_h2 / (4.0 * omega_gamma_h2 * (1.0 + z))


def drag_optical_depth_rate_per_redshift(
    z: float,
    electron_fraction: float,
    params: CEForwardParams,
    helium_mass_fraction_y_p: float,
) -> float:
    """Return d tau_drag/dz for x_e=n_e/n_H.

    The convention matches the standard baryon Euler drag rate:

        tau_drag(z) = integral_0^z [c sigma_T n_e(z')]
                      / [H(z') (1+z') R(z')] dz',
        R = 3 rho_b / (4 rho_gamma),  n_e=x_e n_H,0 (1+z)^3.

    The returned rate is dimensionless per unit redshift.
    """
    if not math.isfinite(electron_fraction) or not 0.0 <= electron_fraction <= 2.0:
        raise ValueError("electron_fraction must lie in [0, 2]")
    n_hydrogen_today = hydrogen_nuclei_number_density_today_m3(
        params,
        helium_mass_fraction_y_p,
    )
    n_electron_m3 = electron_fraction * n_hydrogen_today * (1.0 + z) ** 3
    return (
        SPEED_OF_LIGHT_M_S
        * THOMSON_CROSS_SECTION_M2
        * n_electron_m3
        / (
            early_hubble_rate_s_inverse(z, params)
            * (1.0 + z)
            * baryon_drag_inertia_ratio(z, params)
        )
    )


def eisenstein_hu_drag_redshift(omega_m_h2: float, omega_b_h2: float) -> float:
    """Return the Eisenstein-Hu baryon drag-redshift fitting formula.

    This is a standard-early-physics Selection/assumption imported into the CE
    boundary model. It is not derived internally from CE recombination or
    nuclear physics, and it must not be tuned against the DESI vector.
    """
    if omega_m_h2 <= 0.0 or omega_b_h2 <= 0.0:
        raise ValueError("physical matter and baryon densities must be positive")
    b1 = 0.313 * omega_m_h2 ** (-0.419) * (1.0 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    return (
        1291.0
        * omega_m_h2**0.251
        / (1.0 + 0.659 * omega_m_h2**0.828)
        * (1.0 + b1 * omega_b_h2**b2)
    )


def baryon_photon_sound_speed_km_s(
    a: float,
    omega_b_h2: float,
    omega_gamma_h2: float,
) -> float:
    """Return c/sqrt(3(1+R)) with R=3*rho_b/(4*rho_gamma)."""
    if a < 0.0:
        raise ValueError("scale factor must be non-negative")
    if omega_b_h2 < 0.0 or omega_gamma_h2 <= 0.0:
        raise ValueError("invalid baryon/photon density")
    baryon_loading = 3.0 * omega_b_h2 * a / (4.0 * omega_gamma_h2)
    return C_KM_S / math.sqrt(3.0 * (1.0 + baryon_loading))


def sound_horizon_at_redshift_mpc(
    params: CEForwardParams,
    z_drag: float,
    integration_points: int = EARLY_RD_INTEGRATION_POINTS,
) -> float:
    """Integrate the comoving photon-baryon sound horizon to a declared z_drag."""
    if not math.isfinite(z_drag) or z_drag <= 0.0:
        raise ValueError("z_drag must be finite and positive")
    if integration_points < 3 or integration_points % 2 == 0:
        raise ValueError("integration_points must be an odd integer >= 3")

    omega_gamma_h2 = photon_density_h2(params.tcmb_k)
    omega_r0 = radiation_density_h2(params.tcmb_k, params.n_eff) / (params.h**2)
    a_drag = 1.0 / (1.0 + z_drag)

    def integrand(a: float) -> float:
        sound_speed = baryon_photon_sound_speed_km_s(
            a,
            params.omega_b_h2,
            omega_gamma_h2,
        )
        scaled_hubble = params.h0 * math.sqrt(
            omega_r0 + params.omega_m0 * a + params.omega_lambda0 * a**4
        )
        return sound_speed / scaled_hubble

    a_grid = linspace(0.0, a_drag, integration_points)
    return simpson([integrand(a) for a in a_grid], a_grid)


@lru_cache(maxsize=64)
def early_universe_sound_horizon(
    params: CEForwardParams,
    integration_points: int = EARLY_RD_INTEGRATION_POINTS,
) -> EarlyUniverseSoundHorizon:
    """Derive r_d with standard early-universe physics and Simpson integration.

    The calculation uses CE density-boundary values, external H0 and ``T_CMB``,
    plus a standard-physics ``N_eff`` assumption. Radiation follows the standard
    blackbody and free-streaming-neutrino relations, the drag epoch uses the
    Eisenstein-Hu fit, and

        r_d = integral_0^a_d c_s(a) / [a^2 H(a)] da.

    Eisenstein-Hu/recombination physics is a Selection/assumption, not a
    CE-internal recombination or nuclear-physics derivation and not a precision
    recombination calculation; its fitted drag redshift can bias r_d at the
    precision level.
    No DESI datum is a runtime input to this function; because this mode was
    added after inspecting DR2 residuals, DR2 is not an untouched holdout.
    ``integration_points`` must be odd for composite Simpson.
    """
    if integration_points < 3 or integration_points % 2 == 0:
        raise ValueError("integration_points must be an odd integer >= 3")

    omega_gamma_h2 = photon_density_h2(params.tcmb_k)
    omega_r_h2 = radiation_density_h2(params.tcmb_k, params.n_eff)
    omega_gamma0 = omega_gamma_h2 / (params.h * params.h)
    omega_r0 = omega_r_h2 / (params.h * params.h)
    z_drag = eisenstein_hu_drag_redshift(params.omega_m_h2, params.omega_b_h2)
    a_drag = 1.0 / (1.0 + z_drag)

    rd_mpc = sound_horizon_at_redshift_mpc(
        params,
        z_drag,
        integration_points=integration_points,
    )
    return EarlyUniverseSoundHorizon(
        omega_b_h2=params.omega_b_h2,
        omega_m_h2=params.omega_m_h2,
        omega_gamma_h2=omega_gamma_h2,
        omega_r_h2=omega_r_h2,
        omega_gamma0=omega_gamma0,
        omega_r0=omega_r0,
        z_drag=z_drag,
        a_drag=a_drag,
        sound_speed_drag_km_s=baryon_photon_sound_speed_km_s(
            a_drag,
            params.omega_b_h2,
            omega_gamma_h2,
        ),
        rd_mpc=rd_mpc,
        integration_points=integration_points,
        status=(
            "Selection/approximation: DESI-independent Eisenstein-Hu and standard "
            "early physics; not a CE-internal recombination/nuclear-physics "
            "derivation or a precision recombination r_drag calculation; no DESI "
            "datum is a runtime input, but DR2 is not an untouched holdout"
        ),
    )


def _validate_recombination_history_cosmology(
    history: RecombinationHistory,
    params: CEForwardParams,
    relative_tolerance: float,
) -> None:
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("cosmology_relative_tolerance must be finite and positive")
    expected_values = (
        ("h0_km_s_mpc", history.metadata.h0_km_s_mpc, params.h0),
        ("omega_b_h2", history.metadata.omega_b_h2, params.omega_b_h2),
        ("omega_m_h2", history.metadata.omega_m_h2, params.omega_m_h2),
        ("tcmb_k", history.metadata.tcmb_k, params.tcmb_k),
        ("n_eff", history.metadata.n_eff, params.n_eff),
    )
    for name, history_value, benchmark_value in expected_values:
        absolute_tolerance = relative_tolerance * max(1.0e-30, abs(benchmark_value))
        if not math.isclose(
            history_value,
            benchmark_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"history cosmology mismatch for {name}: "
                f"history={history_value}, benchmark={benchmark_value}"
            )


def drag_optical_depth_benchmark(
    history: RecombinationHistory,
    params: CEForwardParams,
    *,
    maximum_crossing_bracket_width: float = 20.0,
    cosmology_relative_tolerance: float = 1.0e-6,
    rd_integration_points: int = EARLY_RD_INTEGRATION_POINTS,
) -> DragOpticalDepthBenchmark:
    """Compute tau_drag=1, z_d, and r_d from a solver-exported x_e(z) history.

    The external table supplies recombination only. H(z), R(z), and the sound
    horizon use the declared CE boundary plus standard radiation assumptions.
    The history metadata must match those inputs, and the raw table hash must
    already have been verified by :func:`parse_recombination_history_bytes`.

    The opacity rate is evaluated at every solver grid point, treated as
    piecewise linear in redshift, and integrated exactly on that interpolant.
    This is a reproducible adapter benchmark, not itself a CLASS/CAMB/HyRec run
    or an independent precision-recombination calculation.
    """
    if (
        not math.isfinite(maximum_crossing_bracket_width)
        or maximum_crossing_bracket_width <= 0.0
    ):
        raise ValueError(
            "maximum_crossing_bracket_width must be finite and positive"
        )
    _validate_recombination_history_cosmology(
        history,
        params,
        cosmology_relative_tolerance,
    )

    y_p = history.metadata.helium_mass_fraction_y_p
    rates = tuple(
        drag_optical_depth_rate_per_redshift(z_value, x_e, params, y_p)
        for z_value, x_e in zip(history.redshift, history.electron_fraction)
    )
    cumulative_tau = [0.0]
    crossing_index: int | None = None
    for index in range(len(history.redshift) - 1):
        delta_z = history.redshift[index + 1] - history.redshift[index]
        interval_tau = 0.5 * (rates[index] + rates[index + 1]) * delta_z
        next_tau = cumulative_tau[-1] + interval_tau
        if not math.isfinite(next_tau) or next_tau < cumulative_tau[-1]:
            raise ArithmeticError("tau_drag integration became non-finite or decreased")
        cumulative_tau.append(next_tau)
        if crossing_index is None and next_tau >= 1.0:
            crossing_index = index

    if crossing_index is None:
        raise ValueError(
            "tau_drag=1 is not bracketed by the history: "
            f"tau_drag(z_max={history.redshift[-1]})={cumulative_tau[-1]}"
        )

    bracket_low = history.redshift[crossing_index]
    bracket_high = history.redshift[crossing_index + 1]
    bracket_width = bracket_high - bracket_low
    if bracket_width > maximum_crossing_bracket_width:
        raise ValueError(
            "tau_drag=1 crossing grid is too coarse: "
            f"bracket width {bracket_width} exceeds "
            f"{maximum_crossing_bracket_width}"
        )

    remaining_tau = 1.0 - cumulative_tau[crossing_index]
    rate_low = rates[crossing_index]
    rate_delta = rates[crossing_index + 1] - rate_low

    def interval_tau_at(delta_z: float) -> float:
        return (
            rate_low * delta_z
            + 0.5 * rate_delta * delta_z * delta_z / bracket_width
        )

    if remaining_tau <= 0.0:
        crossing_offset = 0.0
    else:
        offset_low = 0.0
        offset_high = bracket_width
        for _ in range(100):
            offset_mid = 0.5 * (offset_low + offset_high)
            if interval_tau_at(offset_mid) < remaining_tau:
                offset_low = offset_mid
            else:
                offset_high = offset_mid
        crossing_offset = 0.5 * (offset_low + offset_high)

    z_drag = bracket_low + crossing_offset
    rd_mpc = sound_horizon_at_redshift_mpc(
        params,
        z_drag,
        integration_points=rd_integration_points,
    )
    return DragOpticalDepthBenchmark(
        solver_family=history.metadata.solver_family,
        solver_version=history.metadata.solver_version,
        recombination_backend=history.metadata.recombination_backend,
        source_label=history.metadata.source_label,
        source_sha256=history.source_sha256,
        history_metadata=history.metadata,
        input_redshift_column=history.redshift_column,
        input_electron_fraction_column=history.electron_fraction_column,
        input_delimiter=history.delimiter,
        convention=(
            "tau_drag(z)=integral_0^z c*sigma_T*n_e/"
            "[H*(1+z)*R] dz; R=3*rho_b/(4*rho_gamma); "
            "x_e=n_e/n_H; tau_drag(z_d)=1"
        ),
        tau_drag_target=1.0,
        z_drag=z_drag,
        a_drag=1.0 / (1.0 + z_drag),
        rd_mpc=rd_mpc,
        rd_unit="Mpc_comoving",
        hydrogen_nuclei_today_m3=hydrogen_nuclei_number_density_today_m3(
            params,
            y_p,
        ),
        tau_drag_at_z_max=cumulative_tau[-1],
        redshift_grid=history.redshift,
        tau_drag_grid=tuple(cumulative_tau),
        crossing_bracket=(bracket_low, bracket_high),
        crossing_bracket_width=bracket_width,
        rd_integration_points=rd_integration_points,
        integration_method=(
            "piecewise-linear d(tau_drag)/dz with exact trapezoid cumulative; "
            "bisection inside crossing bracket; Simpson sound-horizon integral"
        ),
        status=(
            "External hashed recombination-history benchmark; not a CE-internal "
            "recombination derivation and not itself a CLASS/CAMB/HyRec precision "
            "solver result. Accuracy inherits the exported x_e grid and the declared "
            "CE plus standard-radiation background."
        ),
    )


def sound_horizon_selection(params: CEForwardParams) -> SoundHorizonSelection:
    """Select the backward-compatible external or derived early-universe r_d."""
    if params.rd_mode == "external":
        return SoundHorizonSelection(
            mode="external",
            rd_mpc=params.rd_mpc,
            role="external_input",
            source="CLI/default observational baseline",
            note="Externally supplied sound-horizon calibration.",
            early_universe=None,
        )
    early = early_universe_sound_horizon(params)
    return SoundHorizonSelection(
        mode="early-universe",
        rd_mpc=early.rd_mpc,
        role="derived_selection",
        source=(
            "CE density boundary + external H0/Tcmb + Standard-Model Neff + "
            "Eisenstein-Hu/standard early physics"
        ),
        note=early.status,
        early_universe=early,
    )


@dataclass(frozen=True)
class ParameterProvenance:
    name: str
    value: float
    role: str
    source: str
    note: str

    @property
    def is_external_input(self) -> bool:
        return self.role == "external_input"

    @property
    def is_ce_prediction(self) -> bool:
        """Compatibility view of the historical ``role`` field.

        New closure code must use :attr:`closure_role`; the original role is
        retained because older scorecards and reproduction tests serialized
        it.  In particular, ``ce_prediction`` does not mean a blind or
        physically closed prediction.
        """
        return self.role == "ce_prediction"

    @property
    def closure_role(self) -> str:
        """Fail-closed scientific role used by the unified cosmology gate."""
        if self.role == "ce_prediction":
            return "legacy_model_boundary"
        return self.role

    @property
    def qualifies_as_physical_prediction(self) -> bool:
        """Whether this entry is an independently closed CE prediction."""
        return False


def parameter_provenance(params: CEForwardParams) -> tuple[ParameterProvenance, ...]:
    """Return provenance with legacy roles plus fail-closed closure roles.

    The three density values retain the historical ``ce_prediction`` string
    only for serialized compatibility.  Their current :attr:`closure_role` is
    ``legacy_model_boundary`` because the abundance/action bridge is open.
    """
    rd_selection = sound_horizon_selection(params)
    tcmb_role = (
        "external_input" if params.rd_mode == "early-universe" else "inactive_external_input"
    )
    n_eff_role = (
        "model_assumption" if params.rd_mode == "early-universe" else "inactive_model_assumption"
    )
    return (
        ParameterProvenance(
            "omega_b0",
            params.omega_b0,
            "ce_prediction",
            "reality_stone.clarus.constants.ACTIVE_RATIO",
            "CE density-ratio output used as a boundary; its physical identification is a Bridge.",
        ),
        ParameterProvenance(
            "omega_dm0",
            params.omega_dm0,
            "ce_prediction",
            "reality_stone.clarus.constants.STRUCT_RATIO",
            "CE density-ratio output used as a boundary; particle identity is not predicted.",
        ),
        ParameterProvenance(
            "omega_lambda0",
            params.omega_lambda0,
            "ce_prediction",
            "reality_stone.clarus.constants.BACKGROUND_RATIO",
            "CE density-ratio output used as a boundary; microphysical identity remains open.",
        ),
        ParameterProvenance(
            "h0",
            params.h0,
            "external_input",
            "CLI/default observational baseline",
            "Dimensional distance normalization; not predicted by this forward model.",
        ),
        ParameterProvenance(
            "rd_mpc",
            rd_selection.rd_mpc,
            rd_selection.role,
            rd_selection.source,
            rd_selection.note,
        ),
        ParameterProvenance(
            "tcmb_k",
            params.tcmb_k,
            tcmb_role,
            "CLI/default standard CMB temperature",
            "External early-universe input used only when rd_mode=early-universe.",
        ),
        ParameterProvenance(
            "n_eff",
            params.n_eff,
            n_eff_role,
            "CLI/default Standard-Model relativistic-species assumption",
            "Standard-physics assumption used only when rd_mode=early-universe.",
        ),
        ParameterProvenance(
            "sigma8_0",
            params.sigma8_0,
            "external_input",
            "CLI/default observational baseline",
            "Growth-amplitude normalization; not predicted by this forward model.",
        ),
        ParameterProvenance(
            "w0",
            params.w0,
            "model_assumption",
            "CLI/default conservative LambdaCDM limit",
            "Background equation-of-state choice, not a CE prediction in this gate.",
        ),
        ParameterProvenance(
            "wa",
            params.wa,
            "model_assumption",
            "CLI/default conservative LambdaCDM limit",
            "CPL evolution choice, not a CE prediction in this gate.",
        ),
        ParameterProvenance(
            "gravity_mu_coupling",
            params.gravity_mu_coupling,
            "model_assumption",
            "CLI/default GR limit",
            "Phenomenological growth-sector switch, not a CE prediction in this gate.",
        ),
    )


def dark_energy_scale(a: float, w0: float, wa: float) -> float:
    """CPL scale rho_de(a)/rho_de(1), w(a)=w0+wa(1-a)."""
    if a <= 0.0:
        raise ValueError("scale factor must be positive")
    return a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(3.0 * wa * (a - 1.0))


def w_of_a(a: float, w0: float, wa: float) -> float:
    return w0 + wa * (1.0 - a)


def e2_of_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    return params.omega_m0_background * a ** (-3.0) + params.omega_lambda0_background * de


def e_of_z(z: float, params: CEForwardParams) -> float:
    if z < 0.0:
        raise ValueError("redshift must be non-negative")
    a = 1.0 / (1.0 + z)
    return math.sqrt(e2_of_a(a, params))


def omega_m_of_a(a: float, params: CEForwardParams) -> float:
    return params.omega_m0_background * a ** (-3.0) / e2_of_a(a, params)


def omega_de_of_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    return params.omega_lambda0_background * de / e2_of_a(a, params)


def dlnh_dln_a(a: float, params: CEForwardParams) -> float:
    de = dark_energy_scale(a, params.w0, params.wa)
    w = w_of_a(a, params.w0, params.wa)
    e2 = e2_of_a(a, params)
    d_e2 = (
        -3.0 * params.omega_m0_background * a ** (-3.0)
        - 3.0 * (1.0 + w) * params.omega_lambda0_background * de
    )
    return 0.5 * d_e2 / e2


def residual_mu_of_a(a: float, params: CEForwardParams) -> float:
    """Phenomenological growth-sector residual coupling; GR is exactly mu=1."""
    if params.gravity_mu_coupling == 0.0:
        return 1.0
    today_de = omega_de_of_a(1.0, params)
    if today_de <= 0.0:
        return 1.0
    residual_weight = omega_de_of_a(a, params) / today_de
    return 1.0 - params.gravity_mu_coupling * residual_weight


def luminosity_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    if z <= 0.0:
        return 0.0
    c_km_s = 299792.458
    grid = linspace(0.0, z, n)
    inv_e = [1.0 / e_of_z(zz, params) for zz in grid]
    chi = simpson(inv_e, grid)
    return (c_km_s / params.h0) * (1.0 + z) * chi


def transverse_comoving_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    return luminosity_distance_mpc(z, params, n=n) / (1.0 + z)


def hubble_distance_mpc(z: float, params: CEForwardParams) -> float:
    c_km_s = 299792.458
    return c_km_s / (params.h0 * e_of_z(z, params))


def volume_distance_mpc(z: float, params: CEForwardParams, n: int = 2001) -> float:
    if z <= 0.0:
        return 0.0
    dm = transverse_comoving_distance_mpc(z, params, n=n)
    dh = hubble_distance_mpc(z, params)
    return (z * dm * dm * dh) ** (1.0 / 3.0)


@dataclass(frozen=True)
class BAOObservable:
    z: float
    dm_over_rd: float
    dh_over_rd: float
    dv_over_rd: float

    def value(self, kind: str) -> float:
        if kind == "dm":
            return self.dm_over_rd
        if kind == "dh":
            return self.dh_over_rd
        if kind == "dv":
            return self.dv_over_rd
        raise ValueError(f"unknown BAO observable kind: {kind}")


@dataclass(frozen=True)
class BAODataPoint:
    z: float
    kind: str
    value: float
    sigma: float


@dataclass(frozen=True)
class BAODataset:
    name: str
    data: tuple[BAODataPoint, ...]
    covariance: tuple[tuple[float, ...], ...]
    source: str


@dataclass(frozen=True)
class BAOResidualContribution:
    index: int
    z: float
    kind: str
    observed: float
    predicted: float
    residual: float
    sigma: float
    raw_pull: float
    covariance_contribution: float


@dataclass(frozen=True)
class BAOScaleFitDiagnostic:
    scale_factor: float
    chi2: float
    chi2_improvement: float
    additional_fitted_parameter_count: int
    dof: int
    reduced_chi2: float
    survival_p_value: float
    verdict: str
    aic: float
    bic: float
    aic_improvement: float
    bic_improvement: float
    equivalent_rd_mpc_at_fixed_h0: float
    equivalent_h0_at_fixed_rd: float
    note: str


@dataclass(frozen=True)
class BAOFitAssessment:
    chi2: float
    n_observations: int
    fitted_parameter_count: int
    dof: int
    reduced_chi2: float
    survival_p_value: float
    verdict: str
    aic: float
    bic: float
    covariance_mode: str
    contributions: tuple[BAOResidualContribution, ...]
    scale_fit_diagnostic: BAOScaleFitDiagnostic | None


DESI_DR2_ALL_COVARIANCE: tuple[tuple[float, ...], ...] = (
    (5.78998687e-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 2.83473742e-02, -3.26062007e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -3.26062007e-02, 1.83928040e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 3.23752442e-02, -2.37445646e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -2.37445646e-02, 1.11469198e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 2.61732816e-02, -1.12938006e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -1.12938006e-02, 4.04183878e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.05336516e-01, -2.90308418e-02, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.90308418e-02, 5.04233092e-02, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.83020277e-01, -1.95215562e-01, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.95215562e-01, 2.68336193e-01, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.02136194e-02, -2.31395216e-02),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.31395216e-02, 2.82685779e-01),
)


DESI_DR2_ALL_DATA: tuple[BAODataPoint, ...] = (
    BAODataPoint(0.295, "dv", 7.94167639, math.sqrt(DESI_DR2_ALL_COVARIANCE[0][0])),
    BAODataPoint(0.510, "dm", 13.58758434, math.sqrt(DESI_DR2_ALL_COVARIANCE[1][1])),
    BAODataPoint(0.510, "dh", 21.86294686, math.sqrt(DESI_DR2_ALL_COVARIANCE[2][2])),
    BAODataPoint(0.706, "dm", 17.35069094, math.sqrt(DESI_DR2_ALL_COVARIANCE[3][3])),
    BAODataPoint(0.706, "dh", 19.45534918, math.sqrt(DESI_DR2_ALL_COVARIANCE[4][4])),
    BAODataPoint(0.934, "dm", 21.57563956, math.sqrt(DESI_DR2_ALL_COVARIANCE[5][5])),
    BAODataPoint(0.934, "dh", 17.64149464, math.sqrt(DESI_DR2_ALL_COVARIANCE[6][6])),
    BAODataPoint(1.321, "dm", 27.60085612, math.sqrt(DESI_DR2_ALL_COVARIANCE[7][7])),
    BAODataPoint(1.321, "dh", 14.17602155, math.sqrt(DESI_DR2_ALL_COVARIANCE[8][8])),
    BAODataPoint(1.484, "dm", 30.51190063, math.sqrt(DESI_DR2_ALL_COVARIANCE[9][9])),
    BAODataPoint(1.484, "dh", 12.81699964, math.sqrt(DESI_DR2_ALL_COVARIANCE[10][10])),
    BAODataPoint(2.330, "dh", 8.631545674846294, math.sqrt(DESI_DR2_ALL_COVARIANCE[11][11])),
    BAODataPoint(2.330, "dm", 38.988973961958784, math.sqrt(DESI_DR2_ALL_COVARIANCE[12][12])),
)


def named_bao_dataset(name: str) -> BAODataset:
    key = name.strip().lower()
    source = "CobayaSampler/bao_data desi_bao_dr2 gaussian BAO mean/cov ASCII"
    if key == "desi-dr2-bgs":
        return BAODataset(
            name="desi-dr2-bgs",
            data=(DESI_DR2_ALL_DATA[0],),
            covariance=((DESI_DR2_ALL_COVARIANCE[0][0],),),
            source=source,
        )
    if key == "desi-dr2-all":
        return BAODataset(
            name="desi-dr2-all",
            data=DESI_DR2_ALL_DATA,
            covariance=DESI_DR2_ALL_COVARIANCE,
            source=source,
        )
    raise ValueError(f"unknown BAO dataset: {name}")


def bao_observable(z: float, params: CEForwardParams, n: int = 2001) -> BAOObservable:
    rd_mpc = sound_horizon_selection(params).rd_mpc
    return BAOObservable(
        z=z,
        dm_over_rd=transverse_comoving_distance_mpc(z, params, n=n) / rd_mpc,
        dh_over_rd=hubble_distance_mpc(z, params) / rd_mpc,
        dv_over_rd=volume_distance_mpc(z, params, n=n) / rd_mpc,
    )


def parse_bao_data(spec: str) -> tuple[BAODataPoint, ...]:
    """Parse 'z:kind:value:sigma,...' for kind in {dm,dh,dv}."""
    items: list[BAODataPoint] = []
    text = spec.strip()
    if not text:
        return ()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [field.strip() for field in part.split(":")]
        if len(fields) != 4:
            raise ValueError(f"invalid BAO point '{part}': expected z:kind:value:sigma")
        z = float(fields[0])
        kind = fields[1].lower()
        value = float(fields[2])
        sigma = float(fields[3])
        if z <= 0.0:
            raise ValueError("BAO redshift must be positive")
        if kind not in {"dm", "dh", "dv"}:
            raise ValueError("BAO kind must be one of dm, dh, dv")
        if sigma <= 0.0:
            raise ValueError("BAO sigma must be positive")
        items.append(BAODataPoint(z=z, kind=kind, value=value, sigma=sigma))
    return tuple(items)


def parse_covariance_matrix(spec: str) -> tuple[tuple[float, ...], ...]:
    """Parse covariance rows, e.g. '0.04,0.01;0.01,0.09'."""
    text = spec.strip()
    if not text:
        return ()
    rows: list[tuple[float, ...]] = []
    for raw_row in text.split(";"):
        row_text = raw_row.strip()
        if not row_text:
            continue
        row_text = row_text.replace(",", " ")
        values = tuple(float(part) for part in row_text.split())
        if not values:
            continue
        rows.append(values)
    if not rows:
        return ()
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError("covariance matrix must be square")
    for i in range(n):
        if rows[i][i] <= 0.0:
            raise ValueError("covariance diagonal entries must be positive")
        for j in range(i + 1, n):
            if abs(rows[i][j] - rows[j][i]) > 1.0e-10:
                raise ValueError("covariance matrix must be symmetric")
    return tuple(rows)


def invert_matrix(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Invert a small dense matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    if n == 0:
        return ()
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    aug = [
        [float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) <= 1.0e-15:
            raise ValueError("matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [value - factor * pivot_value for value, pivot_value in zip(aug[row], aug[col])]
    return tuple(tuple(row[n:]) for row in aug)


def quadratic_form(vector: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> float:
    if len(matrix) != len(vector):
        raise ValueError("matrix/vector size mismatch")
    total = 0.0
    for i, vi in enumerate(vector):
        for j, vj in enumerate(vector):
            total += vi * matrix[i][j] * vj
    return total


def regularized_gamma_q(shape: float, x: float) -> float:
    """Return Q(shape, x), the regularized upper incomplete gamma function.

    The series is used for x < shape + 1 and a continued fraction otherwise.
    This standard split avoids adding SciPy as a runtime dependency.
    """
    if shape <= 0.0:
        raise ValueError("gamma shape must be positive")
    if x < 0.0:
        raise ValueError("gamma argument must be non-negative")
    if x == 0.0:
        return 1.0

    epsilon = 1.0e-14
    max_iterations = 1000
    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)

    if x < shape + 1.0:
        term = 1.0 / shape
        series = term
        denominator = shape
        for _ in range(max_iterations):
            denominator += 1.0
            term *= x / denominator
            series += term
            if abs(term) <= abs(series) * epsilon:
                break
        else:
            raise ArithmeticError("regularized gamma series did not converge")
        lower = series * math.exp(log_prefactor)
        return min(1.0, max(0.0, 1.0 - lower))

    tiny = 1.0e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0.0:
        d = -d
    fraction = d
    for iteration in range(1, max_iterations + 1):
        coefficient = -float(iteration) * (float(iteration) - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    else:
        raise ArithmeticError("regularized gamma continued fraction did not converge")
    upper = math.exp(log_prefactor) * fraction
    return min(1.0, max(0.0, upper))


def chi_square_survival(chi2: float, dof: int) -> float:
    """Return P[ChiSquare(dof) >= chi2]."""
    if chi2 < 0.0:
        raise ValueError("chi2 must be non-negative")
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    return regularized_gamma_q(0.5 * dof, 0.5 * chi2)


def chi_square_verdict(p_value: float) -> str:
    """Map a fixed-model chi-square survival probability to an audit verdict.

    PASS requires p >= 0.05. TENSION covers 0.0027 <= p < 0.05. REJECT is
    p < 0.0027, using the conventional two-sided Gaussian 3-sigma tail as the
    preregistered severe-mismatch boundary. This is a goodness-of-fit gate for
    a fixed model, not a parameter-estimation or model-selection criterion.
    """
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p-value must lie in [0, 1]")
    if p_value >= 0.05:
        return "PASS"
    if p_value >= 0.0027:
        return "TENSION"
    return "REJECT"


def assess_bao_fit(
    data: tuple[BAODataPoint, ...],
    params: CEForwardParams,
    covariance: tuple[tuple[float, ...], ...] | None = None,
    n: int = 2001,
    fitted_parameter_count: int = 0,
) -> BAOFitAssessment:
    """Assess a compressed BAO fixed-model fit and decompose its full chi-square.

    Degrees of freedom are N - fitted_parameter_count. For the built-in CE
    boundary check no parameters are fitted to the 13 BAO points, so dof=13.
    Per-observation covariance contributions use
    ``r_i (C^-1 r)_i``; they sum exactly to ``r^T C^-1 r`` but individual terms
    may be negative when correlations redistribute the joint discrepancy.

    The returned one-parameter scale ablation analytically fits
    ``q* = (y^T C^-1 d) / (y^T C^-1 y)`` for prediction vector ``y`` and data
    vector ``d``. Because all BAO distance ratios scale as ``1/(H0*rd)``, it
    diagnoses sensitivity to the externally supplied H0*rd normalization.
    AIC=chi2+2k and BIC=chi2+k*log(N) use the total fitted parameter count, so
    their reported improvements include the scale parameter penalty.
    It is a diagnostic fit, not a CE prediction or a promoted model result.
    """
    if not data:
        raise ValueError("BAO data must not be empty")
    if fitted_parameter_count < 0:
        raise ValueError("fitted_parameter_count must be non-negative")
    dof = len(data) - fitted_parameter_count
    if dof <= 0:
        raise ValueError("BAO degrees of freedom must be positive")

    if covariance is None:
        covariance_used = tuple(
            tuple(point.sigma**2 if i == j else 0.0 for j in range(len(data)))
            for i, point in enumerate(data)
        )
        covariance_mode = "diagonal"
    else:
        if len(covariance) != len(data):
            raise ValueError("covariance size must match BAO data length")
        covariance_used = covariance
        covariance_mode = "full"

    predictions = tuple(
        bao_observable(point.z, params, n=n).value(point.kind) for point in data
    )
    residuals = tuple(predicted - point.value for predicted, point in zip(predictions, data))
    inverse = invert_matrix(covariance_used)
    precision_weighted = tuple(
        sum(inverse[i][j] * residuals[j] for j in range(len(data)))
        for i in range(len(data))
    )
    covariance_contributions = tuple(
        residual * weighted for residual, weighted in zip(residuals, precision_weighted)
    )
    chi2 = sum(covariance_contributions)
    aic = chi2 + 2.0 * fitted_parameter_count
    bic = chi2 + fitted_parameter_count * math.log(len(data))
    contributions = tuple(
        BAOResidualContribution(
            index=index,
            z=point.z,
            kind=point.kind,
            observed=point.value,
            predicted=predicted,
            residual=residual,
            sigma=point.sigma,
            raw_pull=residual / point.sigma,
            covariance_contribution=covariance_contribution,
        )
        for index, (point, predicted, residual, covariance_contribution) in enumerate(
            zip(data, predictions, residuals, covariance_contributions)
        )
    )
    p_value = chi_square_survival(chi2, dof)
    scale_fit_diagnostic = None
    scale_fit_dof = dof - 1
    if scale_fit_dof > 0:
        observed = tuple(point.value for point in data)
        prediction_precision_prediction = quadratic_form(predictions, inverse)
        if prediction_precision_prediction <= 0.0:
            raise ValueError("prediction precision norm must be positive")
        prediction_precision_observed = sum(
            predictions[i] * sum(inverse[i][j] * observed[j] for j in range(len(data)))
            for i in range(len(data))
        )
        scale_factor = prediction_precision_observed / prediction_precision_prediction
        scaled_residuals = tuple(
            scale_factor * predicted - point.value
            for predicted, point in zip(predictions, data)
        )
        scale_fit_chi2 = quadratic_form(scaled_residuals, inverse)
        scale_fit_p_value = chi_square_survival(scale_fit_chi2, scale_fit_dof)
        scale_fit_parameter_count = fitted_parameter_count + 1
        scale_fit_aic = scale_fit_chi2 + 2.0 * scale_fit_parameter_count
        scale_fit_bic = scale_fit_chi2 + scale_fit_parameter_count * math.log(len(data))
        selected_rd_mpc = sound_horizon_selection(params).rd_mpc
        scale_fit_diagnostic = BAOScaleFitDiagnostic(
            scale_factor=scale_factor,
            chi2=scale_fit_chi2,
            chi2_improvement=chi2 - scale_fit_chi2,
            additional_fitted_parameter_count=1,
            dof=scale_fit_dof,
            reduced_chi2=scale_fit_chi2 / scale_fit_dof,
            survival_p_value=scale_fit_p_value,
            verdict=chi_square_verdict(scale_fit_p_value),
            aic=scale_fit_aic,
            bic=scale_fit_bic,
            aic_improvement=aic - scale_fit_aic,
            bic_improvement=bic - scale_fit_bic,
            equivalent_rd_mpc_at_fixed_h0=selected_rd_mpc / scale_factor,
            equivalent_h0_at_fixed_rd=params.h0 / scale_factor,
            note="diagnostic fit to external H0*rd scale; not a CE prediction",
        )
    return BAOFitAssessment(
        chi2=chi2,
        n_observations=len(data),
        fitted_parameter_count=fitted_parameter_count,
        dof=dof,
        reduced_chi2=chi2 / dof,
        survival_p_value=p_value,
        verdict=chi_square_verdict(p_value),
        aic=aic,
        bic=bic,
        covariance_mode=covariance_mode,
        contributions=contributions,
        scale_fit_diagnostic=scale_fit_diagnostic,
    )


def bao_chi2(data: tuple[BAODataPoint, ...], params: CEForwardParams, n: int = 2001) -> float:
    """Diagonal compressed BAO chi2."""
    return assess_bao_fit(data, params, n=n).chi2


def bao_chi2_with_covariance(
    data: tuple[BAODataPoint, ...],
    covariance: tuple[tuple[float, ...], ...],
    params: CEForwardParams,
    n: int = 2001,
) -> float:
    """Full compressed BAO chi2 using a supplied covariance matrix."""
    return assess_bao_fit(data, params, covariance=covariance, n=n).chi2


def solve_growth(
    params: CEForwardParams,
    a_min: float = 1.0e-3,
    n: int = 2001,
) -> tuple[list[float], list[float], list[float]]:
    """Solve linear growth D and f=dlnD/dlna, normalized to D(a=1)=1."""
    a_grid = logspace(a_min, 1.0, n)
    ln_a = [math.log(a) for a in a_grid]
    dln = (ln_a[-1] - ln_a[0]) / (len(ln_a) - 1)

    growth = [0.0 for _ in a_grid]
    growth_prime = [0.0 for _ in a_grid]
    growth[0] = a_grid[0]
    growth_prime[0] = a_grid[0]

    def rhs(x: float, d_val: float, dp_val: float) -> tuple[float, float]:
        a = math.exp(x)
        om = omega_m_of_a(a, params)
        mu = residual_mu_of_a(a, params)
        friction = 2.0 + dlnh_dln_a(a, params)
        return dp_val, -friction * dp_val + 1.5 * mu * om * d_val

    for i in range(len(a_grid) - 1):
        x = ln_a[i]
        d_val = growth[i]
        dp_val = growth_prime[i]
        k1 = rhs(x, d_val, dp_val)
        k2 = rhs(x + 0.5 * dln, d_val + 0.5 * dln * k1[0], dp_val + 0.5 * dln * k1[1])
        k3 = rhs(x + 0.5 * dln, d_val + 0.5 * dln * k2[0], dp_val + 0.5 * dln * k2[1])
        k4 = rhs(x + dln, d_val + dln * k3[0], dp_val + dln * k3[1])
        growth[i + 1] = d_val + (dln / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        growth_prime[i + 1] = dp_val + (dln / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])

    norm = growth[-1] if growth[-1] != 0.0 else 1.0
    d_norm = [v / norm for v in growth]
    f_grid = []
    for d_val, dp_val in zip(growth, growth_prime):
        f_grid.append(0.0 if d_val <= 0.0 else dp_val / d_val)
    return a_grid, d_norm, f_grid


def sigma8_at_z(z: float, params: CEForwardParams, a_grid: list[float], d_grid: list[float]) -> float:
    a = 1.0 / (1.0 + z)
    return params.sigma8_0 * interp_linear(a_grid, d_grid, a)


def f_sigma8_at_z(
    z: float,
    params: CEForwardParams,
    a_grid: list[float],
    d_grid: list[float],
    f_grid: list[float],
) -> float:
    a = 1.0 / (1.0 + z)
    d = interp_linear(a_grid, d_grid, a)
    f = interp_linear(a_grid, f_grid, a)
    return f * params.sigma8_0 * d


def s8_today(params: CEForwardParams) -> float:
    return params.sigma8_0 * math.sqrt(params.omega_m0 / 0.3)


def print_report(params: CEForwardParams, z_values: tuple[float, ...]) -> None:
    a_grid, d_grid, f_grid = solve_growth(params)
    rd_selection = sound_horizon_selection(params)
    print("# CE Residual Cosmology Forward Model")
    print()
    print(f"omega_b0 {params.omega_b0:.6f}")
    print(f"omega_dm0 {params.omega_dm0:.6f}")
    print(f"omega_m0 {params.omega_m0:.6f}")
    print(f"omega_lambda0 {params.omega_lambda0:.6f}")
    print(f"h0 {params.h0:.6f}")
    print(f"rd_mode {rd_selection.mode}")
    print(f"rd_mpc {rd_selection.rd_mpc:.9f}")
    print(f"rd_mpc_external_input {params.rd_mpc:.9f}")
    print(f"rd_role {rd_selection.role}")
    print(f"rd_note {rd_selection.note}")
    if rd_selection.early_universe is not None:
        early = rd_selection.early_universe
        print(f"tcmb_k {params.tcmb_k:.6f}")
        print(f"n_eff {params.n_eff:.6f}")
        print(f"omega_b_h2 {early.omega_b_h2:.12g}")
        print(f"omega_m_h2 {early.omega_m_h2:.12g}")
        print(f"omega_gamma_h2 {early.omega_gamma_h2:.12g}")
        print(f"omega_r_h2 {early.omega_r_h2:.12g}")
        print(f"omega_gamma0 {early.omega_gamma0:.12g}")
        print(f"omega_r0 {early.omega_r0:.12g}")
        print(f"z_drag_eisenstein_hu {early.z_drag:.9f}")
        print(f"a_drag {early.a_drag:.12g}")
        print(f"sound_speed_drag_km_s {early.sound_speed_drag_km_s:.9f}")
        print(f"rd_integration_points {early.integration_points}")
    print(f"w0 {params.w0:.6f}")
    print(f"wa {params.wa:.6f}")
    print(f"gravity_mu_coupling {params.gravity_mu_coupling:.6f}")
    print(f"S8_today {s8_today(params):.6f}")
    print("physical_closure INCOMPLETE")
    print("blind_prediction false")
    print()
    print("parameter_provenance(name,value,legacy_role,closure_role,physical_prediction,source)")
    for entry in parameter_provenance(params):
        print(
            f"{entry.name},{entry.value:.9g},{entry.role},{entry.closure_role},"
            f"{str(entry.qualifies_as_physical_prediction).lower()},{entry.source}"
        )
    print()
    print("z,E(z),D_L_Mpc,D_M_over_rd,D_H_over_rd,D_V_over_rd,Omega_m(z),Omega_de(z),sigma8(z),f_sigma8(z)")
    for z in z_values:
        a = 1.0 / (1.0 + z)
        bao = bao_observable(z, params)
        print(
            f"{z:.6f},"
            f"{e_of_z(z, params):.9f},"
            f"{luminosity_distance_mpc(z, params):.6f},"
            f"{bao.dm_over_rd:.9f},"
            f"{bao.dh_over_rd:.9f},"
            f"{bao.dv_over_rd:.9f},"
            f"{omega_m_of_a(a, params):.9f},"
            f"{omega_de_of_a(a, params):.9f},"
            f"{sigma8_at_z(z, params, a_grid, d_grid):.9f},"
            f"{f_sigma8_at_z(z, params, a_grid, d_grid, f_grid):.9f}"
        )
    print()
    print("coverage", ForwardCoverage().summary)


def main() -> int:
    parser = argparse.ArgumentParser(prog="ce_residual_forward_model")
    parser.add_argument("--h0", type=float, default=67.4)
    parser.add_argument("--rd-mpc", type=float, default=147.09)
    parser.add_argument(
        "--rd-mode",
        choices=["external", "early-universe"],
        default="external",
        help=(
            "Use external --rd-mpc (default) or derive rd from CE density boundaries, "
            "external H0/Tcmb, Standard-Model Neff, and Eisenstein-Hu early physics."
        ),
    )
    parser.add_argument("--tcmb-k", type=float, default=TCMB_REFERENCE_K)
    parser.add_argument("--n-eff", type=float, default=DEFAULT_N_EFF)
    parser.add_argument("--sigma8-0", type=float, default=0.811)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--wa", type=float, default=0.0)
    parser.add_argument("--gravity-mu-coupling", type=float, default=0.0)
    parser.add_argument("--z-list", type=str, default="0,0.5,1,2")
    parser.add_argument(
        "--bao-data",
        type=str,
        default="",
        help="Optional diagonal BAO data: z:kind:value:sigma,... where kind is dm, dh, or dv.",
    )
    parser.add_argument(
        "--bao-cov",
        type=str,
        default="",
        help="Optional full BAO covariance rows, e.g. '0.04,0.01;0.01,0.09'.",
    )
    parser.add_argument(
        "--bao-dataset",
        type=str,
        default="",
        choices=["", "desi-dr2-bgs", "desi-dr2-all"],
        help="Optional built-in BAO dataset. Overrides --bao-data/--bao-cov.",
    )
    args = parser.parse_args()

    z_values = tuple(float(part.strip()) for part in args.z_list.split(",") if part.strip())
    params = CEForwardParams(
        h0=args.h0,
        rd_mpc=args.rd_mpc,
        rd_mode=args.rd_mode,
        tcmb_k=args.tcmb_k,
        n_eff=args.n_eff,
        sigma8_0=args.sigma8_0,
        w0=args.w0,
        wa=args.wa,
        gravity_mu_coupling=args.gravity_mu_coupling,
    )
    print_report(params, z_values)
    dataset = named_bao_dataset(args.bao_dataset) if args.bao_dataset else None
    bao_data = dataset.data if dataset is not None else parse_bao_data(args.bao_data)
    if bao_data:
        bao_cov = dataset.covariance if dataset is not None else parse_covariance_matrix(args.bao_cov)
        assessment = assess_bao_fit(
            bao_data,
            params,
            covariance=bao_cov if bao_cov else None,
        )
        print()
        print("bao_chi2", f"{assessment.chi2:.9f}")
        print("bao_fitted_parameter_count", assessment.fitted_parameter_count)
        print("bao_dof", assessment.dof)
        print("bao_reduced_chi2", f"{assessment.reduced_chi2:.9f}")
        print("bao_survival_p_value", f"{assessment.survival_p_value:.12g}")
        print("bao_verdict", assessment.verdict)
        print("bao_aic", f"{assessment.aic:.9f}")
        print("bao_bic", f"{assessment.bic:.9f}")
        print("bao_covariance", assessment.covariance_mode)
        print("bao_n", assessment.n_observations)
        if dataset is not None:
            print("bao_dataset", dataset.name)
            print("bao_source", dataset.source)
        print("bao_note", "compressed_bao_likelihood")
        if assessment.scale_fit_diagnostic is not None:
            scale_fit = assessment.scale_fit_diagnostic
            print()
            print("bao_scale_fit_q", f"{scale_fit.scale_factor:.12f}")
            print("bao_scale_fit_chi2", f"{scale_fit.chi2:.9f}")
            print("bao_scale_fit_delta_chi2", f"{scale_fit.chi2_improvement:.9f}")
            print(
                "bao_scale_fit_additional_parameter_count",
                scale_fit.additional_fitted_parameter_count,
            )
            print("bao_scale_fit_dof", scale_fit.dof)
            print("bao_scale_fit_reduced_chi2", f"{scale_fit.reduced_chi2:.9f}")
            print("bao_scale_fit_survival_p_value", f"{scale_fit.survival_p_value:.12g}")
            print("bao_scale_fit_verdict", scale_fit.verdict)
            print("bao_scale_fit_aic", f"{scale_fit.aic:.9f}")
            print("bao_scale_fit_bic", f"{scale_fit.bic:.9f}")
            print("bao_scale_fit_delta_aic", f"{scale_fit.aic_improvement:.9f}")
            print("bao_scale_fit_delta_bic", f"{scale_fit.bic_improvement:.9f}")
            print(
                "bao_scale_fit_equivalent_rd_mpc_at_fixed_h0",
                f"{scale_fit.equivalent_rd_mpc_at_fixed_h0:.9f}",
            )
            print(
                "bao_scale_fit_equivalent_h0_at_fixed_rd",
                f"{scale_fit.equivalent_h0_at_fixed_rd:.9f}",
            )
            print("bao_scale_fit_note", scale_fit.note)
        print()
        print(
            "bao_contribution("
            "index,z,kind,observed,predicted,residual,raw_pull,covariance_contribution)"
        )
        for contribution in assessment.contributions:
            print(
                f"{contribution.index},"
                f"{contribution.z:.6f},"
                f"{contribution.kind},"
                f"{contribution.observed:.9f},"
                f"{contribution.predicted:.9f},"
                f"{contribution.residual:+.9f},"
                f"{contribution.raw_pull:+.9f},"
                f"{contribution.covariance_contribution:+.9f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
