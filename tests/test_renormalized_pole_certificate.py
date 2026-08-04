from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.renormalized_pole_certificate import (
    CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV,
    RenormalizedKernelManifest,
    RenormalizedPoleStage,
    ScalarDispersionReplica,
    ScalarKernelReplica,
    current_ce_renormalized_pole_certificate,
    renormalized_pole_certificate,
)


ACTION_HASH = "a" * 64
COUNTERTERM_HASH = "b" * 64
BACKGROUND_HASH = "c" * 64
FIELD_ID = "clarus_scalar_control"
CONTROL_MASS_MEV = 2.0


def _manifest() -> RenormalizedKernelManifest:
    return RenormalizedKernelManifest(
        schema_version="1.0",
        field_id=FIELD_ID,
        action_sha256=ACTION_HASH,
        counterterm_sha256=COUNTERTERM_HASH,
        background_sha256=BACKGROUND_HASH,
        renormalization_scheme="on_shell_scalar_control",
    )


def _kernel(
    replica_id: str,
    *,
    gauge_parameter: float,
    renormalization_scale_mev: float,
    pole_s_mev2: float = CONTROL_MASS_MEV**2,
    derivative: float = 1.0,
    imaginary_at_pole_mev2: float = 0.0,
    cut_gap_mev2: float = 20.0,
    action_sha256: str = ACTION_HASH,
    double_zero: bool = False,
) -> ScalarKernelReplica:
    invariant = (
        pole_s_mev2 - 1.0,
        pole_s_mev2,
        pole_s_mev2 + 1.0,
    )
    if double_zero:
        kernel_real = tuple((value - pole_s_mev2) ** 2 for value in invariant)
    else:
        kernel_real = tuple(derivative * (value - pole_s_mev2) for value in invariant)
    return ScalarKernelReplica(
        replica_id=replica_id,
        field_id=FIELD_ID,
        action_sha256=action_sha256,
        counterterm_sha256=COUNTERTERM_HASH,
        background_sha256=BACKGROUND_HASH,
        gauge_parameter=gauge_parameter,
        renormalization_scale_mev=renormalization_scale_mev,
        invariant_s_mev2=invariant,
        kernel_real_mev2=kernel_real,
        kernel_imag_mev2=(
            imaginary_at_pole_mev2,
            imaginary_at_pole_mev2,
            imaginary_at_pole_mev2,
        ),
        first_cut_s_mev2=pole_s_mev2 + cut_gap_mev2,
    )


def _kernel_replicas(
    *,
    derivative: float = 1.0,
) -> tuple[ScalarKernelReplica, ...]:
    return (
        _kernel(
            "baseline",
            gauge_parameter=0.0,
            renormalization_scale_mev=100.0,
            derivative=derivative,
        ),
        _kernel(
            "gauge_holdout",
            gauge_parameter=1.0,
            renormalization_scale_mev=100.0,
            derivative=derivative,
        ),
        _kernel(
            "scale_holdout",
            gauge_parameter=0.0,
            renormalization_scale_mev=200.0,
            derivative=derivative,
        ),
    )


def _dispersion(
    kernel: ScalarKernelReplica,
    *,
    energy_squared_offset_mev2: float = 0.0,
    action_sha256: str = ACTION_HASH,
) -> ScalarDispersionReplica:
    pole_s = kernel.invariant_s_mev2[1]
    momenta = (0.0, 1.0, 3.0)
    energies = tuple(
        math.sqrt(pole_s + momentum**2 + energy_squared_offset_mev2) for momentum in momenta
    )
    return ScalarDispersionReplica(
        replica_id=kernel.replica_id,
        field_id=FIELD_ID,
        action_sha256=action_sha256,
        counterterm_sha256=COUNTERTERM_HASH,
        background_sha256=BACKGROUND_HASH,
        gauge_parameter=kernel.gauge_parameter,
        renormalization_scale_mev=kernel.renormalization_scale_mev,
        momentum_mev=momenta,
        energy_mev=energies,
    )


def _certificate(
    kernels: tuple[ScalarKernelReplica, ...],
    *,
    dispersions: tuple[ScalarDispersionReplica, ...] | None = None,
):
    if dispersions is None:
        dispersions = tuple(_dispersion(kernel) for kernel in kernels)
    return renormalized_pole_certificate(
        registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
        manifest=_manifest(),
        kernel_replicas=kernels,
        dispersion_replicas=dispersions,
    )


def test_current_ce_evidence_stops_at_registered_inverse_correlation_scale() -> None:
    certificate = current_ce_renormalized_pole_certificate()

    assert certificate.registered_inverse_correlation_scale_mev == (
        CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV
    )
    assert certificate.maximum_supported_stage is RenormalizedPoleStage.REGISTERED_SCALE
    assert not certificate.renormalized_kernel_control_pass
    assert not certificate.isolated_simple_pole_control_pass
    assert not certificate.positive_residue_control_pass
    assert not certificate.dispersion_control_pass
    assert certificate.first_blocker == "renormalized action manifest is absent"
    assert "counterterm manifest is absent" in certificate.blockers
    assert "scalar quadratic-kernel replica data are absent" in certificate.blockers


def test_scalar_replica_control_reaches_dispersion_but_never_lsz_or_ce_identity() -> None:
    certificate = _certificate(_kernel_replicas())
    payload = certificate.to_dict()

    assert certificate.kernel_manifest_consistent
    assert certificate.gauge_holdout_present
    assert certificate.scale_holdout_present
    assert certificate.renormalized_kernel_control_pass
    assert certificate.isolated_simple_pole_control_pass
    assert certificate.positive_residue_control_pass
    assert certificate.dispersion_control_pass
    assert certificate.maximum_supported_stage is RenormalizedPoleStage.DISPERSION_CONTROL
    assert all(item.pole_s_mev2 == pytest.approx(4.0) for item in certificate.kernel_diagnostics)
    assert all(item.pole_residue == pytest.approx(1.0) for item in certificate.kernel_diagnostics)
    assert all(item.dispersion_pass for item in certificate.dispersion_diagnostics)
    assert not certificate.spectral_positivity_derived
    assert not certificate.asymptotic_state_derived
    assert not certificate.physical_lsz_particle_derived
    assert not certificate.ce_field_identity_derived
    assert payload["maximum_supported_stage"] == "DISPERSION_CONTROL"


def test_negative_residue_stops_after_isolated_simple_pole() -> None:
    certificate = _certificate(_kernel_replicas(derivative=-1.0))

    assert certificate.isolated_simple_pole_control_pass
    assert not certificate.positive_residue_control_pass
    assert certificate.maximum_supported_stage is (
        RenormalizedPoleStage.ISOLATED_SIMPLE_POLE_CONTROL
    )
    assert all(not item.positive_residue for item in certificate.kernel_diagnostics)
    assert "a scalar pole residue is non-positive" in certificate.blockers


def test_double_zero_is_not_a_simple_pole() -> None:
    kernels = tuple(
        replace(
            kernel,
            kernel_real_mev2=(1.0, 0.0, 1.0),
        )
        for kernel in _kernel_replicas()
    )
    certificate = _certificate(kernels)

    assert certificate.renormalized_kernel_control_pass
    assert not certificate.isolated_simple_pole_control_pass
    assert certificate.maximum_supported_stage is (
        RenormalizedPoleStage.RENORMALIZED_KERNEL_CONTROL
    )
    assert all(not item.simple_root for item in certificate.kernel_diagnostics)


def test_nonzero_imaginary_kernel_at_root_is_not_a_stable_real_pole() -> None:
    kernels = tuple(
        replace(kernel, kernel_imag_mev2=(0.01, 0.01, 0.01)) for kernel in _kernel_replicas()
    )
    certificate = _certificate(kernels)

    assert certificate.renormalized_kernel_control_pass
    assert not certificate.isolated_simple_pole_control_pass
    assert all(not item.imaginary_part_pass for item in certificate.kernel_diagnostics)


def test_negative_mass_squared_root_is_unstable_not_a_particle_pole() -> None:
    kernels = (
        _kernel(
            "baseline",
            gauge_parameter=0.0,
            renormalization_scale_mev=100.0,
            pole_s_mev2=-4.0,
        ),
        _kernel(
            "gauge_holdout",
            gauge_parameter=1.0,
            renormalization_scale_mev=100.0,
            pole_s_mev2=-4.0,
        ),
        _kernel(
            "scale_holdout",
            gauge_parameter=0.0,
            renormalization_scale_mev=200.0,
            pole_s_mev2=-4.0,
        ),
    )
    certificate = _certificate(kernels, dispersions=())

    assert certificate.renormalized_kernel_control_pass
    assert not certificate.isolated_simple_pole_control_pass
    assert all(not item.positive_mass_squared for item in certificate.kernel_diagnostics)


def test_pole_touching_the_first_cut_is_not_isolated() -> None:
    kernels = tuple(
        replace(kernel, first_cut_s_mev2=kernel.invariant_s_mev2[1])
        for kernel in _kernel_replicas()
    )
    certificate = _certificate(kernels)

    assert not certificate.isolated_simple_pole_control_pass
    assert all(not item.cut_separated for item in certificate.kernel_diagnostics)


@pytest.mark.parametrize(
    ("changed_replica", "drift_attribute"),
    [
        ("gauge_holdout", "maximum_gauge_pole_relative_drift"),
        ("scale_holdout", "maximum_scale_pole_relative_drift"),
    ],
)
def test_gauge_or_scale_pole_drift_fails_closed(
    changed_replica: str,
    drift_attribute: str,
) -> None:
    kernels = tuple(
        _kernel(
            kernel.replica_id,
            gauge_parameter=kernel.gauge_parameter,
            renormalization_scale_mev=kernel.renormalization_scale_mev,
            pole_s_mev2=4.1 if kernel.replica_id == changed_replica else 4.0,
        )
        for kernel in _kernel_replicas()
    )
    certificate = _certificate(kernels)

    drift = getattr(certificate, drift_attribute)
    assert drift is not None and drift > 1.0e-2
    assert not certificate.isolated_simple_pole_control_pass
    assert certificate.maximum_supported_stage is (
        RenormalizedPoleStage.RENORMALIZED_KERNEL_CONTROL
    )
    assert "pole gauge/scale drift exceeds tolerance" in certificate.blockers


def test_residue_drift_fails_positive_residue_control() -> None:
    kernels = tuple(
        _kernel(
            kernel.replica_id,
            gauge_parameter=kernel.gauge_parameter,
            renormalization_scale_mev=kernel.renormalization_scale_mev,
            derivative=2.0 if kernel.replica_id == "gauge_holdout" else 1.0,
        )
        for kernel in _kernel_replicas()
    )
    certificate = _certificate(kernels)

    assert certificate.isolated_simple_pole_control_pass
    assert not certificate.positive_residue_control_pass
    assert certificate.maximum_gauge_residue_relative_drift == pytest.approx(0.5)
    assert "residue gauge/scale drift exceeds tolerance" in certificate.blockers


def test_bad_relativistic_dispersion_stops_after_positive_residue() -> None:
    kernels = _kernel_replicas()
    dispersions = tuple(
        _dispersion(
            kernel,
            energy_squared_offset_mev2=(0.5 if kernel.replica_id == "baseline" else 0.0),
        )
        for kernel in kernels
    )
    certificate = _certificate(kernels, dispersions=dispersions)

    assert certificate.positive_residue_control_pass
    assert not certificate.dispersion_control_pass
    assert certificate.maximum_supported_stage is RenormalizedPoleStage.POSITIVE_RESIDUE_CONTROL
    baseline = next(
        item for item in certificate.dispersion_diagnostics if item.replica_id == "baseline"
    )
    assert baseline.maximum_relative_residual is not None
    assert baseline.maximum_relative_residual > 0.1


def test_kernel_action_hash_mismatch_stays_at_registered_scale() -> None:
    kernels = list(_kernel_replicas())
    kernels[1] = replace(kernels[1], action_sha256="d" * 64)
    certificate = _certificate(tuple(kernels))

    assert not certificate.kernel_manifest_consistent
    assert not certificate.renormalized_kernel_control_pass
    assert certificate.maximum_supported_stage is RenormalizedPoleStage.REGISTERED_SCALE
    assert "kernel replica provenance does not match the manifest" in certificate.blockers


def test_dispersion_action_hash_mismatch_cannot_complete_mapping() -> None:
    kernels = _kernel_replicas()
    dispersions = tuple(
        _dispersion(
            kernel,
            action_sha256=("d" * 64 if kernel.replica_id == "baseline" else ACTION_HASH),
        )
        for kernel in kernels
    )
    certificate = _certificate(kernels, dispersions=dispersions)

    assert certificate.positive_residue_control_pass
    assert not certificate.dispersion_control_pass
    assert "dispersion provenance does not match the kernel manifest" in certificate.blockers


@pytest.mark.parametrize("value", [True, 0.0, math.nan, math.inf])
def test_registered_scale_rejects_nonphysical_values(value: object) -> None:
    with pytest.raises(ValueError):
        renormalized_pole_certificate(
            registered_inverse_correlation_scale_mev=value,  # type: ignore[arg-type]
        )


def test_kernel_replica_rejects_nonfinite_or_unsorted_numerical_data() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(_kernel_replicas()[0], gauge_parameter=math.nan)
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            _kernel_replicas()[0],
            invariant_s_mev2=(3.0, 5.0, 4.0),
        )
    with pytest.raises(ValueError, match="matching lengths"):
        replace(
            _kernel_replicas()[0],
            kernel_imag_mev2=(0.0, 0.0, 0.0, 0.0),
        )


def test_manifest_requires_concrete_sha256_provenance() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_manifest(), action_sha256="not-a-digest")
