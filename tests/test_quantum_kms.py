"""Separate scalar-action and bath KMS/Kossakowski reference gates."""

import math

import pytest
import torch

from reality_stone.clarus.quantum import (
    BOLTZMANN_CONSTANT_J_K,
    HBAR_J_S,
    IndependentScalarFieldAction,
    ThermalTransitionRates,
    check_scalar_kms_spectral_pair,
    correlator_fourier_component,
    gksl_rate_from_si_correlation_spectrum,
    kms_detailed_balance_factor,
    kms_thermal_transition_rates_from_reduced_spectrum,
    kossakowski_jump_decomposition,
    lindblad_rhs,
    validate_kossakowski_matrix,
)


def _adjoint(value: torch.Tensor) -> torch.Tensor:
    return value.conj().transpose(-2, -1)


def _direct_kossakowski_dissipator(
    rho: torch.Tensor,
    basis: tuple[torch.Tensor, ...],
    coefficients: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(rho)
    for a, left in enumerate(basis):
        for b, right in enumerate(basis):
            gram = _adjoint(right) @ left
            result = result + coefficients[a, b] * (
                left @ rho @ _adjoint(right)
                - 0.5 * (gram @ rho + rho @ gram)
            )
    return result


class TestIndependentScalarAction:
    def test_fixed_action_convention_closes_the_local_eom_term(self):
        action = IndependentScalarFieldAction(
            mass_squared=4.0,
            curvature_coupling=0.25,
            quartic_coupling=6.0,
        )

        assert action.potential_density(2.0, 8.0) == pytest.approx(16.0)
        assert action.local_eom_term(2.0, 8.0) == pytest.approx(20.0)
        assert action.phi_zero_minkowski_angular_frequency_squared(
            3.0
        ) == pytest.approx(13.0)
        assert action.potential_is_bounded_below_at_curvature(8.0)

    def test_stabilized_negative_mass_squared_is_explicit(self):
        action = IndependentScalarFieldAction(
            mass_squared=-4.0,
            quartic_coupling=6.0,
        )

        assert action.phi_zero_minkowski_angular_frequency_squared(0.0) == -4.0
        assert action.potential_density(2.0, 0.0) == pytest.approx(-4.0)

    def test_curvature_boundedness_is_a_separate_gate(self):
        action = IndependentScalarFieldAction(
            mass_squared=1.0,
            curvature_coupling=1.0,
        )

        assert action.potential_is_bounded_below_at_curvature(0.0)
        assert not action.potential_is_bounded_below_at_curvature(-2.0)
        assert action.potential_density(1_000.0, -2.0) == pytest.approx(
            -500_000.0
        )

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"mass_squared": -1.0}, "requires positive"),
            ({"mass_squared": float("nan")}, "finite"),
            (
                {"mass_squared": 1.0, "quartic_coupling": -1.0},
                "nonnegative",
            ),
            (
                {"mass_squared": 1.0, "curvature_coupling": float("inf")},
                "finite",
            ),
        ],
    )
    def test_action_rejects_invalid_parameters(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            IndependentScalarFieldAction(**kwargs)


class TestCorrelatorAndKMS:
    def test_two_sided_exponential_correlator_has_lorentzian_transform(self):
        times = torch.linspace(-20.0, 20.0, 8_001, dtype=torch.float64)
        correlator = torch.exp(-times.abs()).to(torch.complex128)
        omega = 1.3

        component = correlator_fourier_component(times, correlator, omega)

        assert component.real.item() == pytest.approx(
            2.0 / (1.0 + omega * omega),
            rel=2e-5,
        )
        assert component.imag.item() == pytest.approx(0.0, abs=1e-12)

    def test_correlator_transform_validates_grid_and_does_not_clamp_output(self):
        times = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        negative_correlator = -torch.ones(3, dtype=torch.complex128)

        component = correlator_fourier_component(
            times,
            negative_correlator,
            0.0,
        )

        assert component.real.item() == pytest.approx(-2.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            correlator_fourier_component(
                torch.tensor([0.0, 1.0, 1.0]),
                torch.ones(3),
                1.0,
            )
        with pytest.raises(ValueError, match="negative and positive"):
            correlator_fourier_component(
                torch.tensor([0.0, 1.0, 2.0]),
                torch.ones(3),
                1.0,
            )

    def test_fourier_sign_and_kms_ratio_close_end_to_end_for_scalar_fixture(self):
        omega = 1.0
        temperature = HBAR_J_S / (BOLTZMANN_CONSTANT_J_K * math.log(4.0))
        factor = 0.25
        times = torch.linspace(
            -20.0 * math.pi,
            20.0 * math.pi,
            40_001,
            dtype=torch.float64,
        )
        correlator = (
            torch.exp(-1j * omega * times)
            + factor * torch.exp(1j * omega * times)
        )

        positive = correlator_fourier_component(
            times,
            correlator,
            omega,
        )
        negative = correlator_fourier_component(
            times,
            correlator,
            -omega,
        )

        assert positive.real.item() == pytest.approx(40.0 * math.pi, rel=1e-12)
        assert negative.real.item() / positive.real.item() == pytest.approx(
            factor,
            rel=1e-12,
        )
        assert abs(positive.imag.item()) < 1e-12
        assert abs(negative.imag.item()) < 1e-12
        assert check_scalar_kms_spectral_pair(
            positive.real.item(),
            negative.real.item(),
            omega,
            temperature,
            rtol=1e-12,
        )

    def test_kms_pair_and_rate_completion_use_one_frequency_convention(self):
        omega = 2.0e10
        temperature = 0.1
        positive_spectrum = 3.0
        factor = math.exp(
            -HBAR_J_S * omega / (BOLTZMANN_CONSTANT_J_K * temperature)
        )

        assert kms_detailed_balance_factor(omega, temperature) == pytest.approx(
            factor,
            rel=1e-15,
        )
        assert check_scalar_kms_spectral_pair(
            positive_spectrum,
            positive_spectrum * factor,
            omega,
            temperature,
        )
        assert not check_scalar_kms_spectral_pair(
            positive_spectrum,
            positive_spectrum / factor,
            omega,
            temperature,
        )

        rates = kms_thermal_transition_rates_from_reduced_spectrum(
            coupling=0.2,
            positive_frequency_reduced_spectrum=positive_spectrum,
            angular_frequency_s_inv=omega,
            temperature_kelvin=temperature,
        )
        assert rates.downward_s_inv == pytest.approx(0.12)
        assert rates.upward_s_inv / rates.downward_s_inv == pytest.approx(
            factor,
            rel=1e-15,
        )

    def test_si_hamiltonian_rate_includes_hbar_squared(self):
        coupling_energy = 0.25 * HBAR_J_S
        spectrum_seconds = 8.0

        assert gksl_rate_from_si_correlation_spectrum(
            coupling_energy,
            spectrum_seconds,
        ) == pytest.approx(0.5)

    def test_scalar_kms_zero_and_low_temperature_do_not_false_pass(self):
        assert not check_scalar_kms_spectral_pair(
            0.0,
            5e-13,
            1.0,
            1.0,
        )
        assert kms_detailed_balance_factor(1e16, 1e-6) == 0.0
        assert check_scalar_kms_spectral_pair(
            2.0,
            0.0,
            1e16,
            1e-6,
        )
        assert not check_scalar_kms_spectral_pair(
            2.0,
            1e-300,
            1e16,
            1e-6,
        )

    def test_public_thermal_rate_record_rejects_non_kms_values(self):
        with pytest.raises(ValueError, match="must equal"):
            ThermalTransitionRates(
                downward_s_inv=1.0,
                upward_s_inv=0.3,
                detailed_balance_factor=0.5,
            )

    @pytest.mark.parametrize(
        ("omega", "temperature", "error"),
        [
            (-1.0, 1.0, "nonnegative"),
            (1.0, 0.0, "positive"),
            (1.0, -1.0, "positive"),
        ],
    )
    def test_kms_factor_rejects_nonphysical_inputs(
        self,
        omega,
        temperature,
        error,
    ):
        with pytest.raises(ValueError, match=error):
            kms_detailed_balance_factor(omega, temperature)

    def test_kms_rates_make_the_two_level_gibbs_state_stationary(self):
        omega = 2.0e10
        temperature = 0.1
        rates = kms_thermal_transition_rates_from_reduced_spectrum(
            coupling=0.2,
            positive_frequency_reduced_spectrum=3.0,
            angular_frequency_s_inv=omega,
            temperature_kelvin=temperature,
        )
        boltzmann = rates.detailed_balance_factor
        gibbs_state = torch.diag(
            torch.tensor(
                [1.0 / (1.0 + boltzmann), boltzmann / (1.0 + boltzmann)],
                dtype=torch.complex128,
            )
        )
        hamiltonian = torch.diag(
            torch.tensor([0.0, HBAR_J_S * omega], dtype=torch.complex128)
        )
        lowering = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex128,
        )
        raising = _adjoint(lowering)

        derivative = lindblad_rhs(
            gibbs_state,
            hamiltonian,
            [lowering, raising],
            rates=[rates.downward_s_inv, rates.upward_s_inv],
            hbar=HBAR_J_S,
        )

        assert torch.allclose(
            derivative,
            torch.zeros_like(derivative),
            atol=1e-14,
            rtol=0.0,
        )


class TestKossakowskiGate:
    def test_psd_matrix_decomposition_matches_cross_term_dissipator(self):
        sigma_x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.complex128,
        )
        sigma_y = torch.tensor(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=torch.complex128,
        )
        basis = (sigma_x, sigma_y)
        coefficients = torch.tensor(
            [[2.0, 0.0 + 0.3j], [0.0 - 0.3j, 1.0]],
            dtype=torch.complex128,
        )
        rho = torch.tensor(
            [[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]],
            dtype=torch.complex128,
        )

        decomposition = kossakowski_jump_decomposition(
            basis,
            coefficients,
        )
        decomposed = lindblad_rhs(
            rho,
            torch.zeros((2, 2), dtype=torch.complex128),
            decomposition.jump_operators,
            rates=decomposition.rates,
        )
        direct = _direct_kossakowski_dissipator(rho, basis, coefficients)

        assert len(decomposition.rates) == 2
        assert min(decomposition.rates) > 0.0
        assert decomposition.input_residual_frobenius < 1e-12
        assert torch.allclose(
            decomposition.represented_matrix,
            coefficients,
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(decomposed, direct, atol=1e-12, rtol=1e-12)

    def test_non_psd_or_nonhermitian_matrix_is_rejected(self):
        non_psd = torch.tensor(
            [[1.0, 2.0], [2.0, 1.0]],
            dtype=torch.float64,
        )
        nonhermitian = torch.tensor(
            [[1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )

        with pytest.raises(ValueError, match="positive semidefinite"):
            validate_kossakowski_matrix(non_psd)
        with pytest.raises(ValueError, match="Hermitian"):
            validate_kossakowski_matrix(nonhermitian)
        with pytest.raises(ValueError, match="nonempty"):
            validate_kossakowski_matrix(torch.empty((0, 0)))

        large_nonhermitian = torch.tensor(
            [[2e12, 1e12], [1e12 + 0.5, 2e12]],
            dtype=torch.float64,
        )
        with pytest.raises(ValueError, match="Hermitian"):
            validate_kossakowski_matrix(large_nonhermitian)

    def test_tiny_positive_modes_are_kept_unless_cutoff_is_explicit(self):
        identity = torch.eye(2, dtype=torch.complex128)
        tiny = 1e-12

        represented = kossakowski_jump_decomposition(
            [identity],
            torch.tensor([[tiny]], dtype=torch.float64),
        )
        truncated = kossakowski_jump_decomposition(
            [identity],
            torch.tensor([[tiny]], dtype=torch.float64),
            mode_cutoff=1e-10,
        )

        assert represented.rates == pytest.approx((tiny,))
        assert represented.input_residual_frobenius < 1e-20
        assert truncated.rates == ()
        assert truncated.input_residual_frobenius == pytest.approx(tiny)
        assert truncated.mode_cutoff == pytest.approx(1e-10)

    def test_near_negative_mode_requires_explicit_projection_tolerance(self):
        identity = torch.eye(2, dtype=torch.complex128)
        negative = torch.tensor([[-5e-11]], dtype=torch.float64)

        with pytest.raises(ValueError, match="positive semidefinite"):
            kossakowski_jump_decomposition([identity], negative)

        projected = kossakowski_jump_decomposition(
            [identity],
            negative,
            psd_atol=1e-10,
        )

        assert projected.rates == ()
        assert projected.minimum_input_eigenvalue == pytest.approx(-5e-11)
        assert projected.psd_acceptance_tolerance == pytest.approx(1e-10)
        assert projected.input_residual_frobenius == pytest.approx(5e-11)
        assert torch.equal(
            projected.represented_matrix,
            torch.zeros_like(projected.represented_matrix),
        )

    def test_exact_zero_modes_are_removed_without_inventing_rates(self):
        identity = torch.eye(2, dtype=torch.complex128)
        decomposition = kossakowski_jump_decomposition(
            [identity, identity],
            torch.zeros((2, 2), dtype=torch.float64),
        )

        assert decomposition.jump_operators == ()
        assert decomposition.rates == ()
        assert decomposition.input_residual_frobenius == 0.0
