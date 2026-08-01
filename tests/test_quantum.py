"""Legacy phase-toy tests plus standard finite-dimensional QM baselines."""

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from reality_stone.clarus.quantum import (
    REFERENCE_SCALAR_MASS_MEV,
    ScalarFieldMassGap,
    born_probabilities,
    check_density_matrix,
    check_norm_conservation,
    convergence_inequality,
    density_matrix,
    gksl_rate_from_spectral_density,
    jacobi_rayleigh_scalar,
    lindblad_rhs,
    lindblad_step,
    quantum_phase_step,
    quantum_to_real,
    sample_born,
    unitary_step,
    wick_rotate,
)


class TestQuantumPhase:
    def test_norm_conservation(self):
        psi = torch.randn(16) + 0j
        psi = psi / psi.abs().norm()
        psi_new = quantum_phase_step(psi, energy=1.0, dt=0.01)
        assert check_norm_conservation(psi, psi_new)

    def test_phase_rotation(self):
        psi = torch.ones(4, dtype=torch.complex64)
        psi_new = quantum_phase_step(psi, energy=1.0, dt=0.01)
        assert not torch.allclose(psi.real, psi_new.real)

    def test_zero_energy_no_change(self):
        psi = torch.randn(8, dtype=torch.complex64)
        psi_new = quantum_phase_step(psi, energy=0.0, dt=0.01)
        assert torch.allclose(psi, psi_new)

    def test_real_input_becomes_complex(self):
        psi = torch.randn(8)
        psi_new = quantum_phase_step(psi, energy=1.0)
        assert psi_new.is_complex()

    def test_scalar_energy_is_only_a_global_phase(self):
        psi = torch.tensor([0.8 + 0.0j, 0.0 + 0.6j], dtype=torch.complex128)

        psi_new = quantum_phase_step(psi, energy=2.0, dt=0.3)

        assert torch.allclose(born_probabilities(psi_new), born_probabilities(psi))
        assert torch.allclose(psi_new[1] / psi_new[0], psi[1] / psi[0])


class TestIndependentScalarFieldInputs:
    def test_rayleigh_readout_is_real_and_probe_scale_invariant(self):
        operator = torch.diag(
            torch.tensor([2.0, 8.0], dtype=torch.complex128)
        )
        probe = torch.tensor([1.0, 1.0j], dtype=torch.complex128)

        readout = jacobi_rayleigh_scalar(operator, probe)
        scaled_readout = jacobi_rayleigh_scalar(
            operator,
            (3.0 - 4.0j) * probe,
        )

        assert readout.ndim == 0
        assert not readout.is_complex()
        assert readout.item() == pytest.approx(5.0, abs=1e-12)
        assert scaled_readout.item() == pytest.approx(
            readout.item(),
            abs=1e-12,
        )

    def test_rayleigh_readout_validates_types_shapes_and_operator(self):
        operator = torch.eye(2, dtype=torch.complex128)
        probe = torch.ones(2, dtype=torch.complex128)

        with pytest.raises(TypeError, match="jacobi_operator"):
            jacobi_rayleigh_scalar([[1.0]], probe)
        with pytest.raises(TypeError, match="probe"):
            jacobi_rayleigh_scalar(operator, [1.0, 1.0])
        with pytest.raises(ValueError, match="one-dimensional"):
            jacobi_rayleigh_scalar(operator, probe[:, None])
        with pytest.raises(ValueError, match="square"):
            jacobi_rayleigh_scalar(torch.ones((2, 3)), probe)
        with pytest.raises(ValueError, match="dimension"):
            jacobi_rayleigh_scalar(operator, torch.ones(3))
        with pytest.raises(ValueError, match="nonzero"):
            jacobi_rayleigh_scalar(operator, torch.zeros(2))

        nonhermitian = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex128,
        )
        with pytest.raises(ValueError, match="Hermitian"):
            jacobi_rayleigh_scalar(nonhermitian, probe)

    def test_reference_mass_gap_has_fixed_si_conversions_and_round_trips(self):
        gap = ScalarFieldMassGap(REFERENCE_SCALAR_MASS_MEV)

        assert gap.mass_energy_mev == pytest.approx(29.64757, abs=0.0)
        assert gap.energy_joule == pytest.approx(
            4.750064390887938e-12,
            rel=1e-14,
        )
        assert gap.frequency_hz == pytest.approx(
            7.168750531395956e21,
            rel=1e-14,
        )
        assert gap.angular_gap_s_inv == pytest.approx(
            4.504258800970292e22,
            rel=1e-14,
        )
        assert gap.angular_gap_s_inv == pytest.approx(
            2.0 * math.pi * gap.frequency_hz,
            rel=1e-14,
        )
        assert ScalarFieldMassGap.from_frequency_hz(
            gap.frequency_hz
        ).mass_energy_mev == pytest.approx(gap.mass_energy_mev, rel=1e-14)
        assert ScalarFieldMassGap.from_angular_gap_s_inv(
            gap.angular_gap_s_inv
        ).mass_energy_mev == pytest.approx(gap.mass_energy_mev, rel=1e-14)

        with pytest.raises(FrozenInstanceError):
            gap.mass_energy_mev = 1.0

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_mass_gap_rejects_invalid_values(self, value):
        with pytest.raises(ValueError):
            ScalarFieldMassGap(value)

    def test_gksl_rate_requires_nonnegative_spectral_density(self):
        assert gksl_rate_from_spectral_density(-0.4, 2.5) == pytest.approx(0.4)
        assert gksl_rate_from_spectral_density(0.4, 2.5) == pytest.approx(0.4)

        with pytest.raises(ValueError, match="nonnegative"):
            gksl_rate_from_spectral_density(0.4, -0.1)
        with pytest.raises(ValueError, match="finite"):
            gksl_rate_from_spectral_density(float("nan"), 1.0)
        with pytest.raises(TypeError, match="real"):
            gksl_rate_from_spectral_density(True, 1.0)

    def test_nonnegative_gksl_rate_has_ancilla_cp_reference(self):
        coupling = 0.4
        spectral_density = 2.5
        rate = gksl_rate_from_spectral_density(coupling, spectral_density)
        bell = torch.tensor(
            [1.0, 0.0, 0.0, 1.0],
            dtype=torch.complex128,
        ) / math.sqrt(2.0)
        choi_state = density_matrix(bell)
        zero_hamiltonian = torch.zeros((4, 4), dtype=torch.complex128)
        sigma_z = torch.diag(
            torch.tensor([1.0, -1.0], dtype=torch.complex128)
        )
        local_jump = torch.kron(sigma_z, torch.eye(2, dtype=torch.complex128))
        dt = 0.6

        evolved_choi = lindblad_step(
            choi_state,
            zero_hamiltonian,
            dt,
            [local_jump],
            rates=[rate],
        )

        eigenvalues = torch.linalg.eigvalsh(evolved_choi)
        assert check_density_matrix(evolved_choi, tol=1e-10)
        assert eigenvalues.min().item() >= -1e-12
        assert torch.trace(evolved_choi).real.item() == pytest.approx(
            1.0,
            abs=1e-12,
        )
        assert evolved_choi[0, 3].real.item() == pytest.approx(
            0.5 * math.exp(-2.0 * rate * dt),
            abs=1e-10,
        )


class TestStandardQuantumReference:
    def test_nontrivial_hamiltonian_changes_population_and_preserves_norm(self):
        psi = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        sigma_x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.complex128,
        )

        psi_new = unitary_step(psi, sigma_x, dt=math.pi / 2.0)

        assert check_norm_conservation(psi, psi_new, tol=1e-10)
        assert torch.allclose(
            born_probabilities(psi_new),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            atol=1e-10,
        )

    def test_diagonal_hamiltonian_changes_relative_phase(self):
        psi = torch.tensor([1.0, 1.0], dtype=torch.complex128) / math.sqrt(2.0)
        sigma_z = torch.diag(
            torch.tensor([1.0, -1.0], dtype=torch.complex128)
        )

        psi_new = unitary_step(psi, sigma_z, dt=math.pi / 4.0)

        assert torch.allclose(born_probabilities(psi_new), born_probabilities(psi))
        assert not torch.allclose(psi_new[1] / psi_new[0], psi[1] / psi[0])

    def test_nonhermitian_hamiltonian_is_rejected(self):
        psi = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        nonhermitian = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex64,
        )

        with pytest.raises(ValueError, match="Hermitian"):
            unitary_step(psi, nonhermitian, dt=0.1)

    def test_density_matrix_and_born_probabilities(self):
        psi = torch.tensor(
            [math.sqrt(0.75), 1j * math.sqrt(0.25)],
            dtype=torch.complex128,
        )

        rho = density_matrix(psi)

        assert check_density_matrix(rho, tol=1e-10)
        assert torch.allclose(
            born_probabilities(rho),
            torch.tensor([0.75, 0.25], dtype=torch.float64),
            atol=1e-10,
        )

    def test_seeded_born_sampling_is_reproducible_and_calibrated(self):
        psi = torch.tensor(
            [math.sqrt(0.75), math.sqrt(0.25)],
            dtype=torch.complex128,
        )
        shots = 20_000

        first = sample_born(psi, shots, seed=73)
        second = sample_born(psi, shots, seed=73)
        frequencies = torch.bincount(first, minlength=2).to(torch.float64) / shots

        assert torch.equal(first, second)
        assert torch.allclose(
            frequencies,
            torch.tensor([0.75, 0.25], dtype=torch.float64),
            atol=0.015,
            rtol=0.0,
        )


class TestLindbladReference:
    def test_rhs_preserves_trace_and_hermiticity_tangents(self):
        plus = torch.tensor([1.0, 1.0], dtype=torch.complex128) / math.sqrt(2.0)
        rho = density_matrix(plus)
        zero_hamiltonian = torch.zeros((2, 2), dtype=torch.complex128)
        sigma_z = torch.diag(
            torch.tensor([1.0, -1.0], dtype=torch.complex128)
        )

        derivative = lindblad_rhs(
            rho,
            zero_hamiltonian,
            [sigma_z],
            rates=[0.5],
        )

        assert torch.allclose(
            torch.trace(derivative),
            torch.zeros((), dtype=torch.complex128),
            atol=1e-12,
        )
        assert torch.allclose(
            derivative,
            derivative.conj().transpose(-2, -1),
            atol=1e-12,
        )

    def test_dephasing_step_preserves_density_matrix_and_damps_only_coherence(self):
        plus = torch.tensor([1.0, 1.0], dtype=torch.complex128) / math.sqrt(2.0)
        rho = density_matrix(plus)
        zero_hamiltonian = torch.zeros((2, 2), dtype=torch.complex128)
        sigma_z = torch.diag(
            torch.tensor([1.0, -1.0], dtype=torch.complex128)
        )
        dt = 0.7

        evolved = lindblad_step(
            rho,
            zero_hamiltonian,
            dt,
            [sigma_z],
            rates=[0.5],
        )

        expected_coherence = 0.5 * math.exp(-dt)
        assert check_density_matrix(evolved, tol=1e-10)
        assert torch.allclose(
            torch.diagonal(evolved).real,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            atol=1e-10,
        )
        assert evolved[0, 1].real.item() == pytest.approx(
            expected_coherence,
            abs=1e-10,
        )
        assert evolved[0, 1].imag.item() == pytest.approx(0.0, abs=1e-10)


class TestWickRotation:
    def test_damping(self):
        psi = torch.ones(4, dtype=torch.complex64)
        psi_wick = wick_rotate(psi, energy=1.0, dt=0.1)
        assert psi_wick.abs().norm() < psi.abs().norm()

    def test_real_damping(self):
        psi = torch.ones(4)
        psi_wick = wick_rotate(psi, energy=1.0, dt=0.1)
        assert psi_wick.norm() < psi.norm()


class TestQuantumToReal:
    def test_complex_to_real(self):
        psi = torch.complex(torch.ones(4), torch.ones(4))
        real = quantum_to_real(psi)
        assert not real.is_complex()
        assert torch.allclose(real, torch.ones(4))

    def test_real_passthrough(self):
        psi = torch.randn(4)
        real = quantum_to_real(psi)
        assert torch.allclose(real, psi)


class TestConvergenceInequality:
    def test_satisfied(self):
        assert convergence_inequality(grad_norm=2.0, c_k=0.1, phi_norm=1.0)

    def test_not_satisfied(self):
        assert not convergence_inequality(grad_norm=0.01, c_k=1.0, phi_norm=1.0)

    def test_boundary(self):
        alpha_b = 2.044
        threshold = 0.5 * 1.0 / alpha_b
        assert not convergence_inequality(grad_norm=threshold - 0.001, c_k=0.5, phi_norm=1.0)
