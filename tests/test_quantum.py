"""Quantum phase evolution tests (12_Equation 1.5/4.1/4.7)."""

import torch
import pytest
from clarus.quantum import (
    quantum_phase_step, wick_rotate, quantum_to_real,
    check_norm_conservation, convergence_inequality,
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
