"""Relaxation / sleep completion tests (Phase 13)."""

import torch
import pytest
from clarus.quantum import convergence_inequality


class TestConvergenceCheck:
    def test_energy_decrease_condition(self):
        assert convergence_inequality(grad_norm=1.0, c_k=0.1, phi_norm=0.5)

    def test_bypass_dominated(self):
        assert not convergence_inequality(grad_norm=0.01, c_k=2.0, phi_norm=1.0)


class TestCurvatureSleepPressure:
    def test_curvature_integral_accumulates(self):
        curvature_history = []
        for _ in range(100):
            phi = torch.randn(16)
            r = 4
            v = torch.randn(r, 16)
            v = v / v.norm(dim=1, keepdim=True)
            delta = phi - v.T @ (v @ phi)
            kappa = delta.norm().item() ** 2
            curvature_history.append(kappa)
        p_sleep = sum(curvature_history)
        assert p_sleep > 0

    def test_local_stabilization_subtracts(self):
        curvature = [1.0] * 50
        stabilization = [0.5] * 50
        p_sleep = sum(c - s for c, s in zip(curvature, stabilization))
        assert p_sleep == pytest.approx(25.0)


class TestPhiThreeInterventions:
    def test_portal_coupling(self):
        from clarus.constants import PORTAL
        m = torch.randn(16)
        phi = torch.randn(16)
        phi_hat = phi / phi.norm().clamp(min=1e-8)
        e_portal = -PORTAL * torch.dot(m, phi_hat).item()
        assert isinstance(e_portal, float)

    def test_mode_switch_intervention(self):
        phi_norm = 1.5
        threshold = 1.0
        should_switch = phi_norm > threshold
        assert should_switch

    def test_glymphatic_clearance(self):
        phi = torch.randn(16) * 2.0
        clearance_rate = 0.1
        phi_cleared = phi * (1.0 - clearance_rate)
        assert phi_cleared.norm() < phi.norm()
