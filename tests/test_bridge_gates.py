"""AGI bridge gate measurement regression tests.

Covers gates F2 (ISS ball, docs/7_AGI/12_Equation.md A.1) and F3
(ergodic KL, A.3). F1/F4 are experiment-level and not unit-tested here.
"""

from __future__ import annotations

import math

import pytest
import torch

from clarus.ce_ops import relax
from clarus.quantum import (
    ALPHA_B_DEFAULT,
    estimate_mu,
    iss_ball_radius,
    iss_report,
    pci_regression,
    time_curvature,
)
from clarus.constants import ACTIVE_RATIO
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


def _negdef_weight(n: int = 32, *, scale: float = 0.05, margin: float = 0.1) -> torch.Tensor:
    torch.manual_seed(0)
    w = torch.randn(n, n) * scale
    w = 0.5 * (w + w.T)
    eig_max = float(torch.linalg.eigvalsh(w).max().item())
    return w - (eig_max + margin) * torch.eye(n)


class TestEstimateMu:
    def test_recovers_known_contraction(self) -> None:
        torch.manual_seed(1)
        n = 16
        m_star = torch.randn(n)
        mu_true = 0.4
        dt_over_tau = 0.05
        m = m_star + torch.randn(n) * 0.6
        residuals = []
        for _ in range(400):
            residuals.append(float((m - m_star).norm().item()))
            m = m + dt_over_tau * (-mu_true * (m - m_star))

        mu_hat = estimate_mu(residuals, dt_over_tau=dt_over_tau)

        assert math.isfinite(mu_hat)
        assert abs(mu_hat - mu_true) / mu_true < 0.05

    def test_zero_when_too_short(self) -> None:
        assert estimate_mu([1.0, 0.5], dt_over_tau=0.1) == 0.0

    def test_zero_when_expanding(self) -> None:
        assert estimate_mu([0.1, 0.2, 0.4, 0.8], dt_over_tau=0.1) == 0.0

    def test_rejects_nonpositive_dt(self) -> None:
        with pytest.raises(ValueError):
            estimate_mu([1.0, 0.5, 0.25], dt_over_tau=0.0)


class TestIssBallRadius:
    def test_closed_form(self) -> None:
        r = iss_ball_radius(c_k_max=0.2, phi_inf_norm=0.5, mu=0.4, alpha_b=ALPHA_B_DEFAULT)
        expected = 0.2 * 0.5 / (0.4 * ALPHA_B_DEFAULT)
        assert math.isclose(r, expected, rel_tol=1e-12)

    def test_inf_when_mu_zero(self) -> None:
        assert math.isinf(iss_ball_radius(c_k_max=1.0, phi_inf_norm=1.0, mu=0.0))

    def test_alpha_b_value(self) -> None:
        assert math.isclose(ALPHA_B_DEFAULT, math.exp(1.0 / 3.0) * math.pi ** (1.0 / 3.0))


class TestTimeCurvature:
    def test_zero_when_collinear(self) -> None:
        m = [torch.tensor([1.0, 2.0, 3.0]) * k for k in range(3)]
        assert time_curvature(m) == 0.0

    def test_nonzero_on_acceleration(self) -> None:
        m_hist = [torch.zeros(3), torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.5, 0.0, 0.0])]
        assert time_curvature(m_hist) > 0.0

    def test_zero_when_short(self) -> None:
        assert time_curvature([torch.zeros(3), torch.zeros(3)]) == 0.0


class TestIssReport:
    def test_full_pipeline(self) -> None:
        torch.manual_seed(2)
        n = 16
        m_star = torch.randn(n)
        mu_true = 0.3
        dt_over_tau = 0.1
        m = m_star + torch.randn(n) * 0.5
        hist = []
        for _ in range(200):
            hist.append(m.clone())
            m = m + dt_over_tau * (-mu_true * (m - m_star)) + 1e-3 * torch.randn(n)

        report = iss_report(hist, torch.randn(n) * 0.2, dt_over_tau=dt_over_tau, m_star=m_star)

        assert report["samples"] == 200
        assert report["mu"] > 0.0
        assert math.isfinite(report["iss_ball_radius"])

    def test_empty_history(self) -> None:
        report = iss_report([], torch.zeros(4), dt_over_tau=0.1)
        assert report["samples"] == 0
        assert math.isinf(report["iss_ball_radius"])


class TestRelaxIssIntegration:
    def test_relax_emits_iss(self) -> None:
        n = 32
        w = _negdef_weight(n)
        torch.manual_seed(3)
        b = torch.zeros(n)
        phi = torch.randn(n) * 0.3
        m0 = torch.randn(n)

        _, hist, steps = relax(
            w, b, phi, m0,
            portal=0.03120, bypass=0.4892, t_wake=0.3148,
            beta=1.0, cb_w=0.0,
            lambda0=0.1, lambda_phi=0.0, lambda_var=0.0,
            tau=10.0, dt=0.5, max_steps=200, tol=1e-5,
            anneal_ratio=0.5, noise_scale=0.0, seed=0,
            backend="torch",
        )

        assert "iss" in hist
        iss = hist["iss"]
        assert iss["samples"] == steps
        assert iss["c_k_max"] >= 0.0
        assert iss["phi_inf_norm"] > 0.0
        assert iss["mu"] > 0.0
        assert math.isfinite(iss["iss_ball_radius"])


class TestModeOccupancyKl:
    def _build(self, n: int = 32) -> BrainRuntime:
        torch.manual_seed(4)
        cfg = BrainRuntimeConfig(dim=n, active_ratio=0.05)
        w = torch.randn(n, n) * 0.05
        return BrainRuntime(w, config=cfg, backend="torch")

    def test_empty_runtime_has_nan_kl(self) -> None:
        rt = self._build()
        report = rt.mode_occupancy_kl()
        assert report["samples"] == 0
        assert math.isnan(report["kl_to_p_star"])

    def test_kl_nonnegative_after_steps(self) -> None:
        rt = self._build()
        for i in range(40):
            ext = torch.randn(rt.config.dim) * (0.4 if i < 25 else 0.0)
            rt.step(external_input=ext)
        report = rt.mode_occupancy_kl()
        assert report["samples"] == 40
        assert report["kl_to_p_star"] >= 0.0
        assert math.isclose(
            report["pi_wake"] + report["pi_nrem"] + report["pi_rem"], 1.0, abs_tol=1e-9
        )

    def test_reset_clears_counter(self) -> None:
        rt = self._build()
        for _ in range(10):
            rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        rt.reset_mode_occupancy()
        assert sum(rt.mode_occupancy.values()) == 0

    def test_snapshot_roundtrip_preserves_kl(self) -> None:
        rt = self._build()
        for _ in range(20):
            rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        before = rt.mode_occupancy_kl()
        rt2 = BrainRuntime.from_snapshot(rt.snapshot(), backend="torch")
        after = rt2.mode_occupancy_kl()
        assert before == after

    def test_bridge_gate_report_shape(self) -> None:
        rt = self._build()
        rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        report = rt.bridge_gate_report()
        assert set(report.keys()) == {
            "F1_self_organization",
            "F2_iss_ball",
            "F3_ergodic_kl",
            "F4_pci_regression",
        }
        assert report["F3_ergodic_kl"]["samples"] == 1
        f1 = report["F1_self_organization"]
        assert math.isclose(f1["active_ratio_target"], ACTIVE_RATIO)
        assert "active_ratio_ema" in f1


class TestF1SelfMeasure:
    def _build(self, **overrides) -> BrainRuntime:
        torch.manual_seed(7)
        n = 64
        cfg_kwargs = dict(dim=n, active_ratio=0.30)
        cfg_kwargs.update(overrides)
        cfg = BrainRuntimeConfig(**cfg_kwargs)
        w = torch.randn(n, n) * 0.05
        return BrainRuntime(w, config=cfg, backend="torch")

    def test_off_keeps_static_budget(self) -> None:
        rt = self._build(f1_self_measure=False)
        for _ in range(20):
            rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        assert rt.active_ratio_ema == pytest.approx(0.30, abs=0.30)
        assert rt.config.f1_self_measure is False

    def test_on_pulls_ema_toward_target(self) -> None:
        rt = self._build(f1_self_measure=True, f1_pull_strength=0.9, f1_ema_alpha=0.5)
        initial_ema = rt.active_ratio_ema
        for _ in range(80):
            rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        assert abs(rt.active_ratio_ema - ACTIVE_RATIO) < abs(initial_ema - ACTIVE_RATIO)

    def test_clamp_min_ratio_holds(self) -> None:
        rt = self._build(
            f1_self_measure=True,
            f1_pull_strength=1.0,
            f1_min_ratio=0.02,
            f1_max_ratio=0.5,
        )
        for _ in range(10):
            rt.step(external_input=torch.zeros(rt.config.dim))
        budget = rt._f1_effective_budget(RuntimeMode.WAKE)
        assert budget >= max(1, int(round(rt.config.dim * 0.02)))

    def test_snapshot_preserves_ema(self) -> None:
        rt = self._build(f1_self_measure=True, f1_pull_strength=0.5, f1_ema_alpha=0.3)
        for _ in range(15):
            rt.step(external_input=torch.randn(rt.config.dim) * 0.4)
        before = rt.active_ratio_ema
        rt2 = BrainRuntime.from_snapshot(rt.snapshot(), backend="torch")
        assert rt2.active_ratio_ema == before


class TestPciRegression:
    def test_recovers_known_line(self) -> None:
        alpha, beta = 0.7, 0.05
        stability = [i / 20.0 for i in range(20)]
        pci = [alpha * s + beta for s in stability]
        result = pci_regression(stability, pci)
        assert result["n"] == 20
        assert math.isclose(result["alpha"], alpha, rel_tol=1e-9)
        assert math.isclose(result["beta"], beta, abs_tol=1e-9)
        assert math.isclose(result["r2"], 1.0, abs_tol=1e-9)
        assert math.isclose(result["pearson_r"], 1.0, abs_tol=1e-9)

    def test_too_few_samples(self) -> None:
        result = pci_regression([0.1, 0.2], [0.5, 0.6])
        assert result["n"] == 2
        assert math.isnan(result["alpha"])
        assert result["r2"] == 0.0

    def test_constant_stability(self) -> None:
        result = pci_regression([0.3] * 5, [0.5, 0.4, 0.6, 0.55, 0.45])
        assert math.isnan(result["alpha"])
        assert result["r2"] == 0.0

    def test_noisy_recovery_above_threshold(self) -> None:
        torch.manual_seed(11)
        stability = [i / 50.0 for i in range(50)]
        noise = (torch.randn(50) * 0.02).tolist()
        pci = [0.6 * s + 0.1 + n for s, n in zip(stability, noise)]
        result = pci_regression(stability, pci)
        assert result["r2"] > 0.7
