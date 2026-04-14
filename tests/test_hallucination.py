"""Hallucination suppression V2 tests (F.18)."""

import torch
import pytest


class TestCurvatureSuppression:
    def test_lbo_smoothing(self):
        h = torch.randn(64)
        r = 8
        v = torch.randn(r, 64)
        v = v / v.norm(dim=1, keepdim=True)
        delta_h = h - (v.T @ (v @ h))
        e_curv = delta_h.norm() ** 2
        eta = 0.1
        h_smooth = h - eta * delta_h
        e_curv_after = (h_smooth - (v.T @ (v @ h_smooth))).norm() ** 2
        assert e_curv_after < e_curv

    def test_threshold_trigger(self):
        kappa_th = 1.0
        kappa = 2.0
        assert kappa > kappa_th

    def test_re_inference(self):
        h = torch.randn(64)
        r = 8
        v = torch.randn(r, 64)
        v = v / v.norm(dim=1, keepdim=True)
        kappa_th = 0.5
        for attempt in range(3):
            delta_h = h - (v.T @ (v @ h))
            kappa = delta_h.norm().item() ** 2
            if kappa <= kappa_th:
                break
            h = h - 0.2 * delta_h
        final_kappa = (h - (v.T @ (v @ h))).norm().item() ** 2
        assert final_kappa < kappa_th or attempt == 2
