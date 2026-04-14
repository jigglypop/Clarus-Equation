"""Architecture V2 tests: perturbative mixing + CFC (2_Architecture 2.3/6)."""

import torch
import pytest
from clarus.constants import CFC_XI, GAUGE_ALPHA_S, GAUGE_ALPHA_W, GAUGE_ALPHA_EM


class TestGaugeLattice:
    def test_channel_ratios(self):
        total = GAUGE_ALPHA_S + GAUGE_ALPHA_W + GAUGE_ALPHA_EM
        r3 = GAUGE_ALPHA_S / total
        r2 = GAUGE_ALPHA_W / total
        r1 = GAUGE_ALPHA_EM / total
        assert r3 > r2 > r1
        assert r3 == pytest.approx(0.741, abs=0.01)
        assert r2 == pytest.approx(0.211, abs=0.01)
        assert r1 == pytest.approx(0.049, abs=0.01)

    def test_channel_split(self):
        d = 768
        total = GAUGE_ALPHA_S + GAUGE_ALPHA_W + GAUGE_ALPHA_EM
        d3 = int(round(d * GAUGE_ALPHA_S / total))
        d2 = int(round(d * GAUGE_ALPHA_W / total))
        d1 = d - d3 - d2
        assert d3 + d2 + d1 == d
        assert d3 > d2 > d1


class TestPerturbativeMixing:
    def test_mixing_shape(self):
        d = 64
        r_m = d // 8
        u_down = torch.randn(d, r_m)
        u_up = torch.randn(d, r_m)
        x = torch.randn(d)
        mix = u_down @ (u_up.T @ x)
        assert mix.shape == (d,)

    def test_perturbation_small(self):
        d = 64
        r_m = d // 8
        u_down = torch.randn(d, r_m) * 0.01
        u_up = torch.randn(d, r_m) * 0.01
        diag_norm = torch.randn(d).norm()
        mix_norm = (u_down @ u_up.T).norm()
        ratio = mix_norm / diag_norm
        assert ratio < 0.5

    def test_ffn_with_mixing(self):
        d = 64
        total = GAUGE_ALPHA_S + GAUGE_ALPHA_W + GAUGE_ALPHA_EM
        d3 = int(round(d * GAUGE_ALPHA_S / total))
        d2 = int(round(d * GAUGE_ALPHA_W / total))
        d1 = d - d3 - d2
        x = torch.randn(d)
        x3, x2, x1 = x[:d3], x[d3:d3 + d2], x[d3 + d2:]
        t3 = torch.relu(x3)
        t2 = torch.relu(x2)
        t1 = torch.relu(x1)
        block_diag = torch.cat([t3, t2, t1])
        r_m = d // 8
        u_down = torch.randn(d, r_m) * 0.01
        u_up = torch.randn(d, r_m) * 0.01
        mix = u_down @ (u_up.T @ x)
        output = block_diag + mix
        assert output.shape == (d,)


class TestCFC:
    def test_cfc_damping(self):
        x = torch.ones(16)
        e_curv = 0.5
        coupled = x * (1.0 - CFC_XI * e_curv)
        assert coupled.norm() < x.norm()

    def test_cfc_zero_curvature(self):
        x = torch.ones(16)
        coupled = x * (1.0 - CFC_XI * 0.0)
        assert torch.allclose(coupled, x)

    def test_xi_value(self):
        import math
        assert CFC_XI == pytest.approx(GAUGE_ALPHA_S ** (1 / 3), abs=0.01)
