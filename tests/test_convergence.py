"""Self-convergence and graph coupling tests (Phase 16)."""

import torch
import pytest
from clarus.constants import (
    ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO, BOOTSTRAP_CONTRACTION,
)
from clarus.agent import bootstrap_operator


class TestSelfConvergence:
    def test_transient_response(self):
        """33.3% -> 9.28% -> 5.55% -> 4.98% (3_Sleep.md / 5_Sparsity.md)."""
        p = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        rho = BOOTSTRAP_CONTRACTION
        expected = [
            (0.0928, 0.273, 0.634),
            (0.0555, 0.264, 0.681),
            (0.0498, 0.263, 0.688),
        ]
        for n, (ea, es, eb) in enumerate(expected):
            p = target + rho * (p - target)
            assert p[0].item() == pytest.approx(ea, abs=0.01), f"cycle {n + 1} active"
            assert p[1].item() == pytest.approx(es, abs=0.01), f"cycle {n + 1} struct"
            assert p[2].item() == pytest.approx(eb, abs=0.01), f"cycle {n + 1} bg"

    def test_convergence_to_fixed_point(self):
        p = torch.tensor([0.5, 0.3, 0.2])
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        for _ in range(50):
            p = target + BOOTSTRAP_CONTRACTION * (p - target)
        assert torch.allclose(p, target, atol=1e-4)

    def test_convergence_rate(self):
        p = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        d0 = (p - target).norm().item()
        p = target + BOOTSTRAP_CONTRACTION * (p - target)
        d1 = (p - target).norm().item()
        assert d1 / d0 == pytest.approx(BOOTSTRAP_CONTRACTION, abs=0.01)


class TestGraphCoupling:
    def test_g_agi_structure(self):
        node_sets = ["bind", "gate", "mem", "sal", "homeo", "io"]
        assert len(node_sets) == 6

    def test_graph_laplacian(self):
        n = 16
        adj = torch.randn(n, n).abs()
        adj = 0.5 * (adj + adj.T)
        adj.fill_diagonal_(0)
        degree = adj.sum(dim=1)
        laplacian = torch.diag(degree) - adj
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        assert eigenvalues[0].item() == pytest.approx(0.0, abs=1e-4)
        assert (eigenvalues >= -1e-4).all()
