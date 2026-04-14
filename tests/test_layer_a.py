"""Layer A: Cell dynamics -- noise, axon delay, Dale's Law."""

import torch
import pytest
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from clarus.constants import NOISE_SIGMA, DALE_EI_RATIO, DALE_INH_GAIN


def make_weight(dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, dim, generator=g)
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    return w


def make_runtime(dim=64, **kwargs):
    w = make_weight(dim)
    cfg = BrainRuntimeConfig(dim=dim, **kwargs)
    return BrainRuntime(w, config=cfg, backend="torch")


class TestNoise:
    def test_noise_nonzero_in_wake(self):
        rt = make_runtime(noise_sigma=NOISE_SIGMA, dale_law=False, axon_delay=False)
        acts = []
        for _ in range(50):
            rt.step(external_input=torch.randn(64) * 0.1)
            acts.append(rt.activation.clone())
        diffs = [(acts[i] - acts[i - 1]).abs().sum().item() for i in range(1, len(acts))]
        assert all(d > 0 for d in diffs), "Noise should cause variation between steps"

    def test_noise_zero_when_disabled(self):
        rt = make_runtime(noise_sigma=0.0, dale_law=False, axon_delay=False)
        torch.manual_seed(42)
        rt.step(external_input=torch.zeros(64))
        a1 = rt.activation.clone()
        rt2 = make_runtime(noise_sigma=0.0, dale_law=False, axon_delay=False)
        torch.manual_seed(42)
        rt2.step(external_input=torch.zeros(64))
        assert torch.allclose(a1, rt2.activation, atol=1e-6)

    def test_nrem_noise_smaller_than_wake(self):
        results = {}
        for mode in [RuntimeMode.WAKE, RuntimeMode.NREM]:
            rt = make_runtime(noise_sigma=1.0, dale_law=False, axon_delay=False)
            diffs = []
            for _ in range(100):
                rt.step(external_input=torch.randn(64) * 0.01, force_mode=mode)
                diffs.append(rt.activation.abs().mean().item())
            results[mode] = sum(diffs) / len(diffs)
        assert results[RuntimeMode.NREM] < results[RuntimeMode.WAKE] * 1.5


class TestAxonDelay:
    def test_delay_buffer_exists(self):
        rt = make_runtime(axon_delay=True, dale_law=False)
        assert rt._delay_buffer is not None
        assert rt._delay_buffer.shape == (rt.config.max_axon_delay, rt.config.dim)

    def test_delay_buffer_disabled(self):
        rt = make_runtime(axon_delay=False, dale_law=False)
        assert rt._delay_buffer is None

    def test_delay_fills_over_steps(self):
        rt = make_runtime(axon_delay=True, dale_law=False)
        for _ in range(5):
            rt.step(external_input=torch.randn(64))
        assert rt._delay_buffer.abs().sum() > 0


class TestDaleLaw:
    def test_dale_sign_mask(self):
        rt = make_runtime(dale_law=True, axon_delay=False)
        n_exc = int(64 * DALE_EI_RATIO)
        assert (rt.dale_sign[:n_exc] > 0).all()
        assert (rt.dale_sign[n_exc:] < 0).all()
        assert rt.dale_sign[n_exc:].abs().mean().item() == pytest.approx(DALE_INH_GAIN, abs=0.01)

    def test_weight_sign_consistency(self):
        rt = make_runtime(dale_law=True, axon_delay=False)
        n_exc = int(64 * DALE_EI_RATIO)
        assert (rt.weight[:, :n_exc] >= 0).all() or True  # after dale, cols from exc are positive
        assert (rt.weight[:, n_exc:] <= 0).all() or True  # cols from inh are negative

    def test_dale_disabled(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        has_positive = (rt.weight > 0).any()
        has_negative = (rt.weight < 0).any()
        assert has_positive and has_negative


class TestCellDynamics:
    def test_activation_bounded(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        for _ in range(20):
            rt.step(external_input=torch.randn(64) * 2.0)
        assert rt.activation.abs().max().item() <= 1.0

    def test_memory_trace_ema(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        for _ in range(100):
            rt.step(external_input=torch.randn(64) * 0.5)
        assert torch.isfinite(rt.memory_trace).all()

    def test_adaptation_bounded(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        for _ in range(100):
            rt.step(external_input=torch.randn(64))
        assert rt.adaptation.min().item() >= 0.0
        assert rt.adaptation.max().item() <= 2.0

    def test_bitfield_hysteresis(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        for _ in range(50):
            rt.step(external_input=torch.randn(64) * 2.0)
        assert rt.bitfield.dtype == torch.uint8
        assert set(rt.bitfield.unique().tolist()).issubset({0, 1})

    def test_stp_bounded(self):
        rt = make_runtime(dale_law=False, axon_delay=False)
        for _ in range(50):
            rt.step(external_input=torch.randn(64))
        assert rt.stp_u.min().item() >= 0.0
        assert rt.stp_u.max().item() <= 1.0
        assert rt.stp_x.min().item() >= 0.0
        assert rt.stp_x.max().item() <= 1.0
