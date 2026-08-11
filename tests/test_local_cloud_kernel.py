import math

import numpy as np
import pytest

from reality_stone.clarus.local_cloud_kernel import (
    LocalCloudKernelConfig,
    LocalCloudObservation,
    LocalCloudState,
    LocalCloudTransitionKernel,
)


def _observation(scale=1.0):
    return LocalCloudObservation(
        local=tuple(
            tuple(scale * (row + 1) * (column + 1) / 16.0 for column in range(4))
            for row in range(4)
        ),
        shared=tuple(scale * (column + 1) / 8.0 for column in range(4)),
    )


def test_default_small_gain_certificate_is_typed_and_strict() -> None:
    kernel = LocalCloudTransitionKernel()
    certificate = kernel.certificate
    matrix = np.asarray(certificate.matrix)
    weights = np.asarray(certificate.weights)
    assert certificate.metric == "weighted_local_cloud_sup"
    assert certificate.certified
    assert certificate.spectral_radius < 1.0
    assert certificate.contraction_factor < 1.0
    assert np.max(matrix @ weights / weights) == pytest.approx(certificate.contraction_factor)


def test_unstable_cross_gain_is_rejected() -> None:
    with pytest.raises(ValueError, match="not certified"):
        LocalCloudTransitionKernel(LocalCloudKernelConfig(local_to_cloud=0.75, cloud_to_local=0.75))


def test_step_has_exact_twenty_state_features_and_no_target_input() -> None:
    kernel = LocalCloudTransitionKernel()
    result = kernel.step(kernel.zero_state(), _observation())
    assert len(result.features) == 20
    assert len(result.state.local) == 4
    assert all(len(row) == 4 for row in result.state.local)
    assert len(result.state.cloud) == 4
    assert all(math.isfinite(value) for value in result.features)
    assert any(value != 0.0 for value in result.features)


def test_sequential_composition_is_exact() -> None:
    kernel = LocalCloudTransitionKernel()
    observations = (_observation(0.2), _observation(-0.5), _observation(0.7))
    composed = kernel.compose(kernel.zero_state(), observations)
    current = kernel.zero_state()
    for observation in observations:
        stepped = kernel.step(current, observation)
        current = stepped.state
    assert composed.state == current
    assert composed.features == stepped.features


def test_identity_composition_preserves_state() -> None:
    kernel = LocalCloudTransitionKernel()
    state = kernel.step(kernel.zero_state(), _observation()).state
    identity = kernel.compose(state, ())
    assert identity.state == state


def test_cross_and_reset_lesions_act_on_the_transition_not_the_readout() -> None:
    kernel = LocalCloudTransitionKernel()
    primed = kernel.compose(kernel.zero_state(), (_observation(), _observation(-0.4))).state
    event = _observation(0.1)
    intact = kernel.step(primed, event)
    cross_cut = kernel.step(primed, event, lesion="cross_cut")
    local_reset = kernel.step(primed, event, lesion="local_reset")
    cloud_reset = kernel.step(primed, event, lesion="cloud_reset")
    assert intact.features != cross_cut.features
    assert intact.features != local_reset.features
    assert intact.features != cloud_reset.features


def test_cloud_path_changes_local_state_and_local_path_changes_cloud_state() -> None:
    kernel = LocalCloudTransitionKernel()
    zero = kernel.zero_state()
    cloud_only = LocalCloudState(local=zero.local, cloud=(0.5, -0.5, 0.25, -0.25))
    cloud_effect = kernel.step(cloud_only, _observation(0.0))
    cloud_cut = kernel.step(cloud_only, _observation(0.0), lesion="cross_cut")
    assert cloud_effect.state.local != cloud_cut.state.local

    local = tuple(((0.5 if row == 0 else -0.25),) * 4 for row in range(4))
    local_only = LocalCloudState(local=local, cloud=zero.cloud)
    local_effect = kernel.step(local_only, _observation(0.0))
    local_cut = kernel.step(local_only, _observation(0.0), lesion="cross_cut")
    assert local_effect.state.cloud != local_cut.state.cloud


@pytest.mark.parametrize(
    "kwargs",
    (
        {"local_count": True},
        {"width": 4.0},
        {"local_retentions": [0.2, 0.4, 0.6, 0.8]},
        {"local_retentions": (0.2,)},
        {"input_gain": "0.3"},
        {"cloud_retention": float("nan")},
    ),
)
def test_invalid_config_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        LocalCloudKernelConfig(**kwargs)


def test_invalid_state_and_observation_do_not_mutate_inputs() -> None:
    kernel = LocalCloudTransitionKernel()
    state = kernel.zero_state()
    before = state
    with pytest.raises(ValueError):
        kernel.step(state, LocalCloudObservation(local=((1.0,),), shared=(0.0,) * 4))
    assert state == before


def test_empirical_weighted_sup_ratio_respects_certificate() -> None:
    kernel = LocalCloudTransitionKernel()
    rng = np.random.default_rng(8801)
    weights = kernel.certificate.weights
    observation = _observation(0.3)
    for _ in range(100):
        local_a = rng.uniform(-1.0, 1.0, size=(4, 4))
        cloud_a = rng.uniform(-1.0, 1.0, size=4)
        local_b = rng.uniform(-1.0, 1.0, size=(4, 4))
        cloud_b = rng.uniform(-1.0, 1.0, size=4)
        state_a = LocalCloudState(tuple(map(tuple, local_a)), tuple(cloud_a))
        state_b = LocalCloudState(tuple(map(tuple, local_b)), tuple(cloud_b))
        next_a = kernel.step(state_a, observation).state
        next_b = kernel.step(state_b, observation).state
        local_in = np.max(np.abs(local_a - local_b)) / weights[0]
        cloud_in = np.max(np.abs(cloud_a - cloud_b)) / weights[1]
        denominator = max(local_in, cloud_in)
        local_out = np.max(np.abs(np.asarray(next_a.local) - np.asarray(next_b.local))) / weights[0]
        cloud_out = np.max(np.abs(np.asarray(next_a.cloud) - np.asarray(next_b.cloud))) / weights[1]
        assert max(local_out, cloud_out) <= (
            kernel.certificate.contraction_factor * denominator + 1e-14
        )
