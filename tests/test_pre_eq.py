from __future__ import annotations

import math

import numpy as np

from reality_stone.clarus.pre_eq import (
    born_prior,
    compose_weighted_kernels,
    conditional_values,
    free_energy_for_conditions,
    gibbs_kernel,
    gibbs_reweight,
    joint_gibbs,
    kernel_pushforward,
    manifest_indices,
    marginals,
    nonselected_residual,
    projected_minimizers,
    refined_branch_prior,
    tropical_compose,
    tropicalize,
)


def test_finite_gibbs_concentrates_on_unique_manifest() -> None:
    prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    energy = np.array([0.0, 1.0, 4.0])

    posterior = gibbs_reweight(prior, energy, beta=40.0)

    assert manifest_indices(prior, energy).tolist() == [0]
    assert posterior[0] > 1.0 - 1e-16
    assert posterior[1] < 1e-16
    assert posterior[2] < 1e-60


def test_condition_composition_is_energy_addition() -> None:
    prior = np.array([0.2, 0.3, 0.5])
    e1 = np.array([0.0, 2.0, 1.0])
    e2 = np.array([3.0, 0.0, 1.0])

    sequential = gibbs_reweight(gibbs_reweight(prior, e1, beta=0.7), e2, beta=1.3)
    combined = gibbs_reweight(prior, 0.7 * e1 + 1.3 * e2, beta=1.0)

    assert np.allclose(sequential, combined)


def test_nonselected_residual_keeps_raw_mass_and_conditional_shape() -> None:
    posterior = np.array([0.8, 0.15, 0.05])
    residual = nonselected_residual(posterior, selected=[0])

    assert math.isclose(residual.mass, 0.2)
    assert np.allclose(residual.raw, [0.0, 0.15, 0.05])
    assert np.allclose(residual.conditional, [0.0, 0.75, 0.25])


def test_weighted_kernel_pushforward_respects_composition_before_normalization() -> None:
    prior = np.array([0.4, 0.6])
    first = np.array([[2.0, 1.0], [0.5, 1.5]])
    second = np.array([[1.0, 0.0, 2.0], [0.5, 1.0, 0.5]])

    composed = compose_weighted_kernels(first, second)
    via_two_steps = kernel_pushforward(kernel_pushforward(prior, first), second)
    via_one_step = kernel_pushforward(prior, composed)

    assert np.allclose(via_two_steps, via_one_step)


def test_joint_bayes_readout_matches_rowwise_gibbs_conditionals() -> None:
    prior = np.array([[0.1, 0.2, 0.1], [0.1, 0.2, 0.3]])
    energy = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    beta = 3.0

    posterior = joint_gibbs(prior, energy, beta)
    condition_marginal, value_marginal = marginals(posterior)

    assert math.isclose(float(condition_marginal.sum()), 1.0)
    assert math.isclose(float(value_marginal.sum()), 1.0)
    assert np.allclose(
        conditional_values(posterior, 1),
        gibbs_reweight(prior[1] / prior[1].sum(), energy[1], beta),
    )

    free_energy = free_energy_for_conditions(prior, energy, beta)
    prior_condition_marginal = prior.sum(axis=1)
    reconstructed = prior_condition_marginal * np.exp(-beta * free_energy)
    reconstructed = reconstructed / reconstructed.sum()

    assert np.allclose(condition_marginal, reconstructed)


def test_projected_minimizers_returns_joint_and_marginal_readouts() -> None:
    prior = np.ones((3, 4))
    energy = np.array([
        [2.0, 0.0, 4.0, 5.0],
        [1.0, 0.0, 3.0, 0.0],
        [6.0, 7.0, 8.0, 9.0],
    ])

    coords, conditions, values = projected_minimizers(prior, energy)

    assert coords.tolist() == [[0, 1], [1, 1], [1, 3]]
    assert conditions.tolist() == [0, 1]
    assert values.tolist() == [1, 3]


def test_tropicalized_gibbs_composition_has_log_sum_exp_error_bound() -> None:
    first_energy = np.array([[0.0, 1.0], [2.0, 0.5]])
    second_energy = np.array([[1.0, 3.0], [0.0, 2.0]])
    beta = 25.0

    expected = tropical_compose(first_energy, second_energy)
    composed_kernel = compose_weighted_kernels(
        gibbs_kernel(first_energy, beta),
        gibbs_kernel(second_energy, beta),
    )
    tropicalized = tropicalize(composed_kernel, beta)

    error = expected - tropicalized
    assert np.all(error >= -1e-12)
    assert np.all(error <= math.log(first_energy.shape[1]) / beta + 1e-12)


def test_born_prior_is_phase_invariant_and_matches_branch_refinement() -> None:
    counts = np.array([1, 3, 4])
    total = int(counts.sum())
    amplitudes = np.array([
        math.sqrt(counts[0] / total),
        1j * math.sqrt(counts[1] / total),
        -math.sqrt(counts[2] / total),
    ])

    assert np.allclose(born_prior(amplitudes), refined_branch_prior(counts))
    assert np.allclose(born_prior(np.exp(1j * 0.7) * amplitudes), born_prior(amplitudes))

