from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "reality_stone"
    / "python"
    / "reality_stone"
    / "clarus"
    / "causal_recurrent_geometry_benchmark.py"
)
MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "preregistration"
    / "causal_recurrent_geometry_phase_a_v1.json"
)
RUNNER_PATH = (
    ROOT / "examples" / "agi" / "causal_recurrent_geometry_development_run.py"
)


def _isolated_load(path: Path, prefix: str):
    resolved = path.resolve(strict=True)
    source = resolved.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = f"{prefix}_{digest}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None:
        raise ImportError(f"cannot build spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(
            compile(source, str(resolved), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, digest, source


crg, MODULE_SHA256, MODULE_BYTES = _isolated_load(MODULE_PATH, "_ce_phase_a_test")


def _config(**changes):
    values = {
        "context_count": 3,
        "context_heterogeneity": 0.38,
        "experiment_version": "CE-PHASE-A-V1",
        "heldout_intervention_scale": 1.35,
        "heldout_steps": 96,
        "input_dimension": 2,
        "master_seed": 940_221,
        "noise_sigma": 0.05,
        "ridge": 1.0e-6,
        "state_dimension": 4,
        "train_intervention_scale": 0.75,
        "train_steps": 240,
    }
    values.update(changes)
    return crg.PhaseAConfig(**values)


def _exact_fixture():
    transitions = np.array(
        [
            [[0.25, 0.5], [-0.125, 0.375]],
            [[-0.5, 0.25], [0.75, 0.125]],
        ]
    )
    shared_input = np.array([[0.625], [-0.375]])
    state = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    intervention = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    context = np.array([0, 0, 1, 1, 0])
    next_state = np.stack(
        [
            transitions[context[index]] @ state[index]
            + shared_input @ intervention[index]
            for index in range(state.shape[0])
        ]
    )
    truth = crg.GroundTruth(transitions, shared_input, 0)
    batch = crg.TransitionBatch(state, intervention, context, next_state)
    return truth, batch


def test_isolated_load_hashes_the_exact_executed_bytes_without_parent_import() -> None:
    assert MODULE_SHA256 == hashlib.sha256(MODULE_BYTES).hexdigest()
    assert Path(crg.__file__).resolve() == MODULE_PATH.resolve()
    assert "reality_stone.clarus" not in sys.modules


def test_exact_shared_input_recovery_and_orientation() -> None:
    truth, batch = _exact_fixture()
    fit = crg.fit_context_shared_input(batch, context_count=2, ridge=0.0)
    errors = crg.coefficient_errors(fit, truth)
    assert fit.design.joint_rank == fit.design.required_rank == 5
    assert fit.design.context_state_ranks == (2, 2)
    assert fit.design.residualized_input_rank == 1
    assert errors["max_transition_error"] <= 1.0e-12
    assert errors["max_shared_input_error"] <= 1.0e-12
    assert np.array_equal(crg.predict(fit, batch), batch.next_state)
    assert fit.transitions[0, 0, 1] == pytest.approx(0.5)


def test_rank_deficiency_refuses_exact_edge_even_when_each_context_state_is_full_rank() -> None:
    transitions = np.array(
        [
            [[0.2, 0.1], [0.0, 0.3]],
            [[0.4, -0.2], [0.1, 0.25]],
        ]
    )
    shared_input = np.array([[0.5], [-0.25]])
    state = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ]
    )
    context = np.array([0, 0, 0, 1, 1, 1])
    intervention = state[:, :1].copy()
    next_state = np.stack(
        [
            transitions[context[index]] @ state[index]
            + shared_input @ intervention[index]
            for index in range(state.shape[0])
        ]
    )
    batch = crg.TransitionBatch(state, intervention, context, next_state)
    fit = crg.fit_context_shared_input(batch, context_count=2, ridge=1.0e-3)
    certificate = crg.claim_certificate(
        fit,
        declared_linear_class=True,
    )
    assert fit.design.context_state_ranks == (2, 2)
    assert fit.design.joint_rank < fit.design.required_rank
    assert fit.design.residualized_input_rank == 0
    assert not fit.design.full_rank
    assert not certificate.exact_edge_allowed
    with pytest.raises(PermissionError, match="not claimable"):
        crg.coefficient_errors(fit, crg.GroundTruth(transitions, shared_input, 0))
    zero_ridge_fit = crg.fit_context_shared_input(
        batch, context_count=2, ridge=0.0
    )
    assert np.isfinite(zero_ridge_fit.effective_dof)
    assert zero_ridge_fit.effective_dof == 8.0


def test_exact_edge_is_the_exact_four_term_conjunction_and_observation_refuses() -> None:
    truth, batch = _exact_fixture()
    fit = crg.fit_context_shared_input(batch, context_count=2, ridge=0.0)
    allowed = crg.claim_certificate(
        fit,
        declared_linear_class=True,
    )
    assert allowed.exact_edge_allowed
    assert not crg.claim_certificate(
        fit,
        declared_linear_class=False,
    ).exact_edge_allowed
    masked = crg.observe_batch(batch, "known_mask", [[1.0, 0.0]])
    masked_fit = crg.fit_context_shared_input(masked, context_count=2, ridge=0.0)
    assert not crg.claim_certificate(
        masked_fit, declared_linear_class=True
    ).exact_edge_allowed
    mixed = crg.observe_batch(batch, "unknown_mix", [[1.0, 0.5], [0.0, 1.0]])
    mixed_fit = crg.fit_context_shared_input(mixed, context_count=2, ridge=0.0)
    assert not crg.claim_certificate(
        mixed_fit, declared_linear_class=True
    ).exact_edge_allowed
    with pytest.raises(PermissionError, match="not claimable"):
        crg.coefficient_errors(mixed_fit, truth)
    deficient = replace(fit.design, full_rank=False)
    assert not crg.claim_certificate(
        replace(fit, design=deficient),
        declared_linear_class=True,
    ).exact_edge_allowed
    invalid = replace(fit.design, finite_valid_inputs=False)
    assert not crg.claim_certificate(
        replace(fit, design=invalid),
        declared_linear_class=True,
    ).exact_edge_allowed


def test_known_mask_and_unknown_mix_remain_predictive_coordinate_fixtures() -> None:
    _, batch = _exact_fixture()
    masked = crg.observe_batch(batch, "known_mask", [[1.0, 0.0]])
    mixed = crg.observe_batch(batch, "unknown_mix", [[1.0, 0.5], [0.0, 1.0]])
    assert masked.state.shape == (5, 1)
    assert masked.observation_kind == "known_mask"
    assert mixed.state.shape == batch.state.shape
    assert mixed.observation_kind == "unknown_mix"
    with pytest.raises(ValueError, match="invertible"):
        crg.observe_batch(batch, "unknown_mix", [[1.0, 0.0], [0.0, 0.0]])


def test_prediction_and_nll_refuse_cross_chart_scoring() -> None:
    _, batch = _exact_fixture()
    identity_fit = crg.fit_context_shared_input(batch, context_count=2, ridge=0.0)
    mixed = crg.observe_batch(batch, "unknown_mix", [[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match="observation kind"):
        crg.predict(identity_fit, mixed)
    with pytest.raises(ValueError, match="observation kind"):
        crg.gaussian_nll(identity_fit, mixed, sigma=0.1)


def test_similarity_no_go_changes_latent_support_but_not_observations() -> None:
    fixture = crg.similarity_no_go_fixture()
    assert fixture.support_differs
    assert fixture.observations_identical
    assert np.array_equal(
        fixture.observed_trajectory_a, fixture.observed_trajectory_b
    )
    assert fixture.transition_a[0, 1] == 0.0
    assert fixture.transition_b[0, 1] != 0.0


def test_role_namespace_is_deterministic_and_domain_separated() -> None:
    config = _config()
    roles = (
        "graph",
        "train_trajectory",
        "heldout_trajectory",
        "intervention",
        "train_noise",
        "evaluation_noise",
        "shuffle",
        "bootstrap",
    )
    digests = [
        crg.role_digest(
            config.experiment_version, config.master_seed, role, 1001, 0
        )
        for role in roles
    ]
    assert len(set(digests)) == len(roles)
    assert digests[0] == crg.role_digest(
        config.experiment_version, config.master_seed, "graph", 1001, 0
    )
    assert digests[0] != crg.role_digest(
        config.experiment_version, config.master_seed, "graph", 1002, 0
    )
    with pytest.raises(ValueError, match="unregistered RNG role"):
        crg.role_digest(config.experiment_version, config.master_seed, "reserved", 1)


def test_generator_replay_is_byte_identical_and_role_change_isolated() -> None:
    config = _config()
    generator = crg.DevelopmentGenerator(config, (1001,))
    truth_a = generator.ground_truth(1001)
    truth_b = generator.ground_truth(1001)
    train_a = generator.transition_batch(truth_a, split="train")
    train_b = generator.transition_batch(truth_b, split="train")
    heldout = generator.transition_batch(truth_a, split="heldout")
    assert truth_a.context_transitions.tobytes() == truth_b.context_transitions.tobytes()
    assert train_a.state.tobytes() == train_b.state.tobytes()
    assert train_a.intervention.tobytes() == train_b.intervention.tobytes()
    assert train_a.state.tobytes() != heldout.state.tobytes()
    assert train_a.intervention.tobytes() != heldout.intervention.tobytes()


def test_learner_signature_and_target_mutation_have_no_truth_or_future_path() -> None:
    signature = inspect.signature(crg.fit_context_shared_input)
    assert tuple(signature.parameters) == ("batch", "context_count", "ridge")
    assert "sigma" not in signature.parameters
    assert "truth" not in signature.parameters
    config = _config()
    generator = crg.DevelopmentGenerator(config, (1002,))
    truth = generator.ground_truth(1002)
    training = generator.transition_batch(truth, split="train")
    heldout = generator.transition_batch(truth, split="heldout")
    fit_before = crg.fit_context_shared_input(
        training, context_count=config.context_count, ridge=config.ridge
    )
    mutated = crg.TransitionBatch(
        heldout.state,
        heldout.intervention,
        heldout.context,
        heldout.next_state + 10_000.0,
    )
    assert mutated.next_state.tobytes() != heldout.next_state.tobytes()
    fit_after = crg.fit_context_shared_input(
        training, context_count=config.context_count, ridge=config.ridge
    )
    assert fit_before.transitions.tobytes() == fit_after.transitions.tobytes()
    assert fit_before.shared_input.tobytes() == fit_after.shared_input.tobytes()


def test_primary_arms_share_batch_ridge_sigma_and_publish_dof() -> None:
    config = _config()
    generator = crg.DevelopmentGenerator(config, (1003,))
    truth = generator.ground_truth(1003)
    training = generator.transition_batch(truth, split="train")
    heldout = generator.transition_batch(truth, split="heldout")
    factorized = crg.fit_context_shared_input(
        training, context_count=config.context_count, ridge=config.ridge
    )
    pooled = crg.fit_pooled_shared_input(training, ridge=config.ridge)
    assert factorized.ridge == pooled.ridge == config.ridge
    assert factorized.nominal_dof - pooled.nominal_dof == (
        config.context_count - 1
    ) * config.state_dimension**2
    assert factorized.effective_dof <= factorized.nominal_dof
    assert pooled.effective_dof <= pooled.nominal_dof
    factorized_nll = crg.gaussian_nll(
        factorized, heldout, sigma=config.noise_sigma
    )
    pooled_nll = crg.gaussian_nll(pooled, heldout, sigma=config.noise_sigma)
    factorized_residual = heldout.next_state - crg.predict(factorized, heldout)
    pooled_residual = heldout.next_state - crg.predict(pooled, heldout)
    scalar_count = heldout.next_state.size
    expected_factorized = 0.5 * (
        scalar_count * np.log(2.0 * np.pi * config.noise_sigma**2)
        + np.sum(factorized_residual**2) / config.noise_sigma**2
    )
    expected_delta = (
        np.sum(pooled_residual**2) - np.sum(factorized_residual**2)
    ) / (2.0 * config.noise_sigma**2)
    assert factorized_nll == pytest.approx(expected_factorized)
    assert pooled_nll - factorized_nll == pytest.approx(expected_delta)
    doubled_sigma = 2.0 * config.noise_sigma
    doubled_nll = crg.gaussian_nll(
        factorized, heldout, sigma=doubled_sigma
    )
    constant = 0.5 * scalar_count * np.log(
        2.0 * np.pi * config.noise_sigma**2
    )
    doubled_constant = 0.5 * scalar_count * np.log(
        2.0 * np.pi * doubled_sigma**2
    )
    assert doubled_nll - doubled_constant == pytest.approx(
        (factorized_nll - constant) / 4.0
    )


def test_equal_context_negative_fixture_forbids_false_strict_superiority() -> None:
    transition = np.array([[0.25, 0.5], [-0.125, 0.375]])
    transitions = np.repeat(transition[None, :, :], 2, axis=0)
    shared_input = np.array([[0.625], [-0.375]])
    state = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    intervention = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    context = np.array([0, 0, 1, 1, 0])
    next_state = np.stack(
        [
            transitions[context[index]] @ state[index]
            + shared_input @ intervention[index]
            for index in range(state.shape[0])
        ]
    )
    batch = crg.TransitionBatch(state, intervention, context, next_state)
    factorized = crg.fit_context_shared_input(batch, context_count=2, ridge=0.0)
    pooled = crg.fit_pooled_shared_input(batch, ridge=0.0)
    factorized_prediction = crg.predict(factorized, batch)
    pooled_prediction = crg.predict(pooled, batch)
    assert factorized.design.full_rank
    assert pooled.design.full_rank
    assert factorized_prediction == pytest.approx(pooled_prediction, abs=1.0e-12)
    delta = crg.gaussian_nll(pooled, batch, sigma=0.1) - crg.gaussian_nll(
        factorized, batch, sigma=0.1
    )
    assert delta == pytest.approx(0.0, abs=1.0e-12)


def test_dimensionless_manifest_certificate_is_exact_and_fail_closed() -> None:
    section = {
        "dimension_tags": {
            "gaussian_residual": "DIMENSIONLESS",
            "input": "DIMENSIONLESS",
            "noise": "DIMENSIONLESS",
            "state": "DIMENSIONLESS",
        },
        "normalized_coordinates": True,
        "reference_scales": {
            "input": [1.0, 1.0],
            "noise": [1.0],
            "state": [1.0, 1.0, 1.0, 1.0],
        },
    }
    assert crg.dimensionless_certificate(section).passed
    with pytest.raises(ValueError, match="positive"):
        crg.dimensionless_certificate(
            {
                **section,
                "reference_scales": {**section["reference_scales"], "noise": [0.0]},
            }
        )
    assert not crg.dimensionless_certificate(
        {**section, "normalized_coordinates": False}
    ).passed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("noise_sigma", 0.0, "positive"),
        ("noise_sigma", np.nan, "finite"),
        ("ridge", np.inf, "finite"),
        ("context_count", 1, "at least two"),
    ],
)
def test_config_domain_refusal(field: str, value: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**{field: value})


@pytest.mark.parametrize("value", ["0.1", Decimal("0.1"), object()])
def test_config_real_scalars_refuse_string_decimal_and_object(value: object) -> None:
    with pytest.raises(TypeError, match="finite float"):
        _config(noise_sigma=value)


def test_config_normalizes_numpy_scalar_types_to_builtin_fields() -> None:
    config = _config(
        master_seed=np.int64(940_221),
        noise_sigma=np.float64(0.05),
        ridge=np.float64(1.0e-6),
    )
    assert type(config.master_seed) is int
    assert type(config.noise_sigma) is float
    assert type(config.ridge) is float


def test_batch_shape_nonfinite_empty_context_and_sigma_refusal() -> None:
    _, batch = _exact_fixture()
    with pytest.raises(ValueError, match="axis 0"):
        crg.TransitionBatch(
            batch.state,
            batch.intervention[:-1],
            batch.context,
            batch.next_state,
        )
    invalid = np.array(batch.state, copy=True)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        crg.TransitionBatch(
            invalid, batch.intervention, batch.context, batch.next_state
        )
    with pytest.raises(ValueError, match="every declared context"):
        crg.design_certificate(batch, 3)
    fit = crg.fit_context_shared_input(batch, context_count=2, ridge=0.0)
    with pytest.raises(ValueError, match="positive"):
        crg.gaussian_nll(fit, batch, sigma=0.0)


@pytest.mark.parametrize(
    "invalid_context",
    [
        [0.9, 1.1],
        [True, False],
        ["0", "1"],
        np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64),
    ],
)
def test_context_labels_refuse_lossy_or_noninteger_coercion(
    invalid_context: object,
) -> None:
    state = np.eye(2)
    intervention = np.ones((2, 1))
    with pytest.raises((TypeError, ValueError), match="integer labels|outside int64"):
        crg.TransitionBatch(
            state,
            intervention,
            invalid_context,
            state,
        )


def test_registered_bootstrap_stream_is_shared_and_each_interval_recomputes() -> None:
    config = _config()
    first = np.array([1.0, 2.0, 4.0, 8.0])
    second = np.array([-3.0, 0.0, 5.0, 7.0])
    sample_count = 300
    registered_seed = 771
    first_interval = crg.paired_bootstrap_interval(
        first,
        config=config,
        bootstrap_seed=registered_seed,
        samples=sample_count,
    )
    second_interval = crg.paired_bootstrap_interval(
        second,
        config=config,
        bootstrap_seed=registered_seed,
        samples=sample_count,
    )
    rng = crg.role_rng(config, "bootstrap", 0, registered_seed)
    shared_indices = rng.integers(
        0, first.size, size=(sample_count, first.size)
    )
    assert first_interval == pytest.approx(
        np.quantile(np.mean(first[shared_indices], axis=1), [0.025, 0.975])
    )
    assert second_interval == pytest.approx(
        np.quantile(np.mean(second[shared_indices], axis=1), [0.025, 0.975])
    )


def test_development_seed_level_deltas_bootstrap_shuffle_and_exclusions() -> None:
    config = _config()
    development = tuple(range(1001, 1005))
    result = crg.run_development_benchmark(
        config,
        graph_seeds=development,
        registered_development_graph_seeds=development,
        bootstrap_seed=771,
        bootstrap_samples=400,
    )
    replay = crg.run_development_benchmark(
        config,
        graph_seeds=development,
        registered_development_graph_seeds=development,
        bootstrap_seed=771,
        bootstrap_samples=400,
    )
    assert result["mode"] == "development"
    assert [item["graph_seed"] for item in result["per_graph_seed"]] == list(
        development
    )
    assert result["aggregate"]["graph_seed_count"] == len(development)
    assert result["dof_accounting"]["factorized_minus_pooled"] == 32
    assert result["protocol"]["common_manifest_sigma_scorer_only"] == 0.05
    assert result["protocol"]["frame_is_statistical_unit"] is False
    assert result["protocol"]["nll_aggregation"] == (
        "total_over_graph_seed_transitions_and_coordinates"
    )
    assert result["protocol"]["bootstrap_seed"] == 771
    assert result["protocol"]["bootstrap_samples"] == 400
    assert result["protocol"][
        "bootstrap_stream_shared_across_registered_endpoints"
    ]
    assert not any(result["exclusions"].values())
    assert all(
        item["residual_scalar_count"] == config.heldout_steps * config.state_dimension
        for item in result["per_graph_seed"]
    )
    assert all(
        item["joint_design"]["joint_rank"]
        == item["joint_design"]["required_rank"]
        for item in result["per_graph_seed"]
    )
    aggregate = result["aggregate"]
    expected_h1 = bool(
        aggregate["delta_nll_mean"] > 0.0
        and aggregate["delta_nll_median"] > 0.0
        and aggregate["delta_nll_paired_bootstrap_95"][0] > 0.0
    )
    expected_h2 = bool(
        aggregate["shuffle_penalty_mean"] > 0.0
        and aggregate["shuffle_penalty_median"] > 0.0
        and aggregate["shuffle_penalty_paired_bootstrap_95"][0] > 0.0
    )
    assert (result["claim_status"]["PA-H1"] == "GO") is expected_h1
    assert (result["claim_status"]["PA-H2"] == "GO") is expected_h2
    assert result["primary_gate"] is expected_h1
    assert result["shuffle_integrity_gate"] is expected_h2
    assert crg.canonical_json_bytes(result) == crg.canonical_json_bytes(replay)


def test_reserved_or_partial_seed_block_refuses_before_namespace_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    development = (1001, 1002)
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("namespace generation occurred before seed refusal")

    monkeypatch.setattr(crg, "role_digest", forbidden)
    with pytest.raises(PermissionError, match="exact development seed block"):
        crg.run_development_benchmark(
            config,
            graph_seeds=(9001,),
            registered_development_graph_seeds=development,
            bootstrap_seed=1,
            bootstrap_samples=10,
        )


def _runner():
    return _isolated_load(RUNNER_PATH, "_ce_phase_a_runner_test")


def _manifest_payload() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_runner_isolated_load_and_manifest_validate_without_package_initializer() -> None:
    runner, digest, source = _runner()
    assert digest == hashlib.sha256(source).hexdigest()
    validated = runner.load_and_validate_manifest(MANIFEST_PATH)
    assert validated["manifest_sha256"] == _manifest_payload()["manifest_sha256"]
    assert validated["confirmation"]["status"] == "reserved_unopened"
    assert validated["confirmation"]["execution_authorized"] is False
    assert validated["confirmation"]["raw_seed_material_present"] is False
    assert validated["confirmation"]["reservation_kind"] == "reservation_only"
    assert validated["confirmation"]["custody_status"] == "custody_unverified"
    assert validated["confirmation"]["holdout_status"] == (
        "not_executable_holdout"
    )
    assert not hasattr(runner, "run_confirmation")
    assert b"CONFIRMATION_SEEDS" not in source
    assert "reality_stone.clarus" not in sys.modules


def test_manifest_duplicate_nested_key_nonfinite_and_self_tamper_refuse() -> None:
    runner, _, _ = _runner()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._json_without_duplicate_keys(
            '{"outer":{"value":1,"value":2}}', source="duplicate-fixture"
        )
    with pytest.raises(ValueError, match="nonfinite JSON constant"):
        runner._json_without_duplicate_keys(
            '{"outer":{"value":NaN}}', source="nonfinite-fixture"
        )
    payload = _manifest_payload()
    payload["generator"]["ridge"] = 7.0
    with pytest.raises(ValueError, match="self-hash mismatch"):
        runner.validate_manifest_payload(payload, root=ROOT)


@pytest.mark.parametrize("tamper", ["duplicate_path", "duplicate_hash", "traversal"])
def test_manifest_duplicate_path_hash_and_traversal_refuse(tamper: str) -> None:
    runner, _, _ = _runner()
    payload = _manifest_payload()
    artifacts = payload["required_artifacts"]
    if tamper == "duplicate_path":
        artifacts[1]["path"] = artifacts[0]["path"]
    elif tamper == "duplicate_hash":
        artifacts[1]["sha256"] = artifacts[0]["sha256"]
    else:
        artifacts[1]["path"] = "../outside.py"
    payload["manifest_sha256"] = runner._manifest_digest(payload)
    with pytest.raises(ValueError, match="duplicate|required|invalid"):
        runner.validate_manifest_payload(payload, root=ROOT)


def test_manifest_source_hash_tamper_refuses_before_loading() -> None:
    runner, _, _ = _runner()
    payload = _manifest_payload()
    payload["required_artifacts"][0]["sha256"] = "0" * 64
    payload["manifest_sha256"] = runner._manifest_digest(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.validate_manifest_payload(payload, root=ROOT)


def test_confirmation_seal_has_no_raw_seed_or_namespace_generation_path() -> None:
    runner, _, source = _runner()
    payload = _manifest_payload()
    seal = payload["confirmation"]
    assert set(seal) == {
        "commitment_domain",
        "commitment_scheme",
        "custody_status",
        "disjoint_from_pilot_and_development",
        "execution_authorized",
        "holdout_status",
        "opaque_commitment",
        "raw_seed_material_present",
        "reservation_kind",
        "seed_count",
        "status",
    }
    assert not any("seed" in key and key.endswith("seeds") for key in seal)
    assert set(payload["seed_roles"]) == {
        "development_graph_seeds",
        "pilot_graph_seeds",
    }
    forbidden = copy.deepcopy(payload)
    forbidden["confirmation"]["raw_seed_material_present"] = True
    forbidden["manifest_sha256"] = runner._manifest_digest(forbidden)
    with pytest.raises(PermissionError, match="raw confirmation"):
        runner.validate_manifest_payload(forbidden, root=ROOT)
    assert b"confirmation_rng" not in source
    assert b"confirmation_seed" not in source


def test_result_payload_has_exact_confirmation_status_without_execution_fields() -> None:
    runner, _, _ = _runner()
    registration = {
        "manifest_file_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "required_artifacts": {"artifact": "c" * 64},
    }
    payload = runner._build_result_payload(
        registration,
        {"claim_status": {"PA-H1": "STOP", "PA-H2": "STOP"}},
        loaded_sha256="d" * 64,
    )
    assert payload["confirmation_status"] == "reserved_unopened"
    assert payload["confirmation"]["status"] == payload["confirmation_status"]
    assert payload["confirmation"]["reservation_kind"] == "reservation_only"
    assert payload["confirmation"]["custody_status"] == "custody_unverified"
    assert payload["confirmation"]["holdout_status"] == "not_executable_holdout"
    serialized = runner._canonical_bytes(payload)
    for forbidden in (
        b"confirmation_score",
        b"confirmation_receipt",
        b"confirmation_seed",
        b"confirmation_namespace",
    ):
        assert forbidden not in serialized


def test_manifest_dimensionless_scale_lengths_are_bound_to_generator_shape() -> None:
    runner, _, _ = _runner()
    payload = _manifest_payload()
    payload["dimensionless"]["reference_scales"]["state"].pop()
    payload["manifest_sha256"] = runner._manifest_digest(payload)
    with pytest.raises(ValueError, match="state reference scale length"):
        runner.validate_manifest_payload(payload, root=ROOT)


def test_runner_compiles_the_single_hashed_buffer_without_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _, _ = _runner()
    source_path = tmp_path / "sealed_module.py"
    source_path.write_bytes(b"VALUE = 7\n")
    expected = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original = Path.read_bytes
    calls: list[Path] = []

    def tracked(path: Path) -> bytes:
        if path.resolve() == source_path.resolve():
            calls.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    loaded, digest = runner._isolated_load(source_path, expected)
    assert loaded.VALUE == 7
    assert digest == expected
    assert len(calls) == 1


def test_one_shot_reserves_before_evaluator_and_second_call_never_enters(
    tmp_path: Path,
) -> None:
    runner, _, _ = _runner()
    output = tmp_path / "development-results.json"
    evaluator_entries: list[bytes] = []

    def evaluator() -> dict:
        evaluator_entries.append(output.read_bytes())
        return {"finite": True}

    runner._execute_one_shot_reserved(output, evaluator)
    assert evaluator_entries == [b""]
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": True}
    with pytest.raises(FileExistsError):
        runner._execute_one_shot_reserved(output, evaluator)
    assert evaluator_entries == [b""]


def test_one_shot_failure_preserves_reservation_and_stale_temp_refuses_pre_eval(
    tmp_path: Path,
) -> None:
    runner, _, _ = _runner()
    failed_output = tmp_path / "failed.json"

    def failing_evaluator() -> dict:
        assert failed_output.read_bytes() == b""
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        runner._execute_one_shot_reserved(failed_output, failing_evaluator)
    assert failed_output.is_file()
    assert failed_output.read_bytes() == b""

    stale_output = tmp_path / "stale.json"
    stale_output.with_suffix(".json.tmp").write_bytes(b"audit evidence")
    entered = False

    def forbidden_evaluator() -> dict:
        nonlocal entered
        entered = True
        return {}

    with pytest.raises(FileExistsError, match="stale temporary"):
        runner._execute_one_shot_reserved(stale_output, forbidden_evaluator)
    assert not entered
    assert not stale_output.exists()
