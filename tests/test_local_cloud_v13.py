import numpy as np
import pytest
import torch

from reality_stone.clarus.local_cloud_benchmark import LocalCloudBenchmarkConfig
from reality_stone.clarus.local_cloud_v13_benchmark import (
    V13_PANELS,
    _pattern_bits,
    cell_label,
    generate_episodes_v2,
    holdout_cells,
    holdout_cells_balanced,
    panel_configs_v13,
)
from reality_stone.clarus.gated_local_cloud_v13 import (
    GatedLocalCloudV13,
    GatedLocalCloudV13B,
    model_hash,
    train_gated_v13,
    train_gated_v13b,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "agi"))
import local_cloud_v13_run as v13_run  # noqa: E402


def _small_config() -> LocalCloudBenchmarkConfig:
    return LocalCloudBenchmarkConfig(train_episodes=96, evaluation_episodes=32)


# --- condition split separation -------------------------------------------------


def test_compositional_split_has_no_cell_overlap() -> None:
    config = _small_config()
    train = generate_episodes_v2(4242, 96, config, split="train", condition_split="compositional")
    evaluation = generate_episodes_v2(
        4242, 32, config, split="evaluation", condition_split="compositional"
    )
    train_cells = {(row.context_index, row.local_bits[:3]) for row in train}
    eval_cells = {(row.context_index, row.local_bits[:3]) for row in evaluation}
    assert len(train_cells) == 24
    assert len(eval_cells) == 8
    assert train_cells.isdisjoint(eval_cells)


def test_compositional_split_preserves_every_individual_bit_value_in_train() -> None:
    config = _small_config()
    train = generate_episodes_v2(555, 96, config, split="train", condition_split="compositional")
    for context in range(4):
        context_rows = [row for row in train if row.context_index == context]
        for bit_index in range(3):
            values = {row.local_bits[bit_index] for row in context_rows}
            assert values == {-1, 1}


def test_holdout_cells_are_deterministic_and_bitwise_complementary() -> None:
    first = holdout_cells(99)
    second = holdout_cells(99)
    assert first == second
    assert len(set(first)) == 8
    by_context: dict[int, list[int]] = {}
    for context, pattern in first:
        by_context.setdefault(context, []).append(pattern)
    for context, patterns in by_context.items():
        assert len(patterns) == 2
        a, b = patterns
        assert a + b == 7


def test_iid_condition_split_matches_frozen_v10_generator() -> None:
    from reality_stone.clarus.local_cloud_benchmark import generate_episodes

    config = _small_config()
    a = generate_episodes_v2(17, 32, config, split="train", condition_split="iid")
    b = generate_episodes(17, 32, config, split="train")
    assert a == b


def test_v13_panels_cover_the_registered_five() -> None:
    panels = panel_configs_v13(train_episodes=96, evaluation_episodes=32)
    assert set(panels) == set(V13_PANELS) == {"id", "noise", "horizon", "combined", "heldout"}
    heldout_config, heldout_split = panels["heldout"]
    assert heldout_split == "compositional"
    assert (heldout_config.episode_steps, heldout_config.noise_sigma) == (4, 0.04)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"count": 90, "condition_split": "compositional", "split": "train"},
        {"count": 30, "condition_split": "compositional", "split": "evaluation"},
    ),
)
def test_compositional_count_must_divide_the_fixed_pool_size(kwargs) -> None:
    config = _small_config()
    with pytest.raises(ValueError, match="divisible"):
        generate_episodes_v2(1, **kwargs, config=config)


# --- balanced (fair) condition split ---------------------------------------------


_BALANCED_TEST_SEEDS = (0, 1, 2, 3, 99, 4242, *range(9000, 9016))


def test_balanced_holdout_is_deterministic_label_balanced_and_non_complementary() -> None:
    for seed in _BALANCED_TEST_SEEDS:
        cells = holdout_cells_balanced(seed)
        assert cells == holdout_cells_balanced(seed)
        assert len(set(cells)) == 8
        by_context: dict[int, list[int]] = {}
        for context, pattern in cells:
            by_context.setdefault(context, []).append(pattern)
        assert set(by_context) == {0, 1, 2, 3}
        for context, patterns in by_context.items():
            assert len(patterns) == 2
            a, b = patterns
            # Not a bitwise-complement pair (the anti-identifiable case).
            assert a + b != 7
            # One +1-labelled and one -1-labelled cell per context.
            assert sorted(cell_label(context, p) for p in (a, b)) == [-1, 1]


def test_balanced_split_has_no_cell_overlap_and_balanced_eval_labels() -> None:
    config = _small_config()
    train = generate_episodes_v2(4242, 96, config, split="train", condition_split="balanced")
    evaluation = generate_episodes_v2(
        4242, 32, config, split="evaluation", condition_split="balanced"
    )
    train_cells = {(row.context_index, row.local_bits[:3]) for row in train}
    eval_cells = {(row.context_index, row.local_bits[:3]) for row in evaluation}
    assert len(train_cells) == 24
    assert len(eval_cells) == 8
    assert train_cells.isdisjoint(eval_cells)
    # 4 held-out cells labelled +1 and 4 labelled -1, drawn uniformly.
    assert sum(row.target for row in evaluation) == 0


def _logistic_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    """Plain unregularized logistic regression by full-batch gradient
    descent (with bias); enough iterations that the direction is effectively
    the max-margin one on this tiny separable problem."""
    x = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
    w = np.zeros(x.shape[1])
    for _ in range(5000):
        margins = train_y * (x @ w)
        grad = -(x * (train_y / (1.0 + np.exp(margins)))[:, None]).mean(axis=0)
        w -= 0.5 * grad
    t = np.hstack([test_x, np.ones((test_x.shape[0], 1))])
    return np.where(t @ w >= 0.0, 1, -1)


def test_balanced_holdout_labels_are_identifiable_by_logistic_regression() -> None:
    # The fairness property that motivated the balanced split: for every run
    # seed and every context, logistic regression trained on the 6 train
    # cells predicts the 2 held-out cells exactly. (Under the compositional
    # complement-pair holdout this fails with probability 3/4 per context —
    # the labels are undetermined and the learner outputs the opposite sign.)
    for seed in _BALANCED_TEST_SEEDS:
        held = set(holdout_cells_balanced(seed))
        for context in range(4):
            held_patterns = sorted(p for c, p in held if c == context)
            train_patterns = [p for p in range(8) if (context, p) not in held]
            assert len(held_patterns) == 2 and len(train_patterns) == 6
            train_x = np.array([_pattern_bits(p) for p in train_patterns], dtype=np.float64)
            train_y = np.array(
                [cell_label(context, p) for p in train_patterns], dtype=np.float64
            )
            test_x = np.array([_pattern_bits(p) for p in held_patterns], dtype=np.float64)
            test_y = [cell_label(context, p) for p in held_patterns]
            predictions = _logistic_fit_predict(train_x, train_y, test_x)
            assert list(predictions) == test_y, (seed, context, held_patterns)


def test_panel_configs_heldout_split_selects_balanced() -> None:
    panels = panel_configs_v13(
        train_episodes=96, evaluation_episodes=32, heldout_split="balanced"
    )
    _, heldout_split = panels["heldout"]
    assert heldout_split == "balanced"
    with pytest.raises(ValueError, match="heldout_split"):
        panel_configs_v13(train_episodes=96, evaluation_episodes=32, heldout_split="iid")


# --- model forward shape / determinism ------------------------------------------


def test_gated_v13_forward_shape_and_finite_gradients() -> None:
    model = GatedLocalCloudV13()
    sequence = torch.linspace(-1.0, 1.0, 5 * 8 * 20).reshape(5, 8, 20)
    scores = model(sequence)
    assert scores.shape == (5,)
    assert torch.all(torch.isfinite(scores))
    scores.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())


def test_gated_v13_parameter_count_is_within_the_targeted_budget() -> None:
    model = GatedLocalCloudV13()
    assert 200 <= model.parameter_count() <= 600
    assert model.parameter_count() < 2541  # smaller than the GRU-20 comparator


def test_gated_v13_rejects_bad_input_shape_and_out_of_range_cap() -> None:
    model = GatedLocalCloudV13()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 4, 19))
    with pytest.raises(ValueError):
        GatedLocalCloudV13(retention_cap=1.5)
    with pytest.raises(ValueError):
        GatedLocalCloudV13(retention_cap=0.0)


def test_gated_v13_training_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(31)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    first = train_gated_v13(features, labels, seed=7, epochs=3)
    second = train_gated_v13(features, labels, seed=7, epochs=3)
    assert model_hash(first) == model_hash(second)


def test_gated_v13_retention_respects_the_relaxed_cap() -> None:
    model = GatedLocalCloudV13(retention_cap=1.0)
    local_retention, cloud_retention = model.retentions()
    assert torch.all(local_retention < 1.0) and torch.all(local_retention > 0.0)
    assert torch.all(cloud_retention < 1.0) and torch.all(cloud_retention > 0.0)
    report = model.lipschitz_report()
    assert report["certified"] is False
    assert set(report) >= {
        "local_retention_max",
        "cloud_retention_max",
        "loose_upper_bound_local_lipschitz",
        "loose_upper_bound_cloud_lipschitz",
    }


# --- v13b: convex combination + spectral cap -------------------------------------


def test_v13b_gate_zero_freezes_the_state_exactly() -> None:
    torch.manual_seed(0)
    model = GatedLocalCloudV13B()
    with torch.no_grad():
        model.local_gate.weight.zero_()
        model.local_gate.bias.fill_(-40.0)
        model.cloud_gate.weight.zero_()
        model.cloud_gate.bias.fill_(-40.0)
    model.eval()
    sequence = torch.linspace(-1.0, 1.0, 3 * 6 * 20).reshape(3, 6, 20)
    with torch.no_grad():
        scores = model(sequence)
    # g ~= 0 => h' = h at every tick; state stays at its zero initialization,
    # so the score is exactly the readout bias regardless of the input.
    expected = model.readout.bias.expand(3)
    assert torch.allclose(scores, expected, atol=1e-6)


def test_v13b_gate_one_yields_the_pure_candidate_state() -> None:
    torch.manual_seed(1)
    model = GatedLocalCloudV13B()
    with torch.no_grad():
        model.local_gate.weight.zero_()
        model.local_gate.bias.fill_(40.0)
        model.cloud_gate.weight.zero_()
        model.cloud_gate.bias.fill_(40.0)
    model.eval()
    sequence = torch.linspace(-0.8, 0.9, 1 * 1 * 20).reshape(1, 1, 20)
    with torch.no_grad():
        scores = model(sequence)
        # One tick from the zero state: recurrent/cross/interaction terms all
        # vanish, so g ~= 1 gives h' = h_tilde = tanh(W_in x + b) exactly.
        local_in = sequence[0, 0, :16].reshape(4, 4)
        cloud_in = sequence[0, 0, 16:]
        expected_local = torch.tanh(model.local_input(local_in))
        expected_cloud = torch.tanh(model.cloud_input(cloud_in))
        features = torch.cat((expected_local.reshape(16), expected_cloud))
        expected = model.readout(features)
    assert torch.allclose(scores, expected, atol=1e-6)


def test_v13b_state_stays_inside_the_unit_box() -> None:
    # The convex combination from zero init must keep |state|_inf <= 1, which
    # is what makes the interaction-term bound in lipschitz_report valid.
    torch.manual_seed(2)
    model = GatedLocalCloudV13B()
    with torch.no_grad():
        for name in ("local_input", "cloud_input"):
            getattr(model, name).weight.mul_(10.0)
    model.eval()
    sequence = 5.0 * torch.randn(4, 12, 20)
    with torch.no_grad():
        scores = model(sequence)
    with torch.no_grad():
        readout_weight_l1 = float(model.readout.weight.abs().sum())
        readout_bias = float(model.readout.bias.abs())
    assert torch.all(scores.abs() <= readout_weight_l1 + readout_bias + 1e-5)


def test_v13b_spectral_cap_holds_after_training_and_on_a_forced_expansion() -> None:
    # Forced case: sigma(5*I) = 5 must be capped to 1 exactly.
    model = GatedLocalCloudV13B()
    with torch.no_grad():
        model.local_rec.weight.copy_(5.0 * torch.eye(4))
    model.eval()
    with torch.no_grad():
        capped = model._capped_weight("local_rec")
        assert float(torch.linalg.matrix_norm(capped, ord=2)) <= 1.0 + 1e-5

    # Trained case: after the standard regime, every effective recurrent-path
    # matrix must have spectral norm <= 1 + epsilon (float32 slack only: the
    # cap divides by the exact top singular value).
    rng = np.random.default_rng(77)
    features = rng.normal(size=(32, 8, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    trained = train_gated_v13b(features, labels, seed=5, epochs=50)
    trained.eval()
    with torch.no_grad():
        for name in GatedLocalCloudV13B._CAPPED:
            sigma = float(
                torch.linalg.matrix_norm(trained._capped_weight(name), ord=2)
            )
            assert sigma <= 1.0 + 1e-5, (name, sigma)


def test_v13b_lipschitz_report_is_structural_but_uncertified() -> None:
    model = GatedLocalCloudV13B()
    report = model.lipschitz_report()
    assert report["certified"] is False
    assert "structural_bound" in report
    assert set(report) >= {f"sigma_{name}" for name in GatedLocalCloudV13B._CAPPED}
    # With all caps exactly at sigma <= 1, the coarse per-branch candidate
    # bound of three summed matrices cannot exceed 3 (float32 slack only).
    assert 1.0 <= report["structural_bound"] <= 3.0 + 1e-4


def test_v13b_parameter_count_is_within_30pct_of_v13() -> None:
    v13 = GatedLocalCloudV13().parameter_count()
    v13b = GatedLocalCloudV13B().parameter_count()
    assert v13 == 205
    assert v13b == 197
    assert 0.7 * v13 <= v13b <= 1.3 * v13


def test_v13b_training_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    first = train_gated_v13b(features, labels, seed=9, epochs=3)
    second = train_gated_v13b(features, labels, seed=9, epochs=3)
    assert model_hash(first) == model_hash(second)


def test_v13b_variant_runs_through_the_development_harness() -> None:
    result = v13_run.evaluate_development(
        (61, 62),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=10,
        variant="v13b",
    )
    assert result["variant"] == "v13b"
    for panel in V13_PANELS:
        assert set(result["panel_means"][panel]) == {
            "v13b",
            "v10",
            "elman3",
            "elman20",
            "gru20",
        }
    assert result["overall"] in {"GO", "STOP"}
    ledger = result["seed_rows"][0]["v13_ledger"]
    assert ledger["variant"] == "v13b"
    assert ledger["lipschitz_report"]["certified"] is False


# --- v13c: relaxed spectral cap ---------------------------------------------------


def test_v13b_spectral_cap_parameter_caps_at_the_requested_value() -> None:
    model = GatedLocalCloudV13B(spectral_cap=1.25)
    with torch.no_grad():
        model.local_rec.weight.copy_(5.0 * torch.eye(4))
        capped = model._capped_weight("local_rec")
        assert abs(float(torch.linalg.matrix_norm(capped, ord=2)) - 1.25) <= 1e-5
        # Matrices already below the cap pass through unchanged.
        small = 0.5 * torch.eye(4)
        model.cloud_rec.weight.copy_(small)
        assert torch.allclose(model._capped_weight("cloud_rec"), small)
    report = model.lipschitz_report()
    assert report["spectral_cap"] == 1.25
    assert report["certified"] is False
    with pytest.raises(ValueError):
        GatedLocalCloudV13B(spectral_cap=0.0)
    with pytest.raises(ValueError):
        GatedLocalCloudV13B(spectral_cap=float("inf"))


def test_v13b_default_spectral_cap_is_bit_identical_to_the_original() -> None:
    rng = np.random.default_rng(43)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    default = train_gated_v13b(features, labels, seed=9, epochs=3)
    explicit = train_gated_v13b(features, labels, seed=9, epochs=3, spectral_cap=1.0)
    relaxed = train_gated_v13b(features, labels, seed=9, epochs=3, spectral_cap=1.25)
    assert model_hash(default) == model_hash(explicit)
    assert model_hash(default) != model_hash(relaxed)


def test_v13c_variant_is_registered_as_v13b_with_cap_1_25() -> None:
    assert set(v13_run.VARIANTS) == {"v13", "v13b", "v13c"}
    rng = np.random.default_rng(47)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    model = v13_run.VARIANTS["v13c"](features, labels, seed=3, epochs=2)
    assert isinstance(model, GatedLocalCloudV13B)
    assert model.spectral_cap == 1.25


def test_multi_variant_balanced_harness_runs_with_per_variant_gates() -> None:
    result = v13_run.evaluate_development(
        (61, 62),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=10,
        variant="v13b,v13c",
        split="balanced",
    )
    assert result["split"] == "balanced"
    assert result["variants"] == ["v13b", "v13c"]
    for panel in V13_PANELS:
        assert set(result["panel_means"][panel]) == {
            "v13b",
            "v13c",
            "v10",
            "elman3",
            "elman20",
            "gru20",
        }
    assert set(result["per_variant"]) == {"v13b", "v13c"}
    for name in ("v13b", "v13c"):
        block = result["per_variant"][name]
        assert set(block["gates"]) == {
            "G1_v13_within_5pct_of_gru20_all_panels",
            "G2_v13_beats_elman3_paired_lcb_all_panels",
            "G3_v13_heldout_accuracy_at_least_0_90",
            "G4_integrity",
        }
        assert result["overall"][name] in {"GO", "STOP"}
    ledgers = result["seed_rows"][0]["candidate_ledgers"]
    assert ledgers["v13b"]["lipschitz_report"]["spectral_cap"] == 1.0
    assert ledgers["v13c"]["lipschitz_report"]["spectral_cap"] == 1.25
    # Held-out cells recorded in the row must be the balanced ones.
    assert result["seed_rows"][0]["holdout_cells"] == list(holdout_cells_balanced(61))


# --- gate calculation accuracy ---------------------------------------------------


def test_development_result_has_registered_gate_keys_and_json_shape() -> None:
    result = v13_run.evaluate_development(
        (1, 2),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=5,
    )
    assert set(result["gates"]) == {
        "G1_v13_within_5pct_of_gru20_all_panels",
        "G2_v13_beats_elman3_paired_lcb_all_panels",
        "G3_v13_heldout_accuracy_at_least_0_90",
        "G4_integrity",
    }
    assert result["overall"] in {"GO", "STOP"}
    assert set(result["panel_means"]) == set(V13_PANELS)
    for panel in V13_PANELS:
        assert set(result["panel_means"][panel]) == set(v13_run.MODEL_NAMES)


def test_g1_gate_matches_the_ninety_five_percent_ratio_rule() -> None:
    result = v13_run.evaluate_development(
        (11, 12),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=6,
    )
    for panel in V13_PANELS:
        expected = result["panel_means"][panel]["v13"] >= 0.95 * result["panel_means"][panel]["gru20"]
        assert result["g1_per_panel"][panel] == expected
    assert result["gates"]["G1_v13_within_5pct_of_gru20_all_panels"] == all(
        result["g1_per_panel"].values()
    )


def test_g3_gate_is_an_absolute_heldout_threshold_not_a_relative_one() -> None:
    result = v13_run.evaluate_development(
        (21, 22),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    expected = result["panel_means"]["heldout"]["v13"] >= 0.90
    assert result["gates"]["G3_v13_heldout_accuracy_at_least_0_90"] == expected


def test_development_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        v13_run.evaluate_development(
            (5, 5),
            train_episodes=96,
            evaluation_episodes=32,
            epochs=2,
            bootstrap_samples=100,
            bootstrap_seed=8,
        )


def test_development_is_deterministic_for_repeated_seeds() -> None:
    first = v13_run.evaluate_development(
        (31, 32),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=9,
    )
    second = v13_run.evaluate_development(
        (31, 32),
        train_episodes=96,
        evaluation_episodes=32,
        epochs=2,
        bootstrap_samples=100,
        bootstrap_seed=9,
    )
    assert first["panel_means"] == second["panel_means"]
    assert first["gates"] == second["gates"]
    assert first["integrity"] == second["integrity"]
