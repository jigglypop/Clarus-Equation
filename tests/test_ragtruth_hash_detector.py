from examples.pre_eq.train_ragtruth_hash_detector import (
    best_threshold,
    bucket,
    evaluate_model,
    features,
    load_model,
    predict_rows,
    save_model,
)


def row(answer: str, context: str, *, label: bool = False) -> dict:
    return {
        "answer": answer,
        "context": context,
        "contexts": [context],
        "is_hallucinated": label,
        "model": "m",
        "task_type": "qa",
    }


def test_hash_bucket_is_deterministic() -> None:
    assert bucket("Paris", 128) == bucket("Paris", 128)
    assert bucket("Paris", 128) != bucket("Berlin", 128)


def test_feature_vector_has_normalized_token_weights() -> None:
    feature_map = features(row("Paris Paris", "Paris"), hash_size=256, semantic_features=True)

    l2_norm = sum(value * value for value in feature_map.values()) ** 0.5
    assert l2_norm <= 1.0
    assert any(key.startswith("h:") for key in feature_map)
    assert feature_map["lexical_action"] < 1.0
    assert "max_contradiction_score" in feature_map
    assert "mean_entailment_score" in feature_map
    assert "claim_action_p90" in feature_map


def test_best_threshold_separates_supported_and_hallucinated_scores() -> None:
    threshold, f1 = best_threshold((0.1, 0.2, 0.9), (False, False, True))

    assert 0.2 < threshold <= 0.9
    assert f1 == 1.0


def test_save_load_predict_rows_are_identical(tmp_path) -> None:
    rows = (
        row("Paris is the capital of France.", "Paris is the capital of France."),
        row("Berlin is the capital of France.", "Paris is the capital of France.", label=True),
    )
    model = {
        "threshold": 0.5,
        "hash_size": 128,
        "semantic_features": True,
        "weights": {
            "bias": -2.0,
            "lexical_action": 4.0,
            "novelty_action": 2.0,
        },
    }
    path = tmp_path / "model.json"

    save_model(
        path,
        threshold=model["threshold"],
        hash_size=model["hash_size"],
        weights=model["weights"],
        semantic_features=model["semantic_features"],
    )
    loaded = load_model(path)

    assert loaded["schema_version"] == 2
    assert predict_rows(rows, model) == predict_rows(rows, loaded)


def test_saved_model_can_evaluate_rows_without_training(tmp_path) -> None:
    rows = (
        row("Paris is the capital of France.", "Paris is the capital of France."),
        row("Berlin is the capital of France.", "Paris is the capital of France.", label=True),
    )
    path = tmp_path / "model.json"
    save_model(
        path,
        threshold=0.36,
        hash_size=128,
        semantic_features=True,
        weights={
            "bias": -2.0,
            "lexical_action": 4.0,
            "novelty_action": 2.0,
        },
    )

    result = evaluate_model(path, rows)

    assert result["tp"] == 1.0
    assert result["tn"] == 1.0
    assert result["f1"] == 1.0


def test_tiny_fixture_scores_hallucination_above_supported() -> None:
    rows = (
        row("Paris is the capital of France.", "Paris is the capital of France."),
        row("Berlin is the capital of France.", "Paris is the capital of France.", label=True),
    )
    model = {
        "threshold": 0.5,
        "hash_size": 128,
        "semantic_features": True,
        "weights": {
            "bias": -2.0,
            "lexical_action": 4.0,
            "novelty_action": 2.0,
        },
    }

    supported, hallucinated = predict_rows(rows, model)

    assert hallucinated > supported
