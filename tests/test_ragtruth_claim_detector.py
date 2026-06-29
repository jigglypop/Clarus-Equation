from examples.pre_eq.train_ragtruth_claim_detector import (
    aggregate_response_scores,
    best_threshold,
    build_claim_examples,
    claim_offsets,
    claim_scores,
    label_spans,
    metrics,
    save_model,
    train,
)


def test_claim_offsets_and_label_alignment() -> None:
    row = {
        "id": "r1",
        "answer": "Paris is the capital of France. Berlin is the capital of France.",
        "context": "Paris is the capital of France.",
        "is_hallucinated": True,
        "labels": [
            {
                "start": 32,
                "end": 64,
                "label_type": "Evident Conflict",
            }
        ],
    }

    examples = build_claim_examples((row,), hash_size=128)

    assert claim_offsets(
        row["answer"],
        ("Paris is the capital of France.", "Berlin is the capital of France."),
    ) == ((0, 31), (32, 64))
    assert label_spans(row) == ((32, 64, "Evident Conflict"),)
    assert len(examples) == 2
    assert not examples[0].claim_label
    assert examples[1].claim_label


def test_best_threshold_and_response_metrics() -> None:
    threshold, f1 = best_threshold((0.1, 0.9), (False, True))
    result = metrics((0.1, 0.9), (False, True), threshold)

    assert 0.1 < threshold <= 0.9
    assert f1 == 1.0
    assert result["f1"] == 1.0


def test_claim_detector_scores_labeled_claim_above_supported(tmp_path) -> None:
    rows = (
        {
            "id": "ok",
            "answer": "Paris is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": False,
            "labels": [],
        },
        {
            "id": "bad",
            "answer": "Berlin is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": True,
            "labels": [{"start": 0, "end": 37, "label_type": "Evident Conflict"}],
        },
    )
    examples = build_claim_examples(rows, hash_size=128)
    weights = train(examples, epochs=8, lr=0.4, l2=0.0, seed=1)
    scores = claim_scores(examples, weights)
    responses = aggregate_response_scores(examples, scores)
    threshold, _ = best_threshold(
        tuple(score for _, score, _ in responses),
        tuple(label for _, _, label in responses),
    )
    result = metrics(
        tuple(score for _, score, _ in responses),
        tuple(label for _, _, label in responses),
        threshold,
    )
    path = tmp_path / "claim_model.json"

    save_model(path, threshold=threshold, hash_size=128, weights=weights, result=result)

    assert scores[1] > scores[0]
    assert result["f1"] == 1.0
    assert path.exists()
