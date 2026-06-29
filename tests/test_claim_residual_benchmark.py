import json

from examples.pre_eq.claim_residual_benchmark import (
    apply_thresholds,
    auprc,
    auroc,
    calibrate_thresholds,
    evaluate_jsonl,
    load_jsonl,
    parse_record,
    raw_predictions,
    write_error_csv,
)


def test_parse_record_accepts_common_benchmark_fields() -> None:
    record = parse_record(
        {
            "id": "r1",
            "response": "Paris is the capital of France.",
            "contexts": ["Paris is the capital and largest city of France."],
            "label": "supported",
        },
        0,
    )

    assert record.record_id == "r1"
    assert record.answer == "Paris is the capital of France."
    assert record.contexts == ("Paris is the capital and largest city of France.",)
    assert not record.is_hallucinated


def test_claim_residual_benchmark_scores_supported_and_hallucinated_jsonl(tmp_path) -> None:
    path = tmp_path / "mini.jsonl"
    rows = [
        {
            "id": "supported",
            "answer": "Paris is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": False,
        },
        {
            "id": "hallucinated",
            "answer": "Berlin is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": True,
        },
        {
            "id": "unsupported",
            "answer": "The project uses Redis for durable consensus.",
            "context": "The project stores consensus state in SQLite.",
            "is_hallucinated": True,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    records = load_jsonl(path)
    metrics, predictions = evaluate_jsonl(path)

    assert len(records) == 3
    assert len(predictions) == 3
    assert metrics.total == 3
    assert metrics.true_negative == 1
    assert metrics.true_positive == 2
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0


def test_claim_residual_benchmark_calibrates_threshold_and_ranking_metrics(tmp_path) -> None:
    path = tmp_path / "mini.jsonl"
    rows = [
        {
            "id": "supported",
            "answer": "Paris is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": False,
        },
        {
            "id": "entity_swap",
            "answer": "Berlin is the capital of France.",
            "context": "Paris is the capital of France.",
            "is_hallucinated": True,
        },
        {
            "id": "unsupported",
            "answer": "The project uses Redis for durable consensus.",
            "context": "The project stores consensus state in SQLite.",
            "is_hallucinated": True,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    raw = raw_predictions(path)
    calibration = calibrate_thresholds(raw)
    adjusted = apply_thresholds(
        raw,
        action_threshold=calibration.action_threshold,
        accepted_fraction_threshold=calibration.accepted_fraction_threshold,
    )

    assert calibration.metrics.f1 == 1.0
    assert calibration.metrics.balanced_accuracy == 1.0
    assert auroc(adjusted) == 1.0
    assert auprc(adjusted) == 1.0


def test_claim_residual_benchmark_exports_error_csv(tmp_path) -> None:
    path = tmp_path / "errors.csv"
    write_error_csv(path, ())

    assert path.read_text(encoding="utf-8").splitlines() == [
        "record_id,score,action,accepted_fraction,predicted_hallucinated,actual_hallucinated"
    ]
