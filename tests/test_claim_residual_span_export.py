import csv

from examples.pre_eq.claim_residual_benchmark import (
    BenchmarkRecord,
    claim_predictions_for_record,
    collect_claim_predictions,
    write_claim_csv,
)


def test_write_claim_csv_exports_claim_diagnostics(tmp_path) -> None:
    csv_path = tmp_path / "claims.csv"
    record = BenchmarkRecord(
        record_id="r1",
        answer="Paris is the capital of France. Berlin is the capital of France.",
        contexts=("Paris is the capital of France.",),
        is_hallucinated=True,
    )
    predictions = collect_claim_predictions_for_records((record,))

    write_claim_csv(csv_path, predictions)

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["record_id"] == "r1"
    assert rows[0]["claim_index"] == "0"
    assert rows[1]["claim_index"] == "1"
    assert float(rows[1]["action"]) > float(rows[0]["action"])


def collect_claim_predictions_for_records(records):
    all_predictions = []
    for record in records:
        all_predictions.extend(claim_predictions_for_record(record, semantic_evidence=True))
    return tuple(all_predictions)


def test_collect_claim_predictions_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"id":"r1","answer":"Paris is the capital of France.","context":"Paris is the capital of France.","is_hallucinated":false}\n',
        encoding="utf-8",
    )

    predictions = collect_claim_predictions(path, semantic_evidence=True)

    assert len(predictions) == 1
    assert predictions[0].record_id == "r1"
    assert predictions[0].accepted
