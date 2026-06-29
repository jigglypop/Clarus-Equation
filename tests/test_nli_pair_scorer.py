import json

from examples.pre_eq.score_nli_pairs_transformers import (
    label_kind,
    load_pairs,
    normalize_nli_output,
    score_pairs,
    unpack_batch_outputs,
    write_scores,
)


class FakePipeline:
    def __call__(self, inputs, **kwargs):
        outputs = []
        for item in inputs:
            claim = item["text_pair"]
            if "Berlin" in claim:
                outputs.append(
                    [
                        {"label": "CONTRADICTION", "score": 0.8},
                        {"label": "ENTAILMENT", "score": 0.1},
                        {"label": "NEUTRAL", "score": 0.1},
                    ]
                )
            else:
                outputs.append(
                    [
                        {"label": "ENTAILMENT", "score": 0.85},
                        {"label": "CONTRADICTION", "score": 0.05},
                        {"label": "NEUTRAL", "score": 0.1},
                    ]
                )
        return outputs


def test_label_kind_maps_common_mnli_labels() -> None:
    assert label_kind("ENTAILMENT") == "entailment"
    assert label_kind("contradiction") == "contradiction"
    assert label_kind("LABEL_NEUTRAL") == "neutral"
    assert label_kind("unknown") == "neutral"


def test_normalize_nli_output_returns_probability_simplex() -> None:
    scores = normalize_nli_output(
        [
            {"label": "ENTAILMENT", "score": 2.0},
            {"label": "CONTRADICTION", "score": 1.0},
            {"label": "NEUTRAL", "score": 1.0},
        ]
    )

    assert scores.entailment == 0.5
    assert scores.contradiction == 0.25
    assert scores.neutral == 0.25


def test_score_pairs_and_write_scores_jsonl(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    pairs_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "record_id": "ok",
                    "claim_index": 0,
                    "claim": "Paris is the capital of France.",
                    "evidence": "Paris is the capital of France.",
                },
                {
                    "record_id": "bad",
                    "claim_index": 0,
                    "claim": "Berlin is the capital of France.",
                    "evidence": "Paris is the capital of France.",
                },
            )
        ),
        encoding="utf-8",
    )

    pairs = load_pairs(pairs_path)
    scores = score_pairs(pairs, FakePipeline(), batch_size=2)
    write_scores(scores_path, scores)
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]

    assert scores[0]["entailment"] > scores[0]["contradiction"]
    assert scores[1]["contradiction"] > scores[1]["entailment"]
    assert rows == list(scores)


def test_unpack_batch_outputs_keeps_single_pipeline_result_shape() -> None:
    output = [[{"label": "ENTAILMENT", "score": 0.9}]]

    assert unpack_batch_outputs(output, 1) == tuple(output)
