import json
from dataclasses import asdict

import examples.pre_eq.claim_residual_run_all as run_all
from examples.pre_eq.claim_residual_benchmark import BenchmarkPrediction
from examples.pre_eq.claim_residual_run_all import (
    JsonlRun,
    RunAllSummary,
    external_strength_label,
    run_jsonl,
    run_synthetic,
    strength_label,
    summarize,
)


def test_strength_label_classifies_internal_modes() -> None:
    assert strength_label(exact_accuracy=1.0, hallucination_rate=0.0) == "strong-internal"
    assert strength_label(exact_accuracy=0.88, hallucination_rate=0.1) == "promising"
    assert strength_label(exact_accuracy=0.72, hallucination_rate=0.3) == "prototype"
    assert strength_label(exact_accuracy=0.5, hallucination_rate=0.5) == "weak"


def test_run_synthetic_returns_all_modes() -> None:
    runs = run_synthetic(seed=123, cases=20)

    assert {run.mode for run in runs} == {
        "adversarial",
        "noisy",
        "partial",
        "source",
        "graph",
        "missing",
    }
    assert all(run.exact_accuracy >= 0.0 for run in runs)


def test_run_jsonl_and_summary_reports_external_strength(tmp_path) -> None:
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
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    jsonl_runs = run_jsonl(path)
    summary = summarize((), jsonl_runs)

    assert len(jsonl_runs) == 1
    assert jsonl_runs[0].f1 == 1.0
    assert external_strength_label(jsonl_runs) == "sota-competitive"
    assert summary.external_strength == "sota-competitive"


def test_run_jsonl_forwards_scorer_options(monkeypatch, tmp_path) -> None:
    path = tmp_path / "mini.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    seen = {}

    def fake_raw_predictions(jsonl_path, **kwargs):
        seen["path"] = jsonl_path
        seen["kwargs"] = kwargs
        return (
            BenchmarkPrediction(
                record_id="r1",
                score=0.9,
                action=0.9,
                accepted_fraction=1.0,
                predicted_hallucinated=True,
                actual_hallucinated=True,
            ),
        )

    monkeypatch.setattr(run_all, "raw_predictions", fake_raw_predictions)

    runs = run_all.run_jsonl(
        path,
        accepted_fraction_threshold=0.25,
        max_context_chars=123,
        fast_lexical=True,
        response_level=True,
        enhanced_evidence=True,
        semantic_evidence=True,
        nli_evidence=True,
        nli_scores_jsonl=tmp_path / "nli.jsonl",
    )

    assert len(runs) == 1
    assert seen["path"] == path
    assert seen["kwargs"] == {
        "accepted_fraction_threshold": 0.25,
        "max_context_chars": 123,
        "fast_lexical": True,
        "response_level": True,
        "enhanced_evidence": True,
        "semantic_evidence": True,
        "nli_evidence": True,
        "nli_scores_jsonl": tmp_path / "nli.jsonl",
    }


def test_run_all_json_summary_schema_is_stable() -> None:
    summary = RunAllSummary(
        synthetic=(),
        jsonl=(
            JsonlRun(
                path="mini.jsonl",
                total=1,
                best_action_threshold=0.5,
                accuracy=1.0,
                balanced_accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                auroc=1.0,
                auprc=1.0,
            ),
        ),
        internal_strength="strong-internal",
        external_strength="sota-competitive",
        conclusion="ok",
    )

    data = json.loads(json.dumps(asdict(summary)))

    assert set(data) == {
        "synthetic",
        "jsonl",
        "internal_strength",
        "external_strength",
        "conclusion",
    }
    assert set(data["jsonl"][0]) == {
        "path",
        "total",
        "best_action_threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
    }
