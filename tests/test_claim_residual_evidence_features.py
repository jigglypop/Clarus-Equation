from examples.pre_eq.claim_residual_benchmark import (
    BenchmarkRecord,
    NliScores,
    claim_predictions_for_record,
    collect_nli_pairs,
    deterministic_nli_scores,
    evidence_candidates,
    evaluate_span_predictions,
    extract_entities,
    extract_negations,
    extract_numbers,
    retrieve_evidence_sentences,
    semantic_claim_action,
    write_nli_pairs_jsonl,
)


def test_evidence_extractors_find_numbers_negations_and_entities() -> None:
    text = "Paris did not spend $1,200 in 2024. Berlin did."

    assert "$1200" in extract_numbers(text)
    assert "2024" in extract_numbers(text)
    assert "not" in extract_negations(text)
    assert "paris" in extract_entities(text)
    assert "berlin" in extract_entities(text)


def test_retrieve_evidence_sentences_ranks_relevant_sentence_first() -> None:
    contexts = (
        "Cats sleep often. Paris is the capital of France. Mars is red.",
    )

    evidence = retrieve_evidence_sentences("France has Paris as its capital.", contexts, top_k=1)

    assert evidence == ("Paris is the capital of France.",)


def test_semantic_claim_action_penalizes_entity_swap_more_than_paraphrase() -> None:
    contexts = ("Paris is the capital of France. The city is in Europe.",)

    faithful_action, faithful_accepted = semantic_claim_action(
        "France has Paris as its capital.",
        contexts,
    )
    swapped_action, swapped_accepted = semantic_claim_action(
        "Berlin is the capital of France.",
        contexts,
    )

    assert faithful_accepted == 1.0
    assert swapped_accepted == 0.0
    assert swapped_action > faithful_action


def test_deterministic_nli_axis_distinguishes_entailment_and_contradiction() -> None:
    entailed = deterministic_nli_scores(
        "Paris is the capital of France.",
        "Paris is the capital of France.",
    )
    contradicted = deterministic_nli_scores(
        "Berlin is the capital of France.",
        "Paris is the capital of France.",
    )

    assert entailed.entailment > entailed.contradiction
    assert contradicted.contradiction > entailed.contradiction


def test_evidence_candidates_prefer_true_evidence_over_distractor() -> None:
    contexts = (
        "Paris has a large population and museums. France lists Paris as its capital city.",
    )

    candidates = evidence_candidates("Paris is the capital of France.", contexts)

    assert candidates[0].text == "France lists Paris as its capital city."
    assert candidates[0].nli.entailment > candidates[0].nli.contradiction


def test_claim_predictions_identify_partial_hallucination_claim() -> None:
    record = BenchmarkRecord(
        record_id="r1",
        answer="Paris is the capital of France. Berlin is the capital of France.",
        contexts=("Paris is the capital of France.",),
        is_hallucinated=True,
    )

    claims = claim_predictions_for_record(record, semantic_evidence=True)

    assert len(claims) == 2
    assert claims[0].accepted
    assert not claims[1].accepted
    assert claims[1].action > claims[0].action


def test_span_metrics_count_claim_overlap_with_ragtruth_label() -> None:
    record = BenchmarkRecord(
        record_id="r1",
        answer="Paris is the capital of France. Berlin is the capital of France.",
        contexts=("Paris is the capital of France.",),
        is_hallucinated=True,
        labels=(
            {
                "start": 32,
                "end": 68,
                "label_type": "Evident Conflict",
            },
        ),
    )

    claims = claim_predictions_for_record(record, semantic_evidence=True)
    metrics = evaluate_span_predictions(claims)

    assert claims[1].span_label == "Evident Conflict"
    assert metrics.actual_positive == 1
    assert metrics.true_positive == 1


def test_external_nli_scores_override_claim_diagnostics() -> None:
    record = BenchmarkRecord(
        record_id="r1",
        answer="France has Paris as its capital.",
        contexts=("France has Paris as its capital.",),
        is_hallucinated=False,
    )

    supported = claim_predictions_for_record(
        record,
        semantic_evidence=True,
        nli_scores={("r1", 0): NliScores(entailment=0.98, contradiction=0.01, neutral=0.01)},
    )
    contradicted = claim_predictions_for_record(
        record,
        semantic_evidence=True,
        nli_scores={("r1", 0): NliScores(entailment=0.01, contradiction=0.98, neutral=0.01)},
    )

    assert supported[0].accepted
    assert not contradicted[0].accepted
    assert contradicted[0].action > supported[0].action
    assert contradicted[0].contradiction == 0.98


def test_nli_pair_export_uses_selected_evidence(tmp_path) -> None:
    record_path = tmp_path / "records.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    record_path.write_text(
        '{"id":"r1","answer":"Paris is the capital of France.","context":"Cats sleep. Paris is the capital of France.","is_hallucinated":false}\n',
        encoding="utf-8",
    )

    pairs = collect_nli_pairs(record_path)
    write_nli_pairs_jsonl(pairs_path, pairs)

    text = pairs_path.read_text(encoding="utf-8")
    assert '"record_id": "r1"' in text
    assert "Paris is the capital of France." in text
    assert "Cats sleep." in text
