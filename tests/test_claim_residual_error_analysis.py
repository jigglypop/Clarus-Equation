from examples.pre_eq.analyze_claim_residual_errors import (
    ErrorRow,
    base_confusion,
    quantiles,
    split_errors,
)


def test_error_analysis_splits_fp_and_fn() -> None:
    errors = (
        ErrorRow("fp", 0.8, 0.8, 0.0, True, False),
        ErrorRow("fn", 0.2, 0.2, 1.0, False, True),
    )

    split = split_errors(errors)

    assert split["FP"][0].record_id == "fp"
    assert split["FN"][0].record_id == "fn"


def test_error_analysis_confusion_counts_from_errors() -> None:
    records = {
        "a": {"is_hallucinated": True},
        "b": {"is_hallucinated": True},
        "c": {"is_hallucinated": False},
        "d": {"is_hallucinated": False},
    }
    errors = (
        ErrorRow("b", 0.2, 0.2, 1.0, False, True),
        ErrorRow("c", 0.8, 0.8, 0.0, True, False),
    )

    confusion = base_confusion(records, errors)

    assert confusion == {
        "total": 4,
        "actual_positive": 2,
        "actual_negative": 2,
        "fp": 1,
        "fn": 1,
        "tp": 1,
        "tn": 1,
    }


def test_error_analysis_quantiles() -> None:
    q = quantiles([1.0, 2.0, 3.0, 4.0, 5.0])

    assert q is not None
    assert q.count == 5
    assert q.mean == 3.0
    assert q.p50 == 3.0
