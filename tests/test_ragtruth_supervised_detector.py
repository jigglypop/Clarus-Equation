import random

from examples.pre_eq.train_ragtruth_supervised_detector import (
    Row,
    Weights,
    best_threshold,
    calibrate_task_thresholds,
    category_rates,
    metrics_with_task_thresholds,
    score,
    split_fit_validation,
    threshold_for_row,
)


def make_row(
    y: bool,
    lexical_action: float,
    *,
    model: str = "m",
    task_type: str = "qa",
) -> Row:
    return Row(
        y=y,
        lexical_action=lexical_action,
        model=model,
        task_type=task_type,
        model_task=f"{model}::{task_type}",
    )


def test_best_threshold_finds_f1_optimal_cutpoint() -> None:
    threshold, f1 = best_threshold((0.1, 0.2, 0.9), (False, False, True))

    assert 0.2 < threshold <= 0.9
    assert f1 == 1.0


def test_category_prior_smoothing_and_unseen_fallback() -> None:
    train = (
        make_row(False, 0.1, model="known"),
        make_row(True, 0.9, model="known"),
        make_row(True, 0.8, model="other"),
    )
    global_rate = sum(row.y for row in train) / len(train)
    model_rates = category_rates(train, "model")
    weights = Weights(
        lexical=0.0,
        model_prior=1.0,
        task_prior=0.0,
        model_task_prior=0.0,
        bias=0.0,
    )

    seen = score(
        make_row(False, 0.0, model="known"),
        weights,
        model_rates,
        {},
        {},
        global_rate,
    )
    unseen = score(
        make_row(False, 0.0, model="new"),
        weights,
        model_rates,
        {},
        {},
        global_rate,
    )

    assert 0.0 < model_rates["known"] < 1.0
    assert seen == model_rates["known"]
    assert unseen == global_rate


def test_validation_split_is_separate_when_possible() -> None:
    train = tuple(make_row(idx % 2 == 0, float(idx)) for idx in range(10))

    fit, validation = split_fit_validation(
        train,
        rng=random.Random(123),
        validation_fraction=0.3,
    )

    assert fit
    assert validation
    assert set(fit).isdisjoint(validation)
    assert len(fit) + len(validation) == len(train)


def test_task_thresholds_are_calibrated_with_global_fallback() -> None:
    rows = (
        make_row(False, 0.1, task_type="Summary"),
        make_row(True, 0.9, task_type="Summary"),
        make_row(False, 0.2, task_type="QA"),
    )

    thresholds = calibrate_task_thresholds(
        (0.1, 0.9, 0.2),
        rows,
        global_threshold=0.5,
        min_examples=2,
    )
    result = metrics_with_task_thresholds(
        (0.1, 0.9, 0.2),
        rows,
        task_thresholds=thresholds,
        global_threshold=0.5,
    )

    assert 0.1 < thresholds["Summary"] <= 0.9
    assert thresholds["QA"] == 0.5
    assert threshold_for_row(make_row(False, 0.0, task_type="NewTask"), thresholds, 0.5) == 0.5
    assert result["f1"] == 1.0
