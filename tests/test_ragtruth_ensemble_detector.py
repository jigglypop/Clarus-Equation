from examples.pre_eq.score_ragtruth_ensemble_detector import (
    align_scores,
    blend_score,
    evaluate,
    tune_alpha,
)


def test_align_scores_preserves_labels_and_scores() -> None:
    aligned = align_scores(
        (("a", 0.1, False), ("b", 0.9, True)),
        (("a", 0.2, False), ("b", 0.8, True)),
    )

    assert aligned == (("a", 0.1, 0.2, False), ("b", 0.9, 0.8, True))


def test_tune_alpha_finds_perfect_blend_on_fixture() -> None:
    aligned = (
        ("a", 0.1, 0.9, False),
        ("b", 0.9, 0.1, True),
    )

    alpha, threshold, f1 = tune_alpha(aligned)
    result = evaluate(aligned, alpha=alpha, threshold=threshold)

    assert alpha > 0.5
    assert f1 == 1.0
    assert result["f1"] == 1.0
    assert blend_score(0.9, 0.1, alpha) > blend_score(0.1, 0.9, alpha)
