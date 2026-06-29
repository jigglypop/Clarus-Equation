from examples.pre_eq.convert_faithbench import (
    annotation_is_hallucinated,
    sample_is_hallucinated,
)


def test_faithbench_policy_can_exclude_questionable_labels() -> None:
    annotation = {"label": ["Questionable"]}

    assert annotation_is_hallucinated(annotation, policy="unwanted-or-questionable")
    assert not annotation_is_hallucinated(annotation, policy="unwanted-only")


def test_faithbench_policy_keeps_unwanted_positive() -> None:
    sample = {"annotations": [{"label": ["Unwanted.Intrinsic"]}]}

    assert sample_is_hallucinated(sample, policy="unwanted-only")
    assert sample_is_hallucinated(sample, policy="unwanted-or-questionable")
