"""Print the nested shape tree of processed Tafazoli spike-count fields."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _walk(
    value: Any,
    *,
    path: str,
    depth: int,
    counts: Counter[tuple[int, str, tuple[int, ...]]],
    samples: list[tuple[str, tuple[int, ...], str]],
) -> None:
    if isinstance(value, np.ndarray) and value.dtype == object:
        counts[(depth, "object", value.shape)] += 1
        for index, item in np.ndenumerate(value):
            _walk(
                item,
                path=f"{path}{index}",
                depth=depth + 1,
                counts=counts,
                samples=samples,
            )
        return

    array = np.asarray(value)
    counts[(depth, str(array.dtype), array.shape)] += 1
    if len(samples) < 30:
        samples.append((path, array.shape, str(array.dtype)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-file",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "data"
            / "tafazoli_compositional_v1"
            / "PFC_ClassifierData.mat"
        ),
    )
    args = parser.parse_args()

    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("SciPy is required for this probe") from error

    payload = loadmat(
        args.classifier_file,
        squeeze_me=True,
        struct_as_record=False,
    )
    classifier_options = payload["ClassifierOpts"]
    for name in (
        "DimpredictorsSpkCnt",
        "Dimresponse",
        "DimTrainStimInds",
        "UsedTrainTrials",
        "TrainStimInds",
        "Train2StimInds",
        "Train3StimInds",
    ):
        value = getattr(classifier_options, name)
        counts: Counter[tuple[int, str, tuple[int, ...]]] = Counter()
        samples: list[tuple[str, tuple[int, ...], str]] = []
        _walk(
            value,
            path="",
            depth=0,
            counts=counts,
            samples=samples,
        )
        print(
            f"FIELD {name} top={type(value).__name__} "
            f"shape={np.asarray(value, dtype=object).shape}"
        )
        for key, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"  count={count:<6} depth/type/shape={key}")
        for sample in samples[:12]:
            print(f"  sample={sample}")


if __name__ == "__main__":
    main()

