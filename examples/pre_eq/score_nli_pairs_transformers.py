"""Score exported claim/evidence pairs with a Transformers NLI model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from transformers import pipeline as transformers_pipeline
except Exception:
    transformers_pipeline = None

from examples.pre_eq.claim_residual_benchmark import NliScores  # noqa: E402


NliPipeline = Callable[..., Any]


def load_pairs(path: Path, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pairs.append(json.loads(line))
            if limit is not None and len(pairs) >= limit:
                break
    return tuple(pairs)


def batched(items: Sequence[dict[str, Any]], size: int) -> Iterable[tuple[dict[str, Any], ...]]:
    for start in range(0, len(items), size):
        yield tuple(items[start : start + size])


def label_kind(label: str) -> str:
    normalized = label.lower()
    if "entail" in normalized:
        return "entailment"
    if "contrad" in normalized:
        return "contradiction"
    if "neutral" in normalized:
        return "neutral"
    return "neutral"


def normalize_nli_output(output: Any) -> NliScores:
    if isinstance(output, Mapping):
        rows = (output,)
    else:
        rows = tuple(output)
    scores = {
        "entailment": 0.0,
        "contradiction": 0.0,
        "neutral": 0.0,
    }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        kind = label_kind(str(row.get("label", "")))
        scores[kind] = max(scores[kind], float(row.get("score", 0.0)))
    total = scores["entailment"] + scores["contradiction"] + scores["neutral"]
    if total <= 0.0:
        return NliScores(entailment=0.0, contradiction=0.0, neutral=1.0)
    return NliScores(
        entailment=scores["entailment"] / total,
        contradiction=scores["contradiction"] / total,
        neutral=scores["neutral"] / total,
    )


def unpack_batch_outputs(outputs: Any, expected: int) -> tuple[Any, ...]:
    if isinstance(outputs, Sequence) and len(outputs) == expected:
        return tuple(outputs)
    if expected == 1:
        return (outputs,)
    raise ValueError(f"expected {expected} NLI outputs, got {type(outputs).__name__}")


def score_pairs(
    pairs: Sequence[dict[str, Any]],
    nli_pipeline: NliPipeline,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], ...]:
    scored = []
    for batch in batched(pairs, batch_size):
        inputs = [
            {
                "text": str(pair.get("evidence", "")),
                "text_pair": str(pair.get("claim", "")),
            }
            for pair in batch
        ]
        try:
            outputs = nli_pipeline(inputs, truncation=True, batch_size=batch_size)
        except TypeError:
            outputs = nli_pipeline(inputs)
        for pair, output in zip(batch, unpack_batch_outputs(outputs, len(batch))):
            scores = normalize_nli_output(output)
            scored.append(
                {
                    "record_id": str(pair["record_id"]),
                    "claim_index": int(pair["claim_index"]),
                    "entailment": scores.entailment,
                    "contradiction": scores.contradiction,
                    "neutral": scores.neutral,
                }
            )
    return tuple(scored)


def write_scores(path: Path, scores: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(dict(score), ensure_ascii=False) + "\n")


def build_pipeline(model: str, *, device: int) -> NliPipeline:
    if transformers_pipeline is None:
        raise RuntimeError(
            "transformers is not installed in this Python environment. "
            "Install project dependencies or run this script in an environment with torch/transformers."
        )
    return transformers_pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        top_k=None,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs, limit=args.limit)
    nli_pipeline = build_pipeline(args.model, device=args.device)
    scores = score_pairs(pairs, nli_pipeline, batch_size=args.batch_size)
    write_scores(args.output, scores)
    print("# Transformers NLI pair scoring")
    print(f"pairs {len(pairs)}")
    print(f"scores {len(scores)}")
    print(f"model {args.model}")
    print(f"output {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
