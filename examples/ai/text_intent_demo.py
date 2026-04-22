"""Demo for the topology-aware intent classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarus.text_intent import (
    LabeledIntentExample,
    TopologyIntentClassifier,
)


SAMPLES = (
    "이 구조가 왜 정합한지 설명해줘.",
    "의도분류 프로토타입 하나 구현해봐.",
    "최신 커밋 기준으로 bitfield 부분 분석해줘.",
    "3d, 강화학습, OCR 중 뭐가 가장 해볼만한지 비교해줘.",
    "이 코드 에러나는 부분 고쳐봐.",
    "다음 단계 계획 세워줘.",
    "이 모듈 리팩토링하고 다듬어줘.",
    "관련 논문들 좀 조사해봐.",
)

TRAIN_SET = [
    LabeledIntentExample("이게 왜 맞는지 설명해줘.", "explain"),
    LabeledIntentExample("개념과 원리를 설명해줘.", "explain"),
    LabeledIntentExample("증명을 유도해줘.", "explain"),
    LabeledIntentExample("프로토타입을 하나 구현해봐.", "implement"),
    LabeledIntentExample("새 모듈 하나 만들어줘.", "implement"),
    LabeledIntentExample("스크립트 작성해줘.", "implement"),
    LabeledIntentExample("최신 구조를 분석해줘.", "analyze"),
    LabeledIntentExample("정합한지 검토해줘.", "analyze"),
    LabeledIntentExample("완성도 평가해줘.", "analyze"),
    LabeledIntentExample("둘 중 뭐가 나은지 비교해줘.", "compare"),
    LabeledIntentExample("가장 해볼만한 걸 추천해줘.", "compare"),
    LabeledIntentExample("에러나는 부분 고쳐줘.", "debug"),
    LabeledIntentExample("버그를 수정해줘.", "debug"),
    LabeledIntentExample("왜 이게 안 돼?", "debug"),
    LabeledIntentExample("다음 단계 계획 세워줘.", "plan"),
    LabeledIntentExample("우선순위와 로드맵을 짜줘.", "plan"),
    LabeledIntentExample("이 모듈 리팩토링해줘.", "refactor"),
    LabeledIntentExample("코드 정리하고 개선해줘.", "refactor"),
    LabeledIntentExample("관련 논문 조사해줘.", "research"),
    LabeledIntentExample("주제 관련 리서치해봐.", "research"),
]


def _load_inputs(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        chunks = [part.strip() for part in content.splitlines() if part.strip()]
        return chunks if chunks else [content]
    data = sys.stdin.read().strip()
    if data:
        return [data]
    return list(SAMPLES)


def _print_prediction(prefix: str, sample: str, pred) -> None:
    print(prefix, sample)
    print("label", pred.label)
    print("confidence", f"{pred.confidence:.6f}")
    print("scores", " ".join(f"{k}={v:.6f}" for k, v in sorted(pred.scores.items())))
    if pred.probabilities:
        print(
            "probabilities",
            " ".join(f"{k}={v:.6f}" for k, v in sorted(pred.probabilities.items()))
        )
    print(
        "topology",
        " ".join(f"{k}={v:.6f}" for k, v in sorted(pred.topology_snapshot.items()))
    )
    if pred.matched_keywords:
        print(
            "matches",
            " ".join(f"{k}={','.join(v) if v else '-'}" for k, v in sorted(pred.matched_keywords.items()))
        )
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Topology-aware intent demo")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    args = parser.parse_args()

    base = TopologyIntentClassifier(dim=32)
    fitted = base.fit_centroids(TRAIN_SET)

    for idx, sample in enumerate(_load_inputs(args)):
        print("sample", idx)
        _print_prediction("rule_text", sample, base.predict(sample))
        _print_prediction("centroid_text", sample, fitted.predict(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
