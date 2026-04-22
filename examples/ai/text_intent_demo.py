"""Demo for the topology-aware intent classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarus.text_intent import TopologyIntentClassifier


SAMPLES = (
    "이 구조가 왜 정합한지 설명해줘.",
    "의도분류 프로토타입 하나 구현해봐.",
    "최신 커밋 기준으로 bitfield 부분 분석해줘.",
    "3d, 강화학습, OCR 중 뭐가 가장 해볼만한지 비교해줘.",
    "이 코드 에러나는 부분 고쳐봐.",
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Topology-aware intent demo")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    args = parser.parse_args()

    clf = TopologyIntentClassifier(dim=32)
    for idx, sample in enumerate(_load_inputs(args)):
        pred = clf.predict(sample)
        print("sample", idx)
        print("text", sample)
        print("label", pred.label)
        print("confidence", f"{pred.confidence:.6f}")
        print("scores", " ".join(f"{k}={v:.6f}" for k, v in sorted(pred.scores.items())))
        print("matches", " ".join(f"{k}={','.join(v) if v else '-'}" for k, v in sorted(pred.matched_keywords.items())))
        print(
            "topology",
            " ".join(f"{k}={v:.6f}" for k, v in sorted(pred.topology_snapshot.items()))
        )
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
