"""SessionStart 훅: 원장 요약(active 질문, 최근 항목 3개, escalated)을 문맥에 넣는다.

이유: 헤드리스 루프는 매 세션이 새 문맥으로 시작하므로 상태를 파일에서 불러와야 한다.
출력은 40줄 이하. 실패해도 세션을 막지 않는다(exit 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import ledger

        lines = ledger.summary_lines(ledger.repo_root())
    except Exception as error:  # noqa: BLE001 - 세션 시작을 막지 않는다
        lines = [f"[원장] 요약 실패: {error}"]
    for line in lines[:40]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
