"""진전 원장 §2의 전체 목표와 현재 고리를 700자 미만으로 상기한다.

Claude의 사용자 입력 훅과 Codex의 세션 시작에서 같은 원장을 읽는다.
관측 완료 조건이 현재 하위 목표에 가려지지 않도록 별도 행을 출력한다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "paper" / "진전_원장.md"
QUESTIONS = REPO_ROOT / "ledger" / "questions.yaml"
FIELDS = ("최종 목표", "트랙", "현재 하위 목표", "완료 조건", "kill 조건", "마지막 갱신일")
LIMITS = {"최종 목표": 160, "트랙": 40, "현재 하위 목표": 120, "완료 조건": 90,
          "kill 조건": 60, "마지막 갱신일": 20}


def clean(cell: str) -> str:
    cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
    cell = cell.replace("**", "").strip()
    return re.sub(r"\s+", " ", cell)


def conjecture_line() -> str | None:
    """active 질문의 추측 카드·다음 사다리 단 한 줄. 이유: 식이 먼저라는 규율을 매 턴 상기시킨다."""
    try:
        import yaml

        data = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8-sig")) or {}
    except Exception:
        return None
    for q in data.get("questions") or []:
        if q.get("status") != "active":
            continue
        if q.get("card_status") == "기각":
            return f"[추측] {q.get('id')} 기존 카드 기각 — 새 공리·예측식과 반증 시험을 먼저 고정"
        if q.get("card"):
            ladder = q.get("ladder") or []
            done = sum(1 for s in ladder if s.get("status") in ("closed", "cited"))
            nxt = next((s for s in ladder if s.get("status") in ("open", "blocked")), None)
            step = f"다음 단 {nxt.get('step')}: {str(nxt.get('claim', ''))[:60]}" if nxt else "열린 단 없음"
            return f"[추측] {q.get('id')} {q.get('card_kind', '')} {str(q.get('formula') or '식 미등록')[:50]} 사다리 {done}/{len(ladder)} — {step}"
        if q.get("force_pivot"):
            return f"[추측] {q.get('id')} 카드 없음, force_pivot={q['force_pivot']} — 이번 attempt는 예측식·예산식 카드 작성"
        return f"[추측] {q.get('id')} 카드 없음 — 식을 먼저 세운다(conjecture-first)"
    return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not LEDGER.is_file():
        print("[표적] paper/진전_원장.md 없음 — 세션 표적을 먼저 고정하라.")
        return 0
    text = LEDGER.read_text(encoding="utf-8", errors="replace")
    section = re.search(r"^## 2\..*?(?=^## 3\.)", text, re.M | re.S)
    if not section:
        print("[표적] 진전 원장 §2를 찾지 못함.")
        return 0
    rows: dict[str, str] = {}
    for line in section.group(0).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] in FIELDS:
            value = clean(cells[1])
            limit = LIMITS[cells[0]]
            rows[cells[0]] = value if len(value) <= limit else value[: limit - 1] + "…"
    print("[최종 목표] " + rows.get("최종 목표", "미등록 — 진전 원장 §2의 최종 목표를 고정하라."))
    print("[표적] " + rows.get("트랙", "?") + " — " + rows.get("현재 하위 목표", "?"))
    if "완료 조건" in rows:
        print("[완료] " + rows["완료 조건"])
    line = conjecture_line()
    if line:
        print(line if len(line) <= 120 else line[:119] + "…")
    print("[전환] 공리·예측식·반증 시험·다음 증명 고리를 먼저 제시한다.")
    print("[규율] 새 아이디어는 주차장에 기록. 종료 전 진전 원장 §2·§7 갱신 + links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
