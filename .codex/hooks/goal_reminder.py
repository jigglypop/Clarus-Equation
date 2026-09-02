"""Print the current research target from paper/진전_원장.md §2 in a few lines.

Used by the Claude UserPromptSubmit hook and at Codex session start so the
target stays in context every turn. Output is deliberately short (< 700 chars).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "paper" / "진전_원장.md"
FIELDS = ("트랙", "현재 하위 목표", "완료 조건", "kill 조건", "마지막 갱신일")
LIMITS = {"트랙": 40, "현재 하위 목표": 120, "완료 조건": 90, "kill 조건": 60, "마지막 갱신일": 20}


def clean(cell: str) -> str:
    cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
    cell = cell.replace("**", "").strip()
    return re.sub(r"\s+", " ", cell)


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
    print("[표적] " + rows.get("트랙", "?") + " — " + rows.get("현재 하위 목표", "?"))
    if "완료 조건" in rows:
        print("[완료] " + rows["완료 조건"])
    print("[규율] 아이디어는 주차장에 한 줄, 실행 금지. 종료 전 진전 원장 §2·§7 갱신 + links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
