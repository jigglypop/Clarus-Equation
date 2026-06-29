"""SalienceCell — first gate. Decides how much the event matters and
seeds the field (risk / needs_memory / needs_search flags) that downstream
cells read. Cheap keyword heuristics now; a small classifier later.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind
from ..learn import LEARNED_ACTION_HINTS

_ACTION_HINTS = ("send", "email", "delete", "deploy", "pay", "transfer",
                 "발송", "보내", "삭제", "결제", "이체", "배포", "execute", "run ",
                 # obfuscated imperatives seen in evasion attempts
                 "넘겨", "내보내", "없애", "올려", "쏴", "뿌려", "치워",
                 "밀어", "날려", "처리해", "실행")
_MEMORY_HINTS = ("지난", "이전", "earlier", "last time", "remember", "회의",
                 "previous", "우리가", "결정")
_SEARCH_HINTS = ("최신", "찾아", "검색", "search", "lookup", "latest",
                 "current", "오늘", "뉴스", "price")
# knowledge-question markers: when present (and no tool is proposed) the
# event is an explanation request, so trap keywords must NOT trigger an
# action/route. This is the fail-safe against over-blocking.
_KNOWLEDGE_HINTS = ("뭐야", "무엇", "뭔지", "뭐라", "차이", "개념", "원리",
                    "설명해", "어떻게 동작", "의미가", "왜 중요", "종류 알려",
                    "같은 건 없")


class SalienceCell(Cell):
    name = "salience"

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        e = event.lower()
        tool = bool(field.get("tool"))
        # provenance: where did this text come from? 'user' is trusted;
        # 'tool'/'memory'/'search' content is DATA, not instructions.
        prov = field.get("provenance", "user")
        # A knowledge question with no proposed tool is never an action,
        # even if it contains trap words like '이메일' or '삭제'.
        knowledge = (not tool) and any(h in e for h in _KNOWLEDGE_HINTS)
        field["knowledge"] = knowledge
        # A proposed tool call is itself an action: it must reach the
        # PolicyCell regardless of the text. The allow-list (not the text)
        # decides whether it executes. Text keywords only count when this
        # is not a knowledge question.
        action_detected = not knowledge and (
            any(h in e for h in _ACTION_HINTS)
            or any(h in e for h in LEARNED_ACTION_HINTS))
        field["risk"] = tool or action_detected
        # An action found inside untrusted (non-user) content is an
        # indirect-injection attempt. Quarantine it: it can never satisfy
        # the allow-list, no matter what the embedded text claims.
        field["quarantine"] = (prov != "user") and action_detected
        field["needs_memory"] = (not knowledge
                                 and any(h in e for h in _MEMORY_HINTS))
        field["needs_search"] = (not knowledge
                                 and any(h in e for h in _SEARCH_HINTS))
        return Decision(
            cell=self.name, next="router", kind=EdgeKind.CONTROL,
            reason="scored salience; seeded field",
            data={k: field[k] for k in
                  ("risk", "needs_memory", "needs_search", "knowledge")},
        )
