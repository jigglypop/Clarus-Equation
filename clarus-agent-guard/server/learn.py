"""Learned action hints — the mutable memory that replay writes into.

SalienceCell consults LEARNED_ACTION_HINTS in addition to its static list.
Empty at boot; replay() populates it from past blocked actions so the
guard adapts to obfuscated verbs it has *seen get gated* — without an LLM
and without a human editing the keyword list.
"""

from __future__ import annotations

LEARNED_ACTION_HINTS: set[str] = set()
