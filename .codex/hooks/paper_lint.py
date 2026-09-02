"""Advisory lint for paper/: process narration and English-noun density.

    .codex\\hooks\\python.cmd lint            # report per file
    .codex\\hooks\\python.cmd lint --json     # summary only
    .codex\\hooks\\python.cmd lint --strict   # exit 1 if any file exceeds the ratio ceiling

Two measurements, both about readability for a paper reader, never about the
truth or status of any claim:

1. process words — phrases that describe the research process (sessions,
   commits, next deliverables, snapshots) rather than results; they belong in
   _workspace/ notes.
2. latin ratio — share of Latin-script word tokens among all word tokens after
   removing math, code, links and HTML. High values mean the prose leans on
   untranslated English nouns.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
EXEMPT = {"진전_원장.md", "참고문헌.md"}
RATIO_CEILING = 0.25
PROCESS_WORDS = (
    "이번 세션", "다음 최소 산출물", "다음 산출물", "진행상황", "진행 중 경로",
    "snapshot", "_workspace", "커밋", "40-final", "12-routes", "stage 파일",
    "artifacts/", "시도했으나", "시도했지만", "TODO", "run-id", "CE_RUN",
)
STRIP_PATTERNS = (
    re.compile(r"```.*?```", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\$[^$\n]+\$"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"<[^>\n]+>"),
)
LATIN = re.compile(r"[A-Za-z][A-Za-z\-']+")
HANGUL = re.compile(r"[가-힣]+")


def strip(text: str) -> str:
    for pat in STRIP_PATTERNS:
        text = pat.sub(" ", text)
    return text


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    strict = "--strict" in argv
    as_json = "--json" in argv
    rows = []
    for path in sorted(PAPER_ROOT.rglob("*.md")):
        if path.name in EXEMPT:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = strip(raw)
        latin = len(LATIN.findall(body))
        hangul = len(HANGUL.findall(body))
        ratio = latin / (latin + hangul) if (latin + hangul) else 0.0
        hits = {w: raw.count(w) for w in PROCESS_WORDS if w in raw}
        rows.append({
            "file": path.relative_to(REPO_ROOT).as_posix(),
            "latin_ratio": round(ratio, 3),
            "latin": latin,
            "hangul": hangul,
            "process_hits": sum(hits.values()),
            "process_words": hits,
        })
    over = [r for r in rows if r["latin_ratio"] > RATIO_CEILING]
    summary = {
        "status": "FAIL" if (strict and over) else "PASS",
        "files": len(rows),
        "ratio_ceiling": RATIO_CEILING,
        "files_over_ceiling": len(over),
        "files_with_process_words": sum(1 for r in rows if r["process_hits"]),
        "strict": strict,
    }
    if not as_json:
        print("== latin ratio (top 20) ==")
        for r in sorted(rows, key=lambda r: -r["latin_ratio"])[:20]:
            flag = "OVER " if r["latin_ratio"] > RATIO_CEILING else "     "
            print(f"{flag}{r['latin_ratio']:.3f}  {r['file']}")
        print("== process words (top 15) ==")
        for r in sorted(rows, key=lambda r: -r["process_hits"])[:15]:
            if r["process_hits"]:
                words = ", ".join(f"{k}×{v}" for k, v in r["process_words"].items())
                print(f"{r['process_hits']:3d}  {r['file']}  [{words}]")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
