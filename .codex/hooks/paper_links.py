"""Check paper/ Markdown cross-links, heading/HTML anchors, and orphan documents.

Usage (through the harness wrapper):

    .codex\\hooks\\python.cmd links            # broken file links fail; anchors/orphans warn
    .codex\\hooks\\python.cmd links --strict   # broken anchors also fail
    .codex\\hooks\\python.cmd links --json     # machine-readable summary only

Exit code 0 means no failing category. This is a document-integrity check only;
it says nothing about the mathematical or physical status of any claim.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.M)
HTML_ID_RE = re.compile(r'id="([^"]+)"')
FENCE_RE = re.compile(r"```.*?```", re.S)
ORPHAN_EXEMPT = {"README.md"}


def slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s\-가-힣]", "", text)
    # GitHub keeps one hyphen per removed space, so "A — B" becomes "a--b".
    return re.sub(r"\s", "-", text)


def anchors_of(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        cache[path] = set()
        return cache[path]
    found = {slug(m.group(1)) for m in HEADING_RE.finditer(text)}
    found |= {m.group(1) for m in HTML_ID_RE.finditer(text)}
    cache[path] = found
    return found


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    as_json = "--json" in argv
    if not PAPER_ROOT.is_dir():
        print("paper/ is missing", file=sys.stderr)
        return 2

    md_files = sorted(p for p in PAPER_ROOT.rglob("*.md"))
    cache: dict[Path, set[str]] = {}
    broken: list[tuple[str, str]] = []
    bad_anchor: list[tuple[str, str]] = []
    linked: set[Path] = set()
    total = 0

    for path in md_files:
        text = FENCE_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for match in LINK_RE.finditer(text):
            raw = match.group(1)
            if raw.startswith(("http://", "https://", "mailto:")):
                continue
            total += 1
            target, _, anchor = raw.partition("#")
            target = urllib.parse.unquote(target)
            full = (path.parent / target).resolve() if target else path.resolve()
            rel = path.relative_to(REPO_ROOT).as_posix()
            if not full.exists():
                broken.append((rel, raw))
                continue
            if full.suffix == ".md":
                linked.add(full)
            if anchor and full.suffix == ".md":
                wanted = urllib.parse.unquote(anchor)
                known = anchors_of(full, cache)
                if wanted not in known and slug(wanted) not in known:
                    bad_anchor.append((rel, raw))

    orphans = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in md_files
        if p.resolve() not in linked and p.name not in ORPHAN_EXEMPT
    ]

    failing = bool(broken) or (strict and bool(bad_anchor))
    summary = {
        "status": "FAIL" if failing else "PASS",
        "markdown_files": len(md_files),
        "internal_links": total,
        "broken_links": len(broken),
        "broken_anchors": len(bad_anchor),
        "orphans": len(orphans),
        "strict": strict,
    }
    if as_json:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
        return 1 if failing else 0

    for src, raw in broken:
        print(f"BROKEN  {src} -> {raw}")
    for src, raw in bad_anchor:
        print(f"ANCHOR  {src} -> {raw}")
    for rel in orphans:
        print(f"ORPHAN  {rel}")
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
