"""Normalize Markdown math delimiters without touching code spans or fences."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "paper"
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
TICKS = re.compile(r"`+")


def _replace_outside_code(
    line: str, inline_delimiter: str | None
) -> tuple[str, str | None, int]:
    """Replace inline math delimiters while preserving Markdown code spans."""

    pieces: list[str] = []
    cursor = 0
    replacement_count = 0
    for match in TICKS.finditer(line):
        segment = line[cursor : match.start()]
        if inline_delimiter is None:
            replacement_count += segment.count(r"\(") + segment.count(r"\)")
            segment = segment.replace(r"\(", "$").replace(r"\)", "$")
        pieces.append(segment)

        ticks = match.group(0)
        pieces.append(ticks)
        if inline_delimiter is None:
            inline_delimiter = ticks
        elif ticks == inline_delimiter:
            inline_delimiter = None
        cursor = match.end()

    tail = line[cursor:]
    if inline_delimiter is None:
        replacement_count += tail.count(r"\(") + tail.count(r"\)")
        tail = tail.replace(r"\(", "$").replace(r"\)", "$")
    pieces.append(tail)
    return "".join(pieces), inline_delimiter, replacement_count


def normalize_text(text: str) -> tuple[str, int, int]:
    """Return normalized text and block/inline replacement counts."""

    output: list[str] = []
    fence_character: str | None = None
    fence_width = 0
    inline_delimiter: str | None = None
    block_count = 0
    inline_count = 0

    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        newline = raw_line[len(content) :]
        fence_match = FENCE.match(content)
        if fence_match and inline_delimiter is None:
            marker = fence_match.group(1)
            character = marker[0]
            if fence_character is None:
                fence_character = character
                fence_width = len(marker)
            elif character == fence_character and len(marker) >= fence_width:
                fence_character = None
                fence_width = 0
            output.append(raw_line)
            continue

        if fence_character is not None:
            output.append(raw_line)
            continue

        stripped = content.strip()
        if inline_delimiter is None and stripped in {r"\[", r"\]"}:
            prefix = content[: len(content) - len(content.lstrip())]
            suffix = content[len(content.rstrip()) :]
            output.append(f"{prefix}$${suffix}{newline}")
            block_count += 1
            continue

        converted, inline_delimiter, replacements = _replace_outside_code(
            content, inline_delimiter
        )
        inline_count += replacements
        output.append(converted + newline)

    if inline_delimiter is not None:
        raise ValueError("unclosed inline code span")
    if fence_character is not None:
        raise ValueError("unclosed fenced code block")
    return "".join(output), block_count, inline_count


def markdown_files(inputs: list[Path]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for item in inputs:
        path = item if item.is_absolute() else REPO_ROOT / item
        if path.is_dir():
            files.update(path.rglob("*.md"))
        elif path.suffix.lower() == ".md":
            files.add(path)
    return tuple(sorted(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_ROOT])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    changed: list[Path] = []
    block_count = 0
    inline_count = 0
    for path in markdown_files(args.paths):
        raw = path.read_bytes()
        has_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        normalized, blocks, inlines = normalize_text(text)
        if normalized == text:
            continue
        changed.append(path)
        block_count += blocks
        inline_count += inlines
        if args.write:
            encoded = normalized.encode("utf-8")
            path.write_bytes((b"\xef\xbb\xbf" if has_bom else b"") + encoded)

    action = "normalized" if args.write else "would-normalize"
    print(
        f"{action}: files={len(changed)} block-delimiters={block_count} "
        f"inline-delimiters={inline_count}"
    )
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
