"""Check paper-ready figure captions for the H0 source-role draft."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DRAFT = ROOT / "docs" / "3_상수" / "12_H0_source_role_readout_paper_draft.md"

REQUIRED_CAPTION_PHRASES = [
    "Caption:",
    "Figure 1 visualizes the endpoint split",
    "assigned before H0 comparison",
    "local/high endpoint",
    "global/low endpoint",
    "Figure 2 extends the same selector",
    "bridge/intermediate family",
    "not a joint posterior fit",
]


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    failed = 0

    print("# H0 Paper Caption Gate")
    print()
    print("| caption requirement | status |")
    print("|---|---|")
    for phrase in REQUIRED_CAPTION_PHRASES:
        ok = phrase in text
        failed += 0 if ok else 1
        print(f"| `{phrase}` | {'PASS' if ok else 'FAIL'} |")

    figure_count = text.count("Caption:")
    print()
    print(f"caption count = {figure_count}")
    if figure_count < 2:
        failed += 1

    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: paper draft contains paper-ready captions for the H0 figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
