"""Check the plain-language significance spine of the H0 source-role paper."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DRAFT = ROOT / "docs" / "3_상수" / "12_H0_source_role_readout_paper_draft.md"

REQUIRED_PHRASES = [
    "## Plain-language significance",
    "The proposal is not that one experiment is simply wrong",
    "observation is anchored",
    "CMB and BAO look across the whole cosmic ruler system",
    "Distance ladders start from nearby anchors",
    "Standard sirens naturally sit between those cases",
    "the same universe can be read through different closures",
    "source-role problem",
]


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    failed = 0

    print("# H0 Paper Plain Significance Gate")
    print()
    print("| requirement | status |")
    print("|---|---|")
    for phrase in REQUIRED_PHRASES:
        ok = phrase in text
        failed += 0 if ok else 1
        print(f"| `{phrase}` | {'PASS' if ok else 'FAIL'} |")

    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: paper draft contains a plain-language significance spine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
