"""Check that the H0 readout paper draft contains the required spine."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DRAFT = ROOT / "docs" / "3_상수" / "12_H0_source_role_readout_paper_draft.md"

REQUIRED_SECTIONS = [
    "## Working title",
    "## Abstract draft",
    "## Plain-language significance",
    "## Core claim",
    "## Methods: source-role conductance",
    "## Data provenance",
    "## Figure package",
    "## Numeric results",
    "## Results narrative",
    "## Ablations",
    "## Reviewer objections and safeguards",
    "## Predictions and falsification",
    "## Required limitations",
    "## What the paper gains",
    "## Next tests",
]

REQUIRED_PHRASES = [
    "global/low",
    "bridge/intermediate",
    "local/high",
    "q_F",
    "C_L",
    "C_G",
    "source provenance",
    "static and flipped role ablations",
    "threshold sweeps",
    "branch-selection test",
    "primary gate",
    "scoped bridge abstraction",
    "branch-only",
    "replacement posterior",
    "full joint BAO/SN/TDCOSMO posterior refit",
    "Planck PR3 parameter covariance",
    "event-level posterior samples",
    "source roles are fixed first",
    "falsify",
    "TRGB/JAGB/CCHP",
    "BAO+SN inverse-distance ladder",
    "same universe can be read through different closures",
    "source-role problem",
    "Figure 1 visualizes the endpoint split",
    "Figure 2 extends the same selector",
    "not a joint posterior fit",
]


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    failed = 0

    print("# H0 Paper Draft Gate")
    print()
    print(f"draft = {DRAFT}")
    print()
    print("| requirement | status |")
    print("|---|---|")

    for section in REQUIRED_SECTIONS:
        ok = section in text
        failed += 0 if ok else 1
        print(f"| section `{section}` | {'PASS' if ok else 'FAIL'} |")

    for phrase in REQUIRED_PHRASES:
        ok = phrase in text
        failed += 0 if ok else 1
        print(f"| phrase `{phrase}` | {'PASS' if ok else 'FAIL'} |")

    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: H0 paper draft contains the required claim, figure, and limitation spine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
