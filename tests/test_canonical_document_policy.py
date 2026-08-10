from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "2_경로적분과_응용"
LECTURE_DIR = ROOT / "docs" / "1_강의"
CONSTANTS_DIR = ROOT / "docs" / "3_상수"
FORMAL_DIR = ROOT / "docs" / "9_등호이전"
REFERENCE_DIR = ROOT / "docs" / "참조"

PHYSICS_APPLICATION_MARKDOWN = tuple(
    ROOT / "docs" / "4_공학적_활용" / name
    for name in (
        "01_핵융합_설계.md",
        "02_양자오류보정.md",
        "03_진공에너지.md",
        "04_이론적_한계.md",
        "05_초전도체_설계.md",
        "06_공학식_총람.md",
    )
)

THEORY_DERIVATION_MARKDOWN = tuple(
    ROOT / "docs" / "5_유도" / name
    for name in (
        "01_Navier_Stokes.md",
        "04_Dark_Energy_Derivation.md",
        "06_Master_Action_Universal_Derivation.md",
        "07_Black_Hole_Derivation.md",
    )
)

CANONICAL_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "axium.md",
    ROOT / "docs" / "경로적분.md",
    ROOT / "docs" / "상수.md",
    ROOT / "docs" / "코어_독자_가이드.md",
)

OLD_STATUS_TAG = re.compile(
    r"\[(?:Exact|Selection|Bridge|Phenomenology|Open(?: test)?|Conditional|Rejected)\]"
)
MACHINE_VERDICT = re.compile(
    r"(?<![A-Za-z])(?:PASS|FAIL|CAUTION|WARN|Rejected)(?![A-Za-z])",
    re.IGNORECASE,
)
LEGACY_PROVENANCE = re.compile(
    r"(?:(?<![A-Za-z])(?:Exact|Selection|Bridge|Phenomenology|Conditional)(?![A-Za-z])|"
    r"유도\s*(?:레벨|수준)|(?:^|\s)[ABC]급(?:\s|$))"
)

# These are refuted parent branches.  Their witnesses stay executable in code,
# but the active theory text must not teach or tabulate the parent claims.
REMOVED_PARENT_PATTERNS = (
    ("Vcb LO", re.compile(r"V_?\{?cb\}?[^\n]{0,24}(?:\\mathrm\{LO\}|\bLO\b)")),
    ("Vus tree", re.compile(r"V_?\{?us\}?[^\n]{0,24}(?:\\mathrm\{tree\}|\btree\b)", re.I)),
    ("raw As", re.compile(r"A_s[^\n]{0,20}(?:\\mathrm\{raw\}|\braw\b)", re.I)),
    ("fixed rejected cosmology package", re.compile(r"37\.1003|R_?\{?3L\}?|3[- ]layer", re.I)),
    (
        "excluded 43.77 GeV parent",
        re.compile(r"43\.(?:7\d*|8\d*)\s*(?:\\,)?\s*(?:\\mathrm\{)?GeV", re.I),
    ),
    (
        "baryogenesis ansatz",
        re.compile(r"\\eta_?\{?B\}?[^\n]{0,24}(?:\\rm|\\mathrm)?\{?ansatz\}?", re.I),
    ),
    ("refuted alpha inverse identity", re.compile(r"4\s*\\pi\^?3\s*\+\s*\\pi\^?2\s*\+\s*\\pi")),
    ("refuted proton-electron identity", re.compile(r"6\s*\\pi\^?5")),
    ("raw primordial amplitude", re.compile(r"7\.84\s*(?:\\times\s*10\^?\{?-?9\}?|e-?9)", re.I)),
    ("fitted transition exponent", re.compile(r"h_?\{?\\(?:rm|mathrm)\s*tr\}?", re.I)),
    ("withdrawn H0 closure percentage", re.compile(r"99\.28\s*%")),
    (
        "underdetermined three-equation closure",
        re.compile(r"3\s*방정식\s*[,·]?\s*3\s*미지수"),
    ),
    (
        "random-variable range used as selected set",
        re.compile(r"\\Gamma_\{\\mathrm\{sel\}\}\s*(?::=|=)\s*\\Gamma\(\\Omega\)"),
    ),
    (
        "universal saddle ratio without prefactor",
        re.compile(
            r"\\frac\{A_\{?(?:NS|\\mathrm\{NS\})\}?\}\{A_\{?(?:S|\\mathrm\{S\})\}?\}"
            r"\s*=\s*e\^\{-?\\Delta S(?:_E)?/\\hbar\}"
        ),
    ),
    (
        "finite secretary probability set exactly to one over e",
        re.compile(r"P_\{?(?:\\mathrm\{)?success(?:\})?\}?\s*=\s*\\frac\s*\{?1\}?\s*\{?e\}?", re.I),
    ),
    (
        "incorrect exchange-symmetry factor sign",
        re.compile(r"g\(1-\\epsilon\)\s*=\s*g\(\\epsilon\)"),
    ),
)

POLICY_FILES = (
    "agents/ce-paper-writer.md",
    "agents/ce-math-verifier.md",
    "agents/ce-status-auditor.md",
    "agents/ce-physics-sourcer.md",
    "skills/ce-doc-write/SKILL.md",
    "skills/ce-closure-gate/SKILL.md",
    "skills/ce-dimensionless/SKILL.md",
    "skills/ce-validate/SKILL.md",
)

FORMAL_PROVENANCE = (
    "[정의]",
    "[정리]",
    "[공리]",
    "[산출]",
    "[경험식]",
    "[미완성]",
    "[예측]",
)

SALVAGED_THEORY_ANCHORS = (
    "hodge-closure",
    "portal-boundedness",
    "finite-lattice-measure",
    "noether-stress",
    "euclidean-semigroup",
    "multitype-poisson",
    "canonical-scalar-flrw",
    "flavor-realization",
    "vacuum-stress",
    "logistic-flow",
    "starobinsky-slow-roll",
    "complement-kernel",
    "multitype-threshold",
    "laplace-saddle",
    "secretary-limit",
    "dust-lambda-age",
    "oscillating-scalar-dust",
)


def _active_theory_documents() -> tuple[Path, ...]:
    paper = tuple(sorted(PAPER_DIR.glob("*.md")))
    lectures = tuple(sorted(LECTURE_DIR.glob("*.md")))
    constants = tuple(sorted(CONSTANTS_DIR.glob("*.md")))
    references = tuple(sorted(REFERENCE_DIR.glob("*.md")))
    # 07* is the AGI runtime vocabulary, where lower-case rejected/fail names
    # are machine-state terms rather than theory-document verdict badges.
    formal = tuple(
        path for path in sorted(FORMAL_DIR.glob("*.md")) if not path.name.startswith("07")
    )
    return (
        CANONICAL_MARKDOWN
        + PHYSICS_APPLICATION_MARKDOWN
        + THEORY_DERIVATION_MARKDOWN
        + lectures
        + paper
        + constants
        + formal
        + references
    )


def test_active_theory_uses_formal_provenance_not_machine_verdicts() -> None:
    violations: list[str] = []
    for path in _active_theory_documents():
        text = path.read_text(encoding="utf-8")
        for pattern, label in (
            (OLD_STATUS_TAG, "old status tag"),
            (MACHINE_VERDICT, "machine verdict"),
            (LEGACY_PROVENANCE, "legacy provenance vocabulary"),
        ):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)}")

    assert not violations, "\n".join(violations)


def test_refuted_parent_branches_are_absent_from_active_theory() -> None:
    violations: list[str] = []
    for path in _active_theory_documents():
        text = path.read_text(encoding="utf-8")
        for label, pattern in REMOVED_PARENT_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)}")

    assert not violations, "\n".join(violations)


def test_agent_policies_use_the_same_formal_provenance() -> None:
    violations: list[str] = []
    for relative in POLICY_FILES:
        codex = ROOT / ".codex" / relative
        claude = ROOT / ".claude" / relative
        codex_text = codex.read_text(encoding="utf-8")
        claude_text = claude.read_text(encoding="utf-8")

        if codex_text != claude_text:
            violations.append(f"policy mirror drift: {relative}")
        if relative in {
            "agents/ce-paper-writer.md",
            "agents/ce-status-auditor.md",
            "skills/ce-doc-write/SKILL.md",
        }:
            missing = [tag for tag in FORMAL_PROVENANCE if tag not in codex_text]
            if missing:
                violations.append(f".codex/{relative}: missing provenance: {missing}")
        for pattern, label in (
            (OLD_STATUS_TAG, "old status tag"),
            (MACHINE_VERDICT, "machine verdict"),
            (LEGACY_PROVENANCE, "legacy provenance vocabulary"),
        ):
            for match in pattern.finditer(codex_text):
                line = codex_text.count("\n", 0, match.start()) + 1
                violations.append(f".codex/{relative}:{line}: {label}: {match.group(0)}")

    assert not violations, "\n".join(violations)


def test_salvaged_theory_has_canonical_proofs_and_consistent_eft_signs() -> None:
    ledger = (REFERENCE_DIR / "핵심_정리_증명.md").read_text(encoding="utf-8")
    missing = [
        anchor
        for anchor in SALVAGED_THEORY_ANCHORS
        if f'<a id="{anchor}"></a>' not in ledger
    ]
    assert not missing, f"missing canonical proof anchors: {missing}"

    axiom = (ROOT / "docs" / "axium.md").read_text(encoding="utf-8")
    path_integral = (ROOT / "docs" / "경로적분.md").read_text(encoding="utf-8")
    for text in (axiom, path_integral):
        assert r"-\frac12(\nabla\phi)^2" in text
        assert r"M_{\rm Pl}^2-\xi\phi^2" in text
        assert r"\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}" in text
        assert "Hessian을 물질장으로 동일시" in text or "Hessian 기댓값을 stress tensor로 동일시" in text


def test_reduced_planck_convention_is_consistent_in_cosmology_summary() -> None:
    cosmology = (CONSTANTS_DIR / "9_우주론_수식_의미와_후보.md").read_text(
        encoding="utf-8"
    )
    assert r"M_{\rm Pl}^{-2}=8\pi G" in cosmology
    assert r"S_{\rm dS}=8\pi^2" in cosmology
    assert r"\rho_{\rm crit}" in cosmology
    assert r"=3H^2M_{\rm Pl}^2" in cosmology
