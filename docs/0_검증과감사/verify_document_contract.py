"""Verify the bounded CE Markdown and canonical-document contract.

This checker deliberately does *not* claim to prove the physics in the corpus.
It checks Markdown/math-delimiter hygiene, raw TeX leakage, a small set of
canonical presentation contracts, and quarantining of known superseded values.
Canonical arithmetic is recomputed separately by ``verify_numeric_consistency``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
DOCS_ROOT = HERE.parent
EXPECTED_MARKDOWN_COUNT = 212

MANIFEST_PATH = HERE / "CANONICAL_NUMERIC_MANIFEST_2026-08-06.json"
COMPLETION_PATH = HERE / "FULL_CONSISTENCY_COMPLETION_LOOP_2026-08-06.md"
BASELINE_PATH = HERE / "OBSERVATIONAL_BASELINE_2026-08-06.md"
COSMOLOGY_AUDIT_PATH = HERE / "우주론_양자론_루프_감사.md"
ISSUES_PATH = HERE / "MATHEMATICAL_PHYSICS_ISSUES.md"
README_PATH = DOCS_ROOT / "README.md"
MAIN_PATH = DOCS_ROOT / "경로적분.md"
PROOF_STATUS_PATH = HERE / "PROOF_STATUS_MATRIX.md"
BOOTSTRAP_PATH = DOCS_ROOT / "3_상수" / "3_부트스트랩.md"
H0_ROLE_PATH = DOCS_ROOT / "3_상수" / "9_우주론_수식_의미와_후보.md"
AGI_EQUATION_PATH = DOCS_ROOT / "7_AGI" / "12_Equation.md"
AGI_VERIFICATION_PATH = DOCS_ROOT / "7_AGI" / "13_Verification.md"
AGI_AGENT_LOOP_PATH = DOCS_ROOT / "7_AGI" / "17_AgentLoop.md"
AGI_CODE_MAP_PATH = DOCS_ROOT / "7_AGI" / "18_CodeMap.md"
AGI_ARCHITECTURE_PATH = DOCS_ROOT / "7_AGI" / "2_Architecture.md"
AGI_CONSCIOUSNESS_PATH = DOCS_ROOT / "7_AGI" / "7_Consciousness.md"
AGI_SLEEP_PATH = DOCS_ROOT / "7_AGI" / "3_Sleep.md"
AGI_ROADMAP_PATH = DOCS_ROOT / "7_AGI" / "8_Roadmap.md"


@dataclass(frozen=True, order=True)
class Failure:
    relative_path: str
    line: int
    kind: str
    detail: str

    def render(self) -> str:
        location = self.relative_path
        if self.line:
            location += f":{self.line}"
        return f"[{self.kind}] {location}: {self.detail}"


def relative(path: Path) -> str:
    try:
        return path.relative_to(DOCS_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def add_failure(
    failures: list[Failure], path: Path, line: int, kind: str, detail: str
) -> None:
    failures.append(Failure(relative(path), line, kind, detail))


def read_utf8(path: Path, failures: list[Failure]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        add_failure(failures, path, 0, "REQUIRED_FILE", "required file is missing")
    except UnicodeDecodeError as exc:
        add_failure(failures, path, exc.start, "UTF8", f"invalid UTF-8: {exc}")
    return None


FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def mask_fenced_code(
    path: Path, text: str, failures: list[Failure]
) -> list[str]:
    """Return same-length lines with fenced-code content replaced by spaces."""

    output: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    fence_start = 0

    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE_OPEN_RE.match(line)
        if fence_character is None:
            if match:
                token = match.group(1)
                fence_character = token[0]
                fence_length = len(token)
                fence_start = number
                output.append(" " * len(line))
            else:
                output.append(line)
            continue

        closing = re.match(
            rf"^\s*{re.escape(fence_character)}{{{fence_length},}}\s*$", line
        )
        output.append(" " * len(line))
        if closing:
            fence_character = None
            fence_length = 0
            fence_start = 0

    if fence_character is not None:
        add_failure(
            failures,
            path,
            fence_start,
            "CODE_FENCE",
            f"unclosed {fence_character * fence_length} code fence",
        )
    return output


def mask_inline_code_and_targets(line: str) -> str:
    """Mask inline code and Markdown link destinations without changing offsets."""

    chars = list(line)

    # CommonMark code spans can use arbitrary backtick-run lengths.  This
    # line-local masking is intentionally conservative: an unclosed run masks
    # the rest of the line so command-like example text is not treated as prose.
    cursor = 0
    while cursor < len(line):
        if line[cursor] != "`":
            cursor += 1
            continue
        end_of_run = cursor
        while end_of_run < len(line) and line[end_of_run] == "`":
            end_of_run += 1
        token = line[cursor:end_of_run]
        closing = line.find(token, end_of_run)
        stop = len(line) if closing < 0 else closing + len(token)
        for index in range(cursor, stop):
            chars[index] = " "
        cursor = stop

    masked = "".join(chars)
    target_patterns = (
        re.compile(r"\]\([^\n)]*\)"),
        re.compile(r"<(?:(?:https?|mailto|app)://|[A-Za-z]:\\)[^>]*>"),
    )
    for pattern in target_patterns:
        for match in pattern.finditer(masked):
            for index in range(match.start(), match.end()):
                chars[index] = " "
        masked = "".join(chars)
    return masked


def escaped(text: str, index: int) -> bool:
    count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        count += 1
        cursor -= 1
    return count % 2 == 1


def looks_like_path(line: str, slash_index: int) -> bool:
    """Exclude Windows/relative paths from raw-TeX command diagnostics."""

    prefix = line[max(0, slash_index - 80) : slash_index + 1]
    if re.search(r"(?:[A-Za-z]:|\.{1,2})\\(?:[^\s`<>|]+\\)*$", prefix):
        return True
    if slash_index > 0 and line[slash_index - 1] in "/\\:":
        return True
    suffix = line[slash_index : slash_index + 120]
    if len(re.findall(r"\\[A-Za-z0-9_.-]+", suffix)) >= 2:
        return True
    before = line[:slash_index]
    after = line[slash_index:]
    if re.search(r"(?:^|\s)[A-Za-z0-9_.-]+$", before) and re.match(
        r"\\[A-Za-z0-9_.-]+(?:\\|/[A-Za-z0-9_.-]*|\.[A-Za-z0-9]+)", after
    ):
        return True
    return False


def scan_math_and_raw_tex(
    path: Path, text: str, failures: list[Failure]
) -> None:
    lines = mask_fenced_code(path, text, failures)
    display_mode: str | None = None
    display_start = 0

    for number, original_line in enumerate(lines, 1):
        line = mask_inline_code_and_targets(original_line)
        inline_parenthesis = False
        inline_dollar = False
        cursor = 0

        while cursor < len(line):
            if display_mode == "bracket":
                if line.startswith(r"\]", cursor) and not escaped(line, cursor):
                    display_mode = None
                    display_start = 0
                    cursor += 2
                else:
                    cursor += 1
                continue

            if display_mode == "dollar":
                if line.startswith("$$", cursor) and not escaped(line, cursor):
                    display_mode = None
                    display_start = 0
                    cursor += 2
                else:
                    cursor += 1
                continue

            if inline_parenthesis:
                if line.startswith(r"\)", cursor) and not escaped(line, cursor):
                    inline_parenthesis = False
                    cursor += 2
                else:
                    cursor += 1
                continue

            if inline_dollar:
                if line[cursor] == "$" and not escaped(line, cursor):
                    inline_dollar = False
                cursor += 1
                continue

            if line.startswith(r"\[", cursor) and not escaped(line, cursor):
                display_mode = "bracket"
                display_start = number
                cursor += 2
                continue
            if line.startswith(r"\]", cursor) and not escaped(line, cursor):
                add_failure(
                    failures, path, number, "DISPLAY_MATH", r"stray \] delimiter"
                )
                cursor += 2
                continue
            if line.startswith("$$", cursor) and not escaped(line, cursor):
                display_mode = "dollar"
                display_start = number
                cursor += 2
                continue
            if line.startswith(r"\(", cursor) and not escaped(line, cursor):
                inline_parenthesis = True
                cursor += 2
                continue
            if line.startswith(r"\)", cursor) and not escaped(line, cursor):
                add_failure(
                    failures, path, number, "INLINE_MATH", r"stray \) delimiter"
                )
                cursor += 2
                continue
            if line[cursor] == "$" and not escaped(line, cursor):
                inline_dollar = True
                cursor += 1
                continue

            if line[cursor] == "\\" and not escaped(line, cursor):
                match = re.match(r"\\([A-Za-z]+)", line[cursor:])
                if match:
                    command = match.group(1)
                    if not looks_like_path(line, cursor):
                        add_failure(
                            failures,
                            path,
                            number,
                            "RAW_TEX",
                            f"\\{command} appears outside a math delimiter",
                        )
                    cursor += len(match.group(0))
                    continue
                if cursor + 1 < len(line) and line[cursor + 1] in ",;:!":
                    add_failure(
                        failures,
                        path,
                        number,
                        "RAW_TEX",
                        f"\\{line[cursor + 1]} appears outside a math delimiter",
                    )
                    cursor += 2
                    continue
            cursor += 1

        if inline_parenthesis:
            add_failure(
                failures,
                path,
                number,
                "INLINE_MATH",
                r"\( was not closed on the same line",
            )
        if inline_dollar:
            add_failure(
                failures,
                path,
                number,
                "INLINE_DOLLAR",
                "single-dollar inline math was not closed on the same line",
            )

    if display_mode is not None:
        delimiter = r"\[" if display_mode == "bracket" else "$$"
        add_failure(
            failures,
            path,
            display_start,
            "DISPLAY_MATH",
            f"unclosed {delimiter} display-math block",
        )


QUARANTINE_MARKERS = re.compile(
    r"(?:\blegacy\b|\bcanonical_drift\b|\bhistor(?:y|ical)\b|\bold\b|\bobsolete\b|"
    r"\bdeprecated\b|\bsuperseded\b|\bfrozen\b|\bfixture\b|\barchive\b|"
    r"과거|구형|구버전|이전값|옛|폐기|동결|역사|레거시|"
    r"기록용|비정본|참고용|재현용|현재\s*대표값이\s*아니|현행값이\s*아니)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ForbiddenRule:
    name: str
    pattern: re.Pattern[str]
    detail: str
    context: re.Pattern[str] | None = None


FORBIDDEN_RULES = (
    ForbiddenRule(
        "INSTANTON_SUBSTITUTION",
        re.compile(r"8\\pi\^2\}\{\\alpha_s\}"),
        "incorrect instanton substitution 8pi^2/alpha_s is not quarantined",
    ),
    ForbiddenRule(
        "UNIT_CONVERSION",
        re.compile(r"45\s*\\;?\s*\\text\{neV\}", re.IGNORECASE),
        "known MeV-to-eV conversion error is not quarantined",
    ),
    ForbiddenRule(
        "BOOTSTRAP_ALIAS",
        re.compile(r"\\alpha_s\s*=\s*0\.04865\d*", re.IGNORECASE),
        "bootstrap fraction is actively mislabeled as alpha_s",
    ),
    ForbiddenRule(
        "PHYSICAL_ANGLE_ALIAS",
        re.compile(
            r"(?:s\s*_?\s*\{?W\}?\s*(?:\^\s*\{?2\}?|²)|"
            r"(?:\\sin|sin)\s*(?:\^\s*\{?2\}?|²)\s*"
            r"(?:\\theta|theta)\s*_?\s*\{?W\}?)\s*"
            r"(?:=|:=|≈|\\simeq)\s*0\.231509\d*",
            re.IGNORECASE,
        ),
        "Track-A s_A^2 is actively relabeled as a physical weak-angle value",
    ),
    ForbiddenRule(
        "OLD_SCALAR_KERNEL",
        re.compile(r"249\s*/\s*135|134\.9(?:0*)?", re.IGNORECASE),
        "superseded scalar-kernel/current-result value is not quarantined",
        re.compile(r"(?:scalar|kernel|g\s*[-−]?\s*2|a_?\{?\\?mu\}?|뮤온)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_LIGHT_MASS",
        re.compile(r"29\.6475\d*|29\.648(?:0*)?|29\.65(?:0*)?", re.IGNORECASE),
        "superseded light-bridge mass is not quarantined",
        re.compile(r"(?:MeV|mass|massive|massless|질량|mediator|scalar|pole)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_PORTAL_MASS",
        re.compile(r"43\.767\d*|43\.77(?:0*)?", re.IGNORECASE),
        "superseded portal mass is not quarantined",
        re.compile(r"(?:GeV|portal|포탈|Higgs|mass|질량)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_BAO_CHI2",
        re.compile(r"37\.1003\d*", re.IGNORECASE),
        "superseded BAO chi-square snapshot is not quarantined",
        re.compile(r"(?:BAO|DESI|chi|chi\^2|\\chi|우주론)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_EH_RD",
        re.compile(r"151\.3188\d*", re.IGNORECASE),
        "superseded Eisenstein--Hu sound horizon is not quarantined",
        re.compile(r"(?:Eisenstein|Hu|r_d|r\s*_\s*d|Mpc|sound|BAO)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_PORTAL_BR",
        re.compile(r"0\.825312\d*|0\.772(?:0*)?", re.IGNORECASE),
        "superseded portal invisible branching ratio is not quarantined",
        re.compile(r"(?:BR|branch|inv|invisible|portal|포탈|Higgs)", re.IGNORECASE),
    ),
    ForbiddenRule(
        "OLD_PORTAL_LIMIT",
        re.compile(
            r"(?<![\d.])0\.11(?!\d)|(?<![\d.])11\s*%", re.IGNORECASE
        ),
        "superseded direct invisible limit is not quarantined",
        re.compile(
            r"(?:\bBR\b|\bbranch(?:ing)?\b|\binv(?:isible)?\b|\bportal\b|"
            r"포탈|\bHiggs\b|상한|\blimit\b)",
            re.IGNORECASE,
        ),
    ),
)


def scan_forbidden_values(
    path: Path, text: str, failures: list[Failure]
) -> None:
    prose_lines = mask_fenced_code(path, text, failures=[])
    total = len(prose_lines)
    ledger_scoped_history = False
    if path == HERE / "PROOF_VALIDATION_LEDGER.md":
        preamble = "\n".join(prose_lines[:45])
        ledger_scoped_history = re.search(
            r"(?:§{1,2}\s*6|`?\s*6\s*`?(?:절|장)?)\s*(?:이후|부터)"
            r"[\s\S]{0,240}?(?:역사|historical)"
            r"[\s\S]{0,240}?(?:현행|current)"
            r"[\s\S]{0,120}?(?:아니|not)",
            preamble,
            re.IGNORECASE,
        ) is not None
    current_section: int | None = None
    for number, line in enumerate(prose_lines, 1):
        heading = re.match(r"^##\s+(\d+)\.", line)
        if heading:
            current_section = int(heading.group(1))
        if not line.strip():
            continue
        lower = max(0, number - 6)
        upper = min(total, number + 5)
        window = "\n".join(prose_lines[lower:upper])
        in_scoped_history = (
            ledger_scoped_history
            and current_section is not None
            and current_section >= 6
        )
        if QUARANTINE_MARKERS.search(window) or in_scoped_history:
            continue
        for rule in FORBIDDEN_RULES:
            for match in rule.pattern.finditer(line):
                if rule.context is not None and not rule.context.search(window):
                    continue
                add_failure(
                    failures,
                    path,
                    number,
                    rule.name,
                    f"{rule.detail}: {match.group(0)!r}",
                )


def require_contains(
    failures: list[Failure], path: Path, text: str | None, tokens: Iterable[str], name: str
) -> None:
    if text is None:
        return
    for token in tokens:
        if token not in text:
            add_failure(
                failures,
                path,
                0,
                "CANONICAL_CONTRACT",
                f"{name} is missing exact token {token!r}",
            )


def require_regex(
    failures: list[Failure], path: Path, text: str | None, pattern: str, detail: str
) -> None:
    if text is not None and re.search(pattern, text, re.MULTILINE | re.DOTALL) is None:
        add_failure(failures, path, 0, "CANONICAL_CONTRACT", detail)


def forbid_unquarantined_regex(
    failures: list[Failure],
    path: Path,
    text: str | None,
    pattern: str,
    detail: str,
) -> None:
    if text is None:
        return
    lines = mask_fenced_code(path, text, failures=[])
    total = len(lines)
    compiled = re.compile(pattern, re.IGNORECASE)
    for number, line in enumerate(lines, 1):
        for match in compiled.finditer(line):
            lower = max(0, number - 6)
            upper = min(total, number + 5)
            window = "\n".join(lines[lower:upper])
            if QUARANTINE_MARKERS.search(window):
                continue
            add_failure(
                failures,
                path,
                number,
                "ROLE_REGRESSION",
                f"{detail}: {match.group(0)!r}",
            )


def verify_manifest(failures: list[Failure]) -> None:
    raw = read_utf8(MANIFEST_PATH, failures)
    if raw is None:
        return
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        add_failure(failures, MANIFEST_PATH, exc.lineno, "MANIFEST", str(exc))
        return

    expected_values: dict[tuple[str, str], object] = {
        ("derived", "s_A2"): 0.2315097758079336,
        ("derived", "delta_n"): 0.17791299951329392,
        ("derived", "D_n"): 3.177912999513294,
        ("derived", "bootstrap_x"): 0.04863825851598632,
        ("derived", "dark_ratio_R"): 0.37823869664388306,
        ("derived", "omega_b"): 0.04863825851598632,
        ("derived", "omega_dm"): 0.2610881743576135,
        ("derived", "omega_de"): 0.6902735671264002,
        ("derived", "h0_readout_low_km_s_mpc"): 66.80274566609651,
        ("derived", "h0_readout_high_km_s_mpc"): 72.70237067435626,
        ("derived", "light_bridge_mass_mev"): 29.69915961743591,
        ("derived", "portal_bridge_mass_gev"): 43.8056764706134,
        ("derived", "portal_invisible_branching_ratio"): 0.7708222228518505,
        ("inputs", "higgs_invisible_br_limit_95cl"): 0.107,
        ("derived", "finite_scalar_same_coupling_mu"): 1.6255197624051525e-09,
        ("derived", "inflation_n_s"): 0.9661711384797066,
        ("derived", "inflation_r"): 0.004345610328536992,
        ("diagnostics", "fixed_background_external_rd_chi2"): 40.20145086,
        ("diagnostics", "fixed_background_external_rd_p"): 0.000128283168,
        ("diagnostics", "fixed_background_eh_rd_chi2"): 41.19455358,
        ("diagnostics", "fixed_background_eh_rd_p"): 0.0000886018138,
        ("diagnostics", "fixed_background_verdict"): "REJECT",
    }
    for (section, key), expected in expected_values.items():
        actual = manifest.get(section, {}).get(key)
        if actual != expected:
            add_failure(
                failures,
                MANIFEST_PATH,
                0,
                "MANIFEST",
                f"{section}.{key}={actual!r}; expected {expected!r}",
            )

    definition = manifest.get("definitions", {}).get("ce_neutral_mixing_output", "")
    if "s_A2" not in definition or "physical weak-angle scheme is an open bridge" not in definition:
        add_failure(
            failures,
            MANIFEST_PATH,
            0,
            "MANIFEST",
            "s_A2 must remain a registered output, not a physical weak-angle alias",
        )
    scalar_definition = manifest.get("definitions", {}).get("scalar_g_minus_2", "")
    if "(1-z)^2(1+z)" not in scalar_definition:
        add_failure(
            failures,
            MANIFEST_PATH,
            0,
            "MANIFEST",
            "correct CP-even scalar kernel numerator is missing",
        )


def verify_canonical_documents(failures: list[Failure]) -> None:
    required_paths = (
        COMPLETION_PATH,
        BASELINE_PATH,
        COSMOLOGY_AUDIT_PATH,
        ISSUES_PATH,
        README_PATH,
        MAIN_PATH,
        PROOF_STATUS_PATH,
        BOOTSTRAP_PATH,
        H0_ROLE_PATH,
        AGI_EQUATION_PATH,
        AGI_VERIFICATION_PATH,
        AGI_AGENT_LOOP_PATH,
        AGI_CODE_MAP_PATH,
        AGI_ARCHITECTURE_PATH,
        AGI_CONSCIOUSNESS_PATH,
        AGI_SLEEP_PATH,
        AGI_ROADMAP_PATH,
        HERE / "verify_numeric_consistency.py",
    )
    texts = {path: read_utf8(path, failures) for path in required_paths}

    require_contains(
        failures,
        ISSUES_PATH,
        texts[ISSUES_PATH],
        (
            "0.2315097758079336",
            "0.1779129995132939",
            "3.177912999513294",
            "0.04863825851598632",
            "0.3782386966438831",
        ),
        "Track-A full-precision chain",
    )
    require_regex(
        failures,
        ISSUES_PATH,
        texts[ISSUES_PATH],
        r"s_A\^2\s*&?=0\.2315097758079336",
        "Track-A mixing number must be displayed as s_A^2",
    )

    require_contains(
        failures,
        README_PATH,
        texts[README_PATH],
        (
            "0.0486382585,0.2610881744,0.6902735671",
            "66.802746",
            "72.702371",
            "0.77082222",
            "0.107",
            "162.55198",
            "REJECT",
        ),
        "README canonical presentation",
    )
    require_regex(
        failures,
        README_PATH,
        texts[README_PATH],
        r"0\.77082222[\s\S]{0,220}?0\.107[\s\S]{0,220}?"
        r"(?:통과하지\s*못|`REJECT`)",
        "README portal BR and direct limit must retain the rejection meaning",
    )

    require_contains(
        failures,
        BASELINE_PATH,
        texts[BASELINE_PATH],
        ("0.1180", "125.11", "4.10", "0.107", "PDG 2026"),
        "observational provenance baseline",
    )

    require_contains(
        failures,
        COSMOLOGY_AUDIT_PATH,
        texts[COSMOLOGY_AUDIT_PATH],
        (
            "40.20145086",
            "1.28283168\\times10^{-4}",
            "41.19455358",
            "8.86018138\\times10^{-5}",
            "29.69915961743591",
        ),
        "fixed-background covariance and light-mass presentation",
    )
    require_regex(
        failures,
        COSMOLOGY_AUDIT_PATH,
        texts[COSMOLOGY_AUDIT_PATH],
        r"40\.20145086[\s\S]{0,900}?`REJECT`",
        "external-r_d covariance result must carry a nearby REJECT verdict",
    )
    require_regex(
        failures,
        COSMOLOGY_AUDIT_PATH,
        texts[COSMOLOGY_AUDIT_PATH],
        r"41\.19455358[\s\S]{0,500}?`REJECT`",
        "Eisenstein--Hu covariance result must carry a nearby REJECT verdict",
    )

    require_contains(
        failures,
        MAIN_PATH,
        texts[MAIN_PATH],
        (
            "\\frac{(1-z)^2(1+z)}{(1-z)^2+zr^2}",
            "162.55198\\times10^{-11}",
            "43.805676",
            "0.77082",
            "0.107",
            "0.96617114",
            "0.00434561",
        ),
        "main calculation-chain presentation",
    )
    require_regex(
        failures,
        MAIN_PATH,
        texts[MAIN_PATH],
        r"I_s\(0\)=\frac\{3\}\{2\}|I_s\(0\)=3/2|I_s\(0\).*?3/2",
        "correct scalar kernel must retain the 3/2 light limit",
    )

    require_contains(
        failures,
        PROOF_STATUS_PATH,
        texts[PROOF_STATUS_PATH],
        (
            "Unitary texture benchmark/Open joint likelihood",
            "External normalization input",
        ),
        "proof-status role labels",
    )
    bootstrap = texts[BOOTSTRAP_PATH]
    if bootstrap is not None:
        forbidden_phrase = "A_s 유도의 입력"
        offset = bootstrap.find(forbidden_phrase)
        if offset >= 0:
            add_failure(
                failures,
                BOOTSTRAP_PATH,
                line_number(bootstrap, offset),
                "ROLE_REGRESSION",
                "A_s may normalize lambda_4 but may not be relabeled as a derived input",
            )
    require_contains(
        failures,
        H0_ROLE_PATH,
        texts[H0_ROLE_PATH],
        ("calibration replay", "prospective/holdout test가 아니다"),
        "H0 channel-role quarantine",
    )

    agi_required_tokens = {
        AGI_EQUATION_PATH: (
            "0.1180",
            "0.1779129995",
            "3.1779129995",
            "0.0316530354",
        ),
        AGI_VERIFICATION_PATH: (
            "0.0316530354",
            "0.4904868132",
            "0.3146719247",
            "CANONICAL_DRIFT",
        ),
        AGI_AGENT_LOOP_PATH: (
            "3.177912999513294",
            "0.0316530354",
            "0.4904868132",
            "\\lambda_{\\rm portal}:=\\delta_N^2=0.0316530354",
            "\\xi_{\\rm design}:=\\alpha_s^{1/3}=0.4904868132",
        ),
        AGI_CODE_MAP_PATH: (
            "0.0316530354",
            "0.4904868132",
            "0.3146719247",
            "CANONICAL_DRIFT",
        ),
    }
    for path, tokens in agi_required_tokens.items():
        require_contains(
            failures,
            path,
            texts[path],
            tokens,
            "AGI canonical-constant role contract",
        )
        forbid_unquarantined_regex(
            failures,
            path,
            texts[path],
            r"(?<![\d.])(?:0\.03120|0\.3148|0\.4892)(?!\d)",
            "superseded AGI constant is presented outside a legacy/drift block",
        )
        forbid_unquarantined_regex(
            failures,
            path,
            texts[path],
            r"(?:1\s*/\s*\(?\s*e(?:\^\{?1/3\}?)?\s*\\pi|"
            r"4\s*/\s*\(?\s*e\^\{?4/3\}?\s*\\pi\^\{?4/3\}?)",
            "1/(e*pi)-family implementation constant is outside a legacy/drift block",
        )
        if path in (AGI_VERIFICATION_PATH, AGI_CODE_MAP_PATH):
            agi_lines = mask_fenced_code(path, texts[path] or "", failures=[])
            old_value_pattern = re.compile(
                r"(?<![\d.])(?:0\.03120|0\.3148|0\.4892)(?!\d)"
            )
            for number, line in enumerate(agi_lines, 1):
                if old_value_pattern.search(line) and "CANONICAL_DRIFT" not in line:
                    add_failure(
                        failures,
                        path,
                        number,
                        "ROLE_REGRESSION",
                        "13/18 legacy constants must carry CANONICAL_DRIFT on the same row",
                    )
    require_regex(
        failures,
        AGI_EQUATION_PATH,
        texts[AGI_EQUATION_PATH],
        r"0\.490486813(?:152\d*|2)",
        "AGI equation must contain the canonical bypass coefficient",
    )
    require_regex(
        failures,
        AGI_EQUATION_PATH,
        texts[AGI_EQUATION_PATH],
        r"0\.314671924(?:672\d*|7)",
        "AGI equation must contain the canonical wake coefficient",
    )

    require_contains(
        failures,
        AGI_ARCHITECTURE_PATH,
        texts[AGI_ARCHITECTURE_PATH],
        (
            "L_g:=I-V^\\top V",
            "\\max\\!\\left(1,\\sigma_1",
            "\\exp(-\\xi_{\\text{design}}\\widetilde E",
        ),
        "AGI nonexpansive architecture contract",
    )
    architecture = texts[AGI_ARCHITECTURE_PATH]
    if architecture is not None:
        offset = architecture.find("\\Longleftrightarrow")
        if offset >= 0:
            add_failure(
                failures,
                AGI_ARCHITECTURE_PATH,
                line_number(architecture, offset),
                "ROLE_REGRESSION",
                "determinant and spectral-norm conditions may not be stated as equivalent",
            )

    require_contains(
        failures,
        AGI_CONSCIOUSNESS_PATH,
        texts[AGI_CONSCIOUSNESS_PATH],
        ("M_\\tau:=\\exp", "모니터링 안정도"),
        "metacognition-proxy contract",
    )
    require_contains(
        failures,
        AGI_SLEEP_PATH,
        texts[AGI_SLEEP_PATH],
        ("동일 self-map 수렴 | 미검증",),
        "sleep-loop conditional-convergence contract",
    )
    require_contains(
        failures,
        AGI_ROADMAP_PATH,
        texts[AGI_ROADMAP_PATH],
        ("transformer에서 실패, SNN 미검증",),
        "AGI roadmap non-success contract",
    )
    consciousness_forbidden = (
        "Softmax가 확률 보존 | 충족",
        "자기일관적 정보 처리의 필연적 구조",
        "최소 연산 비용으로 최대 성능",
    )
    consciousness = texts[AGI_CONSCIOUSNESS_PATH]
    if consciousness is not None:
        for phrase in consciousness_forbidden:
            offset = consciousness.find(phrase)
            if offset >= 0:
                add_failure(
                    failures,
                    AGI_CONSCIOUSNESS_PATH,
                    line_number(consciousness, offset),
                    "ROLE_REGRESSION",
                    f"unsupported consciousness/optimality claim remains: {phrase!r}",
                )

    application_forbidden = {
        AGI_SLEEP_PATH: (
            "우주는 $T=0$에서 한 번의 양자 간섭으로 고정점에 도달했다",
            "파괴적 망각(catastrophic forgetting)이 수면 순환으로 자연스럽게 해결된다",
        ),
        AGI_ROADMAP_PATH: ("| 고정점 도달 | 정확 ($T=0$)",),
    }
    for path, phrases in application_forbidden.items():
        text = texts[path]
        if text is None:
            continue
        for phrase in phrases:
            offset = text.find(phrase)
            if offset >= 0:
                add_failure(
                    failures,
                    path,
                    line_number(text, offset),
                    "ROLE_REGRESSION",
                    f"unsupported exact-convergence claim remains: {phrase!r}",
                )

    completion = texts[COMPLETION_PATH]
    if completion is not None:
        for loop in range(6):
            pattern = rf"(?m)^\|\s*L{loop}\s*\|[^\n]*\|\s*`IMPLEMENTED`\s*\|\s*$"
            if re.search(pattern, completion) is None:
                add_failure(
                    failures,
                    COMPLETION_PATH,
                    0,
                    "COMPLETION_STATUS",
                    f"L{loop} is not closed as IMPLEMENTED",
                )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="check Markdown math syntax/raw TeX only; skip bounded semantic contract",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    failures: list[Failure] = []
    markdown_files = sorted(DOCS_ROOT.rglob("*.md"), key=lambda item: item.as_posix())
    if len(markdown_files) != EXPECTED_MARKDOWN_COUNT:
        add_failure(
            failures,
            DOCS_ROOT,
            0,
            "INVENTORY",
            f"found {len(markdown_files)} Markdown files; expected {EXPECTED_MARKDOWN_COUNT}",
        )

    for path in markdown_files:
        text = read_utf8(path, failures)
        if text is None:
            continue
        scan_math_and_raw_tex(path, text, failures)
        if not arguments.syntax_only:
            scan_forbidden_values(path, text, failures)

    if not arguments.syntax_only:
        verify_manifest(failures)
        verify_canonical_documents(failures)

    if failures:
        print(f"DOCUMENT CONTRACT: FAIL ({len(failures)})")
        for failure in sorted(set(failures)):
            print(failure.render())
        return 1

    scope = "syntax" if arguments.syntax_only else "syntax + bounded semantic contract"
    print("DOCUMENT CONTRACT: PASS")
    print(f"Markdown files checked: {len(markdown_files)} ({scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
