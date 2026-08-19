from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import re
from urllib.parse import unquote


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DRAFT = RUN_DIR / "60-paper-draft.md"
NORMALIZER = ROOT / "docs" / "2_경로적분과_응용" / "normalize_markdown_math.py"


def check(condition: bool, label: str, detail: str = "") -> None:
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{label}{suffix}")
    print(f"[OK] {label}{': ' + detail if detail else ''}")


def load_normalizer():
    spec = importlib.util.spec_from_file_location("ce_paper_math_normalizer", NORMALIZER)
    check(spec is not None and spec.loader is not None, "normalizer import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_link_violations(text: str) -> list[str]:
    pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    violations: list[str] = []
    for match in pattern.finditer(text):
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        if not raw or raw.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        target = unquote(raw.split("#", 1)[0].split("?", 1)[0])
        if target and not (DRAFT.parent / target).resolve().exists():
            line = text.count("\n", 0, match.start()) + 1
            violations.append(f"line {line}: {raw}")
    return violations


def non_prose_section_starts(lines: list[str]) -> list[str]:
    exempt = {"## 참고문헌"}
    violations: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith(("## ", "### ")) or line in exempt:
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        first = lines[cursor].strip() if cursor < len(lines) else ""
        if not first or first.startswith(("#", "**[", "**가정", "**정의", "**보조", "**산출", "**핵심", "**따름", "|", "- ", "1. ", "$$", "```")):
            violations.append(f"line {index + 1}: {line} -> {first}")
    return violations


def solve_low_root(d_eff: float) -> float:
    q = math.exp(-d_eff)
    for _ in range(100):
        residual = q - math.exp(-d_eff * (1.0 - q))
        derivative = 1.0 - d_eff * math.exp(-d_eff * (1.0 - q))
        updated = q - residual / derivative
        if updated == q:
            return q
        q = updated
    raise RuntimeError("paper audit root solver did not converge")


def main() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    lines = text.splitlines()

    expected_sections = [
        "## 초록",
        "## 1. 서론",
        "## 2. 물리적 동기와 형식 경계",
        "## 3. 가정과 정의",
        "## 4. 조건부 수학 코어",
        "## 5. 핵심 정리와 수치 산출",
        "## 6. 암흑 표현의 조건부 경계모형",
        "## 7. 남은 문제",
        "## 참고문헌",
        "## 내부 재현 자료",
    ]
    positions = [text.index(section) for section in expected_sections]
    check(positions == sorted(positions), "section order", f"{len(expected_sections)} sections")
    check(not re.search(r"(?m)^#{1,3}\s+결론(?:\s|$)", text), "no standalone conclusion")
    check(not non_prose_section_starts(lines), "reader-oriented section starts")

    normalizer = load_normalizer()
    normalized, block_count, inline_count = normalizer.normalize_text(text)
    check(normalized == text, "renderable math delimiters", f"pending={block_count + inline_count}")
    check(not local_link_violations(text), "relative links resolve")

    equation_numbers = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", text)]
    check(equation_numbers == list(range(1, 22)), "equation numbering", str(equation_numbers))
    proof_openings = len(re.findall(r"(?m)^증명\.", text))
    check(proof_openings == text.count("□") == 7, "proof opening/closing pairs", "7")
    check(len(re.findall(r"(?m)^유도\.", text)) == 1, "derivation marker", "1")

    forbidden = {
        "double passive": r"되어진|보여진",
        "translationese location": r"에 있어서",
        "prose logic symbols": r"[∀∃⇒]",
        "machine verdict badge": r"\[(?:PASS|FAIL|WARN|CAUTION)\]",
    }
    violations = [label for label, pattern in forbidden.items() if re.search(pattern, text)]
    check(not violations, "style exclusions", ", ".join(violations))

    required_boundaries = (
        "양자 진폭에서 비음수 자손 법칙으로 가는 과정",
        "확률-to-density 다리",
        "오늘의 우주론 밀도 예측으로 제시하지 않는다",
        "이 검사는 식의 차원 정합성을 보일 뿐",
        "경험식 E1",
        "과거 우주론 경계모형",
        "$q=1$이라는 다른 고정점",
    )
    missing = [fragment for fragment in required_boundaries if fragment not in text]
    check(not missing, "formal-boundary disclosures", ", ".join(missing))

    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = 3.0 + delta
    q_ext = solve_low_root(d_eff)
    survival = 1.0 - q_ext
    contraction = d_eff * q_ext
    ratio = alpha_s * d_eff
    omega_lambda = survival / (1.0 + ratio)
    omega_dm = survival * ratio / (1.0 + ratio)
    residual = q_ext - math.exp(-d_eff * (1.0 - q_ext))

    check(math.isclose(sin2_theta_w, 0.231222068260755, abs_tol=5e-16), "sin2 chain")
    check(math.isclose(delta, 0.177758423409974, abs_tol=5e-16), "delta chain")
    check(math.isclose(d_eff, 3.177758423409974, abs_tol=5e-16), "effective depth")
    check(math.isclose(q_ext, 0.0486467196440282, abs_tol=5e-17), "low fixed point")
    check(abs(residual) <= 8.0 * math.ulp(q_ext), "fixed-point residual", repr(residual))
    check(contraction < 1.0, "contraction bound", f"Dq={contraction:.16g}")
    check(math.isclose(omega_dm, 0.2592717094, abs_tol=5e-11), "conditional dark matter")
    check(math.isclose(omega_lambda, 0.6920815709, abs_tol=5e-11), "conditional dark energy")
    check(math.isclose(q_ext + omega_dm + omega_lambda, 1.0, abs_tol=1e-15), "conditional closure")

    print("PAPER_DRAFT_AUDIT: PASS")


if __name__ == "__main__":
    main()
