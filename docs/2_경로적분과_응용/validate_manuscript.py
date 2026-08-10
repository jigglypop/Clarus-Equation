"""Deterministic validation gate for the 01–14 manuscript.

This gate checks reproducibility and claim hygiene.  It deliberately does not
declare a physical hypothesis true merely because a numerical identity passes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rejection_loop_engineering import (
    CANONICAL_OBS_AS_1E9,
    CANONICAL_OBS_AS_SIGMA_1E9,
    build_report as build_rejection_loop_report,
    validate_report as validate_rejection_loop_report,
)
from improvement_loop_engineering import (
    build_report as build_improvement_loop_report,
    validate_report as validate_improvement_loop_report,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STATUS_TOKENS = (
    "[정의]",
    "[정리]",
    "[공리]",
    "[산출]",
    "[경험식]",
    "[미완성]",
    "[예측]",
)
LEGACY_STATUS_TOKENS = (
    "[Exact]",
    "[Selection]",
    "[Bridge]",
    "[Phenomenology]",
    "[Open]",
    "[Open test]",
    "[Rejected]",
    "[Conditional]",
)
CHAPTERS = tuple(
    sorted(path for path in HERE.glob("[0-9][0-9]_*.md") if path.name != "00_검증_규약.md")
)
CANONICAL_PROSE = (
    HERE / "00_검증_규약.md",
    *CHAPTERS,
    HERE / "전체_진리값_감사.md",
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ClaimStatusOccurrence:
    filename: str
    line: int
    status: str
    text: str
    taxonomy_definition: bool


def claim_status_inventory() -> tuple[ClaimStatusOccurrence, ...]:
    """Inventory every literal status occurrence without pretending it is proof."""

    records: list[ClaimStatusOccurrence] = []
    for path in CHAPTERS:
        first_occurrence_seen = {token: False for token in STATUS_TOKENS}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            taxonomy_context = line_number <= 15 and (
                "상태 표기" in line
                or stripped.startswith("|")
                or stripped.startswith("- **")
            )
            for token in STATUS_TOKENS:
                for occurrence_index in range(line.count(token)):
                    taxonomy_definition = (
                        not first_occurrence_seen[token] and taxonomy_context
                    )
                    records.append(
                        ClaimStatusOccurrence(
                            filename=path.name,
                            line=line_number,
                            status=token,
                            text=stripped,
                            taxonomy_definition=taxonomy_definition,
                        )
                    )
                    first_occurrence_seen[token] = True
    return tuple(records)


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def fixed_point(D: float) -> float:
    q = math.exp(-D)
    for _ in range(200):
        next_q = math.exp(-D * (1.0 - q))
        if abs(next_q - q) < 1e-15:
            return next_q
        q = next_q
    raise RuntimeError("fixed-point iteration did not converge")


def extinction_probability(D: float) -> float:
    """Return the minimal [0,1] fixed point of the Poisson generating function."""

    if D < 0.0:
        raise ValueError("D must be nonnegative")
    if D <= 1.0:
        return 1.0
    return bisect(lambda x: math.exp(-D * (1.0 - x)) - x, 0.0, 1.0 / D)


def bisect(function, lower: float, upper: float, iterations: int = 200) -> float:
    f_lower = function(lower)
    f_upper = function(upper)
    if f_lower * f_upper > 0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = function(midpoint)
        if f_lower * f_midpoint <= 0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint
    return 0.5 * (lower + upper)


def nonminimal_quartic_observables(xi: float, e_folds: float) -> tuple[float, ...]:
    """Tree-level metric result in reduced-Planck units for V=lambda*phi^4/4."""

    def field_metric(y: float) -> float:
        return (1.0 + xi * (1.0 + 6.0 * xi) * y**2) / (1.0 + xi * y**2) ** 2

    def log_potential_prime(y: float) -> float:
        return 4.0 / (y * (1.0 + xi * y**2))

    def epsilon(y: float) -> float:
        return 0.5 * log_potential_prime(y) ** 2 / field_metric(y)

    y_end = bisect(lambda y: epsilon(y) - 1.0, 0.01, 10.0)

    def e_folds_at(y: float) -> float:
        return (
            (1.0 + 6.0 * xi) * (y**2 - y_end**2) / 8.0
            - 0.75 * math.log((1.0 + xi * y**2) / (1.0 + xi * y_end**2))
        )

    y_star = bisect(lambda y: e_folds_at(y) - e_folds, y_end, 50.0)
    metric = field_metric(y_star)
    u_prime_over_u = log_potential_prime(y_star)
    derivative_u_prime_over_u = -4.0 * (1.0 + 3.0 * xi * y_star**2) / (
        y_star**2 * (1.0 + xi * y_star**2) ** 2
    )
    a = xi * (1.0 + 6.0 * xi)
    derivative_log_metric = (
        2.0 * a * y_star / (1.0 + a * y_star**2)
        - 4.0 * xi * y_star / (1.0 + xi * y_star**2)
    )
    epsilon_star = epsilon(y_star)
    eta_star = (
        (derivative_u_prime_over_u + u_prime_over_u**2) / metric
        - u_prime_over_u * derivative_log_metric / (2.0 * metric)
    )
    n_s = 1.0 - 6.0 * epsilon_star + 2.0 * eta_star
    tensor_ratio = 16.0 * epsilon_star
    return y_end, y_star, n_s, tensor_ratio


def line_allows_missing_reference(line: str) -> bool:
    qualifiers = ("[미완성]", "미존재", "누락", "재현 불가", "stale")
    return any(token in line for token in qualifiers)


def validate() -> list[Check]:
    checks: list[Check] = []

    checks.append(
        Check(
            "chapter-count",
            len(CHAPTERS) == 14,
            f"found {len(CHAPTERS)} chapter files: {[path.name for path in CHAPTERS]}",
        )
    )
    checks.append(
        Check(
            "primordial-observational-snapshot",
            close(CANONICAL_OBS_AS_1E9, 2.099, 5e-15)
            and close(CANONICAL_OBS_AS_SIGMA_1E9, 0.029, 5e-15),
            f"Planck 2018 A_s x 1e9={CANONICAL_OBS_AS_1E9:.3f}"
            f"+/-{CANONICAL_OBS_AS_SIGMA_1E9:.3f}; imported from the canonical "
            "primordial-spectrum gate and used by the inflation normalization",
        )
    )

    missing_status = []
    texts: dict[str, str] = {}
    for path in CHAPTERS:
        text = path.read_text(encoding="utf-8")
        texts[path.name] = text
        if not any(token in text for token in STATUS_TOKENS):
            missing_status.append(path.name)
    checks.append(
        Check(
            "claim-status-taxonomy",
            not missing_status,
            "all chapters use the shared taxonomy"
            if not missing_status
            else f"missing a status token: {missing_status}",
        )
    )
    status_counts = {
        token: sum(text.count(token) for text in texts.values())
        for token in STATUS_TOKENS
    }
    protocol_text = (HERE / "00_검증_규약.md").read_text(encoding="utf-8")
    ledger_text = (HERE / "전체_진리값_감사.md").read_text(encoding="utf-8")
    canonical_prose_texts = {
        path.name: path.read_text(encoding="utf-8") for path in CANONICAL_PROSE
    }
    legacy_status_occurrences = {
        token: sum(text.count(token) for text in canonical_prose_texts.values())
        for token in LEGACY_STATUS_TOKENS
    }
    verdict_word = re.compile(r"\b(?:PASS|FAIL|Rejected)\b", re.IGNORECASE)
    forbidden_verdict_occurrences = [
        f"{path.name}:{line_number}"
        for path in CANONICAL_PROSE
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if verdict_word.search(line)
    ]
    checks.append(
        Check(
            "canonical-status-vocabulary",
            not any(legacy_status_occurrences.values())
            and all(token in protocol_text for token in STATUS_TOKENS),
            "definition plus six formal provenance classes are defined; legacy tags are absent"
            if not any(legacy_status_occurrences.values())
            else f"legacy occurrences={legacy_status_occurrences}",
        )
    )
    checks.append(
        Check(
            "canonical-prose-verdict-words-absent",
            not forbidden_verdict_occurrences,
            "machine verdict words are absent from canonical prose"
            if not forbidden_verdict_occurrences
            else f"forbidden verdict words at {forbidden_verdict_occurrences}",
        )
    )
    checks.append(
        Check(
            "literal-status-inventory-reported",
            all(
                status_counts[token] > 0
                for token in STATUS_TOKENS
                if token != "[예측]"
            )
            and all(
                f"| `{token}` | {status_counts[token]} |" in ledger_text
                for token in STATUS_TOKENS
            ),
            ", ".join(
                f"{token}={status_counts[token]}" for token in STATUS_TOKENS
            )
            + "; zero active [예측] claims is allowed; counts are provenance inventory, not proof",
        )
    )
    status_inventory = claim_status_inventory()
    taxonomy_count = sum(item.taxonomy_definition for item in status_inventory)
    taxonomy_keys = {
        (item.filename, item.status)
        for item in status_inventory
        if item.taxonomy_definition
    }
    substantive_by_status = {
        token: sum(
            item.status == token and not item.taxonomy_definition
            for item in status_inventory
        )
        for token in STATUS_TOKENS
    }
    checks.append(
        Check(
            "claim-status-occurrence-inventory",
            len(status_inventory) == sum(status_counts.values())
            and taxonomy_count == len(taxonomy_keys)
            and taxonomy_count <= len(CHAPTERS) * len(STATUS_TOKENS),
            f"{len(status_inventory)} literal occurrences inventoried; "
            f"{taxonomy_count} taxonomy definitions excluded; substantive="
            f"{substantive_by_status}. This is classification coverage, not proof coverage.",
        )
    )

    all_markdown = tuple(sorted(HERE.glob("*.md")))
    malformed_markdown: list[str] = []
    replacement_characters: list[str] = []
    broken_local_links: list[str] = []
    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in all_markdown:
        text = path.read_text(encoding="utf-8")
        if "\ufffd" in text:
            replacement_characters.append(path.name)
        delimiter_counts = {
            "fence": text.count("```"),
            "display-dollar": text.count("$$"),
            # Ignore LaTeX line breaks such as ``\\[4pt]`` inside arrays.
            "display-open": len(re.findall(r"(?<!\\)\\\[", text)),
            "display-close": len(re.findall(r"(?<!\\)\\\]", text)),
            "inline-open": text.count(r"\("),
            "inline-close": text.count(r"\)"),
        }
        if delimiter_counts["fence"] % 2:
            malformed_markdown.append(f"{path.name}: unclosed code fence")
        if delimiter_counts["display-dollar"] % 2:
            malformed_markdown.append(f"{path.name}: unpaired $$")
        if delimiter_counts["display-open"] != delimiter_counts["display-close"]:
            malformed_markdown.append(f"{path.name}: unpaired \\[...\\]")
        if delimiter_counts["inline-open"] != delimiter_counts["inline-close"]:
            malformed_markdown.append(f"{path.name}: unpaired \\(...\\)")
        for match in markdown_link.finditer(text):
            target_text = match.group(1).strip().strip("<>")
            if (
                not target_text
                or "://" in target_text
                or target_text.startswith(("#", "mailto:"))
            ):
                continue
            local_part = target_text.split("#", 1)[0]
            target = (path.parent / local_part).resolve()
            if not target.exists():
                line_number = text[: match.start()].count("\n") + 1
                broken_local_links.append(f"{path.name}:{line_number} -> {target_text}")
    checks.append(
        Check(
            "markdown-delimiters",
            not malformed_markdown,
            "all Markdown/code/math delimiters are balanced"
            if not malformed_markdown
            else "; ".join(malformed_markdown),
        )
    )
    checks.append(
        Check(
            "utf8-integrity",
            not replacement_characters,
            "all Markdown files decode as UTF-8 without replacement characters"
            if not replacement_characters
            else f"replacement characters in {replacement_characters}",
        )
    )
    checks.append(
        Check(
            "local-markdown-links",
            not broken_local_links,
            "all relative Markdown links resolve"
            if not broken_local_links
            else "; ".join(broken_local_links),
        )
    )

    legacy_patterns = {
        "wrong-dark-energy-unit": re.compile(r"3\.9[^\n]*10\^\{-47\}"),
        "wrong-transition-value": re.compile(r"0\.683"),
        "excluded-invisible-branching": re.compile(r"BR[^\n]*0\.005", re.IGNORECASE),
        "no-extra-particles": re.compile(r"추가 입자 불필요"),
        "zero-free-parameters": re.compile(r"자유.?매개변수 0개"),
        "time-reversal-no-go": re.compile(r"시간 역행 금지"),
        "hodge-d3-overclaim": re.compile(r"Hodge 자기쌍대성의 유일해"),
        "one-dimensional-scattering-no-go": re.compile(r"1차원 산란[^\n]*불가"),
    }
    remaining_legacy_claims: list[str] = []
    for path in CHAPTERS:
        lines = texts[path.name].splitlines()
        for index, line in enumerate(lines):
            for label, pattern in legacy_patterns.items():
                if pattern.search(line):
                    remaining_legacy_claims.append(
                        f"{path.name}:{index + 1} ({label})"
                    )
    checks.append(
        Check(
            "legacy-invalid-claims-removed-from-body",
            not remaining_legacy_claims,
            "known counterexampled legacy claims are absent from 01--14"
            if not remaining_legacy_claims
            else "; ".join(remaining_legacy_claims),
        )
    )

    missing_refs: list[str] = []
    reference_pattern = re.compile(r"examples/physics/[A-Za-z0-9_./-]+\.py")
    for path in CHAPTERS:
        for line_number, line in enumerate(texts[path.name].splitlines(), start=1):
            for match in reference_pattern.finditer(line):
                target = REPO_ROOT / Path(match.group(0))
                if not target.exists() and not line_allows_missing_reference(line):
                    missing_refs.append(f"{path.name}:{line_number} -> {match.group(0)}")
    checks.append(
        Check(
            "referenced-scripts-exist-or-are-disclosed",
            not missing_refs,
            "all runnable references exist; unavailable artifacts are explicitly marked [미완성]"
            if not missing_refs
            else "; ".join(missing_refs),
        )
    )

    linked_artifact_specs = (
        (
            "alpha-closure",
            REPO_ROOT / "examples" / "physics" / "alpha_s_closure_gate.py",
            ("zero free dimensionless parameters",),
        ),
        (
            "holographic-scale",
            REPO_ROOT
            / "examples"
            / "physics"
            / "cosmological_constant_holographic_gate.py",
            ("zero free parameters", "reproduced to <0.2%"),
        ),
        (
            "particle-search-template",
            REPO_ROOT / "examples" / "physics" / "clarus_boson_search_gate.py",
            ("print(f\"  pass:",),
        ),
    )
    artifact_isolation_failures: list[str] = []
    artifact_isolation_details: list[str] = []
    canonical_prose_joined = "\n".join(canonical_prose_texts.values())
    for label, artifact_path, stale_fragments in linked_artifact_specs:
        if not artifact_path.is_file():
            artifact_isolation_failures.append(f"{label}: missing artifact")
            continue
        artifact_text = artifact_path.read_text(encoding="utf-8")
        stale_in_artifact = any(fragment in artifact_text for fragment in stale_fragments)
        stale_in_prose = any(
            fragment in canonical_prose_joined for fragment in stale_fragments
        )
        if stale_in_prose:
            artifact_isolation_failures.append(
                f"{label}: an internal legacy output leaked into canonical prose"
            )
        artifact_isolation_details.append(
            f"{label}={'CODE_ONLY' if stale_in_artifact else 'CLEAN'}"
        )
    checks.append(
        Check(
            "linked-artifact-prose-isolation",
            not artifact_isolation_failures,
            "; ".join(artifact_isolation_details)
            if not artifact_isolation_failures
            else "; ".join(artifact_isolation_failures),
        )
    )

    # Exact algebraic and numerical anchors used throughout the manuscript.
    roots = tuple(d for d in range(0, 10) if d == d * (d - 1) // 2)
    checks.append(Check("dimension-counting-roots", roots == (0, 3), f"roots={roots}"))

    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    D = 3.0 + delta
    q = fixed_point(D)
    checks.append(
        Check(
            "nontrivial-fixed-point",
            close(q, 0.0486466, 5e-7),
            f"D={D:.9f}, extinction q={q:.10f}, survival={1.0-q:.10f}",
        )
    )
    checks.append(
        Check(
            "fixed-point-local-derivative",
            close(D * q, 0.15459, 5e-5),
            f"F'(q)=Dq={D*q:.8f}; this is local convergence, not time dynamics",
        )
    )
    extinction_grid = (0.0, 0.5, 1.0, 1.01, 2.0, D, 10.0)
    extinction_values = tuple(
        extinction_probability(candidate_D) for candidate_D in extinction_grid
    )
    checks.append(
        Check(
            "poisson-extinction-is-minimal-fixed-point",
            all(
                close(
                    value,
                    math.exp(-candidate_D * (1.0 - value)),
                    2e-13,
                )
                for candidate_D, value in zip(extinction_grid, extinction_values)
            )
            and extinction_values[:3] == (1.0, 1.0, 1.0)
            and all(value < 1.0 for value in extinction_values[3:]),
            "q_ext=1 for D<=1 and is the nontrivial minimal root for D>1; "
            f"grid={tuple(round(value, 10) for value in extinction_values)}",
        )
    )

    principal_low_D = fixed_point(0.5)
    checks.append(
        Check(
            "low-D-principal-branch",
            close(principal_low_D, 1.0, 1e-12),
            f"q(0.5)={principal_low_D:.15f}; the physical [0,1] solution is q=1",
        )
    )

    def iterate_scalar_poisson(initial: float, iterations: int = 600) -> float:
        value = initial
        for _ in range(iterations):
            value = math.exp(-D * (1.0 - value))
        return value

    basin_initials = (0.0, q / 2.0, 0.2, 0.9, 1.0 - 1.0e-10)
    basin_limits = tuple(iterate_scalar_poisson(value) for value in basin_initials)
    checks.append(
        Check(
            "poisson-discrete-global-basin",
            all(close(value, q, 5e-13) for value in basin_limits)
            and iterate_scalar_poisson(1.0) == 1.0,
            "all sampled x0 in [0,1) converge to the minimal root while x0=1 "
            f"is the exceptional fixed point; limits={basin_limits}",
        )
    )

    def iterate_multitype_poisson(
        matrix: tuple[tuple[float, ...], ...],
        iterations: int = 1000,
    ) -> tuple[float, ...]:
        value = tuple(0.0 for _ in matrix)
        for _ in range(iterations):
            value = tuple(
                math.exp(
                    -sum(
                        matrix[row][column] * (1.0 - value[column])
                        for column in range(len(matrix))
                    )
                )
                for row in range(len(matrix))
            )
        return value

    subcritical_matrix = ((0.2, 0.1), (0.1, 0.2))
    supercritical_matrix = ((0.8, 0.4), (0.4, 0.8))
    subcritical_extinction = iterate_multitype_poisson(subcritical_matrix)
    supercritical_extinction = iterate_multitype_poisson(supercritical_matrix)
    supercritical_residual = tuple(
        supercritical_extinction[row]
        - math.exp(
            -sum(
                supercritical_matrix[row][column]
                * (1.0 - supercritical_extinction[column])
                for column in range(2)
            )
        )
        for row in range(2)
    )
    checks.append(
        Check(
            "multitype-poisson-spectral-threshold",
            all(close(value, 1.0, 5e-13) for value in subcritical_extinction)
            and all(0.0 < value < 1.0 for value in supercritical_extinction)
            and close(supercritical_extinction[0], supercritical_extinction[1], 1e-14)
            and all(abs(value) < 5e-14 for value in supercritical_residual),
            "representative irreducible matrices give q=(1,1) at rho=0.3 and "
            f"q={supercritical_extinction} at rho=1.2",
        )
    )

    alpha_inverse_formula = 4.0 * math.pi**3 + math.pi**2 + math.pi
    alpha_inverse_codata = 137.035999177
    alpha_inverse_uncertainty = 0.000000021
    alpha_pull = (alpha_inverse_formula - alpha_inverse_codata) / alpha_inverse_uncertainty
    checks.append(
        Check(
            "alpha-inverse-residual-disclosed",
            alpha_pull > 10_000,
            f"formula={alpha_inverse_formula:.12f}, CODATA={alpha_inverse_codata:.12f}, "
            f"pull={alpha_pull:.1f} sigma; it cannot be labelled an exact prediction",
        )
    )

    proton_electron_formula = 6.0 * math.pi**5
    proton_electron_codata = 1836.152673426
    proton_electron_uncertainty = 0.000000032
    mass_ratio_pull = (proton_electron_formula - proton_electron_codata) / proton_electron_uncertainty
    checks.append(
        Check(
            "proton-electron-residual-disclosed",
            abs(mass_ratio_pull) > 1_000_000,
            f"formula={proton_electron_formula:.12f}, CODATA={proton_electron_codata:.12f}, "
            f"pull={mass_ratio_pull:.0f} sigma; it cannot be labelled an exact prediction",
        )
    )

    dark_energy_mev4 = (2.5e-9) ** 4
    checks.append(
        Check(
            "meV-to-MeV-fourth-power",
            close(dark_energy_mev4, 3.90625e-35, 1e-42),
            f"(2.5 meV)^4={dark_energy_mev4:.8e} MeV^4",
        )
    )

    transition_limit = 4.0 * (1.0 / (2.0 * math.pi)) ** (4.0 / 3.0)
    checks.append(
        Check(
            "transition-limit-arithmetic",
            close(transition_limit, 0.34500085, 1e-8),
            f"4(1/2pi)^(4/3)={transition_limit:.9f}",
        )
    )
    derivative_D = 2.0
    derivative_x = 0.5
    partial_D_residual = (1.0 - derivative_x) * math.exp(
        -derivative_D * (1.0 - derivative_x)
    )
    checks.append(
        Check(
            "transition-source-derivative-domain",
            close(partial_D_residual, 0.5 / math.e, 2e-16)
            and not close(partial_D_residual, derivative_x * (1.0 - derivative_x), 1e-3)
            and close(
                (1.0 - q) * math.exp(-D * (1.0 - q)),
                q * (1.0 - q),
                2e-15,
            ),
            "partial_D r=(1-x)e^{-D(1-x)} generally; x(1-x) holds only on r=0",
        )
    )

    # Benchmark scalar-portal width in the convention L_int=-lambda|H|^2 phi^2.
    portal_lambda = delta**2
    v_ew = 246.22
    higgs_mass = 125.25
    scalar_mass = v_ew * delta
    phase_space = math.sqrt(1.0 - 4.0 * scalar_mass**2 / higgs_mass**2)
    invisible_width = portal_lambda**2 * v_ew**2 * phase_space / (8.0 * math.pi * higgs_mass)
    sm_higgs_width = 0.00407
    invisible_branching = invisible_width / (invisible_width + sm_higgs_width)
    checks.append(
        Check(
            "removed-portal-benchmark-counterexample",
            invisible_branching > 0.7,
            f"m_phi={scalar_mass:.3f} GeV, Gamma_inv={invisible_width*1e3:.3f} MeV, "
            f"BR_inv={invisible_branching:.3f}; the false parent is absent from the body",
        )
    )

    charged_lepton_masses = (0.51099895, 105.6583755, 1776.86)
    koide = sum(charged_lepton_masses) / sum(math.sqrt(m) for m in charged_lepton_masses) ** 2
    checks.append(
        Check(
            "koide-identity-value",
            close(koide, 2.0 / 3.0, 1e-5),
            f"Q_K={koide:.10f}; numerical proximity is empirical, not a dynamics proof",
        )
    )
    signed_roots = tuple(
        1.0
        + math.sqrt(2.0)
        * math.cos(math.pi / 3.0 + 2.0 * math.pi * index / 3.0)
        for index in range(1, 4)
    )
    signed_koide = sum(value * value for value in signed_roots) / sum(
        signed_roots
    ) ** 2
    principal_roots = tuple(abs(value) for value in signed_roots)
    physical_koide = sum(value * value for value in principal_roots) / sum(
        principal_roots
    ) ** 2
    checks.append(
        Check(
            "koide-principal-root-domain-counterexample",
            min(signed_roots) < 0.0
            and close(signed_koide, 2.0 / 3.0, 2e-15)
            and close(physical_koide, 0.4093647857764432, 2e-15),
            "signed parametrization gives 2/3 while principal roots give "
            f"{physical_koide:.10f}; the nonnegative-bracket condition is required",
        )
    )

    susceptibility = 75.5**4
    critical_density_mev4 = (2.5e-9) ** 4
    susceptibility_ratio = susceptibility / critical_density_mev4
    conditional_theta_bound = math.sqrt(0.1 / susceptibility_ratio)
    checks.append(
        Check(
            "strong-cp-unit-and-conditional-bound",
            close(conditional_theta_bound, 3.47e-22, 5e-25),
            f"chi/rho={susceptibility_ratio:.3e}, conditional |theta|<"
            f"{conditional_theta_bound:.3e}; this is not a dynamical prediction",
        )
    )

    m1, m2, m3 = 0.306, 8.566, 49.986  # meV; legacy table, arithmetic audit only
    sin2_theta12 = 0.307
    sin2_theta13 = 0.0222
    mee_terms = (
        (1.0 - sin2_theta13) * (1.0 - sin2_theta12) * m1,
        (1.0 - sin2_theta13) * sin2_theta12 * m2,
        sin2_theta13 * m3,
    )
    mee_maximum = sum(mee_terms)
    mee_minimum = max(0.0, 2.0 * max(mee_terms) - mee_maximum)
    checks.append(
        Check(
            "neutrino-effective-mass-triangle",
            close(mee_minimum, 1.25, 0.02) and close(mee_maximum, 3.89, 0.02),
            f"legacy inputs imply {mee_minimum:.3f}<=m_ee<={mee_maximum:.3f} meV; "
            "they do not predict the masses",
        )
    )

    sphaleron_prefactor = 405.0e-7 / (4.0 * math.pi**2 * 106.75 * 0.1)
    old_baryon_low = sphaleron_prefactor * 3.08e-5 * 1.0e-2
    old_baryon_high = sphaleron_prefactor * 3.08e-5 * 1.0e-1
    checks.append(
        Check(
            "removed-legacy-baryogenesis-counterexample",
            close(sphaleron_prefactor, 9.61e-8, 5e-11)
            and close(old_baryon_low, 3.0e-14, 5e-16)
            and close(old_baryon_high, 3.0e-13, 5e-15),
            f"prefactor={sphaleron_prefactor:.3e}, range={old_baryon_low:.3e}--"
            f"{old_baryon_high:.3e}; a transport calculation is still required",
        )
    )

    y_end, y_star, n_s, tensor_ratio = nonminimal_quartic_observables(0.49, 60.0)
    checks.append(
        Check(
            "finite-xi-inflation-arithmetic",
            close(y_end, 1.33905, 2e-5)
            and close(y_star, 11.35775, 2e-5)
            and close(n_s, 0.967717, 2e-6)
            and close(tensor_ratio, 0.0039683, 2e-7),
            f"y_end={y_end:.6f}, y_star={y_star:.6f}, n_s={n_s:.7f}, "
            f"r={tensor_ratio:.7f}; conditional on the proposed action",
        )
    )

    invalid_alpha_em_relation = sin2_theta_w * alpha_s ** (2.0 / 3.0)
    checks.append(
        Check(
            "removed-legacy-alpha-em-relation-counterexample",
            invalid_alpha_em_relation > 0.05,
            f"sW^2*alpha_s^(2/3)={invalid_alpha_em_relation:.6f}, not alpha_em(MZ); "
            "the counterexampled relation must remain absent from the body",
        )
    )

    # Cross-chapter rescue audit.  These checks preserve numerical descendants
    # without promoting their supplied bridges to first-principles predictions.
    alpha_em_mz = 1.0 / 127.95

    def alpha_closure_residual(candidate: float) -> float:
        candidate_sin2 = 4.0 * candidate ** (4.0 / 3.0)
        candidate_alpha2 = alpha_em_mz / candidate_sin2
        return candidate + candidate_alpha2 + alpha_em_mz - 1.0 / (2.0 * math.pi)

    alpha_s_closure = bisect(alpha_closure_residual, 0.10, 0.15)
    sin2_closure = 4.0 * alpha_s_closure ** (4.0 / 3.0)
    alpha2_closure = alpha_em_mz / sin2_closure
    checks.append(
        Check(
            "cross-chapter-alpha-closure",
            close(alpha_s_closure, 0.1173186647, 5e-11)
            and close(sin2_closure, 0.2297291680, 5e-11)
            and close(alpha2_closure, 0.0340207254, 5e-11),
            f"alpha_em(MZ)=1/127.95 supplied -> alpha_s={alpha_s_closure:.10f}, "
            f"sW^2={sin2_closure:.10f}, alpha2={alpha2_closure:.10f}; "
            "this is Bridge/Phenomenology, not a zero-input result",
        )
    )

    alpha_closure_minimum = (alpha_em_mz / 3.0) ** (3.0 / 7.0)
    alpha_closure_minimum_value = alpha_closure_residual(alpha_closure_minimum)
    alpha_closure_small = bisect(
        alpha_closure_residual,
        1.0e-6,
        alpha_closure_minimum,
    )
    alpha_closure_curvature = (
        7.0
        * alpha_em_mz
        / 9.0
        * alpha_closure_minimum ** (-10.0 / 3.0)
    )
    checks.append(
        Check(
            "coupling-closure-two-positive-roots-witness",
            alpha_closure_curvature > 0.0
            and alpha_closure_minimum_value < 0.0
            and 0.0 < alpha_closure_small < alpha_closure_minimum < alpha_s_closure
            and close(alpha_closure_small, 0.05286787, 5e-9)
            and close(alpha_s_closure, 0.11731866, 5e-9),
            f"strictly convex residual has minimum at {alpha_closure_minimum:.8f} "
            f"with fmin={alpha_closure_minimum_value:.8f}; positive roots="
            f"({alpha_closure_small:.8f}, {alpha_s_closure:.8f})",
        )
    )

    alpha_total = 1.0 / (2.0 * math.pi)
    coupling_ratio_sum = (0.01008 + 0.03353 + alpha_s) / alpha_total
    ratio_3layer = alpha_s * (3.0 + q * coupling_ratio_sum) + alpha_s * delta * (
        1.0 + q * delta
    )
    omega_lambda_3layer = (1.0 - q) / (1.0 + ratio_3layer)
    omega_dm_3layer = (1.0 - q) * ratio_3layer / (1.0 + ratio_3layer)
    omega_m_3layer = q + omega_dm_3layer
    checks.append(
        Check(
            "cross-chapter-three-layer-readout",
            close(coupling_ratio_sum, 1.0147344271, 5e-11)
            and close(ratio_3layer, 0.3806266173, 5e-11)
            and close(omega_lambda_3layer, 0.6890735470, 5e-10)
            and close(omega_dm_3layer, 0.2622797333, 5e-10)
            and close(omega_m_3layer, 0.3109264530, 5e-10),
            f"supplied cSigma={coupling_ratio_sum:.10f} -> R={ratio_3layer:.10f}, "
            f"(OmegaL,OmegaDM,Omegam)=({omega_lambda_3layer:.10f},"
            f"{omega_dm_3layer:.10f},{omega_m_3layer:.10f}); selection is Phenomenology",
        )
    )

    n_gauge = 12.0
    transition_count = 1.5 * D * n_gauge
    phase_area = 0.5 * math.pi**2
    log_entropy = phase_area * transition_count - math.pi * delta * (1.0 - q)
    transition_ns = 1.0 - 2.0 / transition_count
    checks.append(
        Check(
            "cross-chapter-phase-area-transition",
            close(transition_count, 57.1996516214, 5e-10)
            and close(log_entropy, 281.7376886303, 5e-10)
            and close(transition_ns, 0.9650347521, 5e-11),
            f"Ne={transition_count:.10f}, logS={log_entropy:.10f}, "
            f"n_s candidate={transition_ns:.10f}; the transition-count bridge is supplied",
        )
    )

    t_planck_s = 5.391247e-44
    mpc_km = 3.0856775814913673e19
    h0_readout = (
        math.sqrt(math.pi) * math.exp(-0.5 * log_entropy) / t_planck_s * mpc_km
    )
    m_planck_ev = 1.220910e28
    omega_lambda_supplied = 0.6891
    rho_lambda_quarter_mev = 1.0e3 * (
        omega_lambda_supplied
        * (3.0 / 8.0)
        * m_planck_ev**4
        / math.exp(log_entropy)
    ) ** 0.25
    checks.append(
        Check(
            "cross-chapter-holographic-readouts",
            close(h0_readout, 67.2472445605, 5e-9)
            and close(rho_lambda_quarter_mev, 2.2412027734, 5e-10),
            f"H0={h0_readout:.9f} km/s/Mpc, "
            f"rhoLambda^(1/4)={rho_lambda_quarter_mev:.10f} meV; "
            "both remain locked to the conditional entropy bridge",
        )
    )

    sigma = 1.0 - q
    projected_drive = (
        (2.0 / math.pi) * sigma ** (D / (D + 1.0)) * q * sigma
    )
    primordial_as = (
        projected_drive**2
        / sigma**2
        * q
        / (2.0 * math.pi * transition_count**2)
    )
    checks.append(
        Check(
            "cross-chapter-primordial-readout",
            close(projected_drive, 0.02836622125, 5e-11)
            and close(primordial_as, 2.1038087465e-9, 5e-19),
            f"Q_cand={projected_drive:.11f}, A_s={primordial_as:.11e}; "
            "the projected observable is Phenomenology and the raw branch fails",
        )
    )

    legacy_ratio = 0.38063
    neutrino_prefactor = (
        delta**4
        * (1.0 - alpha_s / math.pi)
        / ((16.0 * math.pi**2) ** 2 * 32.0 * math.pi**3 * (1.0 + legacy_ratio))
    )
    neutrino_masses_mev = tuple(
        neutrino_prefactor
        * charged_mass ** (5.0 / 8.0)
        * charged_lepton_masses[-1] ** (3.0 / 8.0)
        * 1.0e9
        for charged_mass in charged_lepton_masses
    )
    delta_m21_ev2 = (neutrino_masses_mev[1] ** 2 - neutrino_masses_mev[0] ** 2) * 1e-6
    delta_m31_ev2 = (neutrino_masses_mev[2] ** 2 - neutrino_masses_mev[0] ** 2) * 1e-6
    checks.append(
        Check(
            "cross-chapter-neutrino-legacy-readout",
            all(
                close(value, expected, 7e-4)
                for value, expected in zip(neutrino_masses_mev, (0.306, 8.566, 49.986))
            )
            and close(sum(neutrino_masses_mev), 58.8572, 1e-4)
            and close(delta_m21_ev2, 7.32745e-5, 5e-10)
            and close(delta_m31_ev2, 2.498489e-3, 5e-10),
            f"legacy masses={tuple(round(value, 6) for value in neutrino_masses_mev)} meV, "
            f"sum={sum(neutrino_masses_mev):.6f} meV; arithmetic passes but LNV is absent",
        )
    )

    portal_mass_fixed = 43.7677
    invisible_bound = 0.107
    fixed_phase_space = math.sqrt(
        1.0 - 4.0 * portal_mass_fixed**2 / higgs_mass**2
    )
    allowed_invisible_width = (
        invisible_bound / (1.0 - invisible_bound) * sm_higgs_width
    )
    portal_lambda_limit = math.sqrt(
        allowed_invisible_width
        * 8.0
        * math.pi
        * higgs_mass
        / (v_ew**2 * fixed_phase_space)
    )
    portal_m0_squared = portal_mass_fixed**2 - portal_lambda_limit * v_ew**2
    portal_m0 = math.sqrt(portal_m0_squared)
    checks.append(
        Check(
            "portal-alternative-single-gate",
            close(portal_lambda_limit, 0.00595010827, 5e-12)
            and close(portal_m0_squared, 1554.8904835, 5e-6)
            and close(portal_m0, 39.43209966, 5e-8)
            and 70.0 > higgs_mass / 2.0,
            f"at m=43.7677 GeV, |lambda|<={portal_lambda_limit:.10f}, "
            f"m0={portal_m0:.8f} GeV; 70 GeV closes h->phiphi, "
            "and neither restores the removed m0=0, lambda=delta^2 parent",
        )
    )

    alpha_w_legacy = 0.03352
    j_legacy = 3.12e-5
    g_star_legacy = 106.75
    wall_speed_legacy = 0.118
    eta_b_legacy = (
        405.0
        * 25.0
        * alpha_w_legacy**5
        / (4.0 * math.pi**2 * g_star_legacy * wall_speed_legacy)
        * j_legacy
        / wall_speed_legacy
    )
    checks.append(
        Check(
            "cross-chapter-baryogenesis-ansatz",
            close(eta_b_legacy, 2.2781275739e-10, 5e-20),
            f"distinct eta_B ansatz gives {eta_b_legacy:.11e}; "
            "the reported 6.14e-10 branch uses fitted h=0.01670 and remains Phenomenology",
        )
    )

    neutrino_summary = texts.get("07_중성미자_질량.md", "")
    stale_neutrino_rejection = (
        "| \\(m_\\nu\\propto m_l\\) 또는 \\(m_l^{5/8}\\) | [Rejected]"
        in neutrino_summary
    )
    checks.append(
        Check(
            "neutrino-legacy-status-split",
            not stale_neutrino_rejection
            and "| fractional ansatz의 질량 readout | **[경험식]**"
            in neutrino_summary
            and "| 현재 portal에서 \\(C_{ij}=0\\) | **[정리]**"
            in neutrino_summary
            and neutrino_summary.count("[Rejected]") == 0,
            "the false portal-generation parent is removed; the numerical legacy "
            "ansatz remains empirical and the proved C_ij=0 no-go remains"
            if not stale_neutrino_rejection
            else "a stale blanket Rejected row still swallows the rescued numerical ansatz",
        )
    )

    # Claim-hygiene anchors that must be visible in the revised prose.
    semantic_requirements = {
        "01_차원의_유일성.md": (
            "d = 4",
            "d=0,3",
            "양의 정수해는 \\(d=3\\)",
            "방향과 리만 계량",
        ),
        "02_에스컬레이터.md": (
            "외부 입력",
            "rpp2026-rev-qcd.pdf",
            "0.11731866",
            "1/127.95",
        ),
        "03_자유매개변수.md": (
            "소멸확률",
            "t=-1",
            "X\\ge0",
            "q_{\\rm ext}=\\min",
            "g_D''(x)=-D^2F_D(x)<0",
            "I(P)=P^c",
        ),
        "04_해결한_난제.md": (
            "| 중성미자 질량 | **[미완성]**",
            "| 양성자 반경의 CE 기여 | **[미완성]**",
            "281.73769",
            "0.344351",
            "even-\\(\\varphi\\) loop 총합은",
        ),
        "05_인플레이션.md": (
            "m_{\\varphi,{\\rm phys}}",
            "k_*=0.05",
            "독립 입력",
            "tree-level curvature mass는 **[정리]**",
            "정확한 \\(Z_2\\)",
        ),
        "06_강한_CP.md": (
            "\\bar\\theta",
            "rpp2026-rev-axions.pdf",
            "다시 계산",
            "조건부 진공에너지 경계",
        ),
        "07_중성미자_질량.md": (
            "B-L",
            "\\lambda_5",
            "U(1)_{B-L}",
            "m_{\\varphi,{\\rm tree}}",
            "58.86",
            "legacy spectrum을 입력",
            "fractional ansatz의 질량 readout",
            "2.871\\times10^{-13}",
            "6.064\\times10^{14}",
            "tree-level curvature mass",
        ),
        "08_바리온_비대칭.md": (
            "0.107",
            "hypercharge-neutral",
            "\\Omega_b=q",
            "2.6203\\times10^{-7}",
            "2.66\\times10^9",
            "\\kappa_{\\rm tot}",
            "0.4165",
            "6.38\\times10^9",
            "v_{\\rm DI}=v_{\\rm EW}/\\sqrt2",
        ),
        "09_페르미온_질량.md": ("Q_K=\\frac2N", "공통 \\(A,\\phi,r\\)", "m_t^{\\rm MC}", "principal", "on-shell scheme"),
        "10_공리_정당화.md": (
            "C(x)=1-x",
            "x_Z=q_{\\rm ext}",
            "식별 ansatz",
            "0<Z_0<\\infty",
        ),
        "11_게이지_격자와_인과성.md": (
            "전역",
            "determinant-class",
            "\\mathcal D(e^{a_tH})",
            "graded",
            "\\det g=-N^2\\det h",
            "시간독립 self-adjoint",
            "`[미완성]`인 toy 모식도",
        ),
        "12_전이구간.md": (
            "D>0",
            "q_{\\rm ext}=1",
            "0.96503475",
            "2.1038087",
            "22.452",
            "F_D(x)-x",
            "|_{r=0}",
            "D>1\\)의 비자명 가지",
            "D=0\\)에서는 \\(F_0\\equiv1\\)이므로 역함수가 없다",
        ),
        "13_위상공간.md": (
            "유한 sparse grammar",
            "independent holdout",
            "product Lebesgue measure",
        ),
        "14_자기재귀성_대칭.md": (
            "소멸확률",
            "완전생존",
            "C(x)=1-x",
            "Jacobian과 안정성 `[정리]`",
            "단조극한",
        ),
    }
    missing_semantics: list[str] = []
    for filename, fragments in semantic_requirements.items():
        text = texts.get(filename, "")
        absent = [fragment for fragment in fragments if fragment not in text]
        if absent:
            missing_semantics.append(f"{filename}: {absent}")
    checks.append(
        Check(
            "critical-corrections-present",
            not missing_semantics,
            "all critical corrections are explicit"
            if not missing_semantics
            else "; ".join(missing_semantics),
        )
    )

    rejection_loop_report = build_rejection_loop_report()
    rejection_loop_checks = validate_rejection_loop_report(rejection_loop_report)
    failed_rejection_loop_checks = [
        check.name for check in rejection_loop_checks if not check.passed
    ]
    checks.append(
        Check(
            "rejection-loop-engine-self-checks",
            len(rejection_loop_checks) == 22 and not failed_rejection_loop_checks,
            f"{len(rejection_loop_checks)}/22 fail-closed engine checks pass"
            if not failed_rejection_loop_checks
            else f"failed engine checks: {failed_rejection_loop_checks}",
        )
    )
    checks.append(
        Check(
            "rejection-loop-coverage-lock",
            len(rejection_loop_report.loops) == 16
            and rejection_loop_report.source_rejected_literal_occurrences == 0
            and rejection_loop_report.source_rejected_occurrences == 0
            and rejection_loop_report.routed_rejected_occurrences == 0
            and rejection_loop_report.occurrence_routes == ()
            and rejection_loop_report.source_rejected_literal_occurrences
            == rejection_loop_report.source_rejected_occurrences
            + rejection_loop_report.excluded_taxonomy_occurrences
            and rejection_loop_report.excluded_taxonomy_occurrences == 0
            and rejection_loop_report.routed_rejected_occurrences
            == rejection_loop_report.source_rejected_occurrences
            == len(rejection_loop_report.occurrence_routes)
            and all(
                route.semantic_gate_ids and route.loop_ids
                for route in rejection_loop_report.occurrence_routes
            )
            and len(rejection_loop_report.deleted_parent_regression_witnesses) == 18
            and all(
                witness.passed
                for witness in rejection_loop_report.deleted_parent_regression_witnesses
            )
            and bool(rejection_loop_report.regression_witness_registry_sha256)
            and rejection_loop_report.original_claims_promoted == 0
            and rejection_loop_report.ce_specific_physical_claims_closed == 0,
            "canonical prose contains zero deleted-parent verdict markers; "
            f"{len(rejection_loop_report.loops)} internal loop families and "
            f"{len(rejection_loop_report.semantic_gate_definitions)} semantic gates retained; "
            f"deleted-parent witnesses="
            f"{len(rejection_loop_report.deleted_parent_regression_witnesses)}; "
            f"original promotions="
            f"{rejection_loop_report.original_claims_promoted}, CE closures="
            f"{rejection_loop_report.ce_specific_physical_claims_closed}",
        )
    )

    improvement_report = build_improvement_loop_report()
    improvement_checks = validate_improvement_loop_report(improvement_report)
    failed_improvement_checks = [
        check.name for check in improvement_checks if not check.passed
    ]
    checks.append(
        Check(
            "improvement-loop-engine-self-checks",
            len(improvement_checks) == 22 and not failed_improvement_checks,
            f"{len(improvement_checks)}/22 fail-closed improvement checks pass"
            if not failed_improvement_checks
            else f"failed improvement checks: {failed_improvement_checks}",
        )
    )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    if args.json and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    checks = validate()
    passed = all(check.passed for check in checks)
    inventory = claim_status_inventory()
    substantive_inventory = tuple(
        item for item in inventory if not item.taxonomy_definition
    )
    audit_scope = {
        "implemented_checks_passed": passed,
        "literal_status_occurrences": len(inventory),
        "substantive_status_occurrences": len(substantive_inventory),
        "substantive_by_status": {
            token: sum(item.status == token for item in substantive_inventory)
            for token in STATUS_TOKENS
        },
        "deleted_parent_prose_occurrences": sum(
            path.read_text(encoding="utf-8").count("[Rejected]")
            for path in CANONICAL_PROSE
        ),
        "all_deleted_parents_absent_from_prose": any(
            check.name == "rejection-loop-coverage-lock" and check.passed
            for check in checks
        ),
        "deleted_counterexample_regression_witnesses": 18,
        "all_provenance_claims_independently_proved": False,
        "no_counterexample_implies_truth": False,
        "ce_specific_physical_claims_closed": 0,
        "external_artifact_live_verification": "NOT_RUN_BY_THIS_COMMAND",
    }
    if args.json:
        print(
            json.dumps(
                {
                    "passed": passed,
                    "audit_scope": audit_scope,
                    "checks": [asdict(c) for c in checks],
                    "claim_status_inventory": [asdict(item) for item in inventory],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for check in checks:
            print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        print(f"\nRESULT: {'PASS' if passed else 'FAIL'} ({sum(c.passed for c in checks)}/{len(checks)})")
        print(
            "SCOPE: implemented document/arithmetic/routing checks only; "
            "absence of a counterexample is not proof, claims without a Rejected disposition are not "
            "all independently proved, and CE-specific physical closures remain 0."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
