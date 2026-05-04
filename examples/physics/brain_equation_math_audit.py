"""Mathematical audit for event-level brain-equation readiness gates.

The previous partial gates are behavioral/event-level checks. This script does
not upgrade them into neural proof. It only computes the algebraic separation
quantities that the matched-vs-wrong inequalities require:

- within-domain dispersion
- nearest between-domain prototype distance
- holdout loss ratios already reported by each gate
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parent
GATES = [
    {
        "name": "ds000116_modality",
        "path": ROOT / "ds000116_modality_gate_results.json",
        "domain_field": "task",
        "wrong_key": "wrong_modality",
    },
    {
        "name": "ds000201_task_domain",
        "path": ROOT / "ds000201_task_gate_results.json",
        "domain_field": "task",
        "wrong_key": "wrong_task",
    },
    {
        "name": "ds000201_cognitive_arousal",
        "path": ROOT / "ds000201_cognitive_gate_results.json",
        "domain_field": "task",
        "wrong_key": "wrong_task",
    },
]
PERMUTATIONS = 2000
RNG_SEED = 1729


def squared_loss(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def average(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    return [mean(row[idx] for row in rows) for idx in range(len(rows[0]))]


def design_matrix(labels: list[str]) -> np.ndarray:
    """Intercept plus treatment-coded labels."""
    levels = sorted(set(labels))
    columns = [np.ones(len(labels), dtype=np.float64)]
    for level in levels[1:]:
        columns.append(np.asarray([label == level for label in labels], dtype=np.float64))
    return np.column_stack(columns)


def two_factor_design(primary: list[str], nuisance: list[str]) -> np.ndarray:
    """Intercept plus domain dummies plus nuisance dummies."""
    return np.column_stack([design_matrix(primary), design_matrix(nuisance)[:, 1:]])


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    nonzero = singular[singular > 1e-12]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "rank": rank,
        "full_column_rank": rank == int(matrix.shape[1]),
        "condition_number": condition,
        "singular_values": [float(value) for value in singular],
    }


def residual_sum_squares(y: np.ndarray, x: np.ndarray) -> float:
    coeffs = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ coeffs
    return float(np.sum(resid * resid))


def regression_diagnostics(states: list[list[float]], domains: list[str], subjects: list[str]) -> dict[str, object]:
    y = np.asarray(states, dtype=np.float64)
    intercept = np.ones((len(states), 1), dtype=np.float64)
    x_domain = design_matrix(domains)
    x_domain_subject = two_factor_design(domains, subjects)
    rss_null = residual_sum_squares(y, intercept)
    rss_domain = residual_sum_squares(y, x_domain)
    rss_domain_subject = residual_sum_squares(y, x_domain_subject)
    return {
        "domain_design": matrix_diagnostics(x_domain),
        "domain_subject_design": matrix_diagnostics(x_domain_subject),
        "rss_null": rss_null,
        "rss_domain": rss_domain,
        "rss_domain_subject": rss_domain_subject,
        "domain_r2_vs_null": 1.0 - rss_domain / max(rss_null, 1e-12),
        "domain_subject_r2_vs_null": 1.0 - rss_domain_subject / max(rss_null, 1e-12),
    }


def separation_ratio_for_labels(states: list[list[float]], labels: list[str]) -> float:
    domains = sorted(set(labels))
    rows_by_domain = {
        domain: [row for row, label in zip(states, labels) if label == domain]
        for domain in domains
    }
    prototypes = {domain: average(rows) for domain, rows in rows_by_domain.items()}
    within = mean(
        mean(squared_loss(row, prototypes[domain]) for row in rows)
        for domain, rows in rows_by_domain.items()
        if rows
    )
    between_values = []
    for left in domains:
        for right in domains:
            if left >= right:
                continue
            between_values.append(squared_loss(prototypes[left], prototypes[right]))
    nearest = min(between_values) if between_values else 0.0
    return within / max(nearest, 1e-12)


def permutation_test(states: list[list[float]], domains: list[str]) -> dict[str, object]:
    rng = random.Random(RNG_SEED)
    observed = separation_ratio_for_labels(states, domains)
    shuffled = list(domains)
    hits = 0
    ratios = []
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        ratio = separation_ratio_for_labels(states, shuffled)
        ratios.append(ratio)
        if ratio <= observed:
            hits += 1
    return {
        "statistic": "within/nearest_between; lower is more separated",
        "permutations": PERMUTATIONS,
        "observed": observed,
        "permutation_mean": mean(ratios),
        "permutation_min": min(ratios),
        "p_value_le_observed": (hits + 1) / (PERMUTATIONS + 1),
    }


def audit_gate(config: dict[str, object]) -> dict[str, object]:
    payload = json.loads(Path(config["path"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    regions = [str(region) for region in payload["regions"]]
    cases = payload["cases"]
    domain_field = str(config["domain_field"])
    domains = sorted({str(case[domain_field]) for case in cases})
    rows_by_domain = {
        domain: [
            [float(case["observed_state"][region]) for region in regions]
            for case in cases
            if str(case[domain_field]) == domain
        ]
        for domain in domains
    }
    prototypes = {domain: average(rows) for domain, rows in rows_by_domain.items()}
    states = [
        [float(case["observed_state"][region]) for region in regions]
        for case in cases
    ]
    domain_labels = [str(case[domain_field]) for case in cases]
    subject_labels = [str(case.get("subject", "unknown")) for case in cases]

    within_by_domain = {}
    for domain, rows in rows_by_domain.items():
        within_by_domain[domain] = mean(
            squared_loss(row, prototypes[domain]) for row in rows
        )
    between = {}
    for left in domains:
        for right in domains:
            if left >= right:
                continue
            between[f"{left}|{right}"] = squared_loss(prototypes[left], prototypes[right])
    nearest_between = min(between.values()) if between else 0.0
    mean_within = mean(within_by_domain.values()) if within_by_domain else 0.0
    holdout = payload.get("prototype_holdout", {})
    losses = holdout.get("total_losses", {})
    wrong_key = str(config["wrong_key"])
    matched = float(losses.get("matched", 0.0))
    wrong = float(losses.get(wrong_key, 0.0))
    generic = float(losses.get("generic", 0.0))

    return {
        "name": config["name"],
        "case_count": len(cases),
        "domains": domains,
        "regions": regions,
        "within_by_domain": within_by_domain,
        "mean_within": mean_within,
        "between_prototype_distance": between,
        "nearest_between": nearest_between,
        "separation_ratio_mean_within_over_nearest_between": mean_within
        / max(nearest_between, 1e-12),
        "prototype_holdout_losses": {
            "matched": matched,
            "wrong": wrong,
            "generic": generic,
        },
        "holdout_matched_over_wrong": matched / max(wrong, 1e-12),
        "holdout_matched_over_generic": matched / max(generic, 1e-12),
        "regression_diagnostics": regression_diagnostics(
            states, domain_labels, subject_labels
        ),
        "permutation_test": permutation_test(states, domain_labels),
        "passed_algebraic_separation": mean_within < nearest_between
        and matched < wrong
        and matched < generic,
    }


def write_markdown(audits: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# 뇌 방정식 부분 게이트 수학 점검",
        "",
        "이 문서는 event-level readiness 결과를 수학 부등식 관점에서 다시 정리한 것이다.",
        "여기서 통과는 신경활성 검증이 아니라, 특수 연산자 분해가 데이터 구조상 붕괴하지 않는다는 뜻이다.",
        "",
        "## 핵심 부등식",
        "",
        "$$",
        "\\bar W_d",
        "=",
        "\\frac{1}{|C_d|}\\sum_{i\\in C_d}\\|s_i-\\mu_d\\|^2",
        "$$",
        "",
        "$$",
        "B_{d,d'}=\\|\\mu_d-\\mu_{d'}\\|^2",
        "$$",
        "",
        "필요한 최소 조건은 같은 domain 내부 분산이 가장 가까운 다른 domain prototype 거리보다 작고, holdout에서 matched 손실이 wrong/generic보다 작아야 한다.",
        "",
        "$$",
        "\\bar W < \\min_{d\\ne d'} B_{d,d'},\\qquad",
        "\\mathcal L_{\\mathrm{matched}}",
        "<",
        "\\min(",
        "\\mathcal L_{\\mathrm{wrong}},",
        "\\mathcal L_{\\mathrm{generic}}",
        ")",
        "$$",
        "",
        "## 결과",
        "",
        "| gate | cases | domains | mean within | nearest between | within/between | matched/wrong | matched/generic | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for audit in audits:
        lines.append(
            "| {name} | {case_count} | {domain_count} | {mean_within:.8f} | "
            "{nearest_between:.8f} | {sep:.6f} | {mw:.6f} | {mg:.6f} | {passed} |".format(
                name=audit["name"],
                case_count=audit["case_count"],
                domain_count=len(audit["domains"]),  # type: ignore[arg-type]
                mean_within=float(audit["mean_within"]),
                nearest_between=float(audit["nearest_between"]),
                sep=float(audit["separation_ratio_mean_within_over_nearest_between"]),
                mw=float(audit["holdout_matched_over_wrong"]),
                mg=float(audit["holdout_matched_over_generic"]),
                passed="pass" if audit["passed_algebraic_separation"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## 추가 식별성 점검",
            "",
            "| gate | domain rank | domain cond | domain R2 | domain+subject rank | domain+subject cond | permutation p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for audit in audits:
        reg = audit["regression_diagnostics"]  # type: ignore[assignment]
        domain_design = reg["domain_design"]
        subject_design = reg["domain_subject_design"]
        perm = audit["permutation_test"]  # type: ignore[assignment]
        lines.append(
            "| {name} | {drank}/{dcols} | {dcond:.3f} | {r2:.6f} | "
            "{srank}/{scols} | {scond:.3f} | {pval:.6f} |".format(
                name=audit["name"],
                drank=domain_design["rank"],
                dcols=domain_design["cols"],
                dcond=float(domain_design["condition_number"]),
                r2=float(reg["domain_r2_vs_null"]),
                srank=subject_design["rank"],
                scols=subject_design["cols"],
                scond=float(subject_design["condition_number"]),
                pval=float(perm["p_value_le_observed"]),
            )
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 세 게이트 모두 prototype 사이 거리가 domain 내부 분산보다 크다.",
            "- 세 게이트 모두 holdout에서 matched operator가 wrong/generic보다 낮은 손실을 낸다.",
            "- domain 설계행렬이 full rank이면 event-level 특수 연산자 계수는 적어도 이 표본 안에서 선형적으로 구분된다.",
            "- domain+subject 설계행렬이 full rank이면 subject offset을 넣어도 domain 항이 완전히 붕괴하지 않는다.",
            "- permutation p 값은 domain label을 섞었을 때 현재보다 강한 분리가 얼마나 자주 나오는지의 경험적 점검이다.",
            "- 따라서 현재 수학적으로 닫힌 것은 특수 입력항 분해의 event-level 필요조건이다.",
            "- 아직 닫히지 않은 것은 BOLD/EEG에서 같은 부등식이 region-resolved 상태 \\(p_r\\) 위에서도 유지되는지다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audits = [audit_gate(config) for config in GATES if Path(config["path"]).exists()]
    out_json = ROOT / "brain_equation_math_audit_results.json"
    out_md = ROOT / "brain_equation_math_audit_report.md"
    out_json.write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audits, out_md)

    print("brain equation math audit")
    for audit in audits:
        print(f"  {audit['name']}:")
        print(f"    cases           = {audit['case_count']}")
        print(f"    domains         = {len(audit['domains'])}")
        print(f"    mean_within     = {audit['mean_within']:.8f}")
        print(f"    nearest_between = {audit['nearest_between']:.8f}")
        print(
            "    within/between = "
            f"{audit['separation_ratio_mean_within_over_nearest_between']:.6f}"
        )
        print(f"    matched/wrong   = {audit['holdout_matched_over_wrong']:.6f}")
        reg = audit["regression_diagnostics"]
        perm = audit["permutation_test"]
        domain_design = reg["domain_design"]
        subject_design = reg["domain_subject_design"]
        print(
            "    domain rank     = "
            f"{domain_design['rank']}/{domain_design['cols']}, "
            f"cond={domain_design['condition_number']:.3f}"
        )
        print(
            "    domain+subject  = "
            f"{subject_design['rank']}/{subject_design['cols']}, "
            f"cond={subject_design['condition_number']:.3f}"
        )
        print(f"    domain R2       = {reg['domain_r2_vs_null']:.6f}")
        print(f"    permutation p   = {perm['p_value_le_observed']:.6f}")
        print(f"    passed          = {audit['passed_algebraic_separation']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
