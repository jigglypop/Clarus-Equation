"""Detailed error analysis for CE Claim Residual benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ErrorRow:
    record_id: str
    score: float
    action: float
    accepted_fraction: float
    predicted_hallucinated: bool
    actual_hallucinated: bool

    @property
    def error_type(self) -> str:
        if self.predicted_hallucinated and not self.actual_hallucinated:
            return "FP"
        if not self.predicted_hallucinated and self.actual_hallucinated:
            return "FN"
        return "OK"


@dataclass(frozen=True)
class Quantiles:
    count: int
    mean: float
    p10: float
    p50: float
    p90: float


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def load_error_csv(path: Path) -> tuple[ErrorRow, ...]:
    rows: list[ErrorRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ErrorRow(
                    record_id=str(row["record_id"]),
                    score=float(row["score"]),
                    action=float(row["action"]),
                    accepted_fraction=float(row["accepted_fraction"]),
                    predicted_hallucinated=row["predicted_hallucinated"] == "1",
                    actual_hallucinated=row["actual_hallucinated"] == "1",
                )
            )
    return tuple(rows)


def quantiles(values: Iterable[float]) -> Quantiles | None:
    sorted_values = sorted(values)
    if not sorted_values:
        return None

    def pick(p: float) -> float:
        return sorted_values[min(len(sorted_values) - 1, int(p * (len(sorted_values) - 1)))]

    return Quantiles(
        count=len(sorted_values),
        mean=statistics.mean(sorted_values),
        p10=pick(0.10),
        p50=pick(0.50),
        p90=pick(0.90),
    )


def shorten(value: Any, width: int = 360) -> str:
    return textwrap.shorten(" ".join(str(value).split()), width=width, placeholder=" ...")


def counter_table(counter: Counter[Any], *, limit: int = 12) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- `{key}`: {count}" for key, count in counter.most_common(limit)]


def quantile_line(name: str, values: Iterable[float]) -> str:
    q = quantiles(values)
    if q is None:
        return f"- {name}: n/a"
    return (
        f"- {name}: n={q.count}, mean={q.mean:.4f}, "
        f"p10={q.p10:.4f}, p50={q.p50:.4f}, p90={q.p90:.4f}"
    )


def split_errors(errors: Sequence[ErrorRow]) -> dict[str, tuple[ErrorRow, ...]]:
    return {
        "FP": tuple(error for error in errors if error.error_type == "FP"),
        "FN": tuple(error for error in errors if error.error_type == "FN"),
    }


def base_confusion(records: Mapping[str, Mapping[str, Any]], errors: Sequence[ErrorRow]) -> dict[str, int]:
    actual_positive = sum(1 for record in records.values() if bool(record.get("is_hallucinated")))
    actual_negative = len(records) - actual_positive
    fp = sum(1 for error in errors if error.error_type == "FP")
    fn = sum(1 for error in errors if error.error_type == "FN")
    return {
        "total": len(records),
        "actual_positive": actual_positive,
        "actual_negative": actual_negative,
        "fp": fp,
        "fn": fn,
        "tp": actual_positive - fn,
        "tn": actual_negative - fp,
    }


def ragtruth_section(
    records: Mapping[str, Mapping[str, Any]],
    errors: Sequence[ErrorRow],
) -> list[str]:
    lines = ["## RAGTruth Error Anatomy"]
    confusion = base_confusion(records, errors)
    lines.extend(
        [
            "",
            "### Confusion",
            f"- total: {confusion['total']}",
            f"- actual hallucinated: {confusion['actual_positive']}",
            f"- actual non-hallucinated: {confusion['actual_negative']}",
            f"- true positive: {confusion['tp']}",
            f"- false positive: {confusion['fp']}",
            f"- true negative: {confusion['tn']}",
            f"- false negative: {confusion['fn']}",
        ]
    )
    by_type = split_errors(errors)
    for error_type in ("FN", "FP"):
        subset = by_type[error_type]
        rows = [records[error.record_id] for error in subset if error.record_id in records]
        lines.extend(["", f"### {error_type} Breakdown", quantile_line("action", (e.action for e in subset))])
        lines.append("")
        lines.append("task_type:")
        lines.extend(counter_table(Counter(row.get("task_type", "") for row in rows)))
        lines.append("")
        lines.append("model:")
        lines.extend(counter_table(Counter(row.get("model", "") for row in rows)))
        lines.append("")
        lines.append("label_type:")
        lines.extend(
            counter_table(
                Counter(
                    label_type
                    for row in rows
                    for label_type in row.get("label_types", [])
                )
            )
        )
        lines.append("")
        lines.append("label_count:")
        lines.extend(counter_table(Counter(row.get("label_count", 0) for row in rows)))
        lines.extend(example_rows(records, subset, title=f"{error_type} representative examples"))
    lines.extend(
        [
            "",
            "### RAGTruth Root Causes",
            "- False negatives are mostly lexical-near hallucinations: the response shares many tokens with the source, but changes dates, numbers, negation, scope, causality, or introduces a small unsupported clause.",
            "- False positives are mostly faithful paraphrases, especially in Summary and Data2txt. The lexical residual punishes valid wording changes because it has no entailment model.",
            "- Response-level scoring loses span locality. A small hallucinated span can be diluted by an otherwise faithful response, while a faithful abstract summary can look lexically far from the source.",
            "- The current context is truncated to 2000 characters for speed. Missing late evidence can raise false positives or distort action calibration.",
        ]
    )
    return lines


def load_faithbench_raw(input_dir: Path) -> tuple[dict[str, list[str]], dict[str, str], dict[str, int]]:
    labels_by_id: dict[str, list[str]] = {}
    summarizer_by_id: dict[str, str] = {}
    annotation_count_by_id: dict[str, int] = {}
    for batch_path in sorted(input_dir.glob("batch_*.json")):
        data = json.loads(batch_path.read_text(encoding="utf-8"))
        for sample in data.get("samples", []):
            record_id = f"{batch_path.stem}_{sample.get('sample_id')}"
            labels: list[str] = []
            for annotation in sample.get("annotations", []):
                labels.extend(str(label) for label in annotation.get("label", []))
            labels_by_id[record_id] = labels
            summarizer_by_id[record_id] = str(sample.get("metadata", {}).get("summarizer", ""))
            annotation_count_by_id[record_id] = len(sample.get("annotations", []))
    return labels_by_id, summarizer_by_id, annotation_count_by_id


def faithbench_section(
    records: Mapping[str, Mapping[str, Any]],
    errors: Sequence[ErrorRow],
    raw_dir: Path,
) -> list[str]:
    labels_by_id, summarizer_by_id, annotation_count_by_id = load_faithbench_raw(raw_dir)
    lines = ["## FaithBench Error Anatomy"]
    confusion = base_confusion(records, errors)
    lines.extend(
        [
            "",
            "### Confusion",
            f"- total: {confusion['total']}",
            f"- actual hallucinated: {confusion['actual_positive']}",
            f"- actual non-hallucinated: {confusion['actual_negative']}",
            f"- true positive: {confusion['tp']}",
            f"- false positive: {confusion['fp']}",
            f"- true negative: {confusion['tn']}",
            f"- false negative: {confusion['fn']}",
        ]
    )
    by_type = split_errors(errors)
    for error_type in ("FN", "FP"):
        subset = by_type[error_type]
        ids = [error.record_id for error in subset]
        lines.extend(["", f"### {error_type} Breakdown", quantile_line("action", (e.action for e in subset))])
        lines.append("")
        lines.append("summarizer:")
        lines.extend(counter_table(Counter(summarizer_by_id.get(record_id, "") for record_id in ids)))
        lines.append("")
        lines.append("annotation labels:")
        lines.extend(
            counter_table(
                Counter(label for record_id in ids for label in labels_by_id.get(record_id, []))
            )
        )
        lines.append("")
        lines.append("annotation_count:")
        lines.extend(counter_table(Counter(annotation_count_by_id.get(record_id, 0) for record_id in ids)))
        lines.extend(example_rows(records, subset, title=f"{error_type} representative examples"))
    lines.extend(
        [
            "",
            "### FaithBench Root Causes",
            "- The benchmark is positive-heavy in this converted binary view. A high-recall detector gets strong F1 while balanced accuracy stays weak.",
            "- False positives dominate. Many are faithful paraphrases or summaries with low exact lexical overlap.",
            "- False negatives are rare but semantically important: labels such as `production budget` vs `budget`, or cross-entity conflation, are nearly invisible to token overlap.",
            "- `Benign` annotations appear in false positives, which means binary label mapping and human gray-area categories need a policy-specific calibration.",
        ]
    )
    return lines


def example_rows(
    records: Mapping[str, Mapping[str, Any]],
    errors: Sequence[ErrorRow],
    *,
    title: str,
    limit: int = 5,
) -> list[str]:
    lines = ["", f"#### {title}"]
    for error in errors[:limit]:
        row = records.get(error.record_id)
        if row is None:
            continue
        lines.extend(
            [
                "",
                f"- id: `{error.record_id}`",
                f"  - action: `{error.action:.6f}`",
                f"  - predicted/actual: `{int(error.predicted_hallucinated)}/{int(error.actual_hallucinated)}`",
                f"  - answer: {shorten(row.get('answer', ''))}",
                f"  - context: {shorten(row.get('context', ''))}",
            ]
        )
    return lines


def write_report(
    ragtruth_jsonl: Path,
    ragtruth_errors: Path,
    faithbench_jsonl: Path,
    faithbench_errors: Path,
    faithbench_raw_dir: Path,
    output: Path,
) -> None:
    ragtruth_records = load_jsonl(ragtruth_jsonl)
    ragtruth_error_rows = load_error_csv(ragtruth_errors)
    faithbench_records = load_jsonl(faithbench_jsonl)
    faithbench_error_rows = load_error_csv(faithbench_errors)
    lines: list[str] = [
        "# CE Claim Residual Error Analysis",
        "",
        "This report explains why the current external benchmark strength is `baseline-plus`, not SOTA.",
        "",
        "## Top-Level Diagnosis",
        "",
        "- The CE posterior/action layer is not the primary bottleneck in the current external runs.",
        "- The weak link is the evidence axis: current benchmark mode uses lexical support, not NLI, retrieval, or span localization.",
        "- RAGTruth requires contradiction and unsupported-span detection; FaithBench requires faithful paraphrase tolerance and gray-area label policy.",
        "",
    ]
    lines.extend(ragtruth_section(ragtruth_records, ragtruth_error_rows))
    lines.extend([""])
    lines.extend(faithbench_section(faithbench_records, faithbench_error_rows, faithbench_raw_dir))
    lines.extend(
        [
            "",
            "## Fix Priority",
            "",
            "1. Add an entailment/contradiction axis. This targets RAGTruth FN caused by lexical-near conflicts.",
            "2. Restore claim/span-level scoring for external benchmarks. This separates small hallucinated spans from otherwise faithful responses.",
            "3. Add sentence retrieval/reranking before residual scoring. This reduces faithful paraphrase false positives.",
            "4. Calibrate per dataset and per task type. RAGTruth Summary/Data2txt/QA need different thresholds.",
            "5. Preserve FaithBench gray labels instead of flattening everything to one binary label. `Benign` and `Questionable` need separate policy treatment.",
            "6. Move fast lexical/NLI batch scoring into Rust only after the evidence axis is semantically stronger.",
            "",
            "## Bottom Line",
            "",
            "The current system is internally strong but externally limited by evidence semantics. To close the SOTA gap, improve the claim-evidence mapper before tuning the CE posterior again.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragtruth-jsonl", type=Path, required=True)
    parser.add_argument("--ragtruth-errors", type=Path, required=True)
    parser.add_argument("--faithbench-jsonl", type=Path, required=True)
    parser.add_argument("--faithbench-errors", type=Path, required=True)
    parser.add_argument("--faithbench-raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    write_report(
        args.ragtruth_jsonl,
        args.ragtruth_errors,
        args.faithbench_jsonl,
        args.faithbench_errors,
        args.faithbench_raw_dir,
        args.output,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
