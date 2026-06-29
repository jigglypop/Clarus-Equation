"""Run all local CE Claim Residual Verifier checks.

This runner separates internal synthetic strength from real benchmark strength.
Synthetic modes prove that known failure modes are closed; JSONL benchmark files
are required before making any external SOTA comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from clarus.llm_pre_eq import ClaimResidualVerifierConfig  # noqa: E402

from examples.pre_eq.claim_residual_benchmark import (  # noqa: E402
    calibrate_thresholds,
    raw_predictions,
)
from examples.pre_eq.claim_residual_verifier_sweep import (  # noqa: E402
    score_config,
    synthetic_cases,
)


SYNTHETIC_MODES = ("adversarial", "noisy", "partial", "source", "graph", "missing")


@dataclass(frozen=True)
class SyntheticRun:
    mode: str
    exact_accuracy: float
    answer_rate: float
    hallucination_rate_on_answered: float
    baseline_accuracy: float


@dataclass(frozen=True)
class JsonlRun:
    path: str
    total: int
    best_action_threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float


@dataclass(frozen=True)
class RunAllSummary:
    synthetic: tuple[SyntheticRun, ...]
    jsonl: tuple[JsonlRun, ...]
    internal_strength: str
    external_strength: str
    conclusion: str


def strength_label(*, exact_accuracy: float, hallucination_rate: float) -> str:
    if exact_accuracy >= 0.95 and hallucination_rate <= 0.05:
        return "strong-internal"
    if exact_accuracy >= 0.85 and hallucination_rate <= 0.15:
        return "promising"
    if exact_accuracy >= 0.70:
        return "prototype"
    return "weak"


def external_strength_label(jsonl_runs: tuple[JsonlRun, ...]) -> str:
    if not jsonl_runs:
        return "unmeasured"
    mean_f1 = sum(run.f1 for run in jsonl_runs) / len(jsonl_runs)
    mean_auroc = sum(run.auroc for run in jsonl_runs) / len(jsonl_runs)
    if mean_f1 >= 0.85 and mean_auroc >= 0.90:
        return "sota-competitive"
    if mean_f1 >= 0.75 and mean_auroc >= 0.80:
        return "competitive-prototype"
    if mean_f1 >= 0.60:
        return "baseline-plus"
    return "weak"


def run_synthetic(*, seed: int, cases: int) -> tuple[SyntheticRun, ...]:
    config = ClaimResidualVerifierConfig()
    runs: list[SyntheticRun] = []
    for mode in SYNTHETIC_MODES:
        result = score_config(mode, config, synthetic_cases(mode, seed=seed, n_cases=cases))
        runs.append(
            SyntheticRun(
                mode=mode,
                exact_accuracy=result.exact_accuracy,
                answer_rate=result.answer_rate,
                hallucination_rate_on_answered=result.hallucination_rate_on_answered,
                baseline_accuracy=result.baseline_accuracy,
            )
        )
    return tuple(runs)


def iter_jsonl_files(path: Path | None) -> tuple[Path, ...]:
    if path is None:
        return ()
    if path.is_file():
        return (path,)
    if not path.exists():
        raise FileNotFoundError(path)
    return tuple(sorted(item for item in path.rglob("*.jsonl") if item.is_file()))


def run_jsonl(
    path: Path | None,
    *,
    accepted_fraction_threshold: float = 0.0,
    max_context_chars: int = 2000,
    fast_lexical: bool = False,
    response_level: bool = False,
    enhanced_evidence: bool = False,
    semantic_evidence: bool = False,
    nli_evidence: bool = False,
    nli_scores_jsonl: Path | None = None,
) -> tuple[JsonlRun, ...]:
    runs: list[JsonlRun] = []
    for jsonl_path in iter_jsonl_files(path):
        raw = raw_predictions(
            jsonl_path,
            accepted_fraction_threshold=accepted_fraction_threshold,
            max_context_chars=max_context_chars,
            fast_lexical=fast_lexical,
            response_level=response_level,
            enhanced_evidence=enhanced_evidence,
            semantic_evidence=semantic_evidence,
            nli_evidence=nli_evidence,
            nli_scores_jsonl=nli_scores_jsonl,
        )
        calibration = calibrate_thresholds(
            raw,
            accepted_fraction_threshold=accepted_fraction_threshold,
        )
        metrics = calibration.metrics
        runs.append(
            JsonlRun(
                path=str(jsonl_path),
                total=metrics.total,
                best_action_threshold=calibration.action_threshold,
                accuracy=metrics.accuracy,
                balanced_accuracy=metrics.balanced_accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                auroc=calibration.auroc,
                auprc=calibration.auprc,
            )
        )
    return tuple(runs)


def summarize(synthetic: tuple[SyntheticRun, ...], jsonl: tuple[JsonlRun, ...]) -> RunAllSummary:
    min_exact = min((run.exact_accuracy for run in synthetic), default=0.0)
    max_hallucination = max((run.hallucination_rate_on_answered for run in synthetic), default=1.0)
    internal = strength_label(
        exact_accuracy=min_exact,
        hallucination_rate=max_hallucination,
    )
    external = external_strength_label(jsonl)
    if external == "unmeasured":
        conclusion = (
            "Internal failure modes are measured, but external SOTA strength is unmeasured "
            "until real JSONL benchmarks are supplied."
        )
    else:
        conclusion = f"External benchmark strength is {external}."
    return RunAllSummary(
        synthetic=synthetic,
        jsonl=jsonl,
        internal_strength=internal,
        external_strength=external,
        conclusion=conclusion,
    )


def print_summary(summary: RunAllSummary, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return
    print("# CE Claim Residual run-all")
    print(f"internal_strength {summary.internal_strength}")
    print(f"external_strength {summary.external_strength}")
    print(f"conclusion {summary.conclusion}")
    print()
    print("synthetic_mode,exact_accuracy,answer_rate,hallucination_rate,baseline_accuracy")
    for run in summary.synthetic:
        print(
            f"{run.mode},{run.exact_accuracy:.6f},{run.answer_rate:.6f},"
            f"{run.hallucination_rate_on_answered:.6f},{run.baseline_accuracy:.6f}"
        )
    if summary.jsonl:
        print()
        print("jsonl,total,best_action_threshold,accuracy,balanced_accuracy,precision,recall,f1,auroc,auprc")
        for run in summary.jsonl:
            print(
                f"{run.path},{run.total},{run.best_action_threshold:.6f},"
                f"{run.accuracy:.6f},{run.balanced_accuracy:.6f},"
                f"{run.precision:.6f},{run.recall:.6f},{run.f1:.6f},"
                f"{run.auroc:.6f},{run.auprc:.6f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--jsonl-dir", type=Path)
    parser.add_argument("--accepted-fraction-threshold", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--fast-lexical", action="store_true")
    parser.add_argument("--response-level", action="store_true")
    parser.add_argument("--enhanced-evidence", action="store_true")
    parser.add_argument("--semantic-evidence", action="store_true")
    parser.add_argument("--nli-evidence", action="store_true")
    parser.add_argument("--nli-scores-jsonl", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize(
        run_synthetic(seed=args.seed, cases=args.cases),
        run_jsonl(
            args.jsonl_dir,
            accepted_fraction_threshold=args.accepted_fraction_threshold,
            max_context_chars=args.max_context_chars,
            fast_lexical=args.fast_lexical,
            response_level=args.response_level,
            enhanced_evidence=args.enhanced_evidence,
            semantic_evidence=args.semantic_evidence,
            nli_evidence=args.nli_evidence,
            nli_scores_jsonl=args.nli_scores_jsonl,
        ),
    )
    print_summary(summary, json_output=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
