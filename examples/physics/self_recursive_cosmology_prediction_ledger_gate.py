"""Falsifiable prediction ledger for the self-recursive cosmology package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULT_JSON = BASE_DIR / "self_recursive_cosmology_prediction_ledger_results.json"
REPORT_MD = BASE_DIR / "self_recursive_cosmology_prediction_ledger_report.md"


@dataclass(frozen=True)
class Prediction:
    target: str
    layer: str
    expected_result: str
    decisive_check: str
    falsifier: str
    status: str
    next_artifact: str


PREDICTIONS = [
    Prediction(
        target="real BAO+SN covariance",
        layer="H0 q-selector",
        expected_result="global/low or low-side bridge before H0 comparison",
        decisive_check="source/covariance graph gives q_F near the global family with public covariance labels fixed first",
        falsifier="labelled covariance gives stable local/high q_F before any H0 refit",
        status="future data-facing",
        next_artifact="real BAO+SN Fisher JSON bundle",
    ),
    Prediction(
        target="real SH0ES/CCHP ladder covariance",
        layer="H0 q-selector",
        expected_result="local/high or semi-local high depending on population closures",
        decisive_check="calibrator/anchor graph predicts q_F before final scalar H0 is used",
        falsifier="source graph selects global/low despite endpoint-dominated calibration chain",
        status="future data-facing",
        next_artifact="ladder covariance role adapter",
    ),
    Prediction(
        target="primitive spectrum joint likelihood",
        layer="residual cascade",
        expected_result="A_s, running, and tensor remain compatible with one N_e/A3c cascade",
        decisive_check="same Q_GER and N_e family fit scalar amplitude while r and alpha_spec remain within bounds",
        falsifier="running/tensor data require an observable-specific projection not shared by A_s",
        status="future data-facing",
        next_artifact="primitive spectrum common-readout likelihood gate",
    ),
    Prediction(
        target="CMB large-angle map/covariance likelihood",
        layer="residual cascade",
        expected_result="fixed A_H=2Q_GER/sigma improves or remains competitive against null after mask/trials handling",
        decisive_check="map likelihood tests fixed amplitude without fitting preferred-axis strength",
        falsifier="fixed A_H performs worse than null or requires amplitude refit beyond uncertainty",
        status="future data-facing",
        next_artifact="CMB large-angle fixed-amplitude likelihood gate",
    ),
    Prediction(
        target="FLRW/reheating/horizon scale lift",
        layer="d0 measure transport",
        expected_result="dimensionless S_R transport maps to scale quantities only with a derived physical scale",
        decisive_check="scale map derives curvature/reheating/horizon factor without importing observed H0 as the answer",
        falsifier="scale lift needs an unconstrained calibration per target quantity",
        status="theory-facing",
        next_artifact="FLRW scale-lift derivation gate",
    ),
    Prediction(
        target="late horizon readout dynamics",
        layer="early-late bridge",
        expected_result="late horizon entropy reads channel-corrected primordial phase measure",
        decisive_check="dynamical argument reproduces I_late = I_phase without changing pi^2/2 or q definitions",
        falsifier="dynamics selects local slow-roll entropy growth instead of boundary phase measure",
        status="theory-facing",
        next_artifact="late horizon phase-readout dynamics gate",
    ),
    Prediction(
        target="core kernel deformation",
        layer="kernel guardrail",
        expected_result="no c/kappa deformation is promoted unless fixed before data contact",
        decisive_check="new kernel term comes from independent derivation and improves multiple observables under AIC",
        falsifier="best result relies on tuning c or kappa to one observable and degrades shared readouts",
        status="guardrail",
        next_artifact="kernel no-free-parameter derivation gate",
    ),
]


def main() -> int:
    layers = {row.layer for row in PREDICTIONS}
    statuses = {row.status for row in PREDICTIONS}
    missing_falsifier = [row.target for row in PREDICTIONS if not row.falsifier.strip()]
    missing_next = [row.target for row in PREDICTIONS if not row.next_artifact.strip()]
    passed = (
        len(PREDICTIONS) >= 7
        and {"H0 q-selector", "residual cascade", "d0 measure transport", "early-late bridge", "kernel guardrail"}
        <= layers
        and {"future data-facing", "theory-facing", "guardrail"} <= statuses
        and not missing_falsifier
        and not missing_next
    )
    payload = {
        "gate": "self_recursive_cosmology_prediction_ledger",
        "passed": passed,
        "prediction_count": len(PREDICTIONS),
        "layers": sorted(layers),
        "statuses": sorted(statuses),
        "rows": [asdict(row) for row in PREDICTIONS],
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Self-recursive cosmology prediction ledger",
        "",
        f"- passed: `{passed}`",
        f"- predictions: {len(PREDICTIONS)}",
        "",
        "| target | layer | expected result | decisive check | falsifier | status | next artifact |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in PREDICTIONS:
        lines.append(
            f"| {row.target} | {row.layer} | {row.expected_result} | {row.decisive_check} | "
            f"{row.falsifier} | {row.status} | {row.next_artifact} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"passed": passed, "prediction_count": len(PREDICTIONS)}, indent=2))
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
