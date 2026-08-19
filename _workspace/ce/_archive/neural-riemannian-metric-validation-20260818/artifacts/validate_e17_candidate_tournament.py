"""Validate the frozen E17 candidate-tournament result without SciPy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_TUPLE_COUNTS = {
    "condition_decoder": 44,
    "condition_field": 352,
    "deformation": 2112,
    "directional": 88,
    "distribution": 22,
    "graph": 264,
    "uncertainty": 3024,
}
HORIZONS = ("1", "5", "15", "30")
UNCERTAINTY_CANDIDATES = (
    "S0",
    "S1",
    "S2",
    "S3",
    "S4-H",
    "S5",
    "S12",
    "S13",
    "BASE_FULL",
    "BASE_DIAGONAL",
    "BASE_ISOTROPIC",
    "BASE_PERSISTENCE",
)
DEFORMATION_CANDIDATES = ("S6-H", "S7-H", "S14", "S15")
GRAPH_CANDIDATES = ("G1", "G2", "G3a", "G3b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonstandard JSON numeric constant: {token}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def require_number(value: Any, detail: str) -> float:
    require(finite_number(value), detail)
    return float(value)


def require_close(observed: Any, expected: Any, detail: str) -> None:
    left = require_number(observed, f"{detail}: observed value is not finite")
    right = require_number(expected, f"{detail}: expected value is not finite")
    tolerance = 1e-12 + 1e-10 * max(abs(left), abs(right))
    require(abs(left - right) <= tolerance, f"{detail}: {left} != {right}")


def mean(values: list[float]) -> float:
    require(bool(values), "cannot average an empty list")
    return sum(values) / len(values)


def animal_mean(
    session_values: dict[str, float],
    session_to_animal: dict[str, str],
    animals: set[str],
) -> tuple[float, dict[str, float]]:
    grouped = {
        animal: [
            value
            for session, value in session_values.items()
            if session_to_animal[session] == animal
        ]
        for animal in sorted(animals)
    }
    for animal, values in grouped.items():
        expected = sum(item == animal for item in session_to_animal.values())
        require(len(values) == expected, f"animal aggregation is incomplete for {animal}")
    animal_values = {animal: mean(values) for animal, values in grouped.items()}
    return mean(list(animal_values.values())), animal_values


def require_animal_map(
    observed: dict[str, Any], expected: dict[str, float], detail: str
) -> None:
    require(set(observed) == set(expected), f"{detail}: animal keys differ")
    for animal, value in expected.items():
        require_close(observed[animal], value, f"{detail}/{animal}")


def eligible_intersection(items: list[dict[str, Any]]) -> set[str]:
    common: set[str] | None = None
    for tuples in items:
        eligible = {
            key for key, value in tuples.items() if value.get("status") == "ELIGIBLE"
        }
        common = eligible if common is None else common & eligible
    return common or set()


def validate_uncertainty(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["uncertainty"]
    raw = section["raw_outer_train_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, conditions in raw.items():
        for condition, horizons in conditions.items():
            for horizon, candidates in horizons.items():
                for candidate, tuples in candidates.items():
                    for key, item in tuples.items():
                        if item.get("status") != "ELIGIBLE":
                            continue
                        require_number(item.get("inner_nlpd"), f"uncertainty inner NLPD {session}/{condition}/{horizon}/{candidate}/{key}")
                        require_number(item.get("scale"), f"uncertainty scale {session}/{condition}/{horizon}/{candidate}/{key}")
                        checked += 1

    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [
            session
            for session, animal in session_to_animal.items()
            if animal in train_animals
        ]
        heldout_sessions = [
            session for session, animal in session_to_animal.items() if animal == heldout
        ]
        for horizon in HORIZONS:
            for candidate in UNCERTAINTY_CANDIDATES:
                cells = [
                    raw[session][condition][horizon][candidate]
                    for session in train_sessions
                    for condition in raw[session]
                ]
                common = eligible_intersection(cells)
                choices: list[tuple[float, int, float, str, dict[str, float]]] = []
                for key in sorted(common):
                    session_values: dict[str, float] = {}
                    parameter_count = 0
                    ridge = 0.0
                    for session in train_sessions:
                        scores = []
                        for condition in raw[session]:
                            item = raw[session][condition][horizon][candidate][key]
                            scores.append(require_number(item["inner_nlpd"], "selected uncertainty inner NLPD is nonfinite"))
                            parameter_count = int(item.get("parameter_count", 0))
                            ridge = float(item.get("parameters", {}).get("lambda_c", 0.0))
                        session_values[session] = mean(scores)
                    aggregate, animal_values = animal_mean(
                        session_values, session_to_animal, train_animals
                    )
                    choices.append((aggregate, parameter_count, ridge, key, animal_values))
                observed = folds[heldout][horizon][candidate]
                if not choices:
                    require(observed["status"] == "NO_ELIGIBLE_TUPLE", f"unexpected uncertainty selection {heldout}/{horizon}/{candidate}")
                    continue
                choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                best = choices[0]
                require(observed["status"] == "EVALUATED", f"uncertainty outer evaluation incomplete {heldout}/{horizon}/{candidate}")
                require(observed["tuple_key"] == best[3], f"uncertainty selected tuple mismatch {heldout}/{horizon}/{candidate}")
                require(observed["parameters"] == json.loads(best[3]), f"uncertainty selected parameters mismatch {heldout}/{horizon}/{candidate}")
                require(observed["eligible_tuple_count"] == len(choices), f"uncertainty eligible tuple count mismatch {heldout}/{horizon}/{candidate}")
                require_close(observed["outer_train_inner_nlpd"], best[0], f"uncertainty selection score {heldout}/{horizon}/{candidate}")
                require_animal_map(observed["outer_train_animal_means"], best[4], f"uncertainty train animal means {heldout}/{horizon}/{candidate}")
                require(set(observed["session_scores"]) == set(heldout_sessions), f"uncertainty heldout sessions mismatch {heldout}/{horizon}/{candidate}")
                nlpds = [require_number(item["nlpd"], "heldout uncertainty NLPD is nonfinite") for item in observed["session_scores"].values()]
                energies = [require_number(item["energy_score"], "heldout uncertainty energy is nonfinite") for item in observed["session_scores"].values()]
                require_close(observed["heldout_animal_nlpd"], mean(nlpds), f"uncertainty heldout NLPD aggregation {heldout}/{horizon}/{candidate}")
                require_close(observed["heldout_animal_energy_score"], mean(energies), f"uncertainty heldout energy aggregation {heldout}/{horizon}/{candidate}")

    for horizon in HORIZONS:
        rebuilt = []
        for candidate in UNCERTAINTY_CANDIDATES:
            outcomes = [folds[animal][horizon][candidate] for animal in sorted(animals)]
            if not all(item.get("status") == "EVALUATED" for item in outcomes):
                continue
            animal_values = {
                animal: require_number(folds[animal][horizon][candidate]["heldout_animal_nlpd"], "scoreboard source NLPD is nonfinite")
                for animal in sorted(animals)
            }
            rebuilt.append((mean(list(animal_values.values())), candidate, animal_values))
        rebuilt.sort(key=lambda item: (item[0], item[1]))
        observed_ranking = section["scoreboard"][horizon]["ranking"]
        require([item["candidate"] for item in observed_ranking] == [item[1] for item in rebuilt], f"scoreboard order mismatch at H={horizon}")
        for observed, expected in zip(observed_ranking, rebuilt, strict=True):
            require_close(observed["mean_animal_nlpd"], expected[0], f"scoreboard mean at H={horizon}/{expected[1]}")
            require_animal_map(observed["animal_nlpd"], expected[2], f"scoreboard animal values at H={horizon}/{expected[1]}")
    return checked


def validate_deformation(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["deformation"]
    raw = section["raw_outer_train_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, conditions in raw.items():
        for condition, horizons in conditions.items():
            for horizon, candidates in horizons.items():
                for candidate, tuples in candidates.items():
                    for key, item in tuples.items():
                        if item.get("status") != "ELIGIBLE":
                            continue
                        require_number(item.get("inner_nrmse"), f"deformation inner NRMSE {session}/{condition}/{horizon}/{candidate}/{key}")
                        require_number(item.get("scale"), f"deformation scale {session}/{condition}/{horizon}/{candidate}/{key}")
                        checked += 1
    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [session for session, animal in session_to_animal.items() if animal in train_animals]
        heldout_sessions = [session for session, animal in session_to_animal.items() if animal == heldout]
        for horizon in HORIZONS:
            for candidate in DEFORMATION_CANDIDATES:
                cells = [raw[session][condition][horizon][candidate] for session in train_sessions for condition in raw[session]]
                common = eligible_intersection(cells)
                choices: list[tuple[float, float, str, dict[str, float]]] = []
                for key in sorted(common):
                    session_values = {
                        session: mean([
                            require_number(raw[session][condition][horizon][candidate][key]["inner_nrmse"], "selected deformation NRMSE is nonfinite")
                            for condition in raw[session]
                        ])
                        for session in train_sessions
                    }
                    aggregate, animal_values = animal_mean(session_values, session_to_animal, train_animals)
                    ridge = float(json.loads(key).get("lambda_g", 0.0))
                    choices.append((aggregate, ridge, key, animal_values))
                observed = folds[heldout][horizon][candidate]
                if not choices:
                    require(observed["status"] == "NO_ELIGIBLE_TUPLE", f"unexpected deformation selection {heldout}/{horizon}/{candidate}")
                    continue
                choices.sort(key=lambda item: (item[0], 0, item[1], item[2]))
                best = choices[0]
                require(observed["status"] == "EVALUATED", f"deformation outer evaluation incomplete {heldout}/{horizon}/{candidate}")
                require(observed["tuple_key"] == best[2], f"deformation selected tuple mismatch {heldout}/{horizon}/{candidate}")
                require(observed["parameters"] == json.loads(best[2]), f"deformation parameters mismatch {heldout}/{horizon}/{candidate}")
                require(observed["eligible_tuple_count"] == len(choices), f"deformation eligible tuple count mismatch {heldout}/{horizon}/{candidate}")
                require_close(observed["outer_train_inner_nrmse"], best[0], f"deformation selection score {heldout}/{horizon}/{candidate}")
                require_animal_map(observed["outer_train_animal_means"], best[3], f"deformation train animal means {heldout}/{horizon}/{candidate}")
                require(set(observed["session_scores"]) == set(heldout_sessions), f"deformation heldout sessions mismatch {heldout}/{horizon}/{candidate}")
                scores = [require_number(item["nrmse"], "heldout deformation NRMSE is nonfinite") for item in observed["session_scores"].values()]
                require_close(observed["heldout_animal_nrmse"], mean(scores), f"deformation heldout aggregation {heldout}/{horizon}/{candidate}")
    return checked


def validate_condition_information(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["condition_information"]
    raw = section["raw_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, tuples in raw.items():
        for key, item in tuples.items():
            if item.get("status") == "ELIGIBLE":
                require_number(item.get("inner_balanced_log_loss"), f"decoder inner log loss {session}/{key}")
                checked += 1
            for candidate in ("S8", "S9"):
                for field_key, field in item["fit_field_gates"][candidate].items():
                    if field.get("status") == "ELIGIBLE":
                        require_number(field.get("minimum_eigenvalue"), f"{candidate} minimum eigenvalue {session}/{key}/{field_key}")
                        if candidate == "S8":
                            require_number(field.get("mean_logdet"), f"S8 mean logdet {session}/{key}/{field_key}")
    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [session for session, animal in session_to_animal.items() if animal in train_animals]
        heldout_sessions = [session for session, animal in session_to_animal.items() if animal == heldout]
        common = eligible_intersection([raw[session] for session in train_sessions])
        choices = []
        for key in sorted(common):
            session_values = {
                session: require_number(raw[session][key]["inner_balanced_log_loss"], "selected decoder inner log loss is nonfinite")
                for session in train_sessions
            }
            aggregate, animal_values = animal_mean(session_values, session_to_animal, train_animals)
            choices.append((aggregate, key, animal_values))
        require(bool(choices), f"no condition decoder tuple for {heldout}")
        choices.sort(key=lambda item: (item[0], item[1]))
        best = choices[0]
        observed = folds[heldout]
        require(observed["status"] == "EVALUATED", f"condition decoder outer evaluation incomplete {heldout}")
        require(observed["selected_decoder_tuple"] == json.loads(best[1]), f"decoder selected tuple mismatch {heldout}")
        require_close(observed["outer_train_inner_balanced_log_loss"], best[0], f"decoder selection score {heldout}")
        require_animal_map(observed["outer_train_animal_means"], best[2], f"decoder train animal means {heldout}")
        require(set(observed["session_scores"]) == set(heldout_sessions), f"decoder heldout sessions mismatch {heldout}")
        scores = [require_number(value, "heldout decoder log loss is nonfinite") for value in observed["session_scores"].values()]
        require_close(observed["heldout_animal_balanced_log_loss"], mean(scores), f"decoder heldout aggregation {heldout}")
        for session, fields in observed["field_gates"].items():
            require(fields == raw[session][best[1]]["fit_field_gates"], f"heldout field gate differs from fit-only cache {heldout}/{session}")
    return checked


def validate_graph(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["graph"]
    raw = section["raw_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, conditions in raw.items():
        for condition, candidates in conditions.items():
            for candidate, tuples in candidates.items():
                for key, item in tuples.items():
                    if item.get("status") == "ELIGIBLE":
                        require_number(item.get("inner_spearman_rho"), f"graph inner rho {session}/{condition}/{candidate}/{key}")
                        checked += 1
    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [session for session, animal in session_to_animal.items() if animal in train_animals]
        for candidate in GRAPH_CANDIDATES:
            cells = [raw[session][condition][candidate] for session in train_sessions for condition in raw[session]]
            common = eligible_intersection(cells)
            require(not common, f"graph {candidate} unexpectedly has a common tuple for {heldout}")
            require(folds[heldout][candidate]["status"] == "NO_ELIGIBLE_TUPLE", f"graph no-eligible status mismatch {heldout}/{candidate}")
    return checked


def validate_directional(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["directional_action"]
    raw = section["raw_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, conditions in raw.items():
        for condition, tuples in conditions.items():
            for key, item in tuples.items():
                if item.get("status") == "ELIGIBLE":
                    require_number(item.get("inner_forward_nlpd"), f"D1 inner NLPD {session}/{condition}/{key}")
                    checked += 1
    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [session for session, animal in session_to_animal.items() if animal in train_animals]
        heldout_sessions = [session for session, animal in session_to_animal.items() if animal == heldout]
        cells = [raw[session][condition] for session in train_sessions for condition in raw[session]]
        common = eligible_intersection(cells)
        choices = []
        for key in sorted(common):
            session_values = {
                session: mean([
                    require_number(raw[session][condition][key]["inner_forward_nlpd"], "selected D1 inner NLPD is nonfinite")
                    for condition in raw[session]
                ])
                for session in train_sessions
            }
            aggregate, animal_values = animal_mean(session_values, session_to_animal, train_animals)
            ridge = float(json.loads(key)["lambda_c"])
            choices.append((aggregate, ridge, key, animal_values))
        require(bool(choices), f"no D1 tuple for {heldout}")
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        best = choices[0]
        observed = folds[heldout]
        require(observed["status"] == "EVALUATED", f"D1 outer evaluation incomplete {heldout}")
        require(observed["selected_tuple"] == json.loads(best[2]), f"D1 selected tuple mismatch {heldout}")
        require_close(observed["outer_train_inner_forward_nlpd"], best[0], f"D1 selection score {heldout}")
        require_animal_map(observed["outer_train_animal_means"], best[3], f"D1 train animal means {heldout}")
        require(set(observed["session_scores"]) == set(heldout_sessions), f"D1 heldout sessions mismatch {heldout}")
        for item in observed["session_scores"].values():
            for field in ("forward_nlpd", "reverse_nlpd", "shuffle_nlpd", "reverse_minus_forward", "shuffle_minus_forward"):
                require_number(item.get(field), f"D1 heldout {field} is nonfinite")
        reverse = mean([float(item["reverse_minus_forward"]) for item in observed["session_scores"].values()])
        shuffled = mean([float(item["shuffle_minus_forward"]) for item in observed["session_scores"].values()])
        require_close(observed["heldout_animal_reverse_minus_forward"], reverse, f"D1 reverse aggregation {heldout}")
        require_close(observed["heldout_animal_shuffle_minus_forward"], shuffled, f"D1 shuffle aggregation {heldout}")
    return checked


def validate_distribution(
    result: dict[str, Any], session_to_animal: dict[str, str]
) -> int:
    section = result["distribution"]
    raw = section["raw_inner"]
    folds = section["folds"]
    animals = set(folds)
    checked = 0
    for session, tuples in raw.items():
        for key, item in tuples.items():
            if item.get("status") == "ELIGIBLE":
                for field in ("w2", "permutation_mean", "permutation_sd", "permutation_z", "one_sided_p"):
                    require_number(item.get(field), f"W2 inner {field} {session}/{key}")
                checked += 1
    for heldout in sorted(animals):
        train_animals = animals - {heldout}
        train_sessions = [session for session, animal in session_to_animal.items() if animal in train_animals]
        heldout_sessions = [session for session, animal in session_to_animal.items() if animal == heldout]
        common = eligible_intersection([raw[session] for session in train_sessions])
        choices = []
        for key in sorted(common):
            session_values = {
                session: -require_number(raw[session][key]["permutation_z"], "selected W2 permutation z is nonfinite")
                for session in train_sessions
            }
            aggregate, animal_values = animal_mean(session_values, session_to_animal, train_animals)
            choices.append((aggregate, key, animal_values))
        require(bool(choices), f"no W2 tuple for {heldout}")
        choices.sort(key=lambda item: (item[0], item[1]))
        best = choices[0]
        observed = folds[heldout]
        require(observed["status"] == "EVALUATED", f"W2 outer evaluation incomplete {heldout}")
        require(observed["selected_tuple"] == json.loads(best[1]), f"W2 selected tuple mismatch {heldout}")
        require_close(observed["outer_train_mean_negative_permutation_z"], best[0], f"W2 selection score {heldout}")
        require_animal_map(observed["outer_train_animal_means"], best[2], f"W2 train animal means {heldout}")
        require(set(observed["session_scores"]) == set(heldout_sessions), f"W2 heldout sessions mismatch {heldout}")
        for item in observed["session_scores"].values():
            for field in ("w2", "permutation_mean", "permutation_sd", "permutation_z", "one_sided_p"):
                require_number(item.get(field), f"W2 heldout {field} is nonfinite")
        w2 = mean([float(item["w2"]) for item in observed["session_scores"].values()])
        z = mean([float(item["permutation_z"]) for item in observed["session_scores"].values()])
        require_close(observed["heldout_animal_mean_w2"], w2, f"W2 heldout aggregation {heldout}")
        require_close(observed["heldout_animal_mean_permutation_z"], z, f"W2 z aggregation {heldout}")
    return checked


def main() -> int:
    artifact_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=artifact_root / "e17-candidate-tournament-result-lock-v2.2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_root / "e17-candidate-tournament-validation-v2.2.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace validation output: {args.output}")

    lock = load_json(args.lock)
    result_path = artifact_root / lock["result"]
    freeze_path = artifact_root / lock["freeze"]
    freeze = load_json(freeze_path)
    result = load_json(result_path)

    require(sha256_file(result_path) == lock["result_sha256"], "result hash mismatch")
    require(sha256_file(freeze_path) == lock["freeze_sha256"], "freeze hash mismatch")
    require(freeze["schema_version"] == "2.0.2", "unexpected freeze schema")
    require(freeze["status"] == "FROZEN_BEFORE_V2_2_OUTPUT", "unexpected freeze status")
    require(freeze["candidate_id_count"] == 27, "freeze candidate count changed")
    require(freeze["output"] == result_path.name, "freeze output name mismatch")
    require(freeze["math_fixture_status"] == "PASS", "fixture was not frozen PASS")
    require(sha256_file(artifact_root / "e17_candidate_tournament.py") == freeze["runner_sha256"], "actual runner differs from freeze")
    require(sha256_file(Path(__file__).resolve()) == freeze["validator_sha256"], "actual validator differs from freeze")
    require(sha256_file(artifact_root / "candidate-equation-registry.md") == freeze["registry_markdown_sha256"], "actual registry Markdown differs from freeze")
    require(sha256_file(artifact_root / "candidate-equation-registry.json") == freeze["registry_json_sha256"], "actual registry JSON differs from freeze")
    require(sha256_file(artifact_root / "math" / "candidate_math_fixture.py") == freeze["math_fixture_sha256"], "actual math fixture differs from freeze")
    require(sha256_file(artifact_root / "math" / "candidate_math_fixture_output_v2.2.json") == freeze["math_fixture_output_sha256"], "actual fixture output differs from freeze")
    require(result["code_sha256"] == lock["runner_sha256"], "runner hash mismatch")
    require(result["code_sha256"] == freeze["runner_sha256"], "freeze runner mismatch")
    require(result["schema_version"] == "2.0.2", "unexpected result schema")
    require(result["analysis_status"] == "RETROSPECTIVE_DISCOVERY_ONLY", "analysis class changed")
    require(result["registry_markdown_sha256"] == freeze["registry_markdown_sha256"], "registry Markdown mismatch")
    require(result["registry_json_sha256"] == freeze["registry_json_sha256"], "registry JSON mismatch")

    candidate_ids = result["candidate_ids"]
    require(len(candidate_ids) == 27, "candidate count is not 27")
    require(len(set(candidate_ids)) == 27, "candidate IDs are not unique")
    manifest = result["data_manifest"]
    require(manifest["session_count"] == 11, "session count is not 11")
    require(manifest["animal_count"] == 3, "animal count is not 3")
    observed_inputs = {
        f"Figure2/Data/{Path(item['source_path']).name}": item["source_sha256"]
        for item in manifest["sessions"]
    }
    session_to_animal = {
        item["session_id"]: item["animal"] for item in manifest["sessions"]
    }
    require(observed_inputs == freeze["input_files_sha256"], "result input hashes differ from freeze")
    require(result["freeze_validation"]["status"] == "PASS_BYTE_PINNED_INPUTS_AND_CODE", "freeze validation did not pass")
    require(result["freeze_validation"]["verified_input_count"] == 11, "not all input bytes were verified")

    execution = result["tuple_execution_audit"]
    require(execution["status"] == "PASS_EXACT_EXPECTED_TUPLE_KEYS", "tuple audit did not pass")
    require(execution["all_cells_complete"] is True, "tuple completion is not a JSON boolean true")
    require(execution["failure_count"] == 0, "tuple audit reports failures")
    observed_tuple_total = 0
    for family, expected in EXPECTED_TUPLE_COUNTS.items():
        item = execution["families"][family]
        require(item["expected_tuple_count"] == expected, f"{family} expected tuple count changed")
        require(item["observed_tuple_count"] == expected, f"{family} observed tuple count changed")
        require(item["missing_tuple_count"] == 0, f"{family} has missing tuples")
        require(item["extra_tuple_count"] == 0, f"{family} has extra tuples")
        observed_tuple_total += item["observed_tuple_count"]
    require(observed_tuple_total == 5906, "raw tuple total is not 5906")

    coverage = result["candidate_coverage"]
    require(set(coverage) == set(candidate_ids), "coverage does not match candidate IDs")
    require(coverage["S7-H"]["runtime_status_counts"] == {"ELIGIBLE": 264, "INELIGIBLE_TAUTOLOGY": 88}, "S7-H eligibility counts changed")
    for candidate in ("S8", "S9"):
        require(coverage[candidate]["runtime_status_counts"] == {"ELIGIBLE": 132, "INELIGIBLE_SINGULAR": 44}, f"{candidate} rank-gate counts changed")
        require(coverage[candidate]["strict_tournament_status"] == "FIELD_GATE_ONLY_NO_INDEPENDENT_METRIC_ENDPOINT", f"{candidate} was promoted beyond a field gate")
    for candidate in ("G1", "G2", "G3a", "G3b"):
        require(coverage[candidate]["outer_folds_evaluated"] == 0, f"{candidate} unexpectedly has an outer evaluation")
        require(coverage[candidate]["strict_tournament_status"] == "UNTESTABLE_UNDER_FROZEN_LOAO_INTERSECTION", f"{candidate} graph status changed")
    for candidate in ("D1", "P1/P2"):
        require(coverage[candidate]["outer_folds_evaluated"] == 3, f"{candidate} did not evaluate all outer folds")
    for candidate, item in coverage.items():
        require(type(item["raw_tuple_attempted"]) is bool, f"{candidate} attempted flag is not boolean")

    for fold in result["deformation"]["folds"].values():
        require(fold["1"]["S7-H"]["status"] == "NO_ELIGIBLE_TUPLE", "S7-H H=1 reached an outer test")
    for session in result["condition_information"]["raw_inner"].values():
        for decoder in session.values():
            for candidate in ("S8", "S9"):
                for field in decoder["fit_field_gates"][candidate].values():
                    require(field["source_block"] == "session_fit_only", "condition field used a non-fit source")

    numeric_checks = {
        "uncertainty_eligible_tuples": validate_uncertainty(result, session_to_animal),
        "deformation_eligible_tuples": validate_deformation(result, session_to_animal),
        "condition_decoder_eligible_tuples": validate_condition_information(result, session_to_animal),
        "graph_eligible_tuples": validate_graph(result, session_to_animal),
        "directional_eligible_tuples": validate_directional(result, session_to_animal),
        "distribution_eligible_tuples": validate_distribution(result, session_to_animal),
    }

    claims = result["claim_status"]
    require(claims["population_winner"] == "PROHIBITED", "population winner was promoted")
    require(claims["locked_validation"] == "NOT_RUN_REQUIRES_NEW_COHORT", "locked validation status changed")
    require(result["kill_tests"]["K10"]["status"] == "TRIGGERED_FOR_CONFIRMATION", "opened-data confirmation gate changed")

    output = {
        "schema_version": "2.0.0",
        "status": "PASS",
        "result": result_path.name,
        "result_sha256": lock["result_sha256"],
        "freeze_sha256": lock["freeze_sha256"],
        "candidate_id_count": len(candidate_ids),
        "verified_input_count": len(observed_inputs),
        "verified_raw_tuple_count": observed_tuple_total,
        "numeric_and_selection_checks": numeric_checks,
        "uncertainty_scoreboards_recomputed": len(HORIZONS),
        "s7_h1_tautology_count": coverage["S7-H"]["runtime_status_counts"]["INELIGIBLE_TAUTOLOGY"],
        "s8_zero_ridge_singular_count": coverage["S8"]["runtime_status_counts"]["INELIGIBLE_SINGULAR"],
        "s9_zero_ridge_singular_count": coverage["S9"]["runtime_status_counts"]["INELIGIBLE_SINGULAR"],
        "graph_outer_evaluations": sum(coverage[item]["outer_folds_evaluated"] for item in ("G1", "G2", "G3a", "G3b")),
        "population_winner": claims["population_winner"],
        "locked_validation": claims["locked_validation"],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
