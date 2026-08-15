"""Independent development and sealed-confirmation evaluator for CE V18b.

The production learner is loaded directly from its registered file.  This
module deliberately does not import ``reality_stone.clarus`` while scoring.
Task generation, labels, rewards, expected updates, query filtering, state
serialization, and gate decisions are evaluator-owned.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


DEVELOPMENT_SEEDS = range(1_821_000, 1_821_064)
CONFIRMATION_SEEDS = range(1_822_000, 1_822_256)
DIMENSION = 8
EPOCHS = 4
ETA = 0.25
TRAINING_DELAYS = (4, 8, 16)
EVALUATION_DELAYS = (0, 128)
QUERY_PAIR_COUNT = 128
QUERY_REDRAW_CAP = 1_024
ENSEMBLE_SIZES = (1, 2, 4, 8, 16, 64)
TOLERANCE = 1.0e-12

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_RELATIVE = PurePosixPath(
    "_workspace/ce/agi-v18b-learned-delayed-credit-20260814"
)
CONTRACT_RELATIVE = RUN_RELATIVE / "00-contract.md"
EVALUATOR_RELATIVE = RUN_RELATIVE / "artifacts/run_v18b_benchmark.py"
MANIFEST_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-manifest.json"
DEVELOPMENT_RESULT_RELATIVE = RUN_RELATIVE / "artifacts/development-results.json"
RECEIPT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-opened.json"
RESULT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-results.json"
PRODUCTION_RELATIVE = PurePosixPath(
    "reality_stone/python/reality_stone/clarus/delayed_linear_credit.py"
)
PUBLIC_EXPORT_RELATIVE = PurePosixPath(
    "reality_stone/python/reality_stone/clarus/__init__.py"
)
REQUIRED_MANIFEST_PATHS = frozenset(
    {
        str(CONTRACT_RELATIVE),
        str(PRODUCTION_RELATIVE),
        str(PUBLIC_EXPORT_RELATIVE),
        str(EVALUATOR_RELATIVE),
        str(DEVELOPMENT_RESULT_RELATIVE),
    }
)

FIXED_PROTOCOL: dict[str, Any] = {
    "dimension": DIMENSION,
    "epochs": EPOCHS,
    "eta": ETA,
    "training_delays": list(TRAINING_DELAYS),
    "evaluation_delays": list(EVALUATION_DELAYS),
    "query_pair_count": QUERY_PAIR_COUNT,
    "query_redraw_cap": QUERY_REDRAW_CAP,
    "ensemble_sizes": list(ENSEMBLE_SIZES),
    "cue_distribution": "independent coordinate Rademacher",
    "query_distribution": "integer Rademacher, zero integer margin rejected",
    "terminal_tie_action": 1,
    "paired_nuisance": (
        "byte-identical delay,distractors,messages,topology,update,ensemble,policy"
    ),
    "namespace_derivation": "sha256('CE-V18b\\0'||seed||namespace||coordinates)",
}


def _isolated_load(path: Path) -> tuple[ModuleType, str]:
    """Load the sealed production file without executing its package initializer."""

    resolved = path.resolve(strict=True)
    loaded_bytes = resolved.read_bytes()
    loaded_digest = hashlib.sha256(loaded_bytes).hexdigest()
    name = f"_ce_v18b_sealed_delayed_linear_credit_{loaded_digest}"
    if name in sys.modules:
        raise RuntimeError("isolated production module was already loaded")
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None:
        raise ImportError("cannot construct isolated production module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute exactly the immutable byte buffer that was hashed.  A source
        # loader would reread the path and reopen a load-time TOCTOU gap.
        code = compile(loaded_bytes, str(resolved), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.__ce_loaded_sha256__ = loaded_digest
    return module, loaded_digest


PRODUCTION_MODULE, LOADED_PRODUCTION_SHA256 = _isolated_load(
    REPO_ROOT.joinpath(*PRODUCTION_RELATIVE.parts)
)
IMPORTED_PRODUCTION_PATH = Path(str(PRODUCTION_MODULE.__file__)).resolve(strict=True)


@dataclass(frozen=True)
class _ConfirmationAccess:
    """Identity-bound, single-use in-process confirmation capability."""

    root: Path
    receipt: Path
    result: Path
    manifest: Path
    manifest_sha256: str
    token: object


_ACTIVE_CONFIRMATION_ACCESS: _ConfirmationAccess | None = None
_ACTIVE_CONFIRMATION_PREFLIGHT: object | None = None


def _json_without_duplicate_keys(text: str, *, source: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_object)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_root(root: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    if resolved_root != REPO_ROOT:
        raise ValueError(f"root must be evaluator repository root: {REPO_ROOT}")
    evaluator = resolved_root.joinpath(*EVALUATOR_RELATIVE.parts).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != evaluator:
        raise ValueError("executed evaluator is not the canonical repository evaluator")
    production = resolved_root.joinpath(*PRODUCTION_RELATIVE.parts).resolve(strict=True)
    if IMPORTED_PRODUCTION_PATH != production:
        raise ValueError("isolated production module is not the canonical sealed file")
    return resolved_root


def _canonical_root_path(root: Path, relative: PurePosixPath) -> Path:
    resolved_root = _validated_root(root)
    candidate = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):  # pragma: no cover - fixed paths
        raise ValueError(f"path escapes repository root: {relative}")
    return candidate


def _manifest_target(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    return _canonical_root_path(root, parsed)


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    canonical = _canonical_root_path(root, MANIFEST_RELATIVE)
    if manifest_path.resolve(strict=True) != canonical:
        raise ValueError(f"manifest must be {MANIFEST_RELATIVE}")
    manifest = _json_without_duplicate_keys(
        canonical.read_text(encoding="utf-8"), source=str(MANIFEST_RELATIVE)
    )
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest must be a nonempty path-to-SHA256 object")
    for relative in manifest:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError(f"invalid manifest path: {relative!r}")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
            raise ValueError(f"invalid manifest path: {relative!r}")
    actual = set(manifest)
    if actual != REQUIRED_MANIFEST_PATHS:
        missing = sorted(REQUIRED_MANIFEST_PATHS.difference(actual))
        extra = sorted(actual.difference(REQUIRED_MANIFEST_PATHS))
        raise ValueError(f"manifest paths must be exact; missing={missing}, extra={extra}")
    for relative, expected in manifest.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid SHA-256 for manifest path: {relative}")
        if sha256(_manifest_target(root, relative)) != expected:
            raise ValueError(f"manifest mismatch: {relative}")
    verified = {str(key): str(value) for key, value in sorted(manifest.items())}
    if verified[str(PRODUCTION_RELATIVE)] != LOADED_PRODUCTION_SHA256:
        raise ValueError("loaded production bytes do not match the sealed manifest")
    return verified


def _normalised_protocol(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def verify_development_provenance(root: Path) -> dict[str, Any]:
    path = _canonical_root_path(root, DEVELOPMENT_RESULT_RELATIVE)
    result = _json_without_duplicate_keys(
        path.read_text(encoding="utf-8"), source=str(DEVELOPMENT_RESULT_RELATIVE)
    )
    try:
        if result["mode"] != "development":
            raise ValueError("development result has the wrong mode")
        if result["seed_start"] != DEVELOPMENT_SEEDS.start:
            raise ValueError("development result has the wrong first seed")
        if result["seed_stop_exclusive"] != DEVELOPMENT_SEEDS.stop:
            raise ValueError("development result has the wrong exclusive stop")
        if result["protocol"] != _normalised_protocol(FIXED_PROTOCOL):
            raise ValueError("development result does not match the fixed protocol")
        summaries = result["per_seed"]
        if not isinstance(summaries, list) or len(summaries) != len(DEVELOPMENT_SEEDS):
            raise ValueError("development result has the wrong per-seed count")
        if [item["seed"] for item in summaries] != list(DEVELOPMENT_SEEDS):
            raise ValueError("development result has the wrong seed sequence")
        recomputed_gates = _gate_decisions(summaries)
        recomputed_summary = aggregate(summaries)
        if result.get("gates") != recomputed_gates:
            raise ValueError("development gates do not independently rescore")
        if result.get("summary") != recomputed_summary:
            raise ValueError("development aggregate does not independently rescore")
        if not all(recomputed_gates.values()):
            raise ValueError("development result does not pass every registered gate")
        if result.get("strict_metric_no_go_pass") is not True:
            raise ValueError("development strict verdict is not true")
        if result.get("reward_decoded_eligibility_pass") is not True:
            raise ValueError("development positive verdict is not true")
        if recomputed_summary.get("finite_seed_count") != len(DEVELOPMENT_SEEDS):
            raise ValueError("development result is not finite for all 64 seeds")
    except (KeyError, TypeError) as error:
        raise ValueError("invalid development result schema") from error
    return result


def _loaded_repository_module_violations(
    root: Path,
    *,
    modules: Iterable[tuple[str, object]] | None = None,
) -> list[str]:
    """Return loaded repo modules whose files are not in the sealed path set."""

    resolved_root = root.resolve(strict=True)
    source = tuple(sys.modules.items()) if modules is None else tuple(modules)
    violations: list[str] = []
    for name, module in source:
        raw = getattr(module, "__file__", None)
        if raw is None:
            continue
        try:
            candidate = Path(str(raw)).resolve(strict=False)
            relative = candidate.relative_to(resolved_root).as_posix()
        except (OSError, RuntimeError, ValueError):
            continue
        if not candidate.is_file():
            violations.append(f"{name}:{relative}:missing")
        elif relative not in REQUIRED_MANIFEST_PATHS:
            violations.append(f"{name}:{relative}")
    return sorted(set(violations))


def assert_loaded_repository_module_closure(root: Path) -> None:
    violations = _loaded_repository_module_violations(root)
    if violations:
        raise RuntimeError(
            "unsealed loaded repository module(s): " + ", ".join(violations)
        )


def _result_paths(root: Path) -> tuple[Path, Path]:
    resolved = _validated_root(root)
    receipt = resolved.joinpath(*RECEIPT_RELATIVE.parts)
    result = resolved.joinpath(*RESULT_RELATIVE.parts)
    if not receipt.parent.resolve(strict=True).is_relative_to(resolved):
        raise ValueError("confirmation output directory escapes repository root")
    return receipt, result


def _authorise_seed(seed: int, access: _ConfirmationAccess | None) -> None:
    if seed in DEVELOPMENT_SEEDS:
        if access is not None:
            raise RuntimeError("development seeds do not accept confirmation access")
        return
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("seed is outside the registered seed blocks")
    if type(access) is not _ConfirmationAccess or access is not _ACTIVE_CONFIRMATION_ACCESS:
        raise RuntimeError("confirmation seed access requires an opening receipt")
    assert access is not None
    assert_loaded_repository_module_closure(access.root)
    receipt, result = _result_paths(access.root)
    manifest = _canonical_root_path(access.root, MANIFEST_RELATIVE)
    if (
        access.receipt != receipt
        or access.result != result
        or access.manifest != manifest
        or access.result.exists()
        or not access.receipt.is_file()
    ):
        raise RuntimeError("confirmation access is not bound to an active canonical receipt")
    payload = _json_without_duplicate_keys(
        access.receipt.read_text(encoding="utf-8"), source=str(RECEIPT_RELATIVE)
    )
    if (
        payload.get("status") != "opened-before-seed-access"
        or payload.get("seed_start") != CONFIRMATION_SEEDS.start
        or payload.get("seed_stop_exclusive") != CONFIRMATION_SEEDS.stop
        or payload.get("manifest_path") != str(MANIFEST_RELATIVE)
        or payload.get("manifest_sha256") != access.manifest_sha256
        or payload.get("fixed_protocol") != _normalised_protocol(FIXED_PROTOCOL)
        or sha256(access.manifest) != access.manifest_sha256
    ):
        raise RuntimeError("confirmation receipt payload is invalid")


@dataclass(frozen=True)
class _SeedNamespaces:
    seed: int

    def digest(self, namespace: str, *coordinates: int) -> bytes:
        if not namespace or not namespace.isascii():
            raise ValueError("namespace must be nonempty ASCII")
        payload = bytearray(b"CE-V18b\x00")
        payload.extend(int(self.seed).to_bytes(8, "big", signed=False))
        payload.extend(namespace.encode("ascii"))
        payload.append(0)
        for coordinate in coordinates:
            payload.extend(int(coordinate).to_bytes(8, "big", signed=True))
        return hashlib.sha256(payload).digest()

    def rng(self, namespace: str, *coordinates: int) -> np.random.Generator:
        entropy = int.from_bytes(self.digest(namespace, *coordinates), "big")
        return np.random.default_rng(entropy)

    def token(self, namespace: str, *coordinates: int) -> int:
        return int.from_bytes(self.digest(namespace, *coordinates)[:8], "big")


def _seed_namespaces(
    seed: int, access: _ConfirmationAccess | None
) -> _SeedNamespaces:
    # This is the sole entry point from a numeric scored seed to any namespace.
    # Consequently closure/receipt checking happens immediately before access.
    _authorise_seed(seed, access)
    return _SeedNamespaces(seed)


def _rademacher(rng: np.random.Generator, dimension: int = DIMENSION) -> np.ndarray:
    bits = rng.integers(0, 2, size=dimension, dtype=np.int8)
    return np.where(bits == 0, -1.0, 1.0).astype(np.float64)


def _sign_tie_positive(value: float) -> int:
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite terminal score")
    return -1 if value < 0.0 else 1


def _encoded(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [[field.name, _encoded(getattr(value, field.name))] for field in fields(value)],
        }
    if isinstance(value, np.ndarray):
        return {"__ndarray__": [_encoded(item) for item in value.flat], "shape": list(value.shape)}
    if isinstance(value, (np.floating, float)):
        return {"__float_hex__": float(value).hex()}
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (tuple, list)):
        return [_encoded(item) for item in value]
    if isinstance(value, str) or value is None:
        return value
    raise TypeError(f"unsupported state value: {type(value).__name__}")


def serialize_state(state: Any) -> str:
    return json.dumps(_encoded(state), sort_keys=True, separators=(",", ":"))


def _strict_state_bytes(state: Any) -> bytes:
    """Evaluator-owned exact binary64 factor serialization."""

    if not is_dataclass(state) or _field_names(state) != ("factor",):
        raise TypeError("strict state must expose only factor")
    raw = np.asarray(state.factor)
    if raw.shape != (DIMENSION, DIMENSION) or raw.dtype.kind not in "iuf":
        raise ValueError("strict factor has the wrong shape or type")
    factor = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(factor)):
        raise FloatingPointError("strict factor is nonfinite")
    return b"CE-V18b-strict-factor\x00" + factor.astype("<f8", copy=False).tobytes(order="C")


def _finite_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a binary64 vector") from error
    if result.shape != (DIMENSION,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-{DIMENSION} vector")
    return result.copy()


def _classifier(state: Any) -> np.ndarray:
    if not is_dataclass(state) or not hasattr(state, "classifier"):
        raise TypeError("learner state must expose classifier")
    return _finite_vector(getattr(state, "classifier"), name="classifier")


def _classifier_serialization(state: Any) -> str:
    return json.dumps([float(value).hex() for value in _classifier(state)], separators=(",", ":"))


@dataclass(frozen=True)
class TrainingEpisode:
    epoch: int
    coordinate: int
    cue_sign: int
    cue: tuple[float, ...]
    label: int
    delay: int
    distractors: tuple[tuple[float, ...], ...]
    namespace_tokens: dict[str, int]


@dataclass(frozen=True)
class AcceptedQuery:
    index: int
    rademacher: tuple[int, ...]
    integer_margin: int
    redraw_attempt: int

    @property
    def unit(self) -> np.ndarray:
        return np.asarray(self.rademacher, dtype=np.float64) / math.sqrt(DIMENSION)


def _training_episodes(
    namespaces: _SeedNamespaces, theta: np.ndarray
) -> list[TrainingEpisode]:
    result: list[TrainingEpisode] = []
    serial = 0
    for epoch in range(EPOCHS):
        order = namespaces.rng("epoch-order", epoch).permutation(DIMENSION)
        for position, raw_coordinate in enumerate(order):
            coordinate = int(raw_coordinate)
            cue_sign = int(
                _rademacher(namespaces.rng("cue-sign", epoch, position), 1)[0]
            )
            cue = np.zeros(DIMENSION, dtype=np.float64)
            cue[coordinate] = cue_sign
            label = int(cue_sign * theta[coordinate])
            delay_rng = namespaces.rng("training-delay", epoch, position)
            delay = int(TRAINING_DELAYS[int(delay_rng.integers(0, len(TRAINING_DELAYS)))])
            distractors = tuple(
                tuple(
                    float(value)
                    for value in (
                        _rademacher(
                            namespaces.rng("training-distractor", epoch, position, step)
                        )
                        / math.sqrt(DIMENSION)
                    )
                )
                for step in range(delay)
            )
            result.append(
                TrainingEpisode(
                    epoch=epoch,
                    coordinate=coordinate,
                    cue_sign=cue_sign,
                    cue=tuple(float(value) for value in cue),
                    label=label,
                    delay=delay,
                    distractors=distractors,
                    namespace_tokens={
                        "message": namespaces.token("training-message", serial),
                        "topology": namespaces.token("training-topology", serial),
                        "update": namespaces.token("training-update", serial),
                        "ensemble": namespaces.token("training-ensemble", serial),
                        "policy": namespaces.token("training-policy", serial),
                        "lesion": namespaces.token("training-lesion", serial),
                    },
                )
            )
            serial += 1
    if len(result) != EPOCHS * DIMENSION:
        raise AssertionError("registered training episode coverage failed")
    return result


def _accepted_queries(
    namespaces: _SeedNamespaces, theta: np.ndarray
) -> list[AcceptedQuery]:
    accepted: list[AcceptedQuery] = []
    theta_int = theta.astype(np.int64)
    for query_index in range(QUERY_PAIR_COUNT):
        for attempt in range(QUERY_REDRAW_CAP):
            raw = _rademacher(
                namespaces.rng("evaluation-query", query_index, attempt)
            ).astype(np.int64)
            margin = int(theta_int @ raw)
            if margin != 0:
                accepted.append(
                    AcceptedQuery(
                        index=query_index,
                        rademacher=tuple(int(value) for value in raw),
                        integer_margin=margin,
                        redraw_attempt=attempt,
                    )
                )
                break
        else:
            raise RuntimeError("integer-margin redraw cap exhausted")
    return accepted


def _paired_nuisance(
    namespaces: _SeedNamespaces, query_index: int, delay: int
) -> dict[str, Any]:
    if delay not in EVALUATION_DELAYS:
        raise ValueError("unregistered evaluation delay")
    distractors = [
        [
            float(value)
            for value in (
                _rademacher(
                    namespaces.rng("evaluation-distractor", query_index, delay, step)
                )
                / math.sqrt(DIMENSION)
            )
        ]
        for step in range(delay)
    ]
    nuisance = {
        "delay": delay,
        "distractors": distractors,
        "messages": [
            namespaces.token("evaluation-message", query_index, delay, step)
            for step in range(delay + 1)
        ],
        "topology_seed": namespaces.token("evaluation-topology", query_index, delay),
        "update_seed": namespaces.token("evaluation-update", query_index, delay),
        "ensemble_seed": namespaces.token("evaluation-ensemble", query_index, delay),
        "policy_seed": namespaces.token("evaluation-policy", query_index, delay),
    }
    rendered = json.dumps(nuisance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    nuisance["sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return nuisance


EligibilityLearner = getattr(PRODUCTION_MODULE, "EligibilityLearner")
HardLatchLearner = getattr(PRODUCTION_MODULE, "HardLatchLearner")
HomogeneousLearner = getattr(PRODUCTION_MODULE, "HomogeneousLearner")
StrictMetricControl = getattr(PRODUCTION_MODULE, "StrictMetricControl")
NoTraceControl = getattr(PRODUCTION_MODULE, "NoTraceControl")


def _field_names(state: Any) -> tuple[str, ...]:
    return tuple(field.name for field in fields(state)) if is_dataclass(state) else ()


def _memory_vector(state: Any, kind: str) -> np.ndarray:
    field = {"eligibility": "trace", "hard_latch": "latch"}.get(kind)
    if field is None or not hasattr(state, field):
        raise TypeError(f"{kind} state does not expose its declared memory")
    return _finite_vector(getattr(state, field), name=field)


def _factor(state: Any) -> np.ndarray:
    if not is_dataclass(state) or not hasattr(state, "factor"):
        raise TypeError("factor state does not expose factor")
    try:
        factor = np.asarray(getattr(state, "factor"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be a binary64 matrix") from error
    if factor.shape != (DIMENSION + 1, DIMENSION + 1):
        raise ValueError("homogeneous factor has the wrong shape")
    if not np.all(np.isfinite(factor)):
        raise FloatingPointError("homogeneous factor is nonfinite")
    return factor.copy()


def _homogeneous_metric(state: Any) -> np.ndarray:
    # Evaluator-owned reconstruction; no production metric/readout helper.
    factor = _factor(state)
    metric = factor @ factor.T
    if not np.all(np.isfinite(metric)):
        raise FloatingPointError("reconstructed homogeneous metric is nonfinite")
    return metric


def _homogeneous_eligibility(state: Any) -> np.ndarray:
    metric = _homogeneous_metric(state)
    return 2.0 * metric[:-1, -1]


def _active(state: Any) -> bool:
    if not is_dataclass(state) or not hasattr(state, "active"):
        raise TypeError("learner state must expose active")
    return bool(getattr(state, "active"))


def _state_is_finite(state: Any) -> bool:
    try:
        encoded = _encoded(state)
    except (TypeError, ValueError):
        return False

    def visit(value: Any) -> bool:
        if isinstance(value, dict):
            if "__float_hex__" in value:
                return math.isfinite(float.fromhex(value["__float_hex__"]))
            return all(visit(item) for item in value.values())
        if isinstance(value, list):
            return all(visit(item) for item in value)
        return True

    return visit(encoded)


def _run_positive_training(
    learner: Any,
    kind: str,
    episodes: Sequence[TrainingEpisode],
    *,
    trace_lesion: bool = False,
    invert_reward: bool = False,
    retain_timing: bool = False,
) -> tuple[Any, dict[str, Any]]:
    state = learner.identity_state()
    timing: list[dict[str, Any]] = []
    maximum_update_defect = 0.0
    maximum_recovery_defect = 0.0
    all_finite = _state_is_finite(state)
    all_actions_match = True
    all_updates_match = True
    all_resets_match = True
    identity = learner.identity_state()
    identity_factor_serialized = (
        serialize_state(getattr(identity, "factor")) if kind == "homogeneous" else None
    )

    for serial, episode in enumerate(episodes):
        cue = np.asarray(episode.cue, dtype=np.float64)
        w_start = _classifier(state)
        start_bytes = _classifier_serialization(state)
        state = learner.write_cue(state, cue)
        after_cue_bytes = _classifier_serialization(state)
        if kind == "homogeneous":
            recovered = _homogeneous_eligibility(state)
            metric = _homogeneous_metric(state)
            z = np.concatenate((cue, np.array([1.0])))
            reference_metric = np.eye(DIMENSION + 1) + 0.5 * np.outer(z, z)
            write_metric_defect = float(np.max(np.abs(metric - reference_metric)))
        else:
            recovered = _memory_vector(state, kind)
            write_metric_defect = 0.0
        recovery_defect = float(np.max(np.abs(recovered - cue)))
        maximum_recovery_defect = max(maximum_recovery_defect, recovery_defect)
        after_distractor_equal: list[bool] = []
        for observation in episode.distractors:
            state = learner.distract(state, observation)
            after_distractor_equal.append(_classifier_serialization(state) == start_bytes)
        pre_reward_bytes = _classifier_serialization(state)
        action = int(learner.action(state))
        expected_action = _sign_tie_positive(float(w_start @ cue))
        all_actions_match = all_actions_match and action == expected_action
        reward = int(action == episode.label)
        decoded_reference = action * (2 * reward - 1)
        if decoded_reference != episode.label:
            raise AssertionError("evaluator-owned binary reward decoding failed")
        if trace_lesion:
            state = learner.trace_lesion(state)
            current_eligibility = (
                _homogeneous_eligibility(state)
                if kind == "homogeneous"
                else _memory_vector(state, kind)
            )
        else:
            current_eligibility = recovered
        expected_label = -episode.label if invert_reward else episode.label
        expected_delta = ETA * expected_label * current_eligibility
        state = learner.reward(
            state,
            action,
            reward,
            invert_reward=invert_reward,
        )
        w_after = _classifier(state)
        actual_delta = w_after - w_start
        update_defect = float(np.max(np.abs(actual_delta - expected_delta)))
        maximum_update_defect = max(maximum_update_defect, update_defect)
        update_match = update_defect <= TOLERANCE
        all_updates_match = all_updates_match and update_match
        if kind == "homogeneous":
            reset_match = (
                serialize_state(getattr(state, "factor")) == identity_factor_serialized
                and not _active(state)
            )
        else:
            reset_memory = _memory_vector(state, kind)
            reset_match = bool(np.array_equal(reset_memory, np.zeros(DIMENSION))) and not _active(
                state
            )
        all_resets_match = all_resets_match and reset_match
        all_finite = all_finite and _state_is_finite(state)

        if retain_timing:
            timing.append(
                {
                    "episode": serial,
                    "epoch": episode.epoch,
                    "coordinate": episode.coordinate,
                    "cue_sign": episode.cue_sign,
                    "label": episode.label,
                    "delay": episode.delay,
                    "w_start_hex": [float(value).hex() for value in w_start],
                    "w_after_reward_hex": [float(value).hex() for value in w_after],
                    "w_after_cue_equal": after_cue_bytes == start_bytes,
                    "w_after_every_distractor_equal": all(after_distractor_equal),
                    "checked_distractor_count": len(after_distractor_equal),
                    "w_pre_reward_equal": pre_reward_bytes == start_bytes,
                    "action": action,
                    "expected_action": expected_action,
                    "reward": reward,
                    "independently_decoded_label": decoded_reference,
                    "expected_delta": [float(value) for value in expected_delta],
                    "actual_delta": [float(value) for value in actual_delta],
                    "update_defect": update_defect,
                    "update_match": update_match,
                    "cue_recovery_defect": recovery_defect,
                    "homogeneous_write_metric_defect": write_metric_defect,
                    "atomic_reset": reset_match,
                    "namespace_tokens": episode.namespace_tokens,
                }
            )

    if retain_timing and len(timing) != EPOCHS * DIMENSION:
        raise AssertionError("missing classifier timing episode")
    return state, {
        "finite": all_finite,
        "episode_count": len(episodes),
        "all_actions_match_independent_reference": all_actions_match,
        "all_updates_match_independent_reference": all_updates_match,
        "all_atomic_resets_match": all_resets_match,
        "maximum_update_defect": maximum_update_defect,
        "maximum_cue_recovery_defect": maximum_recovery_defect,
        "final_classifier": [float(value) for value in _classifier(state)],
        "timing": timing,
    }


def _run_no_trace_training(
    learner: Any, episodes: Sequence[TrainingEpisode]
) -> tuple[Any, dict[str, Any]]:
    state = learner.identity_state()
    maximum_delta = 0.0
    for episode in episodes:
        cue = np.asarray(episode.cue, dtype=np.float64)
        w_before = _classifier(state)
        state = learner.write_cue(state, cue)
        for observation in episode.distractors:
            state = learner.distract(state, observation)
        action = int(learner.action(state))
        reward = int(action == episode.label)
        state = learner.reward(state, action, reward)
        maximum_delta = max(
            maximum_delta, float(np.max(np.abs(_classifier(state) - w_before)))
        )
    return state, {
        "finite": _state_is_finite(state),
        "episode_count": len(episodes),
        "maximum_classifier_delta": maximum_delta,
        "final_classifier": [float(value) for value in _classifier(state)],
    }


def _positive_branch(
    learner: Any,
    checkpoint: Any,
    cue: np.ndarray,
    nuisance: dict[str, Any],
) -> dict[str, Any]:
    # Transactional evaluation: every branch restores the supplied checkpoint,
    # and evaluation reward never reaches the learner.
    state = learner.from_snapshot(learner.snapshot(checkpoint))
    checkpoint_serialized = serialize_state(state)
    classifier_before = _classifier_serialization(state)
    state = learner.write_cue(state, cue)
    state = learner.distract_many(state, nuisance["distractors"])
    action = int(learner.action(state))
    classifier_unchanged = _classifier_serialization(state) == classifier_before
    return {
        "action": action,
        "checkpoint_sha256": hashlib.sha256(checkpoint_serialized.encode()).hexdigest(),
        "terminal_state_sha256": hashlib.sha256(serialize_state(state).encode()).hexdigest(),
        "classifier_unchanged": classifier_unchanged,
        "finite": _state_is_finite(state),
    }


def _no_trace_branch(
    learner: Any,
    checkpoint: Any,
    cue: np.ndarray,
    nuisance: dict[str, Any],
) -> dict[str, Any]:
    state = learner.from_snapshot(learner.snapshot(checkpoint))
    checkpoint_serialized = serialize_state(state)
    classifier_before = _classifier_serialization(state)
    state = learner.write_cue(state, cue)
    state = learner.distract_many(state, nuisance["distractors"])
    return {
        "action": int(learner.action(state)),
        "checkpoint_sha256": hashlib.sha256(checkpoint_serialized.encode()).hexdigest(),
        "terminal_state_sha256": hashlib.sha256(serialize_state(state).encode()).hexdigest(),
        "classifier_unchanged": _classifier_serialization(state) == classifier_before,
        "finite": _state_is_finite(state),
    }


def _evaluate_route(
    learner: Any,
    checkpoint: Any,
    queries: Sequence[AcceptedQuery],
    nuisances: dict[tuple[int, int], dict[str, Any]],
    *,
    no_trace: bool = False,
) -> dict[str, Any]:
    delays: dict[str, Any] = {}
    branch_function = _no_trace_branch if no_trace else _positive_branch
    for delay in EVALUATION_DELAYS:
        paired_actions: list[list[int]] = []
        correct_count = 0
        all_checkpoint_equal = True
        all_finite = True
        all_classifier_unchanged = True
        for query in queries:
            nuisance = nuisances[(query.index, delay)]
            cue = query.unit
            plus = branch_function(learner, checkpoint, cue, nuisance)
            minus = branch_function(learner, checkpoint, -cue, nuisance)
            plus_label = 1 if query.integer_margin > 0 else -1
            minus_label = -plus_label
            correct_count += int(plus["action"] == plus_label)
            correct_count += int(minus["action"] == minus_label)
            paired_actions.append([plus["action"], minus["action"]])
            all_checkpoint_equal = all_checkpoint_equal and (
                plus["checkpoint_sha256"] == minus["checkpoint_sha256"]
            )
            all_finite = all_finite and plus["finite"] and minus["finite"]
            all_classifier_unchanged = all_classifier_unchanged and (
                plus["classifier_unchanged"] and minus["classifier_unchanged"]
            )
        branch_count = 2 * len(queries)
        delays[str(delay)] = {
            "branch_count": branch_count,
            "correct_count": correct_count,
            "accuracy": correct_count / branch_count,
            "regret": 1.0 - correct_count / branch_count,
            "paired_actions": paired_actions,
            "all_checkpoint_serializations_equal": all_checkpoint_equal,
            "all_classifier_unchanged": all_classifier_unchanged,
            "finite": all_finite,
        }
    delays["delay_accuracy_difference"] = abs(
        delays[str(EVALUATION_DELAYS[0])]["accuracy"]
        - delays[str(EVALUATION_DELAYS[1])]["accuracy"]
    )
    return delays


def _strict_ensemble_branches(
    learner: Any,
    checkpoints: Sequence[Any],
    cue: np.ndarray,
    nuisance: dict[str, Any],
    checkpoint_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    # States are frozen dataclasses and every production transition returns a
    # new state, so direct reuse is a transactional restore with no mutation.
    states = tuple(checkpoints)
    states = learner.write_cue_many(states, cue)
    states = learner.distract_ensemble(states, nuisance["distractors"])
    results: list[dict[str, Any]] = []
    for state, checkpoint_sha256 in zip(states, checkpoint_hashes, strict=True):
        serialized = _strict_state_bytes(state)
        results.append(
            {
                "state": state,
                "checkpoint_sha256": checkpoint_sha256,
                "serialized": serialized,
                "state_sha256": hashlib.sha256(serialized).hexdigest(),
                "finite": True,
            }
        )
    return results


def _strict_initial_ensemble(
    learner: Any,
    namespaces: _SeedNamespaces,
) -> list[Any]:
    """Create 64 distinct, teacher-independent evaluator-seeded SPD states."""

    states: list[Any] = []
    serializations: set[str] = set()
    for member in range(max(ENSEMBLE_SIZES)):
        rng = namespaces.rng("strict-initial-metric", member)
        raw = rng.normal(0.0, 0.125, size=(DIMENSION, DIMENSION))
        metric = np.eye(DIMENSION) + raw @ raw.T
        state = learner.make_state_from_metric(metric)
        serialization = _strict_state_bytes(state)
        if serialization in serializations:
            raise RuntimeError("strict ensemble initial states must be distinct")
        serializations.add(serialization)
        states.append(state)
    return states


def _train_strict_ensemble(
    learner: Any,
    initial_states: Sequence[Any],
    episodes: Sequence[TrainingEpisode],
) -> tuple[list[Any], dict[str, Any]]:
    states = [learner.from_snapshot(learner.snapshot(state)) for state in initial_states]
    initial_distinct = len({_strict_state_bytes(state) for state in states}) == len(states)
    event_count = 0
    distractor_count = 0
    all_actions_reference = True
    all_reward_no_op = True
    all_finite = True
    timing: list[dict[str, Any]] = []
    for serial, episode in enumerate(episodes):
        cue = np.asarray(episode.cue, dtype=np.float64)
        states = list(learner.write_ensemble(states, cue))
        after_cue_payloads = [_strict_state_bytes(state) for state in states]
        states = list(learner.distract_ensemble(tuple(states), episode.distractors))
        after_distractor_payloads = [_strict_state_bytes(state) for state in states]
        if after_cue_payloads != after_distractor_payloads:
            raise AssertionError("strict distractor transition was not an exact no-op")
        next_states: list[Any] = []
        member_state_hashes: list[str] = []
        for state, before_reward in zip(
            states, after_distractor_payloads, strict=True
        ):
            action = int(learner.action(state))
            reward = int(action == episode.label)
            rewarded = learner.reward(state, action, reward)
            after_reward = _strict_state_bytes(rewarded)
            all_actions_reference = all_actions_reference and action == 1
            all_reward_no_op = all_reward_no_op and before_reward == after_reward
            all_finite = all_finite and _state_is_finite(rewarded)
            member_state_hashes.append(
                hashlib.sha256(after_reward).hexdigest()
            )
            next_states.append(rewarded)
            event_count += 1
            distractor_count += len(episode.distractors)
        states = next_states
        timing.append(
            {
                "episode": serial,
                "delay": episode.delay,
                "member_count": len(states),
                "member_state_hashes": member_state_hashes,
                "all_member_hashes_distinct": len(set(member_state_hashes)) == len(states),
                "action_reference_match": all_actions_reference,
                "reward_no_op": all_reward_no_op,
                "namespace_tokens": episode.namespace_tokens,
            }
        )
    return states, {
        "finite": all_finite,
        "episode_count": len(episodes),
        "member_count": len(states),
        "member_episode_count": event_count,
        "member_distractor_event_count": distractor_count,
        "initial_member_serializations_distinct": initial_distinct,
        "all_actions_match_independent_reference": all_actions_reference,
        "all_rewards_exact_no_op": all_reward_no_op,
        "timing": timing,
    }


def _evaluate_strict(
    learner: Any,
    checkpoints: Sequence[Any],
    queries: Sequence[AcceptedQuery],
    nuisances: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    if len(checkpoints) != max(ENSEMBLE_SIZES):
        raise ValueError("strict evaluation requires the registered 64-member ensemble")
    checkpoint_hashes = [
        hashlib.sha256(_strict_state_bytes(state)).hexdigest() for state in checkpoints
    ]
    delays: dict[str, Any] = {}
    for delay in EVALUATION_DELAYS:
        paired_actions: list[list[int]] = []
        state_pairs: list[dict[str, Any]] = []
        correct_count = 0
        all_state_equal = True
        all_checkpoint_equal = True
        all_finite = True
        aggregate_records: dict[str, list[dict[str, Any]]] = {
            str(size): [] for size in ENSEMBLE_SIZES
        }
        for query in queries:
            nuisance = nuisances[(query.index, delay)]
            plus_members = _strict_ensemble_branches(
                learner, checkpoints, query.unit, nuisance, checkpoint_hashes
            )
            minus_members = _strict_ensemble_branches(
                learner, checkpoints, -query.unit, nuisance, checkpoint_hashes
            )
            plus = plus_members[0]
            minus = minus_members[0]
            plus_label = 1 if query.integer_margin > 0 else -1
            minus_label = -plus_label
            base_action = int(learner.aggregate_action([plus["state"]]))
            correct_count += int(base_action == plus_label)
            correct_count += int(base_action == minus_label)
            paired_actions.append([base_action, base_action])
            member_equal = [
                left["serialized"] == right["serialized"]
                for left, right in zip(plus_members, minus_members, strict=True)
            ]
            state_equal = all(member_equal)
            all_state_equal = all_state_equal and state_equal
            all_checkpoint_equal = all_checkpoint_equal and (
                plus["checkpoint_sha256"] == minus["checkpoint_sha256"]
            )
            all_finite = all_finite and plus["finite"] and minus["finite"]
            state_pairs.append(
                {
                    "plus_sha256": plus["state_sha256"],
                    "minus_sha256": minus["state_sha256"],
                    "equal": state_equal,
                    "member_equal_count": sum(member_equal),
                    "member_count": len(member_equal),
                    "nuisance_sha256": nuisance["sha256"],
                }
            )
            for size in ENSEMBLE_SIZES:
                plus_states = [item["state"] for item in plus_members[:size]]
                minus_states = [item["state"] for item in minus_members[:size]]
                plus_action = (
                    base_action if size == 1 else int(learner.aggregate_action(plus_states))
                )
                # Independent sorted-multiset aggregation verifies production.
                def framed(payloads: list[bytes]) -> bytes:
                    sorted_payloads = sorted(payloads)
                    chunks = [len(sorted_payloads).to_bytes(2, "big")]
                    for payload in sorted_payloads:
                        chunks.extend((len(payload).to_bytes(4, "big"), payload))
                    return b"".join(chunks)

                plus_aggregate = framed(
                    [item["serialized"] for item in plus_members[:size]]
                )
                minus_aggregate = framed(
                    [item["serialized"] for item in minus_members[:size]]
                )
                if plus_aggregate != minus_aggregate:
                    raise AssertionError("paired strict aggregates are not byte-identical")
                # Exact aggregate equality plus a deterministic production
                # function entails the same realized action without a second
                # redundant production call.
                minus_action = plus_action
                independent_action = 1
                aggregate_records[str(size)].append(
                    {
                        "equal": plus_aggregate == minus_aggregate,
                        "plus_sha256": hashlib.sha256(plus_aggregate).hexdigest(),
                        "minus_sha256": hashlib.sha256(minus_aggregate).hexdigest(),
                        "actions": [plus_action, minus_action],
                        "production_actions_match_independent": plus_action
                        == independent_action
                        and minus_action == independent_action,
                        "pair_correct": int(plus_action == plus_label)
                        + int(minus_action == minus_label),
                    }
                )
        branch_count = 2 * len(queries)
        ensembles: dict[str, Any] = {}
        for size, records in aggregate_records.items():
            ensemble_correct = sum(item["pair_correct"] for item in records)
            ensembles[size] = {
                "all_aggregate_serializations_equal": all(item["equal"] for item in records),
                "branch_count": branch_count,
                "correct_count": ensemble_correct,
                "accuracy": ensemble_correct / branch_count,
                "all_production_actions_match_independent": all(
                    item["production_actions_match_independent"] for item in records
                ),
                "records": records,
            }
        delays[str(delay)] = {
            "branch_count": branch_count,
            "correct_count": correct_count,
            "accuracy": correct_count / branch_count,
            "regret": 1.0 - correct_count / branch_count,
            "paired_actions": paired_actions,
            "state_pairs": state_pairs,
            "all_state_serializations_equal": all_state_equal,
            "all_checkpoint_serializations_equal": all_checkpoint_equal,
            "finite": all_finite,
            "ensembles": ensembles,
        }
    return delays


def _state_certificate(learners: dict[str, Any]) -> dict[str, Any]:
    states = {name: learner.identity_state() for name, learner in learners.items()}
    homogeneous_factor = _factor(states["homogeneous"])
    return {
        "eligibility_fields": list(_field_names(states["eligibility"])),
        "homogeneous_fields": list(_field_names(states["homogeneous"])),
        "hard_latch_fields": list(_field_names(states["hard_latch"])),
        "strict_fields": list(_field_names(states["strict"])),
        "no_trace_fields": list(_field_names(states["no_trace"])),
        "homogeneous_factor_shape": list(homogeneous_factor.shape),
        "homogeneous_independent_factor_coordinates": (
            (DIMENSION + 1) * (DIMENSION + 2) // 2
        ),
        "homogeneous_dense_serialized_entries": int(homogeneous_factor.size),
        "homogeneous_added_independent_coordinates": DIMENSION + 1,
        "production_module_path": str(IMPORTED_PRODUCTION_PATH),
        "isolated_module_name": PRODUCTION_MODULE.__name__,
        "package_initializer_imported_by_evaluator": False,
    }


def score_seed(
    seed: int,
    *,
    confirmation_access: _ConfirmationAccess | None = None,
) -> dict[str, Any]:
    try:
        namespaces = _seed_namespaces(seed, confirmation_access)
        theta = _rademacher(namespaces.rng("teacher"))
        episodes = _training_episodes(namespaces, theta)
        queries = _accepted_queries(namespaces, theta)
        nuisances = {
            (query.index, delay): _paired_nuisance(namespaces, query.index, delay)
            for query in queries
            for delay in EVALUATION_DELAYS
        }
        learners = {
            "eligibility": EligibilityLearner(DIMENSION, ETA),
            "homogeneous": HomogeneousLearner(DIMENSION, ETA),
            "hard_latch": HardLatchLearner(DIMENSION, ETA),
            "strict": StrictMetricControl(DIMENSION),
            "no_trace": NoTraceControl(DIMENSION, ETA),
        }
        initial = {
            name: learner.identity_state()
            for name, learner in learners.items()
            if name != "strict"
        }
        strict_initial = _strict_initial_ensemble(learners["strict"], namespaces)

        trained: dict[str, Any] = {}
        training: dict[str, Any] = {}
        trained["eligibility"], training["eligibility"] = _run_positive_training(
            learners["eligibility"], "eligibility", episodes, retain_timing=True
        )
        trained["homogeneous"], training["homogeneous"] = _run_positive_training(
            learners["homogeneous"], "homogeneous", episodes, retain_timing=True
        )
        trained["hard_latch"], training["hard_latch"] = _run_positive_training(
            learners["hard_latch"], "hard_latch", episodes, retain_timing=True
        )
        trained["eligibility_trace_lesion"], training[
            "eligibility_trace_lesion"
        ] = _run_positive_training(
            learners["eligibility"], "eligibility", episodes, trace_lesion=True
        )
        trained["homogeneous_trace_lesion"], training[
            "homogeneous_trace_lesion"
        ] = _run_positive_training(
            learners["homogeneous"], "homogeneous", episodes, trace_lesion=True
        )
        trained["eligibility_reward_inversion"], training[
            "eligibility_reward_inversion"
        ] = _run_positive_training(
            learners["eligibility"], "eligibility", episodes, invert_reward=True
        )
        trained["homogeneous_reward_inversion"], training[
            "homogeneous_reward_inversion"
        ] = _run_positive_training(
            learners["homogeneous"], "homogeneous", episodes, invert_reward=True
        )
        trained["no_trace"], training["no_trace"] = _run_no_trace_training(
            learners["no_trace"], episodes
        )
        trained["strict"], training["strict"] = _train_strict_ensemble(
            learners["strict"], strict_initial, episodes
        )

        route_specs = {
            "eligibility_pre": (learners["eligibility"], initial["eligibility"], False),
            "homogeneous_pre": (learners["homogeneous"], initial["homogeneous"], False),
            "eligibility": (learners["eligibility"], trained["eligibility"], False),
            "homogeneous": (learners["homogeneous"], trained["homogeneous"], False),
            "hard_latch": (learners["hard_latch"], trained["hard_latch"], False),
            "eligibility_trace_lesion": (
                learners["eligibility"],
                trained["eligibility_trace_lesion"],
                False,
            ),
            "homogeneous_trace_lesion": (
                learners["homogeneous"],
                trained["homogeneous_trace_lesion"],
                False,
            ),
            "eligibility_reward_inversion": (
                learners["eligibility"],
                trained["eligibility_reward_inversion"],
                False,
            ),
            "homogeneous_reward_inversion": (
                learners["homogeneous"],
                trained["homogeneous_reward_inversion"],
                False,
            ),
            "no_trace": (learners["no_trace"], trained["no_trace"], True),
        }
        evaluation = {
            name: _evaluate_route(
                learner, checkpoint, queries, nuisances, no_trace=no_trace
            )
            for name, (learner, checkpoint, no_trace) in route_specs.items()
        }
        evaluation["strict"] = _evaluate_strict(
            learners["strict"], trained["strict"], queries, nuisances
        )

        theta_defects = {
            name: float(np.max(np.abs(_classifier(state) - target)))
            for name, state, target in (
                ("eligibility", trained["eligibility"], theta),
                ("homogeneous", trained["homogeneous"], theta),
                ("hard_latch", trained["hard_latch"], theta),
                ("eligibility_trace_lesion", trained["eligibility_trace_lesion"], np.zeros(DIMENSION)),
                ("homogeneous_trace_lesion", trained["homogeneous_trace_lesion"], np.zeros(DIMENSION)),
                ("eligibility_reward_inversion", trained["eligibility_reward_inversion"], -theta),
                ("homogeneous_reward_inversion", trained["homogeneous_reward_inversion"], -theta),
                ("no_trace", trained["no_trace"], np.zeros(DIMENSION)),
            )
        }
        query_records = [
            {
                "index": query.index,
                "rademacher": list(query.rademacher),
                "integer_margin": query.integer_margin,
                "redraw_attempt": query.redraw_attempt,
                "plus_label": 1 if query.integer_margin > 0 else -1,
                "nuisance_sha256": {
                    str(delay): nuisances[(query.index, delay)]["sha256"]
                    for delay in EVALUATION_DELAYS
                },
            }
            for query in queries
        ]
        namespace_certificate = {
            name: namespaces.digest(name).hex()
            for name in (
                "teacher",
                "epoch-order",
                "cue-sign",
                "training-delay",
                "training-distractor",
                "evaluation-query",
                "evaluation-distractor",
                "training-lesion",
                "strict-initial-metric",
                "evaluation-update",
                "evaluation-ensemble",
                "evaluation-policy",
            )
        }
        result = {
            "seed": seed,
            "finite": bool(
                all(item["finite"] for item in training.values())
                and all(
                    delay_result["finite"]
                    for route in evaluation.values()
                    for key, delay_result in route.items()
                    if key in {"0", "128"}
                )
            ),
            "teacher": [int(value) for value in theta],
            "namespace_certificate": namespace_certificate,
            "namespace_digests_unique": len(set(namespace_certificate.values()))
            == len(namespace_certificate),
            "training": training,
            "final_classifier_reference_defects": theta_defects,
            "queries": query_records,
            "evaluation": evaluation,
            "state_certificate": _state_certificate(learners),
        }
        return result
    except (
        AssertionError,
        FloatingPointError,
        OverflowError,
        TypeError,
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "seed": seed,
            "finite": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _route_accuracy(seed_summary: dict[str, Any], route: str, delay: int) -> float:
    return float(seed_summary["evaluation"][route][str(delay)]["accuracy"])


def _all_seed_gate(summaries: Sequence[dict[str, Any]], predicate: Any) -> bool:
    return bool(summaries) and all(bool(predicate(item)) for item in summaries)


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    expected_seed_count = len(summaries)
    complete = [item for item in summaries if item.get("finite") and "evaluation" in item]
    routes = (
        "eligibility_pre",
        "homogeneous_pre",
        "eligibility",
        "homogeneous",
        "hard_latch",
        "eligibility_trace_lesion",
        "homogeneous_trace_lesion",
        "eligibility_reward_inversion",
        "homogeneous_reward_inversion",
        "no_trace",
        "strict",
    )
    route_means: dict[str, Any] = {}
    for route in routes:
        route_means[route] = {
            str(delay): (
                float(np.mean([_route_accuracy(item, route, delay) for item in complete]))
                if complete
                else 0.0
            )
            for delay in EVALUATION_DELAYS
        }
    timing_coverage = {
        route: sum(
            len(item["training"][route]["timing"])
            for item in complete
        )
        for route in ("eligibility", "homogeneous", "hard_latch")
    }
    max_defects = {
        route: max(
            (item["final_classifier_reference_defects"][route] for item in complete),
            default=float(np.finfo(np.float64).max),
        )
        for route in (
            "eligibility",
            "homogeneous",
            "hard_latch",
            "eligibility_trace_lesion",
            "homogeneous_trace_lesion",
            "eligibility_reward_inversion",
            "homogeneous_reward_inversion",
            "no_trace",
        )
    }
    return {
        "seed_count": expected_seed_count,
        "finite_seed_count": len(complete),
        "finite_seed_rate": len(complete) / expected_seed_count if expected_seed_count else 0.0,
        "route_mean_accuracies": route_means,
        "timing_episode_coverage": timing_coverage,
        "maximum_classifier_reference_defects": max_defects,
    }


def _gate_decisions(summaries: Sequence[dict[str, Any]]) -> dict[str, bool]:
    positive = ("eligibility", "homogeneous", "hard_latch")
    trace_lesions = ("eligibility_trace_lesion", "homogeneous_trace_lesion")
    inversions = ("eligibility_reward_inversion", "homogeneous_reward_inversion")
    def timing(item: dict[str, Any], route: str) -> list[dict[str, Any]]:
        return item["training"][route]["timing"]

    return {
        "finite": _all_seed_gate(summaries, lambda item: item.get("finite", False)),
        "namespace_separation": _all_seed_gate(
            summaries, lambda item: item["namespace_digests_unique"]
        ),
        "query_coverage_and_nonzero_integer_margins": _all_seed_gate(
            summaries,
            lambda item: len(item["queries"]) == QUERY_PAIR_COUNT
            and all(query["integer_margin"] != 0 for query in item["queries"]),
        ),
        "pretraining_paired_half": _all_seed_gate(
            summaries,
            lambda item: all(
                _route_accuracy(item, route, delay) == 0.5
                for route in ("eligibility_pre", "homogeneous_pre")
                for delay in EVALUATION_DELAYS
            ),
        ),
        "all_32_classifier_timing_checks": _all_seed_gate(
            summaries,
            lambda item: all(
                len(timing(item, route)) == EPOCHS * DIMENSION
                and item["training"][route][
                    "all_actions_match_independent_reference"
                ]
                and item["training"][route]["all_atomic_resets_match"]
                and all(
                    record["w_after_cue_equal"]
                    and record["w_after_every_distractor_equal"]
                    and record["w_pre_reward_equal"]
                    and record["checked_distractor_count"] == record["delay"]
                    and record["update_match"]
                    and record["atomic_reset"]
                    and record["action"] == record["expected_action"]
                    for record in timing(item, route)
                )
                for route in positive
            ),
        ),
        "intact_exact_learning": _all_seed_gate(
            summaries,
            lambda item: all(
                item["final_classifier_reference_defects"][route] <= TOLERANCE
                and item["training"][route]["all_updates_match_independent_reference"]
                for route in positive
            ),
        ),
        "homogeneous_recovery_and_atomic_reset": _all_seed_gate(
            summaries,
            lambda item: item["training"]["homogeneous"]["maximum_cue_recovery_defect"]
            <= TOLERANCE
            and item["training"]["homogeneous"]["all_atomic_resets_match"],
        ),
        "homogeneous_every_write_metric": _all_seed_gate(
            summaries,
            lambda item: len(timing(item, "homogeneous")) == EPOCHS * DIMENSION
            and all(
                record["homogeneous_write_metric_defect"] <= TOLERANCE
                for record in timing(item, "homogeneous")
            ),
        ),
        "strict_full_training_stream": _all_seed_gate(
            summaries,
            lambda item: item["training"]["strict"]["episode_count"]
            == EPOCHS * DIMENSION
            and item["training"]["strict"]["member_count"] == max(ENSEMBLE_SIZES)
            and item["training"]["strict"]["member_episode_count"]
            == EPOCHS * DIMENSION * max(ENSEMBLE_SIZES)
            and item["training"]["strict"]["initial_member_serializations_distinct"]
            and item["training"]["strict"]["all_actions_match_independent_reference"]
            and item["training"]["strict"]["all_rewards_exact_no_op"]
            and item["training"]["strict"]["member_distractor_event_count"]
            == max(ENSEMBLE_SIZES)
            * sum(
                record["delay"]
                for record in item["training"]["eligibility"]["timing"]
            )
            and len(item["training"]["strict"]["timing"]) == EPOCHS * DIMENSION
            and all(
                record["member_count"] == max(ENSEMBLE_SIZES)
                and record["all_member_hashes_distinct"]
                and record["action_reference_match"]
                and record["reward_no_op"]
                for record in item["training"]["strict"]["timing"]
            ),
        ),
        "evaluation_branch_coverage": _all_seed_gate(
            summaries,
            lambda item: len(item["queries"]) == QUERY_PAIR_COUNT
            and all(
                item["evaluation"][route][str(delay)]["branch_count"]
                == 2 * QUERY_PAIR_COUNT
                and len(item["evaluation"][route][str(delay)]["paired_actions"])
                == QUERY_PAIR_COUNT
                for route in (
                    "eligibility_pre",
                    "homogeneous_pre",
                    "eligibility",
                    "homogeneous",
                    "hard_latch",
                    "eligibility_trace_lesion",
                    "homogeneous_trace_lesion",
                    "eligibility_reward_inversion",
                    "homogeneous_reward_inversion",
                    "no_trace",
                    "strict",
                )
                for delay in EVALUATION_DELAYS
            )
            and all(
                len(item["evaluation"]["strict"][str(delay)]["state_pairs"])
                == QUERY_PAIR_COUNT
                and all(
                    len(
                        item["evaluation"]["strict"][str(delay)]["ensembles"]
                        [str(size)]["records"]
                    )
                    == QUERY_PAIR_COUNT
                    for size in ENSEMBLE_SIZES
                )
                for delay in EVALUATION_DELAYS
            ),
        ),
        "positive_delay_composition": _all_seed_gate(
            summaries,
            lambda item: all(
                _route_accuracy(item, route, delay) == 1.0
                for route in positive
                for delay in EVALUATION_DELAYS
            ),
        ),
        "positive_zero_delay_difference": _all_seed_gate(
            summaries,
            lambda item: all(
                item["evaluation"][route]["delay_accuracy_difference"] == 0.0
                for route in positive
            ),
        ),
        "trace_and_no_trace_half": _all_seed_gate(
            summaries,
            lambda item: all(
                _route_accuracy(item, route, delay) == 0.5
                for route in (*trace_lesions, "no_trace")
                for delay in EVALUATION_DELAYS
            ),
        ),
        "reward_inversion_zero": _all_seed_gate(
            summaries,
            lambda item: all(
                _route_accuracy(item, route, delay) == 0.0
                for route in inversions
                for delay in EVALUATION_DELAYS
            ),
        ),
        "strict_pointwise_paired_half": _all_seed_gate(
            summaries,
            lambda item: all(
                item["evaluation"]["strict"][str(delay)]["accuracy"] == 0.5
                and item["evaluation"]["strict"][str(delay)][
                    "all_state_serializations_equal"
                ]
                and item["evaluation"]["strict"][str(delay)][
                    "all_checkpoint_serializations_equal"
                ]
                and all(
                    pair["equal"]
                    and pair["member_equal_count"] == max(ENSEMBLE_SIZES)
                    and pair["member_count"] == max(ENSEMBLE_SIZES)
                    for pair in item["evaluation"]["strict"][str(delay)]["state_pairs"]
                )
                for delay in EVALUATION_DELAYS
            ),
        ),
        "strict_every_ensemble_half": _all_seed_gate(
            summaries,
            lambda item: all(
                item["evaluation"]["strict"][str(delay)]["ensembles"][str(size)][
                    "accuracy"
                ]
                == 0.5
                and item["evaluation"]["strict"][str(delay)]["ensembles"][str(size)][
                    "all_aggregate_serializations_equal"
                ]
                and item["evaluation"]["strict"][str(delay)]["ensembles"][str(size)][
                    "all_production_actions_match_independent"
                ]
                for delay in EVALUATION_DELAYS
                for size in ENSEMBLE_SIZES
            ),
        ),
        "declared_state_fields": _all_seed_gate(
            summaries,
            lambda item: item["state_certificate"]["eligibility_fields"]
            == ["classifier", "trace", "active"]
            and item["state_certificate"]["homogeneous_fields"]
            == ["classifier", "factor", "active"]
            and item["state_certificate"]["hard_latch_fields"]
            == ["classifier", "latch", "active"]
            and item["state_certificate"]["strict_fields"] == ["factor"]
            and item["state_certificate"]["no_trace_fields"]
            == ["classifier", "active"]
            and item["state_certificate"]["homogeneous_factor_shape"] == [9, 9]
            and item["state_certificate"]["homogeneous_independent_factor_coordinates"] == 45
            and item["state_certificate"]["homogeneous_dense_serialized_entries"] == 81,
        ),
    }


def evaluate(
    seeds: range,
    mode: str,
    *,
    confirmation_access: _ConfirmationAccess | None = None,
) -> dict[str, Any]:
    if mode == "development":
        if seeds != DEVELOPMENT_SEEDS or confirmation_access is not None:
            raise ValueError("development evaluation requires the registered development block")
    elif mode == "confirmation":
        if seeds != CONFIRMATION_SEEDS or type(confirmation_access) is not _ConfirmationAccess:
            raise RuntimeError("confirmation evaluation requires receipt-bound access")
    else:
        raise ValueError("unknown evaluation mode")
    summaries = [score_seed(seed, confirmation_access=confirmation_access) for seed in seeds]
    gates = _gate_decisions(summaries)
    return {
        "mode": mode,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "protocol": _normalised_protocol(FIXED_PROTOCOL),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "summary": aggregate(summaries),
        "gates": gates,
        "strict_metric_no_go_pass": all(
            value for name, value in gates.items() if name.startswith("strict_")
        ),
        "reward_decoded_eligibility_pass": all(gates.values()),
        "per_seed": summaries,
    }


def development() -> dict[str, Any]:
    return evaluate(DEVELOPMENT_SEEDS, "development")


def open_confirmation_block(
    root: Path,
    manifest_path: Path,
    *,
    preflight: object | None = None,
) -> _ConfirmationAccess:
    global _ACTIVE_CONFIRMATION_ACCESS, _ACTIVE_CONFIRMATION_PREFLIGHT

    if preflight is None or preflight is not _ACTIVE_CONFIRMATION_PREFLIGHT:
        raise RuntimeError("confirmation opening requires completed sealed preflight")
    _ACTIVE_CONFIRMATION_PREFLIGHT = None
    receipt, result = _result_paths(root)
    if result.exists():
        raise RuntimeError("confirmation result already exists; seed block is closed")
    if receipt.exists():
        raise RuntimeError("confirmation seed block was already opened")
    assert_loaded_repository_module_closure(root)
    payload = {
        "status": "opened-before-seed-access",
        "seed_start": CONFIRMATION_SEEDS.start,
        "seed_stop_exclusive": CONFIRMATION_SEEDS.stop,
        "manifest_path": str(MANIFEST_RELATIVE),
        "manifest_sha256": sha256(manifest_path.resolve(strict=True)),
        "fixed_protocol": _normalised_protocol(FIXED_PROTOCOL),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "repository_module_closure_verified": True,
    }
    try:
        with receipt.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError("confirmation seed block was already opened") from error
    access = _ConfirmationAccess(
        root=root.resolve(strict=True),
        receipt=receipt,
        result=result,
        manifest=manifest_path.resolve(strict=True),
        manifest_sha256=payload["manifest_sha256"],
        token=object(),
    )
    _ACTIVE_CONFIRMATION_ACCESS = access
    return access


def confirmation(root: Path, manifest_path: Path) -> dict[str, Any]:
    global _ACTIVE_CONFIRMATION_ACCESS, _ACTIVE_CONFIRMATION_PREFLIGHT

    manifest = verify_manifest(root, manifest_path)
    verify_development_provenance(root)
    assert_loaded_repository_module_closure(root)
    preflight = object()
    _ACTIVE_CONFIRMATION_PREFLIGHT = preflight
    try:
        access = open_confirmation_block(root, manifest_path, preflight=preflight)
    finally:
        if _ACTIVE_CONFIRMATION_PREFLIGHT is preflight:
            _ACTIVE_CONFIRMATION_PREFLIGHT = None
    try:
        result = evaluate(
            CONFIRMATION_SEEDS,
            "confirmation",
            confirmation_access=access,
        )
    finally:
        if _ACTIVE_CONFIRMATION_ACCESS is access:
            _ACTIVE_CONFIRMATION_ACCESS = None

    # Closing repository closure and byte rehash are the final reads before the
    # exclusive result creation.  Any failure burns the receipt and writes no
    # confirmation result.
    assert_loaded_repository_module_closure(root)
    closing_manifest = verify_manifest(root, manifest_path)
    if closing_manifest != manifest or sha256(manifest_path) != access.manifest_sha256:
        raise RuntimeError("sealed manifest changed during confirmation")
    result.update({"manifest_verified": True, "manifest": manifest})
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with access.result.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError as error:
        raise RuntimeError("confirmation result path appeared after opening") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "development":
        result = development()
        output = _validated_root(args.root).joinpath(*DEVELOPMENT_RESULT_RELATIVE.parts)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        if args.manifest is None:
            parser.error("confirmation requires --manifest")
        result = confirmation(args.root.resolve(), args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
