"""derivation.md 프론트매터의 verify 블록을 기계검증한다. LLM 호출 없음.

사용: verify_derivation.py <derivations/<Q>/attempt-NN.derivation.md> [--no-write]
출력(stdout): {"symbolic": pass|fail|skipped, "numeric": pass|fail|skipped, "details": [...]}
저장: verify/<Q>/attempt-NN/hook_result.json (--no-write가 없을 때)

검사 유형
  identity   lhs, rhs                  sympy simplify(lhs-rhs)==0 → symbolic. 표본 대조 → numeric
  limit      expr, var, point, expected sympy limit → symbolic
  inequality lhs, rhs, relation        무작위 표본 → numeric
  numeric    expr | lhs, rhs, samples, tol   무작위 표본 |expr| 또는 |lhs-rhs| ≤ tol → numeric

sympy가 없으면 symbolic은 skipped(reason=sympy-not-installed)이고 numeric만 numpy로 돈다.
이유: 이 저장소의 정책 허용 Python에는 sympy가 없을 수 있고, 훅은 fail-open이어야 한다.
난수 씨앗은 20260902로 고정한다. 이유: 재현성 없는 pass는 증거가 아니다.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SEED = 20260902
DEFAULT_SAMPLES = 20
DEFAULT_TOL = 1e-9
SYMBOLIC_TYPES = {"identity", "limit"}
NUMERIC_TYPES = {"numeric", "inequality"}

try:  # sympy는 선택 의존성이다.
    import sympy as sp  # type: ignore

    HAVE_SYMPY = True
except Exception:  # pragma: no cover - 환경 의존
    sp = None  # type: ignore
    HAVE_SYMPY = False


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- 프론트매터


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.S)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as error:
        return {"_frontmatter_error": str(error)}, text[match.end():]
    return (data if isinstance(data, dict) else {}), text[match.end():]


# ---------------------------------------------------------------- 기호 선언


class SymbolSpec:
    def __init__(self, name: str, spec: str) -> None:
        words = set(str(spec or "real").lower().replace(",", " ").split())
        self.name = name
        self.is_function = "function" in words
        self.integer = "integer" in words
        self.positive = "positive" in words
        self.nonnegative = "nonnegative" in words
        self.nonzero = "nonzero" in words or self.positive
        self.complex = "complex" in words

    def sympy_symbol(self):  # type: ignore[no-untyped-def]
        if self.is_function:
            return sp.Function(self.name)
        kwargs: dict[str, bool] = {}
        if self.integer:
            kwargs["integer"] = True
        if self.positive:
            kwargs["positive"] = True
        elif self.nonnegative:
            kwargs["nonnegative"] = True
        elif not self.complex:
            kwargs["real"] = True
        if self.nonzero and not self.positive:
            kwargs["nonzero"] = True
        return sp.Symbol(self.name, **kwargs)

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.integer:
            if self.positive:
                return int(rng.integers(1, 13))
            value = int(rng.integers(-6, 7))
            if self.nonzero and value == 0:
                value = 1
            return value
        if self.positive:
            return float(rng.uniform(0.1, 3.0))
        if self.nonnegative:
            return float(rng.uniform(0.0, 3.0))
        value = float(rng.uniform(-3.0, 3.0))
        if self.nonzero and abs(value) < 1e-3:
            value = 0.5
        return value


def parse_symbols(raw: Any) -> dict[str, SymbolSpec]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): SymbolSpec(str(name), str(spec)) for name, spec in raw.items()}


# ---------------------------------------------------------------- 평가 도구


def _numpy_namespace() -> dict[str, Any]:
    names = {
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "abs": np.abs,
        "Abs": np.abs,
        "pi": math.pi,
        "E": math.e,
        "oo": math.inf,
        "factorial": math.factorial,
        "floor": math.floor,
        "ceiling": math.ceil,
    }
    return {"__builtins__": {}, **names}


def _numpy_eval(expr: str, values: dict[str, float | int]) -> float:
    namespace = _numpy_namespace()
    namespace.update(values)
    expr = expr.replace("^", "**")
    return float(eval(expr, namespace))  # noqa: S307 - 검증 전용 제한 네임스페이스


class Evaluator:
    """sympy가 있으면 lambdify, 없으면 numpy eval로 표본 평가한다."""

    def __init__(self, symbols: dict[str, SymbolSpec]) -> None:
        self.symbols = symbols
        self.rng = np.random.default_rng(SEED)
        self.sym: dict[str, Any] = {}
        if HAVE_SYMPY:
            self.sym = {name: spec.sympy_symbol() for name, spec in symbols.items()}

    def sampleable(self) -> bool:
        return not any(spec.is_function for spec in self.symbols.values())

    def parse(self, expr: str):  # type: ignore[no-untyped-def]
        return sp.sympify(str(expr), locals=self.sym)

    def free_ok(self, *exprs: Any) -> bool:
        """표현식이 선언된 기호만 쓰는지 확인한다. 이유: 미선언 기호는 오타이거나 숨은 가정이다."""
        for expr in exprs:
            for symbol in expr.free_symbols:
                if symbol.name not in self.symbols:
                    return False
        return True

    def samples(self, count: int) -> list[dict[str, float | int]]:
        return [
            {name: spec.sample(self.rng) for name, spec in self.symbols.items()}
            for _ in range(count)
        ]

    def evaluate(self, expr: Any, values: dict[str, float | int]) -> float:
        if HAVE_SYMPY:
            ordered = [self.sym[name] for name in self.symbols]
            fn = sp.lambdify(ordered, expr, modules=["numpy", "math"])
            return float(fn(*[values[name] for name in self.symbols]))
        return _numpy_eval(str(expr), values)


# ---------------------------------------------------------------- 개별 검사


def run_identity(check: dict[str, Any], ev: Evaluator) -> dict[str, Any]:
    detail: dict[str, Any] = {"type": "identity", "lhs": check.get("lhs"), "rhs": check.get("rhs")}
    lhs_s, rhs_s = str(check.get("lhs", "")), str(check.get("rhs", ""))
    if HAVE_SYMPY:
        try:
            lhs, rhs = ev.parse(lhs_s), ev.parse(rhs_s)
            if not ev.free_ok(lhs, rhs):
                detail["symbolic"] = "fail"
                detail["reason"] = "undeclared symbol in expression"
                return detail
            diff = sp.simplify(sp.expand(lhs - rhs))
            if diff == 0:
                detail["symbolic"] = "pass"
            else:
                detail["symbolic"] = "fail"
                detail["residual"] = str(diff)
        except Exception as error:  # noqa: BLE001
            detail["symbolic"] = "fail"
            detail["reason"] = f"sympy error: {error}"
            return detail
    else:
        detail["symbolic"] = "skipped"
        detail["reason"] = "sympy-not-installed"
        lhs, rhs = lhs_s, rhs_s
    _numeric_compare(detail, ev, lhs, rhs, int(check.get("samples", DEFAULT_SAMPLES)), float(check.get("tol", DEFAULT_TOL)))
    return detail


def _numeric_compare(detail: dict[str, Any], ev: Evaluator, lhs: Any, rhs: Any, samples: int, tol: float) -> None:
    if not ev.sampleable():
        detail["numeric"] = "skipped"
        detail.setdefault("reason", "function symbol cannot be sampled")
        return
    worst = 0.0
    try:
        for values in ev.samples(samples):
            a, b = ev.evaluate(lhs, values), ev.evaluate(rhs, values)
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            worst = max(worst, abs(a - b) / (1.0 + abs(b)))
    except Exception as error:  # noqa: BLE001
        detail["numeric"] = "skipped"
        detail["reason"] = f"numeric eval error: {error}"
        return
    detail["max_rel_err"] = worst
    detail["numeric"] = "pass" if worst <= tol else "fail"


def run_limit(check: dict[str, Any], ev: Evaluator) -> dict[str, Any]:
    detail: dict[str, Any] = {"type": "limit", "expr": check.get("expr"), "var": check.get("var"), "point": check.get("point"), "expected": check.get("expected")}
    if not HAVE_SYMPY:
        detail["symbolic"] = "skipped"
        detail["reason"] = "sympy-not-installed"
        return detail
    try:
        expr = ev.parse(str(check.get("expr", "")))
        var = ev.sym.get(str(check.get("var")))
        if var is None:
            detail["symbolic"] = "fail"
            detail["reason"] = "limit var is not a declared symbol"
            return detail
        point = ev.parse(str(check.get("point", "0")))
        expected = ev.parse(str(check.get("expected", "0")))
        direction = str(check.get("dir", "+"))
        result = sp.limit(expr, var, point, dir=direction)
        detail["result"] = str(result)
        detail["symbolic"] = "pass" if sp.simplify(result - expected) == 0 else "fail"
    except Exception as error:  # noqa: BLE001
        detail["symbolic"] = "fail"
        detail["reason"] = f"sympy error: {error}"
    return detail


def run_inequality(check: dict[str, Any], ev: Evaluator) -> dict[str, Any]:
    relation = str(check.get("relation", "<="))
    detail: dict[str, Any] = {"type": "inequality", "lhs": check.get("lhs"), "rhs": check.get("rhs"), "relation": relation}
    ops = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}
    if relation not in ops:
        detail["numeric"] = "fail"
        detail["reason"] = "relation must be one of <=, <, >=, >"
        return detail
    if not ev.sampleable():
        detail["numeric"] = "skipped"
        detail["reason"] = "function symbol cannot be sampled"
        return detail
    try:
        lhs = ev.parse(str(check.get("lhs"))) if HAVE_SYMPY else str(check.get("lhs"))
        rhs = ev.parse(str(check.get("rhs"))) if HAVE_SYMPY else str(check.get("rhs"))
        violations = []
        for values in ev.samples(int(check.get("samples", DEFAULT_SAMPLES))):
            a, b = ev.evaluate(lhs, values), ev.evaluate(rhs, values)
            if not ops[relation](a, b):
                violations.append({"input": values, "lhs": a, "rhs": b})
        detail["violations"] = violations[:5]
        detail["numeric"] = "fail" if violations else "pass"
    except Exception as error:  # noqa: BLE001
        detail["numeric"] = "skipped"
        detail["reason"] = f"numeric eval error: {error}"
    return detail


def run_numeric(check: dict[str, Any], ev: Evaluator) -> dict[str, Any]:
    detail: dict[str, Any] = {"type": "numeric"}
    tol = float(check.get("tol", DEFAULT_TOL))
    samples = int(check.get("samples", DEFAULT_SAMPLES))
    if "expr" in check:
        lhs_s, rhs_s = str(check["expr"]), "0"
        detail["expr"] = check["expr"]
    else:
        lhs_s, rhs_s = str(check.get("lhs", "")), str(check.get("rhs", "0"))
        detail["lhs"], detail["rhs"] = check.get("lhs"), check.get("rhs")
    try:
        lhs = ev.parse(lhs_s) if HAVE_SYMPY else lhs_s
        rhs = ev.parse(rhs_s) if HAVE_SYMPY else rhs_s
    except Exception as error:  # noqa: BLE001
        detail["numeric"] = "fail"
        detail["reason"] = f"parse error: {error}"
        return detail
    _numeric_compare(detail, ev, lhs, rhs, samples, tol)
    return detail


RUNNERS = {"identity": run_identity, "limit": run_limit, "inequality": run_inequality, "numeric": run_numeric}


# ---------------------------------------------------------------- 집계


def aggregate(details: list[dict[str, Any]], key: str) -> str:
    states = [d.get(key) for d in details if key in d]
    if any(state == "fail" for state in states):
        return "fail"
    if any(state == "pass" for state in states):
        return "pass"
    return "skipped"


def verify_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    front, _ = split_frontmatter(text)
    result: dict[str, Any] = {
        "file": str(path),
        "question": front.get("question"),
        "attempt": front.get("attempt"),
        "sympy": HAVE_SYMPY,
    }
    if "_frontmatter_error" in front:
        result.update({"symbolic": "skipped", "numeric": "skipped", "reason": front["_frontmatter_error"], "details": []})
        return result
    checks = front.get("verify")
    if not isinstance(checks, list) or not checks:
        result.update({"symbolic": "skipped", "numeric": "skipped", "reason": "no verify block", "details": []})
        return result
    ev = Evaluator(parse_symbols(front.get("symbols")))
    details: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            details.append({"index": index, "type": None, "symbolic": "fail", "reason": "check is not a mapping"})
            continue
        runner = RUNNERS.get(str(check.get("type", "")))
        if runner is None:
            details.append({"index": index, "type": check.get("type"), "symbolic": "fail", "reason": "unknown check type"})
            continue
        detail = runner(check, ev)
        detail["index"] = index
        details.append(detail)
    result["details"] = details
    result["symbolic"] = aggregate(details, "symbolic")
    result["numeric"] = aggregate(details, "numeric")
    if not HAVE_SYMPY:
        result["reason"] = "sympy-not-installed"
    return result


def artifacts_dir(front_question: Any, front_attempt: Any, root: Path) -> Path | None:
    if not front_question or front_attempt is None:
        return None
    try:
        attempt = int(front_attempt)
    except (TypeError, ValueError):
        return None
    return root / "verify" / str(front_question) / f"attempt-{attempt:02d}"


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = [a for a in argv if not a.startswith("--")]
    write = "--no-write" not in argv
    if not args:
        print("usage: verify_derivation.py <derivation.md> [--no-write]", file=sys.stderr)
        return 2
    root = repo_root()
    path = Path(args[0])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        print(json.dumps({"symbolic": "skipped", "numeric": "skipped", "reason": f"no such file: {path}"}))
        return 0
    result = verify_file(path)
    if write:
        target = artifacts_dir(result.get("question"), result.get("attempt"), root)
        if target is not None:
            target.mkdir(parents=True, exist_ok=True)
            (target / "hook_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            result["artifacts"] = str(target.relative_to(root)).replace("\\", "/")
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
