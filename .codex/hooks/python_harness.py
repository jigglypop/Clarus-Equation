from __future__ import annotations

import json
import importlib.metadata
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMP_PREFIX = "clarus-pytest-"


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Keep finite NumPy/SciPy-style test jobs within the available worker-memory
    # budget. This is child-process resource determinism, not a test setting.
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + existing if existing else str(REPO_ROOT)
    )
    return env


def _validate_interpreter() -> None:
    normalized = str(Path(sys.executable).resolve()).replace("\\", "/").casefold()
    if sys.version_info < (3, 10):
        raise RuntimeError("CE requires a policy-allowed Python >=3.10.")
    if "/.venv/" in normalized or "/uv/python/" in normalized:
        raise RuntimeError(
            "Refusing a workspace or uv-managed interpreter after Application Control rejection."
        )


def _doctor(env: dict[str, str]) -> int:
    import numpy
    import pytest

    def optional_version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    report = {
        "status": "PASS",
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "stdin_is_tty": bool(sys.stdin.isatty()),
        "bytecode_disabled": bool(sys.dont_write_bytecode),
        "pythonpath": env["PYTHONPATH"],
        "versions": {
            "torch": optional_version("torch"),
            "numpy": getattr(numpy, "__version__", "present"),
            "pytest": getattr(pytest, "__version__", "present"),
        },
        "repository_root": str(REPO_ROOT),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def _run_python(arguments: list[str], env: dict[str, str]) -> int:
    if not arguments:
        raise RuntimeError("python mode requires Python arguments.")
    return subprocess.run([sys.executable, "-B", *arguments], env=env, check=False).returncode


def _run_harness(env: dict[str, str]) -> int:
    checker = REPO_ROOT / ".codex" / "hooks" / "repository_harness.py"
    return subprocess.run(
        [sys.executable, "-B", str(checker)], env=env, check=False
    ).returncode


def _run_source(arguments: list[str]) -> int:
    """Parse Python sources in memory without creating bytecode or cache files."""

    targets = [Path(value) for value in arguments] or [
        REPO_ROOT / ".codex",
        REPO_ROOT / "tests",
        REPO_ROOT / "examples",
        REPO_ROOT / "experiments",
        REPO_ROOT / "paper",
    ]
    files: list[Path] = []
    for target in targets:
        resolved = target if target.is_absolute() else REPO_ROOT / target
        if resolved.is_dir():
            files.extend(resolved.rglob("*.py"))
        elif resolved.suffix == ".py" and resolved.is_file():
            files.append(resolved)
        else:
            raise RuntimeError(f"source target is not a Python file or directory: {target}")
    for path in sorted(set(files)):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(json.dumps({"status": "PASS", "python_files": len(set(files))}, sort_keys=True))
    return 0


def _run_pytest(arguments: list[str], env: dict[str, str]) -> int:
    if any(arg == "--basetemp" or arg.startswith("--basetemp=") for arg in arguments):
        raise RuntimeError("The CE harness owns --basetemp; remove the caller value.")

    temp_root = Path(tempfile.gettempdir()).resolve()
    temp_path = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=temp_root)).resolve()
    exit_code = 2
    try:
        command = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            *arguments,
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(temp_path),
        ]
        exit_code = subprocess.run(command, env=env, check=False).returncode
    finally:
        if temp_path.parent != temp_root or not temp_path.name.startswith(TEMP_PREFIX):
            raise RuntimeError(f"Refusing to remove unexpected pytest path: {temp_path}")
        if temp_path.exists():
            shutil.rmtree(temp_path)
    return exit_code


def main(arguments: list[str]) -> int:
    _validate_interpreter()
    env = _child_environment()
    os.environ.update(env)
    source = str(REPO_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    mode = arguments[0] if arguments else "doctor"
    forwarded = arguments[1:]
    if mode == "doctor":
        if forwarded:
            raise RuntimeError("doctor mode takes no arguments.")
        return _doctor(env)
    if mode == "harness":
        if forwarded:
            raise RuntimeError("harness mode takes no arguments.")
        return _run_harness(env)
    if mode == "python":
        return _run_python(forwarded, env)
    if mode == "source":
        return _run_source(forwarded)
    if mode == "pytest":
        return _run_pytest(forwarded, env)
    raise RuntimeError(
        f"Unknown mode {mode!r}; expected doctor, harness, source, python, or pytest."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"CE Python harness: {error}", file=sys.stderr)
        raise SystemExit(2) from error
