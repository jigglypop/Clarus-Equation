from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEX_ROOT = ROOT / ".codex"


def _toml(relative_path: str) -> dict:
    with (CODEX_ROOT / relative_path).open("rb") as stream:
        return tomllib.load(stream)


def test_root_and_delegated_models_are_explicit() -> None:
    config = _toml("config.toml")

    assert config["model"] == "gpt-6-astra"
    assert config["model_reasoning_effort"] == "medium"
    assert config["agents"]["default_subagent_model"] == "gpt-5.6-sol"
    assert config["agents"]["default_subagent_reasoning_effort"] == "low"

    worker = _toml("agents/worker.toml")
    assert worker["model"] == "gpt-5.6-sol"
    assert worker["model_reasoning_effort"] == "low"


def test_codex_surface_stays_compact() -> None:
    files = {
        path.relative_to(CODEX_ROOT).as_posix()
        for path in CODEX_ROOT.rglob("*")
        if path.is_file()
    }

    assert files == {"README.md", "agents/worker.toml", "config.toml"}


def test_active_instructions_do_not_restore_retired_harnesses() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    codex_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(CODEX_ROOT.rglob("*"))
        if path.is_file()
    )

    assert len(instructions.encode("utf-8")) <= 4096
    for retired in ("CE_RUN", "ce-research", "reality_stone", ".codex/hooks"):
        assert retired not in instructions
        assert retired not in codex_docs
