"""Pin the historical successful formula branches without restoring deleted code.

Run from any directory. Archived files are evidence, not imported runtime modules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = [
    ("recursive_cosmology", "210f6f33^", "examples/physics/recursive_cosmology_predictions.py", None),
    ("muon_integral", "210f6f33^", "examples/physics/check_muon_g2_integral.py", None),
    ("holographic_scale", "210f6f33", "examples/physics/cosmological_constant_holographic_gate.py", None),
    ("hubble_flow", "e073bc90", "examples/physics/hubble_tension.py", None),
    ("hubble_numerics", "e073bc90", "examples/physics/cosmology.py", None),
    ("dark_energy_eos", "210f6f33", "examples/physics/xi_derivation.py", None),
    ("paper_dark_split", "210f6f33", "docs/경로적분.md", (338, 369)),
    ("paper_muon", "210f6f33", "docs/경로적분.md", (548, 675)),
    ("paper_muon_finite", "210f6f33", "docs/경로적분.md", (954, 983)),
]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> None:
    target = HERE / "sources"
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, rev, path, span in SOURCES:
        commit = git("rev-parse", rev).decode().strip()
        blob = git("rev-parse", f"{commit}:{path}").decode().strip()
        raw = git("show", f"{commit}:{path}")
        saved = raw if span is None else b"".join(raw.splitlines(keepends=True)[span[0]-1:span[1]])
        dest = target / f"{name}.txt"
        dest.write_bytes(saved)
        manifest.append({
            "id": name, "commit": commit, "original_path": path,
            "git_blob": blob, "original_sha256": hashlib.sha256(raw).hexdigest(),
            "original_lines": list(span) if span else [1, len(raw.splitlines())],
            "saved_path": dest.relative_to(HERE).as_posix(),
            "saved_sha256": hashlib.sha256(saved).hexdigest(),
        })
    (HERE / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Pinned {len(manifest)} historical sources; deleted runtime paths remain untouched.")


if __name__ == "__main__":
    main()
