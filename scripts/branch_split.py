"""Domain branch splitter for the CE repository.

Reads `git ls-files` from the current branch, categorizes every tracked file
into either CORE (kept on every domain branch) or one of five domains
(cosmology, particle, brain-agi, engineering, math-derivations), and emits
the list of files that should be `git rm` -ed when checked out on a given
domain branch.

Usage:
    python scripts/branch_split.py --domain cosmology --print-keep
    python scripts/branch_split.py --domain cosmology --print-remove
    python scripts/branch_split.py --audit       # 불일치/누락 검사
    python scripts/branch_split.py --map         # 전체 매핑 표 출력

본 모듈은 read-only 다. 실제 git rm 은 sync_core.sh 또는 사용자가 직접
실행한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DOMAINS: tuple[str, ...] = (
    "cosmology",
    "particle",
    "brain-agi",
    "engineering",
    "math-derivations",
)


def _git_ls_files() -> list[str]:
    """git ls-files (UTF-8 그대로, octal escape 없음)."""
    out = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
        encoding="utf-8",
    )
    return [s for s in out.split("\0") if s]


# --------------------------------------------------------------------------- #
# 분류 규칙
# --------------------------------------------------------------------------- #

def is_core(path: str) -> bool:
    """모든 도메인 브랜치 공통."""
    if path in {
        ".gitignore", "README.md", "pyproject.toml", "uv.lock", "init.sh",
        "scripts/branch_split.py", "scripts/sync_core.sh",
    }:
        return True
    if path in {
        "docs/README.md", "docs/axium.md",
        "docs/상수.md", "docs/경로적분.md",
    }:
        return True
    if path.startswith("docs/3_상수/"):
        return True
    if path.startswith("docs/참조/"):
        return True
    if path in {
        "clarus/__init__.py",
        "clarus/engine.py",
        "clarus/ce_ops.py",
        "clarus/quantum.py",
    }:
        return True
    return False


def domain_of(path: str) -> str | None:
    """비-CORE 파일의 도메인. 어느 도메인에도 안 맞으면 None."""

    # ------- cosmology -------
    cosmology_docs = {
        "docs/2_경로적분과_응용/05_인플레이션.md",
        "docs/2_경로적분과_응용/08_바리온_비대칭.md",
        "docs/2_경로적분과_응용/12_전이구간.md",
        "docs/4_공학적_활용/03_진공에너지.md",
        "docs/5_유도/04_Dark_Energy_Derivation.md",
        "docs/5_유도/07_Black_Hole_Derivation.md",
    }
    cosmology_examples = {
        "examples/physics/cosmology.py",
        "examples/physics/check_dynamic_de.py",
        "examples/physics/check_dark_matter_paper.py",
        "examples/physics/scorecard.py",
        "examples/physics/transition_correction.py",
    }
    if path in cosmology_docs or path in cosmology_examples:
        return "cosmology"

    # ------- particle -------
    particle_docs = {
        "docs/2_경로적분과_응용/01_차원의_유일성.md",
        "docs/2_경로적분과_응용/02_에스컬레이터.md",
        "docs/2_경로적분과_응용/03_자유매개변수.md",
        "docs/2_경로적분과_응용/04_해결한_난제.md",
        "docs/2_경로적분과_응용/06_강한_CP.md",
        "docs/2_경로적분과_응용/07_중성미자_질량.md",
        "docs/2_경로적분과_응용/09_페르미온_질량.md",
        "docs/2_경로적분과_응용/10_공리_정당화.md",
        "docs/2_경로적분과_응용/11_게이지_격자와_인과성.md",
        "docs/2_경로적분과_응용/13_위상공간.md",
    }
    particle_examples = {
        "examples/physics/ckm_derivation.py",
        "examples/physics/xi_derivation.py",
        "examples/physics/higgs_mass.py",
        "examples/physics/fermion_mass.py",
        "examples/physics/neutrino_mass.py",
        "examples/physics/baryon_inertia.py",
        "examples/physics/check_unification.py",
        "examples/physics/check_open_problems.py",
        "examples/physics/check_muon_g2_integral.py",
        "examples/physics/d0_full_verification.py",
        "examples/physics/d0_origin.py",
        "examples/physics/rho_diagnosis.py",
    }
    if path in particle_docs or path in particle_examples:
        return "particle"

    # ------- engineering -------
    engineering_docs = {
        "docs/4_공학적_활용/01_핵융합_설계.md",
        "docs/4_공학적_활용/02_양자오류보정.md",
        "docs/4_공학적_활용/04_이론적_한계.md",
        "docs/4_공학적_활용/05_초전도체_설계.md",
    }
    engineering_examples = {
        "examples/physics/fusion_trigger_check.py",
        "examples/biology/cancer_mismatch.py",
        "examples/biology/sfe_protein_folding.py",
    }
    if path in engineering_docs or path in engineering_examples:
        return "engineering"
    if path.startswith("quant/"):
        return "engineering"

    # ------- math-derivations -------
    math_docs = {
        "docs/5_유도/01_Navier_Stokes.md",
        "docs/5_유도/02_Riemann_Zeta_Derivation.md",
        "docs/5_유도/03_Protein_Folding_Derivation.md",
        "docs/5_유도/05_Neural_RealityStone_Derivation.md",
        "docs/5_유도/06_Master_Action_Universal_Derivation.md",
    }
    if path in math_docs or path.startswith("docs/8_리만/"):
        return "math-derivations"

    # ------- brain-agi (catch-all for runtime/AI/brain code) -------
    if path.startswith("docs/1_강의/"):
        return "brain-agi"
    if path.startswith("docs/6_뇌/"):
        return "brain-agi"
    if path.startswith("docs/7_AGI/"):
        return "brain-agi"
    if path.startswith("clarus/"):
        return "brain-agi"
    if path.startswith("examples/ai/"):
        return "brain-agi"
    if path.startswith("scripts/"):
        return "brain-agi"
    if path.startswith("tests/"):
        return "brain-agi"
    if path in {
        "examples/physics/brain_cosmos.py",
        "examples/physics/brain_pruning.py",
        "examples/physics/sleep_cycle_analysis.py",
        "examples/physics/sleep_rho_fit.py",
        "examples/physics/sleep_rho_results.json",
    }:
        return "brain-agi"

    # ------- data/ + examples/results/ : 실험 산출물, 도메인별로 분배 -------
    if path.startswith("data/"):
        if "fusion" in path or "quantum_noise" in path or "decoupling" in path:
            return "engineering"
        if "sfe" in path or "sweep" in path:
            return "engineering"
        return "brain-agi"
    if path.startswith("examples/results/"):
        if any(k in path for k in (
            "fusion", "decoupling", "sfe", "quantum",
            "suppression", "lyapunov", "rust_engine",
            "filter", "noise", "duration_sweep", "geometric",
            "error_reduction", "state_evolution",
        )):
            return "engineering"
        return "brain-agi"
    if path == "examples/physics/sleep_rho_results.json":
        return "brain-agi"

    return None


def files_to_keep(domain: str, all_files: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (keep, remove, unmapped) lists for the given domain."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain!r} not in {DOMAINS}")

    keep: list[str] = []
    remove: list[str] = []
    unmapped: list[str] = []

    for path in all_files:
        if is_core(path):
            keep.append(path)
            continue
        d = domain_of(path)
        if d is None:
            unmapped.append(path)
            keep.append(path)         # fail-safe : 분류 못 한 파일은 모두 보존
            continue
        if d == domain:
            keep.append(path)
        else:
            remove.append(path)
    return keep, remove, unmapped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cmd_audit(all_files: list[str]) -> int:
    unmapped: list[str] = []
    multi_assigned: list[str] = []
    counts = {d: 0 for d in DOMAINS}
    counts["CORE"] = 0
    for p in all_files:
        if is_core(p):
            counts["CORE"] += 1
            continue
        d = domain_of(p)
        if d is None:
            unmapped.append(p)
        else:
            counts[d] += 1
    print("== file count by bucket ==")
    print(f"  total tracked  : {len(all_files)}")
    print(f"  CORE           : {counts['CORE']}")
    for d in DOMAINS:
        print(f"  {d:<17}: {counts[d]}")
    print(f"  unmapped       : {len(unmapped)}")
    if unmapped:
        print()
        print("== unmapped (will be kept everywhere as fail-safe) ==")
        for p in unmapped:
            print(f"  {p}")
    return 0 if not unmapped else 0


def cmd_map(all_files: list[str]) -> int:
    print("path,bucket")
    for p in all_files:
        if is_core(p):
            bucket = "CORE"
        else:
            bucket = domain_of(p) or "unmapped"
        print(f"{p},{bucket}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--print-keep", action="store_true")
    parser.add_argument("--print-remove", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args(argv)

    all_files = _git_ls_files()

    if args.audit:
        return cmd_audit(all_files)
    if args.map:
        return cmd_map(all_files)
    if not args.domain:
        parser.error("--domain required (or use --audit / --map)")

    keep, remove, unmapped = files_to_keep(args.domain, all_files)
    if args.print_keep:
        for p in keep:
            print(p)
    elif args.print_remove:
        for p in remove:
            print(p)
    else:
        print(f"== domain : {args.domain} ==")
        print(f"keep    : {len(keep)}")
        print(f"remove  : {len(remove)}")
        if unmapped:
            print(f"unmapped: {len(unmapped)} (kept as fail-safe)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
