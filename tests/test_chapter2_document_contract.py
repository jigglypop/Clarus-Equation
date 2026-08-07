from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "docs" / "2_경로적분과_응용"
NUMBERED = tuple(CHAPTER / f"{index:02d}_{name}.md" for index, name in (
    (1, "차원의_유일성"),
    (2, "에스컬레이터"),
    (3, "자유매개변수"),
    (4, "해결한_난제"),
    (5, "인플레이션"),
    (6, "강한_CP"),
    (7, "중성미자_질량"),
    (8, "바리온_비대칭"),
    (9, "페르미온_질량"),
    (10, "공리_정당화"),
    (11, "게이지_격자와_인과성"),
    (12, "전이구간"),
    (13, "위상공간"),
    (14, "자기재귀성_대칭"),
))
CONTRACT = CHAPTER / "00_수학적_완결성_계약.md"
PROOFS = CHAPTER / "15_보강_정리와_증명.md"
PRIMARY_STATUSES = (
    "Definition",
    "Exact",
    "Exact conditional",
    "Convention",
    "Selection",
    "Bridge",
    "Phenomenology",
    "Calibration input",
    "Calibration output",
    "Open",
    "Rejected",
)


def _markdown_files() -> tuple[Path, ...]:
    return tuple(sorted(CHAPTER.glob("*.md")))


def _local_links(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8")
    links: list[Path] = []
    for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = raw.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        raw_path, _, _ = target.partition("#")
        links.append(path if not raw_path else (path.parent / unquote(raw_path)).resolve())
    return links


def _without_markdown_code(text: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(re.sub(r"`[^`\n]*`", "", line))
    assert not in_fence
    return "".join(kept)


def _assert_math_delimiters_are_ordered(text: str, path: Path) -> None:
    source = _without_markdown_code(text)
    slash_stack: list[str] = []
    dollar_mode: str | None = None
    index = 0
    while index < len(source):
        if source.startswith(r"\(", index):
            assert not slash_stack, (path, index, "nested \\\\(")
            slash_stack.append(r"\(")
            index += 2
            continue
        if source.startswith(r"\)", index):
            assert slash_stack and slash_stack.pop() == r"\(", (path, index, r"\)")
            index += 2
            continue
        if source.startswith(r"\[", index):
            assert not slash_stack, (path, index, "nested \\\\[")
            slash_stack.append(r"\[")
            index += 2
            continue
        if source.startswith(r"\]", index):
            assert slash_stack and slash_stack.pop() == r"\[", (path, index, r"\]")
            index += 2
            continue
        if source[index] == "$" and (index == 0 or source[index - 1] != "\\"):
            token = "$$" if source.startswith("$$", index) else "$"
            if dollar_mode is None:
                dollar_mode = token
            else:
                assert dollar_mode == token, (path, index, dollar_mode, token)
                dollar_mode = None
            index += len(token)
            continue
        index += 1
    assert slash_stack == [], (path, slash_stack)
    assert dollar_mode is None, (path, dollar_mode)


def test_complete_chapter_set_is_utf8_and_structurally_balanced() -> None:
    for path in (CONTRACT, *NUMBERED, PROOFS):
        assert path.is_file(), path
        raw = path.read_bytes()
        assert b"\r" not in raw.replace(b"\r\n", b""), path
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# "), path
        assert len(re.findall(r"(?m)^# ", text)) == 1, path
        assert text.count("$$") % 2 == 0, path
        assert text.count("```") % 2 == 0, path
        assert text.count(r"\[") == text.count(r"\]"), path
        assert text.count(r"\(") == text.count(r"\)"), path
        _assert_math_delimiters_are_ordered(text, path)
        assert "\ufffd" not in text, path
        assert not any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text), path


def test_local_links_and_documented_python_commands_exist() -> None:
    missing: list[str] = []
    for path in _markdown_files():
        for target in _local_links(path):
            if not target.exists():
                missing.append(f"{path.name} -> {target}")

        text = path.read_text(encoding="utf-8")
        for command_path in re.findall(r"(?m)^python\s+([^\s]+\.py)(?:\s|$)", text):
            target = (ROOT / command_path).resolve()
            if not target.exists():
                missing.append(f"{path.name} command -> {target}")
    assert missing == []


def test_markdown_tables_keep_a_stable_column_count() -> None:
    for path in _markdown_files():
        rows = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(rows):
            if not rows[index].lstrip().startswith("|"):
                index += 1
                continue
            block: list[str] = []
            while index < len(rows) and rows[index].lstrip().startswith("|"):
                block.append(rows[index])
                index += 1
            assert len({row.count("|") for row in block}) == 1, (path, block)


def test_shared_status_and_symbol_contract_is_explicit() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    for status in PRIMARY_STATUSES:
        assert status in contract
    for token in (
        r"\(d\in\mathbb N_0\)",
        r"\tau=\kappa D",
        r"\rho(A)",
        r"q_{\rm trig}",
        r"p_{\rm trig}=1-q_{\rm trig}",
        r"x_{\rm path}",
        r"x_E",
        r"R_{\rm CE}",
        r"C_5=C_5^T",
        r"z_\Phi=\Phi/M_P",
        r"s_{\rm ent}",
        r"\mathcal V_N=\pi^N",
        "sub-Markov",
    ):
        assert token in contract
    for malformed in ("(mathcal ", "(kappa", "(alpha_s", "\nho(A)"):
        assert malformed not in contract

    for path in NUMBERED:
        text = path.read_text(encoding="utf-8")
        assert "00_수학적_완결성_계약.md" in text, path
        assert "15_보강_정리와_증명.md" in text, path

    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (*NUMBERED, PROOFS)
    )
    for forbidden_status in (
        "`Calibration`",
        "`Candidate`",
        "`Bridge/Candidate`",
        "`Conditional benchmark`",
        "`Conditional solution`",
        "`Conditional EFT`",
        "`Approximation`",
        "`Framework`",
        "`numerical example`",
        "`Phenomenological matching ansatz`",
    ):
        assert forbidden_status not in combined
    status_prefixes = tuple(status.split()[0] for status in PRIMARY_STATUSES)
    for token in re.findall(r"`([^`\n]+)`", combined):
        if token.startswith(status_prefixes):
            assert token in PRIMARY_STATUSES, token
    status_pattern = "|".join(map(re.escape, PRIMARY_STATUSES))
    assert not re.search(
        rf"`(?:{status_pattern})`\s*\+\s*`(?:{status_pattern})`",
        combined,
    )


def test_strengthened_theorem_suite_covers_each_repaired_layer() -> None:
    proofs = PROOFS.read_text(encoding="utf-8")
    for theorem in range(19):
        assert re.search(rf"(?m)^## 15\.{theorem}(?:\s|$)", proofs)
    for token in (
        "Hodge-type bivector closure",
        "two-channel kernel",
        "reducible 다형 Poisson",
        "에너지 readout",
        "neutral spectral projector",
        "open-system Lyapunov",
        "fixed-background 인과성",
        "inflation EFT",
        "QCD axion",
        "radiative neutrino",
        "wall transport BVP",
        "Koide cone",
        "polydisc와 Gaussian",
        "재매개화 불변 잔차",
        "recursion-semigroup invariant state",
        "재귀 surprisal의 유일성",
    ):
        assert token in proofs


def test_known_semantic_regressions_are_absent() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in _markdown_files())
    for forbidden in (
        r"P_N\equiv\pi^N",
        r"P_N=\pi^N",
        "gravitational_environment_readout_gate.py",
        "a3c_readout_closure_gate.py",
        "a3c_preferred_axis_no_go_gate.py",
        "a3c_conditional_axis_bridge_gate.py",
        "a3c_cmb_axis_ingest_gate.py",
        "a3c_cmb_likelihood_proxy_gate.py",
        "a3c_closure_package_gate.py",
        "flat_energy_readout_omega_b",
        "sigma ** (depth / (depth + 1.0))",
        r"D_N",
        r"\delta_N",
        r"\delta_{\rm CE}",
        r"=s_W^2(1-s_W^2)",
        r"q_D(x)=",
        r"\mathsf DK",
        r"D_iK_i",
        r"P(N=0)",
        r"P_D",
        r"P_{D,x}",
        r"\Omega_b(a_0)=x=",
        r"R=\alpha_sD_A",
        r"z:=\Phi/M_P",
        r"\lambda,\kappa,v>0",
    ):
        assert forbidden not in combined

    chapter5 = NUMBERED[4].read_text(encoding="utf-8")
    assert r"x_{\rm end}" not in chapter5
    assert r"x_*" not in chapter5
    assert r"z_{\Phi,{\rm end}}" in chapter5
    assert r"z_{\Phi,*}" in chapter5

    chapter7 = NUMBERED[6].read_text(encoding="utf-8")
    assert r"\kappa" not in chapter7
    assert r"C_5" in chapter7
    assert r"R_{\rm CI}" in chapter7

    chapter8 = NUMBERED[7].read_text(encoding="utf-8")
    assert r"Y_B=\frac{n_B}{s}" not in chapter8
    assert r"s_{\rm ent}" in chapter8

    chapter14 = NUMBERED[-1].read_text(encoding="utf-8")
    assert r"G^{\circ n}" not in chapter14
    assert "extinction" in chapter14
    assert "suppressor" in chapter14
    assert r"\kappa D" in chapter14 or r"\tau=\kappa D" in chapter14

    chapter12 = NUMBERED[11].read_text(encoding="utf-8")
    assert r"\widehat V" in chapter12
    assert "open-system" in chapter12
    assert r"h_D(x):=" in chapter12

    for path in (NUMBERED[1], NUMBERED[2], NUMBERED[9], NUMBERED[13]):
        text = path.read_text(encoding="utf-8")
        assert r"\delta_A" in text, path
        assert r"\delta_{\rm proj}" in text, path


def test_real_audit_gate_is_the_only_chapter_level_python_gate() -> None:
    gate = ROOT / "examples" / "physics" / "chapter2_mathematical_audit.py"
    tests = ROOT / "tests" / "test_chapter2_mathematical_audit.py"
    assert gate.is_file()
    assert tests.is_file()
    gate_text = gate.read_text(encoding="utf-8")
    assert "physical_bridges_validated: false" in gate_text
    assert "two_channel_path_kernel" in gate_text
    assert "regularized_fixed_point_potential" in gate_text
    assert "operational_tau" in gate_text
    assert "einstein_slope_ratio_per_c6" in gate_text
