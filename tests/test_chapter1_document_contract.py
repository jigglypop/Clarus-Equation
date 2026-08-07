from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "docs" / "1_강의"
FILES = {
    "A": CHAPTER / "A_연역적_유도.md",
    "B": CHAPTER / "B_귀납적_유도.md",
    "C": CHAPTER / "C_다섯_상수.md",
    "D": CHAPTER / "D_정합성_원장.md",
}


def _read(key: str) -> str:
    return FILES[key].read_text(encoding="utf-8")


def _section(text: str, heading: str, next_heading: str) -> str:
    start = text.index(heading)
    end = text.index(next_heading, start + len(heading))
    return text[start:end]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", path.read_text(encoding="utf-8")):
        plain = re.sub(r"[`*_~]", "", heading).lower()
        plain = re.sub(r"[^\w\s-]", "", plain)
        base = re.sub(r"\s+", "-", plain.strip())
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _local_markdown_links(path: Path) -> list[tuple[Path, str]]:
    text = path.read_text(encoding="utf-8")
    targets: list[tuple[Path, str]] = []
    for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = raw.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        raw_path, _, raw_fragment = target.partition("#")
        linked_path = path if not raw_path else (path.parent / unquote(raw_path)).resolve()
        targets.append((linked_path, unquote(raw_fragment)))
    return targets


def test_required_chapter_files_are_present_and_utf8_clean() -> None:
    for path in FILES.values():
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# ")
        assert len(re.findall(r"(?m)^# ", text)) == 1
        assert not any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)


def test_markdown_fences_math_blocks_and_local_links_are_balanced() -> None:
    missing: list[str] = []
    for path in FILES.values():
        text = path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, path
        assert text.count("$$") % 2 == 0, path
        for target, fragment in _local_markdown_links(path):
            if not target.exists():
                missing.append(f"{path.name} -> {target}")
            elif fragment and fragment not in _markdown_anchors(target):
                missing.append(f"{path.name} -> {target.name}#{fragment}")
    assert missing == []


def test_markdown_tables_have_stable_column_counts() -> None:
    for path in FILES.values():
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
            counts = [row.count("|") for row in block]
            assert len(set(counts)) == 1, (path, block)


def test_claim_status_vocabulary_is_shared() -> None:
    a = _read("A")
    for status in (
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
    ):
        assert status in a
    statuses = (
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
    for key in ("B", "C"):
        text = _read(key)
        assert "A 강의의 지위 계약" in text
        for status in statuses:
            assert status in text, (key, status)


def test_survival_and_energy_namespaces_are_explicit() -> None:
    for key in ("A", "B", "C", "D"):
        text = _read(key)
        for label in ("S1", "S2", "S3", "S4", "E1", "E2", "E3", "E4"):
            assert label in text, (key, label)
        assert r"x_E:[0,\infty)\to(0,1]" in text
        assert r"x_E(0)=1" in text
    c = _read("C")
    assert r"S:[0,\infty)\to(0,1]" in c
    assert "어떤 $D_0>0$에서 $S(D_0)<1$" in c
    assert "단위율 $\\kappa_{\\rm surv}=1$" in c
    for key in ("A", "B", "C"):
        assert r"\widetilde S(\widetilde D)" in _read(key)


def test_typed_symbols_do_not_regress_to_ambiguous_identifications() -> None:
    combined = "\n".join(_read(key) for key in FILES)
    forbidden = (
        r"\boxed{R=",
        r"\delta_N",
        r"s_N^2=s_A^2",
        "추가 metric 없이 자연스러운",
        r"\frac12(M_P^2+\xi\phi^2)",
        r"-\frac{\lambda_{H\phi}}2\phi^2H^\dagger H",
        r"e^{-\chi\sqrt A}",
        r"e^{-sA}",
        r"e^{-\ell_P\sqrt A}",
        r"e^{-s_HA}",
        r"e^{-iHt}",
        r"e^{-i\widehat{\mathcal H}t}",
        r"e^{-S_E}",
        r"\Phi_H",
        r"\int\mathcal D\varphi\,e^{iS[\varphi]/\hbar}",
        r"\frac{\delta S}{\delta\varphi}",
        r"\alpha_s=0.1173186647",
        r"D_{\rm act}=(1-x_E)D_A",
        r"\alpha_{\rm total}",
        "$mu_D$",
    )
    for token in forbidden:
        assert token not in combined, token

    a = _read("A")
    for token in (
        r"R_g:=R[g]",
        r"R_{\rm dark}",
        r"q_E",
        r"A_E\ge0",
        r"x_\star",
        r"x_E",
        r"\delta_{\rm proj}",
        r"\delta_A",
        r"\delta_{\rm fold}",
        r"\kappa_{\rm surv}",
        r"\kappa_E",
        r"\beta_E:=\kappa_E c_E",
        r"\Omega_{\rm phys}:=\rho_{\rm tot}/\rho_c",
    ):
        assert token in a

    c = _read("C")
    assert r"e^{-\ell_P\sqrt {A_E}}" in c
    assert r"e^{-s_HA_E}" in c
    assert r"\int\mathcal D\Psi\,e^{iS[\Psi]/\hbar}" in c
    assert r"\frac{\delta S}{\delta\Psi}" in c
    assert r"e^{-i\widehat{\mathcal H}t/\hbar}" in c
    assert r"e^{-S_E/\hbar}" in c
    assert r"\widetilde S(\widetilde D)=e^{-\widetilde D}" in c
    assert r"Z_\phi=1$, $c_i=0" in c
    assert r"\alpha_{s,{\rm SM}}^{(B)}=0.1173186647" in _read("B")
    assert (
        r"C_{\rm CE}:=\alpha_s+\alpha_w+\alpha_{em}"
        in _read("D")
    )
    for key in FILES:
        text = _read(key)
        assert re.search(
            r"C_\{\\rm CE\}.{0,300}`Selection`",
            text,
            re.DOTALL,
        ), key


def test_hodge_hypotheses_and_anomaly_checks_are_complete() -> None:
    a = _section(_read("A"), "## 4. $d=3$ 선택 정리", "## 5.")
    for token in (
        "양의 정부호 내적",
        "orientation",
        "$SO(d)$",
        r"*:\Lambda^2V^*\longrightarrow\Lambda^{d-2}V^*",
        "차원 일치만으로는 자연스러운 동형이 생기지 않는다",
        r"[SU(3)]^3",
        r"[SU(3)]^2U(1)",
        r"[SU(2)]^2U(1)",
        r"[U(1)]^3",
        r"[\mathrm{grav}]^2U(1)",
        "pseudoreality",
        "Witten global anomaly",
    ):
        assert token in a

    compact = _compact(a)
    for equation in (
        r"d-2=1",
        r"[SU(3)]^3&:2-1-1=0",
        r"[SU(3)]^2U(1)&:2\left(\frac16\right)\frac12-\frac23\frac12+\frac13\frac12=0",
        r"[SU(2)]^2U(1)&:3\left(\frac16\right)\frac12-\frac12\frac12=0",
        r"[U(1)]^3&:6\left(\frac16\right)^3+3\left(-\frac23\right)^3+3\left(\frac13\right)^3+2\left(-\frac12\right)^3+1^3=0",
        r"[\mathrm{grav}]^2U(1)&:6\left(\frac16\right)+3\left(-\frac23\right)+3\left(\frac13\right)+2\left(-\frac12\right)+1=0",
    ):
        assert equation in compact, equation


def test_core_and_inflation_branches_use_distinct_fields_and_couplings() -> None:
    a = _read("A")
    core = _section(a, "## 2. 공변 EFT", "## 3.")
    inflation = _section(a, "## 11.", "## 12.")

    for token in (r"F_{\rm core}", r"\xi_{\rm core}", r"R_g", r"\phi"):
        assert token in core
    assert r"Z_\phi>0" in core
    assert r"\xi_{\rm inf}" not in core

    for token in (
        r"F_{\rm inf}",
        r"\xi_{\rm inf}",
        r"\varphi",
        r"\chi_E",
        r"R_J:=R[g_J]",
    ):
        assert token in inflation
    assert re.search(r"\\xi(?!_)", inflation) is None
    assert r"\phi" not in inflation
    assert re.search(r"\\chi(?!_)", inflation) is None
    assert r"F_{\rmcore}(\phi)=M_P^2-\xi_{\rmcore}\phi^2" in _compact(core)
    assert r"F_{\rminf}(\varphi)=M_P^2+\xi_{\rminf}\varphi^2" in _compact(inflation)


def test_vector_scalar_and_energy_bridge_conditions_are_not_hidden() -> None:
    for key in ("A", "B", "C", "D"):
        text = _read(key)
        assert "공통 행합" in text
        assert "next-generation" in text
        assert r"x_\star" in text
        assert r"x_E" in text
    a = _read("A")
    assert r"\mathcal P(F_{\mathsf K}(\boldsymbol x))" in a
    assert (
        r"\mathcalP(F_{\mathsfK}(\boldsymbolx))\neF_D(\mathcalP(\boldsymbolx))"
        in _compact(a)
    )
    assert "정의에 의한 항등식" in _read("C")

    energy_sections = {
        "A": _section(_read("A"), "### 8.2", "## 9."),
        "B": _section(_read("B"), "이제 수학적 근", "## 4."),
        "C": _section(_read("C"), "## 8.", "## 9."),
        "D": _section(_read("D"), "## 4.", "## 5."),
    }
    for key, section in energy_sections.items():
        assert "Bridge" in section and "Open" in section, key
        assert r"\kappa_E" in section, key
        compact = _compact(section)
        assert r"D_{\rmact}=c_E(1-x_E)D_A" in compact, key
        assert r"\beta_E:=\kappa_Ec_E" in compact, key
        assert r"\beta_E=1" in compact, key
        assert re.search(r"\\beta_E=1.{0,350}`Open Bridge`", section, re.DOTALL), key

    positive_measure = energy_sections["A"]
    for token in (
        r"0<Z_D",
        r"d\mu_D",
        "gauge fixing",
        "regularization",
        "sign/phase problem",
    ):
        assert token in positive_measure

    for key in ("C", "D"):
        text = _read(key)
        assert re.search(r"공통\s*행합.{0,100}`Open Bridge`", text, re.DOTALL)

    track_b_sections = {
        "A": _section(_read("A"), "### 5.2", "## 6."),
        "B": _section(_read("B"), "## 6.", "## 7."),
        "D": _section(_read("D"), "## 6.", "## 7."),
    }
    for key, section in track_b_sections.items():
        assert r"s_{W,B}^2:=4(\alpha_s^{(B)})^{4/3}" in section, key
        assert r"\alpha_{w,B}" in section, key
        assert "Bridge" in section and "Open" in section, key
        assert r"\alpha_s>\alpha_w>\alpha_{em}" not in section, key


def test_portal_normalization_and_complete_density_ledger_are_explicit() -> None:
    for key in ("A", "C"):
        text = _read(key)
        assert r"-\lambda_{\rm HP}\phi^2H^\dagger H" in text
        assert r"\lambda^{(1/2)}_{H\phi}=2\lambda_{\rm HP}" in text
    for key in FILES:
        text = _read(key)
        for token in (
            r"\Omega_r",
            r"\Omega_\nu",
            r"\Omega_k",
            r"\Omega_{\rm phys}",
        ):
            assert token in text
        assert re.search(r"late-time\s+truncated", text)
        compact = _compact(text)
        assert r"1=\Omega_{\rmphys}+\Omega_k" in compact, key
        assert r"1=\Omega_{\rmphys}-\Omega_k" not in compact, key
        assert (
            r"\Omega_{\rmrem}:=\Omega_{\rmphys}-\Omega_b-\Omega_r-\Omega_\nu"
            in compact
        ), key
        assert r"\alpha_sD_A(1+x_\star\delta_A)" in compact, key
        assert (
            r"\Omega_{\rmcdm}=\Omega_{\rmrem}\frac{R_{\rmdark}}{1+R_{\rmdark}}"
            in compact
        ), key
        assert (
            r"\Omega_{\rmDE}=\Omega_{\rmrem}\frac1{1+R_{\rmdark}}"
            in compact
        ), key

    inflation = _section(_read("A"), "## 11.", "## 12.")
    assert r"-\lambda_{H\varphi}\varphi^2H^\dagger H" in inflation
    assert "별도 입력" in inflation
    assert "direct SM portal" in inflation
    assert "perturbative decay·annihilation rate는 0" in inflation
    assert "gravitational particle production" in inflation

    for key in ("C", "D"):
        compact = _compact(_read(key))
        assert r"F_{\rmcore}(\phi)=M_P^2-\xi_{\rmcore}\phi^2" in compact, key
        assert r"F_{\rminf}(\varphi)=M_P^2+\xi_{\rminf}\varphi^2" in compact, key


def test_canonical_track_a_ledger_equations_are_fingerprinted() -> None:
    ledger = _compact(_read("D"))
    for equation in (
        r"s_A^2:=4\alpha_s^{4/3}=0.2315097758079336",
        r"\delta_A:=s_A^2(1-s_A^2)=0.17791299951329392",
        r"D_A:=3+\delta_A=3.177912999513294",
        r"x=e^{-D_A(1-x)}",
        r"x_\star=-\frac{W_0(-D_Ae^{-D_A})}{D_A}=0.04863825851598631",
        r"R_{\rmdark}:=\alpha_sD_A(1+x_\star\delta_A)=0.3782386966438831",
    ):
        assert equation in ledger, equation


def test_declared_cosmology_gate_scope_is_not_overstated() -> None:
    for key in ("B", "C", "D"):
        text = _read(key)
        assert re.search(r"BAO-only\s+partial\s+gate", text), key
        assert "DESI DR2" in text, key
        assert "CMB·SN·growth" in text, key
        assert "`Open`" in text, key
        assert "`Rejected`" in text, key
    for key in ("B", "D"):
        text = _read(key)
        compact = _compact(text)
        assert re.search(r"untouched\s+holdout", text), key
        assert "4-sector EH" in text, key
        assert r"p<0.0027" in compact, key
        assert "적합" in text and "0개" in text and "dof" in text, key
        assert r"\Omega_{{\rmrad},0}^{({\rmEH})}(1+z)^4" in compact, key
    assert "sound horizon과 BAO 거리 모두" in _read("B")
    assert "sound horizon과 BAO distance 계산은 모두" in _read("D")
    assert "--density-preset chapter1" in _read("D")
    assert "chapter1_canonical_params" in _read("D")
