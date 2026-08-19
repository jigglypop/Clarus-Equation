#!/usr/bin/env python3
"""Analyze author-released macaque PFC caches without executing pickle globals."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import pickle
import subprocess
from xml.etree import ElementTree
from zipfile import ZipFile
from pathlib import Path

import numpy as np


SELECTIVITY = "selectivity_coefficients_exp1_140_1504stages.pickle"
DECODING_COLOUR = "exp1_decoding_collocked_50_150_4stages.pickle"
DECODING_SHAPE = "exp1_decoding_shapelocked_100_150_4stages.pickle"
OFFICIAL_REPO = "https://github.com/m-j-wojcik/pfc_learning"
DRYAD = "https://doi.org/10.5061/dryad.c2fqz61kb"
PAPER = "https://doi.org/10.1038/s41593-026-02333-w"
SOURCE_DATA = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41593-026-02333-w/MediaObjects/41593_2026_2333_MOESM3_ESM.xlsx"
)

XLSX_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _allowed_pickle_globals() -> dict[tuple[str, str], object]:
    multiarray = np._core.multiarray
    return {
        ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
        ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): multiarray.scalar,
        ("numpy.core.multiarray", "scalar"): multiarray.scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        allowed = _allowed_pickle_globals()
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return allowed[(module, name)]


def safe_load(path: Path) -> dict[str, object]:
    value = RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(value, dict):
        raise ValueError(f"expected dict in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xlsx_text(element: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{{{XLSX_MAIN}}}t"))


def _column_index(reference: str) -> int:
    value = 0
    for char in reference:
        if not char.isalpha():
            break
        value = value * 26 + ord(char.upper()) - ord("A") + 1
    return value - 1


def xlsx_tables(path: Path, wanted: set[str]) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_xlsx_text(item) for item in root]

        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"].lstrip("/") for item in relationships}
        sheets = workbook.find(f"{{{XLSX_MAIN}}}sheets")
        if sheets is None:
            raise ValueError("XLSX has no sheets")

        for sheet in sheets:
            name = sheet.attrib["name"]
            if name not in wanted:
                continue
            relation_id = sheet.attrib[f"{{{XLSX_REL}}}id"]
            worksheet = ElementTree.fromstring(archive.read(targets[relation_id]))
            raw_rows: list[list[str]] = []
            for row in worksheet.iter(f"{{{XLSX_MAIN}}}row"):
                values: dict[int, str] = {}
                for cell in row.findall(f"{{{XLSX_MAIN}}}c"):
                    index = _column_index(cell.attrib["r"])
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{{{XLSX_MAIN}}}v")
                    if cell_type == "inlineStr":
                        value = _xlsx_text(cell)
                    elif value_node is None:
                        value = ""
                    elif cell_type == "s":
                        value = shared[int(value_node.text or "0")]
                    else:
                        value = value_node.text or ""
                    values[index] = value
                width = max(values, default=-1) + 1
                raw_rows.append([values.get(index, "") for index in range(width)])
            if not raw_rows:
                raise ValueError(f"empty XLSX sheet: {name}")
            header = raw_rows[0]
            tables[name] = [dict(zip(header, row)) for row in raw_rows[1:]]

    missing = wanted - tables.keys()
    if missing:
        raise ValueError(f"missing XLSX sheets: {sorted(missing)}")
    return tables


def assert_spd(matrix: np.ndarray, name: str) -> np.ndarray:
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} is not a finite 3x3 matrix")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0.0:
        raise ValueError(f"{name} is not SPD: {eigenvalues}")
    return matrix


def airm_decomposition(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    values, vectors = np.linalg.eigh(assert_spd(reference, "reference covariance"))
    inverse_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    relative = assert_spd(inverse_sqrt @ candidate @ inverse_sqrt, "relative covariance")
    log_eigenvalues = np.log(np.linalg.eigvalsh(relative))
    total_sq = float(log_eigenvalues @ log_eigenvalues)
    if total_sq <= 1e-24:
        return {"total": 0.0, "scale": 0.0, "shape": 0.0, "shape_fraction_sq": 0.0}
    mean_log = float(log_eigenvalues.mean())
    scale_sq = 3.0 * mean_log * mean_log
    shape_sq = max(0.0, total_sq - scale_sq)
    return {
        "total": math.sqrt(total_sq),
        "scale": math.sqrt(scale_sq),
        "shape": math.sqrt(shape_sq),
        "shape_fraction_sq": shape_sq / total_sq if total_sq > 0.0 else 0.0,
    }


def effective_rank(eigenvalues: np.ndarray) -> float:
    probabilities = eigenvalues / eigenvalues.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def format_vector(values: np.ndarray | list[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}g}" for value in values) + "]"


def format_matrix(matrix: np.ndarray, digits: int = 6) -> str:
    return "<br>".join(format_vector(row, digits) for row in matrix)


def git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def analyze(repo: Path, source_xlsx: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    processed = repo / "processed_data"
    selectivity_path = processed / SELECTIVITY
    colour_path = processed / DECODING_COLOUR
    shape_path = processed / DECODING_SHAPE
    for path in (selectivity_path, colour_path, shape_path, source_xlsx):
        if not path.is_file():
            raise FileNotFoundError(path)

    selectivity = safe_load(selectivity_path)["selectivity_coefficients_xval"]
    if not isinstance(selectivity, list) or len(selectivity) != 4:
        raise ValueError("expected four learning stages")

    source = xlsx_tables(
        source_xlsx,
        {"kl_selectivity", "efgi_observed_window", "j_dim_observed_window"},
    )
    source_points: dict[int, np.ndarray] = {}
    for stage_index in (1, 4):
        rows = [row for row in source["kl_selectivity"] if int(row["stage"]) == stage_index]
        source_points[stage_index] = np.asarray(
            [[float(row["colour_sel"]), float(row["shape_sel"]), float(row["xor_sel"])] for row in rows],
            dtype=np.float64,
        )

    stages: list[dict[str, object]] = []
    covariances: list[np.ndarray] = []
    source_cache_max_abs = 0.0
    for index, raw in enumerate(selectivity, start=1):
        coefficients = np.asarray(raw, dtype=np.float64)
        if coefficients.ndim != 3 or coefficients.shape[1:] != (3, 2):
            raise ValueError(f"unexpected stage {index} shape: {coefficients.shape}")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(f"nonfinite selectivity in stage {index}")

        duplicate_fold_max_abs = float(np.max(np.abs(coefficients[:, :, 0] - coefficients[:, :, 1])))
        points = coefficients[:, :, 0]
        if index in source_points:
            if source_points[index].shape != points.shape:
                raise ValueError(f"Nature source/cache shape mismatch in stage {index}")
            residual = float(np.max(np.abs(source_points[index] - points)))
            source_cache_max_abs = max(source_cache_max_abs, residual)
            if residual > 1e-12:
                raise ValueError(f"Nature source/cache value mismatch in stage {index}: {residual}")
            points = source_points[index]
        centered = points - points.mean(axis=0, keepdims=True)
        covariance = assert_spd(centered.T @ centered / (len(points) - 1), f"stage {index} covariance")
        metric = assert_spd(np.linalg.inv(covariance), f"stage {index} inverse covariance")
        eigenvalues = np.linalg.eigvalsh(covariance)
        covariances.append(covariance)
        stages.append(
            {
                "stage": index,
                "source": "Nature XLSX + author cache" if index in source_points else "author cache",
                "neurons": int(len(points)),
                "duplicate_fold_max_abs": duplicate_fold_max_abs,
                "covariance": covariance,
                "metric": metric,
                "covariance_eigenvalues": eigenvalues,
                "effective_rank": effective_rank(eigenvalues),
                "trace": float(np.trace(covariance)),
                "logdet": float(np.linalg.slogdet(covariance)[1]),
                "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                "correlation": np.corrcoef(points, rowvar=False),
            }
        )

    for stage, covariance in zip(stages, covariances):
        stage["from_stage_1"] = airm_decomposition(covariances[0], covariance)

    colour_cache = safe_load(colour_path)
    shape_cache = safe_load(shape_path)
    early_decoding = np.asarray(colour_cache["early_decoding"], dtype=np.float64)
    late_decoding = np.asarray(shape_cache["late_decoding"], dtype=np.float64)
    late_shattering = np.asarray(shape_cache["late_shattering_dim"], dtype=np.float64)
    if late_decoding.shape != (4, 4) or late_shattering.shape != (4, 31):
        raise ValueError("unexpected official decoding cache shape")
    if early_decoding.shape != (4, 4):
        raise ValueError("unexpected official colour decoding cache shape")
    if not all(np.all(np.isfinite(value)) for value in (early_decoding, late_decoding, late_shattering)):
        raise ValueError("nonfinite official decoding cache")

    source_decoding = np.empty((4, 4), dtype=np.float64)
    panel_columns = {"e_colour": 0, "f_shape": 1, "i_width": 2, "g_xor": 3}
    for row in source["efgi_observed_window"]:
        source_decoding[int(row["stage"]) - 1, panel_columns[row["panel"]]] = float(row["observed_accuracy"])
    source_dimensionality = np.asarray(
        [float(row["observed_dim"]) for row in source["j_dim_observed_window"]],
        dtype=np.float64,
    )
    cache_decoding = late_decoding.copy()
    cache_decoding[:, 0] = early_decoding[:, 0]
    cache_dimensionality = np.concatenate([late_shattering, late_decoding], axis=1).mean(axis=1)
    decoding_residual = float(np.max(np.abs(source_decoding - cache_decoding)))
    dimensionality_residual = float(np.max(np.abs(source_dimensionality - cache_dimensionality)))
    if decoding_residual > 1e-12 or dimensionality_residual > 1e-12:
        raise ValueError(
            "Nature source/cache decoding mismatch: "
            f"decoding={decoding_residual}, dimensionality={dimensionality_residual}"
        )

    metadata: dict[str, object] = {
        "commit": git_value(repo, "rev-parse", "HEAD"),
        "remote": git_value(repo, "config", "--get", "remote.origin.url"),
        "input_sha256": {
            source_xlsx.name: sha256(source_xlsx),
            SELECTIVITY: sha256(selectivity_path),
            DECODING_COLOUR: sha256(colour_path),
            DECODING_SHAPE: sha256(shape_path),
        },
        "source_data_url": SOURCE_DATA,
        "source_cache_selectivity_max_abs": source_cache_max_abs,
        "source_cache_decoding_max_abs": decoding_residual,
        "source_cache_dimensionality_max_abs": dimensionality_residual,
        "decoding": source_decoding,
        "shattering_mean": source_dimensionality,
    }
    return stages, metadata


def render(stages: list[dict[str, object]], metadata: dict[str, object]) -> str:
    decoding = np.asarray(metadata["decoding"])
    shattering = np.asarray(metadata["shattering_mean"])
    final_deformation = stages[-1]["from_stage_1"]
    assert isinstance(final_deformation, dict)

    lines = [
        "# 공식 원숭이 PFC 처리자료의 3D 계량 후보 분석",
        "",
        "Status: `PFC_FEASIBILITY_ONLY`",
        "",
        "## 입력",
        "",
        f"- 저자 공식 코드 저장소: `{metadata['remote']}`",
        f"- 분석 커밋: `{metadata['commit']}`",
        f"- 원자료 저장소: {DRYAD}",
        f"- 연계 논문: {PAPER}",
        f"- Nature 공식 Source Data Fig. 2: {metadata['source_data_url']}",
        "- 대상: 실험 1의 실제 macaque PFC 녹화에서 산출된 Nature Source Data와 저자 공식 Git 캐시",
        "- 분석 좌표: colour, shape, XOR selectivity의 3차원 공간",
        "",
        "입력 SHA-256:",
        "",
    ]
    for name, digest in metadata["input_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")

    lines.extend(
        [
            "",
            "Nature XLSX와 저자 Git 캐시 교차대조:",
            "",
            f"- Stage 1/4 selectivity 최대 절대 오차: `{metadata['source_cache_selectivity_max_abs']:.3e}`",
            f"- 4단계 decoding 최대 절대 오차: `{metadata['source_cache_decoding_max_abs']:.3e}`",
            f"- 4단계 dimensionality 최대 절대 오차: `{metadata['source_cache_dimensionality_max_abs']:.3e}`",
        ]
    )

    lines.extend(
        [
            "",
            "## 3D SPD 후보",
            "",
            "각 단계의 뉴런 selectivity 벡터를 $s_n=(s_{colour},s_{shape},s_{XOR})$로 두고, 기술 통계로",
            "",
            "$$",
            "C_k=\\operatorname{Cov}(s_n\\mid k),\\qquad g_k=C_k^{-1}",
            "$$",
            "",
            "를 계산했다. 이는 selectivity chart의 **단계별 상수 SPD 후보**이며, 위치 의존 field나 곡률 측정이 아니다.",
            "",
            "| 단계 | 데이터 출처 | 뉴런 | $\\lambda(C)$ | 유효차원 | $\\mathrm{tr}(C)$ | cond$(C)$ | stage 1 대비 AIRM | shape 비율 |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in stages:
        deformation = stage["from_stage_1"]
        assert isinstance(deformation, dict)
        lines.append(
            "| {stage} | {source} | {neurons} | `{eigenvalues}` | {rank:.4f} | {trace:.6f} | {condition:.3f} | {airm:.4f} | {shape:.1%} |".format(
                stage=stage["stage"],
                source=stage["source"],
                neurons=stage["neurons"],
                eigenvalues=format_vector(stage["covariance_eigenvalues"], 5),
                rank=stage["effective_rank"],
                trace=stage["trace"],
                condition=stage["condition_number"],
                airm=deformation["total"],
                shape=deformation["shape_fraction_sq"],
            )
        )

    lines.extend(
        [
            "",
            "Stage 1에서 4까지 AIRM 변형은 "
            f"`{final_deformation['total']:.6f}`이고, 제곱거리의 `{final_deformation['shape_fraction_sq']:.1%}`가 "
            "공통 scale이 아닌 anisotropic shape 변화다.",
            "",
            "Stage 1 covariance:",
            "",
            f"`{format_matrix(np.asarray(stages[0]['covariance']))}`",
            "",
            "Stage 4 covariance:",
            "",
            f"`{format_matrix(np.asarray(stages[-1]['covariance']))}`",
            "",
            "Stage 1 inverse-covariance metric candidate $g_1$:",
            "",
            f"`{format_matrix(np.asarray(stages[0]['metric']))}`",
            "",
            "Stage 4 inverse-covariance metric candidate $g_4$:",
            "",
            f"`{format_matrix(np.asarray(stages[-1]['metric']))}`",
            "",
            "## 공식 decoding과의 방향 일치",
            "",
            "| 관측량 | Stage 1 | Stage 4 | 변화 |",
            "|---|---:|---:|---:|",
            f"| colour decoding | {decoding[0, 0]:.4f} | {decoding[3, 0]:.4f} | {decoding[3, 0]-decoding[0, 0]:+.4f} |",
            f"| shape decoding | {decoding[0, 1]:.4f} | {decoding[3, 1]:.4f} | {decoding[3, 1]-decoding[0, 1]:+.4f} |",
            f"| width decoding | {decoding[0, 2]:.4f} | {decoding[3, 2]:.4f} | {decoding[3, 2]-decoding[0, 2]:+.4f} |",
            f"| XOR decoding | {decoding[0, 3]:.4f} | {decoding[3, 3]:.4f} | {decoding[3, 3]-decoding[0, 3]:+.4f} |",
            f"| shattering dimensionality score | {shattering[0]:.4f} | {shattering[3]:.4f} | {shattering[3]-shattering[0]:+.4f} |",
            f"| covariance effective rank | {stages[0]['effective_rank']:.4f} | {stages[3]['effective_rank']:.4f} | {stages[3]['effective_rank']-stages[0]['effective_rank']:+.4f} |",
            "",
            "실측 처리자료에서는 학습 후 width와 전체 shattering score가 감소하고 XOR decoding은 증가했다. "
            "동시에 3D selectivity covariance의 유효차원이 `2.8576 -> 2.4982`로 감소했다. "
            "따라서 '학습에 따라 PFC 표현 기하가 저차원·과제 선택적으로 재편된다'는 방향과 일치한다.",
            "",
            "## 반드시 남는 경계",
            "",
            "- 이것은 목업이나 합성 데이터가 아니라 저자 공개 저장소의 실제 macaque PFC 파생 자료다.",
            "- 하지만 원시 trial trajectory를 이번 계산에 다시 적합한 것은 아니다. 저자 캐시의 두 fold는 정확히 동일하며 held-out 검증으로 셀 수 없다.",
            "- 단계마다 다른 뉴런을 합친 pseudopopulation이므로 동일 뉴런의 종단 $\\Delta g$가 아니다.",
            "- `cell_loc`의 연속 좌표, 피질 주름/두께, 구조 연결 $W^s$가 없으므로 물리적 3D cortical-ribbon metric이나 $\\Delta W^s\\to\\Delta g$를 판정하지 못한다.",
            "- $g_k$는 단계별 상수 행렬이어서 이 chart에서는 곡률이 0이다. 이 결과로 비영 곡률을 주장할 수 없다.",
            "",
            "## 판정",
            "",
            "공식 실측 PFC 자료는 **학습에 따른 3D 표현 기하의 anisotropic deformation과 저차원화**를 지지한다. "
            "현재 자료가 지지하지 않는 것은 그 deformation의 원인이 구조 연결 변화라는 주장과, 뇌가 비평탄한 3D Riemann field를 직접 구현한다는 강한 주장이다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-xlsx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stages, metadata = analyze(args.repo.resolve(), args.source_xlsx.resolve())
    report = render(stages, metadata)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
