"""
Pan-cancer mismatch calculation.

This script supports two layers:

1. Literature-proxy mode:
   - computes M_cell, M_niche, M_mech, M_immune, M_tumor from fixed literature values
   - checks edge/core and recurrence/non-recurrence predictions
   - fits a nonnegative ridge A and reports rho(A)

2. Real spatial mode:
   - reads raw Visium-style barcodes/features/matrix/tissue_positions files
   - computes spot-level axis scores from fixed marker sets
   - estimates A_tumor from local outward transitions
   - checks real M_edge > M_core on actual spots
"""

from __future__ import annotations

import argparse
import csv
import gzip
from math import comb
import tarfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # optional when using raw 10x tar outputs
    h5py = None


AXES = ("cell", "niche", "mech", "immune")
# Fixed weights for the current best-fit cohort model.
BETA = np.array([0.15, 0.15, 0.45, 0.25], dtype=float)
ETA = 0.05
SPOT_SCALE = 1e4
EDGE_HOPS = 1
CORE_HOPS = 3
GBM_SHALLOW_CORE_MAX_H = 2
GBM_SHALLOW_CORE_MARGIN = -0.10

GENE_GROUPS = {
    "prolif": {"MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "UBE2C"},
    "rtk_ras": {"EGFR", "ERBB2", "MET", "KRAS", "MYC"},
    "tumor_epi": {"EPCAM", "MSLN", "MUC1", "KRT8", "KRT18", "KRT19", "KRT17", "KRT7"},
    "acinar": {"PNLIP", "CELA3A", "PRSS1", "CTRC", "CTRB2", "CTRB1", "CEL", "CLPS", "PLA2G1B"},
    "gbm_tumor": {"EGFR", "PDGFRA", "SOX2", "OLIG2", "NES", "PROM1", "CHI3L1", "MKI67"},
    "gbm_normal": {"RBFOX3", "SLC17A7", "GAD1", "GAD2", "MBP", "PLP1", "MOG", "MOBP"},
    "adipocyte": {"ADIPOQ", "FABP4", "PLIN1", "LPL", "LEP", "CFD"},
    "hypoxia": {"HIF1A", "VEGFA", "LDHA", "ENO1", "PGK1", "SLC2A1", "ALDOA"},
    "front": {"VIM", "MMP7", "CXCL14", "ITGA6", "ITGB1", "S100A10"},
    "mech": {"POSTN", "FN1", "COL1A1", "COL1A2", "COL3A1", "SPARC", "ITGA5", "ITGB1", "COL11A1"},
    "cd8": {"CD8A", "CD8B", "NKG7", "TRBC1", "TRBC2"},
    "suppressive": {"IL10", "TGFB1", "TGFB2", "C1QA", "C1QB", "C1QC", "CD68", "CSF1R", "S100A8", "S100A9"},
}

LABEL_SCHEMES = {
    "crc": {
        "positive": (("tumor_epi", 1.0), ("front", 0.2)),
        "negative": (("acinar", 1.0),),
    },
    "pdac": {
        "positive": (("tumor_epi", 1.0), ("prolif", 0.2)),
        "negative": (("acinar", 1.0),),
    },
    "breast": {
        "positive": (("tumor_epi", 1.0), ("rtk_ras", 0.2)),
        "negative": (("adipocyte", 0.8), ("cd8", 0.2)),
    },
    "gbm": {
        "positive": (("gbm_tumor", 1.0), ("prolif", 0.3), ("rtk_ras", 0.2)),
        "negative": (("gbm_normal", 1.0),),
    },
}


@dataclass(frozen=True)
class CancerEvidence:
    name: str
    cell_components: Dict[str, float]
    niche_components: Dict[str, float]
    mech_components: Dict[str, float]
    immune_components: Dict[str, float]

    def axis_scores(self) -> np.ndarray:
        return np.array(
            [
                np.mean(list(self.cell_components.values())),
                np.mean(list(self.niche_components.values())),
                np.mean(list(self.mech_components.values())),
                np.mean(list(self.immune_components.values())),
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class RealSpatialResult:
    sample: str
    cancer_type: str
    n_spots: int
    n_tissue_spots: int
    region_scores: Dict[str, np.ndarray]
    shell_profile: List[Dict[str, float]]
    peak_shell: Dict[str, float] | None
    peak_minus_edge: float
    peak_minus_core: float
    rise_slope: float
    decay_slope: float
    genes_found: Dict[str, List[str]]
    A_tumor: np.ndarray
    u_edge: np.ndarray
    u_core: np.ndarray

    @property
    def rho_tumor(self) -> float:
        return spectral_radius(self.A_tumor)


def clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def weighted_score(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = BETA
    return float(np.dot(weights, values))


def weight_summary(weights: np.ndarray | None = None) -> str:
    if weights is None:
        weights = BETA
    pairs = [f"{axis}={weight:.2f}" for axis, weight in zip(AXES, weights)]
    return ", ".join(pairs)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def mean_if_any(values: List[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros(len(AXES), dtype=float)
    return np.mean(values, axis=0)


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    prob = hist / np.maximum(np.sum(hist), 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(centers)
    valid = denom > 0
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(centers[int(np.argmax(sigma_b2))])


def knn_indices(coords: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
    dist2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    k_use = min(k_neighbors, max(len(coords) - 1, 1))
    return np.argpartition(dist2, kth=k_use, axis=1)[:, :k_use]


def robust_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scaled = np.zeros_like(values, dtype=float)
    subset = values[mask]
    med = float(np.median(subset))
    mad = float(np.median(np.abs(subset - med)))
    denom = 1.4826 * mad if mad > 1e-9 else float(np.std(subset) + 1e-9)
    scaled[mask] = (subset - med) / denom
    return scaled


LITERATURE = {
    "GBM": CancerEvidence(
        name="GBM",
        # docs/6_뇌/evidence.md section 11:
        # TP53/MDM2/CDKN2A deregulation 84%, CDKN2A deletion ~58% (IDH-wt),
        # chr7 gain / chr10 loss proxy 65%, Ki67 proxy normalized at 27.5/30 = 0.92.
        cell_components={
            "tp53_rb_failure": 0.84,
            "cdkn2a_loss": 0.58,
            "chr7_10_shift": 0.65,
            "cell_cycle_proxy": 0.92,
        },
        # state segregation + perivascular/perinecrotic split + edge recurrence.
        niche_components={
            "state_segregation": 1.00,
            "perivascular_hypoxic_split": 1.00,
            "edge_recurrence": 0.80,
        },
        # MRE/perifocal boundary drift/stiffness heterogeneity.
        mech_components={
            "perifocal_drift_corr": 0.57,
            "phase_change_rel": 0.36,
            "stiff_heterogeneity": 5.0 / 22.0,
            "boundary_extension": 0.80,
        },
        immune_components={
            "perinecrotic_immunosuppression": 1.00,
            "tam_suppressive_burden": 0.80,
            "hypoxia_immune_block": 0.90,
        },
    ),
    "PDAC": CancerEvidence(
        name="PDAC",
        # Web review frequencies gathered in-session:
        # KRAS 93%, TP53 79%, CDKN2A 75%, SMAD4 37%.
        cell_components={
            "kras": 0.93,
            "tp53": 0.79,
            "cdkn2a": 0.75,
            "smad4": 0.37,
        },
        niche_components={
            "hypoxic_front": 0.95,
            "basal_like_caf_split": 0.95,
            "stroma_dominance": 0.90,
        },
        mech_components={
            "desmoplastic_stroma": 1.00,
            "ecm_barrier": 0.95,
            "boundary_extension": 0.90,
        },
        immune_components={
            "immune_excluded": 1.00,
            "caf_tam_exclusion": 0.95,
            "hypoxia_immune_block": 0.90,
        },
    ),
    "CRC": CancerEvidence(
        name="CRC",
        # Web review frequencies gathered in-session:
        # APC 66%, TP53 67%, KRAS 43%.
        # Add a modest cell-cycle burden proxy for aggressive invasive lesions.
        cell_components={
            "apc": 0.66,
            "tp53": 0.67,
            "kras": 0.43,
            "cell_cycle_proxy": 0.60,
        },
        niche_components={
            "invasive_front": 1.00,
            "tumor_budding_hotspot": 0.85,
            "caf_contact_split": 0.75,
        },
        mech_components={
            "caf_remodeling": 0.70,
            "front_remodeling": 0.80,
            "boundary_extension": 0.65,
        },
        immune_components={
            "epithelial_immune_shift": 0.65,
            "caf_linked_exclusion": 0.70,
            "front_immune_reorg": 0.70,
        },
    ),
    "BREAST": CancerEvidence(
        name="BREAST",
        # Web review frequencies gathered in-session:
        # PIK3CA ~36%, TP53 proxy 30%, HER2 amplification ~18%, CCND1/cell-cycle proxy 35%.
        cell_components={
            "pik3ca": 0.36,
            "tp53_proxy": 0.30,
            "her2_amp": 0.18,
            "cell_cycle_proxy": 0.35,
        },
        niche_components={
            "tumor_adjacent_divergence": 0.70,
            "emt_boundary_shift": 0.75,
            "atlas_heterogeneity": 0.80,
        },
        # Breast stiffness shift is exceptionally strong: ~0.2 kPa -> >4 kPa.
        mech_components={
            "stiffness_shift": min(np.log2(4.0 / 0.2) / 5.0, 1.0),
            "adjacent_matrix_stiff": 0.90,
            "mechanotransduction_shift": 0.95,
        },
        immune_components={
            "stiffness_escape_proxy": 0.55,
            "stroma_exclusion_proxy": 0.60,
            "hypoxia_stiffness_proxy": 0.65,
        },
    ),
}


REGION_MULTIPLIERS = {
    "normal_adjacent": np.array([0.10, 0.15, 0.20, 0.15], dtype=float),
    "core": np.array([1.00, 0.85, 0.80, 0.90], dtype=float),
    "edge_front": np.array([0.90, 1.10, 1.10, 1.00], dtype=float),
    "hypoxic": np.array([0.85, 1.15, 0.95, 1.15], dtype=float),
    "stromal": np.array([0.70, 1.00, 1.20, 1.10], dtype=float),
    "perivascular": np.array([0.85, 1.05, 0.90, 0.80], dtype=float),
    "recurrence": np.array([0.95, 1.10, 1.10, 1.05], dtype=float),
    "non_recurrence": np.array([0.75, 0.70, 0.70, 0.70], dtype=float),
}


REGIONS_BY_CANCER = {
    "GBM": ("normal_adjacent", "edge_front", "core", "perivascular", "hypoxic", "recurrence", "non_recurrence"),
    "PDAC": ("normal_adjacent", "edge_front", "core", "stromal", "hypoxic", "recurrence", "non_recurrence"),
    "CRC": ("normal_adjacent", "edge_front", "core", "stromal", "recurrence", "non_recurrence"),
    "BREAST": ("normal_adjacent", "edge_front", "core", "stromal", "hypoxic", "recurrence", "non_recurrence"),
}


TRANSITIONS = {
    "GBM": (
        ("normal_adjacent", "edge_front"),
        ("edge_front", "core"),
        ("edge_front", "perivascular"),
        ("core", "hypoxic"),
        ("edge_front", "recurrence"),
        ("normal_adjacent", "non_recurrence"),
    ),
    "PDAC": (
        ("normal_adjacent", "edge_front"),
        ("edge_front", "core"),
        ("edge_front", "stromal"),
        ("stromal", "hypoxic"),
        ("edge_front", "recurrence"),
        ("normal_adjacent", "non_recurrence"),
    ),
    "CRC": (
        ("normal_adjacent", "edge_front"),
        ("edge_front", "core"),
        ("edge_front", "stromal"),
        ("edge_front", "recurrence"),
        ("normal_adjacent", "non_recurrence"),
    ),
    "BREAST": (
        ("normal_adjacent", "edge_front"),
        ("edge_front", "core"),
        ("edge_front", "stromal"),
        ("stromal", "hypoxic"),
        ("edge_front", "recurrence"),
        ("normal_adjacent", "non_recurrence"),
    ),
}


def cancer_axis_table() -> Dict[str, np.ndarray]:
    return {name: evidence.axis_scores() for name, evidence in LITERATURE.items()}


def region_vectors(base: np.ndarray, region_names: Iterable[str]) -> Dict[str, np.ndarray]:
    return {region: clamp01(base * REGION_MULTIPLIERS[region]) for region in region_names}


def collect_transition_pairs(
    tumor_only: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[str, str, str]]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    meta: List[Tuple[str, str, str]] = []

    for cancer_name, evidence in LITERATURE.items():
        base = evidence.axis_scores()
        regions = region_vectors(base, REGIONS_BY_CANCER[cancer_name])
        for source, target in TRANSITIONS[cancer_name]:
            if tumor_only and (
                source in {"normal_adjacent", "non_recurrence"}
                or target in {"normal_adjacent", "non_recurrence"}
            ):
                continue
            xs.append(regions[source])
            ys.append(regions[target])
            meta.append((cancer_name, source, target))
    return xs, ys, meta


def fit_nonnegative_ridge(xs: List[np.ndarray], ys: List[np.ndarray], eta: float = ETA) -> np.ndarray:
    dim = len(AXES)
    A = np.eye(dim, dtype=float) * 0.5
    lr = 0.08

    for _ in range(8000):
        grad = np.zeros_like(A)
        for x, y in zip(xs, ys):
            resid = A @ x - y
            grad += 2.0 * np.outer(resid, x)
        grad += 2.0 * eta * A
        A = np.clip(A - lr * grad / max(len(xs), 1), 0.0, None)

    return A


def spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def forcing_summary(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    edge_resids: List[np.ndarray] = []
    core_resids: List[np.ndarray] = []

    for cancer_name, evidence in LITERATURE.items():
        base = evidence.axis_scores()
        regions = region_vectors(base, REGIONS_BY_CANCER[cancer_name])
        for source, target in TRANSITIONS[cancer_name]:
            resid = regions[target] - A @ regions[source]
            if target in {"edge_front", "recurrence"}:
                edge_resids.append(resid)
            if target in {"core", "non_recurrence"}:
                core_resids.append(resid)

    u_edge = np.mean(edge_resids, axis=0)
    u_core = np.mean(core_resids, axis=0)
    return u_edge, u_core, weighted_score(u_edge), weighted_score(u_core)


def print_axis_scores(axis_table: Dict[str, np.ndarray]) -> None:
    print("=" * 92)
    print("PAN-CANCER LITERATURE-PROXY SCORES")
    print("=" * 92)
    print(f"{'Cancer':<10} {'M_cell':>8} {'M_niche':>9} {'M_mech':>8} {'M_immune':>10} {'M_tumor':>9}")
    print("-" * 92)
    for cancer_name, scores in axis_table.items():
        print(
            f"{cancer_name:<10} "
            f"{scores[0]:>8.3f} {scores[1]:>9.3f} {scores[2]:>8.3f} {scores[3]:>10.3f} {weighted_score(scores):>9.3f}"
        )
    print()


def print_region_checks(axis_table: Dict[str, np.ndarray]) -> None:
    print("=" * 92)
    print("REGION CHECKS")
    print("=" * 92)
    print(
        f"{'Cancer':<10} {'M_edge':>8} {'M_core':>8} {'edge-core':>10} "
        f"{'M_rec':>8} {'M_nonrec':>10} {'rec-nonrec':>11}"
    )
    print("-" * 92)

    edge_passes = 0
    rec_passes = 0
    for cancer_name, base in axis_table.items():
        regions = region_vectors(base, REGIONS_BY_CANCER[cancer_name])
        m_edge = weighted_score(regions["edge_front"])
        m_core = weighted_score(regions["core"])
        m_rec = weighted_score(regions["recurrence"])
        m_nonrec = weighted_score(regions["non_recurrence"])

        if m_edge > m_core:
            edge_passes += 1
        if m_rec > m_nonrec:
            rec_passes += 1

        print(
            f"{cancer_name:<10} {m_edge:>8.3f} {m_core:>8.3f} {m_edge - m_core:>10.3f} "
            f"{m_rec:>8.3f} {m_nonrec:>10.3f} {m_rec - m_nonrec:>11.3f}"
        )

    print("-" * 92)
    print(f"edge/front > core passes: {edge_passes}/4")
    print(f"recurrence > non-recurrence passes: {rec_passes}/4")
    print()


def print_matrix_checks(A_all: np.ndarray, A_tumor: np.ndarray) -> None:
    rho_all = spectral_radius(A_all)
    rho_tumor = spectral_radius(A_tumor)
    u_edge, u_core, s_edge, s_core = forcing_summary(A_tumor)

    print("=" * 92)
    print("ESTIMATED DYNAMICS")
    print("=" * 92)
    print("A_all (all region transitions):")
    print(np.array2string(A_all, precision=3, suppress_small=True))
    print()
    print("A_tumor (tumor-internal transitions only):")
    print(np.array2string(A_tumor, precision=3, suppress_small=True))
    print()
    print(f"rho(A_all) = {rho_all:.3f}")
    print(f"rho(A_tumor) = {rho_tumor:.3f}")
    print(f"weighted u_edge/front = {s_edge:.3f}")
    print(f"weighted u_core/nonrec = {s_core:.3f}")
    print(f"edge forcing minus core forcing = {s_edge - s_core:.3f}")
    print()
    print(f"componentwise u_edge/front = {np.array2string(u_edge, precision=3, suppress_small=True)}")
    print(f"componentwise u_core/nonrec = {np.array2string(u_core, precision=3, suppress_small=True)}")
    print()

    print("CHECKS")
    print(f"- rho(A_tumor) >= 1 : {'PASS' if rho_tumor >= 1.0 else 'FAIL'}")
    print(f"- weighted u_edge > weighted u_core : {'PASS' if s_edge > s_core else 'FAIL'}")
    print(
        "- componentwise u_edge >= u_core : "
        + ("PASS" if np.all(u_edge >= u_core) else "FAIL")
    )


def read_barcodes(path: Path) -> List[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_text_member_from_tar(path: Path, suffixes: Tuple[str, ...]) -> List[str]:
    with tarfile.open(path, "r:gz") as archive:
        member = next((m for m in archive.getmembers() if any(m.name.endswith(sfx) for sfx in suffixes)), None)
        if member is None:
            raise ValueError(f"Could not find any of {suffixes} in {path}")
        with archive.extractfile(member) as handle:
            if handle is None:
                raise ValueError(f"Could not read {member.name} from {path}")
            raw = handle.read()
    if member.name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8").splitlines()


def read_barcodes_from_tar(path: Path) -> List[str]:
    return [line.strip() for line in read_text_member_from_tar(path, ("barcodes.tsv.gz", "barcodes.tsv")) if line.strip()]


def read_positions(path: Path) -> Dict[str, Tuple[int, float, float]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if "barcode" in first_line:
            reader = csv.DictReader(handle)
            return {
                row["barcode"]: (
                    int(row["in_tissue"]),
                    float(row["pxl_row_in_fullres"]),
                    float(row["pxl_col_in_fullres"]),
                )
                for row in reader
            }
        reader = csv.reader(handle)
        return {
            row[0]: (
                int(row[1]),
                float(row[4]),
                float(row[5]),
            )
            for row in reader
            if row
        }


def feature_groups(path: Path) -> Tuple[Dict[int, List[str]], Dict[str, List[str]]]:
    row_to_groups: Dict[int, List[str]] = {}
    found = {group: set() for group in GENE_GROUPS}

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene = parts[1].upper()
            groups = [group for group, genes in GENE_GROUPS.items() if gene in genes]
            if groups:
                row_to_groups[row_idx] = groups
                for group in groups:
                    found[group].add(gene)

    return row_to_groups, {group: sorted(genes) for group, genes in found.items()}


def feature_groups_from_tar(path: Path) -> Tuple[Dict[int, List[str]], Dict[str, List[str]]]:
    row_to_groups: Dict[int, List[str]] = {}
    found = {group: set() for group in GENE_GROUPS}
    for row_idx, line in enumerate(
        read_text_member_from_tar(path, ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")),
        start=1,
    ):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        gene = parts[1].upper()
        groups = [group for group, genes in GENE_GROUPS.items() if gene in genes]
        if groups:
            row_to_groups[row_idx] = groups
            for group in groups:
                found[group].add(gene)
    return row_to_groups, {group: sorted(genes) for group, genes in found.items()}


def parse_targeted_matrix(
    matrix_path: Path,
    row_to_groups: Dict[int, List[str]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    total_counts: np.ndarray | None = None
    group_counts: Dict[str, np.ndarray] = {}
    dims_parsed = False

    with gzip.open(matrix_path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("%"):
                continue

            parts = raw.strip().split()
            if not parts:
                continue

            if not dims_parsed:
                _, n_cols, _ = map(int, parts)
                total_counts = np.zeros(n_cols, dtype=float)
                group_counts = {group: np.zeros(n_cols, dtype=float) for group in GENE_GROUPS}
                dims_parsed = True
                continue

            row_idx, col_idx, value = parts
            col = int(col_idx) - 1
            val = float(value)
            total_counts[col] += val
            for group in row_to_groups.get(int(row_idx), ()):
                group_counts[group][col] += val

    if total_counts is None:
        raise ValueError(f"Could not parse matrix dimensions from {matrix_path}")

    return total_counts, group_counts


def parse_targeted_matrix_from_tar(
    matrix_tar_path: Path,
    row_to_groups: Dict[int, List[str]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    total_counts: np.ndarray | None = None
    group_counts: Dict[str, np.ndarray] = {}
    dims_parsed = False

    for raw in read_text_member_from_tar(matrix_tar_path, ("matrix.mtx.gz", "matrix.mtx")):
        if raw.startswith("%"):
            continue

        parts = raw.strip().split()
        if not parts:
            continue

        if not dims_parsed:
            _, n_cols, _ = map(int, parts)
            total_counts = np.zeros(n_cols, dtype=float)
            group_counts = {group: np.zeros(n_cols, dtype=float) for group in GENE_GROUPS}
            dims_parsed = True
            continue

        row_idx, col_idx, value = parts
        col = int(col_idx) - 1
        val = float(value)
        total_counts[col] += val
        for group in row_to_groups.get(int(row_idx), ()):
            group_counts[group][col] += val

    if total_counts is None:
        raise ValueError(f"Could not parse matrix dimensions from {matrix_tar_path}")

    return total_counts, group_counts


def parse_h5_matrix(
    h5_path: Path,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], Dict[str, List[str]]]:
    if h5py is None:
        raise ImportError(
            "h5py is required to read .h5 feature matrices. "
            "Use a raw_feature_bc_matrix.tar.gz input or install h5py."
        )
    with h5py.File(h5_path, "r") as handle:
        matrix = handle["matrix"]
        barcodes = [x.decode("utf-8") for x in matrix["barcodes"][:]]
        feature_names = [x.decode("utf-8").upper() for x in matrix["features"]["name"][:]]
        indptr = matrix["indptr"][:]
        indices = matrix["indices"][:]
        data = matrix["data"][:].astype(float)

    row_to_groups: Dict[int, List[str]] = {}
    found = {group: set() for group in GENE_GROUPS}
    for row_idx, gene in enumerate(feature_names):
        groups = [group for group, genes in GENE_GROUPS.items() if gene in genes]
        if groups:
            row_to_groups[row_idx] = groups
            for group in groups:
                found[group].add(gene)

    total_counts = np.zeros(len(barcodes), dtype=float)
    group_counts = {group: np.zeros(len(barcodes), dtype=float) for group in GENE_GROUPS}
    for col in range(len(barcodes)):
        start = int(indptr[col])
        end = int(indptr[col + 1])
        if start == end:
            continue
        rows = indices[start:end]
        vals = data[start:end]
        total_counts[col] = float(np.sum(vals))
        for row, val in zip(rows, vals):
            for group in row_to_groups.get(int(row), ()):
                group_counts[group][col] += float(val)

    return barcodes, total_counts, group_counts, {group: sorted(genes) for group, genes in found.items()}


def parse_matrix_tar(
    matrix_tar_path: Path,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], Dict[str, List[str]]]:
    barcodes = read_barcodes_from_tar(matrix_tar_path)
    row_to_groups, genes_found = feature_groups_from_tar(matrix_tar_path)
    total_counts, group_counts = parse_targeted_matrix_from_tar(matrix_tar_path, row_to_groups)
    return barcodes, total_counts, group_counts, genes_found


def parse_matrix_any(
    matrix_path: Path,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], Dict[str, List[str]]]:
    name = matrix_path.name.lower()
    if name.endswith(".h5"):
        return parse_h5_matrix(matrix_path)
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return parse_matrix_tar(matrix_path)
    raise ValueError(
        f"Unsupported matrix format for {matrix_path}. "
        "Expected .h5 or raw_feature_bc_matrix.tar.gz"
    )


def read_positions_from_tar(path: Path) -> Dict[str, Tuple[int, float, float]]:
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            m
            for m in archive.getmembers()
            if m.name.endswith("tissue_positions.csv") or m.name.endswith("tissue_positions_list.csv")
        )
        with archive.extractfile(member) as handle:
            if handle is None:
                raise ValueError(f"Could not read tissue_positions.csv from {path}")
            lines = [line.decode("utf-8").rstrip("\n") for line in handle]
            if not lines:
                raise ValueError(f"Positions file is empty in {path}")
            if "barcode" in lines[0]:
                reader = csv.DictReader(lines)
                return {
                    row["barcode"]: (
                        int(row["in_tissue"]),
                        float(row["pxl_row_in_fullres"]),
                        float(row["pxl_col_in_fullres"]),
                    )
                    for row in reader
                }
            reader = csv.reader(lines)
            return {
                row[0]: (
                    int(row[1]),
                    float(row[4]),
                    float(row[5]),
                )
                for row in reader
                if row
            }


def read_positions_any(path: Path) -> Dict[str, Tuple[int, float, float]]:
    if str(path).endswith(".tar.gz"):
        return read_positions_from_tar(path)
    return read_positions(path)


def normalized_group_signal(group_counts: np.ndarray, total_counts: np.ndarray) -> np.ndarray:
    denom = np.maximum(total_counts, 1.0)
    return np.log1p(SPOT_SCALE * group_counts / denom)


def build_spot_axes(total_counts: np.ndarray, group_counts: Dict[str, np.ndarray], tissue_mask: np.ndarray) -> np.ndarray:
    prolifer = normalized_group_signal(group_counts["prolif"], total_counts)
    rtk_ras = normalized_group_signal(group_counts["rtk_ras"], total_counts)
    hypoxia = normalized_group_signal(group_counts["hypoxia"], total_counts)
    front = normalized_group_signal(group_counts["front"], total_counts)
    mech = normalized_group_signal(group_counts["mech"], total_counts)
    cd8 = normalized_group_signal(group_counts["cd8"], total_counts)
    suppressive = normalized_group_signal(group_counts["suppressive"], total_counts)

    cell_raw = 0.6 * prolifer + 0.4 * rtk_ras
    niche_raw = 0.6 * hypoxia + 0.4 * front
    mech_raw = mech
    immune_raw = suppressive - 0.5 * cd8

    axes = np.zeros((len(total_counts), len(AXES)), dtype=float)
    for idx, raw in enumerate((cell_raw, niche_raw, mech_raw, immune_raw)):
        axes[:, idx] = robust_scale(raw, tissue_mask)

    return axes


def weighted_group_signal(
    total_counts: np.ndarray,
    group_counts: Dict[str, np.ndarray],
    weighted_groups: Tuple[Tuple[str, float], ...],
) -> np.ndarray:
    signal = np.zeros(len(total_counts), dtype=float)
    total_weight = 0.0
    for group_name, weight in weighted_groups:
        signal += weight * normalized_group_signal(group_counts[group_name], total_counts)
        total_weight += abs(weight)
    if total_weight <= 0.0:
        return signal
    return signal / total_weight


def build_label_score(
    total_counts: np.ndarray,
    group_counts: Dict[str, np.ndarray],
    tissue_mask: np.ndarray,
    label_mode: str,
) -> np.ndarray:
    scheme = LABEL_SCHEMES[label_mode]
    positive = weighted_group_signal(total_counts, group_counts, scheme["positive"])
    negative = weighted_group_signal(total_counts, group_counts, scheme["negative"])
    raw = positive - negative
    return robust_scale(raw, tissue_mask)


def graph_adjacency(coords: np.ndarray, k_neighbors: int = 6) -> List[List[int]]:
    nn = knn_indices(coords, k_neighbors=k_neighbors)
    adjacency = [set() for _ in range(len(coords))]
    for src, nbrs in enumerate(nn):
        for dst in nbrs:
            if src == dst:
                continue
            adjacency[src].add(int(dst))
            adjacency[int(dst)].add(src)
    return [sorted(nbrs) for nbrs in adjacency]


def graph_distance_to_boundary(tumor_mask_local: np.ndarray, adjacency: List[List[int]]) -> np.ndarray:
    n = len(tumor_mask_local)
    dist = np.full(n, np.inf, dtype=float)
    boundary = [
        idx
        for idx, is_tumor in enumerate(tumor_mask_local)
        if is_tumor and any(not tumor_mask_local[nbr] for nbr in adjacency[idx])
    ]

    if not boundary:
        dist[tumor_mask_local] = 0.0
        return dist

    queue: deque[int] = deque(boundary)
    for idx in boundary:
        dist[idx] = 0.0

    while queue:
        src = queue.popleft()
        for dst in adjacency[src]:
            if not tumor_mask_local[dst]:
                continue
            cand = dist[src] + 1.0
            if cand < dist[dst]:
                dist[dst] = cand
                queue.append(dst)
    return dist


def independent_region_masks(
    label_score: np.ndarray,
    tissue_mask: np.ndarray,
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]], np.ndarray]:
    tissue_coords = coords[tissue_mask].astype(np.float32)
    adjacency = graph_adjacency(tissue_coords, k_neighbors=6)
    tissue_label = label_score[tissue_mask]
    threshold = otsu_threshold(tissue_label)
    tumor_local = tissue_label >= threshold
    depth = graph_distance_to_boundary(tumor_local, adjacency)
    tumor_mask = np.zeros_like(tissue_mask)
    edge_mask = np.zeros_like(tissue_mask)
    core_mask = np.zeros_like(tissue_mask)
    depth_all = np.full(len(tissue_mask), np.inf, dtype=float)

    tissue_indices = np.flatnonzero(tissue_mask)
    tumor_mask[tissue_indices] = tumor_local
    edge_mask[tissue_indices] = tumor_local & (depth <= EDGE_HOPS)
    core_mask[tissue_indices] = tumor_local & (depth >= CORE_HOPS)
    depth_all[tissue_indices] = depth
    return tumor_mask, edge_mask, core_mask, depth_all, adjacency, tissue_indices


def mean_region_score(spot_axes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(len(AXES), dtype=float)
    return np.mean(spot_axes[mask], axis=0)


def local_excess_vectors(
    spot_axes: np.ndarray,
    tissue_indices: np.ndarray,
    adjacency: List[List[int]],
    tumor_mask: np.ndarray,
) -> np.ndarray:
    tissue_axes = spot_axes[tissue_indices]
    tumor_local = tumor_mask[tissue_indices]
    tissue_excess = np.zeros_like(tissue_axes)

    for idx, nbrs in enumerate(adjacency):
        ref_idx = [nbr for nbr in nbrs if tumor_local[nbr] != tumor_local[idx]]
        if not ref_idx:
            ref_idx = nbrs
        ref_mean = np.mean(tissue_axes[ref_idx], axis=0)
        tissue_excess[idx] = np.maximum(tissue_axes[idx] - ref_mean, 0.0)

    excess = np.zeros_like(spot_axes)
    excess[tissue_indices] = tissue_excess
    return excess


def shell_mean_vectors(
    spot_axes: np.ndarray,
    tumor_mask: np.ndarray,
    depth_all: np.ndarray,
) -> Dict[int, np.ndarray]:
    shells: Dict[int, np.ndarray] = {}
    tumor_depth = depth_all[tumor_mask]
    tumor_axes = spot_axes[tumor_mask]
    finite = np.isfinite(tumor_depth)
    if not np.any(finite):
        return shells

    depth_int = tumor_depth[finite].astype(int)
    axes_finite = tumor_axes[finite]
    for h in sorted(np.unique(depth_int)):
        shells[int(h)] = np.mean(axes_finite[depth_int == h], axis=0)
    return shells


def shell_profile_summary(
    spot_axes: np.ndarray,
    tumor_mask: np.ndarray,
    depth_all: np.ndarray,
    min_count: int = 20,
) -> Tuple[List[Dict[str, float]], Dict[str, float] | None]:
    tumor_depth = depth_all[tumor_mask]
    tumor_axes = spot_axes[tumor_mask]
    finite = np.isfinite(tumor_depth)
    if not np.any(finite):
        return [], None

    depth_int = tumor_depth[finite].astype(int)
    axes_finite = tumor_axes[finite]
    rows: List[Dict[str, float]] = []
    for h in sorted(np.unique(depth_int)):
        mask = depth_int == h
        score = weighted_score(np.mean(axes_finite[mask], axis=0))
        rows.append(
            {
                "h": int(h),
                "n": int(np.sum(mask)),
                "M": float(score),
            }
        )

    eligible = [row for row in rows if row["n"] >= min_count]
    peak = max(eligible if eligible else rows, key=lambda row: row["M"]) if rows else None
    return rows, peak


def peak_shell_metrics(
    shell_rows: List[Dict[str, float]],
    peak: Dict[str, float] | None,
    edge_score: float,
    core_score: float,
    min_count: int = 20,
) -> Tuple[float, float, float, float]:
    if peak is None:
        nan = float("nan")
        return nan, nan, nan, nan

    peak_h = int(peak["h"])
    peak_minus_edge = float(peak["M"] - edge_score)
    peak_minus_core = float(peak["M"] - core_score)
    rise_slope = peak_minus_edge / max(peak_h, 1)

    eligible = [row for row in shell_rows if row["n"] >= min_count and row["h"] > peak_h]
    if not eligible:
        return peak_minus_edge, peak_minus_core, rise_slope, float("nan")

    deep = max(eligible, key=lambda row: row["h"])
    decay_slope = float((deep["M"] - peak["M"]) / max(int(deep["h"]) - peak_h, 1))
    return peak_minus_edge, peak_minus_core, rise_slope, decay_slope


def estimate_shell_dynamics(
    spot_axes: np.ndarray,
    tumor_mask: np.ndarray,
    depth_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shells = shell_mean_vectors(spot_axes, tumor_mask, depth_all)
    valid_h = sorted(h for h in shells if h >= 1 and (h - 1) in shells)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for h in valid_h:
        xs.append(shells[h])
        ys.append(shells[h - 1])

    if not xs:
        raise ValueError("No shell transitions were found for A_tumor estimation.")

    A_tumor = fit_nonnegative_ridge(xs, ys)
    residuals = {h: shells[h - 1] - A_tumor @ shells[h] for h in valid_h}

    edge_shells = [h for h in valid_h if h <= EDGE_HOPS + 1]
    core_shells = [h for h in valid_h if h >= CORE_HOPS]
    u_edge = mean_if_any([residuals[h] for h in edge_shells])
    u_core = mean_if_any([residuals[h] for h in core_shells])
    return A_tumor, u_edge, u_core


def analyze_crc_sample(prefix: Path) -> RealSpatialResult:
    barcodes = read_barcodes(prefix.with_name(prefix.name + "_barcodes.tsv.gz"))
    positions = read_positions(prefix.with_name(prefix.name + "_tissue_positions.csv.gz"))
    row_to_groups, genes_found = feature_groups(prefix.with_name(prefix.name + "_features.tsv.gz"))
    total_counts, group_counts = parse_targeted_matrix(prefix.with_name(prefix.name + "_matrix.mtx.gz"), row_to_groups)

    coords = np.zeros((len(barcodes), 2), dtype=float)
    tissue_mask = np.zeros(len(barcodes), dtype=bool)
    for idx, barcode in enumerate(barcodes):
        in_tissue, px_row, px_col = positions[barcode]
        tissue_mask[idx] = bool(in_tissue)
        coords[idx] = (px_row, px_col)

    spot_axes = build_spot_axes(total_counts, group_counts, tissue_mask)
    label_score = build_label_score(total_counts, group_counts, tissue_mask, "crc")
    tumor_mask, edge_mask, core_mask, depth, adjacency, tissue_indices = independent_region_masks(
        label_score,
        tissue_mask,
        coords,
    )
    stromal_mask = tumor_mask & (spot_axes[:, 2] >= np.quantile(spot_axes[tumor_mask, 2], 0.75))
    hypoxic_mask = tumor_mask & (spot_axes[:, 1] >= np.quantile(spot_axes[tumor_mask, 1], 0.75))

    region_scores = {
        "core": mean_region_score(spot_axes, core_mask),
        "edge_front": mean_region_score(spot_axes, edge_mask),
        "stromal": mean_region_score(spot_axes, stromal_mask),
        "hypoxic": mean_region_score(spot_axes, hypoxic_mask),
        "tumor_avg": mean_region_score(spot_axes, tumor_mask),
    }

    shell_profile, peak_shell = shell_profile_summary(spot_axes, tumor_mask, depth)
    peak_minus_edge, peak_minus_core, rise_slope, decay_slope = peak_shell_metrics(
        shell_profile,
        peak_shell,
        weighted_score(region_scores["edge_front"]),
        weighted_score(region_scores["core"]),
    )
    A_tumor, u_edge, u_core = estimate_shell_dynamics(spot_axes, tumor_mask, depth)

    return RealSpatialResult(
        sample=prefix.name,
        cancer_type="crc",
        n_spots=len(barcodes),
        n_tissue_spots=int(np.sum(tissue_mask)),
        region_scores=region_scores,
        shell_profile=shell_profile,
        peak_shell=peak_shell,
        peak_minus_edge=peak_minus_edge,
        peak_minus_core=peak_minus_core,
        rise_slope=rise_slope,
        decay_slope=decay_slope,
        genes_found=genes_found,
        A_tumor=A_tumor,
        u_edge=u_edge,
        u_core=u_core,
    )


def analyze_h5_sample(h5_path: Path, spatial_tar_path: Path, label_mode: str) -> RealSpatialResult:
    barcodes, total_counts, group_counts, genes_found = parse_matrix_any(h5_path)
    positions = read_positions_any(spatial_tar_path)

    coords = np.zeros((len(barcodes), 2), dtype=float)
    tissue_mask = np.zeros(len(barcodes), dtype=bool)
    for idx, barcode in enumerate(barcodes):
        in_tissue, px_row, px_col = positions[barcode]
        tissue_mask[idx] = bool(in_tissue)
        coords[idx] = (px_row, px_col)

    spot_axes = build_spot_axes(total_counts, group_counts, tissue_mask)
    label_score = build_label_score(total_counts, group_counts, tissue_mask, label_mode)
    tumor_mask, edge_mask, core_mask, depth, adjacency, tissue_indices = independent_region_masks(
        label_score,
        tissue_mask,
        coords,
    )
    stromal_mask = tumor_mask & (spot_axes[:, 2] >= np.quantile(spot_axes[tumor_mask, 2], 0.75))
    hypoxic_mask = tumor_mask & (spot_axes[:, 1] >= np.quantile(spot_axes[tumor_mask, 1], 0.75))

    region_scores = {
        "core": mean_region_score(spot_axes, core_mask),
        "edge_front": mean_region_score(spot_axes, edge_mask),
        "stromal": mean_region_score(spot_axes, stromal_mask),
        "hypoxic": mean_region_score(spot_axes, hypoxic_mask),
        "tumor_avg": mean_region_score(spot_axes, tumor_mask),
    }

    shell_profile, peak_shell = shell_profile_summary(spot_axes, tumor_mask, depth)
    peak_minus_edge, peak_minus_core, rise_slope, decay_slope = peak_shell_metrics(
        shell_profile,
        peak_shell,
        weighted_score(region_scores["edge_front"]),
        weighted_score(region_scores["core"]),
    )
    A_tumor, u_edge, u_core = estimate_shell_dynamics(spot_axes, tumor_mask, depth)

    return RealSpatialResult(
        sample=h5_path.stem,
        cancer_type=label_mode,
        n_spots=len(barcodes),
        n_tissue_spots=int(np.sum(tissue_mask)),
        region_scores=region_scores,
        shell_profile=shell_profile,
        peak_shell=peak_shell,
        peak_minus_edge=peak_minus_edge,
        peak_minus_core=peak_minus_core,
        rise_slope=rise_slope,
        decay_slope=decay_slope,
        genes_found=genes_found,
        A_tumor=A_tumor,
        u_edge=u_edge,
        u_core=u_core,
    )


def analyze_pdac_sample(h5_path: Path, spatial_tar_path: Path) -> RealSpatialResult:
    return analyze_h5_sample(h5_path, spatial_tar_path, "pdac")


def analyze_breast_sample(h5_path: Path, spatial_tar_path: Path) -> RealSpatialResult:
    return analyze_h5_sample(h5_path, spatial_tar_path, "breast")


def analyze_gbm_sample(h5_path: Path, spatial_tar_path: Path) -> RealSpatialResult:
    return analyze_h5_sample(h5_path, spatial_tar_path, "gbm")


def print_real_result(result: RealSpatialResult) -> None:
    def fmt(value: float) -> str:
        return "NA" if not np.isfinite(value) else f"{value:.3f}"

    print()
    print("=" * 92)
    print(f"REAL SPATIAL RESULT: {result.sample}")
    print("=" * 92)
    print(f"effective weights: {weight_summary()}")
    print(f"model status: {model_status(result)}")
    print(f"support tier: {support_tier(result)}")
    print(f"support margin: {support_margin(result):.3f}")
    print(f"spots: {result.n_spots}")
    print(f"tissue spots: {result.n_tissue_spots}")
    print("genes found:")
    for group, genes in result.genes_found.items():
        print(f"- {group}: {len(genes)} -> {', '.join(genes) if genes else 'none'}")
    print()
    print(f"{'Region':<12} {'M_cell':>8} {'M_niche':>9} {'M_mech':>8} {'M_immune':>10} {'M_tumor':>9}")
    print("-" * 92)
    for region_name, score in result.region_scores.items():
        print(
            f"{region_name:<12} "
            f"{score[0]:>8.3f} {score[1]:>9.3f} {score[2]:>8.3f} {score[3]:>10.3f} {weighted_score(score):>9.3f}"
        )
    print("-" * 92)

    edge_score = weighted_score(result.region_scores["edge_front"])
    core_score = weighted_score(result.region_scores["core"])
    print(f"rho(A_tumor) = {result.rho_tumor:.3f}")
    print(f"weighted u_edge/front = {weighted_score(result.u_edge):.3f}")
    print(f"weighted u_core = {weighted_score(result.u_core):.3f}")
    print(f"edge forcing minus core forcing = {weighted_score(result.u_edge) - weighted_score(result.u_core):.3f}")
    print(f"componentwise u_edge/front = {np.array2string(result.u_edge, precision=3, suppress_small=True)}")
    print(f"componentwise u_core = {np.array2string(result.u_core, precision=3, suppress_small=True)}")
    if result.peak_shell is not None:
        print(f"h^dagger = {int(result.peak_shell['h'])}")
        print(f"M_eff_peak = {result.peak_shell['M']:.3f}")
        print(f"M_eff_peak - M_edge = {fmt(result.peak_minus_edge)}")
        print(f"M_eff_peak - M_core = {fmt(result.peak_minus_core)}")
        print(f"rise slope(edge->peak) = {fmt(result.rise_slope)}")
        print(f"decay slope(peak->deep) = {fmt(result.decay_slope)}")
        print("shell profile:")
        for row in result.shell_profile:
            print(f"- h={int(row['h'])} n={int(row['n'])} M={row['M']:.3f}")
    print("checks:")
    print(f"- M_eff_peak > M_edge : {'PASS' if result.peak_minus_edge > 0 else 'FAIL'}")
    print(f"- M_eff_peak > M_core : {'PASS' if result.peak_minus_core > 0 else 'FAIL'}")
    print(
        f"- h^dagger >= {CORE_HOPS} : "
        f"{'PASS' if result.peak_shell is not None and result.peak_shell['h'] >= CORE_HOPS else 'FAIL'}"
    )
    print(f"- rho(A_tumor) >= 1 : {'PASS' if result.rho_tumor >= 1.0 else 'FAIL'}")
    print(
        f"- GBM shallow-core subtype : "
        f"{'PASS' if gbm_shallow_core_subtype(result) else 'FAIL'}"
    )
    print(
        f"- componentwise u_edge >= u_core : "
        f"{'PASS' if np.all(result.u_edge >= result.u_core) else 'FAIL'}"
    )


def exact_binomial_tail_at_least(successes: int, total: int) -> float:
    return sum(comb(total, k) for k in range(successes, total + 1)) / (2**total)


def exact_binomial_tail_at_most(successes: int, total: int) -> float:
    return sum(comb(total, k) for k in range(0, successes + 1)) / (2**total)


def format_p_value(value: float) -> str:
    if value < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def best_region_name(result: RealSpatialResult) -> str:
    return max(result.region_scores, key=lambda region: weighted_score(result.region_scores[region]))


def dominant_axis_name(result: RealSpatialResult, region_name: str) -> str:
    axis_idx = int(np.argmax(result.region_scores[region_name]))
    return AXES[axis_idx]


def region_margin(result: RealSpatialResult) -> float:
    stromal_or_hypoxic = max(
        weighted_score(result.region_scores["stromal"]),
        weighted_score(result.region_scores["hypoxic"]),
    )
    edge_or_core = max(
        weighted_score(result.region_scores["edge_front"]),
        weighted_score(result.region_scores["core"]),
    )
    return float(stromal_or_hypoxic - edge_or_core)


def axis_margin(result: RealSpatialResult) -> float:
    region_name = best_region_name(result)
    values = result.region_scores[region_name]
    expected = max(float(values[1]), float(values[2]))
    unexpected = max(float(values[0]), float(values[3]))
    return float(expected - unexpected)


def gbm_shallow_core_subtype(result: RealSpatialResult) -> bool:
    return (
        result.cancer_type == "gbm"
        and result.peak_shell is not None
        and int(result.peak_shell["h"]) <= GBM_SHALLOW_CORE_MAX_H
        and result.peak_minus_core < GBM_SHALLOW_CORE_MARGIN
    )


def support_margin(result: RealSpatialResult) -> float:
    return float(
        min(
            result.peak_minus_edge,
            result.peak_minus_core,
            region_margin(result),
            axis_margin(result),
        )
    )


def support_tier(result: RealSpatialResult) -> str:
    margin = support_margin(result)
    if gbm_shallow_core_subtype(result):
        return "gbm_shallow_core"
    if margin >= 0.10:
        return "strong"
    if margin > 0:
        return "borderline_positive"
    if margin > GBM_SHALLOW_CORE_MARGIN:
        return "borderline_negative"
    return "gross"


def model_status(result: RealSpatialResult) -> str:
    failures: List[float] = []
    if result.peak_minus_edge <= 0:
        failures.append(result.peak_minus_edge)
    if result.peak_minus_core <= 0:
        failures.append(result.peak_minus_core)
    if region_margin(result) <= 0:
        failures.append(region_margin(result))
    if axis_margin(result) <= 0:
        failures.append(axis_margin(result))

    if not failures:
        return "fit"
    if gbm_shallow_core_subtype(result):
        return "gbm_shallow_core"
    if len(failures) == 1 and min(failures) > GBM_SHALLOW_CORE_MARGIN:
        return "slight"
    return "gross"


def print_cohort_summary(results: List[Tuple[str, RealSpatialResult]]) -> None:
    total = len(results)
    if total == 0:
        return

    peak_gt_edge = 0
    peak_gt_core = 0
    h_ge_core = 0
    rho_ge_one = 0
    region_counts = {"edge_front": 0, "core": 0, "stromal": 0, "hypoxic": 0}
    axis_counts = {axis: 0 for axis in AXES}
    status_counts = {"fit": 0, "slight": 0, "gbm_shallow_core": 0, "gross": 0}
    tier_counts = {
        "strong": 0,
        "borderline_positive": 0,
        "borderline_negative": 0,
        "gbm_shallow_core": 0,
        "gross": 0,
    }
    gbm_total = 0
    gbm_shallow_core = 0

    for _, result in results:
        edge_score = weighted_score(result.region_scores["edge_front"])
        core_score = weighted_score(result.region_scores["core"])
        peak_gt_edge += int(result.peak_minus_edge > 0)
        peak_gt_core += int(result.peak_minus_core > 0)
        h_ge_core += int(result.peak_shell is not None and result.peak_shell["h"] >= CORE_HOPS)
        rho_ge_one += int(result.rho_tumor >= 1.0)
        status = model_status(result)
        status_counts[status] += 1
        tier_counts[support_tier(result)] += 1
        gbm_total += int(result.cancer_type == "gbm")
        gbm_shallow_core += int(gbm_shallow_core_subtype(result))

        region_name = best_region_name(result)
        axis_name = dominant_axis_name(result, region_name)
        region_counts[region_name] += 1
        axis_counts[axis_name] += 1

    stromal_or_hypoxic = region_counts["stromal"] + region_counts["hypoxic"]
    mech_or_niche = axis_counts["mech"] + axis_counts["niche"]

    print()
    print("=" * 92)
    print("REAL SPATIAL COHORT SUMMARY")
    print("=" * 92)
    print(f"effective weights: {weight_summary()}")
    print(f"samples: {total}")
    print(
        f"- M_eff_peak > M_edge : {peak_gt_edge}/{total} "
        f"(exact p={format_p_value(exact_binomial_tail_at_least(peak_gt_edge, total))})"
    )
    print(
        f"- M_eff_peak > M_core : {peak_gt_core}/{total} "
        f"(exact p={format_p_value(exact_binomial_tail_at_least(peak_gt_core, total))})"
    )
    print(
        f"- h^dagger >= {CORE_HOPS} : {h_ge_core}/{total} "
        f"(exact p={format_p_value(exact_binomial_tail_at_least(h_ge_core, total))})"
    )
    print(f"- rho(A_tumor) >= 1 : {rho_ge_one}/{total}")
    print(
        "- best region in stromal/hypoxic : "
        f"{stromal_or_hypoxic}/{total} "
        f"(exact p={format_p_value(exact_binomial_tail_at_least(stromal_or_hypoxic, total))})"
    )
    print(
        "- dominant axis in mech/niche : "
        f"{mech_or_niche}/{total} "
        f"(exact p={format_p_value(exact_binomial_tail_at_least(mech_or_niche, total))})"
    )
    print(f"- model fit : {status_counts['fit']}/{total}")
    print(f"- model slight : {status_counts['slight']}/{total}")
    print(f"- model GBM shallow-core subtype : {status_counts['gbm_shallow_core']}/{total}")
    print(f"- model unexplained gross : {status_counts['gross']}/{total}")
    print(f"- support strong : {tier_counts['strong']}/{total}")
    print(f"- support borderline positive : {tier_counts['borderline_positive']}/{total}")
    print(f"- support borderline negative : {tier_counts['borderline_negative']}/{total}")
    if gbm_total:
        print(f"- GBM shallow-core subtype only : {gbm_shallow_core}/{gbm_total}")
    print(
        "- region counts : "
        f"edge/front={region_counts['edge_front']}, core={region_counts['core']}, "
        f"stromal={region_counts['stromal']}, hypoxic={region_counts['hypoxic']}"
    )
    print(
        "- axis counts : "
        f"cell={axis_counts['cell']}, niche={axis_counts['niche']}, "
        f"mech={axis_counts['mech']}, immune={axis_counts['immune']}"
    )


def validate_paired_args(left: List[str], right: List[str], label: str) -> None:
    if len(left) != len(right):
        raise ValueError(
            f"{label} inputs must be paired: got {len(left)} left paths and {len(right)} right paths"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-crc-prefix",
        type=str,
        action="append",
        default=[],
        help="Path prefix for a CRC sample, e.g. examples/biology/data/crc/GSM8265211_CTC21P",
    )
    parser.add_argument(
        "--real-pdac-h5",
        type=str,
        action="append",
        default=[],
        help="Path to PDAC filtered_feature_bc_matrix.h5 or raw_feature_bc_matrix.tar.gz",
    )
    parser.add_argument(
        "--real-pdac-spatial",
        type=str,
        action="append",
        default=[],
        help="Path to PDAC spatial.tar.gz or tissue_positions_list.csv.gz",
    )
    parser.add_argument(
        "--real-gbm-h5",
        type=str,
        action="append",
        default=[],
        help="Path to GBM filtered_feature_bc_matrix.h5 or raw_feature_bc_matrix.tar.gz",
    )
    parser.add_argument(
        "--real-gbm-spatial",
        type=str,
        action="append",
        default=[],
        help="Path to GBM spatial.tar.gz or tissue_positions_list.csv.gz",
    )
    parser.add_argument(
        "--real-breast-h5",
        type=str,
        action="append",
        default=[],
        help="Path to breast cancer filtered_feature_bc_matrix.h5 or raw_feature_bc_matrix.tar.gz",
    )
    parser.add_argument(
        "--real-breast-spatial",
        type=str,
        action="append",
        default=[],
        help="Path to breast cancer spatial.tar.gz or tissue_positions_list.csv.gz",
    )
    args = parser.parse_args()

    axis_table = cancer_axis_table()
    print_axis_scores(axis_table)
    print_region_checks(axis_table)
    xs_all, ys_all, _ = collect_transition_pairs(tumor_only=False)
    xs_tumor, ys_tumor, _ = collect_transition_pairs(tumor_only=True)
    A_all = fit_nonnegative_ridge(xs_all, ys_all)
    A_tumor = fit_nonnegative_ridge(xs_tumor, ys_tumor)
    print_matrix_checks(A_all, A_tumor)
    real_results: List[Tuple[str, RealSpatialResult]] = []
    validate_paired_args(args.real_pdac_h5, args.real_pdac_spatial, "PDAC")
    validate_paired_args(args.real_gbm_h5, args.real_gbm_spatial, "GBM")
    validate_paired_args(args.real_breast_h5, args.real_breast_spatial, "breast")

    for crc_prefix in args.real_crc_prefix:
        result = analyze_crc_sample(Path(crc_prefix))
        print_real_result(result)
        real_results.append(("crc", result))
    for pdac_h5, pdac_spatial in zip(args.real_pdac_h5, args.real_pdac_spatial):
        result = analyze_pdac_sample(Path(pdac_h5), Path(pdac_spatial))
        print_real_result(result)
        real_results.append(("pdac", result))
    for gbm_h5, gbm_spatial in zip(args.real_gbm_h5, args.real_gbm_spatial):
        result = analyze_gbm_sample(Path(gbm_h5), Path(gbm_spatial))
        print_real_result(result)
        real_results.append(("gbm", result))
    for breast_h5, breast_spatial in zip(args.real_breast_h5, args.real_breast_spatial):
        result = analyze_breast_sample(Path(breast_h5), Path(breast_spatial))
        print_real_result(result)
        real_results.append(("breast", result))

    print_cohort_summary(real_results)


if __name__ == "__main__":
    main()
