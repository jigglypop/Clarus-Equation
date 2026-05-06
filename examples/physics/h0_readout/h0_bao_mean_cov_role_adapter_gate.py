"""Adapt DESI BAO mean/covariance files into H0 readout role labels.

This gate deliberately stops before producing an H0 number. The first external
test beyond TDCOSMO should verify source roles: BAO measurements are distance
ratios to a standard ruler, so the adapter should expose them as global
standard-ruler closure nodes rather than local endpoint calibrators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA = Path(__file__).with_name("h0_real_data") / "CobayaSampler_bao_data"
MEAN = DATA / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
COV = DATA / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"


@dataclass(frozen=True)
class BaoDatum:
    z: float
    value: float
    quantity: str

    @property
    def node(self) -> str:
        z_label = f"{self.z:.3f}".replace(".", "p")
        return f"bao_{self.quantity}_z{z_label}"


def read_mean(path: Path) -> list[BaoDatum]:
    out: list[BaoDatum] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        z, value, quantity = line.split()
        out.append(BaoDatum(float(z), float(value), quantity))
    return out


def read_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def validate_covariance(data: list[BaoDatum], cov: list[list[float]]) -> None:
    n = len(data)
    if len(cov) != n:
        raise ValueError(f"covariance row count {len(cov)} does not match mean count {n}")
    for i, row in enumerate(cov):
        if len(row) != n:
            raise ValueError(f"covariance row {i} has length {len(row)}; expected {n}")
        if row[i] <= 0.0:
            raise ValueError(f"covariance diagonal {i} must be positive")
        for j in range(i + 1, n):
            if abs(row[j] - cov[j][i]) > 1e-10:
                raise ValueError(f"covariance must be symmetric at ({i}, {j})")


def role_summary(data: list[BaoDatum]) -> dict[str, list[str]]:
    nodes = [datum.node for datum in data]
    return {
        "observable_nodes": nodes,
        "local_nodes": [],
        "global_nodes": ["sound_horizon_standard_ruler", "bao_distance_ratio_closure", *nodes],
    }


def main() -> int:
    if not MEAN.exists() or not COV.exists():
        raise SystemExit(
            "DESI BAO mean/cov files are missing; run h0_bao_sn_source_scout_gate.py "
            "after fetching CobayaSampler/bao_data."
        )

    data = read_mean(MEAN)
    cov = read_matrix(COV)
    validate_covariance(data, cov)
    roles = role_summary(data)
    quantities = sorted({datum.quantity for datum in data})
    diagonal_min = min(cov[i][i] for i in range(len(cov)))
    diagonal_max = max(cov[i][i] for i in range(len(cov)))

    print("# H0 BAO Mean/Cov Role Adapter Gate")
    print()
    print(f"mean_file = {MEAN.name}")
    print(f"cov_file = {COV.name}")
    print(f"measurements = {len(data)}")
    print(f"quantities = {', '.join(quantities)}")
    print(f"local_nodes = {len(roles['local_nodes'])}")
    print(f"global_nodes = {len(roles['global_nodes'])}")
    print(f"covariance_diagonal_range = {diagonal_min:.8e} .. {diagonal_max:.8e}")
    print()
    print("| node | z | quantity | role |")
    print("|---|---:|---|---|")
    for datum in data:
        print(f"| {datum.node} | {datum.z:.3f} | {datum.quantity} | global standard-ruler observable |")
    print()
    print("Verdict: DESI BAO mean/covariance labels map to a global closure readout candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
