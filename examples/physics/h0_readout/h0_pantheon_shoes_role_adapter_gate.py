"""Adapt Pantheon+SH0ES distance data into H0 readout role labels.

This gate reads the public Pantheon+SH0ES distance table and extracts the
source roles needed for the local distance-ladder contrast channel. The key
flags are:

- IS_CALIBRATOR: SN host has a Cepheid distance anchor.
- USED_IN_SH0ES_HF: SN is used in the SH0ES Hubble-flow sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA = (
    Path(__file__).with_name("h0_real_data")
    / "PantheonPlusSH0ES_DataRelease"
    / "Pantheon+_Data"
    / "4_DISTANCES_AND_COVAR"
)
DISTANCES = DATA / "Pantheon+SH0ES.dat"
COVARIANCE = DATA / "Pantheon+SH0ES_STAT+SYS.cov"


@dataclass(frozen=True)
class ShoesCounts:
    rows: int
    columns: int
    calibrators: int
    hubble_flow: int
    overlap: int
    surveys: int


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    rows = [line.split() for line in lines[1:] if line.strip()]
    return header, rows


def counts(path: Path = DISTANCES) -> ShoesCounts:
    header, rows = read_table(path)
    index = {name: i for i, name in enumerate(header)}
    required = ["IS_CALIBRATOR", "USED_IN_SH0ES_HF", "IDSURVEY"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"missing Pantheon+SH0ES columns: {', '.join(missing)}")

    calibrators = 0
    hubble_flow = 0
    overlap = 0
    surveys = set()
    for row in rows:
        is_calibrator = int(float(row[index["IS_CALIBRATOR"]]))
        is_hubble_flow = int(float(row[index["USED_IN_SH0ES_HF"]]))
        calibrators += is_calibrator
        hubble_flow += is_hubble_flow
        overlap += 1 if is_calibrator and is_hubble_flow else 0
        surveys.add(row[index["IDSURVEY"]])

    return ShoesCounts(
        rows=len(rows),
        columns=len(header),
        calibrators=calibrators,
        hubble_flow=hubble_flow,
        overlap=overlap,
        surveys=len(surveys),
    )


def role_summary(stats: ShoesCounts) -> dict[str, list[str]]:
    return {
        "observable_nodes": ["shoes_h0_ladder_readout"],
        "local_nodes": [
            "cepheid_calibrator_hosts",
            "sn_hubble_flow_endpoint",
            "local_distance_ladder_zero_point",
        ],
        "global_nodes": [],
    }


def main() -> int:
    if not DISTANCES.exists():
        raise SystemExit(f"missing distance table: {DISTANCES}")
    if not COVARIANCE.exists():
        raise SystemExit(f"missing covariance table: {COVARIANCE}")

    stats = counts()
    roles = role_summary(stats)

    print("# H0 Pantheon+SH0ES Role Adapter Gate")
    print()
    print(f"distance_file = {DISTANCES.name}")
    print(f"covariance_file = {COVARIANCE.name}")
    print(f"rows = {stats.rows}")
    print(f"columns = {stats.columns}")
    print(f"surveys = {stats.surveys}")
    print(f"calibrator_rows = {stats.calibrators}")
    print(f"hubble_flow_rows = {stats.hubble_flow}")
    print(f"calibrator_hubble_flow_overlap = {stats.overlap}")
    print(f"local_nodes = {len(roles['local_nodes'])}")
    print(f"global_nodes = {len(roles['global_nodes'])}")
    print()
    print("| role node | interpretation |")
    print("|---|---|")
    print("| cepheid_calibrator_hosts | local Cepheid-host absolute-distance anchors |")
    print("| sn_hubble_flow_endpoint | local Hubble-flow SN endpoint sample |")
    print("| local_distance_ladder_zero_point | ladder zero-point closure |")
    print()

    if stats.calibrators <= 0 or stats.hubble_flow <= 0:
        raise SystemExit("Pantheon+SH0ES local ladder roles were not found")
    if roles["global_nodes"]:
        raise SystemExit("Pantheon+SH0ES role adapter should not declare global standard-ruler nodes")

    print("Verdict: Pantheon+SH0ES source labels map to a local distance-ladder readout candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
