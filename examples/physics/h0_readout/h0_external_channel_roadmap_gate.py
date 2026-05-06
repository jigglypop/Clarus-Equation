"""Roadmap gate for extending the H0 readout test beyond TDCOSMO.

The TDCOSMO gates now close the provenance loop from public notebook sampler
composition to source-aware readout roles. This companion gate names the next
independent channel classes and records what would count as a clean
falsification-grade extension.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalChannel:
    name: str
    expected_role: str
    expected_readout: str
    required_source: str
    blocker: str
    priority: int


CHANNELS = [
    ExternalChannel(
        name="BAO+SN inverse distance ladder",
        expected_role="global standard-ruler closure",
        expected_readout="low-side/global branch",
        required_source="public covariance or compressed likelihood with ruler and SN nuisance roles",
        blocker="map covariance labels into observable/local/global role sets",
        priority=1,
    ),
    ExternalChannel(
        name="SH0ES-style local distance ladder",
        expected_role="local calibrator endpoint closure",
        expected_readout="high-side/local branch",
        required_source="public ladder covariance with Cepheid/TRGB/SN calibration blocks",
        blocker="recover calibration graph rather than treating the final H0 as a scalar",
        priority=2,
    ),
    ExternalChannel(
        name="GW standard sirens",
        expected_role="mixed distance-redshift bridge",
        expected_readout="bridge or intermediate branch",
        required_source="event-level distance-redshift posterior or population covariance",
        blocker="separate detector distance uncertainty from host/redshift anchoring",
        priority=3,
    ),
    ExternalChannel(
        name="CMB acoustic-scale inference",
        expected_role="early global horizon closure",
        expected_readout="low-side/global branch",
        required_source="public parameter covariance or likelihood with acoustic-scale roles",
        blocker="avoid double-counting BAO/SN priors when comparing to late probes",
        priority=4,
    ),
]


def readiness_score(channel: ExternalChannel) -> int:
    score = 0
    if "covariance" in channel.required_source or "posterior" in channel.required_source:
        score += 1
    if "graph" in channel.blocker or "roles" in channel.blocker or "roles" in channel.required_source:
        score += 1
    if channel.priority <= 2:
        score += 1
    return score


def main() -> int:
    print("# H0 External Channel Roadmap Gate")
    print()
    print("| priority | channel | expected role | expected readout | next source requirement | blocker | readiness |")
    print("|---:|---|---|---|---|---|---:|")

    scored = sorted(CHANNELS, key=lambda item: item.priority)
    for channel in scored:
        print(
            f"| {channel.priority} | {channel.name} | {channel.expected_role} | "
            f"{channel.expected_readout} | {channel.required_source} | {channel.blocker} | "
            f"{readiness_score(channel)}/3 |"
        )

    top = scored[0]
    if top.name != "BAO+SN inverse distance ladder":
        raise SystemExit("unexpected roadmap ordering")
    if readiness_score(top) < 3:
        raise SystemExit("top external channel is not ready enough for the next pass")

    print()
    print("Verdict: next falsification target is BAO+SN inverse-distance-ladder covariance provenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
