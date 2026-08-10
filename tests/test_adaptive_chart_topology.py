from pathlib import Path
import numpy as np
from reality_stone.clarus.adaptive_chart_topology import AdaptiveChartBank, run_chart_topology_gate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "adaptive_chart_topology_v1.json"

def test_bank_reuses_near_chart_and_creates_far_chart() -> None:
    bank = AdaptiveChartBank(np.ones(2), 0.5, 0.1, 3)
    bank.observe(np.array([0.0, 0.0]))
    bank.observe(np.array([0.1, 0.0]))
    bank.observe(np.array([1.0, 1.0]))
    assert len(bank.charts) == 2

def test_gate_has_no_external_data() -> None:
    report = run_chart_topology_gate(CONFIG)
    assert report["resource_usage"]["external_download_bytes"] == 0
