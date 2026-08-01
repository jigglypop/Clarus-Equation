from __future__ import annotations

from reality_stone.clarus import origin_life
from reality_stone.clarus.origin_life import (
    evaluate_autocatalysis,
    evaluate_boundary,
    evaluate_heredity,
)
from reality_stone.clarus.origin_life_toy import run_toy_gate


def _rna_data() -> tuple[dict, dict]:
    cycles = {
        3: {
            "negative": (0.3, 1.0, 1.1),
            "seed_1uM": (5.8, 6.2, 7.6),
            "seed_2uM": (12.1, 12.5, 14.2),
        }
    }
    serial = {
        1: {
            "negative": (1.0, 1.1, 1.2),
            "seed_1uM": (4.8, 5.1, 5.5),
            "seed_2uM": (9.5, 9.9, 10.2),
        },
        7: {
            "negative": (1.8, 1.9, 2.6),
            "seed_1uM": (3.7, 4.5, 5.5),
            "seed_2uM": (10.4, 11.3, 14.1),
        },
    }
    return cycles, serial


def test_autocatalysis_requires_effect_separation_and_serial_persistence() -> None:
    cycles, serial = _rna_data()
    passing = evaluate_autocatalysis(cycles, serial)
    assert passing["passed"]

    cycles[3]["seed_1uM"] = (0.8, 1.0, 1.2)
    assert not evaluate_autocatalysis(cycles, serial)["passed"]


def test_autocatalysis_extractor_ignores_later_summary_rows(monkeypatch) -> None:
    raw_rows = []
    summary_rows = []
    for cycle in range(4):
        raw_rows.append((None, cycle, *(float(cycle + offset) for offset in range(9))))
        summary_rows.append((None, cycle, 99.0, 0.1, 3.0, 99.0, 0.1, 3.0, 99.0, 0.1, 3.0))
    monkeypatch.setattr(
        origin_life,
        "read_xlsx_sheet",
        lambda _path, _sheet: tuple(raw_rows + summary_rows),
    )

    extracted = origin_life.extract_salibi_autocatalysis("unused.xlsx")

    assert extracted[3]["negative"] == (3.0, 4.0, 5.0)
    assert extracted[3]["seed_2uM"] == (9.0, 10.0, 11.0)


def test_boundary_gate_rejects_nonpersistent_compartment() -> None:
    boundary_data = {
        "amplification_ratios": {
            1: {"liposome": 50.0, "bulk": 100.0},
            2: {"liposome": 50.0, "bulk": 49.0},
            3: {"liposome": 72.0, "bulk": 18.0},
            4: {"liposome": 198.0, "bulk": 14.0},
            5: {"liposome": 23.0, "bulk": 3.9},
        },
        "liposome": {1: {"16h": 1_700.0}, 5: {"16h": 6_900.0}},
        "bulk": {1: {"16h": 4_100.0}, 6: {"16h": 100.0}},
    }
    assert evaluate_boundary(boundary_data)["passed"]

    boundary_data["liposome"][5]["16h"] = 1_000.0
    assert not evaluate_boundary(boundary_data)["passed"]


def test_heredity_gate_requires_two_unwarned_independent_campaigns() -> None:
    variant = {
        "mutation": "A>G",
        "region": "replicase",
        "trajectory": [0.0001, 0.01, 0.4],
        "low_depth_warning": False,
    }
    assert evaluate_heredity({"campaign_1": [variant], "campaign_2": [variant]})[
        "passed"
    ]

    warned = {**variant, "low_depth_warning": True}
    assert not evaluate_heredity({"campaign_1": [variant], "campaign_2": [warned]})[
        "passed"
    ]


def test_paired_ablation_toy_is_reproducible_and_labeled_as_construction() -> None:
    first = run_toy_gate(draws=8, seed=7)
    second = run_toy_gate(draws=8, seed=7)

    assert first == second
    assert first["passed"]
    assert first["paired_parameters_across_conditions"]
    assert first["operational_necessity_counterexamples"][
        "all_three_ablations_can_pass_after_assumption_changes"
    ]
    assert not first["universal_necessity_proven"]
    assert "does not prove" in first["claim_not_supported"]
