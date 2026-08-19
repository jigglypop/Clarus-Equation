"""Locked multi-session benchmark for bounded audited episodic memory.

This validates latest facts, temporal order, provenance, deletion, abstention,
and typed two-hop recall. It does not validate replay, neural consolidation,
geometric memory, persistence across process restarts, or an end-to-end agent.
"""

from __future__ import annotations

from dataclasses import asdict

from .temporal_memory_protocol import TemporalMemoryBenchConfig, _lcb
from .temporal_memory_scoring import _seed_result


def evaluate_temporal_memory(
    config: TemporalMemoryBenchConfig | None = None,
) -> dict[str, object]:
    cfg = config or TemporalMemoryBenchConfig()
    rows = [_seed_result(1_206_000 + i, cfg) for i in range(cfg.seeds)]
    means = {
        name: {
            metric: sum(row[name][metric] for row in rows) / len(rows)
            for metric in rows[0][name]
        }
        for name in rows[0]
    }
    controls = ("existing", "fifo", "update_off", "temporal_order_shuffle")
    comparisons = {
        name: _lcb(
            [row["candidate"]["composite"] - row[name]["composite"] for row in rows],
            seed=202_608_19 + i,
            draws=cfg.bootstrap_draws,
        )
        for i, name in enumerate(controls)
    }
    candidate = means["candidate"]
    full_context_gap = means["full_context"]["composite"] - candidate["composite"]
    provenance_advantage = (
        candidate["provenance_accuracy"]
        - means["evidence_id_removed"]["provenance_accuracy"]
    )
    abstention_advantage = (
        candidate["abstention_accuracy"]
        - means["abstention_off"]["abstention_accuracy"]
    )
    storage_ratio = candidate["storage_count"] / max(
        means["full_context"]["storage_count"], 1.0
    )
    hard_gate = bool(
        candidate["latest_accuracy"] >= 0.95
        and candidate["provenance_accuracy"] >= 0.95
        and candidate["temporal_order_accuracy"] >= 0.95
        and candidate["multihop_accuracy"] >= 0.90
        and candidate["deleted_multihop_accuracy"] >= 0.90
        and candidate["abstention_accuracy"] >= 0.95
        and candidate["stale_error_rate"] <= 0.02
        and candidate["delete_residual_rate"] <= 0.02
        and candidate["capacity_ok"] == 1.0
        and candidate["update_audit_count"] == candidate["expected_update_audit_count"]
        and candidate["delete_audit_count"] == candidate["expected_delete_audit_count"]
        and all(value > 0.08 for value in comparisons.values())
        and full_context_gap <= 0.03
        and provenance_advantage >= 0.80
        and abstention_advantage >= 0.50
        and storage_ratio <= 0.60
    )
    return {
        "schema": "clarus.temporal-memory.validation.v1",
        "protocol": "locked-multi-session-v1",
        "config": asdict(cfg),
        "means": means,
        "lcb_candidate_composite_minus": comparisons,
        "candidate_full_context_composite_gap": full_context_gap,
        "provenance_advantage_vs_removed": provenance_advantage,
        "abstention_advantage_vs_off": abstention_advantage,
        "candidate_storage_ratio_vs_full_context": storage_ratio,
        "hard_gate": hard_gate,
        "benchmark_score": round(100.0 * candidate["composite"], 4),
        "grade": "GO" if hard_gate else "STOP",
        "claim_limit": (
            "synthetic bounded multi-session typed key/value memory with two-hop recall only; "
            "no process-restart persistence, replay, neural consolidation, geometry, or "
            "end-to-end agent claim"
        ),
    }


__all__ = ["TemporalMemoryBenchConfig", "evaluate_temporal_memory"]
