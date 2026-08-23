import inspect
import json

import torch

from reality_stone.clarus.experiments.runtime_experience_delayed_binding import (
    _experience_block,
    _local_pair,
    analyze_weight_row,
)


from _run_paths import run_dir


DEVELOPMENT = run_dir("brainruntime-local-stochastic-binding-20260822") / "artifacts" / "development-results.json"


def _first_weight() -> tuple[int, torch.Tensor]:
    payload = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    return int(row["seed"]), torch.tensor(row["learned"]["candidate_weights"])


def test_local_pair_reads_only_post_and_presynaptic_event_trace() -> None:
    post = torch.tensor((0.0, .8, 0.0, 0.0))
    pre = torch.tensor((.2, 0.0, 0.0, 0.0))
    pair = _local_pair(post, pre)
    assert pair[1, 0] > 0.0
    assert torch.count_nonzero(pair) == 1
    source = inspect.getsource(_local_pair).lower()
    assert not any(token in source for token in ("target", "decoder", "reward", "endpoint", "label"))


def test_block_has_exact_delay_single_write_and_no_projection() -> None:
    _seed, weight = _first_weight()
    result = _experience_block(weight, condition="learned")
    assert result["mid_block_weight_unchanged"]
    assert result["block_boundary_count"] == 1
    assert result["mutation_count"] == 1
    assert result["outside_support_delta_norm"] == 0.0
    assert result["raw_install_max_error"] <= 1e-7
    assert all(episode["hidden_prearrival_max"] <= 1e-7 for episode in result["episodes"])
    module_source = inspect.getsource(inspect.getmodule(_experience_block)).lower()
    assert "structural_projection" not in module_source


def test_one_weight_row_replays_experience_after_store_cutoff() -> None:
    seed, weight = _first_weight()
    row = analyze_weight_row(seed, weight)
    assert row["status"] == "EXPERIENCE_DELAYED_BINDING_PASS"
    assert row["learned_evaluation"]["accuracy"] == 1.0
    assert all(row["gates"].values())
    assert not row["endpoint_opened"]
