import numpy as np

from reality_stone.clarus.replay_consolidation import ReplayConsolidationConfig
from reality_stone.clarus.replay_consolidation_bridge import ReplayCutoffBridge
from reality_stone.clarus.replay_consolidation_models import ReplayEpisode


def test_bridge_recalls_after_physical_episodic_cutoff() -> None:
    bridge = ReplayCutoffBridge(
        seed=11,
        values=("Busan", "Seoul", "__TOMBSTONE__"),
        config=ReplayConsolidationConfig(dimension=96),
    )
    bridge.stage(ReplayEpisode("A", "lives_in", "Busan", 32, 3.0))
    bridge.consolidate()
    bridge.detach_episodic()
    assert bridge.staged_episode_count == 0
    assert bridge.recall("A", "lives_in").value == "Busan"
    assert bridge.recall("unknown", "lives_in").abstained


def test_exact_radius_attractor_probe_is_supported() -> None:
    bridge = ReplayCutoffBridge(
        seed=12,
        values=("Busan", "Seoul", "__TOMBSTONE__"),
        config=ReplayConsolidationConfig(dimension=128),
    )
    bridge.stage(ReplayEpisode("A", "lives_in", "Busan", 32, 3.0))
    bridge.stage(ReplayEpisode("B", "lives_in", "Seoul", 32, 3.0))
    bridge.consolidate()
    probe = bridge._model.probe_attractor(  # focused patch characterization
        "Busan",
        flip_rate=0.35,
        rng=np.random.default_rng(22),
        exact_hamming=True,
    )
    assert probe.final_similarity > probe.initial_similarity
