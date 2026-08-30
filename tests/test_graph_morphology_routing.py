import importlib.util
from pathlib import Path
import sys


_SOURCE = Path(__file__).parents[1] / "reality_stone/python/reality_stone/clarus/graph_morphology_routing.py"
_SPEC = importlib.util.spec_from_file_location("graph_morphology_routing_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

EDGE_BUDGET = _MODULE.EDGE_BUDGET
COMMUNITY_A = _MODULE.COMMUNITY_A
LABELS = _MODULE.LABELS
TEMPLATES = _MODULE.TEMPLATES
feature_vector = _MODULE.feature_vector
make_split = _MODULE.make_split
run_routing = _MODULE.run_routing
topology_vector = _MODULE.topology_vector


def test_frozen_generator_features_and_development_apparatus() -> None:
    assert all(len(first) == len(second) == EDGE_BUDGET for first, second in TEMPLATES.values())
    split = make_split(41802, 2)
    assert {len(feature_vector(split[0], family)) for family in ("weight_only", "cluster_aware", "topology_trajectory", "topology_static")} == {33, 39, 138, 46}
    assert all(len(topology_vector(graph)) == 33 for pair in TEMPLATES.values() for graph in pair)
    for replicate in range(2):
        weights = {feature_vector(next(s for s in split if s.label == label and s.replicate == replicate), "weight_only") for label in LABELS}
        assert len(weights) == 1
    for label_ordinal, label in enumerate(LABELS):
        for replicate in range(2):
            sample = next(s for s in split if s.label == label and s.replicate == replicate)
            permutation = _MODULE._permutation(41802, label_ordinal, replicate)
            assert all(sample.community[permutation[u]] == (u in COMMUNITY_A) for u in range(12))
            assert tuple(sample.community[permutation[u]] for u in range(12)) == tuple(u in COMMUNITY_A for u in range(12))
            assert _MODULE.community_vector(sample.edges0, sample.community) == (5 / 11, 6 / 11)
            assert _MODULE.community_vector(sample.edges1, sample.community) == (5 / 11, 6 / 11)
    result = run_routing(41802, train_per_label=4, validation_per_label=4)
    assert all(result["gates"].values())
