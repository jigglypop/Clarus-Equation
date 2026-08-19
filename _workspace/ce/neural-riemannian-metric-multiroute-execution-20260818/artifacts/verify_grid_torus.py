"""Independent, deliberately expensive recomputation of frozen Grid v3 output."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import run_grid_torus as runner

P = Path(__file__).resolve().with_name("grid-torus-v3-results.json")
EXPECT = {("Q", 1), ("Q", 2), ("R", 1), ("R", 2), ("R", 3), ("S", 1)}
RTOL, ATOL = 1e-8, 1e-10

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def ids(xs): return {(x["bout_id"], x["block_id"]) for x in xs}
def nonnegative(x): assert math.isfinite(x) and x >= 0

def assert_same(observed, expected, path="root"):
    """Exact discrete equality and documented tolerance for recomputed floats."""
    if isinstance(expected, dict):
        assert set(observed) == set(expected), path
        for key in expected: assert_same(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(observed) == len(expected), path
        for i, value in enumerate(expected): assert_same(observed[i], value, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert math.isfinite(observed) and np.isclose(observed, expected, rtol=RTOL, atol=ATOL), (path, observed, expected)
    else:
        assert observed == expected, (path, observed, expected)

def recompute():
    """Parse official intervals, rebuild blocks/splits/chart/endpoints from raw NPZs."""
    runner.load_analysis_dependencies()
    facts, modules = runner.facts(), []
    for rat, filename, day in runner.FILES:
        loaded = np.load(runner.DATA / filename, allow_pickle=True)
        for module in range(1, 4):
            key = f"spikes_mod{module}"
            if key not in loaded: continue
            spikes = loaded[key].item()
            states = {name: runner.blocks(runner.bin_bouts(spikes, facts[f"rat_{rat.lower()}_{tag}{day}"]), name)
                      for name, tag in (("wake", "OF"), ("REM", "REM"), ("SWS", "SWS"))}
            assert all(states.values()), (rat, module, "empty_state_blocks")
            modules.append(runner.run_module(rat, module, states))
    return modules

def main():
    data = json.loads(P.read_text())
    assert data["schema"] == "nrm-grid-torus-v3"
    assert set((x["rat"], x["module"]) for x in data["modules"]) == EXPECT
    root = Path(__file__).resolve().parents[1] / "artifacts" / "input" / "grid_torus"
    paths = {"utils.py": root / "GridCellTorus-code" / "utils.py",
        "rat_q_grid_modules_1_2.npz": root / "extracted" / "Toroidal_topology_grid_cell_data" / "rat_q_grid_modules_1_2.npz",
        "rat_r_day2_grid_modules_1_2_3.npz": root / "extracted" / "Toroidal_topology_grid_cell_data" / "rat_r_day2_grid_modules_1_2_3.npz",
        "rat_s_grid_modules_1.npz": root / "extracted" / "Toroidal_topology_grid_cell_data" / "rat_s_grid_modules_1.npz"}
    assert {k: sha(v) for k, v in paths.items()} == data["inputs_sha256"]
    frozen = data["frozen"]
    assert frozen["physical_h"] == "unmeasured" and "not_longitudinal" in frozen["wake_reference"] and "wake_C_20_percent" in frozen["split"]
    assert data["population_inference"] == "not_performed_N_equals_3"
    for m in data["modules"]:
        assert m["status"] == "COMPLETE", m; assert set(m["states"]) == {"wake", "REM", "SWS"}; chart = m["chart"]
        assert chart["fit"] == "wake_C_only" and chart["dimension"] == 6 and chart["n_blocks"] > 0 and chart["parameter_sha256"]
        for state, r in m["states"].items():
            assert r["chart_parameter_sha256"] == chart["parameter_sha256"]
            c, a, b = map(ids, (r["calibration_C"], r["analysis_A"], r["analysis_B"])); assert not (c & a or c & b or a & b) and a and b
            if state != "wake": assert not c
            for role in ("A", "B"):
                t, matrix = r[f"topology_{role}"], np.asarray(r[f"mobility_precision_{role}"])
                assert len(t["h1_top3_lifetime"]) >= 2 and t["n_points"] > 0 and t["n_blocks"] > 0 and all(math.isfinite(x) and x >= 0 for x in t["h1_top3_lifetime"])
                assert matrix.shape == (6, 6) and np.isfinite(matrix).all() and np.allclose(matrix, matrix.T) and np.linalg.eigvalsh(matrix).min() > 0
                assert math.isfinite(r[f"mobility_{role}"]["condition"]) and r[f"mobility_{role}"]["condition"] > 0
            for x in r["within_split_noise"].values(): nonnegative(x)
        assert set(m["contrasts"]) == {"wake", "REM", "SWS"}
        for c in m["contrasts"].values():
            assert c["state_reference"] == "wake"
            for role in ("primary", "swap"):
                assert set(c[role]) >= {"topology_ratio", "metric_ratio"}
                for x in c[role].values(): nonnegative(x)
            expected = c["primary"]["topology_ratio"] <= 1 and c["primary"]["metric_ratio"] > 1 and c["swap"]["topology_ratio"] <= 1 and c["swap"]["metric_ratio"] > 1
            assert c["dissociation_compatible"] == expected and c["dissociation_compatible_label"] == "one_split_heuristic_no_p_or_population_inference"
    expected_modules = recompute()
    assert len(expected_modules) == 6
    for observed, expected in zip(data["modules"], expected_modules): assert_same(observed, expected, f"module.{observed['rat']}{observed['module']}")
    print('{"status":"PASS","schema":"nrm-grid-torus-v3","modules":6,"recomputed":"raw_npz"}')

if __name__ == "__main__": main()
