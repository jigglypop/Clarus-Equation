"""Read-only diagnostic on the already-opened A5 development cohort."""

from __future__ import annotations

import json

import numpy as np

import run_a4_dynamic_metric as a4
import run_a5_incremental_metric as a5


def main() -> None:
    manifest = json.loads(a5.MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for asset in manifest["assets"][:3]:
        prep = a4.split_and_standardize(a4.load_asset_verified(asset))
        bidx = a4.a3.pair_indices(prep["b_domain"], a4.STATE_SHIFT, 1)
        tidx = a4.a3.pair_indices(prep["test_domain"], a4.STATE_SHIFT, 1)
        arms, _ = a4.prepare_arms(prep)
        base_geom, _, _ = arms["fixed_geometry"]
        arm_geom, alpha, activation = arms["a4"]
        _, delta_b, _ = a5.base_and_delta_features(prep, bidx, base_geom, arm_geom, alpha, activation)
        _, delta_t, _ = a5.base_and_delta_features(prep, tidx, base_geom, arm_geom, alpha, activation)
        b_rms = float(np.sqrt(np.mean(delta_b * delta_b)))
        t_rms = float(np.sqrt(np.mean(delta_t * delta_t)))
        rows.append({"asset_id": asset["id"], "construction_rms": b_rms, "test_rms": t_rms, "test_to_construction": t_rms / b_rms})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
