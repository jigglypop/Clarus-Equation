from __future__ import annotations

import numpy as np

from reality_stone.clarus.spatial_folding import casimir_cell_conversion_audit
from reality_stone.clarus.targeted_spatial_actuation import (
    causal_target_delivery_audit,
    target_localization_audit,
    throat_scale_window_audit,
)


def main() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    broadcast = target_localization_audit(
        np.ones((3, 3)) * density,
        required_density_j_m3=density,
    )
    delivery = causal_target_delivery_audit(
        distance_m=9.4607304725808e15,
        candidate_count=8,
        requested_activation_s=0.0,
    )
    scale = throat_scale_window_audit(
        candidate_negative_density_j_m3=density,
        ce_correlation_length_m=6.65e-15,
    )

    print("CE TARGET-TO-STRESS ACTUATION LOOP")
    print(" broadcast localized", broadcast.all_commands_localized)
    print(" one-ly earliest command s", delivery.earliest_delivery_s)
    print(" instant remote command", delivery.instantaneous_adaptive_activation)
    print(" density minimum radius m", scale.minimum_radius_from_density_m)
    print(" coherence maximum radius m", scale.maximum_radius_from_coherence_m)
    print(" radius window exists", scale.feasible_radius_window_exists)
    print(" stable wormhole", scale.stable_wormhole_established)


if __name__ == "__main__":
    main()
