"""Run the zero-coordinate boundary target-selection audit."""

from __future__ import annotations

import numpy as np

from reality_stone.clarus.zero_dimensional_targeting import (
    autonomous_d0_targeting,
    boundary_history_targeting,
    coordinate_target_bits,
    target_fixed_point_audit,
)


def main() -> None:
    light_year_m = 9.4607304725808e15
    autonomous = autonomous_d0_targeting(8)
    boundary = boundary_history_targeting(
        np.array([
            [[4.0, 3.0, 1.0], [5.0, 2.0, 0.5]],
            [[3.0, 4.0, 1.5], [4.0, 3.0, 0.25]],
        ]),
        history_prior=[0.4, 0.6],
        time_weights=[0.25, 0.75],
        beta=20.0,
    )
    no_fixed = target_fixed_point_audit([[2.0, 1.0], [1.0, 2.0]])
    unique_fixed = target_fixed_point_audit([[0.0, 1.0], [0.0, 2.0]])

    print("CE ZERO-DIMENSIONAL TARGETING LOOP")
    print(f"  autonomous candidates       {autonomous.candidate_count}")
    print(f"  intrinsic information bits {autonomous.intrinsic_information_bits:.12g}")
    print(f"  target label bits           {autonomous.target_label_bits:.12g}")
    print(f"  autonomous unique target    {autonomous.unique_target_from_d0_state}")
    print(f"  1 ly / 1 m target bits      {coordinate_target_bits(light_year_m, 1.0):.12g}")
    print(f"  boundary scores             {boundary.location_scores}")
    print(f"  boundary target             {boundary.minimizing_locations}")
    print(f"  boundary unique             {boundary.unique_target}")
    print(f"  localized actuation         {boundary.localized_actuation_derived}")
    print(f"  shortcut created            {boundary.spatial_shortcut_created}")
    print(f"  paradox fixed points        {no_fixed.fixed_points}")
    print(f"  consistent fixed points     {unique_fixed.fixed_points}")


if __name__ == "__main__":
    main()
