"""Run complete-history, possibility-shift, and d=0 origin gates."""

from __future__ import annotations

import numpy as np

from reality_stone.clarus.possibility_space import (
    complete_history_readout,
    dimension_origin_audit,
    possibility_shift_audit,
    target_possibility_shift,
)


def main() -> None:
    histories = np.array([
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 0.4, 1.4, 3.0],
        [9.0, 8.0, 7.0, 6.0],
        [0.0, 0.6, 1.2, 2.5],
    ])
    complete = complete_history_readout(
        histories,
        prior=[0.2, 0.2, 0.2, 0.4],
        time_weights=[0.1, 0.2, 0.3, 0.4],
    )
    shift = possibility_shift_audit(
        [0.2, 0.2, 0.2, 0.4],
        past_ids=[1, 1, 2, 1],
        realized_past_id=1,
        target=[False, True, True, False],
        strength=3.0,
    )
    origin = dimension_origin_audit()
    extreme_empty_target, _, _ = target_possibility_shift(
        [0.5, 0.5],
        [False, False],
        strength=1000.0,
    )

    print("CE POSSIBILITY-SPACE LOOP")
    print(f"  complete histories read     {complete.all_histories_used}")
    print(f"  complete time grids read    {complete.all_times_used}")
    print(f"  ensemble history readout    {complete.ensemble_readout:.12g}")
    print(f"  target mass before          {shift.prior_target_mass:.12g}")
    print(f"  target mass after           {shift.posterior_target_mass:.12g}")
    print(f"  target mass increased       {shift.target_mass_increased}")
    print(
        "  target mass numerically up  "
        f"{shift.target_mass_numerically_increased}"
    )
    print(
        "  incompatible pasts zero    "
        f"{shift.incompatible_pasts_remain_impossible}"
    )
    print(f"  support preserved           {shift.support_preserved_by_finite_tilt}")
    print(
        "  float support resolved      "
        f"{shift.floating_point_support_fully_resolved}"
    )
    print(f"  u=1000 empty target finite  {np.all(np.isfinite(extreme_empty_target))}")
    print(f"  dimension roots             {origin.algebraic_roots}")
    print(f"  d0 internal observer        {origin.d0_supports_internal_observer}")
    print(f"  d0 temporally prior proved  {origin.temporal_predecessor_derived}")
    print(f"  d0->d3 dynamics proved      {origin.d0_to_d3_dynamics_derived}")
    print(f"  origin status               {origin.status}")


if __name__ == "__main__":
    main()
