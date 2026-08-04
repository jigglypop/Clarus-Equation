"""Run the already-realized-past intervention/retrodiction audit."""

from __future__ import annotations

import numpy as np

from reality_stone.clarus.past_retrodiction import (
    future_intervention_audit,
    retrodiction_audit,
)


def main() -> None:
    interventions = future_intervention_audit(
        [0.7, 0.3, 0.0],
        np.array([
            [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
            [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]],
        ]),
    )
    retrodiction = retrodiction_audit(
        [0.7, 0.3, 0.0],
        [0.1, 0.8, 1.0],
    )

    print("CE ALREADY-REALIZED-PAST LOOP")
    print(f"  future controls             {interventions.past_marginals.shape[0]}")
    print(f"  past invariant              {interventions.past_invariant}")
    print(
        "  max past residual          "
        f"{interventions.max_past_invariance_residual:.12g}"
    )
    print(f"  future marginals differ     {not np.allclose(*interventions.future_marginals)}")
    print(f"  prior past                  {retrodiction.past_prior}")
    print(f"  posterior past belief       {retrodiction.posterior}")
    print(f"  belief changed              {retrodiction.belief_changed}")
    print(f"  support preserved           {retrodiction.support_preserved}")
    print(
        "  zero-prior past stays zero "
        f"{retrodiction.zero_prior_histories_remain_zero}"
    )


if __name__ == "__main__":
    main()
