"""Run the CE time-travel causality proof/counterexample loop."""

from __future__ import annotations

from reality_stone.clarus.time_travel_causality import (
    global_time_function_audit,
    periodic_time_ctc_audit,
)


def main() -> None:
    local = periodic_time_ctc_audit(period=1.0, steps=128)
    global_time = global_time_function_audit((0.2, 0.3, 0.5))

    print("CE TIME-TRAVEL CAUSALITY LOOP 1")
    print("local determinant/lapse no-go")
    print(f"  positive lapse             {local.positive_lapse}")
    print(f"  determinant <= 1           {local.determinant_bound}")
    print(f"  locally future timelike    {local.locally_future_timelike}")
    print(f"  closes in time quotient    {local.closes_in_quotient}")
    print(f"  proper time accumulated    {local.accumulated_proper_time:.12g}")
    print(f"  local no-go refuted        {local.refutes_local_no_go}")
    print("global-time-function theorem")
    print(f"  strictly increasing        {global_time.strictly_increasing}")
    print(f"  total time change          {global_time.total_time_change:.12g}")
    print(f"  closed endpoint compatible {global_time.closure_requires_zero_change}")
    print(f"  CTC excluded               {global_time.ctc_excluded}")
    print(
        "conclusion: local CE inequalities do not exclude CTCs; "
        "CE A1 excludes them by its global-time-function assumption"
    )


if __name__ == "__main__":
    main()
