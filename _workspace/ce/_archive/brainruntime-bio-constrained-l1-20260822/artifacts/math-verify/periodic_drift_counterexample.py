from __future__ import annotations


def day_map(w: float, *, wake: float, lam: float, kappa: float) -> float:
    pre_sleep = w + wake
    loss_fraction = lam / (1.0 + (pre_sleep / kappa) ** 2)
    return pre_sleep * (1.0 - loss_fraction)


def main() -> None:
    wake = 0.1
    lam = 0.2
    kappa = 1.0
    w = 100.0
    increments: list[float] = []
    for _ in range(1_000):
        next_w = day_map(w, wake=wake, lam=lam, kappa=kappa)
        increments.append(next_w - w)
        w = next_w

    assert min(increments) > wake / 2.0
    assert w > 190.0
    print(
        {
            "initial_w": 100.0,
            "final_w": w,
            "min_increment": min(increments),
            "wake_over_2": wake / 2.0,
            "iterations": len(increments),
            "counterexample": "PASS",
        }
    )


if __name__ == "__main__":
    main()
