from __future__ import annotations


def bottom_loss(total_loss: float, top_share: float, top_loss: float) -> float:
    if not 0.0 < top_share < 1.0:
        raise ValueError("top_share must be in (0, 1)")
    return (total_loss - top_share * top_loss) / (1.0 - top_share)


def main() -> None:
    total = 0.18
    rows = []
    for share in (0.2, 0.5, 0.8):
        for top in (0.0, 0.025, 0.05):
            bottom = bottom_loss(total, share, top)
            reconstructed = share * top + (1.0 - share) * bottom
            assert abs(reconstructed - total) < 1e-12
            rows.append(
                {
                    "top_strength_share": share,
                    "top_fractional_loss": top,
                    "required_bottom_loss": bottom,
                }
            )
    print({"identity": "PASS", "rows": rows})


if __name__ == "__main__":
    main()
