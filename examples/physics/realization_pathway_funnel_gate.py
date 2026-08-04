from __future__ import annotations

from reality_stone.clarus.realization_pathway_funnel import (
    spatial_folding_realization_funnel,
)


def main() -> None:
    print("SPATIAL FOLDING REALIZATION PATHWAY FUNNEL")
    for candidate in spatial_folding_realization_funnel():
        print(
            candidate.physical_gate_count,
            "/ 6",
            candidate.name,
            "=>",
            candidate.verdict,
            "[VETO]" if candidate.fatal_veto else "[ACTIVE]",
        )


if __name__ == "__main__":
    main()
