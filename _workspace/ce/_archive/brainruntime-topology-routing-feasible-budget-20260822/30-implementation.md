# Implementation

Status: COMPLETE

The route module now factors the target-free cluster admissibility calculation
into one helper and computes one circuit-wide sparse budget from the minimum
admissible support over cues. The receipt records that minimum, fraction, and
exact integer budget. Ranking formulas and route masks are otherwise
unchanged. The benchmark runs full/topology binding and all eight factor arms,
then applies the frozen development gates.

Focused tests passed `10/10`, including delay packet transport, backend
fail-closed behavior, target-free constructor checks, formula separation,
degenerate support, shared-budget feasibility, actual delayed runtime cutoff,
and shared-snapshot integrity.
