# CE-JS2: failed joint selection to a new stable candidate

2026-09-21. Authoritative proof:
[chapter 31](../../paper/후속연구_기록과_상태선택/31_질량을_가진_중성장과_안정한_공동선택.md).

The original JS1 radial instability remains valid in that original model.
JS2 adds exactly one periodic, massive, neutral five-dimensional Dirac field.
Its complete finite winding determinant gives a new global existence theorem.
All field content, masses, matching and EFT limits are explicit inputs.

Run from the repository root:

    .venv\Scripts\python.exe experiments/ce_joint_stabilization_20260921/derive_stabilization.py
    .venv\Scripts\python.exe experiments/ce_joint_stabilization_20260921/certify_minimum.py

The first script computes all five derivatives/Hessian directions, successful and
failed mass comparisons, and the spectral/radial inverse. The second script uses
outward interval arithmetic, infinite-sum tails and an analytic third-derivative
bound to prove a unique strict local minimum in a specified five-dimensional ball.
It does not claim global uniqueness, all-loop accuracy, the natural selection of
the added field, or observational validation. The minimum vacuum energy is negative.

The earlier unfinished light/joint auxiliary script in the reverse-bridge folder
is not this proof and is not included in this result's evidence.
The 103 RB claim groups and their check count are not a completion percentage.
