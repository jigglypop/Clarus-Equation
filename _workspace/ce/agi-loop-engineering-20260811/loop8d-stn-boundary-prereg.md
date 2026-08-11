# Loop 8D preregistration — conflict-adaptive STN boundary

Status: LOCKED BEFORE IMPLEMENTATION

## 1. Frozen checkpoint

The Loop 8C PFC–MD–residual trajectory is generated first and then frozen.
The decision layer may read each trial's memory signal but may not write to PFC,
MD, or residual state. No Loop 8B/8C coefficient changes.

## 2. Decision dynamics

For action accumulator `y`, normalized decision time `n`, memory signal `m`,
target sign `q`, coherence `kappa`, and standard Gaussian `xi_n`:

`y_(n+1) = y_n + 0.18 * (0.25 m + q kappa) + sigma_y xi_n`.

The action is the sign of the first boundary hit, or the sign at the registered
deadline. Parameters:

- ID coherence levels `(0.10, 0.20, 0.40, 0.70)` and `sigma_y = 0.35`;
- OOD levels `(0.05, 0.15, 0.30, 0.60)` and `sigma_y = 0.40`;
- deadline `80` steps;
- time cost `0.002` per step;
- low fixed boundary `b0 = 0.70`;
- conflict `C = clamp(1 - kappa/0.70, 0, 1)`;
- adaptive boundary `b = b0 + 1.00 C`;
- matched fixed boundary `b = 1.20`, equal to the ID mean adaptive boundary.

All quantities and probability-law arguments are dimensionless.

## 3. Arms

1. `fixed_low`: `b = 0.70`.
2. `fixed_matched`: `b = 1.20`.
3. `stn_adaptive`: `b = 0.70 + C`.
4. `conflict_shuffle`: the same adaptive equation with conflict values permuted
   across trials while evidence remains in place.

All arms receive identical PFC memory signals, coherence, target, and Gaussian
increments. The shuffle is a destructive control and is never visible to the
candidate.

## 4. Metrics

- high-conflict accuracy and reaction time: two lowest coherence levels;
- low-conflict accuracy and reaction time: two highest levels;
- overall accuracy, reaction time, and utility
  `(+1 correct, -1 wrong) - 0.002 * RT`;
- paired bootstrap LCB with 3,000 draws;
- exact equality of the memory trace before and after every decision arm;
- timeout rate and finite-state check.

ID/OOD each use 32 seeds x 192 trials.

## 5. Locked gates

1. High-conflict accuracy LCB, adaptive minus fixed-low >= `+0.03` ID/OOD.
2. High-conflict accuracy LCB, adaptive minus fixed-matched >= `+0.015` ID/OOD.
3. Adaptive minus shuffled-conflict overall accuracy LCB >= `+0.02` ID/OOD.
4. High-conflict mean RT is at least `2` steps longer than fixed-low ID/OOD.
5. Low-conflict mean RT cost versus fixed-low is at most `3` steps ID/OOD.
6. Utility LCB, adaptive minus fixed-matched >= `0` ID/OOD.
7. Frozen memory traces are bit-identical across arms.
8. Accumulators are finite, `abs(y) <= 10`, and timeout rate <= `0.05`.
9. `future_reads = 0`, `environment_clone_calls = 0`.

Score is `100 GO` only if all gates pass, otherwise `0 STOP`. Passing supports
the narrow claim that conflict-aligned boundary allocation beats equal-average
static allocation in this task. It does not prove biological STN identity or
authorize runtime integration.

## 6. Prohibited after-result actions

- no boundary, noise, coherence, time-cost, or deadline sweep;
- no replacement of the matched-threshold control;
- no seed deletion or timeout exclusion;
- no change to memory dynamics after observing decision results.
