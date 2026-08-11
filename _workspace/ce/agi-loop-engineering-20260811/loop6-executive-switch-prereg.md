# Loop 6 — prefrontal executive rule-switch preregistration

Status: COMPLETE — `0/100 STOP`

The belief-maintenance mechanism was effective, but the registered recursive
surprise feedback was not. Candidate-minus-surprise-off LCB was `-0.00016` ID
and `-0.00456` OOD. Recovery latency was `6.57/6.51 > 6`, and OOD oracle gap
was `0.1556 > 0.15`. No hazard or threshold sweep was run.

## Claim limit

Synthetic hidden-rule maintenance and switching only. This is not biological
PFC validation, general executive intelligence, or AGI.

## Task

Each card exposes three irrelevant/relevant feature channels with four possible
actions. One hidden rule selects which channel determines the correct action.
The rule changes without an explicit cue. The agent sees only its action and a
possibly noisy binary feedback signal; 20% of trials omit feedback as a
working-memory distractor gap.

ID blocks have lengths in `{20, 24, 28}` with feedback flip probability `0.05`.
OOD blocks have lengths in `{12, 16, 32}` with flip probability `0.10`. Seeds,
cards, switches, feedback flips, and gaps are paired across arms.

## Candidate

The executive state is a categorical belief over rules:

\[
\tilde b_t=(1-h_t)b_{t-1}+h_t/K,
\qquad
b_t(r)\propto \tilde b_t(r)P(f_t\mid r,a_t,x_t).
\]

The fixed base hazard is `h=0.02`. If observed feedback has predictive
probability below `0.20`, the next-trial hazard is `0.35`; this is the explicit
metacognitive surprise-to-switch feedback. Missing feedback retains the goal
belief without a likelihood update. The action maximizes posterior expected
correctness. No environment clone or future read is allowed.

## Fixed comparisons

- `hazard_off`: no change-point or surprise release, testing perseveration;
- `surprise_off`: fixed hazard only, testing metacognitive switch feedback;
- `feedback_shuffle`: same feedback multiset with temporal alignment broken;
- `gap_reset`: resets rule belief to uniform whenever feedback is absent;
- `win_stay_shift`: one-step heuristic with no maintained rule distribution;
- oracle hidden-rule controller as a ceiling.

## Hard gates

On both ID and OOD:

1. overall accuracy is at least `0.70`;
2. mean accuracy in trials 3--8 after each switch is at least `0.65`;
3. mean recovery latency to three consecutive correct actions is at most `6`;
4. paired 95% bootstrap LCB(candidate minus hazard-off, feedback-shuffle,
   gap-reset, and win-stay-shift) is positive;
5. candidate minus surprise-off LCB is above `0.01`, isolating the recursive
   surprise feedback rather than belief maintenance alone;
6. candidate is no more than `0.15` below oracle mean accuracy;
7. future reads and environment clone calls are zero, and belief remains a
   finite normalized simplex on every trial.

Any failed gate gives `0/100 STOP`. One unit-scale implementation debug and one
fixed-seed scored run are allowed. No post-run hazard, threshold, block, or
noise sweep is allowed.
