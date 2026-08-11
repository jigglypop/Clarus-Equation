# Loop 7 — active information-seeking executive preregistration

Status: COMPLETE — `0/100 STOP`

Active-minus-reward-only LCB was `-0.00179` ID and `-0.00130` OOD; active also
failed to beat the Loop 6 surprise heuristic. Recovery latency remained above
six. The task supplies no action that changes future observation quality, so
the registered information-gain term has no independent control channel. No
beta sweep was run.

## Claim limit

Synthetic active hidden-rule identification only. This does not establish
general planning, biological PFC function, or AGI.

## Candidate

Loop 6's scalar surprise-hazard boost is removed. The rule posterior uses the
fixed-hazard Bayesian update (`h=0.02`). Action choice adds expected information
gain to immediate correctness:

\[
a_t=\arg\max_a\left[
P(\mathrm{correct}\mid a,b_t)+\beta I(R;F_t\mid a,b_t,x_t)
\right],\qquad \beta=0.25.
\]

Information gain is the current rule entropy minus expected posterior entropy
over positive/negative feedback under feedback error `0.05`. All logarithm
arguments are probabilities and therefore dimensionless.

## Fixed comparisons

- reward-only fixed-hazard belief (`beta=0`), identical to Loop 6
  `surprise_off`;
- Loop 6 surprise heuristic with reward-only action selection;
- information-only action selection, isolating reward/information balance;
- feedback-shuffled active controller;
- win-stay-shift and oracle.

The ID/OOD episodes, seeds, block distributions, gaps, and feedback noise are
identical to Loop 6.

## Hard gates

On both ID and OOD:

1. overall accuracy at least `0.85`;
2. recovery latency to three consecutive correct actions at most `6`;
3. post-switch trials 3--8 accuracy at least `0.75`;
4. paired 95% LCB(active minus reward-only) above `0.01`;
5. paired 95% LCB(active minus surprise heuristic, information-only,
   feedback-shuffle, and win-stay-shift) positive;
6. oracle accuracy gap at most `0.15`;
7. all beliefs remain normalized and future reads/environment clones are zero.

Any failure gives `0/100 STOP`. One unit implementation check and one scored
run are allowed. No post-run beta or entropy threshold sweep is allowed.
