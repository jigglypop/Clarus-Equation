# Loop 4 — modular reward-transfer preregistration

Status: COMPLETE — `0/100 STOP`

Registered failure reasons:

- candidate-minus-stale LCB was nonpositive in all ID contexts and OOD
  `balanced`;
- OOD `cautious` oracle gap was `0.020410 > 0.02`.

Non-scoring baseline audit: the context RNN obtained only `68.76%` train
classification accuracy and predicted the SAFE class zero times despite 269
SAFE labels. Its large negative transfer return therefore cannot support a
structured-planning superiority claim. No architecture or context sweep was
run after the locked result.

## Claim limit

Synthetic modular reward-transfer only. Passing does not establish a general
world model, planning superiority outside this task family, or AGI.

## Question

Can a frozen history belief be combined with a changed utility function at
inference time, without policy retraining, and outperform a small policy RNN
that was trained on raw history plus multiple reward contexts?

For posterior confidence `q`, commit reward `R`, wrong loss `L`, safe reward
`S`, and commit cost `C`, the candidate commits iff

\[
q(R-C)+(1-q)(-L-C)>S.
\]

The belief parameters are exactly the train-only Loop 3 learner. Reward
parameters never enter belief fitting.

## Fixed training and transfer contexts

The context-conditioned two-state tanh RNN policy is trained jointly on the
Cartesian product

- `S in {0.05, 0.25}`;
- `L in {0.8, 1.4}`;
- `C in {0.00, 0.10}`;
- `R = 1.0`.

It receives `(action, observation, mask, S, L, C)` at every step. The frozen
transfer contexts are the unseen compositions:

1. `balanced`: `S=0.15, L=1.10, C=0.05`;
2. `cautious`: `S=0.45, L=1.80, C=0.10`;
3. `asymmetric`: `S=0.30, L=2.20, C=0.00`.

No transfer labels or episodes may update either learner.

## Comparisons

- context-conditioned two-state tanh policy RNN;
- stale planner fixed to the base `S=0.15, L=1.0, C=0.0` utility;
- reactive planner using only the final observed event and the correct utility;
- oracle posterior with the correct utility as a ceiling.

## Hard gates

Across every transfer context on both ID and fixed OOD noise:

1. paired-bootstrap 95% LCB(candidate minus stale) is positive;
2. paired-bootstrap 95% LCB(candidate minus reactive) is positive;
3. the mean paired-bootstrap LCB(candidate minus context-RNN) across contexts
   is above `0.02`, and no individual context is below `-0.03`;
4. candidate-oracle mean return gap is at most `0.02`;
5. candidate success among committed actions is above `0.85` and all outputs
   are finite;
6. policy/belief update counts during transfer, future reads, and environment
   clone calls are exactly zero.

Any failed hard gate gives `0/100 STOP`. A single small implementation-debug
test is allowed before one scored fixed-seed run. No post-run architecture,
threshold, context, or seed sweep is allowed.
