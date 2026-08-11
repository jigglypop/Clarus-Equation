# Experiment B preregistration — value-of-information STN HOLD

## Question

Does an explicit HOLD channel improve decisions when another observation has
positive expected value, and does an action-only common STN offset remain a
provable no-op?

## Generative task

Each episode has hidden binary state \(y\in\{-1,+1\}\). At tick \(t\):

\[
x_t\mid y\sim\mathcal N(y\mu,\sigma^2).
\]

The cortical state is the exact accumulated log-likelihood ratio

\[
L_t=L_{t-1}+\frac{2\mu x_t}{\sigma^2},
\qquad P(y=+1\mid L_t)=\sigma(L_t).
\]

Correct action reward is `+1`, incorrect action reward is `-1`, and each HOLD
costs `c>0`. The horizon is finite, so termination is guaranteed.

## Candidate

At the last tick, act. Before it, compute

\[
Q_{act}(L)=2\max(\sigma(L),1-\sigma(L))-1
\]

and

\[
Q_H(L,t)=-c+E[V(L+\Delta,t+1)\mid L].
\]

The Bellman recursion is

\[
V(L,t)=\max(Q_{act}(L),Q_H(L,t)).
\]

The expectation is evaluated by locked Gauss–Hermite quadrature on a locked LLR
grid. HOLD is selected iff \(Q_H>Q_{act}\). This is the normative source of the
common brake; normalized entropy is recorded but is not hand-tuned into a
threshold.

## Arms

1. immediate action after the first sample;
2. action-only softmax with a common STN offset, which must equal arm 1;
3. always wait until the horizon;
4. Bellman value-of-information HOLD candidate;
5. candidate with HOLD cost sign flipped as a causal control.

## Gates

Across paired untouched seeds:

1. common-offset and immediate actions disagree exactly zero times;
2. candidate utility minus immediate 95% LCB is positive;
3. candidate utility minus always-wait 95% LCB is positive;
4. candidate accuracy is no worse than immediate by more than `0.005`;
5. HOLD rate is higher in the top conflict-entropy quartile than the bottom by
   at least `0.25`;
6. sign-flipped cost does not improve candidate utility;
7. every episode terminates by the declared horizon and all Bellman values are
   finite.
