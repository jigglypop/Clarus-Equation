# Orbit sidecar task, budget, and scaling preregistration

## Goal

Close the three gates left open by the dynamic orbit quotient prototype:

1. an equivariant limited-budget approximation with an honest error bound;
2. measured and structural scaling against the dense cyclic cover;
3. a matched behavioral task comparing dense, quotient-only, and quotient plus
   local causal deviations.

No BrainRuntime execution path may be modified in this experiment.

## Limited-budget rule

At each time slice, candidate deviations are ranked by absolute normalized
activity.  Equal-magnitude candidates form a complete tie class.  A tie class
is retained only when the whole class fits; it is never split by absolute cell
index.  This makes selection commute with cyclic translation.  Dropped values
and budget exhaustion are reported.

For `tanh`, `Lip(phi) <= 1`.  Let

\[
g_d=\max_q\sum_{e:q_e=q,d_e=d}|K_e|,
\qquad r_t=\|e_t^{candidate}-e_t^{retained}\|_\infty.
\]

The registered a-posteriori state-error recursion is

\[
B_t=r_t+\sum_d g_d B_{t-d},\qquad B_s=0\;(s<0).
\]

All terms are dimensionless normalized activities or gains.  `B_t` is a
conservative state bound, not automatically an action certificate: a hard
argmax is certified only when its measured logit margin exceeds the propagated
score uncertainty.

## Behavioral task

Each balanced episode places one positive local cue at one of two well-separated
candidate locations on `C_N`.  The required action is the cued location.  A
fixed horizon-four readout compares deviation energy in the two translated
causal neighborhoods.  Labels are assigned by the environment before any model
rollout; the candidate never reads the dense trajectory.

- covers: `N = 64, 128, 256`, with `256` treated as untouched size OOD;
- deterministic generator seeds: `41064`, `41128`, and locked OOD seed
  `41256`; exactly `18` accepted episodes per action and cover;
- network horizon `4`, candidate offsets `(-12, 0, 12)`, acceptance utility
  margin `0.01`, movement cost `0.01`;
- labels exactly balanced;
- pulse positions are jointly translated by seeded shifts;
- dense and local-cone models use identical kernel, initial state, inputs, and
  readout;
- quotient-only receives only the spatial projection and therefore cannot know
  which translated location was cued.

Hard task gate: exact local-cone accuracy is no worse than dense by `2%`, exact
local-cone trajectory error is `<= 1e-10`, and it beats quotient-only accuracy
by at least `20` percentage points on every cover.

## Budget and scaling gates

- budget curve: `B = 1, 2, 4, 8, 16, exact`;
- actual max state error must never exceed the registered recursive bound by
  more than `1e-12`;
- translating a fixture must translate every retained set and preserve all
  reported residuals;
- candidate state count and counted work at fixed horizon/budget are independent
  of `N`;
- dense state storage grows linearly with `N`;
- wall time is descriptive only and must be reported separately from counted
  work; noisy timing cannot override an algebraic failure.

## Promotion rule

All hard gates must pass and the preregistered readiness score must reach at
least `80/100`.  Passing authorizes only a feature-off, read-only BrainRuntime
sidecar proposal.  It does not authorize replacement of the recurrent runtime,
learning under symmetry, or claims about real-brain geometry.
