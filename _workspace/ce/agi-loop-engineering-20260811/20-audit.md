# Status audit

Status: COMPLETE

| Claim | Status | Evidence | Gate |
|---|---|---|---|
| Optional action-conditioned controller is implemented without changing the default agent path | implementation | focused regression | PASS |
| Belief planning needs previous action polarity | synthetic learned validation | full vs action-agnostic LCB `0.5734` | PASS |
| Delayed horizon is causal | synthetic learned validation | full vs H1 LCB `0.5734` | PASS |
| Belief planner beats a matched recurrent sufficient-statistic policy | not shown | paired difference `0.0` | NONINFERIOR only |
| Primary learned DPC validation | empirical synthetic | return `0.768`, success `0.863`, Brier `0.0723`, ECE `0.0440` | PASS |
| OOD calibration | empirical synthetic | ECE `0.0928 > 0.08` | FAIL diagnostic |
| Signed eligibility carries delayed reward | tabular mechanism | success `1.0` vs three controls `0.4824` | PASS |
| BrainRuntime external signed-credit API | implementation | focused regression `52 passed` | PASS / experimental only |
| Runtime signed-credit efficacy, dense start (Loop 2b) | empirical synthetic | score `0`, structural jump confound | STOP |
| Runtime signed-credit efficacy, matched manifold (Loop 2c) | empirical synthetic | signed-off LCB `-0.06021`; guard delta `+0.07479` | STOP / default off |
| Raw-history controlled state discovery (Loop 3) | empirical synthetic | score `85`; causal ablations pass | GO within claim limit |
| Structured history state superiority over recurrent state | not shown | ID LCB `0.0`; OOD LCB `-0.00498` | NONINFERIOR only |
| Modular reward transfer (Loop 4) | empirical synthetic | stale causal gate and oracle-gap gate fail | STOP |
| Explicit planner superiority over context-RNN | not shown | RNN train accuracy `68.76%`, SAFE predictions `0` | INVALID comparator signal |
| Audited episodic-memory mechanics (Loop 5) | empirical synthetic | corrected score `90`; all registered controls present | GO within claim limit |
| Long-dialogue/SOTA or biological replay efficacy | not tested | bounded synthetic key/value task only | OPEN |

No AGI, world-model superiority, or runtime-learning superiority claim is made.
The tabular signed-credit result does not transfer to the current continuous
BrainRuntime eligibility representation.

