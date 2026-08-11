# Status audit

Status: COMPLETE

## Loop 8B brain-geometry status audit

- Pure LBO heat diffusion as working memory: **deleted** for the compact,
  connected, no-drift model; every nonconstant mode decays.
- Attractor resistance to diffusion: **model choice with passing synthetic
  output**, not a general neural theorem.
- Biological MD equals a Riemannian metric: **incomplete**; this benchmark tests
  continuous gain/landscape modulation only.
- CE residual equals hippocampal replay: **incomplete and untested** here.
- CE cosmological fractions as neural allocation constants: **deleted**; no
  physical or biological map exists.
- Hidden axioms: 2-D state, opposite feature signs, shared-capacity
  cross-inhibition, fixed block distribution/noise, and common cue filter.
- Context shuffle collapses the MD result to chance inside this task.
- Gate: PASS for a separately preregistered residual-replay ablation; BLOCKED
  for canonical/runtime promotion.

## Loop 8C residual status audit

- Feedback residual improves cue-depleted rule switching: **synthetic output**.
- The residual effect is switch-selective rather than a stationary global gain:
  **synthetic output** under the registered domain pair.
- Residual equals hippocampal replay: **not established**. The implemented
  source is supervised action feedback, not spontaneous sequence reinstatement.
- Opposite feature signs make feedback context-identifying: **hidden task
  axiom**, explicitly retained as a scope limit.
- Gate: PASS for a separate BG/STN boundary experiment; BLOCKED for claiming
  episodic replay, hippocampal mechanism, or runtime promotion.

## Loop 8D STN status audit

- Conflict-dependent boundary increases high-conflict accuracy over a low
  boundary: **synthetic output**.
- It beats equal-average fixed allocation: **rejected overall**; OOD high-
  conflict, conflict-shuffle causality, and utility gates failed.
- It preserves memory state: **synthetic invariant**, bit-identical trace check.
- It implements biological STN: **not established**; the boundary map is a
  model choice.
- Gate: STOP. Preserve Loop 8B/8C checkpoints; do not tune this linear boundary
  or promote it to runtime.

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
| Hidden-rule executive belief maintenance (Loop 6) | empirical synthetic | beats hazard-off/shuffle/gap-reset/WSLS | PROMISING mechanism |
| Surprise-triggered metacognitive switch efficacy | not shown | candidate-surprise-off LCB `-0.00016/-0.00456` | STOP |
| Active information-seeking executive (Loop 7) | not shown | active-reward LCB `-0.00179/-0.00130` | STOP |
| Epistemic action value in Loop 7 task | undefined channel | actions do not alter future observation quality | TASK GAP |
| Unified executive equation (Loop 8) | model choice + two narrow theorems | branch-free posterior and policy functional | MATH RESET / implementation locked |

No AGI, world-model superiority, or runtime-learning superiority claim is made.
The tabular signed-credit result does not transfer to the current continuous
BrainRuntime eligibility representation.

