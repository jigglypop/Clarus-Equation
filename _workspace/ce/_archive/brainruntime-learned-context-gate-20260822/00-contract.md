# BA-TR4: learned context-to-mask gate

Status: COMPLETE

Mode: light

PREDECESSOR: `_workspace/ce/brainruntime-context-branch-routing-20260822`

## PREDECESSOR_EVIDENCE

- BA-TR3 R2 source freeze: `artifacts/source-freeze-r2.json`, SHA-256 `9b68715a724c06cf51e78364cf5a4cd83462e1b18ffdbcaf5a7bfd5e184ff302`.
- BA-TR3 R2 development result: `artifacts/development-results-r2.json`, SHA-256 `bc8dd6f1f884e500f691fa4cddd125a11f442201bfd422ccd576d231e0abe2c5`, `CORRECT=1.0`, `WRONG=0.0`, `STATIC_0=STATIC_1=0.5`, 16/16 development seeds.
- Admitted predecessor claim: a supplied context mask is sufficient and necessary for the fixed two-branch task. It did not learn the context-to-mask map. BA-TR3 confirmation remains sealed.

## Question

Can a separate bounded gate learn which one of the two already declared entry branches to open from a context cue and local branch-use eligibility alone, without receiving a target, decoder score, endpoint loss, oracle mask, or branch label as an update input?

## Frozen model

The recurrent state, five-block architecture, exact delay $L=2$, thresholds, payload decoder, branch/trunk weights, recall horizon, and 12-edge mask budget are inherited unchanged from BA-TR3. The only new state is a two-actuator gate. For seed $s$, two fixed orthonormal dimensionless context codes $q_0,q_1\in\mathbb R^4$ are generated independently of payloads. A balanced seed-specific bijection $\sigma_s:\{0,1\}\to\{0,1\}$ determines which experienced branch is paired with each cue; even seeds use the identity and odd seeds use the swap.

Let $Q_b$ be the $m=4$ learned entry edges of branch $b$ and let $E^{(n)}$ be the exact-delay local eligibility measured during an experience that contains only the context cue externally to the gate and the physical $S_{\sigma(c)}\to H_{\sigma(c)}$ pulse pair in the runtime. Define the dimensionless branch-use receipt

$$
u_b^{(n)}=\frac{1}{m}\sum_{(i,j)\in Q_b}\left[E_{ij}^{(n)}\right]_+.
$$

The gate starts at $\Theta_0=0\in\mathbb R^{2\times4}$ and receives only $(q_c,u)$:

$$
\Theta_{n+1}=\operatorname{clip}_{[-4,4]}
\left(\Theta_n+u^{(n)}q_c^{\mathsf T}\right).
$$

At recall the gate is frozen before any payload endpoint is scored:

$$
\ell(c)=\Theta q_c,\qquad
\widehat b(c)=\arg\max_{b\in\{0,1\}}\ell_b(c),
$$

$$
\widehat M_c=Q_{\widehat b(c)}\cup Q_{YH_0}\cup Q_{YH_1},
\qquad \lVert\widehat M_c\rVert_0=12.
$$

An exact logit tie, nonfinite state, wrong shape, mixed budget, output-specific context edge, or any gate mutation after freeze is apparatus invalid. The gate does not discover candidate support; $Q_0,Q_1$ are fixed actuator families inherited from BA-TR3.

## Leakage boundary

The gate update and learned-mask compiler may read context code, local entry eligibility, fixed candidate support, and frozen gate parameters only. They must not accept or inspect payload identity, expected answer, target vector, $Y$ activity, decoder, endpoint score, route label, or oracle mask. Context is never injected into BrainRuntime state or the decoder. The environmental pulse schedule may determine which branch is actually experienced, but that branch index is not an argument to the gate update.

## Frozen controls

`ORACLE`, `LEARNED`, `CONTEXT_SHUFFLE_TRAIN`, `WRONG_CUE`, `POST_CUE_SWAP`, `GATE_LESION_STATIC_0`, `STATIC_1`, `CANONICAL_CUE_MAP`, `RANDOM_MATCHED`, and `FULL`. Every arm except `FULL` uses exactly 12 recurrent edges; `FULL` retains 16 and is a capacity-favourable adverse control. `CONTEXT_SHUFFLE_TRAIN` pairs each physical experience with the other context code while preserving all runtime pulses and update counts.

## Pre-endpoint gates

Both learned masks must contain 12 edges, share the same eight-edge output trunk, differ on exactly eight entry-edge positions, contain no edge outside the frozen support, and select $\sigma_s(0),\sigma_s(1)$ with a positive logit margin. The shuffled-training gate must select the opposite branches. Context codes must be finite, unit norm, mutually orthogonal, payload-independent, and gate/runtime/decoder hashes must be frozen before scoring. BA-TR3 preflight, zero-store cutoff, dense/sparse parity, and no-context-state/decoder gates must also pass.

Five anti-oracle receipts are mandatory. First, a reference evaluator implemented separately from the mask compiler must recompute $\arg\max(\Theta q_c)$ from serialized frozen $\Theta,q$ and agree with the compiled branch and mask for both cues. Second, holding $\Theta,q$, and support fixed while replacing $\sigma_s$, the seed, every seed-derived value, and all schedule metadata must leave compiled learned masks byte-identical. The learned compiler's exact allowed inputs are `(frozen_gate, context_cue, fixed_weight, fixed_blocks)`; it accepts and captures no seed, $\sigma_s$, task object, or schedule. Third, holding $\Theta$ and support fixed while swapping only $q_0,q_1$ must swap the two compiled masks. Fourth, holding $q$ and support fixed while replacing $\Theta$ by the finite non-tied row-swapped matrix $P\Theta$ must swap the compiled actions and masks exactly as an independent $\arg\max(P\Theta q)$ reference predicts. Fifth, the serialized gate digest must be identical before and after every rollout. These are source-level and executable gates; agreement with $\sigma_s$ alone is insufficient.

## Development decision

Use only seeds `97601..97616`; reserve `99601..99632` unopened for confirmation. A development seed passes when `LEARNED>=0.95`, `ORACLE>=0.95`, the learned-oracle gap is at most `0.05`, `WRONG_CUE` and `CONTEXT_SHUFFLE_TRAIN<=0.05` with opposite-payload delivery at least `0.95`, `GATE_LESION_STATIC_0`, `STATIC_1`, and `RANDOM_MATCHED<=0.55`, `FULL<=0.55`, learned advantage over the strongest exact-budget non-oracle control is at least `0.40`, post-cue swap matches the wrong-cue intervention, and every receipt passes. Development GO requires at least 15/16 seed passes and mean `CANONICAL_CUE_MAP<=0.55` across the balanced seed mappings.

Any formula failure may receive one mechanism-only revision. Thresholds, decoder, payload codebook, delay, recall horizon, mask budget, development seeds, and endpoints may not be tuned. Failure after the allowed revision is `LEARNED_CONTEXT_GATE_NOT_IDENTIFIED`; confirmation stays sealed.

## Claim ceiling

A pass establishes only a learned two-choice context selector over a declared synthetic candidate-mask family. The rule is experience-supervised by local branch use. It does not establish support discovery, general graph morphology selection, answer-blind plasticity, cortical biology, curvature-as-memory, physical energy, disease treatment, or AGI.
