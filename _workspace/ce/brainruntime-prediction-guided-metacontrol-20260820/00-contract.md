# Prediction-guided metacontrol contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-native-all-loops-p1-20260819`

## Question and maximum claim

Can a predictor frozen before evaluation causally improve a `BrainRuntime` policy by supplying action-indexed next-state forecasts to the action selector? A positive result supports only this statement: in the frozen synthetic runtime and task, the policy depends on the predictor-to-policy input edge and uses it to reduce a declared one-step goal loss. It does not establish biological metacognition, consciousness, predictive coding in a brain, or optimal control in general.

## PREDECESSOR_EVIDENCE

| Evidence | State | Artifact and SHA-256 | Preserved result | Exclusion carried forward |
|---|---|---|---|---|
| Native Loop 10 | `PASS` 32/32 | `brainruntime-native-all-loops-p1-20260819/artifacts/confirmation-results.json`, file SHA `2fd40c7e32f2ed8b143701bc517393b7df279d36a293483a6846279863726633` | A frozen pretransition native-state ridge predictor beat persistence; mean improvement `0.282425` | It did not choose an action and is not evidence for control or consciousness |
| M3 predictive writer | `STOP` 0/16 | `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-predictor-development-results-v2-frozen.json`, `4e865af022bc7e8ac33a11861a83816b7e4b94fce097ed382f88fc3d45fbdaff` | The registered OOD replay predictor did not beat persistence | Do not retune that writer, replay error, threshold, or seed block |
| Geometric mediator family | `STOP` / `BLOCKED` | global route ledger rows BA-G1--BA-G3D; underlying hashes fixed there | No calibrated metric or response summary earned unique predictive/mediating status | C1 contains no metric feature and makes no mediation claim |

The predecessor has a complete `40-final-report.md`. The contrastive-memory predecessor is also complete. No missing closure is used as positive evidence.

## Candidate admission decision

| Rank | Candidate | Decision before outcomes | Reason |
|---:|---|---|---|
| 1 | C1 predictor-to-policy intervention | selected | It adds a new manipulable causal seam: forecasts can be intact, shuffled only at the planner port, or shuffled only in the displayed readout |
| 2 | S1 SCC feedback lesion | deferred | Current learned support may be dense or one giant SCC, making fair outside/SCC-preserving damage controls impossible |
| 3 | M3 predictor/write retune | retired | Changing seeds, threshold, decoder, or OOD amplitude would repeat the failed mechanism |

## Frozen apparatus

- Development seeds: `97901..97916`; confirmation seeds: `99901..99932`; bootstrap seed: `97998`.
- Dimension `d=48`; Torch CPU; float32 runtime and float64 ridge/score reduction.
- `noise_sigma=0`, `axon_delay=False`, `stdp_enabled=False`, `hippocampal_encoding_enabled=False`, `dale_law=False`, replay gain zero. Hippocampal and temporal rows remain physically zero. Recurrent weight is seed-fixed and never changes after fixture construction.
- Recurrent initialization is exact: a fresh CPU generator seeded by the circuit seed draws `torch.randn(48,48,dtype=float32)`, multiplies by `0.025`, and zeros the diagonal before `BrainRuntime` construction. The post-constructor dense weight and CSR hashes define the common base for every arm.
- Actions are ordered `[-1,0,+1]`. A fresh CPU generator seeded by `97900` draws `torch.randint(0,2,(48,))`, maps zero/one to `-1/+1` in float64, and normalizes the resulting Rademacher vector `u` once; its bytes and hash freeze. Candidate correction is `0.75 a u`. Each episode also has a precommitted context drive `c_e` of norm `0.50`; the actual proposed drive is `c_e+0.75 a u` cast to runtime float32.
- Every episode starts from its own precomputed native snapshot. Sets are disjoint: 128 predictor-fit states, 48 predictor-audit states, and 64 policy-test states. A snapshot is produced only by a fixed four-tick warm-up tape, each tick executed in forced WAKE with `learning_signal=0`, then frozen. A per-circuit CPU generator seeded by `seed+170000` draws the ordered `240 x 4 x 48` Gaussian warm-up table and normalizes each tick drive to norm `0.35`. A separate generator seeded by `seed+180000` draws the ordered `240 x 48` Gaussian context table and normalizes each row to `0.50`. Ordering is fit IDs `000..127`, audit IDs `000..047`, policy IDs `000..063`, with tick order `0..3`; zero/nonfinite rows are rejected. The complete tables, generator rules, ordered snapshot IDs, and associations are hashed before fitting. No policy-test transition becomes a fit row.
- For fit and predictor-audit states, all three candidate actions are evaluated from independent restores. Candidate rollouts are forbidden on policy-test states.

## Predictor and task

Let the native pretransition state contain activation, refractory, memory trace, adaptation, STP `u/x`, and lifecycle. For candidate action $a$, the feature is

$$
\phi(s,c,a)=[a_t,r_t,m_t,\alpha_t,u_t,x_t,\ell_t,c+0.75au,\operatorname{onehot}_3(a),1],
$$

with dimension $8d+4=388$. It contains no post-state, realized loss, target next state, memory row, or candidate rollout. Let $x^{\mathrm{act}}=\operatorname{activation}(s)$. The output is exactly the existing Loop-10 four-summary

$$
y(s)=[\operatorname{mean}(x^{\mathrm{act}}),\lVert x^{\mathrm{act}}\rVert_2,\max_j|x_j^{\mathrm{act}}|,d^{-1}\sum_j\mathbf 1(|x_j^{\mathrm{act}}|>0.1)].
$$

Here $x^{\mathrm{act}}$ never denotes the STP availability vector already named $x_t$ inside $\phi$. This exact schema and its source hash are recorded.

A deterministic float64 ridge map with $\lambda=10^{-4}$ is fit to the 384 fit rows and then hash-frozen. Fit-target mean and population standard deviation (`correction=0`) are computed in float64; the componentwise scale is $\max(\sigma_k,10^{-8})$ and defines the frozen standardizer $z(y)$.

For policy episode $e$, the declared goal is

$$
g_e=z(y(s_e))+0.25q_e,
$$

where $q_e$ cycles through the following bank, fixed without runtime outcomes. An independent CPU generator seeded by `97901` draws $G\in\mathbb R^{16\times4}$ in float64. Reduced QR gives $G=QR$; each column of $Q$ is multiplied by $\operatorname{sign}(R_{kk})$, with zero sign defined as `+1`, and then every nonzero row of $Q$ is individually normalized to unit Euclidean norm. The implementation rejects a zero/nonfinite row, records the pre-normalization matrix and final bank hashes, and requires $|\lVert q_e\rVert_2-1|\le10^{-12}$ for all sixteen rows. This construction does not claim the sixteen rows are mutually orthogonal. The intact planner computes all three predictions algebraically, chooses

$$
a_e=\arg\min_{a\in[-1,0,+1]}\lVert z(\hat y(s_e,c_e,a))-g_e\rVert_2^2,
$$

with lowest-index tie breaking, and only then performs exactly one real runtime step. Primary loss is $L_e=\frac14\lVert z(y_{e+1})-g_e\rVert_2^2$.

## Matched arms

Every arm receives identical snapshots, contexts, goals, candidate set, event count, and exactly one actual transition.

1. `intact`: normal frozen forecasts reach the planner.
2. `edge_shuffle`: a fixed cyclic derangement permutes forecast-to-action labels only at the predictor-to-planner port. True forecasts remain audit-only.
3. `readout_shuffle`: the planner receives intact forecasts; only the emitted display/log copy is permuted after action selection and has no downstream reference.
4. `persistence`: always chooses action zero.
5. `random_balanced`: uses the globally hashed literal schedule `[-1,0,+1]` repeated 21 times followed by `[-1]`, independent of current state, goal, and future. Ordered counts are `(-1,0,+1)=(22,21,21)`.
6. `error_magnitude_only`: uses only the no-correction predicted loss and a fit-only median alarm threshold to choose zero versus nonzero magnitude. For fit snapshot $j$, its fit-only goal uses $q_{j\bmod16}$ in the same goal formula and its own frozen context; sort the 128 predicted zero-action costs and define the even-sample median as the arithmetic mean of zero-based positions `63` and `64`. Nonzero sign follows the independently hashed literal 64-entry tape `[-1,+1]` repeated 32 times. The tape is not rebalanced after alarm filtering.
7. `reactive_mean_effect`: uses the fit-only global mean standardized effect for each action, current summary, and goal, but no state-conditioned counterfactual forecast.

## Integrity and leakage gates

- Predictor parameters, standardizer, banks, schedules, split hashes, source hash, and recurrent weight hash freeze before policy evaluation.
- Selection logs, for every episode and arm, the action-labelled forecast tensor and cost vector before any planner map, the planner-port permutation, selected action, tie rule, and actual drive. A step counter proves zero candidate `BrainRuntime.step` calls and exactly one selected-action step per policy episode.
- Before and after every arm: recurrent weight hash unchanged, dense/CSR parity, zero STDP updates, zero hippocampal/temporal rows, finite state, and identical starting snapshot hash.
- `readout_shuffle` must have selected-action and actual-drive tensors byte-identical to `intact`, action traces byte-identical, and per-episode loss difference at most `1e-12`. Any downstream effect is a wiring failure.
- `edge_shuffle` must use byte-identical pre-map forecast tensors and the same unordered cost multiset as `intact`, then apply a logged nonidentity cyclic action-label permutation only at the planner port.
- Test contexts, goals, noise/step indices, and snapshots are disjoint from fit and predictor-audit rows. Confirmation is inaccessible without a hash-bound passing development manifest.

## Frozen decision rules

The circuit seed is the sole statistical unit. A CPU generator seeded by `97998` draws exactly 16 paired seed-row indices with replacement for each of 10,000 bootstrap replicates. Every replicate statistic must be finite. After ascending sort, the one-sided lower bound is zero-based order statistic `499` and the upper bound is order statistic `9500`. Episode rows are never resampled as independent units; the number of passing original circuit rows is reported separately.

1. Prediction gate: on the independent 48-by-3 audit transitions, both losses use the same raw four-summary components and reduction. Predictor MSE is $\operatorname{mean}_{j,a}\lVert \hat y_{j,a}-y(s'_{j,a})\rVert_2^2/4$; action-conditioned persistence MSE is $\operatorname{mean}_{j,a}\lVert y(s_j)-y(s'_{j,a})\rVert_2^2/4$. The persistence denominator must be finite and strictly positive in every circuit or the route is `STOP`. Their ratio is at most `0.90` in at least 80% of circuits, and the bootstrap upper 95% bound of the mean ratio is at most `0.90`.
2. For each adverse arm $b\in\{$`edge_shuffle`, `persistence`, `random_balanced`, `error_magnitude_only`, `reactive_mean_effect`$\}$ define

$$
A_i^b=\frac{\bar L_i^b-\bar L_i^{\mathrm{intact}}}{\bar L_i^b+\bar L_i^{\mathrm{intact}}+10^{-12}}.
$$

The within-seed minimum across arms must exceed `0.05` in at least 80% of circuits, and its bootstrap lower 95% bound must exceed `0.05`.
3. `edge_shuffle` must change the chosen action on at least 20% of policy episodes in at least 80% of circuits; bootstrap lower bound of mean change rate must exceed `0.20`.
4. `readout_shuffle` exact-equivalence and every integrity gate must pass in every circuit. A seed row is constant-degenerate when the maximum minus minimum of the six mean policy losses (`intact` plus five adverse arms, excluding the required-equivalent readout arm) is at most `1e-12`. Nonfinite or constant-degenerate rows are `STOP`, never excluded; any advantage denominator at most `1e-12` is also `STOP`.

Development `GO` opens confirmation only through a source/result-hash manifest. Confirmation uses the same frozen rules and requires all four gates again on exactly 32 seeds. Otherwise the route is `STOP` and confirmation stays sealed.

## Claim boundary

Even a confirmed `GO` means only causal dependence of this simulated policy on a frozen forecast input under the declared task. It does not identify a brain algorithm, metacognitive awareness, selfhood, phenomenal consciousness, or a biological neural mechanism. Those require independent biological measurements and interventions.
