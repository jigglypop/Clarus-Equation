# V9 nested infinite-SCC source and lineage audit

Status: COMPLETE

Audit date: 2026-08-11 (Asia/Seoul)

## 1. Executive result

This audit supports a narrow design conclusion, not a V9 result.

1. The local V1--V8 line has **not** already implemented a nested recurrent
   causal state. V2 and V4 passed only narrowed synthetic gates; V4's strongest
   result was sequential one-step prediction with the true chart state read
   again at every scored step. The first genuine H20 free-rollout parent (V5)
   failed, as did V7 and the prospective V8 confirmation. V6 was never run.
2. The Adaptive Causal Belief-State Model (ACBSM) is the closest local
   ancestor. It implemented a prefix-only Gaussian belief observer and
   transition-internal residual correction, but its proposed rank-two
   fast/slow state collapsed to rank one in every selected fold. Its eight-fold
   training screen was classified `HOLD`, and the reserved fresh block was not
   opened. ACBSM is therefore a mandatory V9 control, not evidence that V9 has
   already succeeded.
3. A nested V9 mechanism is genuinely new only if **multiple persistent level
   states and cross-scale messages causally mediate predictions/actions**. A
   diagram containing nested cycles, a relabelled low-rank mode, an output
   shrinkage gain, or a finite stack with an analytic-posterior/V5 bypass is not
   a new mechanism.
4. Primary connectome evidence supports finite brains with extensive directed
   recurrence, feedback loops, multiscale organization, and—in one precisely
   thresholded adult-fly graph—a 93.3% giant SCC. It does **not** establish a
   biological direct-limit tower of nested maximal SCCs, infinitely many
   physical neurons, convergence of neural dynamics, AGI, cognition, or
   consciousness.
5. No audited primary source proves that a whole brain is naturally partitioned
   into several *nested maximal SCCs*. On a fixed directed graph maximal SCCs
   are disjoint equivalence classes. Biological nestedness can instead refer
   to nested **strongly connected subgraphs**, communities, anatomical scales,
   functional gradients, or recurrent pathways; those are different objects.

The V8 locked confirmation test block `81100..81355` was **not opened** during
this audit. No V8 locked split, V7 test split, ACBSM fresh block, or new V9
evidence block was simulated.

## 2. Audit scope and immutability

The governing source is `00-contract.md`. The source lane used read-only
inspection of the registered JSON, historical artifacts and Git provenance;
it did not re-fit a route or run an evidence seed. The main local sources are:

- `experiments/preregistration/sparse_causal_bridge_v1.json` through
  `sparse_causal_bridge_v8.json`;
- `_workspace/ce/agi-v7-20260811/10-sources.md`, `20-audit.md`, and
  `31-validation.md`;
- `docs/7_AGI/26_Sparse_Causal_Bridge_V7_Closure.md`;
- `_workspace/ce/agi-v8-breakthrough-20260811/31-validation.md` and
  `40-final-report.md`;
- `_workspace/ce/agi-v8-confirmatory-20260811/10-provenance.md`,
  `11-registration.md`, `20-audit.md`, `30-build-validation.md`,
  `31-validation.md`, and `40-final-report.md`;
- all eight reports plus `training_screen.py` in
  `_workspace/ce/agi-acbsm-development-20260811/`;
- all eight reports in
  `_workspace/ce/agi-unified-breakthrough-map-20260811/`.

The following absence checks were repeated without creating the files:

| Object | Observed at audit time |
| --- | --- |
| `artifacts/agi/sparse_causal_bridge_test_v7.json` | absent |
| `artifacts/agi/sparse_causal_bridge_test_v8.json` | absent |
| `artifacts/agi/integrated_latent_state_bridge_development.json` | absent |
| `experiments/preregistration/sparse_causal_bridge_v9.json` | absent |

Thus this document preserves the V1--V8 registrations, outcomes, seed
partitions, failures, and locks exactly as inherited. In particular, **no V8
locked split was opened**.

## 3. Independent V1--V8 lineage ledger

`PASS` below means only that a stored artifact reports every clause of that
version's registered synthetic gate as true. It is not an AGI or biological
claim.

| Version | Registered change | Exact stored result | What the result permits and forbids |
| --- | --- | --- | --- |
| V1 | Four-chart sparse bridge identification using a geometric candidate prior and paired interventions; programmed edges `A->C=+0.52`, `C->D=-0.48`; target `C` directly confounded. | **Validation FAIL; test absent.** It recovered the exact graph with precision/recall `1/1`, bridge MAE `0.00148372`, and intervention NRMSE `0.0848453`, but causal global RMSE was `0.194811` versus local `0.168985` (`-15.2828%` registered reduction), downstream reduction versus local was `-41.7815%`, the paired lower endpoint was `-0.0323688`, the direct-target lesion moved the wrong way, and the defective permuted-label control still selected both true edges. Seven checks were false. | Permits only development evidence that the programmed paired-do estimator can recover the two edges. Forbids predictive closure, a valid negative-control result, or exact replay of the buggy executable; that V1 implementation is not reachable and no implementation hash was recorded. |
| V2 | Removed direct hidden loading from both bridge targets: train `[1.15,1.25,0,0]`, OOD `[1.15,-1.25,0,0]`; fresh seeds and repaired control. | **Validation PASS; locked test PASS.** Causal global RMSE `0.155070` / `0.155941`; exact two-edge graph on both splits. The predictive-gain selector ratio was `0.999996` / `0.999998`, and equal-probe dense ratio `1.000149` / `1.000516`. | Permits a narrow matched-basis, ideal paired-intervention result with target confounding removed. It does not show a special geometric or sparse algorithmic advantage: the strongest observation-only/equal-information comparators were essentially tied. The V2 integrity record is explicitly a soft, post-run lock, and Git does not independently prove preregistration timing. |
| V3 | Restored target confounding; added an 80-transition OOD-prefix rank-one residual filter and per-seed scalar AR. | **Validation FAIL; test absent.** Exact graph, self MAE `0.0009888`, bridge MAE `0.0047149`, mean loading cosine `0.998759`, and rank-one variance fraction `0.889695` were positive. False clauses were global reduction vs fixed-local `0.5936% < 5%`, downstream `13.5530% < 15%`, mean AR absolute error `0.109347 > 0.08`, and paired lower endpoint `-0.00848675`. | Permits graph/loading recovery as development evidence. Forbids the registered prediction claim and exact source replay: the recorded V3 latent-filter source hash is not the later reachable V4 file. |
| V4 | Replaced per-seed AR with a shared AR fitted from pooled training residuals; one-step sequential prediction. | **Validation PASS; locked test PASS.** Shared AR `0.936926794` (absolute error `0.0230732`); global RMSE `0.157122` / `0.155779`; global reduction vs fixed-local `7.1382%` / `6.5831%`; all registered checks true. | Permits the strongest narrow success in the line. It forbids calling the result free rollout: the true chart state was re-read at each scored step. It also used one fixed synthetic loading family and did not discover unknown charts, hidden rank, or a biological mechanism. |
| V5 | First prefix-only single-origin H20 free rollout from `x[80]`; no future-state reads. | **Validation FAIL; test never opened.** H20 candidate RMSE `0.333083`, persistence `0.389130`, stable adaptive dense `0.312696`; maximum Jacobian radius `0.781420` and all outputs finite. The four false clauses were H5 seed wins vs persistence, H5 CI, H20 CI, and H20 vs stable adaptive dense. H20 paired lower endpoint vs persistence was `-0.014814`. | Permits a finite, leakage-resistant free-rollout implementation and favorable development means. Forbids a passed free-rollout endpoint or treating V5 as a validated parent capability. |
| V6 | Registered prefix consensus proposal extending V5. | **Never implemented or run.** No implementation, validation, test, or integrity artifact exists. | Permits a historical proposal only. It supplies no empirical evidence. |
| V7 | Reliability consensus over frozen V4/V5 components; 96 fresh validation seeds and a locked 96-seed test. | **Validation conjunction FAIL; test unopened.** Mean H20 normalized path RMSE: symmetric dense `0.5633889431`, sparse `0.5634060101`, V5 `0.5569150278`, persistence `0.5738975375`, no-sparse `0.5839581541`, adaptive dense `0.6427024216`. Sparse ablation lower endpoint was positive (`+0.0081743398`), but V5-repair lower endpoint `-0.0274308279`, persistence lower endpoint `-0.0164872019`, and pathwise radius `1.1143092447 > 0.98` failed. | Permits only the isolated observation that the sparse component helped relative to its registered no-sparse ablation. Because the gate was conjunctive, it forbids promotion of the route. The V7 hash lock establishes file integrity, not prospective Git timing. |
| V8 | Prospectively registered fixed parent-anchored output shrinkage `P + g(S-P)` after a disclosed R1 development screen. | R1 development screen passed its route-selection rule on seeds `79100..79355`: `g=0.7868543064870357`, candidate `0.548432992`, V5 `0.554138980`, symmetric-dense shrinkage `0.548593999`, lower endpoint vs V5 `+0.001105704`. **Formal confirmation on seeds `80100..80355` failed one required clause:** candidate `0.5377851901`, V5 `0.5400745832`, improvement `0.0022893931`, 95% interval `[-0.0031918816,+0.0077706678]`; status `R1_NOT_CONFIRMED`. Every other registered statistical, leakage, stability, and hash clause passed. The test `81100..81355` remained unopened. | Permits a correctly locked prospective failure and a small, uncertain mean improvement. It forbids confirmation of R1. Post-failure scalar/lead/chart gains, prefix adaptation, and multi-origin ensembles did not establish a route: best scalar `0.5376955035` still had V5 lower endpoint `-0.0018459594`; prefix adaptation was `0.5396276290`; the multi-origin fit collapsed to weights `[0.7860754,0.0008001,0,0.2131245]`; even hindsight gain `0.8235` had lower endpoint `-0.0021420624`. This closes the tested output-readout family, not all possible state mechanisms. |

The chronology boundary matters. V1--V5 registrations, code, and outcomes
first entered the inspected history together, so their in-file
`locked_pre_implementation` labels do not independently establish prospective
timing. V7's baseline likewise supplies integrity but not pre-outcome history.
V8 is different: the audited ancestor chain is

```text
B0  7c11c04db061d8ef7fc40be68c5a201766c7bd22  retrospective V1--V7 + R1 baseline
R8  6baeaf17d4b33f843785272f33ea104c80b0d8a1  V8 registration only
I8  b0abbaa16e4fddfff14064758098048c6431642d  implementation and exact lock
A8  5f20748fb16f1d1a09694f0a361954e028132d31  failed validation artifact
```

All inspected ancestor relations returned true. This makes V8's failed
confirmation the cleanest causal-bridge evidence event in the line.

## 4. Registration and artifact identity ledger

### 4.1 Registration files

These SHA-256 values were independently recomputed from the current raw files:

| Registration | SHA-256 |
| --- | --- |
| V1 | `a4ab2f9b7ba5049926d77cd7cf90352531c93b02a1af4554a26dbfb6b48dbf0e` |
| V2 | `ae609efae7af917771082caa42d7df5cfd6c41a1ffb0081203eb8b5213f90f19` |
| V3 | `28041d1679cd526f0bde39ea5cce2688aac5d92ff5ea478bd74f0d578e4c0c20` |
| V4 | `b116b3ff73d20b8dd3ce25c0db669bfdaf06e077e98eb7f17d13c9d93f8ad8b6` |
| V5 | `17a8ba5c7889557d0beaf5ab2613f6afe48ec30c8461228563185c8c6059c90d` |
| V6 | `579fabfaec662d89d06fada50a1dae9117f5befa059b10e37222d86e6ae82cf6` |
| V7 | `134ddaa793170b898649b79e11407c10f35d1468ba95701544a06905d9448c3e` |
| V8 | `a175a3d722f031e4878741a3c7136c75b1af229287e8e90442cecbc9591cdafc` |

### 4.2 Result and lock artifacts

| Object | Path / identity | SHA-256 boundary |
| --- | --- | --- |
| Historical V1 validation | recoverable Git artifact; FAIL | canonical Git-LF `58c8d1ccadb6805c46455ba9d2f9a5af70497aa7a78a87414e2a0d1b2280f690`; run-time CRLF attestation `ca9f09f32957124766ff0a5e9dedaa7ae1b5f3f9a3ecdc092d8fbdd3c3c9d282` |
| Historical V2 validation / test | recoverable Git artifacts; PASS / PASS | canonical Git-LF `2b91bd2d883f58b12d173455983c74ef6441fca0ae79ee907b26f49aef397f14` / `67cdae06a77f9b1eceb60f9239d25725714402c2d67cf8b55da21aa35c294c7f` |
| Historical V3 validation | recoverable Git artifact; FAIL | canonical Git-LF `62da4ed018a38bbc9cd8b1aff5753dded622f78822f24de9489c6e8b726c3310` |
| V4 validation | `artifacts/agi/sparse_causal_bridge_validation_v4.json` | embedded/run-time `41c17778c7aa2adcd36557ca0042ea0d2de90c817acbd8730bbc97424f553986` |
| V5 validation | `artifacts/agi/sparse_causal_bridge_validation_v5.json` | `6dd4999e385fc47ea5ccd2e3e1233c60f2d1968554b82dd5cb95a34524f9e9a0` |
| V7 validation | `artifacts/agi/sparse_causal_bridge_validation_v7.json` | `f172447ecc0d19ac206c6625bf5911805f28214bb5adc1d2a215c59dc3bc4e12` |
| V8 implementation lock | `artifacts/agi/sparse_causal_bridge_v8_implementation_lock.json` | canonical LF `afb7d296bb8694741310f34b839ffe13282514d00e53678d012eaa4658fc9ea3`; current Windows CRLF bytes `e8a58ca32156aec07535812b2f38949bd13abe532a0c603588a2f18c98466eac` |
| V8 failed validation | `artifacts/agi/sparse_causal_bridge_validation_v8.json` | canonical LF `b5850cafa1b76313f4717e924a2c4e5906095bf8cf7be3b9ceed75bf13cf53e7`; current Windows CRLF bytes `4467b8e6aff4da7e5761cb7b4b6e59c7ce569b50f66780522be53ef1dd27b855` |

The LF/CRLF pairs are a serialization boundary already documented by the
lineage; they are not evidence of semantic JSON drift. A reproducer must use
the canonical-byte convention rather than compare arbitrary checkout bytes.

Additional current identities for the nearest post-V8 design work are:

- ACBSM final report:
  `e489dfa17ac469dc13915a9ed8a52d69e79010bf5541b00b983d16255bb3997d`;
- ACBSM training screen:
  `f1de77ed7506b45956740f1add65671220edec976e178615a94a219866a92cc4`;
- unified-breakthrough final report:
  `a35d6716b25ddae01bf9a00be6021ed814244b338e0e7fb0706a7eeab7c221c3`.

## 5. ACBSM and unified-breakthrough audit

### 5.1 What was actually built

The unified-breakthrough report selected one proposed Adaptive Causal
Belief-State Model: a frozen sparse mechanism plus a prefix-inferred
fast/slow low-rank residual belief, posterior covariance, and
transition-internal correction. Regime belief, resets, episodic priors,
planning, hypothesis beams, and edge adaptation were explicitly reserved but
disabled. The unified report itself ran no model, seed, or V9 gate.

The subsequent ACBSM development lane implemented:

- a prefix-only linear-Gaussian posterior observer;
- episode-boundary-safe residual moments;
- rank-one/rank-two PSD decompositions;
- correction applied inside each rollout transition rather than as a final
  output gain;
- a predeclared stability rule that collapses unsupported modes.

Focused tests passed (`8` ACBSM tests; `65` inherited V1--V8 plus ACBSM tests),
but these are implementation tests, not evidence of predictive superiority.

### 5.2 Exact training-only result

Only eight inherited training episodes `45100..45107` were screened in an
eight-fold leave-one-episode-out procedure:

| Quantity | Observed value |
| --- | ---: |
| legacy mean H20 RMSE | `0.6300720964` |
| ACBSM mean H20 RMSE | `0.6057545840` |
| mean improvement | `0.0243175124` (`3.8594809%`) |
| paired 95% episode interval | `[-0.0223078218,+0.0709428467]` |
| held-episode wins | `5/8` |
| sparse-core / symmetric-dense ratio | `1.0020100` |
| promise score | `67.12584746369484 / 100` |
| classification | `HOLD` |

The raw fast mode was supported in only `4/8` folds, with fitted poles ranging
from `0.075` to `0.8`. The frozen stability rule therefore collapsed the
selected candidate to **rank one in every fold**. Rank two added effectively
zero improvement over rank one.

No seed in the reserved fresh ACBSM block `82100..82355` was simulated, no
implementation lock was issued, and no V9 was registered. The result supports
keeping a rank-one observer as a development ancestor; it does not validate a
two-timescale state, nested hierarchy, or V9.

### 5.3 Why a nested recurrent state can still be new

ACBSM is more than V8 output shrinkage because its residual belief enters the
transition. But after rank collapse it is still a **single residual state**
computed by an analytically fitted Gaussian observer. It contains no explicit
level-indexed tower, no learned cross-scale messages, no actions/rewards or
policy, and no causal lesion study showing that stored history mediates
held-out decisions.

A V9 candidate is a genuinely new causal-state mechanism only if all of the
following are true:

1. It exposes at least two explicit persistent states `z_t^(0), z_t^(1), ...`
   and registered cross-level messages, rather than merely naming two
   eigenmodes or layers.
2. Candidate and controls receive the same raw observation/action/reward
   streams. Prediction/action is computed **exclusively** from the registered
   state and current legal input. There is no analytic posterior, frozen V5
   output, future observation, oracle state, or direct-current-input bypass
   capable of reproducing the decision.
3. Same-current-input/different-history trials change the internal state and
   prediction in the registered direction; decoding tests show that the
   relevant history is present in state.
4. Level resets, cross-level cuts, delay/time shifts, sign flips, and message
   shuffles cause predeclared held-out changes. Matched random lesions and
   parameter/compute-matched controls do not explain the result.
5. It beats V5 and matched ACBSM, monolithic recurrent, finite-depth,
   no-cross-scale, and same-information controls on a fresh development block
   before any confirmation is unlocked. Parameter count, persistent-state
   size, latency, and MAC budgets are frozen and reported for every arm.
6. Any direct-limit or quotient claim has explicit compatible maps and a
   finite truncation/convergence error. Topological recurrence alone is not a
   numerical convergence theorem.

Conversely, V9 is **not new** if it is only:

- an SCC decomposition of a frozen adjacency matrix;
- a nested visualization of subgraphs or communities;
- more output gains, lead-wise shrinkage, or ensemble weights from the V8
  family;
- duplicated finite layers with no persistent level-specific history;
- a relabelled rank-one/rank-two ACBSM posterior;
- a generator queried at several indices while decisions bypass the generated
  states;
- a finite computation called “infinite” without a direct-limit object,
  compatible embeddings/semiconjugacy, and a truncation certificate.

## 6. Required graph-language distinctions

The distinction is structural, not terminological. Tarjan's original SCC
algorithm paper is [Depth-First Search and Linear Graph Algorithms,
DOI 10.1137/0201010](https://doi.org/10.1137/0201010).

| Object | Formal content | May be nested? | What it does **not** imply |
| --- | --- | --- | --- |
| Maximal SCC of one fixed directed graph | An equivalence class under mutual directed reachability. The SCCs uniquely partition the vertex set; the condensation graph is acyclic. | Distinct maximal SCCs are disjoint, so not properly nested. | A stable attractor, convergence, shared function, or a multiscale hierarchy. |
| Strongly connected subgraph | Any selected subgraph in which every pair is mutually reachable inside that subgraph; it need not be maximal. | Yes. A chain `G_0 subset G_1 subset ...` can be nested. In the union graph, finite levels are generally strongly connected subgraphs, not separate maximal SCCs. | That the selected boundaries are unique, data-driven, or biological. |
| Community/module | A partition or cover optimized for density, modularity, spectral cuts, block models, anatomy, or another objective. Direction may be discarded. | Yes, depending on method and resolution. | Mutual directed reachability. A community is not automatically an SCC. |
| Hierarchy/gradient | An ordering, tree, nested partition, partial order, laminar level, or continuous coordinate across areas. | Often by construction. | Cycles, SCCs, causal direction, or dynamical recurrence. |
| Dynamical recurrence | State at time `t+1` depends on earlier state through a feedback update or recurrent path. | Can occur within or across scales. | Structural strong connectivity; convergence requires separate dynamical assumptions. |
| Positive-delay event unroll | Nodes are time-stamped events and edges advance time. | It may contain repeated motifs, but its forward-time graph is a DAG. | Nontrivial SCCs in the event graph. Recurrence belongs to the finite substrate/operator, not the time-unrolled DAG. |

This yields a precise biological reading. If 93.3% of a fixed fly connectome is
one giant maximal SCC, a recurrent microcircuit inside that giant component is
a strongly connected **subgraph**, not another SCC of the same graph. A V9
“tower” must therefore declare whether its levels are subgraphs under
injective inclusions, SCCs of different scale-dependent quotient graphs, or
dynamical state spaces. Interchanging these meanings invalidates the claim.

## 7. Primary neuroscience and connectome evidence

Only primary research papers and their official data records are used for the
claim ledger below. The audit is current through 2026-08-11.

| Primary source and exact graph construction | What it permits | What it forbids for V9 |
| --- | --- | --- |
| **Adult fly, FlyWire network statistics.** Lin et al., 2024, [DOI 10.1038/s41586-024-07968-y](https://doi.org/10.1038/s41586-024-07968-y), data [Zenodo 10.5281/zenodo.10676865](https://doi.org/10.5281/zenodo.10676865). The analyzed v630 directed neuron graph had `127,978` neurons and `2,613,129` connections after removing synapses with confidence `<50` and requiring at least five synapses per neuron-pair edge. | A directly computed giant SCC contains `93.3%` of neurons; giant WCC `98.8%`; mean directed shortest path `4.42`, maximum `13`. The giant components persisted across reported threshold and edge-pruning controls. This is strong evidence for a large recurrent core in **that constructed graph**. | It does not show several nested maximal SCCs, any infinite substrate, an attractor, or cognition. v630 optic lobes were about 80% complete; the graph is threshold-dependent, chemical-synapse based, from one adult female, and omits unknown gap-junction/extrasynaptic effects. |
| **Adult fly finite reconstruction.** Dorkenwald et al., 2024, [DOI 10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y). The later v783 reconstruction contains `139,255` proofread neurons and `54.5` million assigned chemical synapses; it reports remaining false-negative/false-positive and threshold limitations. | Establishes a finite physical substrate with brain-wide directed chemical edges and explicit reconstruction uncertainty. | Does not license “infinitely many physical agents/neurons,” and v783 size must not be silently mixed with v630's 93.3% SCC statistic. |
| **Larval fly whole brain.** Winding et al., 2023, [DOI 10.1126/science.add9330](https://doi.org/10.1126/science.add9330), `3,016` neurons and about `548,000` synapses. Hierarchical connectivity clustering produced nested neuron-type clusters; recurrence was defined by up-to-five-hop cascades returning from downstream partners to a source under the paper's connection rules. | The authors report that `41%` of brain neurons participated in at least one such recurrent loop and explicitly describe a “nested recurrent architecture.” This is direct precedent for recurrent pathways distributed across a hierarchical clustering. | The clusters were neuron types with similar connection profiles and were explicitly not necessarily dense within-cluster communities. The recurrence census is not an SCC computation and does not prove a direct-limit tower, maximal nested SCCs, persistent multilevel state, or convergence. “Nested recurrent” here cannot be imported as the V9 theorem. |
| **Adult fly brain-and-nerve-cord, BANC.** Bates et al., 2026, [DOI 10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w), final materialization v888; static data [Harvard Dataverse 10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N), image data [BossDB 10.60533/boss-2025-941r](https://doi.org/10.60533/boss-2025-941r). It reports `155,916` proofread/roughly proofread neurons. For the 13 “CNS networks,” the authors selected `50,568` neurons, symmetrized normalized directed weights as `a_ij=(w_ij+w_ji)/2`, formed a **weighted undirected** graph, then used spectral embedding and k-means with `k=13` chosen as a useful coarse resolution. | Supports a finite, embodied, distributed control architecture with local feedback loops, long-range ascending/descending links, reciprocal cross-network connectivity, and multiscale organization. | The 13 networks are spectral communities, not SCCs. Sensory, effector, and optic-lobe intrinsic neurons were excluded from that clustering. BANC is a living, single-specimen reconstruction; lamina and ocellar ganglion are missing, only `18%` of detected synapses had identified cells on both sides in the report, and known artefact boxes remain. It supplies no nested-SCC census or V9 causal-state lesion. |
| **C. elegans directed connectome.** Varshney et al., 2011, [DOI 10.1371/journal.pcbi.1001066](https://doi.org/10.1371/journal.pcbi.1001066). The chemical graph has `279` neurons and directed chemical edges; a separate union adds gap junctions as two opposed directed edges. | Chemical-only graph: giant SCC `237`, one 2-neuron SCC, and 40 neurons outside those nontrivial SCCs. Chemical-plus-electrical union: giant SCC `274/279`. This demonstrates that SCC membership changes with the chosen edge layer and direction convention. | It does not prove functional nestedness. Treating electrical junctions as reciprocal directed edges is a modelling convention, and the historical aggregate connectome is not an individual-invariant exact graph. |
| **C. elegans reconstruction uncertainty.** Cook et al., 2019, [DOI 10.1038/s41586-019-1352-7](https://doi.org/10.1038/s41586-019-1352-7), whole-animal connectomes of both sexes; Witvliet et al., 2021, [DOI 10.1038/s41586-021-03778-8](https://doi.org/10.1038/s41586-021-03778-8), eight individuals across development, data [BossDB 10.60533/BOSS-2020-FQI7](https://doi.org/10.60533/BOSS-2020-FQI7). | Establishes finite directed chemical/electrical reconstructions and empirically important individual/developmental variability. | Forbids treating one thresholded/aggregated adjacency matrix or one SCC partition as a timeless species-level ground truth. |
| **Mouse mesoscale directed atlas.** Oh et al., 2014, [DOI 10.1038/nature13186](https://doi.org/10.1038/nature13186). `469` unilateral anterograde AAV tracer experiments were fit under regional-homogeneity and projection-additivity assumptions to a weighted directed model among `213` regions. | Supports dense reciprocal mesoscale projections, multiscale regional organization, and a direction-bearing anatomical dataset suitable for an explicitly registered SCC analysis. | The atlas is a fitted regional model, not a synaptic graph or an already reported whole-brain SCC tower. Static structure was explicitly described as necessary but insufficient for function. Threshold, parcellation, tracer sensitivity, symmetry, and model assumptions can change any SCC census. |
| **Mouse cortical/thalamic hierarchy.** Harris et al., 2019, [DOI 10.1038/s41586-019-1716-z](https://doi.org/10.1038/s41586-019-1716-z). Around one thousand new layer/cell-class tracer experiments were used to model projection patterns as feedforward or feedback and assign hierarchical positions. | Supports cell-class-specific feedback and a shallow corticothalamic hierarchy. | A fitted feedforward/feedback hierarchy is not an SCC decomposition, and hierarchy does not by itself yield nested recurrent causal states. |
| **Mouse cortico-basal-ganglia-thalamic loops.** Foster et al., 2021, [DOI 10.1038/s41586-021-03993-3](https://doi.org/10.1038/s41586-021-03993-3). Tracer mapping identified six parallel subnetworks; thalamic projections returned to the originating corticostriatal neurons, with monosynaptic electrophysiological confirmation in the reported associative loop. | Strong primary evidence for anatomically and physiologically supported closed recurrent loops arranged as parallel subnetworks. | Six loops/subnetworks are not six maximal SCCs, do not form an infinite nested tower, and do not establish V9's history-mediation or cross-scale lesion gate. |
| **Interareal module hierarchy.** Rubinov, 2016, [DOI 10.1038/ncomms13812](https://doi.org/10.1038/ncomms13812). Mouse and fly interareal directed weighted connectomes were analyzed with modularity maximization over resolution parameter `gamma`; stable low/high-resolution partitions defined a two-level module hierarchy. | Supports multiresolution communities and rich-club/module organization in interareal connectomes. | The modules are objective- and resolution-dependent communities, not SCCs. The paper itself warns that simultaneous module/hub/hierarchy characterizations can involve circular constraints (“double dipping”). |
| **Human lifespan functional hierarchy.** Taylor et al., 2026, [DOI 10.1038/s41586-026-10219-x](https://doi.org/10.1038/s41586-026-10219-x). Resting-state fMRI correlations from five cohorts were embedded into three dominant functional-connectivity gradients; `3,972` gradient sets from `3,556` individuals survived QC. | Supports reproducible multiscale functional gradients and age-dependent hierarchy-like organization. | The paper explicitly says these axes are continuous processing motifs, not strict causal or unidirectional pathways, and calls the analysis exploratory. Symmetric fMRI correlation does not supply edge polarity, SCCs, nested recurrence, or a mechanistic causal state. |
| **Finite human substrate estimate.** Azevedo et al., 2009, [DOI 10.1002/cne.21974](https://doi.org/10.1002/cne.21974), isotropic-fractionator estimate `86.1 +/- 8.1` billion NeuN-positive cells and `84.6 +/- 9.8` billion non-neuronal cells in adult male human brains. | Reinforces that a biological brain is a large but finite physical system. | Does not identify connectivity or dynamics and forbids literalizing the mathematical direct limit as infinitely many physical human neurons. |

### 7.1 Direct answer to the “whole brain is SCCs” question

- **Already established in a precise limited sense:** every fixed finite
  directed graph decomposes into maximal SCCs. Lin et al. directly measured
  that one SCC contains 93.3% of the v630 thresholded adult-fly neuron graph;
  Varshney et al. reported a `237/279` giant chemical SCC and `274/279` giant
  union SCC for C. elegans under their edge conventions.
- **Not established:** that biologically meaningful whole-brain modules are
  several nested maximal SCCs; that the same SCC partition persists across
  animals, thresholds, synapse layers, parcellations, or time; or that SCCs
  are stable computational states.
- **No source located in this primary audit** reports the V9 object: a proper
  tower of finite strongly connected subgraphs with registered inclusions,
  compatible recurrent state maps, causal cross-level messages, and a
  certified direct-limit/truncation semantics.

## 8. Claim-permission gate

| Candidate statement | Status after source audit | Exact boundary |
| --- | --- | --- |
| “A fixed directed brain graph can be decomposed into SCCs.” | **Permitted theorem-level graph statement.** | The graph, nodes, directed edge layer, threshold, and time snapshot must be fixed. |
| “A giant recurrent core has been observed in a whole adult fly graph.” | **Permitted empirical statement.** | Say `93.3%` of the FlyWire **v630, confidence>=50, >=5-synapse-edge** graph; do not substitute v783 or BANC counts. |
| “Brains exhibit feedback, recurrent pathways, communities, and hierarchies across scales.” | **Permitted broad empirical summary.** | Preserve each paper's graph construction; do not collapse these four notions into SCC. |
| “The whole brain is several nested SCCs.” | **Forbidden as written.** | Maximal SCCs of one fixed graph are disjoint. Replace with a registered tower of strongly connected subgraphs or SCCs of explicitly different quotient graphs, then test it. |
| “The brain has infinitely many neurons/agents.” | **Forbidden.** | Physical connectomes and cell counts are finite. “Infinite” may denote an ideal direct limit or indefinitely extendable event computation only. |
| “A cycle or SCC guarantees convergence/fixed point.” | **Forbidden.** | Topology gives reachability. Convergence needs a specified update, norm/metric, invariant domain, and contraction/dissipativity/ergodic theorem with uniform control across levels. |
| “ACBSM already proves the nested V9 mechanism.” | **Forbidden.** | ACBSM was training-only `HOLD`; rank two collapsed to rank one; no fresh block or causal hierarchy lesions were run. |
| “V9 is new because its graph is nested.” | **Forbidden.** | Novelty must reside in causally necessary persistent multi-level state and cross-scale messages under matched controls, not topology or naming. |
| “Passing a future V9 synthetic gate proves brain identity, cognition, consciousness, or AGI.” | **Forbidden.** | At most it would support the registered synthetic mechanism and intervention claims. Biological identity and AGI remain separate, untested bridges. |

## 9. Source-lane handoff

The defensible V9 route is therefore:

```text
finite physical substrate
    + explicitly chosen nested strongly connected subgraphs/quotients
    + persistent level-indexed recurrent states
    + registered cross-scale causal messages
    + compatible maps and finite-truncation certificate
    + history mediation and level-specific lesions
    + fresh, matched-control predictive advantage
```

The SCC tower is a useful **representation/design prior**. It becomes a new
mechanism only when the state variables, updates, observables, interventions,
and falsifiers make the tower causally indispensable. Until that gate is run,
V9 remains `THEORY/DESIGN`, the V1--V8 outcomes remain immutable, ACBSM remains
`HOLD`, and the V8 locked test remains unopened.
