# Brain-algorithm route ledger

Status: ACTIVE

## Latest admission decision — 2026-08-21

This section is the authoritative next-route order and supersedes the older
active-candidate ordering below.  It changes route management only; it does not
promote a simulator or schema audit into a biological mechanism claim.

| ID | Route status | Formal status | Preserved claim | Evidence | Never-retry / resume rule |
|---|---|---|---|---|---|
| BA-EMP-RANDI | `NATIVE_EVENT_RECOVERY_COMPLETE / PARTIAL_SOURCE_INDEX_SCHEMA / EMPIRICAL_ROUTE_BLOCKED` | **[관측 비교: 입력 감사]** | 113개 numeric-prefix session 중 87개 complete session에서 4,457 stimulation row를 검사했다. 3,537개 nonnegative source-index join 중 2,710개만 nonblank local label에 도달했고 827개는 blank였다. 이것은 event 번호→recording-local source index→일부 local label의 부분 스키마이며, canonical identity 또는 효과·라우팅 결과가 아니다. `PASS_SOURCE_INDEX_JOIN`, `PASS_SOURCE_JOIN`, `PASS_ASSIGNMENT_RECEIPT`은 모두 false다. | `randi-native-event-recovery-20260820/20-audit.md`, `9f9b02797bb2d418ec71a5833cbbd15675777eae6fe655b8a46b5eca72f3ef4f`; validation `31-validation.md`, `3bff495394eb224dfe50acabc923f0fe82004ddb3ed1879f61a1825bb5d362a3`; final `40-final-report.md`, `5c8e01113da8537576a96302f91bc6b03377c916dba7219e0ddfe56eba7de2ec`; machine artifact `artifacts/native_event_audit.json`, `ba55f83e2436f4b2be373c0254d60378efff7d3dca5494895c68a1771c8ebbec` | outcome-tuned spatial matching, pre/post, failed-autoresponse 선택으로 source identity·무광 대조·효과를 만들지 않는다. 재개는 (a) complete original event assignment receipt와 canonical identity confidence/provenance를 함께 가진 immutable 원자료, 또는 (b) frame·unit/z·transform·radius·ambiguity·tie-break를 함께 고정한 완전한 immutable geometric validation object로 한정한다. |
| BA-A6-P | `MATH_PASS / EMPIRICAL_UNTESTED` | **[조건부 정리]** | 고정 history·input path와 $G_T\succ0$에서 flow derivative $J_T$가 full rank이면 $g_{\rm pass}=J_T^\top G_TJ_T$는 상태좌표의 passive pullback Riemann metric이다. rank loss이면 PSD degeneracy만 남는다. 회로 변화 $W(\varepsilon)$의 $\dot g_\Gamma$는 같은 조건의 국소 민감도다. 실제 뇌, AGI, 또는 피질 주름 효과는 여기서 나오지 않는다. | `brain-circuit-manifold-equations-20260821/11-math.md`, `ec6a23a6093df33204306759e73b86d63086e3a8728d2ae5eb9067ba9fccaf83`; audit `20-audit.md`, `7af493542bedd94986c719e38b121d9d68b966344ddebae9f0e8ebb8e3f6993c`; validation `31-validation.md`, `23531a1fdaf6fab5882a42c61e4bc2692b34a9063c84a3615c6627ec0860c2e9` (`17 passed`, deterministic math witness PASS) | anatomy bridge는 `BLOCKED_INPUT`: longitudinal embedding, thickness, growth/material law, boundary/observation map이 필요하다. independent calibration 없이는 $b_i$와 neuron threshold $\theta_i$가 $b_i-\theta_i$로만 식별된다. |
| BA-A6-C | `MATH_PASS / EMPIRICAL_UNTESTED` | **[조건부 정리]** | fixed LTV delayed tangent system, $R\succ0$, reachable terminal $v$에서 $E_T^*(v)=v^\top\mathcal W_c(T)^\dagger v$는 finite-horizon endpoint minimum-control-energy다; unreachable $v$에는 $+\infty$다. 이는 actuator $B$, cost $R$, horizon $T$, reference trajectory에 의존하는 endpoint quadratic form이며 local Riemann/sub-Riemannian metric이 아니다. | `brain-circuit-manifold-equations-20260821/00-contract.md`, `54b5d7716c6e57df9113ecaf770bcc61452e8e5019efce92f39bf2940581286b`; math `11-math.md`, `ec6a23a6093df33204306759e73b86d63086e3a8728d2ae5eb9067ba9fccaf83`; final `40-final-report.md`, `80878ce5793b8526c7517dbdc74332e991db3cec0a5b2594454c4ab82df6f3a6` | real edge delay, signed strength, efficacy, calibrated offsets, actuator map and fixed cost must be joined in one session/event frame before empirical use. No A3--A5 threshold·clip·RMS·ridge·horizon/seed retune is admissible; their disjoint real-data failures remain retired. |

## A6 property-loop normalization (2026-08-21)

This normalization supersedes the status, evidence, and retry cells for
`BA-A6-P` and `BA-A6-C` in the admission table above.  It does not broaden
their mathematical domains or convert synthetic properties into biological or
AGI evidence.

| ID | Route status | Formal status | Frozen property evidence | Preserved boundary / next rule |
|---|---|---|---|---|
| BA-A6-P | `MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED` | **[conditional theorem]** | Eight of eight frozen smooth delayed fixtures passed passive tangent, total circuit derivative, coordinate-covariance, and rank-gate checks. Property audit `brain-circuit-manifold-property-loop-20260821/20-audit.md`, `3eddcd5830c4eb2d31ec073c05264cd51bd5da2d8ea8827becbe196401f08dd8`; validation `31-validation.md`, `7dbc77f3a844ccd773301f2bec135d80ba1b61fed131417741e08148393a0654`; final source/result `c67f1f790c291f622db6362f31eeb58b9b7e4bc147d7c8d99d08f48fe511074d` / `f6130d58cfe5e8d20b6ea9987c467e473ad9c4a183c4c54f19678e3d1d8a89bd`; final `40-final-report.md`, `d35491b525a6d6b11e16ef46a87f5a2ca85fb65bcb48d5835cc0b260fbf1fd27`. | Revision 1 corrected only the implementation gates and receipt provenance: it kept formula, seeds, fixture, finite-difference step, and tolerances frozen. No formula revision occurred. Cortical anatomy remains `BLOCKED_INPUT` pending longitudinal embedding, thickness, growth/material law, and boundary/observation receipts. Without independent calibration, only $b_i-\theta_i$ is identifiable. |
| BA-A6-C | `MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED` | **[conditional theorem]** | Eight of eight frozen fixtures passed Gramian PSD, least-norm energy, chart-energy, and $\dot E$ checks; unreachable and rank-deficient adverse controls were killed. Property audit `brain-circuit-manifold-property-loop-20260821/20-audit.md`, `3eddcd5830c4eb2d31ec073c05264cd51bd5da2d8ea8827becbe196401f08dd8`; validation `31-validation.md`, `7dbc77f3a844ccd773301f2bec135d80ba1b61fed131417741e08148393a0654`; final source/result `c67f1f790c291f622db6362f31eeb58b9b7e4bc147d7c8d99d08f48fe511074d` / `f6130d58cfe5e8d20b6ea9987c467e473ad9c4a183c4c54f19678e3d1d8a89bd`; final `40-final-report.md`, `d35491b525a6d6b11e16ef46a87f5a2ca85fb65bcb48d5835cc0b260fbf1fd27`. | Revision 1 is an implementation-gate correction only, not a formula revision. Empirical use still requires real edge delay, signed strength, efficacy, calibrated offsets, actuator map, and fixed cost in one session/event frame; A3--A5 threshold/clip/RMS/ridge/horizon/seed retuning remains inadmissible. |

## A7-H discrete-hybrid normalization (2026-08-21)

This entry supersedes the former `BA-A7-H OPEN / UNTESTED` admission row. It
records a frozen synthetic runtime property only; it does not promote the
result to evidence about biological neurons, learning, AGI, cortical folding,
or physical manifold deformation.

| ID | Route status | Formal status | Frozen property evidence | Preserved revisions / boundary / next rule |
|---|---|---|---|---|
| BA-A7-H | `DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED / HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED / EMPIRICAL_UNTESTED` | **[conditional theorem / implementation property]** | Frozen actual-`BrainRuntime` fixture passed: Torch/mirror one-step max error `4.8676e-8`; full $24\times24$ fixed-branch Jacobian normalized error `2.6270e-12`; reachable clip-face one-sided error `6.5854e-8`; delay-ring first arrival at call $t=2$ (`3.5435e-3`); lifecycle next-tick effect `2.3413e-2`; permutation residuals `0`; and no-delay Torch/Rust error `7.4506e-9`. The delay-on adverse control reproduced the expected activation mismatch `3.3332e-2`, so it is a blocker rather than a parity PASS. Contract `00-contract.md`, `86f72a7f188e63f03edb19efbe9eb67613b254f3aa0db4214953e684658ac1fe`; audit `20-audit.md`, `e74f8875b10da49a2b8eaba48b72ce56fb76d365be1530e5a970566d28748861`; validation `31-validation.md`, `3da81fba48d92497d42f151e41e9362a285a81354169674ecbdc374d3441e67`; final `40-final-report.md`, `0928792665b79de2bf40bfa98d308aeeb10625c50a4cdc326c0bdad567c6ab9e`; final witness/result `9021d90352933d903b7af6c716d47d27773792978060805d931994de6ad8fad8` / `cc6954b7f8120fb231494a4cc5cc0498895270ff1a93dfa834027f3fb22c9e6a`. | P2 apparatus revision 00 preserved the pre-evaluation bad Rust-source-path witness (`4f598977359d0eb6c9488c8d825e2f4aeef3151e98a1d5ef5edb67a55852dd1b`) and corrected only two path literals. P2 apparatus revision 01 preserved the passing witness/result (`2878bca53478dfbbc966f6b6189d7cfddd3611c1aec561061957560c4f847462` / `52ed6b73fa4fdc94132027ed7a5ce1dbb39de85aaa6460469a43028d46bc1d20`) and added only the source-tree package-version receipt `reality_stone.__version__=0.2.10`; every formula, fixture, direction, step, tolerance, source hash, gate, and claim ceiling remained frozen. Discrete bit/TopK/lifecycle crossings remain receipt-only with undefined derivatives; continuous-time saltation and arbitrary chart covariance remain prohibited. Next independent routes are A8-T threshold vectors, then A8-D Rust delay repair. |

## Immediate next priority

| Priority | Candidate | Admission state | Required falsifier / STOP |
|---:|---|---|---|
| 1 | BA-A8-T | `OPEN / UNTESTED` | **[unfinished]** Implement neuronwise runtime configuration/state for $\theta_i^-$, $\theta_i^+$, and $\vartheta_i$. The scalar configuration must remain an exactly equivalent broadcast compatibility path. Freeze permutation and guard-receipt checks; stop on any scalar-regression, index-permutation, or guard-semantics mismatch. This is a threshold implementation route, not evidence for biological threshold distributions, AGI, or anatomy. |
| 2 | BA-A8-D | `OPEN / UNTESTED` | **[unfinished]** In a separate code/contract route, repair Rust to expose and apply the same delay ring buffer, unbounded counter, read-before-write, and snapshot semantics as Torch. Reuse A7-H's delay-on mismatch as the fail-closed adverse control; stop on any mismatch after a frozen shared-state parity fixture. Do not retune the delay mismatch away. |

| Priority | Candidate | Admission state | Required falsifier / STOP |
|---:|---|---|---|
| 1 | BA-EMP-IBL | `OPEN_INPUT__PREDICTIVE_TRANSFER_ONLY` | **[미완성]** 공개 IBL 연구의 단순 재실행은 하지 않는다. 공식 cache의 exact simultaneous session에서 source×state interaction 모델이 additive state model보다 held-out predictive transfer를 개선하는지만 사전 고정한다. intervention이 없으므로 causal routing 주장은 금지한다. |
| 2 | BA-EMP-CLOUD-G | `PASS_INPUT_METRIC_ONLY` | Output-Fisher geometry may be tested within recordings; anatomical source-target routing and causal claims remain blocked. |
| 3 | BA-BIO-LONG | `BLOCKED_INPUT` | Requires same-cell/synapse longitudinal structure, calibrated activity and intervention data. |
| 4 | BA-S1 | `DEFERRED_SYNTHETIC` | Do not substitute a synthetic support threshold or lesion seed for real structural evidence. |

이 원장은 BrainRuntime 실험에서 어떤 알고리즘 후보가 살아 있고 어떤 경로가 기각·퇴역했는지를 빠르게 감사하기 위한 문서다. 실제 뇌를 설명하는 논문이나 의식 이론의 서사가 아니다.

## 정의와 경계

**[정의]** 이 원장의 `CONFIRMED_SIMULATOR`, `STOP`, `APPARATUS_INVALID`, `BLOCKED_NOT_IDENTIFIED`, `OPEN`은 실행 경로의 관리 상태다. CE의 형식 지위나 생물학적 진리 판정이 아니다.

**[공리: 모델 경계]** 고정된 BrainRuntime 코드·설정·seed에서 재현한 결과만 해당 simulator 주장에 사용한다. simulator 결과를 실제 뇌의 기억, 학습, connectome, SCC 또는 의식과 동일시하지 않는다.

**[정의]** 새 후보는 이전 실패와 다른 mechanism 또는 개입 seam을 가지며, 그 차이를 죽일 독립 falsifier와 matched control을 사전에 포함해야 한다. threshold·seed·endpoint·decoder만 바꾼 반복은 새 후보가 아니다.

## 실행 증거 원장

| ID | 후보·mechanism | 실행 상태 | 형식 지위 | 보존되는 좁은 주장 | 증거 경로와 SHA-256 | 재시도·재개 조건 |
|---|---|---|---|---|---|---|
| BA-M0 | supervised rank-4 recurrent write | `CONFIRMED_SIMULATOR` 32/32 | **[산출]** 고정 simulator 계산 | rank 4에 32개 연상의 supervised capacity가 있다. 획득 규칙의 증거는 아니다 | `brain-mechanism-alternative-routes-20260819/artifacts/m0-m1-confirmation-results.json`, `536590a9d38669c5c7fc7485b388f7c4af2d413e213de1cfa68613c287c8f8bb` | brain-algorithm 후보로 재실행하지 않는다. capability ceiling으로만 사용 |
| BA-M1 | fixed-clock delayed local eligibility + replay | `CONFIRMED_SIMULATOR` 32/32 | **[산출]** 고정 simulator 계산 | zero-store cue/value binding acquisition과 여섯 adverse-control 분리를 확인했다 | 같은 confirmation artifact와 SHA | factor transfer·prediction·biological consolidation을 별도 입증해야 한다 |
| BA-T1 | frozen M1 factorized held-out composition | `STOP` 11/16 | **[산출]** 부정 계산 | M1 binding은 안정된 조합 전이를 보장하지 않는다 | `brain-memory-contrastive-predictive-routes-20260819/artifacts/t1-development-results-v2-audited.json`, `1c1914b952ead084a21a88a35abca983314dccb013e57c333d1d1075436841fa` | 같은 불균형 schedule의 threshold/seed retune 금지. factor-balanced mechanism이 새로 필요 |
| BA-M2 | positive-minus-negative lag contrastive write | `STOP` 0/16 | **[산출]** 부정 계산 | frozen schedule에서 negative phase가 정확히 0이고 positive write도 recall을 만들지 못했다 | binding `brain-memory-contrastive-predictive-routes-20260819/artifacts/m2-binding-development-results-v2-frozen.json`, `0ddbe12b5b78c6c3e9a4f1d4d14a1b5b24690807a0d6f45646f84ec368535567`; factor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m2-factor-development-results-v2-frozen.json`, `f52a2ee5bf5424c27823eb2441514357b0236d939c704f3dea2cfb0ab9764d11` | nonzero negative phase를 만드는 독립 mechanism 없이는 퇴역 |
| BA-M3 | teacher-forced replay residual write | `STOP` | **[산출]** 부정 계산 | binding capability는 있으나 predictor가 persistence를 이기지 못했고 transition-order shuffle이 binding을 재현했다 | predictor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-predictor-development-results-v2-frozen.json`, `4e865af022bc7e8ac33a11861a83816b7e4b94fce097ed382f88fc3d45fbdaff`; binding `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-binding-development-results-v2-frozen.json`, `80db6e32ce50a84a716f9c989441fdea8f5dce19b3f011113bf46188a8bd0879`; factor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-factor-development-results-v2-frozen.json`, `2b89b82ffb757e9d6cbb97f7a52ebeafbfba79998092845dfb9ab8cf6f7db8b3` | predictor→policy 인과 seam처럼 학습 write와 독립된 새 시험만 허용 |
| BA-G1 | directed $\operatorname{do}(W)$ → SPD response + endpoint | `STOP` 0/16 | **[산출]** 부정 계산 | 평균 효과는 양수였지만 preregistered per-circuit effect gate를 통과하지 못했다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g1-development-results-v1.json`, `3e2a69b22bce0ae906bdf4b3fd3a2830421df6f385969e19577a0ca6b1cca6f3` | 효과크기 문턱·noise control을 결과 후 바꾼 재시도 금지 |
| BA-G2 | compressed SPD metric feature for fixed-W prediction | `STOP` 0/16 | **[산출]** 부정 계산 | $g$는 유효한 압축 표현이지만 raw horizon $B_h$, direct quadratic, $C$ terms보다 유용하지 않았다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g2-development-results-v1.json`, `c4fbeeee6cc3e71f596238e902007c30f942cad389d9bf6e45884cdd0437b489` | “metric sufficiency/고유 정보” 경로는 퇴역. 새 독립 mediator 개입 없이는 재개 금지 |
| BA-G3D-v1 | response/recall diagnostic, first apparatus | `APPARATUS_INVALID` | **[미완성]** | 과학 결과로 사용하지 않는다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g3-diagnostic-development-results-v1.json`, `2075e3516ffab8c21691535540443f8b4743609b227416bbd420b40c15e2ee9e` | seed `97701..97716` 영구 퇴역 |
| BA-G3D | independent response-summary/recall co-change diagnostic | `STOP`; mediation `BLOCKED_NOT_IDENTIFIED` | **[산출]** 부정 계산 + **[미완성]** mediation | M1 continuous recall advantage는 강하지만 global SPD-change 우위와 same-arm correlation은 실패했다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g3-diagnostic-development-results-v2.json`, `a0a9321aca3366d1c6c7d4e12f7cfe97d1387e86c6e05799ee90543a01a665b7` | independently manipulable mediator 또는 충분한 causal state model 없이는 mediation 재개 금지 |
| BA-C1 | frozen action-conditioned predictor at a planner port | `STOP` 0/16 advantage gate | **[산출]** 고정 simulator 부정 계산 | 예측 MSE와 planner-port 개입은 통과했지만, guided policy가 zero-action persistence와 reactive mean-effect control보다 손실이 컸다 | `brainruntime-prediction-guided-metacontrol-20260820/artifacts/c1-development-results.json`, `4e21f994ae1f6f2563a2c00bc13fba7ec9ac812e02701fed54d6eb6cdd49ae0b`; canonical results `de2b59cb54de3a6d2007c3282b13b0c3bb80e45a55c5ec9af923003157be8b28` | seed·goal·threshold·action set retune 금지. confirmation은 봉인하고 합성 controller 후보로 재개하지 않는다 |

## 활성 후보 순위

| 순위 | ID | 상태 | 선택 이유 | 필수 falsifier / stop 조건 |
|---:|---|---|---|---|
| 1 | BA-EMP-CLOUD | `OPEN_REAL_DATA` | 로컬에 이미 보존된 실제 *C. elegans* 전뇌 칼슘 활동과 동시 locomotion 시계열에서 task-output Fisher pullback과 lagged predictive routing을 같은 recording 안에서 독립 추정한다 | 원자료 provenance·MAT schema·시간축·행동 정렬이 통과해야 한다. covariance inverse를 metric으로 부르거나 structural connectome ID를 억지로 결합하면 STOP |
| 2 | BA-EMP-IBL | `OPEN_INPUT` | IBL Brain-wide Map의 동시 multi-area Neuropixels와 probability block을 이용해 context-sensitive geometry와 source→target predictive transfer를 실제 mammalian brain에서 검정한다 | exact simultaneous VISp/MOs session, quality units, leave-block-out split와 공식 cache가 없으면 실행하지 않는다. lagged transfer를 causal routing으로 승격하지 않는다 |
| 3 | BA-BIO-LONG | `BLOCKED_INPUT` | 실제 뇌의 구조→기능→행동 사슬에 필요한 same-cell/synapse longitudinal $W$, activity, intervention 자료 경로다 | 직접 구조 측정·독립 metric calibration·held-out dynamics·animal-level intervention이 없으면 열지 않는다 |
| 4 | BA-S1 | `DEFERRED_SYNTHETIC` | 합성 M1 SCC lesion은 실제 자료 경로보다 우선하지 않는다 | 실제자료 분석이 끝나기 전 새 synthetic support threshold나 lesion seed를 만들지 않는다 |

## 오케스트레이터 선택 규칙

1. 활성 후보 1번부터 capability dependency를 검사한다. 앞 후보가 STOP이어도 다음 후보가 독립 mechanism이면 진행할 수 있다.
2. 새 결과는 해당 run의 12/31/40과 artifact hash가 안정된 뒤 이 원장에 한 번만 반영한다.
3. 양성 결과를 찾기 위해 seed·threshold·endpoint를 순회하지 않는다. apparatus 결함은 과학 결과와 분리해 전 seed block을 퇴역시키고 새 contract로만 교체한다.
4. `CONFIRMED_SIMULATOR`는 실제 뇌 증거가 아니다. 생물학적 승격은 BA-BIO-LONG의 독립 입력과 개입 gate가 있어야 한다.

## 현재 인계

**[미완성]** BA-C1은 `STOP`으로 닫혔다. 다음 오케스트레이터는 새 합성 seed 경로를 만들지 않고 BA-EMP-CLOUD의 원자료 적격성부터 판정한다. CloudCell의 neuron identity가 별도 구조 connectome과 검증 가능하게 정렬되지 않으면 구조 라우팅 주장은 삭제하고, 같은 recording 안의 Fisher geometry와 lagged predictive transfer만 독립적으로 보고한다. BA-EMP-IBL은 공식 mammalian replication 입력이 확보된 뒤 연다.
