# Repository reuse map — causal recurrent geometry / connectome memory

Status: COMPLETE

Date: 2026-08-16 (Asia/Seoul)

Scope: `_workspace/ce/agi-connectome-geometric-memory-20260816/00-contract.md`에 고정된 새 연구 가설을 대상으로, 현재 저장소의 SCC, 유한 metric, causal/invariant recovery, memory benchmark 자산을 읽기 전용으로 조사했다. 이 문서는 재사용 지도이지 정본 승격, 제품 통합, 생물학적 동일시 또는 AGI 판정이 아니다. 제품 코드와 정본은 수정하지 않았다.

## 1. 결론

현재 저장소에는 새 연구를 시작하기에 충분한 **부품과 반례**가 있다. 그러나 이미 완성된 하나의 모델은 없다. 가장 안전한 연구 순서는 다음과 같다.

1. 첫 논문/실험은 계약의 축 A, 즉 **합성 환경에서의 causal recurrent geometry 식별 가능성**으로 한정한다.
2. SCC는 새 모델의 학습 대상 또는 압축 단위로 쓰되, `scc_atlas.py`의 유한 maximal-SCC/condensation 의미론을 그대로 재사용한다.
3. 동역학에서 방향 구조를 회복하는 레인은 V4 synthetic bridge의 개입·누수 방지 장치를 재사용하되, V5/V7/V8 실패와 real-data 0/4를 숨기지 않는 새 벤치마크로 다시 연다.
4. metric 레인은 V15의 유한 그래프 경계, V16의 vector-cost NARROW GO, V17의 sign-blind no-go를 모두 제약조건으로 삼는다. metric만으로 방향성, 비가역 전이, 부호 기억을 얻었다고 주장할 수 없다.
5. V9의 첫 task architecture는 STOP이다. 상태 매개와 lesion 효과만 살아남았으며, 새 후보는 level별 timescale, local/shared 분리 readout, 동일 compute 대조군, fresh seed로 다시 구현해야 한다.
6. 현재 checkout에는 정본보다 넓은 infinite-tail 코드/테스트가 있으므로 이를 새 가설의 근거로 쓰지 않는다. 먼저 아래의 **P0 repository reconciliation**을 끝내고 정확한 파일 해시로 characterization한 뒤에만 재사용한다.

## 2. Provenance와 P0 repository reconciliation

### 2.1 현재 checkout의 충돌

현재 `NestedTowerGenerator`에는 uniform infinite-tail 및 rollout-tail certificate 구현이 존재한다(`reality_stone/python/reality_stone/clarus/nested_scc_tower.py:813-890`). 대응 테스트도 certificate 성공과 bound를 직접 검사한다(`tests/test_nested_scc_tower.py:439-483`). 반면 활성 정본은 generic append-zero boundary를 `REFUSED`하고 “truncation bound나 infinite-horizon convergence를 주장하지 않는다”고 제한한다(`docs/7_AGI/28_Nested_Infinite_SCC_V9.md:519-524`). CodeMap도 `certify_prefix`를 “no infinite-tail or truncation certificate”로 기록한다(`docs/7_AGI/18_CodeMap.md:569-574`). 같은 구현 파일 안에서도 controller의 `requires_extension`은 여전히 “not a truncation-error certificate”라고 명시한다(`reality_stone/python/reality_stone/clarus/nested_scc_tower.py:892-903`).

따라서 현 상태는 “새로운 infinite-SCC 정리가 검증됨”이 아니라 **code/test와 정본 사이의 형식 지위 불일치**다. 구현의 주석도 exact direct-limit certificate가 아니라 unit-ball shellwise product에서의 analytic uniform-domain bound라고 한정한다(`reality_stone/python/reality_stone/clarus/nested_scc_tower.py:813-840`). 이 차이를 해소하기 전에는 infinite-tail 결과를 새 CGM 가설의 전제로 사용하지 않는다.

### 2.2 Git provenance 스냅샷

2026-08-16 작업 트리 기준으로 다음 경계가 확인되었다.

| 자산 | checkout 상태 | 재사용 판정 |
|---|---|---|
| `scc_atlas.py`, `tests/test_scc_atlas.py`, `nested_scc_tower.py`, `tests/test_nested_scc_tower.py` | tracked, 해당 파일 local diff 없음 | 유한 SCC 코어는 재사용 가능. infinite-tail 부분은 P0 보류 |
| `docs/7_AGI/28_Nested_Infinite_SCC_V9.md`, `docs/7_AGI/18_CodeMap.md` | tracked, dirty | 현재 줄 내용과 base 정본을 분리해 감사해야 함 |
| `unified_metric.py`, `tests/test_unified_metric.py` | tracked, dirty | V15 당시 STOP 이후 수리된 현 checkout 후보. stable API로 승격 금지 |
| `covariant_metric_flow.py`, `tests/test_covariant_metric_flow.py`, `tests/test_v16_benchmark.py` | untracked | V16의 local predecessor snapshot일 뿐 committed dependency가 아님 |
| `homogeneous_signed_cue.py`, 관련 unit/benchmark tests | untracked | V17의 local predecessor snapshot일 뿐 committed dependency가 아님 |
| `_workspace/ce/_archive/agi-v9-*`, `agi-v15-*`, `agi-v16-*`, `agi-v17-*` | `_archive/`가 현재 untracked relocation | 과거 판정 근거는 보존하되 현재 정본/committed run으로 오인 금지 |

P0 characterization에 사용할 현재 파일 SHA-256은 다음과 같다. 이는 **현 checkout 식별자**이지 정본 승격 표식이 아니다.

| 파일 | SHA-256 |
|---|---|
| `nested_scc_tower.py` | `854673216E5FEACA5FF0E3619DA63B789B15BDFE35994DE539F61C4EAE83A717` |
| `tests/test_nested_scc_tower.py` | `9F15513E3ABCD4C62ABE9C5F06C78255411A0161F92BAA9FFEDC541E82EC9666` |
| `28_Nested_Infinite_SCC_V9.md` | `A9E010E6A82F47E71EEEB4897BC3FB1CCE6FBC300C571668635445B3666696F7` |
| `18_CodeMap.md` | `8CB9B83F850C8EC50CEFABFB29E35E637CCB5D38042711F7910D4E9C2CFBF7D5` |
| `unified_metric.py` / test | `73BBB8C3EF39B56A2C061DBA31470C257A2C2038CCEC0C22689E27A45B1A67CD` / `F0AEE28A5BA5BC7F8EFC53540732189D5D7E114974C5E95D48CCE2E24723CB8C` |
| `covariant_metric_flow.py` / test / V16 benchmark test | `E2AD7AAE4BA5F2B18FF737071AC81C0703E2B2C627DED3068146DBF142422EAC` / `EE6F121E6F2D1C043ABBD53425ADA8E17D27CCC90D649D6629FE2BFB6F3DBA33` / `B48DC39FAA6E569CE60A03AF207D06916832189D8206FD72A0788520DD6CF3D0` |
| `homogeneous_signed_cue.py` / test / V17 benchmark test | `5E8DE35DDA08D238A6F9FBDCF68A5377F18A7DB5B7008F2F50A6C8D56FCB72D3` / `2E438C4B257FB9BFE3DC07EFC56E896E7EE0411EBA4C8055125A51A35C02CD4B` / `0AA81A133341B400DF8CB01CF1A41257ACDA64B4CCE7315E1150C6E5B552A42B` |

현재 untracked relocation에 놓인 판정문도 별도 고정한다. V9 final/validation은 각각 `3D39334ED14613F076477904F38CB632ADC97FC90B9BF33A5BC12A0DE2F33404`, `101473DBD5E6FD84885696EC0406DABE61CA0D98B23F2E040A59A11C3B495CE4`; V15/V16/V17 final report는 각각 `54126F80E742EE207BCA1862E1A5D207644ED80AAEF5401C8DAFD230825E3DBD`, `38E902B21F34DA90ADEF086F736E3E2D8A547BC56FE4D83061CAABE00F64A2F3`, `754B427798535D6DB0732FBCC9D1F3F2A495BDFB149F248DDB4A3627D56166B2`다. 해당 파일의 판정 위치는 `_workspace/ce/_archive/agi-v9-loop-engineering-20260812/40-final-report.md:7-25`, `31-validation.md:8-15`, `_workspace/ce/_archive/agi-v15-unified-metric-score-20260813/40-final-report.md:9-25`, `_workspace/ce/_archive/agi-v16-covariant-metric-flow-20260813/40-final-report.md:7-22`, `_workspace/ce/_archive/agi-v17-metric-delayed-credit-20260813/40-final-report.md:7-27`이다.

P0 완료 조건은 (a) base/working-tree diff와 수학 가정을 분리한 characterization, (b) unit-ball norm·boundary residual·schedule·direct-limit 지위를 명시한 독립 수학 감사, (c) 위 해시를 묶은 재현 테스트, (d) 정본과 CodeMap 중 어느 문구가 활성 지위인지 별도 closure gate로 결정하는 것이다. 이 문서는 그 결정을 대신하지 않는다.

## 3. SCC 자산 지도

### 3.1 유한 SCC 코어 — 즉시 재사용 가능

`reality_stone/python/reality_stone/clarus/scc_atlas.py`는 생물학적 whole-brain parcellation이 아니라 선언된 유한 directed graph용 기반임을 먼저 제한한다(`:1-11`). 핵심 재사용 표면은 다음과 같다.

| 기능 | 구현 | 검증 | 새 연구에서의 용도 |
|---|---|---|---|
| SCC/condensation 자료형 | `scc_atlas.py:30-39` | `tests/test_scc_atlas.py:68-87` | recovered graph의 maximal SCC와 quotient DAG |
| threshold filtration | `scc_atlas.py:43-68,328-412` | `tests/test_scc_atlas.py:148-211` | scale/threshold 안정성 검사. 같은 고정 그래프의 “중첩 maximal SCC” 주장 대체 |
| deterministic maximal SCC와 DAG | `scc_atlas.py:233-326` | exhaustive n≤4 비교 `tests/test_scc_atlas.py:68-87`; edge-addition coarsening `:97-129` | graph seed별 구조 평가 및 permutation-invariant summary |
| forward time unroll | `scc_atlas.py:90-99,414-470` | `tests/test_scc_atlas.py:214-231` | recurrence graph와 event-time DAG를 구분하는 negative control |
| ARCH-1 검증/구성 | `scc_atlas.py:103-126,504-597` | `tests/test_scc_atlas.py:234-265` | 명시적 block architecture fixture |
| block gain와 유한 error bound | `scc_atlas.py:127-173,598-753` | `tests/test_scc_atlas.py:268-355` | SCC quotient 상의 finite rollout bound와 encoder/decoder error budget |

반드시 보존할 의미론은 `M[target, source]` 방향, maximal SCC, acyclic condensation, `q<1` fail-closed다(`scc_atlas.py:598-699`). topology와 dynamics가 다름을 테스트가 명시한다(`tests/test_scc_atlas.py:132-146`). 따라서 SCC 존재만으로 attractor, 안정성, 기억, 의식을 추론할 수 없다.

### 3.2 nested tower — 유한 구조만 재사용, task 후보는 재설계

`nested_scc_tower.py`에서 바로 재사용 가능한 부분은 typed specification/certificate(`:73-242`), 독립 유한 SCC 구성(`:243-282`), deterministic finite nested prefix와 audit(`:284-463`), finite causal cone와 forward unroll(`:477-541`), schedule별 normalization/step/contraction(`:567-744`), append-zero incompatibility witness(`:746-811`)다. 테스트는 proper nesting과 “lower level은 동일 고정 그래프의 두 번째 maximal SCC가 아님”을 확인하고(`tests/test_nested_scc_tower.py:82-113`), causal cone/event unroll(`:218-252`)과 compatibility/contraction/topology 경계(`:254-324`)를 검사한다.

controller는 grow-only finite mechanism이며 event/token/intervention/snapshot을 typed하게 묶는다(`reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py:71-242`). depth 결정과 exhaustion은 `:385-419`, atomic observe/update/token은 `:567-663`, token validation과 state-only forecast/policy는 `:665-723`, snapshot lifecycle은 `:727-855`다. stale/foreign token fail-closed와 real intervention, snapshot integrity는 `tests/test_adaptive_scc_tower_controller.py:122-210,343-615,620-881`에서 검증된다.

runtime 연결은 opt-in만 허용된다. 기본값은 `nested_scc_enabled=False`이며 belief control과 상호 배타적이다(`reality_stone/python/reality_stone/clarus/agent.py:259-285,414-430`). action은 발급된 state token 경계를 통과해야 한다(`agent.py:493-531`); 기존 경로 보존과 fail-closed는 `tests/test_agent.py:248-333`에서 검사된다.

### 3.3 고정 그래프 hierarchy의 수학 경계

고정된 한 그래프에서 maximal SCC들은 서로소이므로 진정한 strict nested chain이 될 수 없고, forward event-unroll은 DAG여서 nontrivial recurrent SCC가 사라진다(`docs/7_AGI/28_Nested_Infinite_SCC_V9.md:22-29,119-143`). condensation DAG에 SCC를 다시 적용해도 같은 의미의 다층 hierarchy가 생기지 않는다. 새 가설의 “iterated SCC hierarchy”는 반드시 **threshold, 시간창, graph semantics, observation scale 중 무엇이 변하는지** 선언해야 한다. 그 선언 없이 같은 그래프에 SCC를 반복 적용하는 구현은 no-go fixture로 분류한다.

## 4. V9 STOP 지도

V9의 수학적 finite SCC/causal-state 장치는 살아 있지만 첫 task architecture의 등록 판정은 STOP이다.

- 개발 실행에서 V9 accuracy는 `0.3457`, matched monolithic은 `0.6116`, paired improvement는 `-0.2659`이고 95% bootstrap interval은 `[-0.2788,-0.2524]`였다(`docs/7_AGI/28_Nested_Infinite_SCC_V9.md:657-664`; `_workspace/ce/_archive/agi-v9-loop-engineering-20260812/31-validation.md:8-15`).
- upper-reset loss와 cross-cut loss는 각각 약 `0.0635`로 좁은 causal contribution gate만 통과했다(`28_Nested_Infinite_SCC_V9.md:660-668`). 이는 cross-level state가 출력을 바꾼다는 증거이지 task advantage가 아니다.
- confirmation seed `10000..10255`는 열리지 않았다(`28_Nested_Infinite_SCC_V9.md:662-670`; `_workspace/ce/_archive/agi-v9-loop-engineering-20260812/40-final-report.md:7-25`).
- 사후 진단은 모든 level이 같은 약한 recurrence를 공유하고 readout이 지연된 약한 복사본을 평균했다는 것이다(`28_Nested_Infinite_SCC_V9.md:672-675`).
- 재설계 요구는 local temporal state, typed transition kernel, full 대 local/cloud의 matched compute 비교, 양팔 level별 timescale, local/shared increment 분리, fresh seeds다(`28_Nested_Infinite_SCC_V9.md:677-686`).

`nested_scc_memory_benchmark.py`는 episode/config/result(`reality_stone/python/reality_stone/clarus/nested_scc_memory_benchmark.py:63-140`), evaluation(`:141-258`), seed aggregate와 gate(`:265-327`), hash-bound preregistration(`:335-380`), development/confirmation sealing(`:383-434`)을 제공한다. 그러나 파일 자체가 locked V9 synthetic benchmark임을 밝힌다(`:1-5`). 따라서 **manifest/receipt/one-shot locking 패턴만 재사용**하고 candidate architecture, 개발/confirmation seed block, 과거 점수는 새 연구에 재사용하지 않는다.

## 5. Metric 자산 지도

### 5.1 V15 — finite metric boundary와 STOP

V15 score run은 ordinary tests와 수학 fixture가 통과했어도 finite core와 AGI에 STOP을 내렸다(`_workspace/ce/_archive/agi-v15-unified-metric-score-20260813/40-final-report.md:9-25`). 이유는 다음과 같다.

- projection 기반 chart 처리는 affine covariance가 아니었다.
- static symmetric metric direction alone 및 symmetric source-free goal은 방향/비가역 dynamics를 식별하지 못한다.
- finite endpoint consistency는 continuum Riemannian manifold를 식별하지 못한다(`40-final-report.md:38-44`).
- tiny positive edge에서 Dijkstra predecessor cycle이 발생해 adversarial numeric gate가 0/8이었다(`40-final-report.md:54-64`).
- metric은 oracle/privileged input이었고 관측에서 학습되지 않았다(`40-final-report.md:68-81`).

현재 dirty `unified_metric.py`는 이 중 일부 구현 결함을 수리한 checkout 후보다. 정확히 하나의 `metric` state field를 둔다(`reality_stone/python/reality_stone/clarus/unified_metric.py:225-247`), projection 없는 affine chart transport를 분리한다(`:303-367`), finite graph core/state를 정의한다(`:370-445`), 외부 source update와 tensor deformation memory readout을 제공한다(`:509-548`), local/edge length와 repaired Dijkstra를 구현한다(`:550-698`), dimensionless surprise와 tie-preserving targets를 제공한다(`:700-795`). certificate는 continuum, irreversible dynamics, AGI, biology, cosmology를 모두 false로 유지한다(`:797-830`; tests `tests/test_unified_metric.py:463-480`).

테스트 자산은 affine covariance와 external source update(`tests/test_unified_metric.py:108-179`), 한 metric의 다중 readout(`:182-217`), ties/tiny edge/cycle guard(`:220-285`), dimensionless/extreme input rejection(`:289-437`)이다. 다만 source/test가 dirty이므로 P0 hash-bound characterization 전에는 stable core로 의존하지 않는다.

새 가설과의 직접 경계는 명확하다. 이 구현의 “memory”는 외부 source에 의해 바뀐 두 finite tensor 사이의 deformation readout이지, trajectory에서 학습된 Riemannian memory trace가 아니다. 또한 symmetric metric으로 directed `F_theta`, time arrow, control, source 또는 goal을 대체할 수 없다. 그 항들은 모델에 별도 typed object로 남겨야 한다.

### 5.2 V16 — NARROW GO

V16은 **nonzero vector observation + positive scalar cost**의 finite online metric flow에만 NARROW GO다(`_workspace/ce/_archive/agi-v16-covariant-metric-flow-20260813/40-final-report.md:7-22`). benchmark 결과는 accuracy `0.9642334`, regret `0.000439384`, metric error `0.0339121`, chart action rate `1.0`이었다(`40-final-report.md:71-81`). 이는 raw perception, delayed credit, continuum geometry 또는 AGI 증거가 아니다(`40-final-report.md:105-118`).

현재 untracked `covariant_metric_flow.py`의 재사용 가능한 표면은 factor-only state(`reality_stone/python/reality_stone/clarus/covariant_metric_flow.py:195-219`), certificate(`:233-257`), SPD factor construction(`:300-323`), predict/residual(`:388-413`), rank-one update(`:441-493`), route/tie choice(`:495-536`)다. full identifiability without excitation span, noisy fixed-rate convergence, raw/delayed/continuum/AGI를 false로 둔다(`:544-568`). unit tests는 one-factor contract(`tests/test_covariant_metric_flow.py:44-55`), SPD/covariance/contraction/natural-gradient behavior(`:67-149`), extremes와 route/snapshot(`:158-235`), false broad claims(`:238-250`)을 제공한다. benchmark manifest/receipt locking은 `tests/test_v16_benchmark.py:95-188`에 있다.

재사용은 single-metric baseline adapter로 한정한다. 관측→tangent/cost encoder, semantic OOD, delayed memory는 새 구현 대상이다. untracked source/test/archive이므로 위 SHA-256을 고정한 characterization 후에만 predecessor로 인용한다.

### 5.3 V17 — sign-blind no-go와 좁은 escape

V17은 original-space의 strict metric-only family가 full `GL(d)` covariance 아래 signed cue를 구분할 수 없음을 닫았다. strict arm은 0.5/0.5이고 finite 또는 countable SCC 복제로도 lost odd information이 복구되지 않는다(`_workspace/ce/_archive/agi-v17-metric-delayed-credit-20260813/40-final-report.md:7-27,31-53,105-125`). 한 비트의 odd/covector 정보를 추가하면 conditional separation은 가능하지만, homogeneous lift는 anchor/covector/scalar와 `d+1` coordinate를 추가하므로 strict metric-only가 아니다(`40-final-report.md:55-91`). 판정은 homogeneous lift NARROW GO, AGI STOP이다.

현재 untracked `homogeneous_signed_cue.py`는 narrow one-cue memory임을 제한한다(`reality_stone/python/reality_stone/clarus/homogeneous_signed_cue.py:1-12`), state/readout/certificate(`:79-124`), lift/write(`:126-218`), cost/readout/snapshot(`:223-265`), strict sign-blind control(`:271-337`)을 제공한다. tests는 chart covariance(`tests/test_homogeneous_signed_cue.py:34-109`), factor/strict sign-even serialization(`:111-155`), added-coordinate deletion killing test(`:158-180`), narrow certificate flags(`:217-239`)를 제공한다. benchmark locking은 `tests/test_v17_benchmark.py:82-420`에 있다.

따라서 새 geometric memory는 signed/directional trace가 필요하면 odd covector, oriented transition, anchor, eligibility/history state 중 하나를 **명시적으로 비용에 포함**해야 한다. metric mixture나 SCC 수를 늘리는 것만으로 sign blindness를 해결했다고 주장해서는 안 된다.

## 6. Causal / invariant recovery 자산 지도

### 6.1 graph dynamics — leakage-resistant하지만 anatomical recovery 아님

`graph_dynamics.py`는 effective predictive graph이며 anatomical connectome이 아님을 명시한다(`reality_stone/python/reality_stone/clarus/graph_dynamics.py:1-17`). directed/diffusion graph 구성은 `:314-399`, context regime와 dynamic features는 `:403-500`, chronological evaluation은 `:529-860`, artifact claim boundary는 `:870-940`이다. synthetic direction/context와 train-graph reuse는 `tests/test_graph_dynamics.py:82-178`, test block이 graph를 바꾸지 못하는 leakage lock은 `:178` 이후에 있다.

활성 문서는 synthetic PASS와 real AML310 exploratory FAIL 0/4를 함께 기록한다(`docs/7_AGI/23_Graph_Dynamics_Loop.md:1-3,67-98`). 그러므로 real neural activity에서 anatomical 또는 causal connectome을 이미 회복했다고 인용할 수 없다. 새 연구에서는 observation-derived effective graph와 anatomical prior/ground truth를 별도 필드와 별도 metric으로 유지한다.

### 6.2 sparse/latent causal bridge — V4 좁은 synthetic 선행 결과

`sparse_causal_bridge.py`는 paired interventions가 있는 finite sparse causal fixture다(`reality_stone/python/reality_stone/clarus/sparse_causal_bridge.py:1-4`). synthetic environment(`:148-220`), common-noise paired intervention과 permutation(`:220-310`), finite graph geometry proposal(`:314-351`), observational/interventional estimate와 causal selection(`:369-532`), model/lesion evaluation(`:561-668`), locked gate(`:706-1040`)가 재사용 가능하다. tests는 orientation, lower-bound proposal, paired `do`, confounder, truth-free selector, controls를 검사한다(`tests/test_sparse_causal_bridge.py:36-188`)고 evaluation leakage를 막는다(`:206-259`).

`latent_causal_bridge.py`는 residual filter/full mechanism/pool AR와 sequential prediction(`reality_stone/python/reality_stone/clarus/latent_causal_bridge.py:18-159`), latent evaluation(`:199-356`), locked gate(`:356-688`)를 제공한다. fresh seeds, rank-one filter, no-current-outcome, calibration boundary, V4 pass는 `tests/test_latent_causal_bridge.py:30-177`에 있다. V4 artifact는 narrow synthetic exact recovery를 기록하지만(`artifacts/agi/sparse_causal_bridge_validation_v4.json:6-7,505-506`), 그것이 일반 recurrent geometry identifiability는 아니다.

후속 lineage는 성공하지 않았다. V5 free rollout validation은 false이고, V7 closure는 persistence/stability 실패와 unopened test를 명시한다(`docs/7_AGI/26_Sparse_Causal_Bridge_V7_Closure.md:75-92`); V8도 validation false다. 따라서 새 Paper A는 V4의 generator/intervention/integrity 장치를 재사용하되, 더 넓은 graph/environment blind split과 nonlinear/context families를 가진 **새 claim**이어야 한다. V4를 이름만 바꿔 재출판하거나 V5/V7/V8 실패를 누락해서는 안 된다.

추가 integrity pattern으로 `free_rollout_bridge.py`, `reliability_rollout_bridge.py`, `parent_anchored_rollout_bridge.py`, `integrated_latent_state_bridge.py`와 대응 tests가 있다. 특히 no-future/no-hidden API, prelock, finite stability, unopened-test 패턴은 재사용 가치가 있다(`tests/test_free_rollout_bridge.py:54-192`; `tests/test_reliability_rollout_bridge.py:49-182`; `tests/test_parent_anchored_rollout_bridge.py:50-152`; `tests/test_integrated_latent_state_bridge.py:36-139`). 이들은 성공한 candidate lineage가 아니라 실패를 보존하는 evaluation harness다.

## 7. Memory benchmark 자산 지도

| 자산 | 재사용 가능한 것 | 경계 |
|---|---|---|
| episodic memory | audited ADD/UPDATE/DELETE/NOOP와 abstaining recall (`reality_stone/python/reality_stone/clarus/episodic_memory.py:1-120`); FIFO, merge-off, abstention-off, existing, no-memory baselines와 hard gate (`episodic_memory_benchmark.py:35-181`) | bounded key/value interference benchmark이며 geometric trajectory regeneration이 아님 |
| local temporal memory | same-unit lag score와 chronological 60/20/20 embargo evaluation (`local_memory.py:1-5,30-180,225-399`); current-only/null controls (`:185-223`) | graph memory가 아니며 biological interpretation은 GFP-only control/headroom correction으로 기각됨 (`docs/7_AGI/25_Local_Temporal_Memory_Confirmation.md:4-14,176-186`) |
| brain geometry benchmark | context-switch trial generator(`brain_geometry_benchmark.py:57-77`), diffusion vs attractor arms(`:80-133`), paired LCB/gate(`:257-368`) | synthetic MD-like attractor benchmark이며 learned Riemannian metric, biology, AGI가 아님 (`:1-5`) |
| nested SCC memory | locked config/evaluation/gate/manifest/receipt (`nested_scc_memory_benchmark.py:63-434`) | V9 candidate와 seed는 STOP이므로 새 연구에서 재사용 금지 |
| delayed linear credit | eligibility/hard latch, homogeneous candidate, strict metric control (`reality_stone/python/reality_stone/clarus/delayed_linear_credit.py:219-848`) | source/test와 V18b run이 untracked/incomplete이고 externally given marker를 쓰므로 characterization 전 predecessor로만 취급 (`reality_stone/python/reality_stone/clarus/delayed_linear_credit.py:1-9`; `tests/test_delayed_linear_credit.py:443-464`) |

새 geometric-memory benchmark의 최소 baseline은 Euclidean latent state, activity-only recurrent memory, V16 single metric, ordinary episodic memory, 그리고 parameter/FLOP-matched context metric mixture다. 각 arm은 같은 encoder information과 같은 train/validation/test split을 받아야 한다. “memory” primary endpoint는 단순 다음-step accuracy 하나가 아니라 delayed trajectory regeneration 또는 controlled long-horizon retrieval처럼 사전에 하나로 고정해야 한다.

## 8. Brain/connectome 자산과 데이터 경계

`brain_scc_study.py`는 실제 생물 데이터를 읽는 구현이 아니라 typed future study design이다(`reality_stone/python/reality_stone/clarus/brain_scc_study.py:1-6`). 각 scale graph에 node/edge/direction semantics를 요구하고(`:15-49`), cross-scale map 의미론을 별도로 선언한다(`:52-73`), scale별 SCC image compatibility와 incomplete mapping을 감사하면서 fixed-graph nested maximal 및 biological identity를 false로 둔다(`:95-167`). compatible fine→coarse와 split/incomplete/reverse fail-closed tests는 `tests/test_brain_scc_study.py:10-75`다. 이것이 새 축 B의 schema로 가장 적합하다.

기존 뇌 문서는 binary adjacency보다 weighted/effective connectivity를 우선하고(`docs/6_뇌/09_생명에서지능까지/01_개요와공통식.md:195,238-264`), memory trace는 아직 문맥적 항으로만 둔다(`:318`). C. elegans는 weighted connectome과 recurrence를 다루지만 trial-behavior 연결에 데이터 경계가 있다(`02_c_elegans.md:30-65,263-265`). FlyWire는 structural closure이며 time-aligned dynamics가 없다(`03_drosophila.md:121-244`). current assessment는 cross-species action carrier와 dynamic link가 미완성이고 readiness가 0/5임을 기록한다(`06_현재판정과다음병목.md:56-86,408-418`). V9 정본도 FlyWire/C. elegans를 finite recurrence evidence 이상으로 올리지 않는다(`docs/7_AGI/28_Nested_Infinite_SCC_V9.md:438-463`).

현재 저장소에는 `MICrONS` 파일 또는 텍스트 참조가 없다(`rg --files` 및 repository-wide `rg -i MICrONS` 기준 0건). 따라서 사용자 노트의 MICrONS 분석은 새 데이터 lane이다. 실행 전에 source URL/version, license, checksum, cell/edge/direction semantics, activity alignment, inclusion/exclusion, split unit, anatomical-vs-effective label을 가진 manifest를 만들어야 한다. 데이터가 없는 상태에서 MICrONS 결과를 예측하거나 기존 FlyWire 결과로 대체할 수 없다.

## 9. 새 가설과 기존 자산의 중복·충돌

| 새 가설 요소 | 기존 중복 | 충돌/살아남는 새 연구 질문 |
|---|---|---|
| CGM-D1: SCC condensation은 DAG | 이미 구현·exhaustive-tested theorem (`scc_atlas.py:233-326`; `tests/test_scc_atlas.py:68-87`) | 새 empirical claim으로 승격하지 말고 정의/도구로 사용 |
| dynamics → directed structure recovery | V4 narrow synthetic exact recovery와 paired interventions | V5/V7/V8 rollout/closure 실패, real graph dynamics 0/4. broader identifiability under graph/environment OOD가 새 질문 |
| iterated SCC hierarchy | finite nested prefixes와 cross-scale study schema 존재 | 같은 고정 graph의 maximal SCC 반복은 no-go. scale/threshold/window semantics와 maps를 새로 선언해야 함 |
| SCC가 recurrent attractor/memory substrate | V9 finite causal state와 lesion contribution | topology≠stability; V9 task utility STOP. dynamics, contraction, task advantage를 별도 gate로 검증해야 함 |
| Riemannian metric as memory | V15 finite tensor deformation, V16 learned finite vector-cost metric | V15은 oracle/external-source이고 continuum 아님; V16은 raw/delayed memory 아님. observation encoder와 long-horizon memory가 새 구현 |
| metric drives transition/goal | V15 one-metric readouts | static symmetric metric의 direction/irreversibility/source-free goal은 no-go. directed drift/control/source를 별도 객체로 유지 |
| context metric mixture | 저장소에 안정된 mixture learner 없음 | 진짜 신규 표면. rank/parameter/FLOP matching, permutation/gauge audit가 필요하며 V16 one-factor state를 몰래 변형하면 안 됨 |
| signed/directional trace | V17 strict no-go와 homogeneous escape | metric/SCC 복제만으로 odd information 복구 불가. 추가 covector/anchor/history 비용을 공개해야 함 |
| connectome structure↔activity | brain scale schema, FlyWire/C. elegans 문서 | anatomy와 effective causal graph는 별개; time alignment와 interventions 없이는 causal recovery 주장 금지 |
| consciousness/AGI | 일부 문서의 장기 동기 | 계약상 제외. SCC=consciousness, one metric generates everything, 구현 성공=AGI를 주장하지 않음 |

## 10. 최소 후속 구현 표면

제품 또는 기존 정본을 건드리지 않고, 승인된 다음 run에서 새 isolated research surface만 추가하는 것이 최소다.

### P0. Repository reconciliation — Paper A 전에 필수

1. 위 SHA-256 snapshot으로 nested tail, V15/V16/V17 파일을 묶은 read-only characterization manifest를 만든다.
2. `nested_scc_tower.py:813-890`의 norm/domain/schedule/boundary assumptions와 `28_Nested_Infinite_SCC_V9.md:519-524`의 더 좁은 지위를 독립 감사한다.
3. dirty/untracked 파일을 committed API처럼 import하지 않는다. 후속 benchmark는 snapshot hash가 일치할 때만 optional adapter를 연다.
4. 정본 변경은 이 inventory 범위 밖의 별도 closure decision으로 남긴다.

### P1. 첫 연구 구현 — causal recurrent geometry benchmark

- `reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py`
  - linear VAR, nonlinear recurrent, context-switch generator
  - exact `G`, `F`, `z`, `g` truth와 full/partial observation
  - paired single-node/module interventions
  - graph seed와 environment seed를 모두 blind split
  - candidate adapter, SCC quotient, matched baselines, compute accounting
- `experiments/preregistration/causal_recurrent_geometry_v1.json`
  - fresh development/confirmation seed roles
  - primary endpoint 하나, alpha/CI/STOP rule
  - graph/environment OOD, parameter/FLOP budget
  - V9/V16/V17 sealed seed 재사용 금지
- `examples/agi/causal_recurrent_geometry_run.py`
  - one-shot manifest/receipt runner; 기존 V9/V16/V17 locking pattern만 재사용
- `tests/test_causal_recurrent_geometry_benchmark.py`
  - observational-equivalence no-go와 intervention separation
  - no-future/no-hidden API
  - `M[target,source]` orientation
  - exact SCC/condensation 및 fixed-graph hierarchy refusal
  - direction/SCC-label/geometry shuffle ablations
  - parameter/FLOP equality와 test mutation lock

이 네 표면이면 축 A를 검증하는 데 충분하다. MICrONS ingest, context metric mixture, runtime agent integration은 Paper A의 선행 조건이 아니다.

### P2. Paper A가 통과한 뒤의 geometric memory

별도 `context_metric_mixture.py`와 별도 memory benchmark를 추가한다. V16은 hash-bound single-metric baseline adapter로만 감싼다. Euclidean/activity/episodic/strict-metric/homogeneous controls를 함께 두고, V17 no-go를 깨는 odd state가 있다면 좌표 수와 parameter/FLOP 비용을 공개한다. 이 단계도 기존 `UnifiedMetricState.metric` 또는 `CovariantMetricState.factor` contract를 수정하지 않는다.

## 11. 변경하면 안 되는 compatibility 경계

1. `scc_atlas.py`의 maximal SCC, condensation DAG, threshold fail-closed, `M[target,source]`, `q<1` 의미론을 바꾸지 않는다.
2. 고정 graph의 SCC 반복을 nested hierarchy로 이름 바꾸지 않는다. cross-scale map이 없으면 hierarchy claim을 거부한다.
3. V9 locked preregistration, score, STOP, unopened confirmation, seed blocks를 수정·재사용하지 않는다.
4. `nested_scc_enabled=False`, belief-control 상호 배타, issued-token-only runtime 경계를 유지한다(`agent.py:259-285,493-531`).
5. dirty `UnifiedMetricState.metric`의 one-field contract와 false continuum/AGI certificate를 넓히지 않는다(`unified_metric.py:225-247,797-830`).
6. untracked `CovariantMetricState.factor`의 one-factor/no hidden optimizer contract를 변경해 mixture를 숨기지 않는다(`covariant_metric_flow.py:195-219`; `tests/test_covariant_metric_flow.py:44-55`).
7. homogeneous lift를 strict original-space metric-only로 보고하지 않는다. extra coordinate/anchor/covector 비용과 deletion test를 유지한다.
8. `graph_dynamics`의 “effective predictive, not anatomical” 라벨과 chronological split을 보존한다.
9. V4–V8 validation artifacts, unopened-test locks, 실패 판정을 덮어쓰지 않는다.
10. local-memory의 biological no-go와 GFP-only negative control을 새 memory 근거로 뒤집지 않는다.
11. untracked V16/V17/V18b 파일은 exact hash manifest 없이 dependency나 정본으로 취급하지 않는다.
12. P0를 닫기 전 현재 infinite-tail code/test를 정본의 증명 또는 새 가설의 근거로 승격하지 않는다.
13. anatomy, effective connectivity, causal graph, learned metric, task memory, consciousness를 서로 다른 typed status로 유지한다.
14. 이 연구 run에서는 제품 코드, runtime default, 활성 정본을 변경하지 않는다.

## 12. 최종 재사용 판정

- **즉시 재사용:** clean finite `scc_atlas` primitives/tests, `brain_scc_study` schema/tests, sparse/latent bridge의 generator·paired intervention·leakage guard, episodic/local/brain-geometry baseline harness.
- **패턴만 재사용:** V9/V16/V17 manifest·receipt·confirmation locks, free/reliability/anchored rollout의 no-future/no-hidden tests.
- **P0 hash-bound characterization 후 재사용:** current infinite-tail candidate, dirty V15 core/test, untracked V16/V17 source/tests/archive.
- **재사용 금지:** V9 candidate architecture와 seed blocks, failed V5/V7/V8 lineage를 성공 모델로 사용하는 것, strict metric-only signed memory 주장, same-graph repeated-SCC hierarchy, finite metric의 continuum/biology/AGI 승격.

이 경계 안에서는 사용자 가설을 충분히 독립 연구주제로 전개할 수 있다. 첫 유효 산출물은 “AGI가 증명되었다”가 아니라, fresh blind graph/environment split에서 intervention이 어떤 관측 동치류를 깨고 directed recurrent structure와 finite geometry를 어느 범위까지 식별하는지에 대한 명시적 정리·반례·benchmark 판정이어야 한다.
