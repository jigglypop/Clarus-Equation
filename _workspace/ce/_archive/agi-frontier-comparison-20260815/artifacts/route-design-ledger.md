# CE-AGI CR3 route design ledger

작성일·접근일: 2026-08-15  
범위: 현재 작업 트리의 비-RBE AGI 문서·코드와 공개 benchmark의 공식 자료  
독립성: `10-sources.md`, `11-math.md`, `20-audit.md` 등 다른 lane 산출물은 읽지 않았다.

## 1. 계산 규약

- `d_tune`: 공통 backbone·optimizer·데이터 처리 선택을 제외하고, CE 경로 때문에 새로 생기는 독립 조정 선택 수다.
- `L`: 사전등록 전 열어 볼 CE 전용 grid의 크기다. 후보별로 $L=\prod_j |\Theta_j|$로 센다.
- `target-aware`: 후보를 고정하기 전에 같은 내부 실패 또는 외부 benchmark의 목표·결과를 보았는지를 내부/외부로 나눠 적는다.
- 아래 `d_tune`과 `L`은 증분 자유도만 센다. 실제 연구에서는 backbone, optimizer, seed, prompt, decoder 선택의 전체 탐색 횟수도 별도로 원장에 더해야 한다.
- 네 경로의 CE 전용 grid 기회 수는 $6+4+4+2=16$이다. 서로 다른 축의 결과를 하나의 “AGI score”로 합치거나 16개 중 최고값만 보고하면 안 된다.
- 모든 효과 판정은 paired seed/task 차이의 95% interval을 사용한다. 하한이 0보다 크지 않으면 우위 주장은 죽는다. 정확한 검정법과 multiplicity correction은 구현 전 등록한다.

## 2. 현재 내부 출발점

| 축 | 보존 가능한 좁은 사실 | 경로가 넘어서야 할 음성 결과·경계 |
|---|---|---|
| nested SCC | finite prefix, 수축 certificate, immutable state token, 실제 reset/cut intervention이 구현됨 (`nested_scc_tower.py:1-11`, `adaptive_scc_tower_controller.py:281-300`, `:488-643`) | V9는 256-seed 개발에서 0.3457 대 matched monolithic 0.6116, paired improvement -0.2659로 STOP. cross-level lesion 효과만 생존 (`28_Nested_Infinite_SCC_V9.md:649-670`). 현재 depth 선택은 residual/난이도가 아니라 경계 compatibility가 없으면 매 tick 한 층씩 증가한다 (`adaptive_scc_tower_controller.py:385-418`). |
| local/continual | V10 local/shared transition은 좁은 합성 과제에서 자체 factorial control보다 우수했고 실제 lesion 효과가 있었다 (`29_Local_Cloud_Kernel_V10.md:43-75`). learnable small-gain operator와 replay/guard API가 구현됨 (`learnable_small_gain_local_cloud.py:15-68,80-215`; `sleep.py:55-77,1139-1285`). | V11에서 compute-matched Elman-3를 포함한 모든 강한 recurrent 비교에 패했고 10/14 gate가 실패했다 (`30_Strong_Recurrent_OOD_V11.md:8-44`). 총론은 V12-V13c도 ABANDONED/STOP으로 기록한다 (`1_AGI.md:16-19`). STDP·sleep의 continual 효능은 확립되지 않았다 (`1_AGI.md:7`). |
| causal/OOD/agent | sparse bridge는 paired one-step intervention으로 방향을 정하는 구현이며 (`sparse_causal_bridge.py:1-6,445-520`), `BeliefController`는 committed action 열만 인과 transition에서 갱신한다 (`belief_control.py:140-205`). | V7 locked test는 열리지 않아 test 일반화 증거가 0이다 (`26_Sparse_Causal_Bridge_V7_Closure.md:80-90`). bridge 구현은 정확히 네 chart를 강제하므로 외부 환경 adapter가 필요하다 (`sparse_causal_bridge.py:113-138`). |
| metacognition/self-correction | agent loop, critic, bounded working memory와 text loop의 실행 surface가 있다 (`agent.py:364-588,683-721`). | `metacognition_step`은 답을 재검증·수정하지 않고 입력 deviation에 $\rho$를 반복 곱한다 (`agent.py:176-215`). `TextEnvironment`는 hash vector와 템플릿 응답 fixture다 (`agent.py:608-672`). 따라서 현재 구현은 외부 self-correction evidence가 아니다. |

## 3. R1 — residual-gated nested SCC for recurrent/latent computation

### 목표

동일 backbone·parameter 수에서 item별 recurrent depth를 바꾸는 nested-SCC adapter가 fixed-depth 및 compute-matched recurrent baseline보다 정확도–compute Pareto와 length/difficulty OOD를 개선하는지 검증한다. 값의 방향은 `accuracy 증가`, `평균 반복·비용 감소 또는 동일`, 층위는 `public reasoning benchmark`, 정의역은 동일 tokenizer/backbone과 동일 train data다.

### 새 공리 1개

**[공리: residual halting]** target을 읽지 않는 정규화 prediction residual $r_t$만이 다음 recurrent level 실행을 허가하며, $r_t\le\tau$이면 halt, 아니면 $D_{max}$까지 정확히 한 level만 추가한다. 기존의 compatibility-driven grow-every-tick 규칙은 benchmark adapter 안에서 사용하지 않는다.

### 조정 자유도와 target-awareness

- $D_{max}\in\{4,8\}$, $\tau\in\{0.02,0.05,0.10\}$: `d_tune=2`, `L=6`.
- injection 위치, normalizer, shell width, readout은 한 번 고정하고 sweep하지 않는다.
- 내부 target-aware: `예` — V9 STOP과 사후 진단을 본 뒤 설계했다.
- 외부 target-aware: `예` — ARC-AGI-2의 공개 목표·score page와 BABILong 공개 결과가 보이는 공식 페이지를 열람했다. 따라서 public set은 development로만 쓰고 semi-private/private 또는 새 봉인 seed를 최종 판정에 써야 한다.

### benchmark·baseline

1. 저비용 killing stage: BABILong의 짧은 길이에서 학습하고 더 긴 context와 지원 사실 수에서 평가한다.
2. 생존 시 CR3 stage: ARC-AGI-2 public eval을 development로, 공식 semi-private/private evaluation을 최종 판정으로 사용하고 pass@2와 task당 compute/cost를 함께 보고한다.
3. baseline: one-pass backbone, fixed depth 4/8, maximum-depth, token/compute-matched self-consistency, parameter-matched recurrent/ACT-style adapter, 기존 V9 readout diagnostic. 모든 baseline은 같은 backbone·data·search budget을 받는다.
4. ablation: cross-scale cut, upper reset, residual shuffle, depth decision replay, fixed-depth replacement.

### 교차 예측

- 독립 difficulty가 높은 item에서만 active depth가 증가하고, 쉬운 item은 일찍 halt한다.
- cross-scale cut/reset의 손실은 짧은 단일-fact보다 긴 compositional item에서 더 커야 한다.
- residual 감소와 다음-step 정답 개선은 held-out item 안에서 양의 연관을 가져야 한다.
- 개선이 있다면 accuracy 하나가 아니라 accuracy–FLOP/cost Pareto에서도 fixed-depth envelope 밖에 있어야 한다.

### 죽이는 반증

- strongest compute-matched baseline 대비 paired 정확도 차이의 95% 하한이 0 이하.
- 선택 depth가 거의 항상 1 또는 $D_{max}$로 붕괴하거나 독립 difficulty와 무관.
- cut/reset가 상태 tensor를 바꾸어도 held-out 성능 손실의 95% 하한이 0 이하.
- 같은 정확도에서 평균 compute가 fixed-depth보다 낮지 않거나, 같은 compute에서 정확도가 높지 않음.

### 최소 구현

`scc_latent_adapter.py`, benchmark-neutral residual/halting trace schema, BABILong/ARC adapter 각 1개, 동일-budget evaluator와 preregistration manifest. 기존 tower generator·intervention 코드는 재사용하되 기존 V9 점수나 confirmation seed는 재사용하지 않는다.

## 4. R2 — frozen-core contractive plastic shell for continual/local learning

### 목표

frozen encoder 위의 작은 contractive local/cloud adapter만 순차 학습하고 guarded replay로 commit할 때, matched-memory replay·LoRA continual baseline보다 평균 성능과 forgetting을 개선하는지 검증한다. 방향은 `average accuracy·forward transfer 증가`, `forgetting·memory·update 비용 감소`, 정의역은 동일 task order와 동일 frozen backbone이다.

### 새 공리 1개

**[공리: frozen-core plastic shell]** 순차 stream 중 base encoder는 불변이고, small-gain certificate가 있는 adapter/head만 변경할 수 있으며, 독립 guard set의 과거-task 성능이 사전 허용폭을 넘게 떨어지는 update는 원자적으로 폐기한다.

### 조정 자유도와 target-awareness

- replay capacity $M\in\{16,64\}$, update/sleep interval $S\in\{4,16\}$: `d_tune=2`, `L=4`.
- STDP-only, replay-only, guard-off는 tuning 후보가 아니라 고정 ablation arm이다.
- 내부 target-aware: `예` — V11과 V13 계열 STOP 및 STDP 경계를 본 뒤 설계했다.
- 외부 target-aware: `예` — TRACE 논문의 공개 baseline 결과를 본 뒤 benchmark를 선택했다. 최종 task order와 test split을 새로 봉인해야 한다.

### benchmark·baseline

1. 저비용 killing stage: CLINC150을 class-incremental task stream으로 재현하되 원 dataset의 train/validation/test와 OOS test를 보존하고, 최소 5개 봉인 task order를 사용한다.
2. 생존 시 CR3 stage: TRACE 8-task continual benchmark의 공식 preprocessing/evaluation. 1–3B 공개 backbone의 adapter-only 비교를 먼저 하고 전체 scale 주장은 하지 않는다.
3. baseline: sequential fine-tune, joint/offline upper bound, frozen retrieval, equal-memory experience replay, LoRA, EWC, OGD/GEM 또는 TRACE 공식 지원 arm. optimizer·update 수·memory byte를 맞춘다.
4. metric: final average accuracy, average forgetting, backward/forward transfer, OOS, instruction/safety retention, peak memory, update FLOPs, guard accept/reject rate.

### 교차 예측

- replay 제거는 과거-task forgetting을, guard 제거는 harmful commit을 선택적으로 늘려야 한다.
- contractive shell의 이득은 frozen retrieval보다 새 task 학습에, unconstrained adapter보다 과거-task 보존에 각각 나타나야 한다.
- distribution boundary에서 reject rate와 retained-state reset 효과가 증가해야 하며, 평시에는 거의 증가하지 않아야 한다.
- task order를 바꿔도 효과 부호가 유지되어야 한다.

### 죽이는 반증

- strongest equal-memory baseline 대비 final average accuracy 차이의 95% 하한이 0 이하이거나 forgetting이 더 큼.
- guard가 거의 모든 update를 거부해 plasticity가 사라지거나, harmful update를 받아 과거 guard를 깨뜨림.
- replay/guard/local-state lesion이 예상 축을 선택적으로 바꾸지 않음.
- 한 task order에서만 양수이고 봉인된 다른 order에서 효과 부호가 뒤집힘.

### 최소 구현

frozen embedding adapter, `LearnableSmallGainLocalCloud`의 streaming `partial_fit/state_dict`, byte-accounted replay buffer, transactional guard rollback, CLINC/TRACE adapter, order manifest. V10 hand-designed kernel을 반복하지 않고 V11 Elman-3/GRU 계열 대조를 보존한다.

## 5. R3 — intervention-authorized causal world model for embodied OOD

### 목표

paired intervention으로 반복 확인된 directed effect만 world-model cross-channel로 허가할 때, observational/dense transition model보다 held-out intervention과 novel environment의 success·sample efficiency를 개선하는지 검증한다. 방향은 `OOD success/return·action efficiency 증가`, 층위는 `embodied interactive benchmark`다.

### 새 공리 1개

**[공리: interventional authorization]** 동일 random exogenous noise를 공유한 paired do-intervention의 효과가 사전 threshold와 seed-stability를 통과한 directed edge만 policy/world-model의 cross-group message가 될 수 있다. observational correlation만으로는 edge를 열지 않는다.

### 조정 자유도와 target-awareness

- directed edge budget $K\in\{8,16\}$, graph refresh interval $H\in\{32,128\}$: `d_tune=2`, `L=4`.
- state grouping과 action interface는 benchmark metadata로 한 번 고정하고 tuning하지 않는다.
- 내부 target-aware: `예` — V7 test 미개봉과 기존 four-chart 제한을 본 뒤 설계했다.
- 외부 target-aware: `예` — CausalWorld 공식 baseline surface와 ARC-AGI-3 공개 score/goal 자료를 본 뒤 단계 경로를 확정했다. ARC-AGI-3 private environments만 최종 판정력이 있다.

### benchmark·baseline

1. 저비용 killing stage: CausalWorld structured observation과 공식 do-intervention/generalization protocol. 5개 이상 seed에서 ID, appearance-only, mass/dynamics, goal-shape, combined shift를 분리한다.
2. 생존 시 CR3 stage: ARC-AGI-3 public games는 adapter development/behavior audit에만 쓰고 공식 hidden evaluation에서 score, levels, action efficiency를 측정한다.
3. baseline: same policy without world model, dense learned transition, observational sparse/Granger edge, random sparse edge, current rank-1 `BeliefController`, oracle graph upper bound, PPO/SAC 등 official or strong same-budget control.
4. ablation: intervention-pair permutation, selected-edge cut, direction reversal, graph freeze, uncertainty removal.

### 교차 예측

- ID 이득보다 mass/dynamics intervention OOD 이득이 커야 하고 shift severity와 이득이 함께 증가해야 한다.
- appearance-only shift에는 causal graph의 선택적 이득이 작거나 없어야 한다.
- edge cut/reversal은 해당 descendant outcome만 선택적으로 악화시켜야 한다.
- seed/task 사이 선택 edge 안정성과 OOD 성능이 양의 관계를 가져야 한다.

### 죽이는 반증

- strongest equal-sample/compute baseline 대비 OOD success/return 차이의 95% 하한이 0 이하.
- intervention-pair permutation이 실제 pair와 같은 성능을 내거나 selected-edge lesion이 무효.
- ID 이득만 있고 held-out intervention severity가 커질수록 이득이 사라짐.
- 선택 edge의 seed 안정성이 preregistered 하한보다 낮아 동일 mechanism을 재현하지 못함.

### 최소 구현

hard-coded four-chart validation을 generic registered groups로 바꾸는 별도 adapter, CausalWorld structured-state wrapper, graph-to-policy sidecar, intervention audit log, ARC-AGI-3 SDK adapter. 먼저 vector/structured observation만 허용하고 vision encoder는 이 경로가 CausalWorld에서 살아남은 뒤 별도 계약으로 연다.

## 6. R4 — verifier-monotone metacognitive self-correction

### 목표

독립 실행 verifier를 state로 되먹이는 self-correction loop가 동일 model·token budget의 single pass, fixed revision, self-consistency보다 실제 해결률을 높이면서 regression을 만들지 않는지 검증한다. 방향은 `resolved rate 증가`, `false correction·tokens per solved 감소`다.

### 새 공리 1개

**[공리: verifier-monotone commit]** 수정은 독립 executable verifier에서 이전보다 엄격히 더 많은 test를 통과하고 기존 통과 test를 깨뜨리지 않을 때만 commit된다. 모델의 자기평가 문장만으로는 commit할 수 없다.

### 조정 자유도와 target-awareness

- 최대 revision 수 $R\in\{1,3\}$: `d_tune=1`, `L=2`.
- commit threshold는 strict monotonicity로 고정하고 tune하지 않는다.
- 내부 target-aware: `예` — 현재 metacognition이 단순 $\rho$-decay이고 text environment가 fixture라는 코드를 본 뒤 설계했다.
- 외부 target-aware: `아니오(점수 기준)` — SWE-bench Verified의 official dataset/evaluator 구조는 확인했지만 특정 agent의 leaderboard 수치를 경로 선택에 쓰지 않았다.

### benchmark·baseline

1. killing stage: 봉인한 SWE-bench Verified 50-instance development subset, gold patch 미접근.
2. 생존 시 CR3 stage: official SWE-bench Verified 500 instance와 official Docker evaluator; 결과·trajectory·cost를 공개한다.
3. baseline: same model의 mini-SWE-agent 또는 동일 bash-only scaffold single pass, fixed 1/3 revisions, token-matched self-consistency, verbal reflection, test-feedback-only loop.
4. metric: resolved %, pass-to-pass regression, failed-to-pass conversion, verifier false accept, tokens/latency/cost per solved, accepted revision count.

### 교차 예측

- 이득은 실행 가능한 failing-test signal이 있는 item에 집중되고, 처음부터 통과한 item은 거의 수정하지 않아야 한다.
- verifier delta가 다음 revision 성공을 예측하고, shuffled/withheld verifier state에서는 이 상관과 이득이 사라져야 한다.
- revision 수를 늘려도 commit 수는 포화하고 regression은 0에 가까워야 한다.

### 죽이는 반증

- strongest token-matched baseline 대비 resolved 차이의 95% 하한이 0 이하.
- previously passing test regression 또는 hidden-test/gold-patch leakage가 한 건이라도 확인됨.
- verifier state shuffle 후에도 같은 이득이 나와 self-correction의 인과 매개가 부정됨.
- 대부분의 수정이 strict verifier gate에서 거부되거나, accepted revision이 최종 해결률과 무관.

### 최소 구현

`RevisionState`/`VerifierDelta` schema, transactional patch commit/rollback, same-budget runner, official SWE-bench prediction exporter. 실행은 untrusted code와 model artifact를 격리하는 sandbox가 준비되기 전에는 열지 않는다. 현재 `TextEnvironment` 응답 fixture를 benchmark agent로 재사용하지 않는다.

## 7. 공식 benchmark 원장

아래 링크는 benchmark의 존재·공개 evaluation surface를 확인하기 위해 열람했다. 최신 frontier 결과의 해석은 source lane 소관이며 이 문서는 우열 수치를 채택하지 않는다.

| benchmark | 공식 1차 자료 | 이 경로에서의 역할 |
|---|---|---|
| BABILong | [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c0d62e70dbc659cc9bd44cbcf1cb652f-Paper-Datasets_and_Benchmarks_Track.pdf), [official code](https://github.com/booydar/babilong) | recurrent memory/length OOD 저비용 killing stage |
| ARC-AGI-2 | [official benchmark](https://arcprize.org/arc-agi/2), [2026 competition](https://arcprize.org/competitions/2026/arc-agi-2) | latent reasoning의 private held-out CR3 stage |
| CLINC150 | [author repository](https://github.com/clinc/oos-eval) | low-cost continual/OOS sentinel; frontier 자체로 간주하지 않음 |
| TRACE | [OpenReview benchmark paper](https://openreview.net/forum?id=3qa4YLkcEw), [author code](https://github.com/BeyonderXX/TRACE) | LLM continual learning full stage |
| CausalWorld | [author repository](https://github.com/rr-learning/CausalWorld), [paper](https://arxiv.org/abs/2010.04296) | do-intervention과 causal transfer killing stage; 오래된 기반 benchmark임을 명시 |
| ARC-AGI-3 | [official benchmark](https://arcprize.org/arc-agi/3), [official docs](https://docs.arcprize.org/), [2026 competition](https://arcprize.org/competitions/2026/arc-agi-3) | novel interactive environment/continual adaptation CR3 stage |
| SWE-bench Verified | [official benchmark](https://www.swebench.com/verified.html), [official dataset guide](https://www.swebench.com/SWE-bench/guides/datasets/) | executable-verifier self-correction CR3 stage |

## 8. 권고 순서와 중단 규칙

1. **R2**: 구현 재사용률이 가장 높고 CLINC killing stage가 가장 싸다. 여기서 equal-memory replay를 못 이기면 full TRACE를 열지 않는다.
2. **R1**: C4를 가장 직접적으로 시험한다. BABILong에서 depth–difficulty·lesion 교차 예측이 없으면 ARC-AGI-2를 열지 않는다.
3. **R3**: CausalWorld structured-state 단계가 살아남아야 vision/ARC-AGI-3 adapter를 별도 계약으로 연다.
4. **R4**: dof는 작지만 container·LLM 비용과 안전 경계가 크다. 50-instance gate와 sandbox가 먼저다.

어느 경로도 다른 경로의 실패를 보상하지 않는다. 한 경로의 PASS는 해당 benchmark·backbone·budget에서의 CR3 비교 가능성만 만들며 AGI, 생물학적 동일성, nested SCC 우위를 닫지 않는다.
