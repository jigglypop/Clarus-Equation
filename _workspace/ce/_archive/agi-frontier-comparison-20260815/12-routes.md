# C4 대안 경로: CE-AGI를 CR3 공개 비교로 연결하기

Status: COMPLETE

## 독립성·판정 경계

이 레인은 계약의 **C4 [가설]**만 대상으로 했고 다른 lane 파일은 읽지 않았다. RBE는 전부 제외했다. 현재 정본은 CE-AGI를 “수학 정리, 실행 가능한 brain-like runtime, 좁은 합성 PoC”로 한정하며 AGI가 아니라고 명시한다 (`docs/7_AGI/1_AGI.md:9-25`). 따라서 아래 후보는 승격안이 아니라 fresh held-out에서 죽일 수 있는 CR3 연결안이다.

V9의 cross-level state는 실제 intervention으로 output을 바꾸지만 task utility는 matched monolithic보다 크게 낮아 STOP였다 (`docs/7_AGI/28_Nested_Infinite_SCC_V9.md:649-670`). V10의 좁은 합성 효과는 남지만 V11과 V13 계열의 강한 recurrent/OOD 경로는 STOP다 (`docs/7_AGI/29_Local_Cloud_Kernel_V10.md:43-89`; `docs/7_AGI/30_Strong_Recurrent_OOD_V11.md:8-50`; `docs/7_AGI/1_AGI.md:16-19`). 이 음성 결과를 새 이름으로 재사용하지 않는다.

## 후보 순위

| 순위·후보 | 한 문장 목표 | 새 공리 | `d_tune / L` | target-aware | 공개 benchmark·강한 baseline | 교차 예측 | 죽이는 조건 | 최소 구현 |
|---|---|---|---:|---|---|---|---|---|
| 1. R2 guarded continual shell | frozen backbone의 contractive local adapter가 equal-memory continual baseline보다 덜 잊고 새 task를 학습하는가 | frozen-core plastic shell 1개 | `2 / 4` | 내부·외부 결과 인지 | CLINC150→TRACE; replay, LoRA, EWC, OGD/GEM, joint upper bound | guard-off는 harmful commit, replay-off는 forgetting을 선택적으로 증가 | strongest equal-memory baseline 대비 평균 정확도 CI 하한 $\le0$ 또는 forgetting 악화 | streaming adapter, byte-accounted replay, rollback guard, dataset/order manifest |
| 2. R1 residual-gated nested SCC | item별 SCC depth가 fixed-depth보다 accuracy–compute Pareto와 reasoning OOD를 개선하는가 | residual halting 1개 | `2 / 6` | 내부·외부 결과 인지 | BABILong→ARC-AGI-2 private; one-pass, fixed/max-depth, ACT/recurrent, token-matched self-consistency | 난이도↑→depth↑, 장기 item에서 cut/reset 효과↑ | compute-matched 정확도 CI 하한 $\le0$, depth collapse, lesion 무효, Pareto 미개선 | latent adapter, halt trace, benchmark adapters, equal-budget evaluator |
| 3. R3 interventional causal world model | paired do-intervention으로 허가한 edge가 embodied intervention OOD를 개선하는가 | interventional authorization 1개 | `2 / 4` | 내부·외부 결과 인지 | CausalWorld→ARC-AGI-3 hidden; dense/observational/random/oracle graph, PPO/SAC, rank-1 belief | shift severity↑→causal 이득↑; appearance-only에는 작은 이득; edge lesion은 descendant만 손상 | OOD CI 하한 $\le0$, permuted intervention 동률, edge lesion 무효, seed 불안정 | generic chart adapter, CausalWorld wrapper, graph sidecar, ARC SDK adapter |
| 4. R4 verifier-monotone self-correction | 독립 executable verifier를 되먹인 수정이 token-matched agent보다 해결률을 높이는가 | verifier-monotone commit 1개 | `1 / 2` | 내부 인지; 외부 점수 blind | SWE-bench Verified; mini-SWE-agent single pass, fixed revision, self-consistency, test-feedback-only | verifier delta가 해결을 예측하고 shuffle하면 이득 소실 | resolved CI 하한 $\le0$, regression/leakage 1건, verifier shuffle에도 이득 유지 | revision schema, transactional patch rollback, official evaluator/exporter, sandbox |

`d_tune`은 CE 전용 증분 선택 수이고 `L`은 그 grid 크기다. 네 후보를 모두 본 뒤 최고 점수만 고르면 최소 16번의 선택 기회가 생기므로 경로별 endpoint를 독립 등록해야 한다. 상세 산식·근거·공식 benchmark 원장은 `artifacts/route-design-ledger.md`에 있다.

## 1. R2 — guarded continual shell

- **구조 경로:** 기존 learnable small-gain operator (`learnable_small_gain_local_cloud.py:15-68,80-215`)와 replay/guard primitive (`sleep.py:55-77,1139-1285`)를 frozen encoder의 작은 plastic shell로 제한한다. V10 hand-designed transition을 반복하지 않고 V11의 Elman-3/GRU 대조를 유지한다.
- **자유 부분:** replay capacity $\{16,64\}$와 update interval $\{4,16\}$만 연다. STDP-only, replay-only, guard-off는 조정 후보가 아니라 ablation이다.
- **판정:** 다섯 개 봉인 task order에서 average accuracy, forgetting, transfer, OOS, memory byte, update FLOP를 함께 본다. CLINC sentinel을 못 넘으면 TRACE를 열지 않는다.

## 2. R1 — residual-gated nested SCC

- **구조 경로:** finite SCC generator·immutable token·실제 cut/reset은 재사용하되, 현재 compatibility-driven grow-every-tick (`adaptive_scc_tower_controller.py:281-287,385-418`)을 target-free normalized residual halting으로 교체한 benchmark adapter를 만든다.
- **자유 부분:** $D_{max}\in\{4,8\}$와 $\tau\in\{0.02,0.05,0.10\}$만 연다. injection 위치와 readout은 고정한다.
- **판정:** BABILong에서 depth–difficulty, lesion–horizon, accuracy–compute 세 교차 예측이 모두 살아야 ARC-AGI-2의 봉인 평가로 간다. V9 seed와 unopened confirmation block은 재사용하지 않는다.

## 3. R3 — intervention-authorized causal world model

- **구조 경로:** 기존 paired intervention estimator (`sparse_causal_bridge.py:1-6,445-520`)와 action-conditioned belief update (`belief_control.py:140-205`)를 외부 structured environment에 연결한다. 현재 four-chart 강제 (`sparse_causal_bridge.py:113-138`)는 registered generic group adapter로 격리한다.
- **자유 부분:** edge budget $\{8,16\}$와 refresh interval $\{32,128\}$만 연다. state grouping은 benchmark metadata에서 사전 고정한다.
- **판정:** CausalWorld의 ID/appearance/mass-dynamics/goal/combined shift를 분리한다. causal 특이 예측이 없으면 vision encoder나 ARC-AGI-3 hidden evaluation을 열지 않는다.

## 4. R4 — verifier-monotone self-correction

- **구조 경로:** 현재 `metacognition_step`은 deviation에 $\rho$를 반복 곱할 뿐이고 (`agent.py:176-215`), text loop는 hash-vector/template fixture다 (`agent.py:608-721`). 이를 독립 executable verifier의 delta를 state로 갖는 transactional revision loop로 새로 정의한다.
- **자유 부분:** 최대 revision $\{1,3\}$만 연다. strict test monotonicity는 고정 규칙이다.
- **판정:** 봉인 50-instance gate 후에만 SWE-bench Verified 전체를 연다. previously passing test regression, gold/hidden-test leakage, verifier-state shuffle에도 유지되는 이득 중 하나라도 나오면 경로를 죽인다.

## 우선 실행 결론

가장 싼 순서는 `R2 CLINC killing gate → R1 BABILong killing gate → R3 CausalWorld structured gate → R4 SWE-bench 50-instance gate`다. 각 gate가 살아남을 때만 각각 TRACE, ARC-AGI-2, ARC-AGI-3, SWE-bench Verified 전체로 확장한다. 성공해도 판정은 해당 축의 **CR3 비교 가능**에 한정하며 C4, AGI, 생물학적 동일성을 닫지 않는다.
