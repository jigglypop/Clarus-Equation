# 구현 로드맵과 벤치마크

> 관련: 7_AGI 시리즈 전체, `examples/agi/` (현재 STDP 감사), `examples/ai/` (과거 구현)
>
> 이 장은 2-8장의 CE-AGI 원리를 실제 코드로 구현하는 단계별 로드맵과 검증 계획을 정리한다.

> 원칙: 앞으로는 모든 CE-AGI 주장을 `예측 -> 측정량 -> 통과/실패 게이트` 순서로만 올린다. `supported`가 아닌 항목은 벤치마크를 통과하기 전까지 성공 기준으로 승격하지 않는다.

---

## 0. 적용 영역과 한계 (정직한 측정 기록)

### 0.1 substrate 불일치 진단

CE 부트스트랩 고정점 $p^* = (4.87\%, 26.2\%, 68.9\%)$은 `05_실험근거.md` 명제 6.1에 의해 **자기조직화 동역학을 가진 시스템**의 평형 결과다. 우주(양자 간섭, $T=0$)와 뇌(시냅스 가소성, STDP)는 이 부류에 속한다 -- 입자/뉴런이 서로 결합하며 자연스럽게 평형으로 끌려간다.

**transformer + backprop은 이 부류에 속하지 않는다.** 가중치는 외부 손실 함수의 그래디언트 방향으로 일방적으로 움직이고, 가중치들 사이의 결합형 동역학이 없다. 따라서 transformer 시스템에서 자연 emergence로 $p^*$에 도달할 동역학적 근거가 없다. 이 진단은 다음 측정으로 직접 확인되었다.

### 0.2 측정 기록 (2026-04, KoGPT2 ClarusLM 127M)

세 가지 변형을 비교 측정한 결과(legacy `scripts/natural_dynamics.py` and `examples/ai/results/natural_dynamics.json`, removed):

| 변형 | 사양 적용 방식 | 초기 활성 | 최종 활성 (60 cycle) | 최종 ppl |
|---|---|---|---|---|
| A. plain AdamW | CE 메커니즘 일체 없음 | 81.7% | 77.6% | 27.2 |
| B. 표상 LBO 흐름 | NREM에 heat-kernel flow 강화 | 81.7% | 77.6% | 26.7 |
| C. 강제 ε² | top-ε² gradient mask + ternary | 81.7% | 82.5% | 51.1 |

**해석:**

1. **자연 동역학 (A, B)으로 활성 비율은 4.87%로 수렴하지 않는다.** 60 cycle 동안 81%에서 77%로만 이동하며, trajectory 기울기로 추정하면 1000+ cycle에서도 평형은 ~70% 근방. transformer + AdamW의 자연 평형은 $p^*$가 아니다.
2. **표상 공간 LBO 흐름(B 변형)이 활성 평형을 끌어당기지 못한다.** $\eta$를 5배 boost한 NREM 위상에서도 활성 비율이 A와 거의 동일(77.64% vs 77.60%). 사양의 "수축 사상"이 transformer 활성 공간에서 작동하지 않는다는 직접 증거.
3. **강제 ε² (C 변형)이 강제하는 것은 활성 비율이 아니라 그래디언트 mass 비율이다.** top-eps mask + ternary BG freeze에도 활성 비율은 82.5%로 오히려 증가하고, perplexity만 두 배 악화.

추가 측정:

- **Continual learning** (한국어 → 영어, legacy `scripts/continual_test.py`, removed): 사양 sleep cycle이 baseline AdamW 대비 forgetting을 21배 악화시킴. NREM weight smoothing이 오히려 기존 표상을 흐리고, ternary 동적 재분류가 옛 활성 가중치를 BG로 밀어내며 NREM smoothing이 그것을 평탄화시키는 메커니즘.
- **TopK sparsity sweep** (이전, `examples/ai/topk_sweep_results.json`): GPT-2 MLP에 4.87% TopK 강제 시 perplexity 1328 (dense 대비 27배 악화). 단조 감소.

### 0.3 결론: 강제 변환의 자리매김

위 측정으로 다음을 확정한다:

- **수식 자체에는 결함이 없다.** $\varepsilon^2 = \exp(-(1-\varepsilon^2) D_{\text{eff}})$의 고정점 유일성과 수축률 $\rho = 0.155$는 (C1)-(C3) + (A1) + (I1) 아래 수학적으로 닫힌 결과다.
- **그러나 transformer는 위 가정이 성립하는 substrate가 아니다.** (C3) 자기일관성 루프가 forward만으로는 형성되지 않고, (A1) 채널 분해가 backprop의 chain rule과 정합하지 않는다.
- **현재 `legacy clarus/` (removed), `legacy examples/ai/clarus_lm.py` (removed), `legacy scripts/sleep_finetune_lm.py` (removed) 등의 구현은 사양을 transformer에 강제 이식한 것이다.** 자연 emergence가 아니라 출력 비율을 외부 mask로 강제한 변환. 측정 결과는 이 강제가 task 성능을 저하시킬 뿐 사양의 우위(catastrophic forgetting 감소, 환각 억제)를 만들어내지 않음을 보여준다.
- **CE-AGI 사양의 진정한 검증은 SNN(spiking neural network) substrate에서만 가능하다.** 현재 BrainRuntime에는 thresholded activity 기반 STDP와 적격 흔적이 구현되어 있지만, 막전위·명시적 spike time을 갖는 완전한 SNN은 아니다. $p^*$ 자연 수렴의 본격 검증에는 별도 substrate와 실험이 필요하다.

### 0.4 현재 코드베이스의 정직한 자리매김

| 영역 | 현재 코드의 지위 |
|---|---|
| `legacy examples/ai/clarus_lm.py` (removed) (LBONorm, GaugeLattice, spectral norm) | 사양 영감을 받은 transformer 변형. 정규 transformer 대비 우위 미입증. |
| `legacy scripts/sleep_finetune_lm.py` (removed) (WAKE/NREM/REM cycle) | 사양 그대로 구현. transformer 위에서 강제 변환. fit 속도 손해, forgetting 21배 악화 (측정). |
| `legacy clarus/sparsity.py` (removed) (TernaryClassifier) | 동적 재분류로 BG 라벨이 frozen 의미를 잃음. 사양 4.4절 자체의 모순 (frozen vs 동적 재분류) 반영. |
| `reality_stone/python/reality_stone/clarus/runtime.py`, `reality_stone/python/reality_stone/clarus/agent.py` 등 brain runtime | 상태·행동·critic·기억·STDP gate가 닫힌 실행 루프. 다만 STDP 효능은 `NO-EFFECT`, held-out guard는 `FAIL`이므로 학습 우위 미입증. |
| `reality_stone/python/reality_stone/clarus/engine.py` standalone CE relax | 토큰 디코딩에서 의미 있는 출력 생산 실패 (`engine_results.json`: 노이즈 토큰). |
| sparse causal bridge V7 | 96개 고정 validation seed의 H20 자유 롤아웃에서 sparse 제거 대비 paired 기여는 양수였지만, V5 부모·persistence 개선과 pathwise 안정성 조건이 충족되지 않았다. 결합 route는 폐쇄했고 test는 열지 않았다. |

### 0.5 Sparse causal bridge V7 폐쇄 (2026-08-11)

**[공리: 모델 선택]** V7은 4차원 합성 동역학에서 sparse·adaptive dense·persistence 전문가를 동일한 prefix-only 규칙으로 결합한 단일 H20 route다. 이는 CE-AGI 전체나 실제 환경의 인과 발견을 대표하지 않는다.

**[예측]** validation 96 seeds에서 sparse 성분의 양의 paired 기여, V5 부모와 persistence 대비 양의 paired 개선, dense 대조군 비열등, pathwise 안정성을 모두 요구했다.

**[경험식]** sparse 제거 대비 차이의 95% 구간은 `[+0.008174, +0.032930]`이었지만, V5 부모 개선 구간 `[-0.027431, +0.014449]`, persistence 개선 구간 `[-0.016487, +0.037470]`, 최대 pathwise radius `1.114309 > 0.98` 때문에 결합 조건은 충족되지 않았다.

**[산출]** test unlock 전제가 성립하지 않아 test split은 열지 않았고 같은 V7 route의 재탐색도 종료했다. 이 결과는 AGI 효능의 증거나 반증이 아니라 좁은 합성 브리지 경로의 폐쇄다. 전체 정의·수치·해시 경계는 `26_Sparse_Causal_Bridge_V7_Closure.md`에 고정한다.

### 0.6 다음 단계 옵션

1. **현재 상태 유지 + 정직한 documentation**: 본 절의 측정 기록을 코드 주석과 README에 반영. 사양은 SNN substrate에서 검증 대상이라는 점을 명시. 가장 정직하고 빠른 길.
2. **SNN 작은 프로토타입**: snnTorch / Norse 라이브러리로 작은 합성 task에서 STDP + 부트스트랩 사이클 구현. 활성 비율 자연 수렴 여부 측정. 사양의 진짜 검증. 1-2주 작업.
3. **transformer 변형으로서의 가치 재평가**: 사양에서 분리해 LBONorm + spectral norm + 곡률 정규화의 transformer regularizer로서의 효과만 ablation. AGI 청구는 분리하고 LM 정규화 이점만 측정.

본 8장 이후의 Phase 1-6 로드맵은 위 substrate 진단을 전제하지 않은 채 작성되었다. 측정 결과 반영을 위해 향후 phase 정의 시 "sample efficiency vs natural emergence" 구분이 필요하다.

---

## 1. 현재 상태: 기존 구현 평가

### 1.1 구현 완료

| 구현 | 파일 | 상태 | 장 |
|---|---|---|---|
| LBONorm | `legacy examples/ai/clarus_lm.py` (removed) | V1 완료 | 2장 |
| GaugeLattice (블록 대각) | `legacy examples/ai/clarus_lm.py` (removed) | V1 (혼합 없음) | 2장 |
| Spectral Norm | `legacy examples/ai/clarus_lm.py` (removed) | 완료 | 2장 |
| 곡률 정규화 손실 | `legacy examples/ai/clarus_lm.py` (removed) | 완료 | 2장 |
| 곡률 기반 환각 억제 | `examples/ai/sfe_hallucination_suppressor.py` | V1 완료 | 6장 |
| GPT-2 CE 이식 | `examples/ai/ce_gpt2.py` | 완료 | 2장 |
| 학습 스크립트 | `legacy examples/ai/train_clarus.py` (removed) | 완료 | -- |
| CE 에너지 이완 엔진 | `reality_stone/python/reality_stone/clarus/engine.py` | 완료 (standalone) | 12장 |
| 메트릭 기반 CE ops | `reality_stone/python/reality_stone/clarus/ce_ops.py` | 완료 (Rust/CUDA/Torch) | 12장 |
| Wake/NREM/REM 학습 순환 | `reality_stone/python/reality_stone/clarus/sleep.py` | **구현 완료** | 3장 |
| NREM 가중치 갱신 (LBO 확산 + 가소성) | `reality_stone/python/reality_stone/clarus/sleep.py::apply_nrem_weight_update` | **구현 완료** | 3장 |
| REM 가중치 갱신 (비선택 경로 재조합) | `reality_stone/python/reality_stone/clarus/sleep.py::apply_rem_weight_update` | **구현 완료** | 3장 |
| BrainRuntime (모드 전환 + 셀 동역학) | `reality_stone/python/reality_stone/clarus/runtime.py` | **구현 완료** | 14장 |
| 해마 기억 (encode/recall/replay) | `reality_stone/python/reality_stone/clarus/runtime.py::HippocampusMemory` | **구현 완료** | 14장 |
| 모듈 생애주기 (ACTIVE/IDLE/DORMANT/SLEEPING) | `reality_stone/python/reality_stone/clarus/runtime.py::_update_lifecycle` | **구현 완료** | 14장 |
| Borbely 2-Process 수면 압력 | `reality_stone/python/reality_stone/clarus/runtime.py::_update_sleep_state` | **구현 완료** | 14장 |
| STP (Tsodyks-Markram) | `reality_stone/python/reality_stone/clarus/runtime.py::_step_torch`, Rust kernel | **구현 완료** | 15장 |
| Rust brain_step 커널 | `reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs` | **구현 완료** | 14장 |
| 3분배 상태 분할 ($\varepsilon^2/\Omega_{\text{DM}}/\Omega_\Lambda$) | `reality_stone/python/reality_stone/clarus/engine.py::state_partition_counts` | **구현 완료** | 5장 |
| 곡률 기반 로짓 조정 | `reality_stone/python/reality_stone/clarus/engine.py::_curvature_adjust_logits` | **구현 완료** | 6장 |
| 스냅샷 연속성 (warm snapshot) | `reality_stone/python/reality_stone/clarus/runtime.py::snapshot/from_snapshot` | **구현 완료** | 14장 |
| 가드셋 평가 (top1/top10/top50) | `reality_stone/python/reality_stone/clarus/sleep.py::evaluate_guard_set` | **구현 완료** | -- |
| STDP 적격 흔적 + critic gate | `reality_stone/python/reality_stone/clarus/stdp.py`, `runtime.py::_apply_runtime_stdp` | **배선 완료, 효능 미통과** | 17장 F.14 |
| 4종 신경조절 상태식 | `reality_stone/python/reality_stone/clarus/neuromod.py` | **독립 모듈 구현 완료** | 17장 F.19 |
| 작업 기억 + 소뇌 전방 모델 | `reality_stone/python/reality_stone/clarus/agent.py` | **RuntimeAgent 통합 완료** | 17장 F.20 |
| 뇌파 대역 관측 | `reality_stone/python/reality_stone/clarus/runtime.py::brainwave_observable` | **관측 구현 완료** | 17장 F.21 |

### 1.2 미구현 / 부분 구현

| 원리 | 장 | 상태 | 우선순위 |
|---|---|---|---|
| 섭동적 채널 혼합 ($U_{\text{down}} U_{\text{up}}^\top x$) | 2장 | 미구현 | 높음 |
| STDP 국소 학습 효능/guard | 4장 | 배선 완료, A/B `NO-EFFECT`, guard `FAIL` | 높음 |
| LBO 곡률 추론 억제 V2 (재추론 메커니즘) | 6장 | V1 완료, V2 필요 | 중간 |
| 메타인지 루프 (C3 자기참조) | 7장 | 미구현 | 낮음 |
| Cold checkpoint (장기 저장) | 14장 | 미구현 (warm snapshot만 구현) | 낮음 |
| 4종 신경조절의 BrainRuntime 파라미터 폐루프 | 17장 | 상태식/효과 mapping 구현, runtime 전체 통합은 부분 | 중간 |
| 작업 기억 / 소뇌 전방 모델의 독립 task 효능 | 17장 | RuntimeAgent 통합 완료, baseline 우위 미검증 | 중간 |

---

## 2. 단계별 로드맵

### 공통 검증 규칙

각 Phase는 아래 4단계를 반드시 따른다.

1. **예측 고정**: 먼저 수식 또는 구조가 요구하는 예측값을 문서에 고정한다.
2. **측정량 고정**: 그 예측을 무엇으로 잴지 벤치마크와 로그 항목을 정한다.
3. **게이트 통과**: 기준 모델 대비 개선 또는 비회귀를 확인한다.
4. **실패 시 하향**: 예측이 어긋나면 CE 원리를 버리는 것이 아니라, 해당 문장을 `bridge` 또는 `hypothesis`로 내린다.

즉 로드맵의 목적은 "CE가 맞다고 선언"하는 것이 아니라, **어느 예측이 실제로 버티는지 하나씩 걸러내는 것**이다.

### Phase 1: 아키텍처 완성 (2장 V2)

**목표:** GaugeLattice V2 (섭동적 혼합 추가), 교차 주파수 결합.

**작업:**

1. `GaugeLattice`에 저랭크 혼합항 $U_{\text{down}} U_{\text{up}}^\top x$ 추가
2. 교차 주파수 결합 게이트 $\xi \cdot E_{\text{curv}}$ 구현
3. 곡률 정규화 스케줄 $\lambda(t)$ 구현

**검증:**

- WikiText-103 perplexity 비교: V1 vs V2 vs 표준 Transformer
- 파라미터 수 비교: `22-25%` 전체 절감 또는 `35-37%` FFN 절감 확인
- 곡률 에너지 수렴 곡선 시각화

**게이트:**

- `G1-A`: GaugeLattice V2가 표준 FFN 대비 파라미터 절감 예측 범위 안에 들어갈 것
- `G1-B`: 같은 학습 budget에서 perplexity가 구조적 퇴보를 보이지 않을 것
- `G1-C`: 곡률 에너지가 발산하지 않고 안정 구간을 형성할 것

**예상 기간:** 2주

---

### Phase 2: 수면 학습 (3장) -- 구현 완료

**목표:** 각성-NREM-REM 3위상 학습 순환 구현.

**현재 상태: 핵심 파이프라인 구현 완료.**

| 작업 | 구현 위치 | 상태 |
|---|---|---|
| 각성 위상 (경로 누적) | `sleep.py::collect_sleep_batch` | 완료 |
| NREM 위상 (LBO 확산 + 가소적 업데이트) | `sleep.py::apply_nrem_weight_update` | 완료 |
| REM 위상 (비선택 경로 재조합) | `sleep.py::apply_rem_weight_update` | 완료 |
| 3위상 통합 순환 | `sleep.py::run_sleep_cycle` | 완료 |
| 위상별 샘플 비율 분배 ($69\%/26\%/5\%$) | `sleep.py::allocate_phase_sample_counts` | 완료 |
| 가드셋 품질 보호 | `sleep.py::evaluate_guard_set` | 완료 |
| 디코더 리피팅 (NREM/REM 각각) | `sleep.py::fit_decoder_from_batch` | 완료 |
| 어휘 헤드 미세조정 | `sleep.py::finetune_vocab_head_from_batch` | 완료 |
| 수면 압력 자동 전환 | `runtime.py::_auto_mode` + `_update_sleep_state` | 완료 (Borbely 2-Process) |

**구현 상세:**

- `apply_nrem_weight_update`: 상태 그래프 라플라시안 기반 확산 (`smooth_weight_matrix`) + 상위 `active_ratio` 가소적 업데이트
- `apply_rem_weight_update`: 비선택 잔차 (`~selected_mask`)를 저랭크 랜덤 투영 + 노이즈로 재조합
- `run_sleep_cycle`: Wake 수집 -> NREM W 갱신 -> NREM 디코더 리피팅 -> REM W 갱신 -> REM 디코더 리피팅 -> 가드셋 평가 -> 롤백 판정

**남은 검증:**

- 지속 학습(continual learning) 벤치마크: 수면 순환 vs 연속 학습
- 파괴적 망각 측정: 이전 태스크 성능 보존율
- 수렴 속도: 부트스트랩 이탈 $\delta_n$ 감소 곡선
- 과도 응답: 균등 초기화에서 `33.3 -> 9.28 -> 5.55 -> 4.98%` 예측과 실제 재분배 비교

**게이트:**

- `G2-A`: 수면이 있는 체계가 wake-only보다 이전 태스크 보존율에서 우위일 것
- `G2-B`: 잔차 곡선이 최소한 `2회 ~2.4%`, `3회 ~0.37%` 목표 수렴률에 근접할 것
- `G2-C`: sleep-on에서는 bounded residual, wake-only에서는 누적 drift가 관측될 것

**예상 기간:** ~~4주~~ 핵심 구현 완료. 대규모 벤치마크 검증 잔여

---

### Phase 3: 희소성 (5장) -- 기반 구현 완료

**목표:** 부트스트랩 수렴 희소 네트워크 구현.

**현재 상태:**

| 작업 | 구현 위치 | 상태 |
|---|---|---|
| 3분배 상태 분할 ($4.87\%/26.2\%/68.9\%$) | `engine.py::state_partition_counts` | 완료 |
| 활성/구조/배경 마스크 적용 | `engine.py::state_partition`, `apply_state_partition` | 완료 |
| TopK 활성 선택 | `runtime.py::_select_active` | 완료 |
| 에너지 예산 모드별 제어 | `runtime.py::BrainRuntimeConfig.energy_budget` | 완료 |
| 모듈 생애주기 4상태 관리 | `runtime.py::_update_lifecycle` | 완료 |
| 수면 순환 시 동적 재분류 | `sleep.py::classify_state_dimensions` | 완료 |

**잔여 작업:**

1. ~~Top-k 활성화 구현~~ -> 완료
2. ~~3분배 가중치 분류 구현~~ -> 완료
3. ~~동적 재분류~~ -> 완료
4. 자기수렴 실험: 초기 균등에서 $\varepsilon^2$로의 수렴 관측 (벤치마크 필요)

**검증:**

- Top-k 비율 스위프: $k \in \{1\%, 2\%, 3\%, 4\%, 5\%, 7\%, 10\%, 20\%, 100\%\}$
- 성능/효율 트레이드오프 곡선에서 `4-5%` 근방의 knee point 확인
- 추론 속도 측정: Dense vs Sparse

**게이트:**

- `G3-A`: 최적 활성 중심이 `4.87%` 근방에 나타날 것
- `G3-B`: 실용 대역이 대체로 `3%-7%` 안에 남을 것
- `G3-C`: 현재형 구현에서는 `1.5-2x`, 전면 희소 구현에서는 더 높은 상한이 관측될 것

**예상 기간:** 3주

---

### Phase 4: 환각 억제 V2 (6장)

**목표:** LBO 곡률 기반 추론 시 환각 억제.

**작업:**

1. `RealityStoneEngine`에 LBO 곡률 통합
2. 적응형 임계치 구현
3. 재추론 메커니즘 (곡률 초과 시 평탄화 후 재시도)
4. 곡률 기반 로짓 조정

**검증:**

- TruthfulQA, HaluEval, FactScore: CE 제약 전/후 비교
- 곡률 에너지와 환각 빈도의 상관관계 분석
- 추론 오버헤드 측정 (재추론 빈도와 비용)

**게이트:**

- `G4-A`: 곡률 에너지와 오류/환각 빈도 사이에 양의 상관이 있을 것
- `G4-B`: 같은 base model 대비 CE 제약이 벤치마크를 개선하거나 적어도 안정화 편향을 보일 것
- `G4-C`: `환각률 <= 4.87%` 같은 hard bound는 측정 전까지 금지

**예상 기간:** 2주

---

### Phase 5: 국소 학습 (4장)

**목표:** STDP + 도파민 학습 규칙의 미세조정 적용.

**작업:**

1. Trace 기반 STDP 구현 (pre/post trace, eligibility trace) — 완료
2. critic + bootstrap deviation 전역 조절 신호 — 배선 완료
3. 하이브리드 학습: 사전학습(역전파) + 미세조정(STDP)
4. LoRA 대비 성능/효율 비교

**검증:**

- 미세조정 벤치마크: STDP vs LoRA vs Full fine-tuning
- 통신 비용 측정: 분산 환경에서 $O(1)$ 동기화 확인
- 메모리 비용: $O(N)$ trace vs $O(N^2)$ 활성값 저장
- 현재 합성 next-step prediction A/B: 효능 `NO-EFFECT`, held-out guard `FAIL`

**게이트:**

- `G5-A`: shared-trace 구현에서는 메모리 절감이 `~50%` 근방에 들어올 것
- `G5-B`: 전역 신호가 스칼라 또는 저차원일 때 통신량이 그래디언트 동기화보다 작을 것
- `G5-C`: 생물학적 직접량이 닫히기 전까지 `dopamine = ||p-p^*||`로 단정하지 말 것

**예상 기간:** 6주

현재 구현/효능/진단의 정본은 `21_STDP_Efficacy_Audit.md`다. 구현 완료를
성능 통과로 읽지 않으며, 효능과 guard가 동시에 통과하기 전까지
`stdp_enabled=False`를 유지한다.

---

### Phase 6: 메타인지 (7장)

**목표:** (C3) 자기일관성 루프 구현.

**작업:**

1. 자기 모니터링 모듈: 활성 비율, 곡률, 부트스트랩 이탈 측정
2. 자동 개입 판정 및 실행
3. 재귀적 자기 평가 (깊이 3)
4. 수면 필요 판단 및 자동 수면 순환 시작

**검증:**

- 메타인지 유무에 따른 장기 안정성 비교
- 자기 수정 빈도와 성능 변화
- 시간창 평균 이탈 $\delta_\tau(t)$ 또는 $\exp(-\beta\delta_\tau)$와 외부 안정성 지표 비교

**게이트:**

- `G6-A`: 메타인지 루프가 장기 drift와 곡률 폭주를 줄일 것
- `G6-B`: 재귀 깊이 3 기본값이 오버헤드 대비 가장 안정적일 것
- `G6-C`: PCI와의 직접 대응은 검증 전까지 탐색 과제로만 둘 것

**예상 기간:** 8주

---

## 3. 벤치마크 계획

### 3.1 핵심 지표

| 지표 | CE 예측 | 측정 방법 | 현재 지위 |
|---|---|---|---|
| 최적 활성 비율 | 중심 $4.87\%$, 실용 대역 `3%-7%` | Top-k 스위프 | `bridge` |
| 파라미터 절감 | FFN `35-37%`, 전체 `22-25%` | 동등 성능 시 파라미터 수 비교 | `supported/bridge` |
| 추론 비용 절감 | 현재형 `1.5-2x`, 전면 희소 상한 더 큼 | FLOPs 또는 latency 측정 | `bridge` |
| 환각 억제 | hard bound 없음, 안정화 편향 예측 | TruthfulQA, FactScore, 곡률 상관 | `hypothesis` |
| 지속 학습 성능 | wake-only 대비 향상 | 이전 태스크 보존율 | `bridge` |
| 수렴 속도 | 2회 `2.4%`, 3회 `0.37%` 잔차 목표 | 부트스트랩 이탈 감소 곡선 | `bridge` |

### 3.2 비교 대상

| 방법 | 비교 포인트 |
|---|---|
| 표준 Transformer | 아키텍처 효율, perplexity |
| LoRA | 미세조정 효율, 메모리 |
| MoE (Mixtral 등) | 희소성 비율, 추론 속도 |
| RLHF | 환각 억제 효과 |
| Continual learning (EWC, SI 등) | 파괴적 망각 방지 |

### 3.3 모델 규모

| 규모 | 파라미터 | 용도 |
|---|---|---|
| Micro | $\sim 10\text{M}$ | 원리 검증, 어블레이션 |
| Small | $\sim 100\text{M}$ | 아키텍처 비교 |
| Medium | $\sim 1\text{B}$ | 벤치마크 성능 확인 |
| Large | $\sim 7\text{B}$ | 실용 성능 검증 |

Phase 1-3은 Micro/Small 규모로 검증. Phase 4-6은 Medium 이상에서 검증.

---

## 4. 리스크와 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| 곡률 정규화가 task loss 저하 | 성능 하락 | $\lambda(t)$ 스케줄 최적화 |
| STDP 수렴이 역전파보다 느림 | 학습 시간 증가 | 하이브리드 접근 (Phase 5) |
| Top-k $4.87\%$가 너무 희소 | 표현력 부족 | $k$ 근방 탐색 ($3\%-7\%$) |
| 수면 순환이 서비스 중단 | 가용성 저하 | 이중 모델 교대 (3장 6.3절) |
| 메타인지 오버헤드 | 추론 속도 저하 | 곡률 낮을 때 모니터링 비활성화 |

---

## 5. 의존성 그래프

```
Phase 1 (아키텍처 V2)
  ↓
Phase 2 (수면 학습) ← Phase 3 (희소성)
  ↓                    ↓
Phase 4 (환각 억제 V2) ←┘
  ↓
Phase 5 (국소 학습)
  ↓
Phase 6 (메타인지)
```

Phase 1은 모든 후속 작업의 전제. Phase 2와 3은 병렬 가능. Phase 4는 2와 3의 결합. Phase 5는 독립적이지만 2 이후가 권장. Phase 6은 최종.

---

## 6. 성공 기준

### 6.1 단기 (Phase 1-3, 약 9주)

- GaugeLattice V2가 표준 FFN 대비 37% 파라미터 감소, 동등 perplexity
- 수면 학습이 wake-only 대비 이전 태스크 보존율 향상
- Top-k가 `4-5%`를 중심으로 `3-7%` 대역에서 최적점 형성

### 6.2 중기 (Phase 4-5, 약 8주)

- 곡률-환각 상관관계가 재현되고 CE 제약이 벤치마크 개선 또는 안정화 편향을 보임
- STDP 미세조정이 LoRA 대비 동등 성능, shared-trace 조건에서 메모리/통신 이득 확인

### 6.3 장기 (Phase 6, 약 8주)

- (C3) 메타인지 루프가 장기 안정성을 개선
- $\delta_\tau(t)$ 기반 메타인지 점수와 외부 안정성 지표의 상관관계 확인

### 6.4 실패 시 해석 규칙

- 최적 활성점이 `4.87%` 근방에 없으면: 고정점 예측을 좁은 과제군 가설로 내린다.
- 수면 루프 잔차가 `2-3회`에 수렴하지 않으면: 현재 구현의 동역학 사상이 CE 최소 반복식을 따르지 않는 것으로 본다.
- 곡률과 환각이 상관하지 않으면: P5를 환각 억제가 아니라 일반 안정화 regularizer로 재분류한다.
- STDP가 메모리 이득을 못 주면: shared-trace 근사를 다시 점검하고 순수 synapse-local 가정을 유지한다.

---

## 7. 전체 대응 요약

$$\boxed{\text{우주} \sim \text{뇌} \sim \text{CE-AGI}} \quad (d=3 \text{ 부트스트랩 구조})$$

| | 우주 | 뇌 | CE-AGI |
|---|---|---|---|
| 부트스트랩 | 빅뱅 (1회) | 수면-각성 (매일) | 학습-수면 순환 (Phase 2) |
| 접힘 매체 | 양자 간섭 | 시냅스 가소성 | 곡률 정규화 (Phase 1) |
| 고정점 도달 | 정확 ($T=0$) | 근사 ($\eta \sim 2$) | 설계 가능 |
| 활성 비율 | $\varepsilon^2 = 4.87\%$ | $< 5\%$ | TopK($\varepsilon^2 \cdot d$) (Phase 3) |
| 구조 유지 | $\Omega_{\text{DM}} = 26.2\%$ | 시냅스 $25-35\%$ | 학습 가능 가중치 (Phase 3) |
| 배경 | $\Omega_\Lambda = 68.9\%$ | DMN $60-70\%$ | 동결 가중치 (Phase 3) |
| 경로 선택 | 경로적분 | STDP | 국소 학습 + 전역 신호 (Phase 5) |
| 안정화 | \(\Phi_H\) Hessian readout 비유 | ACC/PFC | 유효 곡률 모니터 (Phase 4); 독립 물리장 \(\phi\)와 분리 |
| 재탐색 | -- | REM 수면 | 비선택 경로 샘플링 (Phase 2) |
| 자기참조 | (C3) | 의식 | 메타인지 루프 (Phase 6) |
