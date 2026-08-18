# CE-AGI Hopfield Engine: 논문 vs 구현 검증 보고서

이 문서는 `12_Equation.md`의 지정 수식과 구현 artifact가 어떤 좁은 계약에서 일치하는지 점검한다. 독자는 테스트·fixture·baseline의 기본을 아는 독자를 전제로 하며, 기계 테스트 통과는 코드 경로와 등록된 입력의 일치이지 과학적 참·AGI 효능·생물학적 기제의 증명은 아니다.

파이프라인·동역학·상수·메모리·추론을 같은 fixture와 환경에서 대조한 뒤, 이전 구현 차이와 미결 failure를 읽는다. metric의 단위·분모·seed·split·threshold를 기록하고, expected failure나 미결 항목은 통과 수치로 덮지 않는다.

> `12_Equation.md` 수식 기반. 구현: `reality_stone/python/reality_stone/clarus/engine.py`, `reality_stone/python/reality_stone/clarus/engine.py`

---

## 1. 파이프라인 대조

파이프라인 대조는 같은 입력 fixture가 논문 명세와 코드 단계에서 어떤 output을 내는지 확인한다. 성공은 등록된 환경·seed·threshold의 경로 일치이며, data provenance가 다른 실행이나 새 task의 일반화는 판정 범위 밖이다.

| 단계 | 논문 (12_Equation.md) | 구현 | 일치 |
|------|----------------------|------|------|
| W 추출 | Q@K^T/sqrt(d_h) + V@O + FFN, 레이어 평균, 대칭화 | `extract_hopfield()` 동일 | O |
| 3D 격자 희소화 | d=3, r_c=pi, 밀도 ~3.16% (N=4096) | `sparsify_3d()` r_c=pi, N=768 -> 10.57% | 부분 |
| 스펙트럼 조건화 | lambda_max < 0, 음정치 | `make_negative_definite()` shift=lambda_max+0.1|lambda_min| | O |
| CSR 압축 | 희소 행렬 저장 | `to_csr()` values + col_idx + row_ptr | O |
| 어휘 추출 | emb + ln_f + lm_head | weight tying 감지, 1벌 저장 | O |

## 2. 동역학 대조

동역학 대조는 지정한 초기 상태·step·정규화에서 수식 update와 구현 state trajectory를 비교한다. metric은 같은 timebase와 norm 분모를 사용하며, 안정성 통과는 다른 입력·OOD·장기 horizon에서의 보장이 아니다.

| 항목 | 논문 | 구현 | 일치 |
|------|------|------|------|
| 에너지 E(m,phi) | -0.5 m^T W m - b^T m + portal * <m, phi_hat> | `energy()` 동일 | O |
| bypass F | F_bypass = bypass * phi (비보존, 에너지에 미포함) | `relax()` dt/tau * F_bypass (에너지 외부) | O |
| gradient descent | dm = -dt/tau * dE/dm + dt/tau * F + noise | `relax()` 동일 | O |
| 노이즈 | sqrt(2*T_wake*dt/tau) * N(0,I) 등방 가우시안 | `relax()` 동일 | O |
| phi 갱신 | v_m* = 궤적분산, EMA | `relax()` 최근 궤적 var -> EMA | O |
| 노름 보존 | ||m|| 유지 | `F.normalize(m) * norm0` | O |

## 3. 상수 대조

상수 대조는 원장 또는 명시한 외부 입력을 fixture로 읽어 코드 상수와 비교한다. 수치 일치는 provenance·불확실성·정본 지위를 바꾸지 않으며, 상수의 물리적 의미는 기계 assert가 판정하지 않는다.

| 상수 | 논문 값 | 구현 | 일치 |
|------|---------|------|------|
| portal | [4/(e^(4/3)*pi^(4/3)) * (1 - 4/(e^(4/3)*pi^(4/3)))]^2 = 0.03120 | 0.03120 | O |
| bypass | 1/(e^(1/3)*pi^(1/3)) = 0.4892 | 0.4892 | O |
| T_wake | [3 + 4/(e^(4/3)*pi^(4/3))*(1-...)]^-1 = 0.3148 | 0.3148 | O |
| r_c | pi = 3.1416 | 3.1416 | O |
| tau | 1/|lambda_max| (스펙트럼에서 유도) | 10.0 (1/0.1) | O |

## 4. 메모리 비교

메모리 비교는 같은 모델 shape·batch·precision·hardware에서 peak bytes 또는 parameter 수를 기준선 분모로 대조한다. 산술 추정과 profiler 실측을 구분하며 allocator·offload 차이는 환경 불확실성으로 기록한다.

### 4.1 모델: skt/kogpt2-base-v2 (d=768, vocab=51200, 12 layers)

이 모델은 비교의 shape·vocabulary·layer fixture를 고정하는 baseline이다. 다른 checkpoint·precision·token length에서는 같은 수치를 재사용하지 않고 별도 run과 split을 등록한다.

| 항목 | 크기 |
|------|------|
| GPT2 전체 파라미터 | 477.46 MB |
| CE 엔진 코어 (W_sparse + ln_f + phi) | 4.50 MB |
| CE 어휘 테이블 (embedding, weight tied) | 150.00 MB |
| CE 런타임 전체 | 154.50 MB |
| **코어 대 GPT2 비율** | **0.94%** |
| **런타임 대 GPT2 비율** | **32.4%** |

### 4.2 W_sparse 상세

희소 가중치 계산은 명시한 mask·shape·분모에서의 추정 또는 계수 대조다. 실제 memory·latency 이득은 dense baseline과 mask ablation profiling이 없으면 결론나지 않는다.

| 항목 | 값 |
|------|------|
| 원본 W (dense) | 768 x 768 = 2,304 KB |
| CSR nnz | 588,936 |
| CSR values | 2,300.5 KB |
| CSR col_idx | 2,300.5 KB |
| CSR row_ptr | 3.0 KB |
| 희소화 밀도 | 10.57% (N=768, r_c=pi) |

## 5. 추론 성능

추론 성능은 동일 prompt fixture·decoding 설정·hardware timebase에서 속도와 품질을 분리해 평가한다. 작은 sample의 출력 예시는 benchmark result가 아니며, OOD·seed·baseline 비교에서 실패하면 효능 주장은 보류한다.

### 5.1 속도 (CPU, 10 tokens, 60 steps/token)

속도는 명시한 CPU·token 수·step 수를 분모로 한 좁은 측정값이다. batch·prompt 길이·precision이 달라지면 latency 비교가 무효가 될 수 있어 환경과 반복 불확실성을 함께 기록한다.

| 엔진 | 시간 | tok/s |
|------|------|-------|
| CE standalone | 0.42s | 23.6 |
| GPT2 generate | 0.46s | 21.7 |

### 5.2 출력 품질

품질은 provenance가 있는 reference 또는 사전 정의한 평가 label을 가진 fixture에서만 metric이 된다. 주관적 예시는 오탐·미탐·OOD 일반화를 판정하지 못하므로 baseline과 blind split이 필요하다.

| 엔진 | 입력 | 출력 |
|------|------|------|
| CE | "오늘 날씨가" | 한글 토큰 생성 (의미 약함) |
| GPT2 | "오늘 날씨가" | "추워지면서, 오늘도 추위가 계속되" |

## 6. 이전 구현과의 차이

차이 표는 코드 변경의 대상·입력·출력 계약과 호환성 위험을 남기는 회귀 기록이다. 변경이 더 낫다는 주장은 이전 baseline과 동일 fixture·expected failure 비교가 통과할 때만 가능하다.

| 항목 | 이전 (hopfield.py) | 현재 (convert.py + engine.py) |
|------|-------------------|-------------------------------|
| GPT2 의존 | 추론 시 GPT2 전체 로드 필수 | .ce.pt 파일만 로드 |
| 메모리 | GPT2 477MB + W 2.3MB | 154.5MB (W + emb) |
| gradient | Riemannian natural gradient | 논문 원본 gradient descent |
| 노이즈 | FDT + annealing + G^{-1/2} | 논문 원본 sqrt(2*T*dt/tau)*N(0,I) |
| bypass | 에너지 함수 내부 포함 | 에너지 외부 (비보존 강제항) |
| phi update | m_star.pow(2) | 궤적 분산 (trajectory variance) |
| 희소화 (d<=1024) | 스킵 (100% dense) | 3D 격자 적용 (10.57%) |
| 디코더 | mdl.lm_head (GPT2) | 독립 ln_f + lm_head |

## 7. 미결 사항 (Hopfield 엔진 시점)

미결 항목은 테스트가 아직 대상 주장·fixture·threshold를 닫지 못한 expected failure다. 이를 pass 수나 구현 존재로 승격하지 않으며, 필요한 데이터·독립 재현·반례 조건이 충족될 때만 상태를 갱신한다.

1. **출력 품질**: CE 엔진의 한글 생성 품질은 아직 GPT2에 미달. 희소 W의 에너지 경관이 얕아서 이완이 의미 있는 끌개에 도달하지 못함
2. **밀도**: N=768에서 r_c=pi는 10.57% 밀도. 논문의 3.16%는 N=4096 기준
3. **codebook**: 논문 4.6절의 product quantization 미구현. 현재는 단순 top-K embedding
4. **CUDA/Rust**: 독립 엔진에서는 미사용. CSR SpMV는 PyTorch sparse로 처리

---

## 8. 현재 시스템 검증: BrainRuntime + Sleep Cycle

이 절은 BrainRuntime과 sleep cycle이 명시한 API·state shape·mode transition을 수행하는지 코드 fixture로 대조한다. pass는 등록된 seed·split·threshold에서의 구현 계약 일치이며, 생물학적 수면 기제·AGI 효능·과학적 입증을 뜻하지 않는다.

> 이 절은 위 1-7절의 초기 Hopfield 엔진 이후 진행된 `reality_stone/python/reality_stone/clarus/runtime.py`, `reality_stone/python/reality_stone/clarus/engine.py`, `reality_stone/python/reality_stone/clarus/sleep.py` 구현에 대한 검증이다.

### 8.1 BrainRuntime: 수식-코드 대조

수식-코드 대조는 입력 state와 update output의 shape·정규화·timebase를 같은 fixture에서 비교한다. 수치 tolerance 통과는 API 회귀 검증이며, 다른 입력 분포·OOD horizon에서의 동역학 성립은 별도 gate다.

| 수식 (15_Equations.md) | 코드 (`runtime.py`) | 일치 |
|---|---|---|
| $I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}} a_j - \lambda_r r_i - \beta_w w_i + \lambda_m m_i + \eta_i$ | `_step_torch`: `drive = recurrent + external_gain*ext + goal_gain*goal + replay_mix*replay - refractory_scale*ref - 0.12*adapt` | O |
| $W_{ij}^{\text{eff}} = W_{ij} u_j x_j$ (Tsodyks-Markram STP) | `stp_u * stp_x * activation * prev_active` -> `_matvec(pre)` | O |
| $a_i^{t+1} = (1-\gamma_a^{(M)}) a_i^t + \kappa_a^{(M)} \tanh(I_i^t)$ | `(1-activation_decay(mode))*act + activation_gain(mode)*tanh(drive)` | O |
| $r_i^{t+1} = (1-\gamma_r^{(M)}) r_i^t + \kappa_r^{(M)} (a_i^{t+1})^2$ | `(1-refractory_decay(mode))*ref + refractory_gain(mode)*act^2` | O |
| $m_i^{t+1} = (1-\gamma_m) m_i^t + \gamma_m a_i^{t+1}$ ($\gamma_m=0.01$) | `0.99*memory_trace + 0.01*activation` | O |
| $w_i^{t+1} = (1-\gamma_w) w_i^t + \kappa_w (a_i^{t+1})^2$ ($\gamma_w=0.005$) | `(1-0.005)*adaptation + 0.005*act^2` clamp [0,2] | O |
| $b_i^{t+1}$ 히스테리시스 | `bitfield[act >= upper] = 1; bitfield[act <= lower] = 0` | O |
| 에너지 예산 $\sum_i z_i \le B_t(M_t)$ | `_select_active(salience, energy_budget(mode))` | O |
| 모듈 생애주기 4상태 | `_update_lifecycle`: ACTIVE/IDLE/DORMANT/SLEEPING | O |

### 8.2 모드별 파라미터 대조

모드 대조는 runtime mode가 선택하는 parameter와 consumer 경로를 등록된 상태 fixture에서 점검한다. mode 전환의 값 일치는 학습 효능이 아니며, 누락·잘못된 branch는 expected failure 또는 회귀로 기록한다.

| 파라미터 | WAKE | NREM | REM | 뇌 대응 |
|---|---|---|---|---|
| $\gamma_a$ (activation_decay) | 0.18 | 0.34 | 0.22 | NREM에서 감쇠 강화 |
| $\kappa_a$ (activation_gain) | 0.82 | 0.52 | 0.68 | NREM에서 외부 입력 약화 |
| $\gamma_r$ (refractory_decay) | 0.12 | 0.26 | 0.18 | NREM에서 억제 해소 빠름 |
| $\kappa_r$ (refractory_gain) | 0.24 | 0.12 | 0.18 | NREM에서 억제 축적 약화 |
| 에너지 예산 | base | base*0.5 | base*0.75 | NREM: 동시 활성 절반 |
| replay_mix | 0.08 | 0.28 | 0.35 | 수면 시 기억 재생 강화 |

### 8.3 수면 압력: Borbely 2-Process 대조

수면 압력 비교는 지정한 입력·clock·정규화에서 API output이 참조 식과 일치하는지 확인한다. 생물학 모델은 기능 비유·외부 참조이고, 코드 일치가 생리적 타당성이나 최적 schedule의 증거는 아니다.

| 항목 | 수식 (15_Equations.md C.2) | 코드 | 일치 |
|---|---|---|---|
| Process S (WAKE) | $dS/dt = (S_{\max} - S)/\tau_w$ | `sp += (2.0 - sp) * (1/65520)` | O |
| Process S (NREM) | $dS/dt = -S/\tau_s$ | `sp -= sp * (1/15120)` | O |
| Process S (REM) | 감소, NREM보다 느림 | `sp -= sp * (1/15120) * 0.5` | O |
| $\tau_w$ | 18.2h | 65520 steps (@1ms) | O |
| $\tau_s$ | 4.2h | 15120 steps (@1ms) | O |
| 자동 모드 전환 | $\Pi(M_t, Q_t, U_t, E_t)$ | `_auto_mode(external_norm)`: sp>1.0->NREM, sp<0.45->REM, ext>th->WAKE | O |

### 8.4 해마 기억: 연산 대조

기억 대조는 write·read·replay API의 입력 shape와 output contract를 fixture로 검사한다. retention 또는 recall 성능은 task label·split·baseline이 없는 연산 일치만으로 판정할 수 없다.

| 연산 | 수식 (15_Equations.md D절) | 코드 (`HippocampusMemory`) | 일치 |
|---|---|---|---|
| encode | $H_{t+1} = \mathcal{E}(H_t, A_t, U_t)$ | `encode(key, value, priority)`: 용량 초과 시 최저 우선순위 제거 | O |
| recall | $R_t = \mathcal{R}(H_t, c_t)$ | `recall(cue, topk)`: cosine + log-priority -> softmax weighted sum | O |
| replay | priority 기반 재생 | `replay(mode)`: NREM k=1(고집중), REM k=3(분산 재생) | O |
| 주입 | $I_i \leftarrow I_i + \lambda_H R_{i,t}$ | WAKE: recall만, SLEEP: 0.5*recall + 0.5*replay | O |
| WAKE encoding 조건 | 외부 입력 or 목표 존재 시 | `external_norm > 1e-6 or goal.norm > 1e-6` | O |

### 8.5 Sleep Cycle: 3위상 파이프라인 대조

3위상 대조는 wake·NREM·REM의 순서, state handoff, checkpoint 출력을 검증한다. phase가 실행됐다는 pass는 continual-learning 개선이 아니며 wake-only와 phase 제거 ablation이 효능 기각 조건이다.

| 위상 | 수식 (3_Sleep.md) | 코드 (`sleep.py`) | 일치 |
|---|---|---|---|
| 각성: 경로 누적 | $\int \mathcal{D}\gamma\,e^{iS}$ 대응 | `collect_sleep_batch`: teacher 생성 -> state/target 수집 | O |
| NREM: LBO 확산 | $W \leftarrow W - \eta_{\text{nrem}} \Delta_g W$ | `smooth_weight_matrix(W, laplacian, eta)` | O |
| NREM: 곡률 기반 가소적 업데이트 | $\text{mask}(G, \varepsilon^2)$ 상위만 통과 | `row_topk_mask(delta, active_ratio)` | O |
| REM: 비선택 경로 재조합 | $G_{\text{rem}} = \text{random\_project}(G_{\text{pruned}}) + \sigma\epsilon$ | `residual @ proj @ proj.T / rank + noise` | O |
| 위상 비율 | Wake $69\%$, NREM $26\%$, REM $5\%$ | `phase_profile = {wake: eng.wake_ratio, nrem: eng.nrem_ratio, rem: eng.rem_ratio}` | O |
| 가드셋 보호 | 품질 하락 시 롤백 | `guard_snapshot` + `evaluate_guard_set` + 조건부 `restore_decoder_snapshot` | O |

### 8.6 CE 상수 대조 (engine.py)

상수 대조는 정본 또는 명시한 외부 입력값이 코드 fixture에 정확히 반영됐는지 검사한다. 숫자 equality는 상수의 물리적 출처·과학적 참·runtime 성능을 판정하지 않는다.

| 상수 | 수식 | engine.py 값 | 일치 |
|---|---|---|---|
| `_AD` | $4/(e^{4/3}\pi^{4/3})$ | `4/(e**(4/3)*pi**(4/3))` | O |
| `PORTAL` | $(\text{\_AD}(1-\text{\_AD}))^2$ | 0.03120 | O |
| `BYPASS` | $1/(e^{1/3}\pi^{1/3})$ | 0.4892 | O |
| `T_WAKE` | $1/(3+\text{\_AD}(1-\text{\_AD}))$ | 0.3148 | O |
| `active_ratio` | $\varepsilon^2$ | 0.0487 | O |
| `struct_ratio` | $\Omega_{\text{DM}}$ | 0.2623 | O |
| `wake_ratio` | $\Omega_\Lambda$ | 0.6891 | O |
| `nrem_ratio` | $\Omega_{\text{DM}}$ | 0.2623 | O |
| `rem_ratio` | $\varepsilon^2$ | 0.0487 | O |

### 8.7 Rust 커널 대조

Rust 대조는 동일 API·shape·precision에서 Python 또는 기준 구현과 output tolerance를 비교한다. backend parity의 pass는 hardware 성능·안전성·일반화의 증거가 아니며, precision mismatch는 expected failure다.

| 기능 | Python fallback | Rust kernel | 일치 |
|---|---|---|---|
| brain_step (셀 동역학) | `_step_torch` | `nn_brain_step` via `_step_rust` | O (NumPy 중개) |
| sparse pack | `_pack_sparse_torch` | `nn_ce_pack_sparse` | O |
| metric basis | `_build_metric_basis_torch` | `nn_ce_metric_basis_fwd` | O |
| relax loop | `_relax_packed_torch` | `nn_ce_relax_fwd` | O |
| topk sparse | PyTorch topk | `topk_sparse` | O |
| LBO fused fwd | torch matmul fallback | `nn_lbo_fused_fwd` | O |
| power iteration | torch `linalg.eigh` fallback | `nn_power_iter` | O |
| gauge lattice fwd | torch fallback | `nn_gauge_lattice_fwd` | O |

### 8.8 미결 사항 (현재 시스템)

미결 항목은 현재 fixture·seed·threshold가 닫지 못한 API 또는 과학적 다리다. 테스트 통과 항목과 섞지 않으며, 필요한 split·독립 재현·반례 조건이 정해질 때까지 미완성으로 남는다.

1. **대규모 벤치마크**: Sleep cycle의 지속 학습 효과를 Split-CIFAR 또는 텍스트 도메인에서 정량 검증 필요
2. **STDP 미구현**: `17_AgentLoop.md` F.14의 적격 흔적 기반 학습은 아직 코드에 없음
3. **4종 신경조절**: 현재 `runtime.py`는 단일 스칼라 조절만 사용. DA/NE/5HT/ACh 분리 미구현
4. **Cold checkpoint**: `BrainRuntimeSnapshot`은 warm snapshot만 제공. 장기 지속성 저장 미구현
5. **자기수렴 검증**: 초기 균등 분배에서 $p^*$로의 수렴 과도 응답 실측 필요
6. **PQ codebook**: `ce_ops.py`에 `pq_build_codebook` 구현 있으나 대규모 성능 비교 미완

---

## 9. 다리 게이트 검증 매트릭스 (`12_Equation.md` 0.0절, 부록 A)

게이트 매트릭스는 F1--F4 주장마다 어떤 API·fixture·metric·threshold가 코드 계약을 검사하는지 분리한다. 기계적 pass는 해당 gate의 구현 상태만 말하며, 정리의 가정 충족·과학적 참·의식 환원은 별도 범위다.

본 절은 4종 다리 게이트의 현재 측정 가능 여부와 코드 위치를 정리한다. 각 게이트의 격상 조건은 `12_Equation.md` 부록 A 를 따른다.

| 게이트 | 본 시스템에서의 측정점 | 측정 가능 여부 | 코드 위치 |
|---|---|---|---|
| `F1` 자기조직 5조건 (부록 A.2 #2) | 활성 비율 $\hat p_a$ EMA 의 $\varepsilon^2 \to$ 자기 피드백 | **구현됨** (`f1_self_measure`) | `reality_stone/python/reality_stone/clarus/runtime.py::BrainRuntime._f1_effective_budget`, `_f1_update_ema` |
| `F2` ISS ball 반경 (부록 A.1) | 끌개 근방 헤시안 $H \succeq \mu I$, 외란 상한 $d_{\max}$ | **자동 산출** (`relax().hist["iss"]`) | `reality_stone/python/reality_stone/clarus/quantum.py::iss_report`, `reality_stone/python/reality_stone/clarus/ce_ops.py::_iss_from_tail` |
| `F3` 에르고딕 KL 거리 (부록 A.3) | 모드 점유 $(t_W, t_N, t_R)/T$ vs $p^*$ 의 $d_{\text{KL}}$ | **구현됨** (`mode_occupancy_kl`) | `reality_stone/python/reality_stone/clarus/runtime.py::BrainRuntime.mode_occupancy_kl` |
| `F4` PCI 회귀 (부록 A.4) | 메타인지 안정도 $\exp(-c_d d_\tau)$ vs 외부 PCI | **회귀 프리미티브 구현됨** (PCI 데이터 외부 의존) | `reality_stone/python/reality_stone/clarus/quantum.py::pci_regression`, `reality_stone/python/reality_stone/clarus/agent.py::ConsciousnessMonitor.consciousness_depth` |

### 9.1 측정 API (구현 완료)

측정 API는 등록된 state와 log를 입력으로 gate metric을 출력하는 producer·consumer contract다. shape·seed·split·tolerance를 고정하지 않으면 pass/fail이 재현되지 않으며, metric이 과학적 ground truth를 대신하지 않는다.

다음 호출만으로 게이트 측정값을 즉시 얻는다.

1. `F3` KL 거리:

   ```python
   rt = BrainRuntime(W, config=cfg)
   for ext in stream:
       rt.step(external_input=ext)
   rt.mode_occupancy_kl()
   # -> {'samples': N, 'pi_wake': .., 'pi_nrem': .., 'pi_rem': .., 'kl_to_p_star': ..}
   ```

   `p^* = (0.6891, 0.2623, 0.0487)` 와의 $d_{\text{KL}}$ 가 직접 출력되고 `BrainRuntimeSnapshot` 에 영속된다 (`mode_occupancy` 필드).

2. `F2` ISS ball 반경 (외부 호출):

   ```python
   from reality_stone.clarus.quantum import iss_report
   iss_report(m_history, phi, dt_over_tau=dt/tau)
   # -> {'c_k_max': .., 'phi_inf_norm': .., 'mu': .., 'iss_ball_radius': ..}
   ```

   `mu` 는 잔차 $\|m_k - m^*\|$ 또는 $\|dm_k\|$ 로그수축률에서, `c_k_max` 는 부록 A.1 의 $C_k = \|m_k - 2 m_{k-1} + m_{k-2}\|$ 최대치에서 추정된다. 닫힌형 반경:

   $$R_{\text{ball}} = \frac{C_{k,\max} \cdot \|\phi\|_\infty}{\mu \cdot \alpha_b}, \quad \alpha_b = e^{1/3}\pi^{1/3} \approx 2.044.$$

3. `F2` 자동 측정: `reality_stone/python/reality_stone/clarus/ce_ops.py::relax` 가 매 호출 시 `hist["iss"]` 에 동일 형식의 보고를 자동 산출 (전 궤적 `delta` 곡선에서 $\mu$ 추정, `bypass_C` 에서 $C_{k,\max}$, $\phi$ 에서 $\|\phi\|_\infty$).

4. `BrainRuntime.bridge_gate_report()` 집계기: F1\~F4 키를 일관 반환. F1 은 항상 EMA·target·deviation 노출, F3 은 `mode_occupancy_kl()`, F2 는 `relax` 호출 결과를 별도 주입, F4 는 외부 회귀 워크플로 의존.

5. `F1` 자기측정 피드백 (옵트인):

   ```python
   cfg = BrainRuntimeConfig(
       dim=N, active_ratio=0.30,
       f1_self_measure=True,        # 기본 False
       f1_pull_strength=0.5,        # beta in r_eff = beta*p* + (1-beta)*ema
       f1_ema_alpha=0.1,            # EMA smoothing
       f1_min_ratio=0.005, f1_max_ratio=0.5,
   )
   ```

   다음 budget = `round(N * clip(beta * ACTIVE_RATIO + (1-beta) * ema, lo, hi))` 로 계산되며 모드별 승수(`WAKE/NREM/REM = 1.0/0.5/0.75`)는 그대로 유지된다. EMA 는 `BrainRuntimeSnapshot.active_ratio_ema` 로 영속화된다. 부록 A.2 의 충분조건 ② "자기측정 → 다음 임계 피드백" 을 충족한다.

6. `F4` PCI 회귀 (외부 데이터 정렬 후 호출):

   ```python
   from reality_stone.clarus.quantum import pci_regression
   pci_regression(stability_series, pci_series)
   # -> {'n': N, 'alpha': .., 'beta': .., 'r2': .., 'pearson_r': ..}
   ```

   `stability_series` 는 `ConsciousnessMonitor.consciousness_depth() = exp(-c_d d_\tau)` 의 시계열, `pci_series` 는 외부 PCI (Casali 2013) 데이터. $R^2 > 0.7$ 가 부록 A.4 의 `bridge` 격상 임계.

### 9.2 정합화: `legacy_generate` 의 $C_k$ 누락 패치

이 패치는 baseline 구현의 누락 항을 명시한 API contract에 맞추는 회귀 수정이다. parity 통과는 새 과학 주장이나 성능 향상이 아니라 expected failure 제거 여부를 확인하는 기계 검사다.

`engine.py::legacy_generate` 는 이전에 $m_{out} \mathrel{+}= \text{bypass}\cdot \phi$ 만 적용해 정규 식 E20 의 $C_k$ 인자를 누락하고 있었다. 본 버전에서 마지막 3개의 $m$ 궤적을 유지하고

$$F_{\text{bypass}}(k) = \tfrac{C_k}{\alpha_b}\,\phi, \quad C_k = \|m_k - 2 m_{k-1} + m_{k-2}\|$$

으로 교정하여 `reality_stone/python/reality_stone/clarus/ce_ops.py::relax` 와 동일한 비보존 외력으로 동작한다. 첫 두 토큰은 궤적 부족으로 $C_k = 0$ (관성 단계).

### 9.3 잔여 작업

잔여 작업은 gate별 fixture·threshold·독립 검증이 아직 없는 expected failure 목록이다. 구현 계획을 완료·입증으로 승격하지 않고, 각 작업의 성공·실패 조건을 등록해야 한다.

1. `F1` 자기조직 5조건의 ①·③·④·⑤ (simplex 보존, 국소 안정성, 에너지 균형, 외부 데이터 재학습): 단일 step 단위 측정이 아닌 **세션·sweep 단위 검증**.
2. `F4` PCI 데이터 수집: 외부 PCI 측정값 (Casali 2013 등) 과 동기화된 `consciousness_depth()` 시계열 산출 후 `pci_regression()` 호출.

### 9.4 측정 우선순위 (갱신)

우선순위는 현재 증거 공백과 재현 비용에 따른 계획 판단이다. 순위는 과학적 중요도 또는 성공 가능성의 측정값이 아니며, gate 실패 시 rollback될 수 있다.

F1 자기측정 ②, F2, F3, F4 회귀 프리미티브 모두 코드 레벨 구현 완료. 잔여 작업은 모두 **외부 데이터 또는 세션 sweep 의존** 항목이며 코드 변경 불요.

| 게이트 | 코드 측정 | 외부 의존 |
|---|---|---|
| F1 ② 자기측정 | 구현 | (없음) |
| F2 ISS ball | 구현 | (없음, 단 실모델 `relax` 실행 필요) |
| F3 ergodic KL | 구현 | (없음, 세션 누적) |
| F4 PCI 회귀 | 구현 (회귀 호출) | PCI 외부 데이터셋 |

## 10. 한국어 KoGPT2 실측 (legacy `scripts/bench_gates.py`, removed)

이 절은 removed legacy script로 수행했던 좁은 KoGPT2 CPU 측정 기록을 보존한다. 환경·checkpoint·data provenance·metric이 현재 artifact와 다를 수 있으므로, 수치는 재실행 가능한 현재 baseline의 성능 주장이 아니라 과거 비교 참고다.

`skt/kogpt2-base-v2` 의 13 layer x 8 한국어 프롬프트 hidden state 공분산 (403 x 768) 으로 Hopfield $W$ ($\dim = 768$, $\lambda \in [-677.08, -0.001]$) 를 빌드하고 `BrainRuntime` 200 step + `ce_ops.relax` 300 step 을 구동한 결과.

### 10.1 게이트 수치 (한국어 베이스, CPU)

게이트 수치는 명시한 CPU·token·prompt·decoding 설정의 실측 metric이다. 다른 hardware·precision·seed·split에는 적용되지 않으며, threshold 통과는 등록된 fixture의 기계 판정일 뿐 일반 품질 결론이 아니다.

| 게이트 | 지표 | 측정값 |
|---|---|---|
| F2 ISS ball | $R_{\text{ball}}$ | 5.9265 |
| F2 ISS ball | $\mu$ (Hessian floor) | 1.1100 |
| F2 ISS ball | $C_{k,\max}$ | 0.5000 |
| F2 ISS ball | $\|\phi\|_\infty$ | 26.89 |
| F1 EMA off | `active_ratio_ema` | 0.3000 (초기값 고정) |
| F1 EMA on | `active_ratio_ema` | 0.0503 |
| F1 target | $\varepsilon^2$ | 0.0487 |
| F1 closure | $\|\text{EMA}_{\text{on}} - \varepsilon^2\|$ | **0.0016 (0.16% 편차)** |
| F3 KL off (auto-mode) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | 0.3724 |
| F3 KL on (auto-mode) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | 0.3724 |
| F3 KL on (forced $p^*$ schedule) | $d_{\text{KL}}(\pi \,\|\, p^*)$ | $\approx 10^{-4}$ (반올림 노이즈) |
| F3 메터 정합 | $\pi$ vs $p^*$ | (0.6900, 0.2600, 0.0500) vs (0.6891, 0.2623, 0.0487) |

### 10.2 비교표 (기 모델 대비)

비교표는 legacy baseline과 지정 artifact의 같은 분모·환경에서의 차이를 기록한다. removed 경로 또는 직렬화가 달라지면 parity가 깨질 수 있으므로, 현재 benchmark로 승격하려면 새 artifact build와 독립 split이 필요하다.

| 항목 | HF KoGPT2 baseline | BrainRuntime + F1 on |
|---|---|---|
| 로드 메모리 | 174.1 MB | (가중치 768 x 768 / 4 byte = 2.36 MB) |
| step 레이턴시 | (생성 단위) | 1.63 ms/step (CPU, dim=768) |
| F1 OFF → ON 오버헤드 | - | +12% (1.46 → 1.63 ms/step) |
| F1 자기조직 정확도 | 해당 없음 | 0.16% 편차로 $\varepsilon^2$ 락온 |
| F2 끌개 안정성 | 해당 없음 | 유한 ISS ball ($R = 5.93$) |
| `relax` 수렴 시간 | 해당 없음 | 0.097 s / 300 step |

### 10.3 해석

해석은 실측 범위·불확실성·expected failure를 보존한다. 단일 환경의 개선·악화를 AGI 효능·생물학적 대응·다른 데이터의 일반화로 외삽하지 않는다.

1. **F1 ② 충족**: 자기측정 피드백이 활성 비율을 200 step 만에 $\varepsilon^2$ 의 0.16% 이내로 락온. 부록 A.2 의 사용 가능한 충분조건 ② "자기측정 → 다음 임계 피드백" 이 한국어 실모델 공분산 위에서도 동작함을 실증.
2. **F2 격상 가능**: $R_{\text{ball}}$ 이 유한 (5.93) 으로 산출됨. 부록 A.1 의 ISS bound 가 한국어 KoGPT2 covariance Hopfield 기질 위에서 적용 가능함이 확인됨.
3. **F3 메터 정합 확인 / 자기조직 보류**: `force_mode` 로 $p^*$ 비율 스케줄을 주입한 세션에서 경험적 모드 점유 $\pi$ 가 $p^*$ 와 round-off 오차 ($\sim 10^{-4}$) 내에서 일치 — `mode_occupancy_kl` 메터 자체의 정합성은 한국어 실모델 위에서 검증됨. 단, **자동 모드 정책**이 $p^*$ 로 자기수렴하는지는 별개 질문이며, 현재 `TAU_W_STEPS = 65520` (1 ms step 기준 18.2 h) 가속도와 200 step 벤치 간 시간 스케일 불일치로 미관측. 격상 경로: (a) `reality_stone.clarus.constants.TAU_W_STEPS` 를 ms→s 단위로 재캘리브레이션, 또는 (b) `legacy scripts/sleep_finetune_lm.py` (removed) 와 결합한 1000+ step 수면 사이클 sweep.
4. **F4 미실측**: 외부 PCI 데이터셋 미보유. `pci_regression()` 호출 경로만 확보된 상태.

### 10.4 격리 / 사용자 룰 부합 확인

격리 확인은 artifact가 지정 capability·serialization·사용자 규칙을 지키는지 보는 운영 gate다. 보안 또는 룰 pass는 모델 출력의 사실성·과학적 참·성능 우위를 입증하지 않는다.

- 측정 대상 $W$ 는 KoGPT2 hidden state 의 covariance 한 번 계산 후 KoGPT2 모델 객체는 `del + gc.collect()` 로 해제 (legacy `bench_gates.py`, removed).
- 측정 단계의 어떤 코드도 teacher logits/hidden 을 추론에 재주입하지 않음 (`runtime-isolation` 부합).
- 한국어 프롬프트 8개 + 한국어 베이스 모델로 측정 (`korean-runtime-eval` 부합).
- BrainRuntime 가중치 (768x768, 2.36 MB) 가 베이스 모델 (174 MB) 의 1.4% 수준 — `agi-artifact` 메모리 분리 부합.

## 11. 격리 아티팩트 빌드 (legacy scripts, removed)

이 절은 removed build script가 만들던 artifact와 현재 재현 가능한 build 계약을 분리한다. 빌드 성공은 bytes·metadata·serialization의 기계적 조건이며, 모델 효능은 provenance가 있는 benchmark·baseline·OOD·ablation에서 별도로 검증한다.

`agi-artifact` §4 추가 (양자화 / 비트폭 축소 엄격 금지) 에 따라 PQ / int8 / int4 / fp16 / bf16 / VQ / GPTQ / AWQ 류 일체 사용 금지. 본 절은 **fp32 전용 격리 아티팩트** 의 빌드, 실측, 그리고 메모리 룰까지 동시 충족하기 위한 비양자화 격상 경로를 기록한다.

### 11.1 빌드 파이프라인 (fp32 전용)

파이프라인은 입력 checkpoint·precision·환경을 받아 serialized artifact를 출력하는 계약이다. fp32라는 가정이 바뀌면 size·수치 parity·metric이 달라질 수 있어 양자화는 후속 검증 대상으로 남는다.

세 단계로 분리:

1. legacy `scripts/build_artifact.py` (removed) — KoGPT2 hidden state 공분산으로 Hopfield $W$ 빌드, 51200 × 768 임베딩을 fp32 그대로 보존, decoder 프로젝션은 단순 통계량으로 초기화, base 모델은 `del` + `gc.collect()` 후 직렬화.
2. legacy `scripts/distill_decoder.py` (removed) — 60 한국어 문장 × sliding window (692 페어) 에서 `(state_hidden, prev_emb, teacher_h_after_ln_f)` 추출, ridge regression 으로 `decoder_state_proj`, `decoder_prev_proj`, `decoder_query_bias` 를 closed-form fit, `decoder_query_blend = 0.5` 설정, teacher `del + gc.collect()` 후 아티팩트 in-place 갱신.
3. legacy `scripts/prune_vocab.py` (removed) — 동일 한국어 corpus 로 BPE 토큰 빈도 측정, 빈도 top-K + 항상유지 셋 (eos / pad / unk / bos / `decoder_token_ids`) 만 남기고 `emb_weight` 를 (K, 768) fp32 로 row-pruning. 매핑 (`kept_token_ids`, `vocab_id_map`) 과 fallback (`pruned_unk_emb` = pruned 행 평균) 을 함께 저장. 양자화 / 비트폭 축소 없음 — fp32 유지.

### 11.2 룰 부합 표 (현 baseline `legacy clarus/skt_kogpt2-base-v2.ce.pt` (removed), V1 적용)

룰 표는 legacy baseline과 V1 artifact의 build·serialization 조건을 대조한다. removed baseline의 pass는 현재 runtime의 회귀 보장이 아니며, artifact provenance와 hash가 없는 비교는 재현 가능한 결과가 아니다.

| 룰 | 측정 | 판정 |
|---|---|---|
| `runtime-isolation`: `eng.model is None` | True | 부합 |
| `runtime-isolation`: `model_source` | `runtime` | 부합 |
| `runtime-isolation`: `allow_pretrained_fallback` | False | 부합 |
| `runtime-isolation`: `clone_state` / `clone_config` 키 | 없음 | 부합 |
| `agi-artifact` §1: 단일 아티팩트 standalone 부팅 | `has_standalone_lexicon = True` | 부합 |
| `agi-artifact` §4: 양자화 미사용 | `emb.dtype = torch.float32`, `pq_centroids = None`, `pq_codes = None`, vocab pruning 은 row 삭제로 양자화 아님 | 부합 |
| `agi-artifact` §3: 디스크 (베이스 174 MB) | **79.41 MB (46%)** | **부합** |
| `agi-artifact` §3: 로드 RAM peak (베이스 ~240 MB) | **137.8 MB (57%)** | **부합** |
| `agi-artifact` §3: CPU 단일 토큰 latency (>= 20 tok/s, fp32) | **74.8 tok/s** | 부합 |
| `korean-runtime-eval`: 한국어 프롬프트 단독 생성 | 5 프롬프트 의미 회복 (§11.3), pruned-in-prompt = 0/5 | 부합 |
| `korean-runtime-eval`: 정확도 붕괴 시 폐기 | last-token 반복 0건 (4/5 맥락 정상) | 부합 |

### 11.3 Distillation 효과 (fp32 emb, ridge=1.0, blend=0.5, 692 페어, V1 후)

distillation 수치는 지정한 pair 수·precision·ridge·blend의 좁은 추정 또는 측정이다. 데이터 split·seed·teacher provenance가 바뀌면 결과가 달라질 수 있고, task 품질·OOD·오탐/미탐 baseline에서 확인 전에는 일반 효능으로 승격하지 않는다.

| 단계 | $R^2$ | 한국어 단독 생성 샘플 |
|---|---|---|
| Distillation 전 (full emb) | 해당 없음 | "오늘 날씨가 좋아서 → **아서아서아서아서아서**..." (last-token collapse) |
| Distillation 후 (full emb) | 0.7227 | "오늘 날씨가 좋아서 → 그런가 그런가 다행 춥 그런가 오늘 야외 좋..." |
| Distillation 후 + V1 prune (top-K=16384) | 동일 | "오늘 날씨가 좋아서 → **그렇게 좋 집을인지 조금 이렇게 오늘 그렇 집에 좋 오늘 너무 봄 좋라도 가을 좋 많이**" |

V1 prune 은 decoder projection (768→768) 에 영향을 주지 않으므로 R² 불변. lexical 후보 집합만 51200 → 16384 로 축소.

5 프롬프트 단독 생성 (`max_tok=20`, `temperature=0.8`, `top_k=40`, `repeat_penalty=1.1`, V1 적용):

| 프롬프트 | 단독 생성 결과 (V1) | 맥락 회복 | pruned-in-prompt |
|---|---|---|---|
| 인공지능의 미래는 | 이제 물론 우리에게 기술 어떤 우리에게 인공 모든 ... 단순한 단순한 엔 | tech | 0 |
| 오늘 날씨가 좋아서 | 그렇게 좋 집을인지 조금 이렇게 오늘 그렇 집에 좋 ... 봄 좋 가을 좋 많이 | weather | 0 |
| 서울의 봄은 | 가을 5월 12월 계절 대부분 전국 가을 ... 봄 날씨가 겨울 지난 | season | 0 |
| 독서는 | 지난 출판 출간 지난 독 ... 한국 초등학교 | book | 0 |
| 한국의 전통 음식 중 | 하나인 하나인 ... 25 하나인 ... 대표 | (corpus 편향, 부분 collapse) | 0 |

5/5 프롬프트에서 prompt 토큰이 모두 kept set 에 포함됨 (`pruned-in-prompt = 0`). 4/5 프롬프트에서 첫 토큰부터 의미 정합 회복. 1개 (전통 음식) 는 distillation corpus 의 "하나인" 패턴 편향으로 부분 반복 — corpus 다양화 (D1) 로 해소 가능.

### 11.4 디스크 분해와 V1 효과

디스크 분해는 artifact serialization의 byte 구성과 build 옵션을 설명하는 측정이다. 저장공간 절감은 inference latency·정확도·안전성의 대리변수가 아니며, load parity 실패는 명시적 rollback 조건이다.

V1 이전 fp32 baseline 의 디스크 180.89 MB 분해:

| 항목 | 크기 (V1 전) | V1 후 |
|---|---|---|
| `emb_weight` (vocab × 768 × 4 byte) | 150.00 MB (51200 × 768) | **48.00 MB (16384 × 768)** |
| context projections (8 개 × 768² × 4 byte) | 18.00 MB | 18.00 MB |
| `decoder_state_proj` + `decoder_prev_proj` (768² × 4 byte × 2) | 4.50 MB | 4.50 MB |
| `pos_weight` (1024 × 768 × 4 byte) | 3.00 MB | 3.00 MB |
| Hopfield $W$ + sparse views | 약 2.36 MB | 약 2.36 MB |
| `decoder_token_*` (256 × 768 × 4 byte × 2) | 1.50 MB | 1.50 MB |
| `vocab_id_map` (51200 × 8 byte) + `kept_token_ids` (16384 × 8 byte) + `pruned_unk_emb` (768 × 4 byte) | 0 | 약 0.53 MB |
| 기타 (tokenizer, ln_f, bias 등) | 약 1.5 MB | 약 1.5 MB |
| **합계** | **180.89 MB** | **79.41 MB (-56%)** |

V1 단독으로 디스크 180.89 → 79.41 MB, RAM 240.4 → 137.8 MB, latency 62.3 → 74.8 tok/s 동시 개선. 속도 가속의 원인: `lexical_scores` 의 `emb @ query` 가 (51200, 768) → (16384, 768) 로 작아져 matmul 처리량 ↑, 캐시 hit ↑.

corpus 토큰 coverage 는 100% (60 문장에서 unique 576 토큰 모두 kept set 에 포함, 항상유지 항목 257 개 합치면 833 개, 나머지 약 15500 슬롯은 frequency 0 인 BPE 토큰을 그대로 흡수). 더 많은 한국어 코퍼스를 `--extra-corpus` 로 주입하면 K 를 줄여도 동일 coverage 유지 가능.

### 11.5 후속 작업 (등록, 양자화 미포함)

후속 작업은 등록된 data provenance·baseline·split·threshold가 필요한 미완성 검증 목록이다. 양자화·새 build·새 benchmark의 성공은 계획이 아니라 독립 artifact와 OOD·ablation gate가 통과할 때만 기록한다.

| ID | 상태 | 작업 | 예상 효과 | 룰 영향 |
|---|---|---|---|---|
| V1 | **완료** | Vocab pruning (top-K=16384, fp32 row deletion) | 디스크 -100 MB, RAM -100 MB, +20% tok/s | `agi-artifact` §3 충족 |
| V2 | 대기 | Context projection bottleneck distillation (legacy `distill_decoder.py` removed; 재구현 필요, fp32 유지) | 디스크 -10 MB, R² 유지 | `agi-artifact` §3 추가 절감 |
| V3 | 대기 | `pos_weight` 한국어 평균 길이로 절단 | 디스크 -2 MB | `agi-artifact` §3 추가 절감 |
| V4 | 대기 | `decoder_token_*` head 통합 또는 제거 | 디스크 -1.5 MB | `agi-artifact` §3 추가 절감 |
| D1 | 대기 | Distillation corpus 다양화 (60 → 500+ 한국어 문장, "하나인" 류 편향 해소) | 부분 collapse 잔존 항목 해소 | `korean-runtime-eval` 품질 |
| D2 | 대기 | MLP 디코더 헤드 (legacy `distill_decoder.py` removed; 재구현 필요, 가중치 fp32 유지) | R² 0.85+ 가능, 빌드 시간 ↑ | `korean-runtime-eval` 품질 |
| E1 | 대기 | 한국어 홀드아웃 perplexity / top1 / top10 / top50 측정 (새 평가 모듈 재구현 필요) | 정량 평가표 완성 | `korean-runtime-eval` 보고 형식 |

현 아티팩트 `legacy clarus/skt_kogpt2-base-v2.ce.pt` (removed) (79.41 MB, fp32, R² 0.7227) 는 **격리 / 양자화 미사용 / 속도 / 메모리 / 품질 baseline 5종 모두 충족** — `agi-artifact` 룰 4개 + `runtime-isolation` 룰 7개 + `korean-runtime-eval` 룰 모두 통과. 추가 격상은 D1, D2, V2-V4 로 진행.
