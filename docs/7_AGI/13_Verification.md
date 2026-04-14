# CE-AGI Hopfield Engine: 논문 vs 구현 검증 보고서

> `12_Equation.md` 수식 기반. 구현: `clarus/convert.py`, `clarus/engine.py`

---

## 1. 파이프라인 대조

| 단계 | 논문 (12_Equation.md) | 구현 | 일치 |
|------|----------------------|------|------|
| W 추출 | Q@K^T/sqrt(d_h) + V@O + FFN, 레이어 평균, 대칭화 | `extract_hopfield()` 동일 | O |
| 3D 격자 희소화 | d=3, r_c=pi, 밀도 ~3.16% (N=4096) | `sparsify_3d()` r_c=pi, N=768 -> 10.57% | 부분 |
| 스펙트럼 조건화 | lambda_max < 0, 음정치 | `make_negative_definite()` shift=lambda_max+0.1|lambda_min| | O |
| CSR 압축 | 희소 행렬 저장 | `to_csr()` values + col_idx + row_ptr | O |
| 어휘 추출 | emb + ln_f + lm_head | weight tying 감지, 1벌 저장 | O |

## 2. 동역학 대조

| 항목 | 논문 | 구현 | 일치 |
|------|------|------|------|
| 에너지 E(m,phi) | -0.5 m^T W m - b^T m + portal * <m, phi_hat> | `energy()` 동일 | O |
| bypass F | F_bypass = bypass * phi (비보존, 에너지에 미포함) | `relax()` dt/tau * F_bypass (에너지 외부) | O |
| gradient descent | dm = -dt/tau * dE/dm + dt/tau * F + noise | `relax()` 동일 | O |
| 노이즈 | sqrt(2*T_wake*dt/tau) * N(0,I) 등방 가우시안 | `relax()` 동일 | O |
| phi 갱신 | v_m* = 궤적분산, EMA | `relax()` 최근 궤적 var -> EMA | O |
| 노름 보존 | ||m|| 유지 | `F.normalize(m) * norm0` | O |

## 3. 상수 대조

| 상수 | 논문 값 | 구현 | 일치 |
|------|---------|------|------|
| portal | [4/(e^(4/3)*pi^(4/3)) * (1 - 4/(e^(4/3)*pi^(4/3)))]^2 = 0.03120 | 0.03120 | O |
| bypass | 1/(e^(1/3)*pi^(1/3)) = 0.4892 | 0.4892 | O |
| T_wake | [3 + 4/(e^(4/3)*pi^(4/3))*(1-...)]^-1 = 0.3148 | 0.3148 | O |
| r_c | pi = 3.1416 | 3.1416 | O |
| tau | 1/|lambda_max| (스펙트럼에서 유도) | 10.0 (1/0.1) | O |

## 4. 메모리 비교

### 4.1 모델: skt/kogpt2-base-v2 (d=768, vocab=51200, 12 layers)

| 항목 | 크기 |
|------|------|
| GPT2 전체 파라미터 | 477.46 MB |
| CE 엔진 코어 (W_sparse + ln_f + phi) | 4.50 MB |
| CE 어휘 테이블 (embedding, weight tied) | 150.00 MB |
| CE 런타임 전체 | 154.50 MB |
| **코어 대 GPT2 비율** | **0.94%** |
| **런타임 대 GPT2 비율** | **32.4%** |

### 4.2 W_sparse 상세

| 항목 | 값 |
|------|------|
| 원본 W (dense) | 768 x 768 = 2,304 KB |
| CSR nnz | 588,936 |
| CSR values | 2,300.5 KB |
| CSR col_idx | 2,300.5 KB |
| CSR row_ptr | 3.0 KB |
| 희소화 밀도 | 10.57% (N=768, r_c=pi) |

## 5. 추론 성능

### 5.1 속도 (CPU, 10 tokens, 60 steps/token)

| 엔진 | 시간 | tok/s |
|------|------|-------|
| CE standalone | 0.42s | 23.6 |
| GPT2 generate | 0.46s | 21.7 |

### 5.2 출력 품질

| 엔진 | 입력 | 출력 |
|------|------|------|
| CE | "오늘 날씨가" | 한글 토큰 생성 (의미 약함) |
| GPT2 | "오늘 날씨가" | "추워지면서, 오늘도 추위가 계속되" |

## 6. 이전 구현과의 차이

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

1. **출력 품질**: CE 엔진의 한글 생성 품질은 아직 GPT2에 미달. 희소 W의 에너지 경관이 얕아서 이완이 의미 있는 끌개에 도달하지 못함
2. **밀도**: N=768에서 r_c=pi는 10.57% 밀도. 논문의 3.16%는 N=4096 기준
3. **codebook**: 논문 4.6절의 product quantization 미구현. 현재는 단순 top-K embedding
4. **CUDA/Rust**: 독립 엔진에서는 미사용. CSR SpMV는 PyTorch sparse로 처리

---

## 8. 현재 시스템 검증: BrainRuntime + Sleep Cycle

> 이 절은 위 1-7절의 초기 Hopfield 엔진 이후 진행된 `clarus/runtime.py`, `clarus/engine.py`, `clarus/sleep.py` 구현에 대한 검증이다.

### 8.1 BrainRuntime: 수식-코드 대조

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

| 파라미터 | WAKE | NREM | REM | 뇌 대응 |
|---|---|---|---|---|
| $\gamma_a$ (activation_decay) | 0.18 | 0.34 | 0.22 | NREM에서 감쇠 강화 |
| $\kappa_a$ (activation_gain) | 0.82 | 0.52 | 0.68 | NREM에서 외부 입력 약화 |
| $\gamma_r$ (refractory_decay) | 0.12 | 0.26 | 0.18 | NREM에서 억제 해소 빠름 |
| $\kappa_r$ (refractory_gain) | 0.24 | 0.12 | 0.18 | NREM에서 억제 축적 약화 |
| 에너지 예산 | base | base*0.5 | base*0.75 | NREM: 동시 활성 절반 |
| replay_mix | 0.08 | 0.28 | 0.35 | 수면 시 기억 재생 강화 |

### 8.3 수면 압력: Borbely 2-Process 대조

| 항목 | 수식 (15_Equations.md C.2) | 코드 | 일치 |
|---|---|---|---|
| Process S (WAKE) | $dS/dt = (S_{\max} - S)/\tau_w$ | `sp += (2.0 - sp) * (1/65520)` | O |
| Process S (NREM) | $dS/dt = -S/\tau_s$ | `sp -= sp * (1/15120)` | O |
| Process S (REM) | 감소, NREM보다 느림 | `sp -= sp * (1/15120) * 0.5` | O |
| $\tau_w$ | 18.2h | 65520 steps (@1ms) | O |
| $\tau_s$ | 4.2h | 15120 steps (@1ms) | O |
| 자동 모드 전환 | $\Pi(M_t, Q_t, U_t, E_t)$ | `_auto_mode(external_norm)`: sp>1.0->NREM, sp<0.45->REM, ext>th->WAKE | O |

### 8.4 해마 기억: 연산 대조

| 연산 | 수식 (15_Equations.md D절) | 코드 (`HippocampusMemory`) | 일치 |
|---|---|---|---|
| encode | $H_{t+1} = \mathcal{E}(H_t, A_t, U_t)$ | `encode(key, value, priority)`: 용량 초과 시 최저 우선순위 제거 | O |
| recall | $R_t = \mathcal{R}(H_t, c_t)$ | `recall(cue, topk)`: cosine + log-priority -> softmax weighted sum | O |
| replay | priority 기반 재생 | `replay(mode)`: NREM k=1(고집중), REM k=3(분산 재생) | O |
| 주입 | $I_i \leftarrow I_i + \lambda_H R_{i,t}$ | WAKE: recall만, SLEEP: 0.5*recall + 0.5*replay | O |
| WAKE encoding 조건 | 외부 입력 or 목표 존재 시 | `external_norm > 1e-6 or goal.norm > 1e-6` | O |

### 8.5 Sleep Cycle: 3위상 파이프라인 대조

| 위상 | 수식 (3_Sleep.md) | 코드 (`sleep.py`) | 일치 |
|---|---|---|---|
| 각성: 경로 누적 | $\int \mathcal{D}\gamma\,e^{iS}$ 대응 | `collect_sleep_batch`: teacher 생성 -> state/target 수집 | O |
| NREM: LBO 확산 | $W \leftarrow W - \eta_{\text{nrem}} \Delta_g W$ | `smooth_weight_matrix(W, laplacian, eta)` | O |
| NREM: 곡률 기반 가소적 업데이트 | $\text{mask}(G, \varepsilon^2)$ 상위만 통과 | `row_topk_mask(delta, active_ratio)` | O |
| REM: 비선택 경로 재조합 | $G_{\text{rem}} = \text{random\_project}(G_{\text{pruned}}) + \sigma\epsilon$ | `residual @ proj @ proj.T / rank + noise` | O |
| 위상 비율 | Wake $69\%$, NREM $26\%$, REM $5\%$ | `phase_profile = {wake: eng.wake_ratio, nrem: eng.nrem_ratio, rem: eng.rem_ratio}` | O |
| 가드셋 보호 | 품질 하락 시 롤백 | `guard_snapshot` + `evaluate_guard_set` + 조건부 `restore_decoder_snapshot` | O |

### 8.6 CE 상수 대조 (engine.py)

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

1. **대규모 벤치마크**: Sleep cycle의 지속 학습 효과를 Split-CIFAR 또는 텍스트 도메인에서 정량 검증 필요
2. **STDP 미구현**: `17_AgentLoop.md` F.14의 적격 흔적 기반 학습은 아직 코드에 없음
3. **4종 신경조절**: 현재 `runtime.py`는 단일 스칼라 조절만 사용. DA/NE/5HT/ACh 분리 미구현
4. **Cold checkpoint**: `BrainRuntimeSnapshot`은 warm snapshot만 제공. 장기 지속성 저장 미구현
5. **자기수렴 검증**: 초기 균등 분배에서 $p^*$로의 수렴 과도 응답 실측 필요
6. **PQ codebook**: `ce_ops.py`에 `pq_build_codebook` 구현 있으나 대규모 성능 비교 미완
