# 이론-코드 정합 맵

> 이 문서는 `15_Equations.md`의 Layer A--E 수식과 `17_AgentLoop.md`의 Layer F가 실제 코드의 어디에서 구현되는지를 1:1로 대응시킨다.
> 코드를 읽을 때 "이 변수가 어떤 수식인지" 또는 수식을 읽을 때 "이 항이 어디에 구현되어 있는지"를 즉시 찾을 수 있도록 한다.

---

## 1. 전체 아키텍처 대응

```
15_Equations.md            clarus/
+-------------------+      +-----------------------------+
| Layer A: 셀 동역학  | <--> | runtime.py::_step_torch     |
|                   |      | core/src/engine/kernel.rs   |
+-------------------+      +-----------------------------+
| Layer B: 필드 결합  | <--> | runtime.py::_matvec (CSR)   |
|                   |      | core/src/engine/field.rs    |
+-------------------+      +-----------------------------+
| Layer C: 전역 모드  | <--> | runtime.py::_auto_mode      |
|                   |      | runtime.py::_update_sleep   |
+-------------------+      +-----------------------------+
| Layer D: 해마/기억  | <--> | runtime.py::HippocampusMemory|
+-------------------+      +-----------------------------+
| Layer E: 전역 요약  | <--> | runtime.py::RuntimeStep     |
|                   |      | runtime.py::BrainRuntimeSnapshot |
+-------------------+      +-----------------------------+
| Layer F: 에이전트   | <--> | engine.py::CEEngine (이완)   |
|          루프      |      | sleep.py::run_sleep_cycle   |
+-------------------+      +-----------------------------+
```

---

## 2. Layer A: 셀 동역학

### 2.1 상태 변수

| 수식 기호 | canonical 이름 | Python 변수 | Rust 변수 | 초기값 |
|---|---|---|---|---|
| $a_i$ | activation | `self.activation` | `activation` | 0 |
| $r_i$ | refractory | `self.refractory` | `refractory` | 0 |
| $m_i$ | memory_trace | `self.memory_trace` | `memory_trace` | 0 |
| $w_i$ | adaptation | `self.adaptation` | `adaptation` | 0 |
| $b_i$ | bitfield | `self.bitfield` | `bitfield` | 0 |
| $u_i$ | stp_u | `self.stp_u` | `stp_u` | 0.5 |
| $x_i$ | stp_x | `self.stp_x` | `stp_x` | 1.0 |

### 2.2 입력 계산 (A.2)

$$I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}} a_j - \lambda_r r_i - \beta_w w_i + \lambda_m m_i + \eta_i$$

```python
# runtime.py::_step_torch, line ~508
pre = stp_u * stp_x * self.activation * prev_active  # W_eff = u*x*a
recurrent = self._matvec(pre)                         # sum_j W_ij * pre_j
adapt_force = 0.12 * self.adaptation                  # beta_w * w_i

drive = (
    recurrent                                 # sum_j W_ij_eff * a_j
    + self.config.external_gain * external    # u_i (external input)
    + self.config.goal_gain * self.goal       # goal contribution
    + self.config.replay_mix(mode) * replay   # lambda_H * R_i (hippocampus)
    - self.config.refractory_scale * self.refractory  # -lambda_r * r_i
    - adapt_force                             # -beta_w * w_i
)
```

### 2.3 활성 갱신 (A.3)

$$a_i^{t+1} = (1 - \gamma_a^{(M)}) a_i^t + \kappa_a^{(M)} \tanh(I_i^t)$$

```python
# runtime.py::_step_torch, line ~516
activation = (
    (1.0 - self.config.activation_decay(mode)) * self.activation
    + self.config.activation_gain(mode) * torch.tanh(drive)
).clamp(-1.0, 1.0)
```

### 2.4 억제 갱신 (A.4)

$$r_i^{t+1} = (1 - \gamma_r^{(M)}) r_i^t + \kappa_r^{(M)} (a_i^{t+1})^2$$

```python
# runtime.py::_step_torch, line ~520
refractory = (
    (1.0 - self.config.refractory_decay(mode)) * self.refractory
    + self.config.refractory_gain(mode) * activation.square()
)
```

### 2.5 기억 흔적 (A.5)

$$m_i^{t+1} = (1 - \gamma_m) m_i^t + \gamma_m a_i^{t+1}, \quad \gamma_m = 0.01$$

```python
# runtime.py::_step_torch, line ~524
memory_trace = 0.99 * self.memory_trace + 0.01 * activation
```

### 2.6 적응 변수 (A.6 / J.20)

$$w_i^{t+1} = (1 - \gamma_w) w_i^t + \kappa_w (a_i^{t+1})^2, \quad \gamma_w = 0.005$$

```python
# runtime.py::_step_torch, line ~526
adaptation = ((1.0 - 0.005) * self.adaptation + 0.005 * activation.square()).clamp(0.0, 2.0)
```

### 2.7 비트 갱신 (A.7)

$$b_i^{t+1} = \begin{cases} 1 & a_i > \tau^+ \\ 0 & a_i < \tau^- \\ b_i^t & \text{otherwise} \end{cases}$$

```python
# runtime.py::_step_torch, line ~528
bitfield[activation >= self.config.bit_upper_threshold] = 1   # tau+ = 0.30
bitfield[activation <= self.config.bit_lower_threshold] = 0   # tau- = 0.10
```

### 2.8 STP (Tsodyks-Markram, J.19)

$$u_j \leftarrow u_j + (-u_j/\tau_f + u_0(1-u_j)\delta(t-t_j^*))$$
$$x_j \leftarrow x_j + ((1-x_j)/\tau_r - u_j x_j \delta(t-t_j^*))$$

```python
# runtime.py::_step_torch, line ~492
stp_u = self.stp_u + (-tau_fac_inv * self.stp_u + u_base * (1 - self.stp_u) * spike)
stp_x = self.stp_x + (tau_rec * (1 - self.stp_x) - self.stp_u * self.stp_x * spike)
```

---

## 3. Layer B: 필드 결합

| 수식 | 코드 위치 | 구현 방식 |
|---|---|---|
| $W_{ij}$ (sparse) | `runtime.py::__init__` | `pack_sparse` -> CSR `(values, col_idx, row_ptr)` |
| $\sum_j W_{ij} a_j$ | `runtime.py::_matvec` | `torch.sparse.mm(sparse_weight, x)` |
| $W_{ij}(g) = \exp(-d_g^2/\sigma^2) \chi_{ij}$ | `core/src/engine/field.rs` | Rust 구현 |
| Dale's Law ($w_I/w_E = 4$, E:I = 80:20) | `core/src/engine/kernel.rs::apply_dale_sign` | Rust 구현 |

---

## 4. Layer C: 전역 모드

### 4.1 모드 전환 ($\Pi$)

$$M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$$

```python
# runtime.py::_auto_mode
def _auto_mode(self, external_norm):
    if self.mode is WAKE:
        if self.sleep_pressure > 1.0 and external_norm < wake_threshold:
            return NREM       # 수면 압력 높고 외부 자극 약함
        return WAKE
    if self.mode is NREM:
        if external_norm > wake_threshold * 1.5:
            return WAKE       # 강한 외부 자극 -> 즉시 각성
        if self.sleep_pressure < 0.45:
            return REM        # 수면 압력 충분히 해소 -> REM 전환
        return NREM
    # REM
    if external_norm > wake_threshold or self.sleep_pressure < 0.15:
        return WAKE           # 외부 자극 또는 수면 완료 -> 각성
    return REM
```

### 4.2 수면 압력 (Borbely 2-Process, C.2)

$$\frac{dS}{dt} = \begin{cases} (S_{\max} - S)/\tau_w & \text{WAKE} \\ -S/\tau_s & \text{NREM} \\ -S/(2\tau_s) & \text{REM} \end{cases}$$

```python
# runtime.py::_update_sleep_state
# tau_w = 18.2h = 65520 steps @1ms, tau_s = 4.2h = 15120 steps @1ms
if mode is WAKE:
    self.sleep_pressure += (s_max - self.sleep_pressure) * tau_w_inv
elif mode is NREM:
    self.sleep_pressure -= self.sleep_pressure * tau_s_inv
else:  # REM
    self.sleep_pressure -= self.sleep_pressure * tau_s_inv * 0.5
```

### 4.3 모드별 파라미터 ($\Theta^{(M)}$)

```python
# runtime.py::BrainRuntimeConfig
#                        WAKE   NREM   REM
# activation_decay:      0.18   0.34   0.22
# activation_gain:       0.82   0.52   0.68
# refractory_decay:      0.12   0.26   0.18
# refractory_gain:       0.24   0.12   0.18
# energy_budget:         base   0.5x   0.75x
# replay_mix:            0.08   0.28   0.35
```

---

## 5. Layer D: 해마/기억

| 수식 | 코드 위치 | 구현 |
|---|---|---|
| $H_t = (K_t, V_t, P_t)$ | `HippocampusMemory._keys, _values, _priority` | list[Tensor] |
| $\mathcal{E}(H_t, A_t, U_t)$ | `HippocampusMemory.encode(key, value, priority)` | 용량 초과 시 최저 priority drop |
| $R_t = \mathcal{R}(H_t, c_t)$ | `HippocampusMemory.recall(cue, topk)` | cosine + log(priority) -> softmax weighted sum |
| replay injection | `HippocampusMemory.replay(mode)` | NREM: k=1, REM: k=3 |
| $I_i \leftarrow I_i + \lambda_H R_{i,t}$ | `runtime.py::step` | WAKE: recall, SLEEP: 0.5*recall + 0.5*replay |

### 5.1 encode 조건

```python
# runtime.py::step, line ~589
# WAKE: 외부 입력 또는 목표가 있을 때만 기억
if mode is WAKE and (external_norm > 1e-6 or goal.norm > 1e-6):
    hippocampus.encode(activation, value=memory_trace, priority=priority)
# SLEEP: 기존 기억 + 현재 활성의 혼합을 통합
elif mode is not WAKE and len(hippocampus) > 0:
    consolidated = 0.85 * activation + 0.15 * replay
    hippocampus.encode(consolidated, value=memory_trace, priority=priority * 0.5)
```

---

## 6. Layer E: 전역 요약

| 수식 | 코드 위치 |
|---|---|
| $G_t = (M_t, A_t^{\text{summary}}, H_t, Q_t, \mu_t)$ | `RuntimeStep(step, mode, energy, active_modules, replay_norm, sleep_pressure, arousal, lifecycle_counts)` |
| $\mathcal{W}$ (warm snapshot) | `BrainRuntimeSnapshot`: config + 전체 상태 텐서 + 해마 state_dict |
| snapshot 저장/복원 | `BrainRuntime.snapshot()` / `BrainRuntime.from_snapshot()` |

---

## 7. Layer F: 에이전트 루프 (CE 에너지 이완 경로)

> CE 에너지 이완 추론은 Layer A-B의 brain cell dynamics와는 별도 경로다.
> `engine.py::CEEngine`이 Hopfield 에너지 이완을 수행하고, `sleep.py`가 3위상 학습 순환을 관리한다.

### 7.1 에너지 이완 ($R$)

$$E(m, \phi) = -\frac{1}{2} m^\top W m - m^\top b + \text{portal} \cdot m^\top \hat\phi + E_{\text{cb}} + E_{\text{bypass}}$$

```python
# ce_ops.py::_energy_parts_torch
E_hop    = -0.5 * dot(m, W @ m)        # Hopfield
E_bias   = -dot(m, m0)                 # bias toward initial state
E_portal = -portal * dot(m, phi_hat)   # portal coupling
E_cb     = codebook Boltzmann          # log-sum-exp over codebook
E_bypass = bypass_coeff * dot(m, phi)  # non-conservative bypass
```

### 7.2 이완 루프

```python
# ce_ops.py::_relax_packed_torch
for step in range(n_steps):
    grad = -W @ m - b + portal * phi_hat + ...   # dE/dm
    natural_dir = metric_aware_direction(grad)     # natural gradient
    noise = fdt_noise(T, dt, tau)                  # FDT-compliant noise
    m = m - dt/tau * natural_dir + noise
    m = normalize(m) * norm0                       # norm preservation
    phi = update_phi(phi, m_star, phi_var)          # auxiliary field update
```

### 7.3 Sleep Cycle (3위상 학습)

$$\text{Wake} \to \text{NREM} \to \text{REM} \to \text{evaluate}$$

| 위상 | 코드 함수 | 핵심 연산 |
|---|---|---|
| Wake | `collect_sleep_batch` | teacher 기반 state/target 수집 |
| NREM W 갱신 | `apply_nrem_weight_update` | Laplacian 확산 + 상위 `active_ratio` 가소적 업데이트 |
| NREM 디코더 | `fit_decoder_from_batch` | ridge 회귀로 state->logit 투영 리피팅 |
| NREM 어휘 헤드 | `finetune_vocab_head_from_batch` | AdamW soft-target 미세조정 |
| REM W 갱신 | `apply_rem_weight_update` | 비선택 잔차 저랭크 투영 + 노이즈 재조합 |
| REM 디코더/어휘 | 위와 동일 (rem_weight, rem_mix 적용) | hard sample 가중 |
| 가드셋 보호 | `evaluate_guard_set` | top1/top10/top50 품질 체크, 조건부 롤백 |

### 7.4 위상 비율

$$\text{wake} : \text{nrem} : \text{rem} = \Omega_\Lambda : \Omega_{\text{DM}} : \varepsilon^2 = 68.91\% : 26.23\% : 4.87\%$$

```python
# sleep.py::run_sleep_cycle
phase_profile = {
    "wake": eng.wake_ratio,   # 0.6891
    "nrem": eng.nrem_ratio,   # 0.2623
    "rem":  eng.rem_ratio,    # 0.0487
}
phase_budget = allocate_phase_sample_counts(total_cycle_samples, phase_profile)
```

---

## 8. CE 상수 -> 코드 값

| 수식 기호 | 유도식 | 코드 변수 | 값 |
|---|---|---|---|
| $\text{\_AD}$ | $4/(e^{4/3}\pi^{4/3})$ | `engine._AD` | 0.1726... |
| Portal | $(\text{\_AD}(1-\text{\_AD}))^2$ | `engine.PORTAL` | 0.03120 |
| Bypass | $1/(e^{1/3}\pi^{1/3})$ | `engine.BYPASS` | 0.4892 |
| $T_{\text{wake}}$ | $1/(3+\text{\_AD}(1-\text{\_AD}))$ | `engine.T_WAKE` | 0.3148 |
| $\varepsilon^2$ | bootstrap fixed point | `eng.active_ratio` | 0.0487 |
| $\Omega_{\text{DM}}$ | bootstrap fixed point | `eng.struct_ratio` | 0.2623 |
| $\Omega_\Lambda$ | bootstrap fixed point | `eng.wake_ratio` | 0.6891 |
| $r_c$ | $\pi$ | `eng.sparsity_radius` | 3.1416 |
| target W density | $N=4096, r_c=\pi$ | `eng.target_w_density` | 0.0316 |
| codebook weight | $(\text{\_AD}(1-\text{\_AD}))^2$ | `ce_ops.DEFAULT_CB_W` | 0.03120 |

---

## 9. 백엔드 분기

```
ce_ops.ce_backend(device, requested) -> "cuda" | "rust" | "torch"
    |
    +-- "cuda":  clarus.kernels (CUDA custom ops)    -- 미포함 (선택적)
    +-- "rust":  clarus._rust   (PyO3 바인딩)         -- clarus/core/
    +-- "torch": pure PyTorch fallback               -- ce_ops 내부
```

| 연산 | Torch fallback | Rust (`_rust`) | CUDA |
|---|---|---|---|
| pack_sparse | `_pack_sparse_torch` | `nn_ce_pack_sparse` | -- |
| build_metric_basis | `_build_metric_basis_torch` | `nn_ce_metric_basis_fwd` | -- |
| codebook_pull | `_codebook_pull_torch` | `nn_ce_codebook_pull` | -- |
| relax_packed | `_relax_packed_torch` | `nn_ce_relax_fwd` | -- |
| brain_step | `_step_torch` | `nn_brain_step` | -- |
| topk_sparse | torch.topk | `topk_sparse` | -- |
| LBO fused fwd | torch mm | `nn_lbo_fused_fwd` | -- |
| power iter | `linalg.eigh` | `nn_power_iter` | -- |
| gauge lattice | torch mm | `nn_gauge_lattice_fwd` | -- |

---

## 10. 파일 책임 분리

| 파일 | 책임 | Layer |
|---|---|---|
| `clarus/runtime.py` | 셀 동역학, 모드 전환, 해마, 생애주기, 스냅샷 | A, B, C, D, E |
| `clarus/engine.py` | CE 에너지 이완, 디코딩, 상태 분할, 곡률 억제 | F (이완), 6장 |
| `clarus/ce_ops.py` | 수치 백엔드 분기, 에너지/이완/메트릭/PQ | F (수치 핵심) |
| `clarus/sleep.py` | Wake/NREM/REM 학습 순환, 가드셋, 디코더 리피팅 | F (학습) |
| `clarus/device.py` | 디바이스 자동 감지 | 인프라 |
| `clarus/core/src/engine/kernel.rs` | brain_step 핵심 루프, Dale's Law | A |
| `clarus/core/src/engine/field.rs` | 필드 결합, 리만 거리 | B |
| `clarus/core/src/engine/manifold.rs` | 다양체 연산 | B |
| `clarus/core/src/engine/nn_ops.rs` | NN 연산 (topk, LBO, gauge) | 2장 |
| `clarus/core/src/engine/ce_riemann.rs` | CE 리만 수치 | 물리 |
| `clarus/core/src/engine/constants.rs` | 물리 상수 유도 | 3_상수 |
| `clarus/core/src/engine/config.rs` | 런타임 설정 | 인프라 |
| `clarus/core/src/engine/runtime_types.rs` | CellState, Mode 등 타입 | A, C |

---

## 11. 미구현 대조

| 수식/개념 | 문서 위치 | 코드 상태 |
|---|---|---|
| STDP 적격 흔적 | F.14 | 미구현 |
| 4종 신경조절 (DA/NE/5HT/ACh) | F.19 | 미구현 (단일 스칼라) |
| 소뇌 전방 모델 | F.20 | 미구현 |
| 작업 기억 용량 제한 $|h_t| \le T_h$ | F.20 | 미구현 |
| 뇌파 대역 분해 | F.21 | 미구현 |
| (C3) 메타인지 재귀 루프 | F.17 | 미구현 |
| Cold checkpoint ($\mathcal{C}$) | 14장 7절 | 미구현 (warm만 있음) |
| Live journal ($\mathcal{J}$) | 14장 7절 | 미구현 |
| 섭동적 채널 혼합 | 2장 2.3절 | 미구현 |
| 교차 주파수 결합 게이트 | 2장 6절 | 미구현 |
