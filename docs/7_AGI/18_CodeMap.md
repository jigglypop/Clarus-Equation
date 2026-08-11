# 이론-코드 정합 맵

> 이 문서는 `15_Equations.md`의 Layer A--E 수식과 `17_AgentLoop.md`의 Layer F가 실제 코드의 어디에서 구현되는지를 1:1로 대응시킨다.
> 코드를 읽을 때 "이 변수가 어떤 수식인지" 또는 수식을 읽을 때 "이 항이 어디에 구현되어 있는지"를 즉시 찾을 수 있도록 한다.

> 실험 격리: `brain_geometry_benchmark.py`는 Loop 8B의 PFC 끌개–MD 연속조절
> 가설을 검증하는 합성 벤치이며 canonical Layer A–F 런타임에는 연결되지
> 않는다. 통과 결과도 생물학적 동일성이나 런타임 승격을 뜻하지 않는다.

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
    +-- "cuda":  reality_stone.clarus.kernels (CUDA custom ops)    -- 미포함 (선택적)
    +-- "rust":  reality_stone.clarus._rust   (PyO3 바인딩)         -- reality_stone/python/reality_stone/clarus/core/
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

## 10. 자기참조재귀 구현 대응

AI 응용에서 핵심은 단일 모듈 성능이 아니라 \(S_t \to R(S_t) \to C_t \to S_{t+1}\) 루프가 닫히는지다. 현재 코드 대응은 다음처럼 읽는다.

| 재귀 항 | 의미 | 현재 코드 위치 | 구현 판정 |
|---|---|---|---|
| \(S_t\) | 전역 상태: mode, activation, memory, pressure, lifecycle | `runtime.py::BrainRuntime`, `BrainRuntimeSnapshot` | 부분 구현 |
| \(R(S_t)\) | 내부 이완/수렴: 셀 동역학 반복, sparse activation | `runtime.py::step`, `engine.py::CEEngine`, `ce_ops.py::relax` | 구현됨 |
| \(C_t\) | 자기비평: 예측오차, 일관성, 놀라움, 곡률 점수 | `agent.py`, `stdp.py`, `ce_laplacian.py` 후보 | 부분/분산 구현 |
| \(\mathcal M\) | 기억 갱신과 replay | `runtime.py::HippocampusMemory`, `sleep.py` | 구현됨 |
| \(\phi_t\) | 잔류장/불확실성/탈락 경로 보존 | `engine.py`, `sleep.py` | 부분 구현 |
| \(\mathcal U\) | 다음 전역 상태 구성 | `runtime.py::step`, `snapshot()/from_snapshot()` | 구현됨 |

따라서 현재 구현의 강점은 `runtime.py`의 상태-기억-모드 루프이고, 약점은 \(C_t\)가 하나의 표준 self-critic API로 아직 고정되지 않았다는 점이다. LLM 응용을 강화하려면 새 attention 변형을 늘리기보다, `agent.py`/`runtime.py`/`sleep.py` 사이에 self-critic score와 잔류장 업데이트를 표준 계약으로 묶는 것이 우선이다.

### 10.1 수학량과 로그 항목

닫힌 루프 실험에서는 아래 양을 같은 run에서 기록해야 한다.

| 수학량 | 코드에서 읽을 후보 | 필수성 |
|---|---|---|
| \(\|S_{t+1}-S_t\|\) | `RuntimeStep`, snapshot tensor 차이 | 수축률 \(\hat\rho_t\) 계산 |
| \(\bar c_t\) | agent critic score, STDP learning gate, curvature score | 자기비평 강도 |
| \(I_c\) | critic on/off ablation의 `activation` 또는 logits 차이 | critique가 제어량인지 검증 |
| \(I_m\) | hippocampus recall on/off ablation | memory 재주입 영향 |
| \(\|\phi_t\|\) | `engine.py` / `sleep.py` 잔류장 후보 | 잔류장 유계성 |
| \(M_t\) | `RuntimeMode` | WAKE/NREM/REM 별 \(\rho\) 분리 |
| active ratio | `active_modules / dim` | \(\varepsilon^2\) 근처 수렴 여부 |

> 정정 노트 (2026-07, F.14.2 게이트): $g[t]=\alpha_g\,d\bar c/dt+(1-\alpha_g)\,\text{bootstrap\_dev}$ 의 미분항은 **같은 척도의 critic 신호 차분**이어야 한다. `runtime.py::_apply_runtime_stdp` 가 이전에는 drive로 `critic_score`(≈1.0), prev로 `energy`(≈0.3)를 써서 서로 다른 척도를 빼는 바람에 게이트 부호가 거의 무작위였다. 현재는 `_stdp_prev_critic_score` 에 이번 tick의 `gate_drive`(critic, 없으면 energy proxy)를 그대로 저장해 일관된 시간미분이 되도록 수정됨. `stdp_enabled=False` 극한 환원 불변식은 그대로 유지(트래커 미생성 시 조기 반환).

> 효능 판정 (2026-07-30): 인과 배선 테스트는 통과했지만 기본 STDP A/B는
> next-step prediction에서 `NO-EFFECT`, held-out guard에서 `FAIL`이다.
> 구현과 효능을 분리한 수치·재현 명령은 `21_STDP_Efficacy_Audit.md`를 따른다.

최소 closed-loop 판정:

$$
I_c>0,\qquad I_m>0,\qquad
\operatorname{median}_t \hat\rho_t < 1.
$$

더 강한 판정은 open-loop baseline 대비 task score가 좋아지는 동시에 잔류 반경이 커지지 않는 것이다.

$$
G_{\rm rec}>0,
\qquad
\Delta r_\phi \le 0.
$$

### 10.2 계층 gain 로그

`17_AgentLoop.md` F.-1.5의 계층 정리를 코드 실험으로 옮기려면 각 모듈 또는 agent마다 아래 값을 로그로 남긴다.

| 수학량 | 코드 추정 방법 | 판정 |
|---|---|---|
| \(\rho_\ell\) | 같은 입력에서 연속 state delta 비율 `state_delta_next / state_delta` | 모듈 자체 수축률 |
| \(g_\uparrow\) | 하위 모듈 state perturbation이 상위 summary를 바꾸는 norm ratio | aggregation gain |
| \(g_\downarrow\) | 상위 goal/critic perturbation이 하위 activation을 바꾸는 norm ratio | feedback gain |
| \(\rho(G)\) | 추정 gain matrix의 spectral radius | 전체 계층 안정성 |

최소 2층 실험에서는 solver agent와 critic agent만 둔다.

$$
G=
\begin{bmatrix}
\rho_{\rm solver} & g_{\rm down}\\
g_{\rm up} & \rho_{\rm critic}
\end{bmatrix}.
$$

안정 조건은

$$
\rho(G)<1,
$$

또는 보수적으로

$$
\max(\rho_{\rm solver},\rho_{\rm critic})
+
\sqrt{g_{\rm up}g_{\rm down}}
<1
$$

로 로그 판정할 수 있다. 이 값이 1에 가까워지면 상위 critic이 하위 solver를 교정하는 것이 아니라 흔들어 불안정하게 만드는 regime으로 본다.

---

## 11. 파일 책임 분리

| 파일 | 책임 | Layer |
|---|---|---|
| `reality_stone/python/reality_stone/clarus/runtime.py` | 셀 동역학, 모드 전환, 해마, 생애주기, 스냅샷 | A, B, C, D, E |
| `reality_stone/python/reality_stone/clarus/engine.py` | CE 에너지 이완, 디코딩, 상태 분할, 곡률 억제 | F (이완), 6장 |
| `reality_stone/python/reality_stone/clarus/ce_ops.py` | 수치 백엔드 분기, 에너지/이완/메트릭/PQ | F (수치 핵심) |
| `reality_stone/python/reality_stone/clarus/sleep.py` | Wake/NREM/REM 학습 순환, 가드셋, 디코더 리피팅 | F (학습) |
| `reality_stone/python/reality_stone/clarus/belief_control.py` | `[경험식]` action-conditioned rank-1 posterior와 짧은 관측공간 MPC; 기본 비활성 | F (실험 제어) |
| `reality_stone/python/reality_stone/clarus/dpc_benchmark.py` | `[예측]` belief/action/horizon 인과성의 합성 DPC 검증 | F (실험) |
| `reality_stone/python/reality_stone/clarus/credit_control.py` | `[경험식]` signed TD eligibility와 signed homeostasis 분리 probe | F.14 (실험 학습) |
| `reality_stone/python/reality_stone/clarus/delayed_credit_benchmark.py` | `[예측]` 지연 보상 credit의 합성 인과 ablation | F.14 (실험) |
| `reality_stone/python/reality_stone/clarus/device.py` | 디바이스 자동 감지 | 인프라 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs` | brain_step 핵심 루프, Dale's Law | A |
| `reality_stone/python/reality_stone/clarus/core/src/engine/field.rs` | 필드 결합, 리만 거리 | B |
| `reality_stone/python/reality_stone/clarus/core/src/engine/manifold.rs` | 다양체 연산 | B |
| `reality_stone/python/reality_stone/clarus/core/src/engine/nn_ops.rs` | NN 연산 (topk, LBO, gauge) | 2장 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/ce_riemann.rs` | CE 리만 수치 | 물리 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/constants.rs` | 물리 상수 유도 | 3_상수 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/config.rs` | 런타임 설정 | 인프라 |
| `reality_stone/python/reality_stone/clarus/core/src/engine/runtime_types.rs` | CellState, Mode 등 타입 | A, C |

---

## 12. 미구현 대조

| 수식/개념 | 문서 위치 | 코드 상태 |
|---|---|---|
| STDP 적격 흔적 | F.14 | 구현·runtime 연결 완료; 효능 `NO-EFFECT`, guard `FAIL`, 기본 off |
| 4종 신경조절 (DA/NE/5HT/ACh) | F.19 | `neuromod.py` 상태식/효과 mapping 구현; 전체 runtime 폐루프는 부분 |
| 소뇌 전방 모델 | F.20 | `agent.py::CerebellumPredictor`와 RuntimeAgent 연결 구현; 독립 효능 미검증 |
| 작업 기억 용량 제한 $|h_t| \le T_h$ | F.20 | `agent.py::WorkingMemory` FIFO capacity 구현·RuntimeAgent 연결 |
| 뇌파 대역 분해 | F.21 | `runtime.py::brainwave_observable` FFT 5대역 구현 |
| (C3) 메타인지 재귀 루프 | F.17 | `ConsciousnessMonitor.metacognition_step` 수축 toy 구현; 실제 agent feedback 미구현 |
| Cold checkpoint ($\mathcal{C}$) | 14장 7절 | 미구현 (warm만 있음) |
| Live journal ($\mathcal{J}$) | 14장 7절 | 미구현 |
| 섭동적 채널 혼합 | 2장 2.3절 | 미구현 |
| 교차 주파수 결합 게이트 | 2장 6절 | 미구현 |
| action-conditioned belief + MPC | F.2, F.4, F.7 | `belief_control.py`와 RuntimeAgent 선택 경로 구현; 합성 DPC validation `86.32/100 GO`, recurrent sufficient-statistic baseline과는 동률이므로 일반 world-model 우위는 미완성 |
| signed temporal credit | F.14 | `credit_control.py`의 tabular mechanism probe `100/100 GO`; `runtime_credit_benchmark.py`의 Loop 2b/2c는 모두 `0/100 STOP`, 기본 off |
| raw-history controlled belief | F.2, F.4 | `history_state_benchmark.py` 합성 ID/OOD `85/100 GO`; 실제 2-state tanh RNN에는 비열등일 뿐 우위 미확립 |
| modular reward transfer | F.4, F.7 | `reward_transfer_benchmark.py` 구현; locked Loop 4 `0/100 STOP`, context-RNN SAFE class 미학습으로 planner 우위 주장 불가 |
| audited episodic memory | D.1--D.6 | `episodic_memory.py` ADD/UPDATE/DELETE/NOOP·evidence·abstention 후보 구현; corrected Loop 5 `90/100 GO`, runtime 기본 경로 미연결 |
| executive rule belief | F.4, F.7, F.17 | `executive_control.py` 후보 구현; Loop 6 전체 `0/100 STOP`, belief maintenance는 유망하나 surprise 재귀 feedback 효능 실패 |
| active executive information gain | F.7, F.17 | `ActiveExecutiveController` 구현; Loop 7 `0/100 STOP`, 현 과제에 action-dependent sensing/probe가 없어 독립 효능 식별 불가 |
| unified executive posterior/control | F.1, F.4, F.7, F.17 | Loop 8 수식 후보를 research workspace에 고정; canonical·runtime 미편입, 정확 finite-state solver 전까지 구현 잠금 |

### 12.1 2026-08-11 loop-engineering 상태

- `[경험식]` action-conditioned sufficient statistic과 충분한 planning horizon은
  DPC-2/DPC-3 합성 validation에서 reactive, action-agnostic, H1 대조군을
  이겼다. 학습된 모델의 per-delay return은 `0.76797`, reactive는 `0.15`,
  paired 95% LCB는 `0.57344`였다.
- `[미완성]` 같은 sufficient statistic을 가진 recurrent 대조군과의 paired
  차이는 `0.0`이다. 따라서 model-based planning의 독립 우위나 AGI 주장은
  닫히지 않았다.
- `[경험식]` signed delayed eligibility는 tabular probe에서 success `1.0`,
  trace-off/absolute-TD/reward-shuffle은 각각 `0.48242`였다.
- `[경험식]` 외부 signed signal을 받는 선택적 runtime 경로, apply-interval
  누적, snapshot 복원, 8개 대조군 벤치가 구현되었다. 기본 gate는 바꾸지 않았다.
- `[미완성]` runtime Loop 2b는 dense-to-sparse 첫 투영 혼입을 발견했고,
  이를 제거한 Loop 2c도 `0/100 STOP`이었다. signed-off 95% LCB는
  `-0.06021`, reward-shuffle 대비 LCB는 `-0.00208`, held-out guard 증가는
  `+0.07479 > 0.02`였다. 따라서 현재 오차개선량×STDP eligibility 식의
  runtime 효능은 지지되지 않으며 `stdp_enabled=False`를 유지한다.
- `[미완성]` 다음 핵심 공백은 observation/action history에서 latent state를
  학습하는 것이다. 현재 DPC planner는 sufficient statistic을 직접 받아
  같은 정보를 가진 recurrent 대조군과 동률이다.
- `[경험식]` Loop 3은 ordered raw history에서
  $h_t=\rho^{\Delta t}h_{t-1}+m_ta_{t-1}y_t$와 frozen likelihood를 train-only로
  학습해 `85/100 GO`를 얻었다. ID/OOD ECE는 `0.02021/0.02815`이며 action
  제거·shuffle·history 절단 대조군을 모두 이겼다.
- `[미완성]` 교정된 2-state tanh RNN 대비 LCB는 ID `0.0`, OOD `-0.00498`로
  비열등만 성립한다. 다음 게이트는 동일 task 분류가 아니라 명시적 belief와
  planner가 정책 재학습 없이 intervention/reward 변화에 전이하는지다.
- `[미완성]` Loop 4 reward-transfer는 `0/100 STOP`이다. stale-planner
  대조가 6개 cell 중 4개에서 분리되지 않았고 OOD oracle gap 하나가
  `0.020410 > 0.02`였다. context-RNN은 train accuracy `68.76%`와 SAFE 예측
  `0`으로 미학습이므로 큰 candidate-RNN 차이를 planner 우위로 해석하지 않는다.
- `[경험식]` 교정된 Loop 5 bounded-memory probe에서 explicit UPDATE,
  evidence ID, margin abstention, DELETE audit 후보가 `90/100 GO`를 얻었다.
  최신 값·근거·abstention·삭제 정확도는 각각 `1.0`이고 기존 memory 대비
  composite LCB는 `+0.75`다.
- `[미완성]` 위 결과는 UPDATE 간섭을 겨냥한 합성 key/value task다.
  LoCoMo/LongMemEval, multi-session temporal reasoning, replay consolidation,
  BrainRuntime 기본 경로 효능은 아직 검증하지 않았다.
- `[경험식]` Loop 6 hidden-rule task에서 rule posterior 유지 후보는 ID/OOD
  accuracy `0.8771/0.8444`였고 hazard-off·feedback shuffle·gap reset·
  win-stay-shift를 paired LCB로 이겼다.
- `[미완성]` surprise→hazard 증폭은 surprise-off보다 낫지 않았고 전체
  gate는 `0/100 STOP`이다. 메타인지 효능이나 PFC 구현 완료로 승격하지
  않으며, 다음 후보는 명시적 change-point/context inference여야 한다.
- `[미완성]` Loop 7 정보이득 행동도 reward-only보다 낫지 않았다. 현재
  rule-switch 과제에는 미래 관측 품질을 바꾸는 probe 행동이 없으므로
  epistemic action value를 식별할 수 없다. 다음 검증에는 비용 있는 inspect,
  action-conditioned observation, delayed payoff가 함께 필요하다.
- `[미완성]` Loop 8은 분기식 대신
  $q_t(x,c,v,\kappa)$ posterior와 단일 정책 functional로 executive core를
  재정의했다. 이는 아직 research model choice이며 canonical 식이나 PFC
  대응으로 승격하지 않는다. exact finite-state identifiability gate가 먼저다.
- 재현 artifact와 제한된 해석은
  `_workspace/ce/agi-loop-engineering-20260811/`에 기록한다.
