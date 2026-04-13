# Brain Runtime Specification v0.1

> 위치: `12_Equation.md`의 canonical runtime 5계층을 구현 관점에서 재정의한다.
> 의존: `12_Equation.md`(수식 정본), `6_뇌/evidence.md`(근거 판정), `6_뇌/proof.md`(검증 매트릭스)
>
> 이 문서는 수식 체계 정비, 라이브러리 아키텍처 분리, 단계별 개발 계획, 검증 루프 설계를 다룬다.

---

## 0. 개발 목표

$$\boxed{\text{리만 결합 위에서 작동하는 국소 상태셀들의 지속 실행형 뇌형 런타임}}$$

최종 산출물 3개:

| 산출물 | 책임 |
|---|---|
| **Clarus Kernel** | 셀/필드 갱신 엔진 (Rust) |
| **Brain Runtime** | 모드, 활성/휴면, 해마, 스냅샷 (Python orchestration) |
| **LLM Bridge** | 기존 LLM hidden state와 연결하는 브리지 |

---

## 1. 핵심 전환: 사진에서 영화로

### 1.1 기존 LLM의 한계

기존 LLM은 정적 깊이의 1회성 계산기에 가깝다.

$$h^{\ell+1} = h^\ell + f_\ell(h^\ell)$$

입력이 들어오면 고정된 레이어를 한 번 통과하고 토큰 다음 것을 예측한 뒤 끝난다. 메모리도 context window, KV cache, 외부 RAG 같은 식으로 붙는다. 잠도 없고, 꿈도 없고, 모듈이 쉬지도 않고, 전역 상태가 계속 살아있지도 않다.

### 1.2 Clarus 구조

$$s_i^{t+1} = F_i^{(M_t)}\big(s_i^t,\; u_i^t,\; \sum_j W_{ij}(g)\,s_j^t,\; h_i^t\big)$$

각 단위가 상태를 가지고, 그 상태가 계속 돌고, 모듈마다 깨어났다 잠들고, 전역 모드가 바뀌고, 해마 같은 별도 기억계가 있고, sleep/REM/wake가 계산 모드이며, 전역 출력은 국소 모듈들의 집단 리듬이다.

### 1.3 왜 while 모듈인가: 주기함수 병목의 해결

리만기하학의 표현력은 무궁무진하지만, 사인/코사인 같은 전역 주기 기저를 쓰면 다음 문제가 생긴다.

- 같은 위상으로 되돌아오는 aliasing
- 이력(history) 보존 불가
- 비트성/비가역성 표현 불가

국소 상태모듈(while문)로 바꾸면 각 모듈이 자기 내부 상태를 유지하므로, 같은 입력이라도 이전 상태에 따라 결과가 달라진다. 즉 비트필드에 필요한 것은 periodic code가 아니라 hysteretic dynamical code다.

리만기하학은 이때 표현 좌표가 아니라 **결합 구조**(배선망)를 제공한다.

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right)$$

---

## 2. 수식 층분리 원칙 (Layer A--E)

현재 제일 큰 문제는 물리식, 구현식, 비유식이 한 레벨에 섞여 있다는 것이다. 반드시 5층으로 나눈다.

### 2.1 Layer A: 순수 셀 동역학

셀 $i$의 최소 상태:

$$s_i^t = (a_i^t,\; r_i^t,\; b_i^t)$$

- $a_i$: activation
- $r_i$: refractory / inhibition
- $b_i$: hysteretic bit

최소 입력:

$$I_i^t = u_i^t + \sum_j W_{ij}\,a_j^t - \lambda_r^{(M_t)}\,r_i^t + \lambda_m^{(M_t)}\,m_i^t + \eta_i^t$$

활성 갱신:

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

억제 갱신:

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^t)^2$$

비트 갱신 (히스테리시스):

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \tau_i^- \le a_i^{t+1} \le \tau_i^+ \end{cases}$$

이 Layer A는 순수하고 작아야 한다. 해마도, 자아도, sleep도 넣지 않는다.

### 2.2 Layer B: 필드 결합

셀들의 연결 구조. 가장 단순하게는 sparse graph:

$$W \in \mathbb{R}^{N \times N}$$

리만 해석을 살리려면:

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

$\chi_{ij}$는 sparse mask. 구현은 먼저 graph coupling abstraction까지만.

### 2.3 Layer C: 전역 모드

$$M_t \in \{\mathrm{WAKE},\;\mathrm{NREM},\;\mathrm{REM}\}$$

모드별 파라미터:

$$\Theta^{(M)} = (\gamma_a^{(M)},\;\kappa_a^{(M)},\;\lambda_r^{(M)},\;B^{(M)},\;\dots)$$

셀 식은 같고, 파라미터만 바뀐다. 모드 전환식:

$$M_{t+1} = \Pi(M_t,\;Q_t,\;U_t,\;E_t)$$

- $Q_t$: sleep pressure / arousal
- $U_t$: external input load
- $E_t$: energy budget state

초기에는 규칙 기반. 학습시키지 않는다.

| 모드 | 특성 |
|---|---|
| WAKE | 외부입력 coupling 강함, 감각/추론 주도 |
| NREM | 감쇠 큼, 정리/복원/synaptic down-selection |
| REM | 외부입력 약화, 내부결합/기억 중심, 재조합 |

### 2.4 Layer D: 해마/기억

기억은 셀에 넣지 말고 분리한다.

해마 상태:

$$H_t = (K_t,\;V_t,\;P_t)$$

- $K_t$: cue/index
- $V_t$: stored episode embedding
- $P_t$: replay priority

encode:

$$H_{t+1} = \mathcal{E}(H_t,\;A_t,\;U_t)$$

recall:

$$R_t = \mathcal{R}(H_t,\;c_t)$$

replay injection:

$$I_i^t \leftarrow I_i^t + \lambda_H\,R_{i,t}$$

기억은 셀 로컬 상태가 아니라 **외부 메모리 루프**로 둔다.

### 2.5 Layer E: 자아/전역 상태

전역 상태:

$$G_t = (M_t,\;A_t^{\text{summary}},\;H_t,\;Q_t,\;\mu_t)$$

자아는 이걸 관측한 higher-order summary다.

$$\text{Self}_t = \mathcal{S}(G_t)$$

초기에 구현 안 해도 된다. 문서상 변수만 두고 넘어간다.

---

## 3. 최소 전체식: 6개 핵심 수식

$$I_i^t = u_i^t + \sum_j W_{ij}(g)\,a_j^t - \lambda_r^{(M_t)}\,r_i^t + \lambda_H\,R_{i,t}$$

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^t)^2$$

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \text{otherwise} \end{cases}$$

$$H_{t+1} = \mathcal{E}(H_t, A_t), \quad R_t = \mathcal{R}(H_t, c_t)$$

$$M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$$

이 6개만 흔들리지 않게 잡으면 나머지는 구현으로 내릴 수 있다.

---

## 4. 자기참조 재귀식

### 4.1 최소 재귀

$$S_{t+1} = U_\theta(S_t,\;z_t,\;a_t,\;o_t,\;c_{t+1},\;m_{t+1})$$

with:

$$z_t = R_\theta(S_t) \quad\text{(relax/converge)}$$

$$a_t = \pi_\theta(z_t) \quad\text{(action selection)}$$

$$o_t = E(a_t, S_t) \quad\text{(execution)}$$

$$c_{t+1} = C_\theta(S_t, a_t, o_t) \quad\text{(self-critique)}$$

$$m_{t+1} = M_\theta(m_t, z_t, a_t, o_t, c_{t+1}) \quad\text{(memory update)}$$

### 4.2 에너지 기반 자기참조

$$E_t(z) = E_{\text{task}}(z;g) + \lambda_m E_{\text{mem}}(z;m_t) + \lambda_c E_{\text{crit}}(z;c_t) + \lambda_h E_{\text{hist}}(z;h_t)$$

$$z_t^* = \arg\min_z E_t(z)$$

### 4.3 Clarus 통합형

$$X_{t+1} = B\big[X_t + \lambda_1 R_{\text{self}}(X_t) + \lambda_2 R_{\text{obs}}(X_t) + \lambda_3 C(X_t) - \lambda_4 S(X_t)\big]$$

- $R_{\text{self}}$: 자기참조로 생기는 내부 수정
- $R_{\text{obs}}$: 관찰/측정/메타인지로 생기는 상태 전이
- $C$: 구조적 정렬, 의미 응집, 통찰 방향 (Clarus field)
- $S$: 잡음, 허위 attractor, 불필요 branch 억제 (Suppression field)
- $B$: brain-like integrator

---

## 5. 국소 뇌파 해석

### 5.1 while 모듈 = 국소 파동 발생기

각 모듈의 활성도 $a_i^t$ 시계열이 국소 리듬 성분을 만든다.

$$\psi_i(t) = a_i^t$$

### 5.2 전역 뇌파 = 합성 관측량

$$\Psi_{\text{global}}(t) = \sum_i \omega_i\,a_i^t$$

대역별 분해: $\Psi_\delta(t),\;\Psi_\theta(t),\;\Psi_\alpha(t),\;\Psi_\beta(t),\;\Psi_\gamma(t)$

EEG-like 관측:

$$\text{EEG}(t) = O\!\left(\{a_i^t\}_{i=1}^N\right)$$

모듈이 리듬을 만들고, 전역 뇌파는 그 리듬들의 합성된 관측량이다.

### 5.3 뇌파 대역과 뇌 회로 대응

| 대역 | 주요 회로 |
|---|---|
| delta | 전두 slow-wave, 깊은 수면 |
| theta | 해마, 기억 인코딩/회상, REM |
| alpha | 후두-두정 시각계, 게이팅 |
| mu | 감각운동 피질, 운동 억제 |
| beta | 전두-운동계, 현재 상태 유지 |
| sigma/spindle | 시상-피질, NREM2 |
| gamma | 국소 피질 회로, 결합/집중 |
| ripple | 해마, 기억 재생 |

---

## 6. 모듈 생애주기와 에너지 예산

### 6.1 모듈 상태

$$Z_i^t \in \{\text{ACTIVE},\;\text{IDLE},\;\text{DORMANT},\;\text{SLEEPING}\}$$

| 상태 | 의미 |
|---|---|
| ACTIVE | 지금 연산 참여 |
| IDLE | 바로 깨울 수 있는 대기 |
| DORMANT | 장기 휴면, coupling 거의 끊김 |
| SLEEPING | 내부 정리/압축 중 |

### 6.2 에너지 예산

$$\sum_i z_i^t \le B_t$$

$B_t$는 모드에 따라 달라진다: $B_t(\text{NREM}) < B_t(\text{WAKE})$

### 6.3 활성 조건

$$z_i^{t+1} = \mathbf{1}\!\left[\alpha_u \|u_i^t\| + \alpha_m \|m_i^t\| + \alpha_n \sum_j W_{ij} a_j^t - \alpha_r r_i^t + \alpha_q q_i^t > \theta_i^{(M_t)}\right]$$

### 6.4 희소 활성 원칙

반드시 $A_t \ll N$이어야 한다. 전 모듈 상시 활성은 에너지 폭발.

### 6.5 상태 업데이트

$$s_i^{t+1} = z_i^t\;F_i^{(M_t)}(s_i^t,\;u_i^t,\;\textstyle\sum_j W_{ij} s_j^t) + (1-z_i^t)\;H_i^{(M_t)}(s_i^t)$$

$F_i$: 활성 모듈 업데이트. $H_i$: 휴면 중 느린 decay/유지.

---

## 7. 스냅샷/지속성 계층

프로세스 종료 = 부분적 기억상실에 가깝다. 3계층 저장이 필수.

| 계층 | 주기 | 내용 |
|---|---|---|
| cold checkpoint | 가끔 | 전체 구조, 장기 기억, 안정 상태: $\mathcal{C} = (\Theta, W, \text{long\_memory})$ |
| warm snapshot | 자주 | 현재 해마 상태, 활성 모듈군, 전역 모드: $\mathcal{W} = (M_t, H_t, \text{active\_set}_t)$ |
| live journal | 실시간 append | 중요 이벤트, 새 기억 인덱스, 모드 전환 로그: $\mathcal{J} = (\text{events}, \text{transitions})$ |

---

## 8. 라이브러리 아키텍처

### 8.1 Python / Rust 분리 원칙

$$\text{Python} = \text{orchestration/policy/experiment}$$
$$\text{Rust} = \text{pure computation kernel}$$

Python은 Rust의 존재를 모르게 한다. backend protocol로 분리.

### 8.2 Backend Protocol

```
class CEBackend(Protocol):
    def relax(self, state, weights, cfg) -> RelaxResult: ...
    def logits(self, hidden, lm_head, bias) -> Tensor: ...
    def sample(self, logits, cfg) -> Tensor: ...
    def consolidate(self, memory, cfg) -> Memory: ...
    def critic_scores(self, state, goal, output, cfg) -> Scores: ...
```

backend 선택은 한 군데서만. `load_backend(prefer="auto")`.

### 8.3 목표 디렉터리 구조

```
clarus/
  kernel/          # 순수 계산: cell.rs, field.rs, coupling.rs, config.rs, traits.rs
  runtime/         # 모드, 활성/휴면, scheduler: brain.rs, mode.rs, lifecycle.rs, energy.rs, snapshot.rs
  memory/          # 해마, replay, trace: hippocampus.rs, replay.rs, trace.rs
  bridge/          # PyTorch/LLM/Python 연결: pytorch.rs, python_api.rs, llm_bridge.rs
  apps/            # CLI, demos, experiments
```

### 8.4 책임 분리

| 모듈 | 핵심 인터페이스 |
|---|---|
| kernel | $s_i^{t+1} = F(s_i^t, u_i^t, n_i^t, \theta)$ |
| runtime | $X_{t+1} = \mathcal{U}(X_t, \text{input}_t)$ |
| memory | encode / recall / replay |
| bridge | PyTorch/LLM/Python 연결 |

---

## 9. LLM 변환 대응

### 9.1 Transformer 부품 대응

| Transformer | Clarus 대응 |
|---|---|
| hidden state $h_t$ | 국소 모듈 상태 집합 $\{s_1^t, \dots, s_N^t\}$ |
| attention | 리만 결합 $\sum_j W_{ij}(g)\,a_j^t$ |
| FFN | 국소 모듈 업데이트 $F_i$ |
| residual connection | 상태 지속성 ($s_i^t$ 유지) |
| layer depth | 시간 반복 $n_{\text{iter}}$ |
| KV cache | 해마 + 압축 기억 |

### 9.2 변환 경로

1단계(호환형): 기존 LLM hidden state를 모듈장으로 매핑
2단계(모사형): 기존 LLM 출력 분포를 비슷하게 재현
3단계(초과형): 더 적은 파라미터, 더 긴 지속성, 더 나은 자기수정

---

## 10. 성능 예측 (vs gpt-oss 기준)

### 10.1 용량

$$P_{\text{brain}} \approx N\,(p_{\text{loc}} + k\,p_{\text{edge}}) + P_{\text{io}} + P_{\text{mode}}$$

sparse graph면 $P_{\text{brain}} = O(N)$.

### 10.2 런타임 메모리

$$M_{\text{run}} \approx M_{\text{weights}} + A_t\,d_s\,b_s + k\,A_t\,b_e + R\,d_h\,b_h$$

긴 컨텍스트를 raw token이 아니라 state로 압축하면 메모리 이점이 생긴다.

### 10.3 속도

$$C_{\text{reply}} \approx n_{\text{iter}}\,A_t\,(p_{\text{loc}} + k\,p_{\text{edge}})$$

step은 싸게 만들 수 있지만, 전체 속도는 수렴 step 수에 달린다.

### 10.4 정확도

$$Q_{\text{brain}} = Q_{\text{base}} - \Delta_{\text{lang-prior}} - \Delta_{\text{instability}} + \Delta_{\text{self-correction}} + \Delta_{\text{persistent-memory}} + \Delta_{\text{mode-specialization}}$$

초기 정확도는 기존 LLM보다 낮을 가능성이 크고, 구조가 안정화되면 특정 과제(장기 상태 유지, 자기수정, agentic)에서 역전 가능.

### 10.5 유리한 위치

짧은 one-shot LM 정답률이 아니라:

- stateful brain-like runtime
- long-horizon agent
- persistent memory reasoner

---

## 11. 단계별 개발 계획

### Phase 0: 정리 주간

- naming cleanup
- Layer A--E 수식 문서 작성 (`15_Equations.md`)
- backend observable과 runtime state 분리

### Phase 1: Clarus Kernel v0

- `ClarusCellState { a, r, b }`
- `ClarusCellParams`
- `ClarusField` + sparse coupling
- deterministic step
- 완료 기준: 64개 셀로 안정적으로 1만 step

### Phase 2: Runtime v0

- 전역 모드 `WAKE / SLEEP`
- energy budget
- active/idle/dormant lifecycle
- scheduler
- 완료 기준: 일부만 활성, sleep 모드에서 활성 수 감소

### Phase 3: Hippocampus v0

- trace cache 대체
- encode / recall / replay
- priority replay
- 완료 기준: cue 주면 replay가 활성 셀에 영향

### Phase 4: Mode v1

- WAKE / NREM / REM
- mode switch rules
- 완료 기준: 같은 입력에 다른 모드가 다른 evolution

### Phase 5: Snapshot / Persistence

- cold checkpoint + warm snapshot + live journal
- 완료 기준: 저장 후 복구 시 동역학 연속성

### Phase 6: Python / PyTorch Bridge

- Rust kernel 유지
- Python wrapper는 runtime orchestration만
- PyTorch는 parameter learning / experimental reference

### Phase 7: LLM Bridge

- $h_t^{\text{LLM}} \to U_t^{\text{Clarus}}$
- $A_t^{\text{summary}} \to \hat{h}_t$
- 처음엔 adapter만. stateful sidecar runtime으로 붙인다.

---

## 12. 모듈 수 가이드

| 버전 | 모듈 수 $N$ | 동시 활성 $A_t$ | 용도 |
|---|---|---|---|
| v0 proto | 8--16 | 4--8 | 개념 검증 |
| v1 start | **64** | 8--12 | 현실적 시작점 (추천) |
| v2 brain-like | 256--1024 | 16--64 | 뇌형 확장 |
| v3+ | 10k+ | sparse | MSA급 분산 동역학계 |

---

## 13. 뇌 대응 비유 요약

| 개념 | Clarus 대응 | 프론트엔드 비유 |
|---|---|---|
| 국소 회로 | ClarusCell | 자기 상태를 갖는 스마트 컴포넌트 |
| 뇌파 | 모듈 활성도의 집단 리듬 | 전체 UI의 분위기/활동량 |
| 해마 | HippocampusIndex | 최근 활동 캐시 + 중요 상태 저장소 |
| 수면 | NREM/REM 모드 전환 | 백그라운드 정리/압축 |
| 자아 | 전역 모드 + 기억 + 연속성 | root state |
| 기저핵 | 게이트 오토마타 (action selection) | dispatch / access control |
| 전전두엽 | 작업기억 상태 유지 | live state |
| 신경조절계 | 전역 모드 전환 | global mode manager |

---

## 14. 현재 brain.rs와의 대응

| formal 변수 | brain.rs 구현 |
|---|---|
| $m_t$ (주 상태) | `field.phi`, `field.dphi` |
| $\phi_t$ (잔류장) | `field` + `pooled memory` |
| $q_t$ (제어) | `arc` 추정 상태 (`r_est`, `k_est`) |
| $g$ (goal) | `goal` 직접 존재 |
| $h_t$ (memory) | `memory` EMA |
| suppression | `geo_sup`, `suppression.apply_to_signal` |

현재 구현은 brain-field core가 있으나, critic/action/output loop는 아직 부족하다.

---

## 15. 한 줄 원칙

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$
