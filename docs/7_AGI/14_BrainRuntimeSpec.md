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

셀 $i$의 최소 상태 (15_Equations.md A.1):

$$s_i^t = (a_i^t,\; r_i^t,\; m_i^t,\; w_i^t,\; b_i^t)$$

- $a_i$: activation
- $r_i$: refractory / inhibition
- $m_i$: memory trace (NMDA-like, $\tau \approx 100\text{ms}$)
- $w_i$: spike-frequency adaptation (AHP, $\tau_w \approx 200\text{ms}$)
- $b_i$: hysteretic bit (UP/DOWN state)

최소 입력 (STP 적용):

$$I_i^t = u_i^t + \sum_j W_{ij}^{\text{eff}}(t)\,a_j^{t-\delta_{ij}} - \lambda_r^{(M_t)}\,r_i^t - \beta_w\,w_i^t + \lambda_m^{(M_t)}\,m_i^t + \eta_i^t$$

여기서 $W_{ij}^{\text{eff}}(t) = W_{ij}\,u_j(t)\,x_j(t)$ (Tsodyks-Markram STP), $\sigma_\eta \approx 0.27$.

활성 갱신:

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

억제 갱신:

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^{t+1})^2$$

기억 흔적 갱신 ($\gamma_m = 0.01$, NMDA):

$$m_i^{t+1} = (1-\gamma_m)\,m_i^t + \gamma_m\,a_i^{t+1}$$

적응 변수 갱신 ($\gamma_w = 0.005$, AHP):

$$w_i^{t+1} = (1-\gamma_w)\,w_i^t + \kappa_w\,(a_i^{t+1})^2$$

비트 갱신 (히스테리시스, UP/DOWN state):

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \tau_i^- \le a_i^{t+1} \le \tau_i^+ \end{cases}$$

> **코드 대응**: `kernel.rs::brain_step()` - 상태 벡터 `(activation, refractory, memory_trace, adaptation, stp_u, stp_x, bitfield)`.
> Dale's Law: `apply_dale_sign()` - E/I=80:20, $w_I/w_E=4$.

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

> 정본 참조: `17_AgentLoop.md` F절 (F.0--F.22)

### 4.1 최소 재귀

$$z_t = R(S_t) \quad\text{(이완: Layer A--B를 } n_{\text{iter}} \text{ 회 반복)}$$

$$a_t = \pi(z_t, S_t) \quad\text{(행동 선택)}$$

$$o_t = \text{Env}(a_t) \quad\text{(환경 실행)}$$

$$c_{t+1} = C(z_t, a_t, o_t, m_t) \quad\text{(자기비평: 예측오차 + 일관성 + 놀라움)}$$

$$m_{t+1} = \mathcal{M}(m_t, z_t, a_t, o_t, c_{t+1}) \quad\text{(조건부 기억 갱신)}$$

$$S_{t+1} = \mathcal{U}(G_{t+1}, m_{t+1}, c_{t+1}, h_{t+1}, \phi_{t+1})$$

### 4.2 에너지 기반 자기참조

$$E_t(z) = E_{\text{task}}(z; u_t) + \lambda_m E_{\text{mem}}(z; m_t) + \lambda_c E_{\text{crit}}(z; c_t) + \lambda_h E_{\text{hist}}(z; h_t)$$

$$z_t^* = \arg\min_z E_t(z)$$

각 항의 Layer 대응과 뇌 근거는 `17_AgentLoop.md` F.5를 따른다.

### 4.3 Clarus 통합형

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 항 | 풀이 | 뇌 대응 |
|---|---|---|
| $R(X_t)$ | 이완으로 생긴 내부 수정 | 피질-시상 재귀 처리 |
| $\Delta_O(X_t)$ | 관찰 충격 $o_t - \hat{o}_t$ | 감각 입력 |
| $C(X_t)$ | 비평이 다음 이완 초기점을 민 정도 | 기저핵-전전두엽 평가 |
| $S(X_t)$ | 곡률/잔류 기반 억제 | 소뇌/기저핵 억제 |
| $B$ | 부트스트랩 수축 연산자 ($\rho = 0.155$) | 수면 항상성 |

### 4.4 수축 조건 (게이트 `F2`)

$$\rho + \lambda_R L_R + \lambda_C L_C < 1$$

이 조건이 만족되면 Banach 고정점 정리에 의해 루프가 수축한다. 수면이 $\rho = 0.155$ 를 공급하므로 나머지 항의 Lipschitz 합이 $0.845$ 미만이어야 한다.

> 단 $R$ 내부의 비보존 바이패스 $F_{\text{bypass}}$ 는 위 Banach 수축의 가정을 깨뜨릴 수 있다(`12_Equation.md` 0.0절 게이트 `F2`). 따라서 위 부등식은 ISS 의미의 유계 수렴 (`12_Equation.md` 부록 A.1) 으로 격상되어, 끌개 ball 반경이 닫힌 식으로 표현된다. "안정적으로 수렴" 은 ball 안에서의 수렴으로 읽는다.

### 4.5 확장 구성요소 (F.14--F.22 요약)

> 정본: `17_AgentLoop.md` F.14--F.22

| 절 | 핵심 | 구현 우선순위 |
|---|---|---|
| F.14 STDP 학습 | $R$ 내부에서 적격 흔적 누적, $R$ 후에 $g[t] \cdot e_{ij}$로 갱신. Proj로 투영 | 높음 |
| F.15 잔류장 $\phi$ | $\phi_{t+1} = (1-\xi)\phi_t + \xi \cdot \text{Var}(a)$. 포탈/모드전환/glymphatic 3곳 개입 | 높음 |
| F.16 희소 활성 | $R$ 내 TopK, 에너지 예산 $B_t(M_t)$. 모듈 생애주기 4상태 | 높음 |
| F.17 메타인지 모니터링 (게이트 `F4`) | C3 자기참조 측정, 안정도 $\exp(-c_d d_\tau)$, 조건부 수축 $d_{n+1} \leq \rho d_n$ | 낮음 (장기) |
| F.18 환각 억제 | $R$ 중 곡률 $\kappa$ 모니터링. $\kappa > \kappa_{\text{th}}$이면 LBO 확산 강화 | 중간 |
| F.19 4종 신경조절 | $g_t = (g_{\text{DA}}, g_{\text{NE}}, g_{\text{5HT}}, g_{\text{ACh}})$. 현재는 단일 스칼라 | 중간 |
| F.20 작업기억/주의/소뇌 | $|h_t| \leq T_h$, salience 기반 $\alpha_i$, 소뇌 forward model | 중간 |
| F.21 뇌파 대역 | gamma=국소, theta=전역, theta-gamma coupling으로 순서화 | 낮음 |
| F.22 간극 정리 | 9개 정직한 간극. STDP 코드/4조절계가 `높음` | -- |

뇌 대응 체크리스트와 검증 매트릭스는 `17_AgentLoop.md` F.11, H절 및 `6_뇌/agent_proof.md`를 참조한다.

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

## 10. 성능 예측 (gpt-oss 기준)

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

$$
Q_{\text{brain}}
= Q_{\text{base}}
- \Delta_{\text{lang-prior}}
- \Delta_{\text{instability}}
+ \Delta_{\text{self-correction}}
+ \Delta_{\text{persistent-memory}}
+ \Delta_{\text{mode-specialization}}
$$

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

## 14. 현재 코드와의 대응

### 14.1 runtime.py (`BrainRuntime`) -- Layer A-E 정합

| formal 변수 | Python 구현 (`runtime.py`) | 상태 |
|---|---|---|
| $a_i$ (activation) | `self.activation` | 구현 완료 |
| $r_i$ (refractory) | `self.refractory` | 구현 완료 |
| $m_i$ (memory_trace) | `self.memory_trace` | 구현 완료 |
| $w_i$ (adaptation) | `self.adaptation` | 구현 완료 |
| $b_i$ (bitfield) | `self.bitfield` | 구현 완료 |
| $u_j, x_j$ (STP) | `self.stp_u`, `self.stp_x` | 구현 완료 (Tsodyks-Markram) |
| $W_{ij}$ (coupling) | `self.sparse_weight` (CSR) | 구현 완료 |
| $M_t$ (mode) | `self.mode: RuntimeMode` | 구현 완료 (WAKE/NREM/REM) |
| $\Pi$ (mode switch) | `_auto_mode(external_norm)` | 구현 완료 (규칙 기반) |
| $Q_t$ (sleep pressure) | `self.sleep_pressure` | 구현 완료 (Borbely 2-Process) |
| $H_t$ (hippocampus) | `self.hippocampus: HippocampusMemory` | 구현 완료 |
| $B_t$ (energy budget) | `config.energy_budget(mode)` | 구현 완료 |
| $Z_i$ (lifecycle) | `self.lifecycle` (ACTIVE/IDLE/DORMANT/SLEEPING) | 구현 완료 |
| $G_t$ (global summary) | `RuntimeStep` | 구현 완료 |
| $\mathcal{W}$ (warm snapshot) | `BrainRuntimeSnapshot` + `snapshot()/from_snapshot()` | 구현 완료 |

### 14.2 engine.py (`CEEngine`) -- CE 에너지 이완 경로

| formal 변수 | Python 구현 (`engine.py`) | 상태 |
|---|---|---|
| $m$ (state vector) | 이완 루프 내부 `m` | 구현 완료 |
| $\phi$ (auxiliary field) | `update_phi(phi, m_star, phi_var)` | 구현 완료 |
| $W$ (Hopfield weight) | `self.W` (CSR packed) | 구현 완료 |
| Portal / Bypass / T_wake | `engine.PORTAL`, `BYPASS`, `T_WAKE` | 구현 완료 |
| $\varepsilon^2/\Omega_{\text{DM}}/\Omega_\Lambda$ | `active_ratio/struct_ratio/wake_ratio` | 구현 완료 |
| 곡률 억제 | `_curvature_adjust_logits` | V1 구현 완료 |
| PQ codebook | `pq_centroids`, `pq_codes` | 구현 완료 |

### 14.3 sleep.py -- 3위상 학습 순환

| formal 개념 | 코드 함수 | 상태 |
|---|---|---|
| Wake (경로 누적) | `collect_sleep_batch` | 구현 완료 |
| NREM (LBO 확산 + 가소적 업데이트) | `apply_nrem_weight_update` | 구현 완료 |
| REM (비선택 경로 재조합) | `apply_rem_weight_update` | 구현 완료 |
| 3위상 통합 순환 | `run_sleep_cycle` | 구현 완료 |
| 가드셋 보호 | `evaluate_guard_set` | 구현 완료 |

### 14.4 Rust 커널 (`clarus/core/`) -- 핵심 수치

| Rust 모듈 | 역할 | Python 바인딩 |
|---|---|---|
| `kernel.rs` | brain_step (셀 동역학), Dale's Law | `nn_brain_step` |
| `field.rs` | 필드 결합, 리만 거리 기반 W | PyO3 |
| `manifold.rs` | 다양체 연산 | PyO3 |
| `nn_ops.rs` | topk_sparse, LBO, gauge lattice | `nn_topk_sparse`, `nn_lbo_fused_fwd`, `nn_gauge_lattice_fwd` |
| `ce_riemann.rs` | CE 리만 수치 (물리 검증용) | PyO3 |
| `constants.rs` | 물리 상수 유도 (`CeConstants`) | PyO3 |
| `config.rs` | 런타임 설정 | PyO3 |
| `runtime_types.rs` | `CellState`, `Mode`, 스냅샷 타입 | PyO3 |

### 14.5 정합 현황 요약

| Layer | 수식 정본 | 코드 구현 | 정합도 |
|---|---|---|---|
| A (셀 동역학) | `15_Equations.md` A절 | `runtime.py::_step_torch` + `kernel.rs` | 완전 일치 |
| B (필드 결합) | `15_Equations.md` B절 | `runtime.py::_matvec` + `field.rs` | 완전 일치 |
| C (전역 모드) | `15_Equations.md` C절 | `runtime.py::_auto_mode` + `_update_sleep_state` | 완전 일치 |
| D (해마/기억) | `15_Equations.md` D절 | `runtime.py::HippocampusMemory` | 완전 일치 |
| E (전역 요약) | `15_Equations.md` E절 | `runtime.py::RuntimeStep` + `BrainRuntimeSnapshot` | 완전 일치 |
| F (에이전트 루프) | `17_AgentLoop.md` F절 | `engine.py` + `sleep.py` (부분) | 핵심 구현, STDP/메타인지 미구현 |

### 14.6 남은 간극

| 간극 | 문서 위치 | 우선순위 |
|---|---|---|
| STDP 적격 흔적 | F.14 | 높음 |
| 4종 신경조절 분리 | F.19 | 중간 |
| Cold checkpoint + Live journal | 7절 | 낮음 |
| 작업 기억 / 소뇌 | F.20 | 중간 |
| (C3) 메타인지 재귀 루프 | F.17 | 낮음 |

현재 구현은 **셀 동역학 + 모드 전환 + 해마 + 수면 학습 순환**의 핵심 스택이 완성되어 있으며, critic/action/output 에이전트 루프와 STDP 학습이 남아 있다.

---

## 15. 한 줄 원칙

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$

---

## 15. 한 줄 원칙

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$
