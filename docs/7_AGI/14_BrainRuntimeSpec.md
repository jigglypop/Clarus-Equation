# Brain Runtime Specification v0.1

이 문서는 BrainRuntime 후보의 상태 tensor, 모드 전환, 기억, 전역 readout을 구현 계약으로 정의한다. 독자는 state machine·tensor shape·이산 update의 기본을 아는 독자를 전제로 하며, 뇌파·자아·기억 명칭은 producer와 consumer를 설명하는 비유이지 생물학적 기제의 확인이 아니다.

개발 목표 뒤에 Layer A--E와 여섯 수식의 입력·출력, 재귀 수축 조건, 국소 관측 비유를 차례로 읽는다. 상태 값은 지정한 정규화의 무차원 tensor이고 timebase는 runtime tick이며, 안정성·효능은 baseline·ablation·OOD gate가 없는 한 설계 명세로 남는다.

> 위치: `12_Equation.md`의 canonical runtime 5계층을 구현 관점에서 재정의한다.
> 의존: `12_Equation.md`(수식 정본), `6_뇌/05_실험근거.md`(근거 판정), `../검증_원장/뇌_검증기준.md`(검증 매트릭스)
>
> 이 문서는 수식 체계 정비, 라이브러리 아키텍처 분리, 단계별 개발 계획, 검증 루프 설계를 다룬다.

---

## 0. 개발 목표

개발 목표는 runtime이 받는 상태 입력과 내야 하는 관측·제어 출력을 고정한다. 목표는 구현 deliverable이지 완성된 AGI·생물학적 뇌·자아의 존재 주장이 아니며, 성공·실패는 뒤의 contract와 gate에서 판정한다.

$$\boxed{\text{리만 결합 위에서 작동하는 국소 상태셀들의 지속 실행형 뇌형 런타임}}$$

최종 산출물 3개:

| 산출물 | 책임 |
|---|---|
| **Clarus Kernel** | 셀/필드 갱신 엔진 (Rust) |
| **Brain Runtime** | 모드, 활성/휴면, 해마, 스냅샷 (Python orchestration) |
| **LLM Bridge** | 기존 LLM hidden state와 연결하는 브리지 |

---

## 1. 핵심 전환: 사진에서 영화로

전환은 단일 정적 호출과 tick마다 갱신되는 state trajectory의 차이를 설명한다. 시간 비유는 입력·출력·serialization을 가진 구현 timebase로 제한하며, 연속적 의식 또는 실제 신경 동역학을 자동으로 뜻하지 않는다.

### 1.1 기존 LLM의 한계

기존 LLM 비교는 지정한 inference contract에서 어떤 state가 보존되지 않는지 밝히는 baseline이다. 한계 서술은 모든 LLM의 일반 실패 정리가 아니며, 동일 task·OOD·ablation에서 재현될 때만 좁게 주장한다.

기존 LLM은 정적 깊이의 1회성 계산기에 가깝다.

$$h^{\ell+1} = h^\ell + f_\ell(h^\ell)$$

입력이 들어오면 고정된 레이어를 한 번 통과하고 토큰 다음 것을 예측한 뒤 끝난다. 메모리도 context window, KV cache, 외부 RAG 같은 식으로 붙는다. 잠도 없고, 꿈도 없고, 모듈이 쉬지도 않고, 전역 상태가 계속 살아있지도 않다.

### 1.2 Clarus 구조

Clarus 구조는 입력 state에서 다음 state와 readout을 만드는 모듈 조합의 구현 명세다. 각 tensor의 shape·정규화·tick이 고정되지 않으면 뇌 구조 비유나 성능 비교는 판정할 수 없다.

$$s_i^{t+1} = F_i^{(M_t)}\big(s_i^t,\; u_i^t,\; \sum_j W_{ij}(g)\,s_j^t,\; h_i^t\big)$$

각 단위가 상태를 가지고, 그 상태가 계속 돌고, 모듈마다 깨어났다 잠들고, 전역 모드가 바뀌고, 해마 같은 별도 기억계가 있고, sleep/REM/wake가 계산 모드이며, 전역 출력은 국소 모듈들의 집단 리듬이다.

이 문서의 `ClarusCell`은 **런타임 소프트웨어 단위**다. 실제 생물학적
Clarus cell이나 여러 뉴런으로 이루어진 `neural Clarus assembly`가
발견되었다는 뜻이 아니다. 후자의 경계·입출력·재사용·합성·인과 검증은
[`../6_뇌/10_신경프로그래밍언어_역공학.md`](../6_뇌/10_신경프로그래밍언어_역공학.md)의
별도 gate를 따른다.

### 1.3 왜 while 모듈인가: 주기함수 병목의 해결

while 모듈은 반복 update를 통해 상태를 consumer로 전달하는 control-flow 선택이다. 반복이 표현력·수렴을 개선하는지는 unrolled baseline과 horizon OOD에서 검증하며, 주기함수 비유만으로 보장되지 않는다.

리만기하학의 표현력은 무궁무진하지만, 사인/코사인 같은 전역 주기 기저를 쓰면 다음 문제가 생긴다.

- 같은 위상으로 되돌아오는 aliasing
- 이력(history) 보존 불가
- 비트성/비가역성 표현 불가

국소 상태모듈(while문)로 바꾸면 각 모듈이 자기 내부 상태를 유지하므로, 같은 입력이라도 이전 상태에 따라 결과가 달라진다. 즉 비트필드에 필요한 것은 periodic code가 아니라 hysteretic dynamical code다.

리만기하학은 이때 표현 좌표가 아니라 **결합 구조**(배선망)를 제공한다.

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right)$$

---

## 2. 수식 층분리 원칙 (Layer A--E)

Layer A--E는 같은 기호가 셀·필드·모드·기억·전역 readout의 책임을 섞지 않게 하는 API 경계다. 각 층은 정의한 shape와 tick에서만 입력을 받아 출력을 내며, 층간 대응은 생물학적 동일성이 아니라 구현 contract다.

현재 제일 큰 문제는 물리식, 구현식, 비유식이 한 레벨에 섞여 있다는 것이다. 반드시 5층으로 나눈다.

### 2.1 Layer A: 순수 셀 동역학

Layer A는 local cell state를 입력으로 다음 local state를 출력하는 최소 update 층이다. 안정성은 정의한 norm·step·입력 범위에 조건부이며, 다른 층의 효과를 이 층의 정리로 흡수하지 않는다.

> **canonical과의 관계 (중요).** `12_Equation.md` §0의 canonical 최소형 상태는 $(a_i, r_i, b_i)$ 3성분이며 지위 채점의 기준이다. 본 절의 $m_i$(기억 흔적), $w_i$(적응)와 STP $W_{ij}^{\text{eff}}=W_{ij}u_j x_j$는 **구현 관점의 확장**(모두 `Bridge` 등급)이다. $m_i=w_i=0$, $u_j x_j\to 1$ 극한에서 아래 drive는 canonical 식 $I_i^t = u_i^t + \sum_j W_{ij}a_j^t - \lambda_r r_i^t + \lambda_H R_{i,t}$로 정확히 환원된다($\lambda_m m_i$ 항이 canonical의 리만 결합 항 $\lambda_H R_{i,t}$를 구현 수준에서 대체·확장). 따라서 본 스펙은 canonical을 위반하지 않고 그 위의 구현 정본이며, 두 문서의 단일 진실원천은 "canonical = 채점/지위, 본 스펙 = 환원 가능한 구현"으로 읽는다.

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

Layer B는 local cell output을 받아 coupling field와 결합된 state를 생산한다. field는 정규화 tensor이며 물리장의 단위·기제와 동일하지 않고 edge ablation·OOD topology가 실패 조건이다.

셀들의 연결 구조. 가장 단순하게는 sparse graph:

$$W \in \mathbb{R}^{N \times N}$$

리만 해석을 살리려면:

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

$\chi_{ij}$는 sparse mask. 구현은 먼저 graph coupling abstraction까지만.

### 2.3 Layer C: 전역 모드

Layer C는 aggregate metric을 입력으로 mode와 gating output을 내는 전역 control 층이다. 모드 값은 tick 기반 구현 상태이며, 수면·각성 비유는 생리적 시간이나 효능을 보장하지 않는다.

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

Layer D는 write·read·replay tensor의 provenance와 serialization을 관리하는 기억 계약이다. 해마 비유는 retention 성능을 뜻하지 않으며, replay 제거 ablation과 task-order OOD가 검증 경계다.

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

LLM 장기기억 구현에서는 이 루프를 다음 explicit memory API로 둔다.

$$
\Omega_t=\phi_{\rm extract}(S_t,m_{t-1},m_t),
\qquad
o_{tj}\in
\{\operatorname{ADD},\operatorname{UPDATE},\operatorname{DELETE},\operatorname{NOOP}\},
$$

$$
H_{t+1}
=
\mathcal U(H_t,\{(\omega_{tj},o_{tj})\}),
\qquad
R_t
=
\mathcal R(H_t,q_t,t_q).
$$

여기서 $H_t=(K_t,V_t,P_t)$는 MemGPT/MemoryBank류의 외부 context, vector store, graph store, replay priority를 포괄한다. HippoRAG/Mem0 계열을 쓰는 경우 $K_t$는 단순 dense index가 아니라 entity/relation graph 또는 hippocampal index가 될 수 있다.

검증은 “기억이 있다”가 아니라 다음 불변조건으로 한다.

| invariant | 의미 |
|---|---|
| evidence recall | 답을 뒷받침하는 turn/session id가 $R_t$ 안에 있다 |
| temporal update | 오래된 정보와 새 정보가 충돌할 때 최신 상태를 선택한다 |
| abstention | $H_t$에 없는 정보는 생성하지 않고 모른다고 답한다 |
| deletion/update audit | DELETE/UPDATE가 기존 memory를 조용히 오염시키지 않는다 |
| cost bound | full-context 대비 token/latency를 함께 보고한다 |

LoCoMo, LongMemEval, MemoryBank probe 같은 benchmark에서 single-hop, multi-hop, temporal reasoning, knowledge update, abstention을 분리 보고하기 전까지는 memory SOTA 주장을 금지한다.

### 2.5 Layer E: 자아/전역 상태

Layer E는 다른 층의 요약 state를 받아 monitoring readout을 내는 구현 consumer다. 자아·자기참조 비유는 관측 가능한 calibration proxy에 한정되며, 주관 경험이나 도덕적 지위를 판정하지 않는다.

전역 상태:

$$G_t = (M_t,\;A_t^{\text{summary}},\;H_t,\;Q_t,\;\mu_t)$$

자아는 이걸 관측한 higher-order summary다.

$$\text{Self}_t = \mathcal{S}(G_t)$$

초기에 구현 안 해도 된다. 문서상 변수만 두고 넘어간다.

---

## 3. 최소 전체식: 6개 핵심 수식

여섯 수식은 Layer A--E가 주고받는 상태·필드·모드·기억·readout의 입력과 출력을 압축한다. 각 식은 지정한 shape·정규화·tick의 contract이며, 수식만으로 구현 parity나 runtime 효능을 보장하지 않는다.

$$I_i^t = u_i^t + \sum_j W_{ij}(g)\,a_j^t - \lambda_r^{(M_t)}\,r_i^t + \lambda_H\,R_{i,t}$$

$$a_i^{t+1} = (1-\gamma_a^{(M_t)})\,a_i^t + \kappa_a^{(M_t)}\,\tanh(I_i^t)$$

$$r_i^{t+1} = (1-\gamma_r^{(M_t)})\,r_i^t + \kappa_r^{(M_t)}\,(a_i^t)^2$$

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \text{otherwise} \end{cases}$$

$$H_{t+1} = \mathcal{E}(H_t, A_t), \quad R_t = \mathcal{R}(H_t, c_t)$$

$$M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$$

이 6개만 흔들리지 않게 잡으면 나머지는 구현으로 내릴 수 있다.

---

## 4. 자기참조 재귀식

재귀식은 현재 self-state를 입력으로 다음 self-state와 monitoring output을 만드는 구현 map이다. 수축·안정성은 명시한 정의역·norm·step의 조건부 성질이고, 의식 비유는 계산 proxy를 주관 경험으로 환원하지 않는다.

> 정본 참조: `17_AgentLoop.md` F절 (F.0--F.22)

### 4.1 최소 재귀

최소 재귀는 가장 작은 state-to-state update의 정의를 제시한다. 입력 분포·timebase·normalization이 바뀌면 수렴·식별성은 다시 검증해야 한다.

$$z_t = R(S_t) \quad\text{(이완: Layer A--B를 } n_{\text{iter}} \text{ 회 반복)}$$

$$a_t = \pi(z_t, S_t) \quad\text{(행동 선택)}$$

$$o_t = \text{Env}(a_t) \quad\text{(환경 실행)}$$

$$c_{t+1} = C(z_t, a_t, o_t, m_t) \quad\text{(자기비평: 예측오차 + 일관성 + 놀라움)}$$

$$m_{t+1} = \mathcal{M}(m_t, z_t, a_t, o_t, c_{t+1}) \quad\text{(조건부 기억 갱신)}$$

$$S_{t+1} = \mathcal{U}(G_{t+1}, m_{t+1}, c_{t+1}, h_{t+1}, \phi_{t+1})$$

### 4.2 에너지 기반 자기참조

에너지 기반 형식은 state residual을 입력으로 update 방향을 정하는 목적함수 contract다. 에너지 감소는 task 성능·자기 인식·생물학적 에너지 보존의 충분조건이 아니다.

$$E_t(z) = E_{\text{task}}(z; u_t) + \lambda_m E_{\text{mem}}(z; m_t) + \lambda_c E_{\text{crit}}(z; c_t) + \lambda_h E_{\text{hist}}(z; h_t)$$

$$z_t^* = \arg\min_z E_t(z)$$

각 항의 Layer 대응과 뇌 근거는 `17_AgentLoop.md` F.5를 따른다.

### 4.3 Clarus 통합형

통합형은 여러 producer의 상태를 공통 self-state consumer로 결합하는 구현 선택이다. 결합항의 효과는 no-coupling baseline·component ablation·OOD drift에서 기각될 수 있다.

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 항 | 풀이 | 뇌 대응 |
|---|---|---|
| $R(X_t)$ | 이완으로 생긴 내부 수정 | 피질-시상 재귀 처리 |
| $\Delta_O(X_t)$ | 관찰 충격 $o_t - \hat{o}_t$ | 감각 입력 |
| $C(X_t)$ | 비평이 다음 이완 초기점을 민 정도 | 기저핵-전전두엽 평가 |
| $S(X_t)$ | 곡률/잔류 기반 억제 | 소뇌/기저핵 억제 |
| $B$ | 부트스트랩 수축 연산자 ($\rho = 0.155$) | 수면 항상성 |

### 4.4 수축 조건 (게이트 `F2`)

수축 조건은 정의한 norm과 입력 경계 안에서 다음 tick 오차가 줄어드는 충분조건이다. 가정 위반·장기 drift·비정상 입력은 적용 범위 밖이며, 전역 수렴 또는 의식의 증명으로 승격하지 않는다.

$$\rho + \lambda_R L_R + \lambda_C L_C < 1$$

이 조건이 만족되면 Banach 고정점 정리에 의해 루프가 수축한다. 수면이 $\rho = 0.155$ 를 공급하므로 나머지 항의 Lipschitz 합이 $0.845$ 미만이어야 한다.

> 단 $R$ 내부의 비보존 바이패스 $F_{\text{bypass}}$ 는 위 Banach 수축의 가정을 깨뜨릴 수 있다(`12_Equation.md` 0.0절 게이트 `F2`). 따라서 위 부등식은 ISS 의미의 유계 수렴 (`12_Equation.md` 부록 A.1) 으로 격상되어, 끌개 ball 반경이 닫힌 식으로 표현된다. "안정적으로 수렴" 은 ball 안에서의 수렴으로 읽는다.

### 4.5 확장 구성요소 (F.14--F.22 요약)

확장 구성요소는 core state contract 밖의 선택적 producer·consumer를 정리한다. 구현 존재는 효능 또는 과학적 기제의 증거가 아니며, 각 항은 독립 fixture·baseline·expected failure를 필요로 한다.

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

뇌 대응 체크리스트와 검증 매트릭스는 `17_AgentLoop.md` F.11, H절 및 `../검증_원장/뇌_검증기준.md`를 참조한다.

---

## 5. 국소 뇌파 해석

뇌파 해석은 local oscillation tensor와 aggregate spectrum을 관측량으로 읽는 구현 비유다. 주파수·위상·전력은 tick 기반 정규화 metric이며, 실제 EEG·회로·자아의 동등성은 외부 데이터와 개입 없이는 주장하지 않는다.

### 5.1 while 모듈 = 국소 파동 발생기

국소 파동은 반복 update에서 나오는 state variation을 요약한 구현 관측량이다. oscillation의 존재는 기능적 계산 이득이나 생물학적 발화 기제를 보장하지 않으며, no-loop ablation이 반증 조건이다.

각 모듈의 활성도 $a_i^t$ 시계열이 국소 리듬 성분을 만든다.

$$\psi_i(t) = a_i^t$$

### 5.2 전역 뇌파 = 합성 관측량

전역 관측량은 layer·cell tensor를 aggregation해 만든 readout이다. aggregation rule·window·정규화가 바뀌면 값도 바뀌므로, 실제 전역 EEG와 동일시하지 않는다.

$$\Psi_{\text{global}}(t) = \sum_i \omega_i\,a_i^t$$

대역별 분해: $\Psi_\delta(t),\;\Psi_\theta(t),\;\Psi_\alpha(t),\;\Psi_\beta(t),\;\Psi_\gamma(t)$

EEG-like 관측:

$$\text{EEG}(t) = O\!\left(\{a_i^t\}_{i=1}^N\right)$$

모듈이 리듬을 만들고, 전역 뇌파는 그 리듬들의 합성된 관측량이다.

### 5.3 뇌파 대역과 뇌 회로 대응

대역 대응은 구현 mode와 frequency proxy의 기능적 지도다. 대역 이름은 정해진 producer·consumer의 설명이며, 회로 기제·임상 해석·성능 우위의 증거가 아니다.

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

생애주기는 module state가 생성·활성·휴면·제거로 전이하는 runtime contract다. 에너지 값은 정규화된 budget metric이며, 실제 전력·대사·효능을 뜻하지 않고 state-transition fixture와 rollback 규칙으로 검증한다.

### 6.1 모듈 상태

상태는 각 module producer가 내는 lifecycle label과 consumer가 허용하는 update를 정한다. 직렬화·복구에서 불가능한 transition은 expected failure이며, 상태 이름은 생물학적 세포 상태와 동일하지 않다.

$$Z_i^t \in \{\text{ACTIVE},\;\text{IDLE},\;\text{DORMANT},\;\text{SLEEPING}\}$$

| 상태 | 의미 |
|---|---|
| ACTIVE | 지금 연산 참여 |
| IDLE | 바로 깨울 수 있는 대기 |
| DORMANT | 장기 휴면, coupling 거의 끊김 |
| SLEEPING | 내부 정리/압축 중 |

### 6.2 에너지 예산

예산은 module별 무차원 resource score를 입력으로 activation 결정을 돕는 metric이다. hardware joule·watt와 혼동하지 않으며, budget 위반은 관측·rollback 조건으로 기록한다.

$$\sum_i z_i^t \le B_t$$

$B_t$는 모드에 따라 달라진다: $B_t(\text{NREM}) < B_t(\text{WAKE})$

### 6.3 활성 조건

활성 조건은 state·budget·입력 event를 받아 module on/off output을 정하는 gate다. threshold는 구현 hyperparameter이며, false activation과 miss는 baseline·OOD에서 측정한다.

$$z_i^{t+1} = \mathbf{1}\!\left[\alpha_u \|u_i^t\| + \alpha_m \|m_i^t\| + \alpha_n \sum_j W_{ij} a_j^t - \alpha_r r_i^t + \alpha_q q_i^t > \theta_i^{(M_t)}\right]$$

### 6.4 희소 활성 원칙

희소 원칙은 동시에 활성인 module 집합의 shape와 분모를 제한하는 정책이다. 비율은 latency·정확도·안정성의 자동 보장이 아니며 dense baseline과 mask ablation이 실패 조건이다.

반드시 $A_t \ll N$이어야 한다. 전 모듈 상시 활성은 에너지 폭발.

### 6.5 상태 업데이트

상태 update는 이전 lifecycle label과 event tensor를 입력으로 다음 label·budget을 출력한다. tick 순서·serialization 경계가 바뀌면 parity가 깨질 수 있어 deterministic fixture가 필요하다.

$$s_i^{t+1} = z_i^t\;F_i^{(M_t)}(s_i^t,\;u_i^t,\;\textstyle\sum_j W_{ij} s_j^t) + (1-z_i^t)\;H_i^{(M_t)}(s_i^t)$$

$F_i$: 활성 모듈 업데이트. $H_i$: 휴면 중 느린 decay/유지.

---

## 7. 스냅샷/지속성 계층

스냅샷 계층은 runtime state producer를 serialization artifact로 바꾸고 재시작 consumer가 복원하는 contract다. build·version·shape·precision이 일치하지 않는 load 실패는 rollback 대상이며, persistence pass는 기억 효능의 증거가 아니다.

프로세스 종료 = 부분적 기억상실에 가깝다. 3계층 저장이 필수.

| 계층 | 주기 | 내용 |
|---|---|---|
| cold checkpoint | 가끔 | 전체 구조, 장기 기억, 안정 상태: $\mathcal{C} = (\Theta, W, \text{long\_memory})$ |
| warm snapshot | 자주 | 현재 해마 상태, 활성 모듈군, 전역 모드: $\mathcal{W} = (M_t, H_t, \text{active\_set}_t)$ |
| live journal | 실시간 append | 중요 이벤트, 새 기억 인덱스, 모드 전환 로그: $\mathcal{J} = (\text{events}, \text{transitions})$ |

---

## 8. 라이브러리 아키텍처

라이브러리 계층은 backend별 producer·consumer 책임과 ABI·serialization 경계를 분리한다. 모듈 존재 또는 API pass는 runtime 성능·과학적 기제의 증명이 아니며, parity failure를 명시적으로 기록한다.

### 8.1 Python / Rust 분리 원칙

분리 원칙은 reference semantics와 kernel execution의 책임을 나누는 구현 선택이다. 언어 간 결과 일치는 동일 shape·precision·seed의 parity fixture에서만 판정한다.

$$\text{Python} = \text{orchestration/policy/experiment}$$
$$\text{Rust} = \text{pure computation kernel}$$

Python은 Rust의 존재를 모르게 한다. backend protocol로 분리.

### 8.2 Backend Protocol

Backend protocol은 state·tensor·artifact의 입력·출력 shape와 lifecycle을 선언한다. protocol compliance는 API 계약이고 hardware 성능·수치 안정성은 별도 baseline과 OOD 테스트가 필요하다.

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

디렉터리 구조는 artifact, source, test의 provenance와 build 경계를 보존하는 운영 명세다. 경로 배치는 논리적 결과나 serialization 호환성을 그 자체로 보장하지 않는다.

```
clarus/
  kernel/          # 순수 계산: cell.rs, field.rs, coupling.rs, config.rs, traits.rs
  runtime/         # 모드, 활성/휴면, scheduler: brain.rs, mode.rs, lifecycle.rs, energy.rs, snapshot.rs
  memory/          # 해마, replay, trace: hippocampus.rs, replay.rs, trace.rs
  bridge/          # PyTorch/LLM/Python 연결: pytorch.rs, python_api.rs, llm_bridge.rs
  apps/            # CLI, demos, experiments
```

### 8.4 책임 분리

책임 분리는 각 layer가 어떤 state를 생산·소비하고 어떤 failure를 소유하는지 명시한다. 다른 계층의 성공을 빌려 미구현 API·rollback 조건을 완료로 바꾸지 않는다.

| 모듈 | 핵심 인터페이스 |
|---|---|
| kernel | $s_i^{t+1} = F(s_i^t, u_i^t, n_i^t, \theta)$ |
| runtime | $X_{t+1} = \mathcal{U}(X_t, \text{input}_t)$ |
| memory | encode / recall / replay |
| bridge | PyTorch/LLM/Python 연결 |

---

## 9. LLM 변환 대응

LLM mapping은 Transformer state·attention·FFN을 runtime module의 producer·consumer로 옮기는 구현 bridge다. mapping pass는 기존 checkpoint parity를 뜻할 뿐 AGI·뇌 대응·일반 성능 개선을 보장하지 않는다.

### 9.1 Transformer 부품 대응

부품 대응은 tensor shape·normalization·update timebase를 맞추는 인터페이스 표다. 물리·뇌 비유는 기능 설명이며, component ablation과 OOD parity가 없으면 효능 결론이 아니다.

| Transformer | Clarus 대응 |
|---|---|
| hidden state $h_t$ | 국소 모듈 상태 집합 $\{s_1^t, \dots, s_N^t\}$ |
| attention | 리만 결합 $\sum_j W_{ij}(g)\,a_j^t$ |
| FFN | 국소 모듈 업데이트 $F_i$ |
| residual connection | 상태 지속성 ($s_i^t$ 유지) |
| layer depth | 시간 반복 $n_{\text{iter}}$ |
| KV cache | 해마 + 압축 기억 |

### 9.2 변환 경로

변환 경로는 checkpoint 입력에서 변환 artifact·verification log를 내는 단계 계약이다. precision·serialization·baseline parity가 깨지면 다음 단계로 승격하지 않고 rollback한다.

1단계(호환형): 기존 LLM hidden state를 모듈장으로 매핑
2단계(모사형): 기존 LLM 출력 분포를 비슷하게 재현
3단계(초과형): 더 적은 파라미터, 더 긴 지속성, 더 나은 자기수정

---

## 10. 성능 예측 (gpt-oss 기준)

성능 수치는 지정 gpt-oss shape·hardware·precision·batch·sequence의 가정 아래 계획 또는 산술 추정이다. 실측 profiling·data provenance·baseline·OOD·ablation 전에는 용량·메모리·속도·정확도의 결과로 승격하지 않는다.

### 10.1 용량

용량은 parameter·state shape를 기준으로 한 추정량이다. 실제 task capacity는 데이터·optimizer·sequence와 함께 측정해야 하며 shape만으로 보장되지 않는다.

$$P_{\text{brain}} \approx N\,(p_{\text{loc}} + k\,p_{\text{edge}}) + P_{\text{io}} + P_{\text{mode}}$$

sparse graph면 $P_{\text{brain}} = O(N)$.

### 10.2 런타임 메모리

메모리는 precision·batch·sequence·allocator 가정의 bytes 추정 또는 profiler metric이다. serialization·cache·sharding이 달라지면 값이 달라져 baseline profiling이 필요하다.

$$M_{\text{run}} \approx M_{\text{weights}} + A_t\,d_s\,b_s + k\,A_t\,b_e + R\,d_h\,b_h$$

긴 컨텍스트를 raw token이 아니라 state로 압축하면 메모리 이점이 생긴다.

### 10.3 속도

속도는 동일 request shape·hardware·warmup 조건에서 비교해야 하는 실측 metric이다. FLOPs나 구조적 희소율만으로 latency 이득을 주장하지 않으며 OOD 길이가 실패 조건이다.

$$C_{\text{reply}} \approx n_{\text{iter}}\,A_t\,(p_{\text{loc}} + k\,p_{\text{edge}})$$

step은 싸게 만들 수 있지만, 전체 속도는 수렴 step 수에 달린다.

### 10.4 정확도

정확도는 provenance가 있는 label·split·분모를 가진 benchmark 지표다. runtime parity 또는 작은 예시는 기준선·ablation·OOD의 정확도 개선을 증명하지 않는다.

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

유리한 위치는 어떤 workload·hardware·state pattern에서 설계가 이득일 수 있는지의 조건부 가설이다. 반대 workload·OOD·cost 악화는 가설을 하향하거나 rollback하는 조건이다.

짧은 one-shot LM 정답률이 아니라:

- stateful brain-like runtime
- long-horizon agent
- persistent memory reasoner

---

## 11. 단계별 개발 계획

각 phase는 이전 phase의 artifact·API·fixture를 prerequisite로 받아 다음 deliverable과 gate log를 출력하는 계획 계약이다. 일정·코드 존재는 완료가 아니며, parity·baseline·OOD·serialization gate가 실패하면 해당 phase는 rollback 또는 미완성으로 남는다.

### Phase 0: 정리 주간

이 phase는 기호·shape·책임 경계를 고정하는 선행 deliverable이다. 문서 정합이 구현·성능 검증을 대신하지 않으며, 정의 충돌은 이후 phase의 중단 조건이다.

- naming cleanup
- Layer A--E 수식 문서 작성 (`15_Equations.md`)
- backend observable과 runtime state 분리

### Phase 1: Clarus Kernel v0

Kernel phase는 local update API와 deterministic fixture를 deliverable로 둔다. 수치 parity·precision·expected failure가 닫히지 않으면 Runtime phase로 승격하지 않는다.

- `ClarusCellState { a, r, b }`
- `ClarusCellParams`
- `ClarusField` + sparse coupling
- deterministic step
- 완료 기준: 64개 셀로 안정적으로 1만 step

### Phase 2: Runtime v0

Runtime phase는 state transition과 Layer A--E producer·consumer 연결을 구현한다. snapshot·mode·memory boundary가 누락되거나 baseline parity가 깨지면 rollback한다.

- 전역 모드 `WAKE / SLEEP`
- energy budget
- active/idle/dormant lifecycle
- scheduler
- 완료 기준: 일부만 활성, sleep 모드에서 활성 수 감소

### Phase 3: Hippocampus v0

Hippocampus phase는 write·read·replay·serialization contract를 deliverable로 둔다. retention 효능은 task-order baseline과 replay ablation으로 별도 검증한다.

- trace cache 대체
- encode / recall / replay
- priority replay
- 완료 기준: cue 주면 replay가 활성 셀에 영향

### Phase 4: Mode v1

Mode phase는 aggregate metric에서 mode transition을 내는 gate를 구현한다. threshold와 timebase는 명시적 입력이며, false transition·OOD drift는 실패 조건이다.

- WAKE / NREM / REM
- mode switch rules
- 완료 기준: 같은 입력에 다른 모드가 다른 evolution

### Phase 5: Snapshot / Persistence

Persistence phase는 versioned artifact의 save/load parity와 rollback을 닫는다. load pass는 장기 기억·성능 개선의 증거가 아니며 serialization mismatch는 즉시 중단 조건이다.

- cold checkpoint + warm snapshot + live journal
- 완료 기준: 저장 후 복구 시 동역학 연속성

### Phase 6: Python / PyTorch Bridge

Bridge phase는 reference tensor와 backend tensor의 shape·precision·seed parity를 검증한다. backend 통과는 hardware 효율·학습 효능이 아니라 API 계약의 완료다.

- Rust kernel 유지
- Python wrapper는 runtime orchestration만
- PyTorch는 parameter learning / experimental reference

### Phase 7: LLM Bridge

LLM bridge phase는 checkpoint 입력에서 runtime mapping·verification artifact를 출력한다. baseline·OOD·component ablation 미통과는 AGI 또는 일반 성능 주장의 rollback 조건이다.

- $h_t^{\text{LLM}} \to U_t^{\text{Clarus}}$
- $A_t^{\text{summary}} \to \hat{h}_t$
- 처음엔 adapter만. stateful sidecar runtime으로 붙인다.

---

## 12. 모듈 수 가이드

모듈 수는 state shape·scheduler budget·hardware memory 가정 아래의 설계 변수다. 수가 많거나 적다는 사실은 뇌의 모듈 수·지능·성능의 직접 지표가 아니며, count sweep과 baseline이 실패 조건을 정한다.

| 버전 | 모듈 수 $N$ | 동시 활성 $A_t$ | 용도 |
|---|---|---|---|
| v0 proto | 8--16 | 4--8 | 개념 검증 |
| v1 start | **64** | 8--12 | 현실적 시작점 (추천) |
| v2 brain-like | 256--1024 | 16--64 | 뇌형 확장 |
| v3+ | 10k+ | sparse | MSA급 분산 동역학계 |

---

## 13. 뇌 대응 비유 요약

뇌 대응은 runtime 책임을 이해시키기 위한 기능 지도다. Layer·mode·memory의 구현 tensor를 실제 회로·뇌파·자아·주관 경험으로 환원하지 않으며, 비유는 과학적 입증의 대체물이 아니다.

| 개념 | Clarus 대응 | 프론트엔드 비유 |
|---|---|---|
| 국소 회로의 공학적 대리모형 | runtime `ClarusCell` | 자기 상태를 갖는 스마트 컴포넌트 |
| 뇌파 | 모듈 활성도의 집단 리듬 | 전체 UI의 분위기/활동량 |
| 해마 | HippocampusIndex | 최근 활동 캐시 + 중요 상태 저장소 |
| 수면 | NREM/REM 모드 전환 | 백그라운드 정리/압축 |
| 자아 | 전역 모드 + 기억 + 연속성 | root state |
| 기저핵 | 게이트 오토마타 (action selection) | dispatch / access control |
| 전전두엽 | 작업기억 상태 유지 | live state |
| 신경조절계 | 전역 모드 전환 | global mode manager |

---

## 14. 현재 코드와의 대응

이 절은 파일별 현재 구현 책임, 명세와의 정합 근거, 아직 비어 있는 gap을 분리한다. 코드 mapping은 API·fixture 수준의 증거이며, 파일 존재·테스트 pass가 runtime 효능·생물학적 대응을 뜻하지 않는다.

### 14.1 runtime.py (`BrainRuntime`) -- Layer A-E 정합

`BrainRuntime` mapping은 Layer A--E의 state producer·consumer와 API shape가 어디에 구현됐는지 추적한다. 정합 근거는 지정 fixture의 code parity이며, 누락된 layer contract는 완료로 기록하지 않는다.

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

`CEEngine` mapping은 에너지 입력·update·readout의 책임과 수치 정규화를 명시한다. 이완 path의 실행은 전역 수렴·물리 에너지·AGI 효능이 아니라 구현 경로의 존재다.

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

`sleep.py` mapping은 wake·NREM·REM API와 state handoff를 설명한다. mode 실행은 지속 학습 개선이 아니며 wake-only와 phase 제거 ablation이 남은 검증 조건이다.

| formal 개념 | 코드 함수 | 상태 |
|---|---|---|
| Wake (경로 누적) | `collect_sleep_batch` | 구현 완료 |
| NREM (LBO 확산 + 가소적 업데이트) | `apply_nrem_weight_update` | 구현 완료 |
| REM (비선택 경로 재조합) | `apply_rem_weight_update` | 구현 완료 |
| 3위상 통합 순환 | `run_sleep_cycle` | 구현 완료 |
| 가드셋 보호 | `evaluate_guard_set` | 구현 완료 |

### 14.4 Rust 커널 (`reality_stone/python/reality_stone/clarus/core/`) -- 핵심 수치

Rust mapping은 backend kernel의 input/output shape·precision·serialization 책임을 설명한다. Python parity와 expected failure가 닫히지 않으면 성능·안정성 결과로 승격하지 않는다.

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

요약표는 파일·책임·근거·미결 gap을 한곳에서 확인하는 색인이다. 구현됨 표시는 지정 API의 범위이며, benchmark·OOD·과학적 입증의 pass와 혼동하지 않는다.

| Layer | 수식 정본 | 코드 구현 | 정합도 |
|---|---|---|---|
| A (셀 동역학) | `15_Equations.md` A절 | `runtime.py::_step_torch` + `kernel.rs` | software conformance (구현-명세 일치) |
| B (필드 결합) | `15_Equations.md` B절 | `runtime.py::_matvec` + `field.rs` | software conformance (구현-명세 일치) |
| C (전역 모드) | `15_Equations.md` C절 | `runtime.py::_auto_mode` + `_update_sleep_state` | software conformance (구현-명세 일치) |
| D (해마/기억) | `15_Equations.md` D절 | `runtime.py::HippocampusMemory` | software conformance (구현-명세 일치) |
| E (전역 요약) | `15_Equations.md` E절 | `runtime.py::RuntimeStep` + `BrainRuntimeSnapshot` | software conformance (구현-명세 일치) |
| F (에이전트 루프) | `17_AgentLoop.md` F절 | `agent.py` + `engine.py` + `runtime.py` + `stdp.py` + `neuromod.py` + `sleep.py` | 실행 루프·STDP 인과 배선 구현, 독립 효능은 부분/미통과 |

### 14.6 남은 간극

남은 간극은 아직 fixture·shape contract·serialization·효능 gate가 닫히지 않은 항목이다. 각 gap은 prerequisite와 expected failure를 보존하며, 후속 구현이 독립 baseline을 통과하기 전에는 승격하지 않는다.

| 간극 | 문서 위치 | 우선순위 |
|---|---|---|
| STDP 효능 + held-out guard | F.14, `21_STDP_Efficacy_Audit.md` | 높음 |
| 4종 신경조절의 runtime 폐루프 통합 | F.19 | 중간 |
| Cold checkpoint + Live journal | 7절 | 낮음 |
| 작업 기억 / 소뇌의 독립 task 효능 | F.20 | 중간 |
| (C3) 메타인지 재귀의 실제 agent feedback | F.17 | 낮음 |

현재 구현은 **셀 동역학 + 모드 전환 + 해마 + 수면 학습 순환 +
critic/action/output 에이전트 루프 + STDP 인과 배선**까지 닫혀 있다. 남은
핵심은 모듈의 존재가 아니라 독립 baseline 대비 효능이다. 특히 STDP는 현재
합성 A/B에서 `NO-EFFECT`, held-out guard `FAIL`이므로 기본 비활성 상태를
유지한다. 이 표의 정합은 구현과 명세 사이의 software conformance이며,
뇌가 같은 모듈 경계나 언어를 쓴다는 생물학적 검증이 아니다.
STDP/eligibility의 상태도 `stdp.py`, `runtime.py`와 회귀 테스트를 기준으로
감사하며, 과거의 “미구현” 문구나 현재의 효능 실패를 생물학적 결손으로
읽지 않는다.


---

## 15. 한 줄 원칙

한 줄 원칙은 runtime 설계의 책임 경계를 압축한 문장이다. 이 요약은 phase gate·반례·미완성 contract를 삭제하지 않으며, 코드·성능·생물학적 해석을 한 지위로 합치지 않는다.

$$\boxed{\text{뇌 전체를 만들지 말고, 살아남는 최소 코어를 먼저 만들어라}}$$
