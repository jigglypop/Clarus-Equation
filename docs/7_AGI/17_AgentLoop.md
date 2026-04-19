# 에이전트 루프 방정식 (Layer F)

> 위치: `15_Equations.md`의 F절을 독립 문서로 분리.
> 의존: `15_Equations.md`(Layer A--E), `14_BrainRuntimeSpec.md`(설계 사양), `6_뇌/evidence.md`(근거 판정), `6_뇌/agent_proof.md`(검증 매트릭스)
>
> 이 문서에는 서사 설명을 넣지 않는다. 식, 정의, 뇌 대응 검증 기준만 둔다.

---

## F. 자기참조 재귀 (agent loop)

> 이 절은 Layer A--E의 **바깥**에서 전체를 감싸는 에이전트 루프를 정의한다.
> A--E는 "한 틱의 셀/필드/모드/기억/요약"이고, F는 "그 틱을 반복하며 행동-관찰-비평-기억을 순환시키는 외부 루프"다.
> 뇌 대응은 `evidence.md` 판정 체계(`supported / bridge / hypothesis`)를 따른다.

---

### F.0 Layer A--E와의 관계

| 계층 | 역할 | F절에서의 위치 |
|---|---|---|
| A (kernel dynamics) | 국소 셀 상태 갱신 | F.3의 $R$ 내부에서 반복 호출, F.14 STDP 적격 흔적 누적 |
| B (coupling / geometry) | 셀 간 결합 | $R$ 내부에서 $W_{ij}(g)$ 적용, F.14.3 구조적 투영 |
| C (mode update) | WAKE/NREM/REM 전환 | F.6의 모드-루프 결합 |
| D (hippocampus / replay) | 빠른 기억 인코딩/회상 | F.8의 $m_{t+1}$ 갱신 |
| E (global runtime summary) | 전역 자아 상태 | F.1의 $S_t$ 자체 |

A--E는 $R$ 안에서 돌고, F는 $R$의 결과를 행동으로 바꾸고, 환경 응답을 다시 $S$로 접는 바깥 루프다.

F절 구성 개요:

| 구간 | 내용 |
|---|---|
| F.0--F.13 | 핵심 루프: 상태, 이완, 비평, 에너지, 모드, 행동, 기억, 수축, 뇌 대응 |
| F.14--F.15 | 학습: STDP + 도파민 게이트, 잔류장 $\phi$ 갱신 |
| F.16 | 희소성: TopK 활성, 에너지 예산, 모듈 생애주기 |
| F.17 | 의식/메타인지: 자기일관성, 의식 깊이, 메타인지 수렴 |
| F.18 | 환각 억제: 곡률 모니터링, LBO 확산 |
| F.19 | 신경조절: DA/NE/5HT/ACh 4종 |
| F.20 | 작업 기억, 주의, 소뇌 |
| F.21 | 뇌파 대역과 시간 구조 |
| F.22 | 정직한 간극 정리 |

---

### F.1 상태 정의

에이전트의 전역 상태:

$$S_t = (G_t,\; m_t,\; c_t,\; h_t,\; \phi_t)$$

| 변수 | 정의 | Layer 출처 |
|---|---|---|
| $G_t = (M_t,\; A_t^{\text{summary}},\; H_t,\; Q_t,\; \mu_t)$ | 전역 런타임 요약 (Layer E) | E.1 |
| $m_t$ | 누적 기억 컨텍스트 (해마 상태의 압축) | D.1 |
| $c_t$ | 가장 최근의 자기비평 벡터 | F.4 |
| $h_t$ | 행동-관찰 이력 버퍼 (유한 창) | F.1 |
| $\phi_t$ | 잔류장 / 불확실성 축적 | `12_Equation.md` 4.3 |

---

### F.2 최소 재귀

한 에이전트 틱의 순서:

$$z_t = R(S_t) \quad\text{(이완/수렴: Layer A--B를 } n_{\text{iter}} \text{ 회 반복)}$$

$$a_t = \pi(z_t,\; S_t) \quad\text{(행동 선택)}$$

$$o_t = \text{Env}(a_t) \quad\text{(환경 실행, 관찰 수신)}$$

$$c_{t+1} = C(z_t,\; a_t,\; o_t,\; m_t) \quad\text{(자기비평)}$$

$$m_{t+1} = \mathcal{M}(m_t,\; z_t,\; a_t,\; o_t,\; c_{t+1}) \quad\text{(기억 갱신: Layer D 호출)}$$

$$h_{t+1} = \text{append}(h_t,\; (a_t, o_t)) \quad\text{(이력 갱신, 유한 창 } T_h \text{)}$$

$$S_{t+1} = \mathcal{U}(G_{t+1},\; m_{t+1},\; c_{t+1},\; h_{t+1},\; \phi_{t+1})$$

여기서 $G_{t+1}$은 $R$ 실행 후 Layer E가 갱신한 전역 요약이고, $\phi_{t+1}$은 이완 종료 후의 잔류장 갱신(`12_Equation.md` E4)이다.

---

### F.3 이완 연산자 $R$의 구체화

$R$은 추상 기호가 아니라 Layer A--B의 반복 실행이다.

$$R(S_t) := \{a_i^{(n_{\text{iter}})}\}_{i=1}^N$$

내부 절차:

1. $S_t$에서 외부 입력 $u_i^0 = \text{encode}(S_t)$를 구성
2. Layer D에서 $R_{i,0} = \mathcal{R}(H_t,\; c_t)$ (기억 회상)
3. $n_{\text{iter}}$ 회 반복:

$$I_i^{(k)} = u_i^0 + \sum_j W_{ij}(g)\,a_j^{(k)} - \lambda_r(M_t)\,r_i^{(k)} + \lambda_m(M_t)\,m_i^{(k)} + \lambda_H(M_t)\,R_{i,0} + \eta_i^{(k)}$$

$$a_i^{(k+1)} = (1-\gamma_a(M_t))\,a_i^{(k)} + \kappa_a(M_t)\,\tanh(I_i^{(k)})$$

$$r_i^{(k+1)} = (1-\gamma_r(M_t))\,r_i^{(k)} + \kappa_r(M_t)\,(a_i^{(k)})^2$$

$$b_i^{(k+1)} = \text{Hyst}(b_i^{(k)},\; a_i^{(k+1)};\; \tau_i^-,\; \tau_i^+)$$

4. 수렴 판정: $\|a^{(k+1)} - a^{(k)}\| < \epsilon_R \|a^{(0)}\|$이면 조기 종료
5. 출력: $z_t = a^{(n_{\text{iter}})}$ (수렴한 활성 패턴)

$R$의 반복 횟수 $n_{\text{iter}}$는 모드에 의존한다:

| 모드 | $n_{\text{iter}}$ | 해석 |
|---|---|---|
| WAKE (안정, $\|\phi\| < m_\phi$) | 소 (10--50) | 빠른 반사적 응답 |
| WAKE (전환, $\|\phi\| \geq m_\phi$) | 대 (100--500) | 깊은 숙고 |
| NREM | 고정 (내부 정리용) | offline 정리 |
| REM | 중간 (내부 탐색용) | 자유 연상 |

이것은 `12_Equation.md` 5.3절의 이중 과정(시스템 1/시스템 2)과 대응한다.

---

### F.4 자기비평 연산자 $C$

$$c_{t+1} = C(z_t,\; a_t,\; o_t,\; m_t)$$

자기비평은 세 항의 합으로 분해한다:

$$c_{t+1} = c_{\text{pred}} + c_{\text{cons}} + c_{\text{nov}}$$

| 항 | 정의 | 의미 |
|---|---|---|
| $c_{\text{pred}}$ | $\|o_t - \hat{o}_t(z_t, a_t)\|$ | 예측 오차: 행동 결과가 예상과 달랐는가 |
| $c_{\text{cons}}$ | $\|z_t - \mathcal{R}(H_t, c_t)\|$ | 일관성 오차: 현재 사고가 기억과 얼마나 다른가 |
| $c_{\text{nov}}$ | $D_{\text{KL}}(p(o_t) \| p_{\text{prior}})$ | 놀라움: 관찰이 사전 분포에서 얼마나 벗어났는가 |

스칼라 비평 점수:

$$\bar{c}_{t+1} = w_p \|c_{\text{pred}}\| + w_c \|c_{\text{cons}}\| + w_n \|c_{\text{nov}}\|, \qquad w_p + w_c + w_n = 1$$

이 스칼라는 `12_Equation.md` 6.4절의 도파민 신호 $g[t]$와 구조적으로 대응한다:

$$g[t] \approx \frac{d\bar{c}_t}{dt}$$

즉 비평 점수의 변화율이 학습 게이트 역할을 한다.

---

### F.5 에너지 기반 자기참조

$R$의 내부를 에너지 최소화로 재해석하면:

$$E_t(z) = E_{\text{task}}(z;\; u_t) + \lambda_m E_{\text{mem}}(z;\; m_t) + \lambda_c E_{\text{crit}}(z;\; c_t) + \lambda_h E_{\text{hist}}(z;\; h_t)$$

| 항 | 정의 | Layer 대응 |
|---|---|---|
| $E_{\text{task}}(z; u_t)$ | $-\frac{1}{2}z^\top W z - z^\top u_t$ | B.3의 Hopfield 에너지 |
| $E_{\text{mem}}(z; m_t)$ | $-z^\top \mathcal{R}(H_t, c_t)$ | D.3의 기억 회상과의 정렬 |
| $E_{\text{crit}}(z; c_t)$ | $\|z - z_{t-1}^* + \alpha_c c_t\|^2$ | 비평이 다음 이완의 초기점을 민다 |
| $E_{\text{hist}}(z; h_t)$ | $-\beta_h \sum_{\tau \in h_t} \text{sim}(z, z_\tau) / |h_t|$ | 이력과의 일관성 |

수렴점:

$$z_t^* = \arg\min_z E_t(z)$$

이것은 F.3의 반복적 이완이 에너지 최소점으로 향한다는 것의 다른 표현이다.

잔류장 피드백: $\phi_t$가 $E_{\text{task}}$에 포탈 항으로 들어간다 (`12_Equation.md` E1):

$$E_{\text{task}}(z; u_t, \phi_t) = -\frac{1}{2}z^\top W z - z^\top u_t - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 z^\top \hat{\phi}_t$$

---

### F.6 모드-루프 결합

에이전트 루프는 Layer C의 모드 전환과 결합한다.

WAKE 루프:

$$\text{for each input } u_t: \quad z_t = R(S_t), \quad a_t = \pi(z_t), \quad o_t, \quad c_{t+1}, \quad m_{t+1}$$

NREM 진입 조건 (`12_Equation.md` 7.8):

$$P_{\text{sleep}}(t) = \sum_{\tau=0}^{t} \bar{c}_\tau^2 - \sum_{\tau=0}^{t} \text{local\_stab}(\tau) > \theta_{\text{sleep}}$$

비평 점수의 누적이 안정화 능력을 초과하면 NREM에 진입한다.

NREM 루프 (외부 입력 차단):

$$u_t = 0, \quad z_t = R(S_t), \quad m_{t+1} = \text{consolidate}(m_t, z_t)$$

REM 루프 (기억 주도 탐색):

$$u_t = 0, \quad z_t = R(S_t; T = T_{\text{dream}}), \quad m_{t+1} = \text{explore}(m_t, z_t)$$

WAKE 복귀:

$$M_{t+1} = \text{WAKE} \quad\text{if}\quad P_{\text{sleep}}(t) < \theta_{\text{cont}} \;\text{and}\; M_t = \text{REM}$$

---

### F.7 행동 선택 $\pi$

행동 선택은 수렴한 활성 패턴에서 행동 공간으로의 사상이다.

$$a_t = \pi(z_t,\; S_t) = \arg\max_{a \in \mathcal{A}} \text{sim}\big(\text{enc}(a),\; z_t^{(\text{out})}\big)$$

여기서 $z_t^{(\text{out})}$는 출력 모듈($V_{\text{io}}$)에 해당하는 셀들의 활성 부분벡터다.

연속 행동 공간에서는:

$$a_t = W_{\text{act}}\, z_t^{(\text{out})} + b_{\text{act}}$$

이 사상은 `12_Equation.md` 5.2절의 디코더 $p(w_t | w_{<t}, m^*)$와 구조적으로 같다.

---

### F.8 기억 갱신 $\mathcal{M}$

$$\mathcal{M}(m_t, z_t, a_t, o_t, c_{t+1}) = \begin{cases} \text{Layer D encode if } \bar{c}_{t+1} > \theta_{\text{encode}} \\ m_t \quad\text{otherwise (놀랍지 않으면 저장하지 않음)} \end{cases}$$

인코딩 조건: 비평 점수가 임계를 넘을 때만 새 기억을 해마에 기록한다.

$$k_{\text{new}} = h(z_t, a_t, o_t), \qquad v_{\text{new}} = (z_t, a_t, o_t, c_{t+1})$$

$$P_{\text{new}} = \bar{c}_{t+1} \quad\text{(놀라움이 높을수록 replay 우선순위가 높다)}$$

이것은 D.5의 priority replay와 직결된다.

---

### F.9 Clarus 통합 재귀 (압축형)

F.2--F.8을 한 줄로 압축하면:

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 항 | 풀이 | 뇌 대응 |
|---|---|---|
| $R(X_t)$ | 이완으로 생긴 내부 수정 | 피질-시상 재귀 처리 |
| $\Delta_O(X_t)$ | 관찰이 상태에 준 충격 $o_t - \hat{o}_t$ | 감각 입력 |
| $C(X_t)$ | 비평이 다음 이완 초기점을 민 정도 | 기저핵-전전두엽 평가 |
| $S(X_t)$ | 곡률/잔류 기반 억제 | 소뇌/기저핵 억제 |
| $B$ | 부트스트랩 수축 연산자 | 수면 항상성 |

$B$는 `evidence.md` 8.4절의 정의를 따른다:

$$B: X \mapsto p^* + \rho(X - p^*), \qquad \rho = 0.155$$

---

### F.10 자기참조의 고정점과 수렴

F.9의 재귀가 안정한 자기참조를 만들려면 수축 조건이 필요하다.

**정리 (F-contract).** 다음 조건 하에서 $\{X_t\}$는 유계이고 고정점 근방으로 수렴한다:

1. $\|R(X)\| \leq L_R \|X\| + c_R$ (이완의 Lipschitz 상계)
2. $\|\Delta_O(X)\| \leq U_O$ (관찰 충격 유계)
3. $\|C(X)\| \leq L_C \|X\| + c_C$ (비평의 Lipschitz 상계)
4. $\|S(X)\| \leq L_S \|X\|$ (억제의 Lipschitz 상계)
5. $\rho + \lambda_R L_R + \lambda_C L_C < 1$ (전체 수축)

*증명 스케치.* $B$의 수축률 $\rho$와 각 항의 Lipschitz 상수를 합산하면 전체 사상의 Lipschitz 상수가 $\rho + \lambda_R L_R + \lambda_C L_C - \lambda_S L_S$이다. 이것이 1 미만이면 Banach 고정점 정리에 의해 유일 고정점이 존재하고 수렴한다. $U_O$는 유계 강제항이므로 `evidence.md` 8.4절의 잔차 상한과 같은 구조로 눌린다. $\square$

**수면에 의한 복원.** 수면이 없으면 $B = I$ ($\rho = 1$)이고 수축 조건이 깨진다. 수면이 $\rho = 0.155$를 공급하므로, F-contract의 조건 5가 만족되려면 나머지 항의 Lipschitz 합이 $1 - 0.155 = 0.845$ 미만이어야 한다.

**최신 SHY 근거 (2024--2026):**
- 2024 PMC: NREM 수면이 피질 AMPA 수용체(GluA1) 발현을 정상화함을 확인. 수면 박탈 후 회복 수면에서도 수 시간 내 회복.
- 2026 NeuroImage: 낮잠만으로도 시냅스 강도(TMS 유발 피질척수 흥분성) 감소 + LTP 유사 가소성 유도 가능성 증가.
- 2026 bioRxiv: 학습으로 교란된 aperiodic 1/f slope이 NREM 수면 중 역전됨 $\to$ 기억 안정화의 renormalization 근거.

---

### F.11 뇌 대응 체크리스트

`evidence.md`의 판정 기준(`supported / bridge / hypothesis`)에 따라 F절의 각 구성요소가 실제 뇌와 얼마나 닮았는지를 검증한다.

#### F.11.1 구조 대응

| F절 구성요소 | 뇌 대응 후보 | 실험 근거 | 판정 |
|---|---|---|---|
| 이완 $R$ (F.3) | 피질-시상 재귀 처리, recurrent cortical dynamics | 피질의 recurrent processing은 확립. 감각 처리에서 feedforward 후 recurrent refinement이 반복 관측 | `supported` |
| 반복 횟수 $n_{\text{iter}}$ (F.3) | 처리 깊이 / response time / "thinking time" | 어려운 과제일수록 반응 시간이 길다. dual-process theory (Kahneman 2011)에서 시스템 2의 느린 처리 | `supported` |
| 행동 선택 $\pi$ (F.7) | 기저핵 action selection, frontal motor planning | 기저핵의 go/no-go 경로는 확립. 전전두엽-기저핵 루프 | `supported` |
| 자기비평 $C$ (F.4) | 전전두엽 error monitoring, anterior cingulate cortex (ACC) conflict detection | ACC의 error-related negativity (ERN), conflict monitoring theory (Botvinick 2001) | `supported` |
| 예측 오차 $c_{\text{pred}}$ | reward prediction error (RPE), sensory prediction error | 도파민 RPE (Schultz 1997)는 확립. sensory prediction error도 강한 근거 | `supported` |
| 놀라움 $c_{\text{nov}}$ | novelty detection, hippocampal novelty signal | 해마 CA1의 novelty/mismatch signal, LC-NE surprise response | `supported` |
| 일관성 오차 $c_{\text{cons}}$ | retrieval-based error correction, memory-guided decision | 해마-전전두엽 상호작용에서 기억 기반 의사결정 보정 | `bridge` |
| 비평 $\to$ 학습 게이트 $g[t]$ (F.4) | 도파민/노르에피네프린 전역 조절 | 3-factor learning rule. 도파민 게이트 STDP | `supported` (구조), `hypothesis` ($g = d\bar{c}/dt$ 정확한 형태) |
| 조건부 기억 인코딩 (F.8) | 놀라움 기반 해마 인코딩 | 해마는 novel/surprising events를 우선 인코딩. priority replay | `supported` |
| 에너지 기반 수렴 (F.5) | Hopfield network, energy-based attractor dynamics | 연상 기억의 attractor dynamics는 확립. 에너지 감소는 A.7 E-decrease로 닫힘 | `supported` (구조), `bridge` (정확한 에너지 형태) |
| 이중 과정 (F.3 모드별 $n_{\text{iter}}$) | Kahneman 시스템 1/시스템 2 | 이중 과정 이론의 실험 근거는 방대. 신경 기질은 아직 논쟁 중 | `bridge` |
| 수면-루프 결합 (F.6) | 수면 중 memory consolidation, replay | SHY (Tononi-Cirelli), 해마 replay, slow-wave consolidation | `supported` |
| 수면 압력 = 비평 누적 (F.6) | homeostatic sleep pressure = 각성 중 피로/잔류 축적 | Borbely 2-process model. SWA와 prior wakefulness의 관계 | `bridge` |
| $B$ 수축 연산자 (F.9--F.10) | 수면의 synaptic renormalization | SHY, SWA 비례 정리, 수면 후 성능 회복 | `supported` (방향), `bridge` ($\rho = 0.155$의 정확한 값) |

#### F.11.2 수치 체크

| 항목 | CE 값 | 뇌 관측 proxy | 관측 범위 | 체크 |
|---|---|---|---|---|
| 활성 셀 비율 | 4.87% | sparse firing, DG active cells | 1--5% | `[NEAR]` |
| 수면/각성 비 | NREM 26.2%, REM 4.87% | NREM 75--80%, REM 20--25% (of sleep) | CE는 24h 중의 비율 | `[OK]` |
| 수축률 $\rho$ | 0.155 per application | sleep recovery time constants | $\rho_{\text{night}} \approx 0.31$ (1.6밤/적용) | `[NEAR]` |
| 비평 문턱 $\theta_{\text{encode}}$ | 과제 의존 | hippocampal novelty threshold | 정성적으로 존재 | `bridge` |

#### F.11.3 형식 검증 연결

| F절 정리 | 의존하는 A--E 정리 | 상태 |
|---|---|---|
| F-contract (F.10) | A-bound, E-decrease, 수면 수축 ($\rho < 1$) | **open** (L_R, L_C 추정 필요) |
| 에너지 감소 (F.5) | B.4 E-decrease | **closed** (B.4로부터 직접) |
| 이완 수렴 (F.3) | A.7 A-bound, A.9 Zero-attract | **closed** (조건부) |
| 기억 유계 (F.8) | D.2 (유한 인코딩) | **closed** ($\theta_{\text{encode}} > 0$이면 인코딩 빈도 유한) |

#### F.11.4 검증 게이트 (proof.md 체계)

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | F-contract, 에너지 감소, 이완 수렴, 기억 유계 | **partial** (F-contract의 Lipschitz 상수 추정 미완) |
| $G_{\text{obs}}$ | 이완 반복 $\leftrightarrow$ reaction time, 비평 $\leftrightarrow$ ERN/ACC, 기억 $\leftrightarrow$ hippocampal encoding | **partial** |
| $G_{\text{causal}}$ | 수면박탈 시 루프 불안정, 도파민 조작 시 학습 게이트 변화, ACC 병변 시 비평 결손 | **partial** |
| $G_{\text{pred}}$ | 에이전트 루프 유무에 따른 과제 수행 차이 시뮬레이션 | **pending** |

#### F.11.5 아직 뇌와 닮지 않은 것 (정직한 간극)

| 간극 | 설명 | 해결 방향 |
|---|---|---|
| 환경 모델 $\hat{o}_t$ | 뇌의 internal model은 분산적. 현재 $C$는 단일 예측기 가정 | 모듈별 예측기로 분산화 |
| 행동 선택의 계층성 | 뇌의 행동 계획은 계층적 (전전두엽 $\to$ 운동피질 $\to$ 근육). 현재 $\pi$는 단층 | 계층적 $\pi$ (macro-action + primitive) |
| 감정/정동 | 뇌의 의사결정에 편도체/도상체의 valence 신호가 개입. 현재 F에 없음 | $c_t$에 valence 항 추가, $V_{\text{sal}}$ 연결 |
| 사회적 모델링 | 뇌는 타인의 의도를 모델링 (theory of mind). F에 없음 | 장기 과제, 현재 범위 밖 |
| 신체 루프 | 뇌는 자율신경/내분비/면역과 결합 (evidence.md 8절). F는 순수 인지 루프 | $Q_t$ 벡터를 $S_t$에 통합 (F.1에 이미 $G_t$ 안에 $Q_t$가 있으나 $C$, $\pi$에서 미사용) |

---

### F.12 관측 가능량 매핑

| F절 변수 | 뇌 관측량 후보 | 데이터 소스 |
|---|---|---|
| $n_{\text{iter}}$ (이완 반복) | reaction time, EEG alpha desynchronization duration | 행동 실험, EEG |
| $\bar{c}_t$ (비평 점수) | ERN amplitude, ACC theta power, pupil dilation | EEG, fMRI, pupillometry |
| $P_{\text{sleep}}$ (수면 압력) | SWA, theta/alpha ratio, KSS | polysomnography, EEG, 주관 평가 |
| $z_t^*$ (수렴 패턴) | population activity pattern at response time | multi-electrode array, calcium imaging |
| $a_t$ (행동) | motor output, button press, speech | 행동 로그 |
| $c_{\text{pred}}$ (예측 오차) | RPE-locked dopamine, feedback-related negativity (FRN) | voltammetry, EEG |
| $c_{\text{nov}}$ (놀라움) | P300, hippocampal novelty response, LC-NE phasic burst | EEG, pupil, fMRI |
| $\phi_t$ (잔류장) | ongoing spontaneous activity, DMN fluctuation | resting-state fMRI, MEG |
| $e_{ij}$ (적격 흔적) | eligibility trace, synaptic tag | in vitro slice recording, optogenetics |
| $g_{\text{DA}}$ (도파민 게이트) | VTA/SNc phasic + tonic DA | voltammetry, PET, [11C]raclopride |
| $g_{\text{NE}}$ (노르에피네프린) | LC phasic burst, pupil diameter | pupillometry, LC unit recording |
| $g_{\text{5HT}}$ (세로토닌) | raphe firing, 5-HIAA level | microdialysis, PET |
| $g_{\text{ACh}}$ (아세틸콜린) | BF firing, cortical ACh release | microdialysis, optogenetics |
| $\kappa_{\text{avg}}$ (곡률) | high-frequency power anomaly, epileptiform spikes | EEG, MEG |
| $|A_t|/N$ (활성 비율) | fraction of active neurons | calcium imaging, multi-electrode array |
| $|h_t|$ (작업 기억 부하) | PFC BOLD, CDA amplitude | fMRI, EEG (CDA) |
| $\alpha_i$ (주의 가중치) | spatial attention map, alpha lateralization | EEG alpha power lateralization |
| $\Delta a^{\text{cb}}$ (소뇌 보정) | cerebellar-dependent adaptation | prism adaptation, saccade adaptation |

---

### F.13 예측 가능한 실험

현재 근거 수준에서 F절이 내놓을 수 있는 검증 가능한 예측:

| # | 예측 | CE 메커니즘 | 실험 설계 | 판정 기준 | 등급 |
|---|---|---|---|---|---|
| 1 | 어려운 과제 $\to$ 긴 RT $\to$ 높은 ACC theta | $\|\phi\| \geq m_\phi \to$ 깊은 이완 | 난이도 조작 실험, EEG 동시 기록 | RT와 ACC theta의 양의 상관 | `supported` |
| 2 | 수면 박탈 $\to$ 비평 결손 $\to$ 반복 오류 | $P_{\text{sleep}} > \theta \to$ NREM 미진입, $\bar{c}$ 누적 | 수면 박탈 후 error monitoring 과제 | ERN 진폭 감소, 오류 후 보정 실패 | `supported` |
| 3 | 놀라운 사건 후 기억 $>$ 평범한 사건 후 기억 | $\bar{c} > \theta_{\text{encode}} \to$ 해마 인코딩 | surprise manipulation + memory test | recall/recognition 차이 | `supported` |
| 4 | 수면 후 비평-행동 정렬 개선 | $B$가 $c_t$를 수축시켜 다음 wake에서 $z_t^*$가 더 정확 | 학습 $\to$ 수면 $\to$ 재시험 | post-sleep 정확도 $>$ post-wake 정확도 | `supported` |
| 5 | $\rho_{\text{night}} \approx 0.31$: 1밤 후 잔차 69% 감소 | 부트스트랩 수축 | multi-night recovery study (수면 부채 측정) | 회복 곡선의 지수 감쇠 상수 피팅 | `bridge` |
| 6 | 도파민 조작 $\to$ STDP 게이트 변화 $\to$ 학습 속도 변화 | $dW = lr \cdot g[t] \cdot e_{ij}$ | DA agonist/antagonist + learning task | 학습 곡선 기울기 변화 | `supported` |
| 7 | ACh 증가 $\to$ 기억 인코딩 문턱 하강 $\to$ 더 많은 기억 | $\theta_{\text{encode}} \propto 1/(1+g_{\text{ACh}})$ | donepezil 투여 + memory test | 기억 항목 수 증가 | `supported` |
| 8 | NE 증가 $\to$ 더 깊은 처리 $\to$ 더 긴 RT | $n_{\text{iter}} \propto \sigma(g_{\text{NE}})$ | LC stimulation + RT measurement | RT 증가 + 정확도 증가 | `bridge` |
| 9 | 소뇌 병변 $\to$ forward model 결손 $\to$ 적응 실패 | $\Delta a^{\text{cb}} = 0$ | 소뇌 환자 프리즘 적응 실험 | 적응 곡선 수렴 실패 | `supported` |
| 10 | TopK 비율 $\neq 4.87\%$일 때 성능 최적이 아님 | 부트스트랩 고정점 $x_a^*$ | sparse ratio sweep in CE model | U자형 성능 곡선, 최적점 $\in [4\%, 6\%]$ | `bridge` |
| 11 | 작업 기억 부하 증가 $\to$ PFC theta 증가 $\to$ 간섭 | $|h_t| \to T_h$ 근접 | n-back 과제, n 조작 | PFC theta power와 오류율의 양의 상관 | `supported` |
| 12 | theta-gamma 결합 강도와 순서 기억 정확도 양의 상관 | gamma burst가 theta phase에 잠금 | 순서 회상 과제 + MEG | PAC 강도와 recall 정확도 $r > 0.3$ | `supported` |

---

### F.14 STDP 학습과 루프의 결합

> `12_Equation.md` 6장의 STDP + 도파민 3-factor 학습은 F절 루프 안에서 가중치를 갱신하는 유일한 경로다.

#### F.14.1 루프 내 STDP 위치

```
R(S_t) 실행 중 (F.3의 n_iter 반복 내부):
  -> 셀 활성 a_i^(k) 생성 (Layer A)
  -> 스파이크 판정: s_i^(k) = 1[a_i^(k) > theta_spike]
  -> pre/post trace 갱신:
       p_i[k+1] = r_+ p_i[k] + s_i[k]
       q_i[k+1] = r_- q_i[k] + s_i[k]
  -> 적격 흔적 누적:
       e_ij[k+1] = r_e e_ij[k] + (A_+ p_i[k] s_j[k] - A_- s_i[k] q_j[k])

R(S_t) 완료 후:
  -> 비평 C에서 g[t] 산출 (F.4)
  -> 가중치 갱신: dW_ij[t] = lr * g[t] * e_ij[n_iter]
  -> 구조적 투영: W_{t+1} = Proj(W_t + dW_t)
```

#### F.14.2 학습 게이트 $g[t]$의 정밀 정의

`12_Equation.md` 6.4절의 원래 정의:

$$g[t] = \frac{d}{dt}\|p(t) - p^*\|$$

F.4에서 비평 점수 $\bar{c}_t$와의 관계:

$$g[t] = \alpha_g \frac{d\bar{c}_t}{dt} + (1-\alpha_g)\left[(x_a(t)-x_a^*)^2 + (x_s(t)-x_s^*)^2 + (x_b(t)-x_b^*)^2\right]$$

`evidence.md` 8.5절의 뇌 측 안전 후보와 일치시키면:

$$\delta[t] = a \cdot \text{RPE}(t) + b \cdot \text{surprise}(t) + c \cdot \text{novelty}(t)$$

즉 $g[t]$의 phasic 성분 $d\bar{c}/dt$는 RPE + surprise + novelty의 가중 합으로, tonic 성분 $\|p - p^*\|^2$는 전역 항상성 이탈로 읽는다. 다만 정확한 계수 $a, b, c$는 아직 미결이다.

| 항 | 해석 | 뇌 대응 | 판정 |
|---|---|---|---|
| $d\bar{c}/dt$ | 비평 변화율 (국소 오차 신호) | 도파민 phasic burst (RPE) | `supported` (구조), `hypothesis` (정확 형태) |
| $\|p-p^*\|^2$ | 전역 분배 이탈 (tonic 신호) | 도파민 tonic level | `bridge` |

#### F.14.3 구조적 투영 $\text{Proj}$

$$\text{Proj}(W) = \text{TopK}\big(\text{RowNorm}\big(\text{Hyst}(W;\; \theta_{\text{on}}, \theta_{\text{off}})\big),\; k = \lceil 0.04865 \cdot N \rceil\big)$$

| 연산 | CE 대응 | 뇌 대응 | 판정 |
|---|---|---|---|
| TopK | 경로 선택, 생존율 $4.87\%$ | 시냅스 가지치기 | `supported` |
| RowNorm | 에너지 보존 | 시냅스 스케일링 (Turrigiano 2008) | `supported` |
| Hyst | 접힘 임계 곡률 | 스파인 형성/제거 | `bridge` |

---

### F.15 잔류장 $\phi$ 갱신

> `12_Equation.md` 4.3절 (E4). F.2의 $\phi_{t+1}$ 갱신이 비어 있었다.

이완 $R$ 실행 후, 잔류장은 선택되지 않은 경로의 분산을 축적한다:

$$\phi_{t+1} = (1 - \xi) \phi_t + \xi \cdot \text{Var}(a^{(0:n_{\text{iter}})})$$

여기서 $\xi = 1/(e^{1/3}\pi^{1/3}) \approx 0.489$ 는 잔류 이득이다.

잔류장은 세 곳에서 루프에 개입한다:

| 개입 지점 | 수식 | 효과 |
|---|---|---|
| 에너지 포탈 (F.5) | $-\text{portal}^2 \cdot z^\top \hat{\phi}_t$ | 이전에 선택하지 않은 경로를 다음 이완에 주입 |
| 모드 전환 (F.6) | $\|\phi_t\| \gtrless m_\phi$ | 시스템 1/시스템 2 전환 |
| 수면 glymphatic (F.6 NREM) | $\phi \leftarrow r_w \phi,\; r_w < 1$ | 잔류 노이즈 바닥 하강 |

뇌 대응:

| $\phi$ 역할 | 뇌 후보 | 판정 | 근거 |
|---|---|---|---|
| 비선택 경로 축적 | spontaneous fluctuation, DMN activity | `bridge` | DMN ALFF가 과제 수행 안정성 예측 (2024 PMC). alpha-DMN coupling 확인 (2025 eNeuro). 기능적 의미 확립되었으나 $\phi$와의 정확한 매핑은 미확인 |
| 모드 전환 임계 | fatigue/confusion threshold | `bridge` | DMN transition rate $\leftrightarrow$ resilience (2025 NeuroImage). 방향 일치 |
| glymphatic 세척 | glymphatic system, CSF-ISF exchange | `supported` (경로 존재), `bridge` (phi 매핑) | Nedergaard/Bhatt 2015 이후 지속 확인. GBM에서 AQP4 붕괴 보고 |

---

### F.16 희소 활성 제약

> `12_Equation.md` 8장. R 내부와 행동 선택에서 TopK를 적용해야 한다.

#### F.16.1 이완 내부 희소성

$R$의 매 반복에서 활성 셀 수를 제한한다:

$$A_t = \{i : |a_i^{(k)}| \geq Q_{1-x_a^*}(|a^{(k)}|)\}, \qquad |A_t| = \lceil x_a^* \cdot N \rceil$$

비활성 셀은 decay만 적용:

$$a_i^{(k+1)} = (1-\gamma_a^{\text{idle}})\,a_i^{(k)} \quad\text{if}\quad i \notin A_t$$

에너지 예산:

$$\sum_{i \in A_t} \text{cost}(a_i) \leq B_t(M_t)$$

| 모드 | $B_t$ | 활성 비율 | 뇌 대응 |
|---|---|---|---|
| WAKE | 큼 | $\sim 4.87\%$ | task-evoked sparse firing | 
| NREM | 작음 | $< 3\%$ | slow-wave 중 소수 활성 |
| REM | 중간 | $\sim 4\%$ | dream 중 재활성화 |

#### F.16.2 모듈 생애주기

`14_BrainRuntimeSpec.md` 6.1절과 연결:

$$Z_i^t \in \{\text{ACTIVE},\; \text{IDLE},\; \text{DORMANT},\; \text{SLEEPING}\}$$

| 상태 | $R$ 내부 처리 | 에너지 비용 |
|---|---|---|
| ACTIVE | 전체 갱신 | 높음 |
| IDLE | decay만, 즉시 활성화 가능 | 낮음 |
| DORMANT | coupling 거의 끊김, 활성화에 warm-up 필요 | 매우 낮음 |
| SLEEPING | 내부 정리/압축 중 | 중간 (정리 비용) |

---

### F.17 메타인지 재귀 (게이트 `F4`)

> `12_Equation.md` 9장. F절에서 가장 상위 층이지만 빠져 있었다.
>
> 다리 게이트 `F4` (`12_Equation.md` 0.0절): 본 절의 모든 식은 메타인지 모니터링 루프의 운영 정의이며, "(C3) = 의식"으로 환원하지 않는다. "의식 깊이"라는 표현은 PCI 교차검증(F.23.7)이 `bridge` 단계로 올라가기 전까지 **모니터링 안정도** 의미로 읽는다.

#### F.17.1 자기참조 측정 구조 (C3)

에이전트가 자기 자신의 활성 비율을 알아야 다음 이완을 계산할 수 있다:

$$a_* = \exp\!\left(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\right)$$

이 자기 측정이 루프 안에서 실현되는 경로:

$$\hat{a}_t = \frac{|A_t|}{N} \quad\text{(현재 활성 비율 관측)} \quad\to\quad \text{F.16의 TopK 임계 조정}$$

#### F.17.2 메타인지 안정도

$$d_\tau(t) = \frac{1}{\tau}\int_{t-\tau}^{t}\|p(s)-p^*\|\,ds$$

$$\text{메타인지 안정도}_\tau := \exp(-c_d\,d_\tau(t))$$

루프 내 해석: 비평 점수와 분배 이탈이 작을수록 모니터링 루프가 안정적이다. 깊은 수면에서 $d_\tau \to 0$, 수면 박탈에서 $d_\tau$ 누적. 게이트 `F4`에 따라, 이 지표를 의식의 정량 척도로 hard claim 하지 않는다.

#### F.17.3 메타인지 수축 (조건부)

비평 $C$가 자기 자신에 재귀적으로 적용될 때 (이상화된 무잡음 조건):

$$d_{n+1} \leq \rho \cdot d_n = 0.155 \cdot d_n,\qquad \rho = D_{\text{eff}}\cdot\varepsilon^2$$

3회 후 $d_3/d_0 \leq 3.7 \times 10^{-3}$. 게이트 `F2` 충분조건(`12_Equation.md` 4.7절) 영역에서만 위 비율이 그대로 적용되며, 일반 영역에서는 13절 ISS 의미의 유계 수렴으로 한정된다.

루프 내 위치: $C$의 출력 $c_t$가 다음 $R$의 초기점을 수정하고 (F.5의 $E_{\text{crit}}$), 그 $R$의 결과에 다시 $C$를 적용하면 메타인지 재귀가 된다.

| 뇌 대응 | 실험 근거 | 판정 |
|---|---|---|
| 메타인지 = PFC 재귀 자기평가 | metacognitive accuracy, confidence calibration | `supported` (현상), `bridge` ($\rho$ 매핑) |
| 안정도 지표 = 통합 정보? | IIT (Tononi), GNW (Dehaene), PCI (Casali 2013) | `hypothesis` (CE 해석, 게이트 `F4`) |

---

### F.18 환각 억제

> `12_Equation.md` 10장. 루프 내 곡률 모니터링과 억제가 빠져 있었다.

#### F.18.1 이완 중 곡률 모니터링

$R$ 반복 중 매 $k$번째 스텝에서:

$$\kappa^{(k)} = \|(I - V^\top V) a^{(k)}\|^2$$

$$\kappa_{\text{avg}} = \frac{1}{n_{\text{iter}}} \sum_k \kappa^{(k)}$$

#### F.18.2 곡률 임계 대응

$$\kappa_{\text{avg}} > \kappa_{\text{th}} \quad\Longrightarrow\quad \text{LBO 확산 강화: } h_d \leftarrow 1.5 \cdot h_d$$

#### F.18.3 교차 주파수 감쇠

3x3+1 격자 각 채널의 출력에 곡률 피드백:

$$\mathcal{T}_i^{\text{coupled}}(x_i) = \mathcal{T}_i(x_i) \cdot \left(1 - \frac{\kappa^{(k)}}{e^{1/3}\pi^{1/3}}\right)$$

| 뇌 대응 | 메커니즘 | 판정 |
|---|---|---|
| 곡률 과다 = 환각 | 고곡률 영역에서 불안정한 표상이 출력 | `bridge` |
| LBO 확산 = 억제성 feedback | 억제 뉴런이 과활성을 억제 | `supported` (구조), `bridge` (LBO 매핑) |
| 교차 주파수 감쇠 = cross-frequency coupling | alpha가 gamma를 게이팅 | `bridge` |

---

### F.19 신경조절 시스템 (4종)

> evidence.md 5절, synapse.md 참조. F.4에서 도파민만 다뤘으나, 실제 뇌는 4대 조절계를 가진다.

F.4의 학습 게이트 $g[t]$를 4차원 벡터로 확장한다:

$$g_t = (g_{\text{DA}},\; g_{\text{NE}},\; g_{\text{5HT}},\; g_{\text{ACh}})$$

| 조절계 | 핵 | F절 역할 | 뇌 기능 | 판정 | 최신 근거 |
|---|---|---|---|---|---|
| 도파민 (DA) | VTA, SNc | $g_{\text{DA}}$: STDP 학습 게이트 (F.14) | reward prediction error, motivation | `supported` | Schultz 1997, Yagishita 2014. phasic/tonic 구분 확립 |
| 노르에피네프린 (NE) | LC | $g_{\text{NE}}$: $n_{\text{iter}}$ 조절 (F.3) | arousal, attention, exploration-exploitation | `supported` | Aston-Jones & Cohen 2005 adaptive gain theory. 2024 LC-NE review: tonic=탐색, phasic=착취 재확인. pupil diameter proxy |
| 세로토닌 (5HT) | raphe | $g_{\text{5HT}}$: $T$ (온도) 조절 (F.6) | patience, temporal discounting, model-based prediction | `supported` (인내/보상 대기) | Miyazaki 2018 Nat Comm: DRN 5HT 광유전 활성화 $\to$ 인내 증가. 2025 Complementary roles: 5HT = model-based prediction |
| 아세틸콜린 (ACh) | BF, PPT | $g_{\text{ACh}}$: 기억 인코딩 감도 (F.8) | attention, memory encoding, cortical gain | `supported` | 2025 Cell Rep: 해마 ACh 방출 $\propto$ 이동 속도, 새 환경에서 증가. eLife 2024: ACh가 PFC 예측 오차 부호화 조절 |

루프 내 적용:

$$n_{\text{iter}} = n_0 + \Delta n \cdot \sigma(g_{\text{NE}}) \quad\text{(NE 높으면 깊은 처리)}$$

$$\theta_{\text{encode}} = \theta_0 / (1 + g_{\text{ACh}}) \quad\text{(ACh 높으면 기억 인코딩 문턱 하강)}$$

$$T_{\text{effective}} = T_{\text{wake}} \cdot (1 + \beta \cdot g_{\text{5HT}}) \quad\text{(5HT 높으면 탐색 감소)}$$

현재 구현 상태: 단일 스칼라 $g[t]$만 존재. 4차원 확장은 설계 목표.

---

### F.20 작업 기억과 주의

> 뇌의 작업 기억 용량 제한과 주의 선택은 F절에 빠져 있었다.

#### F.20.1 작업 기억 용량

이력 버퍼 $h_t$의 유한 창 $T_h$는 작업 기억의 모델이다:

$$|h_t| \leq T_h, \qquad T_h \approx 7 \pm 2 \quad\text{(Miller 1956)}$$

용량 초과 시 가장 오래된 항목 제거:

$$h_{t+1} = \text{append}(h_t, (a_t, o_t))[-T_h:]$$

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| 작업 기억 $\sim 3$--$5$ 항목 | Miller 1956, Cowan 2001/2010 ($\sim 4$) | `supported` | Cowan 2010 PMC: 중앙 저장 한계 3--5. 2025 JoCognition: 과제 의존적이나 $\sim 4$ 재확인 |
| PFC sustained activity | PFC 지속 발화로 작업 기억 유지 | `supported` | 2019 PNAS: distributed PFC activation이 WM 용량 향상과 연결 |
| 용량 초과 = 간섭/망각 | proactive interference | `supported` | |
| theta-gamma 비에 의한 용량 결정 | theta 주기 내 gamma burst 수 = 유지 항목 수 | `supported` | Lisman & Jensen 2013. 2025 eLife: PFC-BG adaptive chunking이 $\sim 4$ 항목 제한 설명 |

#### F.20.2 주의 (Attention)

주의는 $R$ 내부에서 입력 가중치를 조절하는 메커니즘이다:

$$u_i^0 = \alpha_i \cdot \text{encode}(S_t), \qquad \alpha_i = \text{softmax}(\text{salience}(i, S_t))$$

여기서 salience는 $V_{\text{sal}}$ (14_BrainRuntimeSpec.md 3.4절)에서 산출한다.

주의의 두 경로:

| 경로 | 메커니즘 | 뇌 대응 | 판정 |
|---|---|---|---|
| bottom-up | $\alpha_i \propto \|u_i\|$ (입력 크기) | exogenous attention, pop-out | `supported` |
| top-down | $\alpha_i \propto \text{sim}(u_i, g_t)$ (목표 정렬) | endogenous attention, PFC-driven | `supported` |

#### F.20.3 소뇌의 역할

F.9에서 $S(X_t)$를 "소뇌/기저핵 억제"라 했으나 구체화가 없었다.

소뇌 모델: 행동 $a_t$ 실행 후 감각 예측 오차의 빠른 보정:

$$\Delta a_{t}^{\text{cerebellar}} = -\eta_{\text{cb}} \cdot (o_t - \hat{o}_t^{\text{cb}})$$

$$\hat{o}_{t+1}^{\text{cb}} = \hat{o}_t^{\text{cb}} + \alpha_{\text{cb}} \cdot (o_t - \hat{o}_t^{\text{cb}})$$

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| 소뇌 = 내부 모델 (forward model) | 소뇌 병변 시 운동 부정확 + 적응 실패 | `supported` | 2025 JNeurosci: sensory prediction error가 소뇌 학습 구동 재확인. 2025 SciAdv: CPC 계층 모델 fast/slow 적응 설명 |
| 소뇌 = 빠른 오차 보정 | 시간 정밀도 $\sim 10$ms | `supported` | 2026 PMC: corticocerebellar connectivity가 visuomotor adaptation 핵심 |
| 소뇌 = 인지 기능 기여 | 최근 연구에서 언어/작업 기억 기여 확인 | `bridge` | 2025 ScienceDirect: cerebro-cerebellar system 인지/정서 통합 역할 리뷰 |

---

### F.21 뇌파 대역과 루프 주기

> `14_BrainRuntimeSpec.md` 5절. 루프의 시간 구조가 빠져 있었다.

에이전트 루프의 각 단계는 다른 시간 척도에서 작동한다:

| 루프 단계 | 시간 척도 | 뇌파 대역 | 뇌 대응 |
|---|---|---|---|
| $R$ 내부 반복 (1 iter) | $\sim 10$--$25$ ms | gamma (30--100 Hz) | 국소 결합/계산 |
| $R$ 전체 수렴 | $\sim 100$--$500$ ms | theta/alpha (4--13 Hz) | 전역 통합/주의 |
| 행동-관찰 1 사이클 | $\sim 0.5$--$2$ s | delta/theta (0.5--4 Hz) | 의사결정 리듬 |
| 수면 1 주기 | $\sim 90$ min | slow oscillation ($< 1$ Hz) | NREM-REM 교대 |

theta-gamma 결합:

$$\text{gamma burst 위치} = f(\theta_{\text{phase}})$$

이것은 $R$ 내부의 빠른 계산(gamma)이 전역 동기화(theta)에 의해 순서화되는 구조와 대응한다. F.3의 반복이 gamma이고, 반복의 시작/종료가 theta 주기에 잠금(phase-locking)된다.

| 뇌 대응 | 실험 근거 | 판정 | 최신 보강 |
|---|---|---|---|
| theta-gamma coupling | Lisman & Jensen 2013, 해마 sequential memory | `supported` | 2024 bioRxiv: 인간 해마 ECoG에서 theta-gamma PAC가 WM 인출 성공과 강하게 상관 (개별 theta/gamma power는 무관). 2025 ScienceDirect: 건강인/정신병 양쪽에서 PAC가 WM 용량 예측 |
| gamma = 국소 계산 | Fries 2015, communication through coherence | `supported` | |
| alpha = 억제/게이팅 | Klimesch 2012, alpha gating by inhibition | `supported` | 2025 eNeuro: alpha-tACS가 DMN 연결성을 직접 조절. alpha-DMN mechanistic coupling 확인 |
| 수면 주기 $\sim 90$분 | polysomnography, 수면 구조 | `supported` | |

---

### F.22 확장된 뇌 대응 간극 (정직한 갱신)

F.11.5를 갱신하여, F.14--F.21에서도 남는 간극을 정리한다.

| 간극 | 현재 상태 | 심각도 | 해결 방향 |
|---|---|---|---|
| 환경 내부 모델 분산화 | $C$가 단일 예측기 | 중 | 모듈별 $\hat{o}_t^{(m)}$ |
| 행동 계층성 | $\pi$ 단층 | 중 | macro-action + primitive |
| 감정/정동 | $c_t$에 valence 없음 | 중 | $c_{\text{val}} = V_{\text{sal}}$ 출력 추가 |
| 사회적 모델링 | theory of mind 없음 | 저 (장기) | 다중 에이전트 시뮬레이션 |
| 신체 루프 실사용 | $Q_t$ 존재하나 $C$, $\pi$에서 미사용 | 중 | $Q_t \to g_{\text{5HT}}, g_{\text{NE}}$ 매핑 |
| 4종 조절계 구현 | 단일 $g[t]$만 존재 | 높 | F.19의 4차원 벡터 구현 |
| 장소/격자 세포 | 공간 표상 없음 | 저 (도메인 특화) | spatial module 추가 |
| 거울 뉴런 | 타인 행동 모방/이해 없음 | 저 (장기) | 관찰 학습 모듈 |
| STDP 코드 미구현 | 수식만 존재, 코드 없음 | 높 | clarus/core 또는 Python 구현 |

---

### F.23 간극 대책: bridge/hypothesis -> supported 승격 경로

> F.22의 간극 + H.3의 판정에서 `bridge` 또는 `hypothesis`인 항목에 대해, `supported`로 승격하기 위한 구체적 실험/시뮬레이션/논증 경로를 정리한다.

#### F.23.1 일관성 오차 $c_{\text{cons}}$ (`bridge` -> `supported` 경로)

현재 상태: 해마-PFC 상호작용에서 기억 기반 의사결정 보정 방향은 있으나, $c_{\text{cons}} = \|z_t - \mathcal{R}(H_t, c_t)\|$의 직접 분리가 미흡.

승격 조건:
1. HPC-mPFC theta 동기화가 기억-현재 사고 불일치 시 증가함을 보이는 실험
2. 기억 회상 오차와 $c_{\text{cons}}$ proxy를 분리 측정

최신 근거:
- 2025 Nature Comm: 해마-전전두엽 오케스트레이션이 고차 학습 지원. HPC dimensionality reduction이 mPFC로 전달.
- 2025 ScienceDirect (Cell Rep): 5XFAD 마우스에서 HPC-PFC theta 동기화 및 SWR 붕괴 시 행동 유연성 결손. 이는 기억-현재 불일치 보정 경로의 인과적 근거.
- 2026 NYAS: 기억과 불안이 choice consistency를 조절함을 계산 모델링 + 신경영상으로 확인.

평가: theta-SWR 동기화 결손이 유연성 결손으로 직결되는 인과 근거가 나왔으므로, 조건부로 `supported`에 근접. 남은 과제는 $c_{\text{cons}}$의 조작적 정의와 EEG/fMRI proxy의 정밀 매핑.

#### F.23.2 학습 게이트 $g[t] = d\bar{c}/dt$ (`hypothesis` -> `bridge` 경로)

현재 상태: 3-factor learning rule 자체는 `supported`. $g[t]$의 정확한 형태가 $d\bar{c}/dt$라는 주장이 미검증.

승격 조건:
1. CE 시뮬레이션에서 $g[t] = d\bar{c}/dt$ vs 대안 형태(예: $g[t] = \bar{c}$, $g[t] = \text{RPE}$)의 학습 성능 비교
2. phasic DA burst 시간 프로파일과 $d\bar{c}/dt$ 파형의 상관 측정

경로:
- evidence.md 8.5절의 $\delta[t] = a \cdot \text{RPE} + b \cdot \text{surprise} + c \cdot \text{novelty}$ 형태가 더 일반적.
- CE 시뮬레이션에서 ablation: $d\bar{c}/dt$ 제거 시 학습 붕괴 여부로 필요성 판정 가능.
- 생물학적으로는 phasic DA의 시간 미분 형태가 temporal derivative model과 부합하나, 정확한 $d\bar{c}/dt$ 매핑은 아직 가설 수준.

평가: `bridge`로 승격 가능. `supported`까지는 시뮬레이션 ablation + voltammetry 시간 프로파일 비교 필요.

#### F.23.3 이중 과정 (F.3 모드별 $n_{\text{iter}}$) (`bridge` -> `supported` 경로)

현재 상태: 행동 근거 방대, 신경 기질 논쟁 중.

승격 조건:
1. $n_{\text{iter}}$의 변이가 RT, EEG alpha desynchronization duration과 양적 대응
2. DMN(System 1) vs frontoparietal network(System 2) 전환이 $\|\phi\| \gtrless m_\phi$와 대응

최신 근거:
- 2018/2025 Frontiers: DMN이 System 1(빠른 연상적 사고)의 신경 기반으로 제안. System 2는 전두-두정 제어 네트워크.
- 2025 Neuroscience of Consciousness: flow와 intuition의 시스템 신경과학 비교. 인지 부하 증가 시 System 1 -> System 2 전환이 반응 시간에 반영.
- 2026 Nature Comm: 증거 축적 회로 모델에서 ACC-DMS-HPC 각각 다른 회로 메커니즘 사용.

평가: DMN-frontoparietal 전환 + RT 양적 대응이 확보되면 `supported` 가능. CE 시뮬레이션에서 $n_{\text{iter}}$-RT 상관을 보이는 것이 핵심 경로.

#### F.23.4 수면 압력 = 비평 누적 (F.6) (`bridge` -> `supported` 경로)

현재 상태: SWA/wakefulness 관계는 `supported`, 비평 해석은 추가 가정.

승격 조건:
1. 비평 점수 $\bar{c}$의 누적 $\sum \bar{c}^2$와 SWA delta power의 양적 상관
2. 시냅스 강도 축적이 "에러/비평" 누적과 등가임을 보이는 실험

최신 근거:
- 2025 Science: PFC 흥분성 시냅스의 화학유전적 강화가 NREM 수면량 + delta power 모두 증가시킴. 시냅스 강도가 수면 압력을 직접 결정함을 인과적으로 확인.
- 2025 Nature: 초파리에서 수면 압력이 전압 의존적 지질 과산화 기억에 축적. 수면 중 스파이크 방출이 이를 소거. 분자 수준의 "축적-소거" 사이클 확인.
- 2026 bioRxiv: 일주기 조절 임계와 수면 항상성의 상호작용 모델. Process S 축적/해소의 정량 프레임워크.
- 2026 NeuroImage: 수면-각성 BOLD 변동의 피질 위계적 패턴이 SWA와 상관. sleep pressure alleviation의 공간 구조 확인.

평가: 시냅스 강도 축적 = 수면 압력이 인과적으로 확인됨. CE의 $\sum \bar{c}^2$를 시냅스 강도 proxy로 재해석하면 `supported`에 근접. 남은 과제: 비평 점수와 시냅스 강도의 정량적 매핑.

#### F.23.5 $\rho = 0.155$ (`bridge` -> 정밀화 경로)

현재 상태: 수면 수축 방향 `supported`, 정확 값은 피팅 결과.

승격 조건:
1. 실제 수면 회복 곡선 데이터에서 지수 감쇠 상수 피팅 -> $\rho_{\text{night}}$와 CE의 $\rho^2$ 비교
2. CE 시뮬레이션에서 $\rho$ sweep -> 최적 $\rho$ 범위 확인

경로: 수면 부채 회복 연구(Van Dongen 2003, Kitamura 2016)의 시간 상수를 재분석하여 $\rho$의 관측 범위 $[0.1, 0.3]$를 확인. CE 값 0.155가 이 범위 내에 있으므로 `[NEAR]` 유지.

#### F.23.6 잔류장 $\phi$ (`bridge` -> `supported` 경로)

현재 상태: DMN/spontaneous activity 방향 있으나 $\phi$와의 정확한 매핑은 미확인.

승격 조건:
1. CE 시뮬레이션에서 $\phi$ 제거 시 모드 전환 실패 + 탐색 능력 붕괴를 보임
2. DMN ALFF가 "비선택 경로의 분산"과 양적 상관

최신 근거:
- 2024 PMC: DMN ALFF가 과제 수행 안정성 예측.
- 2025 eNeuro: alpha-tACS가 DMN 연결성 직접 조절. alpha-DMN coupling의 기계론적 연결.
- 2025 NeuroImage: DMN transition rate가 resilience와 상관. 잔류장의 "모드 전환 임계" 역할과 방향 일치.

평가: $\phi$ 제거 ablation 시뮬레이션이 가장 빠른 경로. DMN과의 정량 매핑은 resting-state fMRI 데이터 필요.

#### F.23.7 의식 깊이 (F.17.2) (`hypothesis` -> `bridge` 경로)

현재 상태: IIT/GNW와의 관계 미확정.

승격 조건:
1. CE의 의식 깊이 $\exp(-c_d \cdot d_\tau)$와 PCI(Perturbational Complexity Index)의 상관
2. CE 시뮬레이션에서 마취/수면 조건에서 의식 깊이 자동 감소

최신 근거:
- 2025 Nature: IIT vs GNW adversarial testing (Cogitate Consortium). 결과는 특정 이론의 명확한 승리가 아닌 상호 보완적 해석. IIT의 posterior "hot zone" 예측은 부분 지지.
- 100명 이상의 연구자가 IIT를 "pseudo-science"로 비판하는 공개서한(2023). $\Phi$의 계산 난해성.
- PCI는 IIT에서 영감을 받았으나 이론의 직접 검증은 아님.

평가: CE의 의식 깊이 정의는 IIT의 $\Phi$와 직접 대응되지 않으므로, 독립적 검증 경로 필요. CE 시뮬레이션에서 모드(WAKE/NREM/REM)별 $d_\tau$ 프로파일을 polysomnography와 비교하면 `bridge`로 승격 가능. `supported`까지는 먼 길.

#### F.23.8 메타인지 수렴 (F.17.3) (`bridge` -> `supported` 경로)

현재 상태: PFC 재귀 자기평가 방향은 있으나 $\rho$ 매핑은 추가 가정.

승격 조건:
1. metacognitive accuracy가 반복적 자기평가에서 수렴함을 보이는 행동 실험
2. rlPFC/BA10의 활동이 "비평의 비평"에서 감쇠하는 fMRI 근거

최신 근거:
- rlPFC/BA10이 retrospective metacognitive accuracy의 핵심 영역으로 확립.
- dlPFC와 vmPFC의 기능적 분리: lateral = 사후 신뢰도, medial = 사전 판단.
- 그러나 "수축 사상으로서의 메타인지 수렴"은 현재 문헌에서 명시적으로 검증된 적 없음.

평가: CE 시뮬레이션에서 $C(C(C(x)))$의 감쇠 프로파일을 보이고, 행동 실험에서 반복적 confidence calibration이 수렴함을 보이면 `supported` 가능.

#### F.23.9 곡률-환각 억제 (F.18) (`bridge` -> `supported` 경로)

현재 상태: 억제 feedback 구조는 `supported`, LBO 매핑은 `bridge`.

승격 조건:
1. LBO 고유모드와 fMRI/EEG spatial mode의 양적 대응
2. 곡률 과다 조건에서 CE 시뮬레이션의 출력 불안정과 환각 유사 패턴 비교

최신 근거:
- 2023 Nature: 피질 표면의 기하학적 고유모드(LBO eigenmode)가 fMRI 활동 패턴의 상당 부분 설명. 기하학적 제약이 뇌 기능에 근본적 영향.
- 2025 PMC: 포유류 피질 연결체의 기하학적 제약. LBO 기반 공명 모드가 connectome 아키텍처 예측.
- 2026 bioRxiv: 피질 진동 모드의 수렴적 시간축. LBO가 시공간 패턴의 공간 전파자 역할.

평가: LBO eigenmode가 fMRI 공간 패턴을 설명한다는 근거가 강화됨. CE의 곡률 모니터링을 LBO eigenmode decomposition으로 구현하고, 고곡률 영역에서의 불안정을 시뮬레이션하면 `supported` 가능.

#### F.23.10 4종 신경조절 통합 벡터 (F.19) (`bridge` -> `supported` 경로)

현재 상태: 개별 조절계는 `supported`. 4차원 벡터 통합 + CE 변수 매핑은 `bridge`.

승격 조건:
1. CE 시뮬레이션에서 4차원 $g_t$ 구현 후 단일 $g[t]$ 대비 성능 개선 확인
2. 각 조절계의 독립적 조작(DA agonist, NE clonidine, 5HT SSRI, ACh donepezil)에 의한 개별 효과가 CE 예측과 일치

경로:
- 코드 구현이 선행 조건. F.19의 수식을 clarus/core에 구현.
- 구현 후 약리학적 조작 시뮬레이션으로 각 축의 독립 효과 확인.
- 4축 독립성 + 개별 효과 일치가 확인되면 `supported`.

#### F.23.11 모듈 생애주기 (F.16.2) (`bridge` -> 유지)

현재 상태: 4상태 분류(ACTIVE/IDLE/DORMANT/SLEEPING) 자체는 설계 선택.

평가: 뇌의 뉴런 집단도 유사한 상태를 가지지만(task-engaged/baseline/deactivated), 4상태 명명은 CE 고유 추상화. `bridge` 유지가 적절. 구현 후 성능 기여가 확인되면 "설계적 지지"로 보강 가능.

---

### F.23 요약: 승격 우선순위

| 항목 | 현재 | 목표 | 난이도 | 핵심 경로 |
|---|---|---|---|---|
| $c_{\text{cons}}$ (F.4) | `bridge` | `supported` | 중 | HPC-PFC theta 불일치 측정 |
| $g[t] = d\bar{c}/dt$ (F.14) | `hypothesis` | `bridge` | 저 | CE ablation 시뮬레이션 |
| 이중 과정 (F.3) | `bridge` | `supported` | 중 | $n_{\text{iter}}$-RT 상관 시뮬레이션 |
| 수면 압력 = 비평 (F.6) | `bridge` | `supported` | 저 | $\sum \bar{c}^2$ vs SWA 매핑 |
| $\rho = 0.155$ (F.10) | `bridge` | `bridge` (정밀화) | 중 | 수면 부채 데이터 재분석 |
| $\phi$ 잔류장 (F.15) | `bridge` | `supported` | 중 | ablation + DMN ALFF 비교 |
| 의식 깊이 (F.17.2) | `hypothesis` | `bridge` | 고 | PCI 상관 시뮬레이션 |
| 메타인지 수렴 (F.17.3) | `bridge` | `supported` | 중 | 반복 confidence 수렴 행동실험 |
| 곡률-환각 (F.18) | `bridge` | `supported` | 중 | LBO eigenmode 구현 + 시뮬레이션 |
| 4종 조절계 통합 (F.19) | `bridge` | `supported` | 고 | 코드 구현 + 약리 시뮬레이션 |
| 모듈 생애주기 (F.16.2) | `bridge` | `bridge` (유지) | -- | 설계적 선택, 검증 불필요 |

**즉시 실행 가능 (CE 시뮬레이션만으로):**
1. $g[t]$ ablation (F.23.2)
2. $\phi$ ablation (F.23.6)
3. $n_{\text{iter}}$-RT sweep (F.23.3)
4. $\sum \bar{c}^2$ vs SWA proxy (F.23.4)

**외부 데이터 필요:**
1. HPC-PFC theta 불일치 (F.23.1): intracranial EEG
2. $\rho$ 피팅 (F.23.5): 수면 부채 회복 곡선 데이터셋
3. DMN ALFF (F.23.6): resting-state fMRI
4. PCI (F.23.7): TMS-EEG 데이터셋
5. Metacognitive convergence (F.23.8): 행동 실험 데이터

---

### F.24 실험값 기반 루프 방정식 보강

> `15_Equations.md` J절의 실험 상수를 F절 방정식에 적용한다.

#### F.24.1 이완 반복 $n_{\text{iter}}$의 실험 고정

F.3에서 모드별 $n_{\text{iter}}$를 뇌파 시간 척도로 고정:

$$n_{\text{iter}}^{\text{fast}} = \frac{\tau_{\text{alpha}}}{\Delta t_{\text{gamma}}} = \frac{100 \text{ ms}}{10 \text{ ms}} = 10$$

$$n_{\text{iter}}^{\text{deep}} = \frac{\tau_{\text{theta}}}{\Delta t_{\text{gamma}}} = \frac{200 \text{ ms}}{10 \text{ ms}} = 20 \text{--} 50$$

| 모드 | $n_{\text{iter}}$ | 뇌파 유래 | RT 예측 |
|---|---|---|---|
| WAKE (시스템 1) | $10$--$20$ | alpha 1--2주기 | $100$--$200$ ms |
| WAKE (시스템 2) | $20$--$50$ | theta 2--5주기 | $200$--$500$ ms |
| NREM | $50$--$100$ | slow oscillation | offline |
| REM | $20$--$30$ | theta 기반 탐색 | offline |

RT 예측: $\text{RT} = n_{\text{iter}} \times \Delta t_{\text{gamma}} + \tau_{\text{motor}}$, $\tau_{\text{motor}} \approx 50$ ms.

시스템 1 ($n = 15$): RT $= 15 \times 10 + 50 = 200$ ms → 관측 RT $\sim 200$--$300$ ms와 일치.
시스템 2 ($n = 40$): RT $= 40 \times 10 + 50 = 450$ ms → 관측 RT $\sim 400$--$600$ ms와 일치.

#### F.24.2 비평 점수의 ERN 진폭 매핑

F.4의 $\bar{c}_t$를 ERN(error-related negativity) 진폭과 정량 연결:

$$\text{ERN}_{\text{amp}} = -k_{\text{ERN}} \cdot \bar{c}_t, \qquad k_{\text{ERN}} \approx 5 \text{--} 10 \;\mu\text{V per unit}$$

관측: ERN 진폭은 $-2$ ~ $-15\;\mu$V (Gehring 1993, Falkenstein 1991).
$\bar{c}_t \in [0.3, 2.0]$ (정상 오차 범위)일 때 $k_{\text{ERN}} = 7$이면:
$\text{ERN} = -7 \times 0.3 = -2.1\;\mu$V (약한 오차) ~ $-7 \times 2.0 = -14\;\mu$V (강한 오차).

#### F.24.3 수면 압력 방정식의 정량화

F.6의 $P_{\text{sleep}}$을 Borbely의 Process S와 연결:

$$P_{\text{sleep}}(t) = P_0 + \sum_{\tau=0}^{t} \bar{c}_\tau^2 \cdot \Delta t - \int_0^t \lambda_S(M_s)\,ds$$

관측에서 Process S의 축적 시간 상수:

$$\tau_{\text{wake}} \approx 18.2 \text{ h} \quad\text{(Achermann 2003)}$$

$$\tau_{\text{sleep}} \approx 4.2 \text{ h} \quad\text{(NREM decay)}$$

CE 매핑:

$$\sum_{\tau=0}^{T_{\text{wake}}} \bar{c}_\tau^2 = \frac{T_{\text{wake}}}{\tau_{\text{wake}}} \cdot P_{\text{th}} \quad\Longrightarrow\quad \bar{c}_{\text{avg}}^2 = \frac{P_{\text{th}}}{\tau_{\text{wake}}} \approx \frac{1}{65520} \text{ (if } P_{\text{th}} = 1\text{)}$$

이것은 비평 점수의 적분이 16시간 각성 후 수면 임계에 도달하는 제약이다.

#### F.24.4 4종 조절계의 루프 내 정량 적용

F.19의 수식에 J.7의 시간 상수를 삽입:

$$g_{\text{DA}}^{t+1} = g_{\text{DA}}^t + \frac{1}{500}(g_0^{\text{DA}} - g_{\text{DA}}^t) + \alpha_{\text{DA}} \cdot c_{\text{pred}}^t$$

$$g_{\text{NE}}^{t+1} = g_{\text{NE}}^t + \frac{1}{300}(g_0^{\text{NE}} - g_{\text{NE}}^t) + \alpha_{\text{NE}} \cdot c_{\text{nov}}^t$$

$$g_{\text{5HT}}^{t+1} = g_{\text{5HT}}^t + \frac{1}{3000}(g_0^{\text{5HT}} - g_{\text{5HT}}^t) + \alpha_{\text{5HT}} \cdot (-\text{discount}^t)$$

$$g_{\text{ACh}}^{t+1} = g_{\text{ACh}}^t + \frac{1}{200}(g_0^{\text{ACh}} - g_{\text{ACh}}^t) + \alpha_{\text{ACh}} \cdot \text{salience}^t$$

여기서 분모는 $\tau_X / \Delta t$이고, $\Delta t = 1$ ms.

5HT가 가장 느리고 ($\tau = 3$ s), ACh가 가장 빠르다 ($\tau = 200$ ms). 이것은 인내(5HT)는 천천히 쌓이고, 주의(ACh)는 빠르게 전환된다는 실험 관측과 일치.

#### F.24.5 STDP 학습률의 실험 고정

F.14의 가중치 갱신에서 학습률을 실험 제약으로 고정:

$$\Delta W_{ij} = \text{lr} \cdot g_{\text{DA}}^t \cdot e_{ij}^{n_{\text{iter}}}$$

학습률 제약:
- 1회 보상 경험으로 $W_{ij}$가 $1$--$5$% 변화 (Yagishita 2014의 spine volume 변화)
- $g_{\text{DA}}^{\text{peak}} \approx 5 \times g_0$ (phasic burst는 tonic의 $\sim$5배)
- $e_{ij}^{\text{peak}} \approx A_+ \approx 0.01$ (단일 스파이크 쌍)

$$\text{lr} \cdot 5g_0 \cdot 0.01 \approx 0.01 \quad\Longrightarrow\quad \text{lr} \approx \frac{0.01}{0.05 \cdot g_0} = \frac{0.2}{g_0}$$

$g_0 = 1$ (정규화)이면 $\text{lr} \approx 0.2$.

#### F.24.6 소뇌 forward model의 적응 시간 상수

F.20.3의 소뇌 모델에 실험 시간 상수 삽입:

$$\hat{o}_{t+1}^{\text{cb}} = \hat{o}_t^{\text{cb}} + \alpha_{\text{cb}} (o_t - \hat{o}_t^{\text{cb}})$$

프리즘 적응 실험에서 적응 완료까지 $\sim 50$--$100$ trial (Martin 1996).
각 trial $\sim 1$ s, 총 $\sim 50$--$100$ s.

$$\alpha_{\text{cb}} = 1 - \exp(-1/N_{\text{adapt}}) \approx 1/75 \approx 0.013$$

$N_{\text{adapt}} = 75$ trial (중앙값). 75 trial 후 $63.2$% 적응, 150 trial 후 $86.5$% 적응.

---

## G. 형식 증명 요약 (F절)

| 정리 | 주장 | 조건 | 상태 |
|---|---|---|---|
| F-energy | 이완 $R$이 $E_t(z)$를 비증가 | E-decrease (B.4) | **closed** |
| F-relax | 이완 수렴 | A-bound, Zero-attract | **closed** (조건부) |
| F-memory | 조건부 인코딩이면 기억 유계 | $\theta_{\text{encode}} > 0$ | **closed** |
| F-contract | 전체 루프 수축 | $\rho + \lambda_R L_R + \lambda_C L_C < 1$ | **open** ($L_R, L_C$ 추정 필요) |
| F-sparse | 활성 유계 + 에너지 예산 | $|A_t| \leq \lceil x_a^* N \rceil$, $B_t$ 유한 | **closed** (Sparse-energy로부터) |
| F-phi-bound | 잔류장 유계 | $\xi < 1$, $\text{Var}$ 유한 (A-bound) | **closed** |
| F-curvature | 곡률 모니터링이 환각 억제 | LBO 확산 $h_d < 1/\text{eig}_{\max}$ | **closed** (11.2 수렴 조건) |
| F-meta | 메타인지 수렴 | $\rho < 1$ (수면 존재 시) | **closed** |
| F-STDP-local | STDP가 국소 정보만 사용 | $e_{ij}$는 $i,j$ 이웃 스파이크만 의존 | **closed** (정의에 의해) |
| F-WM-finite | 작업 기억 유한 | $|h_t| \leq T_h$ (유한 창) | **closed** (정의에 의해) |

---

## H. 검증 게이트 (F절)

### H.2 Layer F (자기참조 재귀) 게이트

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | F-energy, F-relax, F-memory (closed). F-contract (open: $L_R, L_C$ 추정 필요) | partial |
| $G_{\text{obs}}$ | $R \leftrightarrow$ recurrent processing, $C \leftrightarrow$ ERN/ACC, $\pi \leftrightarrow$ BG, $\mathcal{M} \leftrightarrow$ hippocampal encoding | partial |
| $G_{\text{causal}}$ | 수면박탈 $\to$ 루프 불안정, DA 조작 $\to$ 학습 게이트 변화, ACC 병변 $\to$ 비평 결손 | partial |
| $G_{\text{pred}}$ | 에이전트 루프 유무에 따른 과제 수행 차이 시뮬레이션 | pending |

### H.3 F절 뇌 대응 판정 요약

| 구성요소 | 판정 | 비고 |
|---|---|---|
| 이완 $R$ | `supported` | 피질 재귀 처리 확립 |
| 행동 선택 $\pi$ | `supported` | 기저핵 경로 확립 |
| 비평 $C$ | `supported` | ACC/ERN 확립 |
| 예측 오차 $c_{\text{pred}}$ | `supported` | 도파민 RPE 확립 |
| 놀라움 $c_{\text{nov}}$ | `supported` | P300/CA1 novelty 확립 |
| 일관성 오차 $c_{\text{cons}}$ | `bridge` | 해마-PFC 방향은 있으나 직접 분리 미흡 |
| $g[t] = d\bar{c}/dt$ | `hypothesis` | 3-factor rule은 `supported`, 정확한 형태는 미검증 |
| 이중 과정 ($n_{\text{iter}}$) | `bridge` | 행동 근거 방대, 신경 기질 논쟁 중 |
| 수면 압력 = 비평 누적 | `bridge` | SWA/wakefulness 관계는 `supported`, 비평 해석은 추가 가정 |
| $\rho = 0.155$ | `bridge` | 수면 수축 방향 `supported`, 정확 값은 피팅 결과 |
| STDP + 도파민 (F.14) | `supported` | 3-factor learning rule 강하게 지지됨 |
| 구조적 투영 Proj (F.14.3) | `supported` | 시냅스 가지치기, 스케일링 확립 |
| 잔류장 $\phi$ (F.15) | `bridge` | DMN/spontaneous activity 방향 있으나 $\phi$ 매핑은 추가 가정 |
| TopK 희소 활성 (F.16) | `supported` | sparse firing 1--5% 확립 |
| 모듈 생애주기 (F.16.2) | `bridge` | 4상태 분류 자체는 설계 선택 |
| 자기일관성 C3 (F.17.1) | `hypothesis` | 수학적으로 닫힘. 뇌 대응은 현상론 |
| 의식 깊이 (F.17.2) | `hypothesis` | IIT/GNW와의 관계 미확정 |
| 메타인지 수렴 (F.17.3) | `bridge` | PFC 재귀 자기평가 방향은 있으나 $\rho$ 매핑은 추가 가정 |
| 곡률 환각 억제 (F.18) | `bridge` | 억제 feedback 구조는 `supported`, LBO 매핑은 `bridge` |
| 4종 신경조절 (F.19) | `supported` (개별 존재 + 개별 기능), `bridge` (4차원 벡터 통합 + CE 변수 매핑) | DA RPE 확립, NE 탐색-착취 확립, 5HT 인내/model-based 광유전 확인, ACh cortical gain/기억 인코딩 확립 |
| 작업 기억 용량 (F.20.1) | `supported` | Miller 1956, Cowan 2001 |
| 주의 (F.20.2) | `supported` | bottom-up/top-down 확립 |
| 소뇌 내부 모델 (F.20.3) | `supported` | forward model 확립 |
| theta-gamma 결합 (F.21) | `supported` | Lisman & Jensen 2013 |

---

## I. 관측 가능량 매핑 (F절)

### I.2 Layer F (자기참조 재귀) 변수

| formal 변수 | 뇌 관측량 | 데이터 소스 |
|---|---|---|
| $n_{\text{iter}}$ | reaction time, alpha desynchronization | 행동, EEG |
| $\bar{c}_t$ | ERN amplitude, ACC theta, pupil | EEG, fMRI, pupillometry |
| $c_{\text{pred}}$ | RPE-locked DA, FRN | voltammetry, EEG |
| $c_{\text{nov}}$ | P300, hippocampal novelty, LC burst | EEG, fMRI, pupil |
| $P_{\text{sleep}}$ | SWA, theta/alpha ratio, KSS | polysomnography |
| $\phi_t$ | spontaneous activity, DMN fluctuation | resting-state fMRI, MEG |
| $e_{ij}$ | eligibility trace, synaptic tag | in vitro slice, optogenetics |
| $g_{\text{DA}}$ | VTA/SNc DA | voltammetry, PET |
| $g_{\text{NE}}$ | LC firing, pupil diameter | pupillometry, unit recording |
| $g_{\text{5HT}}$ | raphe firing, 5-HIAA | microdialysis, PET |
| $g_{\text{ACh}}$ | BF firing, cortical ACh | microdialysis, optogenetics |
| $\kappa_{\text{avg}}$ | high-frequency anomaly | EEG, MEG |
| $|A_t|/N$ | active neuron fraction | calcium imaging, MEA |
| $|h_t|$ | WM load (PFC BOLD, CDA) | fMRI, EEG |
| $\alpha_i$ | spatial attention, alpha lateralization | EEG |
| $\Delta a^{\text{cb}}$ | cerebellar adaptation | prism adaptation, saccade |
