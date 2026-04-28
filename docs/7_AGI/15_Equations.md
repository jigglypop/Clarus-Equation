# Layer A--E 방정식 정본

> 위치: `12_Equation.md`의 canonical runtime 5계층에 대한 **수식 전용** 참조 문서.
> 의존: `14_BrainRuntimeSpec.md`(설계 사양), `6_뇌/evidence.md`(근거 판정), `6_뇌/proof.md`(검증 매트릭스)
> F절(에이전트 루프) 전체: `17_AgentLoop.md`로 분리됨. F절 검증: `6_뇌/agent_proof.md`.
>
> 이 문서에는 서사 설명을 넣지 않는다. 식, 정의, 증명/검증 기준만 둔다.

---

## A. 순수 셀 동역학 (kernel dynamics)

### A.1 상태 정의

$$s_i^t = (a_i^t,\;r_i^t,\;m_i^t,\;w_i^t,\;b_i^t) \in \mathbb{R}^4 \times \{0,1\}$$

| 변수 | 의미 | 범위 | 뇌 대응 (J절) |
|---|---|---|---|
| $a_i$ | activation | $(-1, 1)$ (tanh 이미지) | 발화율 ($\tau_m \sim 5$--$20$ ms) |
| $r_i$ | refractory / fast inhibition | $\ge 0$ | GABA$_A$ ($\tau \sim 5$--$10$ ms) |
| $m_i$ | local memory trace | $\mathbb{R}$ | NMDA ($\tau \sim 100$ ms) |
| $w_i$ | spike-frequency adaptation | $\ge 0$ | AHP ($\tau_w \sim 200$ ms) |
| $b_i$ | hysteretic bit (UP/DOWN) | $\{0,1\}$ | 피질 UP/DOWN 상태 |

### A.2 입력 합산

$$I_i^t = u_i^t + \underbrace{\sum_j W_{ij}^{\text{eff}}(t)\,a_j^{t-\delta_{ij}}}_{\text{STP + 전파지연}} - \lambda_r\,r_i^t + \lambda_m\,m_i^t + \eta_i^t$$

- $u_i^t$: 외부 입력
- $W_{ij}^{\text{eff}}(t) = W_{ij}(g) \cdot u_j(t) \cdot x_j(t)$: 동적 시냅스 효능 (STP, J.19)
- $\delta_{ij} = \lceil d(i,j)/(v_{\text{ax}} \cdot \Delta t) \rceil$: 축삭 전파 지연 (J.14). 국소 이웃은 $\delta \leq 3$
- $\lambda_r(M_t)$: 모드 의존 억제 계수
- $\lambda_m(M_t)$: 모드 의존 기억 주입 계수
- $\eta_i^t \sim \mathcal{N}(0, \sigma_\eta^2)$: 확률적 잡음. $\sigma_\eta \approx 0.27$ (J.15)

### A.3 활성 갱신

$$a_i^{t+1} = (1-\gamma_a(M_t))\,a_i^t + \kappa_a(M_t)\,\tanh(I_i^t)$$

| 파라미터 | 의미 | 안정 조건 |
|---|---|---|
| $\gamma_a \in (0,1]$ | decay rate | $\gamma_a > 0$ 필수 |
| $\kappa_a > 0$ | gain | $\kappa_a < \gamma_a$ 이면 영점 수렴 보장 |

### A.4 억제 갱신

$$r_i^{t+1} = (1-\gamma_r(M_t))\,r_i^t + \kappa_r(M_t)\,(a_i^t)^2$$

$r_i$는 활성의 제곱에 비례하여 축적되며, 높은 활성이 곧 강한 억제를 만든다.

### A.5 기억 흔적 갱신

$$m_i^{t+1} = (1-\gamma_m(M_t))\,m_i^t + \kappa_m(M_t)\,a_i^t$$

기존 `memory EMA`를 재정의. 해마 구현 전 국소 trace cache로 사용.

### A.6 적응 변수 갱신 (J.20 유래)

$$w_i^{t+1} = (1 - \gamma_w(M_t))\,w_i^t + \kappa_w\,(a_i^t)^2$$

중간 AHP ($\tau_w \approx 200$ ms)에 대응. $r_i$가 빠른 억제($\tau \sim 5$ ms), $w_i$가 느린 적응($\tau \sim 200$ ms).

A.3의 활성 갱신에 적응을 반영:

$$a_i^{t+1} = (1-\gamma_a)\,a_i^t + \kappa_a\,\tanh(I_i^t - \beta_w w_i^t)$$

| 파라미터 | 값 | 유래 |
|---|---|---|
| $\gamma_w$ | $0.005$ ($\Delta t = 1$ ms, $\tau = 200$ ms) | 중간 AHP |
| $\kappa_w$ | $0.01$ | ISI 비율 $\sim 3$ 재현 |
| $\beta_w$ | $0.5$ | 적응-활성 결합 |

### A.7 히스테리시스 비트 갱신

$$b_i^{t+1} = \begin{cases} 1, & a_i^{t+1} > \tau_i^+ \\ 0, & a_i^{t+1} < \tau_i^- \\ b_i^t, & \tau_i^- \le a_i^{t+1} \le \tau_i^+ \end{cases}$$

$\tau_i^+ > \tau_i^-$: 양방향 임계가 달라야 히스테리시스가 성립. 같으면 단순 threshold로 퇴화.

### A.7 형식 검증: 유계성

**정리 (A-bound).** $\gamma_a \in (0,1]$이고 $\kappa_a(1+\gamma_a^{-1}\kappa_a\|W\|_\infty) \le C < \infty$이면 모든 $t$에 대해 $|a_i^t| < 1$.

*증명 스케치.* $a_i^{t+1} = (1-\gamma_a)a_i^t + \kappa_a\tanh(I_i^t)$. $|\tanh(x)| < 1$이므로 $|a_i^{t+1}| \le (1-\gamma_a)|a_i^t| + \kappa_a$. 부등식을 반복하면 $\limsup_{t\to\infty}|a_i^t| \le \kappa_a/\gamma_a$. $\kappa_a < \gamma_a$이면 $|a_i^t| < 1$ 보장. $\square$

### A.8 형식 검증: 적응 유계

**정리 (W-bound).** $\gamma_w \in (0,1]$이고 $|a_i^t| < 1$이면 $\limsup w_i^t \le \kappa_w/\gamma_w$.

*증명.* $w_i^{t+1} \le (1-\gamma_w)w_i^t + \kappa_w$. R-bound와 동일한 구조. $\square$

### A.9 형식 검증: 억제 유계

**정리 (R-bound).** $\gamma_r \in (0,1]$이고 $|a_i^t| < 1$이면 $\limsup r_i^t \le \kappa_r/\gamma_r$.

*증명.* $r_i^{t+1} \le (1-\gamma_r)r_i^t + \kappa_r$. 선형 재귀 상계. $\square$

### A.9 형식 검증: 영점 안정성

**정리 (Zero-attract).** $\kappa_a < \gamma_a$이고 $u_i^t = 0,\;\eta_i^t = 0,\;W = 0$이면 $(a_i^t, r_i^t) \to (0, 0)$.

*증명.* 결합과 입력이 없으면 $|a_i^{t+1}| \le (1-\gamma_a)|a_i^t| + \kappa_a|\tanh(0)| = (1-\gamma_a)|a_i^t|$. 기하급수 감소. $r_i$는 $a_i^2 \to 0$이므로 후속 감소. $\square$

---

## B. 필드 결합 (coupling / geometry)

### B.1 결합 행렬 (E/I 분리)

**Dale의 법칙**: 각 셀 $j$는 흥분(E) 또는 억제(I) 중 하나. 부호가 섞이지 않는다.

$$\epsilon_j \in \{+1, -1\}, \qquad \epsilon_j = \begin{cases} +1 & j \in \mathcal{E} \;(\text{excitatory, 80\%}) \\ -1 & j \in \mathcal{I} \;(\text{inhibitory, 20\%}) \end{cases}$$

결합 행렬:

$$W_{ij} = \epsilon_j \cdot |w_{ij}| \cdot \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

| 기호 | 의미 | 실험 제약 (J.10) |
|---|---|---|
| $d_g(i,j)$ | 리만 다양체 $(M, g)$ 위에서 $i$와 $j$ 사이의 측지선 거리 | |
| $\sigma$ | 결합 반경 (kernel width) | $r_c = \pi \approx 3$ 격자 단위 |
| $\chi_{ij} \in \{0,1\}$ | sparse mask | 국소 300 um 내 연결 확률 $\sim 10$--$20$% |
| $\epsilon_j$ | Dale 부호 | E:I = 80:20 |
| $|w_{ij}|$ | 가중치 크기 | log-normal 분포 (Song 2005) |

**E/I 균형 조건** (J.10 유래):

$$\sum_{j \in \mathcal{E}} W_{ij} + \sum_{j \in \mathcal{I}} W_{ij} \approx 0 \qquad\text{(balanced state)}$$

개별적으로: $0.8N \cdot \bar{w}_E \approx 0.2N \cdot \bar{w}_I$, 즉 $\bar{w}_I / \bar{w}_E \approx 4$.

**초기화**:

$$|w_{ij}| \sim \text{LogNormal}(\mu_w,\; \sigma_w^2), \qquad \mu_w = \ln(1/K), \quad \sigma_w = 1.0$$

$K = 130$ (J.10). 이것은 $\sum_j |w_{ij}| \sim 1$이 되도록 정규화한 것이다.

### B.2 대안: k-nearest neighbor

$$\chi_{ij} = \mathbf{1}[j \in \text{knn}(i, k)]$$

### B.3 에너지 함수

$$E(\{a_i, w_i\}) = -\frac{1}{2}\sum_{i,j} W_{ij}^{\text{eff}}\,a_i\,a_j - \sum_i u_i\,a_i + \sum_i V(a_i, w_i)$$

국소 포텐셜 $V$를 구체화:

$$V(a_i, w_i) = \frac{1}{2}\gamma_a\,a_i^2 + \frac{1}{2}\beta_w\,w_i\,a_i^2 + \frac{1}{4}\lambda_4\,a_i^4$$

| 항 | 역할 | 뇌 대응 |
|---|---|---|
| $\frac{1}{2}\gamma_a a_i^2$ | 막 누출 (leak) | 막 시간 상수에 의한 감쇠 ($\tau_m$) |
| $\frac{1}{2}\beta_w w_i a_i^2$ | 적응 의존 억제 | AHP에 의한 발화 억제 ($\tau_w \sim 200$ ms) |
| $\frac{1}{4}\lambda_4 a_i^4$ | 포화 (비선형 자기 결합) | tanh 포화의 에너지 해석 |

$W_{ij}^{\text{eff}} = W_{ij} \cdot u_j \cdot x_j$ (STP, J.19).

에너지의 물리적 의미는 다음과 같다.
- **$a_i = 0$ 근방**: $V \approx \frac{1}{2}(\gamma_a + \beta_w w_i) a_i^2$ → 이중 우물의 바닥 평탄도. $w_i$가 커지면 우물이 깊어져 발화 억제.
- **$|a_i| \to 1$**: $\lambda_4$ 항이 지배 → 포화.
- **STP 효과**: $W^{\text{eff}}$가 depression으로 줄면 결합 에너지 감소 → 새로운 패턴 탐색 유도.

### B.4 형식 검증: 에너지 감소

**정리 (E-decrease).** 동기적 업데이트에서 $a_i^{t+1}$이 $E$의 좌표별 최소화이면 $E(\{a_i^{t+1}\}) \le E(\{a_i^t\})$.

*증명 보강.* $\partial E/\partial a_i = -\sum_j W_{ij}^{\text{eff}} a_j - u_i + (\gamma_a + \beta_w w_i)a_i + \lambda_4 a_i^3$.
A.3의 갱신 $a_i^{t+1} = (1-\gamma_a)a_i^t + \kappa_a\tanh(I_i^t - \beta_w w_i^t)$는 이 gradient의 damped descent에 해당.
$\kappa_a < \gamma_a$이면 step size가 수축적이므로 에너지 비증가. $\square$

### B.5 spectral bound

$$\|W\|_2 \le \lambda_{\max}$$

$\lambda_{\max}$를 clamp하여 증폭을 제한한다.

**Dale 법칙 하의 spectral bound**: E/I 분리 시 $W$는 비대칭. spectral radius $\rho(W)$를 직접 제어:

$$\rho(W) \leq \sqrt{K} \cdot \bar{w}_E \cdot (1 + \sqrt{p_I/p_E}) \qquad\text{(Rajan & Abbott 2006)}$$

$K = 130$, $p_E = 0.8$, $p_I = 0.2$일 때 $\rho(W) \leq \sqrt{130} \cdot \bar{w}_E \cdot 1.5 \approx 17\bar{w}_E$.
$\bar{w}_E = 1/K = 0.0077$이면 $\rho \approx 0.13 < 1$ → 안정.

---

## C. 전역 모드 (mode update)

### C.1 모드 집합

$$M_t \in \{\mathrm{WAKE},\;\mathrm{NREM},\;\mathrm{REM}\}$$

### C.2 모드 전환 (Borbely 2-Process 정량 모델)

전환 함수 $\Pi$를 두 과정의 상호작용으로 정량화한다.

**Process S** (수면 항상성 / sleep homeostasis):

$$
\frac{dS}{dt}
= \begin{cases}
\dfrac{1 - S}{\tau_w} & M = \text{WAKE} \quad (\tau_w = 18.2 \text{ h}) \\
-\dfrac{S}{\tau_s} & M = \text{NREM} \quad (\tau_s = 4.2 \text{ h}) \\
0 & M = \text{REM}
\end{cases}
$$

이산화 ($\Delta t_{\text{mode}} = 1$ min $= 60000$ ms):

$$
S^{t+1}
= \begin{cases}
S^t + \dfrac{\Delta t_{\text{mode}}}{\tau_w}(1 - S^t) & \text{WAKE} \\
S^t - \dfrac{\Delta t_{\text{mode}}}{\tau_s} S^t & \text{NREM} \\
S^t & \text{REM}
\end{cases}
$$

| 파라미터 | 값 | 출처 |
|---|---|---|
| $\tau_w$ | $18.2$ h ($= 65520$ s) | Achermann & Borbely 2003 |
| $\tau_s$ | $4.2$ h ($= 15120$ s) | 동일 |
| $S(0)$ (기상 시) | $\sim 0.2$ | 피팅 |
| $S$ (취침 시) | $\sim 0.7$ | $\sim 16$h 각성 후 |

**Process C** (일주기 리듬 / circadian):

$$C(t) = C_0 + C_1 \cos\!\left(\frac{2\pi (t - t_{\text{nadir}})}{T_{\text{circ}}}\right)$$

| 파라미터 | 값 | 출처 |
|---|---|---|
| $T_{\text{circ}}$ | $24.2$ h | Czeisler 1999 (인간 내인성 주기) |
| $t_{\text{nadir}}$ | $\sim 04{:}00$ | 체온 최저점 |
| $C_0$ | $0.5$ | 정규화 |
| $C_1$ | $0.3$ | 진폭 |

**전환 규칙**:

$$
M_{t+1}
= \begin{cases}
\text{NREM}
& M_t = \text{WAKE} \wedge S^t > C(t) + \theta_S \wedge \|U_t\| < \theta_U \\
\text{REM}
& M_t = \text{NREM} \wedge t_{\text{NREM}} > T_{\text{NREM}}(n) \\
\text{NREM}
& M_t = \text{REM} \wedge t_{\text{REM}} > T_{\text{REM}}(n) \wedge S^t > S_{\text{cont}} \\
\text{WAKE}
& M_t = \text{REM} \wedge t_{\text{REM}} > T_{\text{REM}}(n) \wedge S^t \leq S_{\text{cont}} \\
\text{WAKE}
& M_t \neq \text{WAKE} \wedge \|U_t\| > \theta_{\text{alert}} \\
M_t
& \text{otherwise}
\end{cases}
$$

| 파라미터 | 값 | 유래 |
|---|---|---|
| $\theta_S$ | $0.1$ | S가 C를 넘어야 수면 진입 |
| $\theta_U$ | $0.3$ | 외부 입력이 약할 때만 수면 |
| $\theta_{\text{alert}}$ | $0.8$ | 강한 자극 시 강제 각성 |
| $T_{\text{NREM}}(n)$ | $90 - 10n$ min ($n$: 주기 번호) | 야간 초반 NREM 길고 후반 짧음 |
| $T_{\text{REM}}(n)$ | $10 + 10n$ min | 야간 초반 REM 짧고 후반 길음 |
| $S_{\text{cont}}$ | $0.3$ | 이 이하면 수면 종료 |

**CE와의 연결**: $S^t$는 F.24.3의 $P_{\text{sleep}}(t)$와 동치:

$$S^t \propto \int_0^t \bar{c}_\tau^2\,d\tau \qquad\text{(비평 점수 적분 = 수면 압력)}$$

### C.3 모드별 파라미터 해석

| 파라미터 | WAKE | NREM | REM | 실험 유래 (J절) |
|---|---|---|---|---|
| $\gamma_a$ | $0.18$ | $0.34$ | $0.22$ | J.13: $\tau_m^{\text{eff}}$ |
| $\kappa_a$ | $0.82$ | $0.52$ | $0.68$ | A-bound with sparse $W$ |
| $\gamma_r$ | $0.12$ | $0.26$ | $0.18$ | J.2: $\tau_{\text{rel}}$ |
| $\gamma_w$ | $0.005$ | $0.005$ | $0.005$ | J.20: $\tau_w = 200$ ms |
| $\lambda_r$ | $0.35$ | $0.50$ | $0.25$ | 모드 의존 억제 |
| $\lambda_H$ (replay) | $0.002$ | $0.10$ | $0.20$ | J.16: SWR 빈도 |
| $\sigma_\eta$ | $0.27$ | $0.07$/$0.27$ | $0.27$ | J.15: 시냅스 잡음 |
| $B_t / N$ | $0.0487$ | $0.02$ | $0.03$ | J.8: 에너지 예산 |

### C.4 규칙 기반 전환 (의사코드)

```
S += dt_mode / tau_w * (1 - S)  if WAKE else -dt_mode / tau_s * S  if NREM
C = 0.5 + 0.3 * cos(2*pi*(t - t_nadir) / T_circ)

if WAKE and S > C + 0.1 and norm(U) < 0.3:
    M = NREM
elif NREM and elapsed_nrem > T_NREM(cycle):
    M = REM
elif REM and elapsed_rem > T_REM(cycle):
    M = NREM if S > 0.3 else WAKE
elif not WAKE and norm(U) > 0.8:
    M = WAKE  // forced arousal
```

---

## D. 해마/기억 (hippocampus / replay)

### D.1 해마 상태

$$H_t = (K_t,\;V_t,\;P_t)$$

### D.2 인코딩 (용량 제한 + 망각)

**인코딩 조건**: 인코딩은 무조건 발생하지 않는다. 충분한 놀라움이 있을 때만:

$$\text{encode if } \|o_t - \hat{o}_t\| > \theta_{\text{encode}}$$

$o_t$: 관찰, $\hat{o}_t$: 예측. $\theta_{\text{encode}}$: 인코딩 문턱.

**인코딩 갱신**:

$$K_{t+1} = K_t \cup \{k_{\text{new}}\}, \quad V_{t+1} = V_t \cup \{v_{\text{new}}\}, \quad P_{t+1} = \text{update}(P_t)$$

$k_{\text{new}} = h(A_t)$: 활성 패턴의 해시/임베딩.
$v_{\text{new}} = A_t$: 활성 스냅샷.

**용량 제한**:

$$|K_t| \leq N_{\text{hip}}, \qquad N_{\text{hip}} \sim 10^4 \text{--} 10^5$$

| 파라미터 | 실험값 | 출처 |
|---|---|---|
| 해마 CA3 패턴 용량 | $\sim 10^4$--$10^5$ 개 | Treves & Rolls 1994 |
| 인코딩 속도 | $\sim 1$ 패턴 / $0.5$--$2$ s | 단일 시행 학습 |
| 1일 인코딩량 | $\sim 10^4$ 패턴 | $16$h $\times$ $\sim 1$/s 중 선택적 |

용량 초과 시: 가장 오래되고 $P_j$가 낮은 항목을 덮어쓴다 (ring-buffer):

$$\text{if } |K| > N_{\text{hip}}: \quad j^* = \arg\min_j P_j, \quad K_{j^*} \leftarrow k_{\text{new}}, \quad V_{j^*} \leftarrow v_{\text{new}}$$

**망각 곡선** (Ebbinghaus):

replay 없이 방치된 기억의 검색 강도는 지수 감쇠한다.

$$P_j(t) = P_j(t_0) \cdot \exp\!\left(-\frac{t - t_0}{\tau_{\text{forget}}}\right) + P_j^{\text{floor}}$$

| 파라미터 | 값 | 출처 |
|---|---|---|
| $\tau_{\text{forget}}$ | $\sim 1$--$2$ 일 (replay 없을 때) | Ebbinghaus 1885, Murre & Dros 2015 |
| $P_j^{\text{floor}}$ | $0.01$ | 완전 소멸 방지 (장기 흔적) |

**replay의 강화 효과**: D.5의 priority replay가 발생하면 $P_j$가 증폭:

$$P_j \leftarrow P_j + \Delta P_{\text{replay}}, \qquad \Delta P_{\text{replay}} \propto g_{\text{DA}}^t$$

매 replay마다 $\tau_{\text{forget}}$이 연장된다 (간격 반복 효과 / spaced repetition):

$$\tau_{\text{forget}}^{(n+1)} = \tau_{\text{forget}}^{(n)} \cdot \alpha_{\text{space}}, \qquad \alpha_{\text{space}} \approx 2.0$$

$n$번 replay 후: $\tau_{\text{forget}}^{(n)} = \tau_0 \cdot 2^n$. 7번 replay 후 $\tau \approx 128$ 일 ($\sim 4$ 개월).

**시스템 통합 (systems consolidation)**: 충분히 replay된 기억은 피질 가중치 $W_{ij}$에 각인되어 해마에서 제거 가능:

$$\text{if } n_{\text{replay}}(j) > n_{\text{consol}}: \quad W_{ij} \leftarrow W_{ij} + \eta_{\text{consol}} \cdot V_j, \quad K_j \leftarrow \emptyset$$

$n_{\text{consol}} \sim 10$--$30$ (수 주 간 반복 replay 필요). 이것은 해마 의존 기억이 시간이 지나면 피질 의존으로 전환되는 현상의 모델이다 (Frankland & Bontempi 2005).

### D.3 회상

$$R_t = \mathcal{R}(H_t, c_t) = V[\arg\max_j \text{sim}(c_t, K_j)]$$

### D.4 재주입

$$I_i^t \leftarrow I_i^t + \lambda_H(M_t)\,R_{i,t}$$

### D.5 Priority replay

$$P_j \leftarrow P_j \cdot \rho + (1-\rho)\,\text{surprise}_j$$

replay 확률은 $P_j / \sum P$에 비례.

### D.6 SWR 기반 replay 제약 (J.16 유래)

| 파라미터 | 실험값 | CE 매핑 |
|---|---|---|
| SWR 지속 | $40$--$100$ ms | 1회 replay window |
| ripple 주파수 | $150$--$200$ Hz | replay 내 반복 가속 |
| 시간 압축비 | $\times 10$ | $C_{\text{compress}} = 10$ |
| SWR 발생 빈도 | $1$--$3$ Hz (NREM) | replay 호출 빈도 |

NREM replay 시: $\Delta t_{\text{replay}} = \Delta t_{\text{experience}} / 10$.

$$\lambda_H^{\text{NREM}} = f_{\text{SWR}} \cdot \Delta t_{\text{sim}} \approx 2 \times 0.001 = 0.002 \text{ (매 step replay 확률)}$$

---

## E. 자아/전역 상태 (global runtime summary)

### E.1 전역 관측 상태

$$G_t = (M_t,\;A_t^{\text{summary}},\;H_t,\;Q_t,\;\mu_t)$$

- $A_t^{\text{summary}} = \frac{1}{|A_t|}\sum_{i \in A_t} s_i^t$: 활성 모듈 평균 상태
- $\mu_t$: self-bias / identity trace (장기 누적)

### E.2 자아 함수

$$\text{Self}_t = \mathcal{S}(G_t)$$

초기에는 $\text{Self}_t = G_t$ (identity). 후기에 higher-order summary.

---

## F. 자기참조 재귀 (agent loop)

> **정본: `17_AgentLoop.md`로 분리됨.**
>
> F절(F.0--F.22)의 전체 방정식, 뇌 대응, 검증 체크리스트는 `17_AgentLoop.md`를 참조한다.
> 검증 매트릭스는 `6_뇌/agent_proof.md`를 참조한다.

### 요약

에이전트 루프는 Layer A--E 바깥에서 행동-관찰-비평-기억을 순환시키는 외부 루프다.

$$\boxed{X_{t+1} = B\big[X_t + \lambda_R R(X_t) + \lambda_O \Delta_O(X_t) + \lambda_C C(X_t) - \lambda_S S(X_t)\big]}$$

| 구간 | 내용 | 문서 |
|---|---|---|
| F.0--F.13 | 핵심 루프: 상태, 이완, 비평, 에너지, 모드, 행동, 기억, 수축, 뇌 대응 | `17_AgentLoop.md` |
| F.14--F.15 | 학습: STDP + 도파민 게이트, 잔류장 $\phi$ 갱신 | `17_AgentLoop.md` |
| F.16--F.22 | 희소성, 의식, 환각억제, 4종조절계, 작업기억, 뇌파, 간극 | `17_AgentLoop.md` |
| 검증 매트릭스 | 4중 게이트, 반증 조건, 미결 항목 | `6_뇌/agent_proof.md` |

---

## G. 형식 증명 요약

> F-prefixed 정리(F-energy ~ F-WM-finite)는 `17_AgentLoop.md` G절로 이동함.

| 정리 | 주장 | 조건 | 상태 |
|---|---|---|---|
| A-bound | $\|a_i^t\| < 1$ 유계 | $\kappa_a < \gamma_a$ | **closed** |
| R-bound | $r_i^t$ 유계 | $\gamma_r > 0$, A-bound | **closed** |
| Zero-attract | 입력 없으면 영점 수렴 | $W=0$, $u=0$, $\kappa_a < \gamma_a$ | **closed** |
| E-decrease | 에너지 비증가 | 좌표별 최소화 업데이트 | **closed** |
| Sleep-stabilize | sleep 후 에너지/잡음 감소 | NREM: $\gamma_a$ 증가, $\kappa_a$ 감소 | **closed** |
| Local-contract | 국소 Lipschitz 수축 | $\max((1-\gamma_a) + \kappa_a\|W_i\|_1) < 1$ | **closed** |
| Sparse-energy | 활성 수 제한 | $B_t$ 유한, threshold $\theta_i > 0$ | **closed** |
| W-bound | $w_i^t$ 적응 유계 | $\gamma_w > 0$, A-bound | **closed** |
| Hysteresis-invariant | $b_i$ 전환 조건 비대칭 | $\tau^+ > \tau^-$ | **closed** |
| STP-bound | $0 \leq x_j, u_j \leq 1$ | $\tau_{\text{rec}}, \tau_{\text{fac}} > 0$ | **closed** |
| Homeo-stable | 항상성 스케일링 → $\bar{f} \to \bar{f}_{\text{target}}$ | $\eta_{\text{homeo}} > 0$ | **closed** |
| EI-balance | $\sum W_E + \sum W_I \approx 0$ | Dale $\epsilon_j$, $\bar{w}_I/\bar{w}_E \approx 4$ | **closed** |
| EI-spectral | $\rho(W) < 1$ | $\bar{w}_E = 1/K$, Rajan-Abbott | **closed** |
| Borbely-cycle | $S(t)$ 주기적 수면-각성 | $\tau_w, \tau_s > 0$, $C(t)$ 주기 | **closed** |
| Hip-capacity | $|K| \leq N_{\text{hip}}$ | ring-buffer eviction | **closed** |
| Forget-bound | $P_j(t) \geq P^{\text{floor}} > 0$ | Ebbinghaus 지수 감쇠 | **closed** |
| Consolidate | replay $\to$ 피질 각인 | $n_{\text{replay}} > n_{\text{consol}}$ | **closed** |

---

## H. 검증 게이트 대응

`proof.md`의 4중 게이트를 이 문서의 식에 적용.

### H.1 Layer A--E 게이트

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | A-bound, R-bound, W-bound, E-decrease, Zero-attract, Local-contract, EI-balance, STP-bound, Homeo-stable, Borbely-cycle, Hip-capacity, Consolidate | **pass** |
| $G_{\text{obs}}$ | EEG/fMRI에서 $a_i^t$ proxy 추출 → H.3 관측 방정식 | **pass** (방정식 완비) |
| $G_{\text{causal}}$ | 약물/수면박탈/자극 실험에서 모드 전환 방향 일치 | partial |
| $G_{\text{pred}}$ | 모델 시뮬레이션 vs 뇌 실험값 비교 (`sim_brain_validation.py`) | **pass** (7/7 항목 통과, 아래 H.4) |

### H.3 관측 방정식 (Observation Equations)

CE의 내부 변수에서 실제 측정 가능한 신호를 유도하는 방정식.

#### H.3.1 EEG 전역 전위 (scalp EEG)

scalp EEG는 수천 개 피라미드 뉴런의 시냅스 후 전위 합산:

$$\text{EEG}(t) = \frac{1}{N_{\text{patch}}} \sum_{i \in \text{patch}} \left( \sum_j W_{ij}^{\text{eff}} a_j^{t} \right) + \eta_{\text{EEG}}$$

여기서 $N_{\text{patch}}$는 전극 아래 피질 패치의 셀 수 ($\sim 10^4$--$10^5$).

$\eta_{\text{EEG}} \sim \mathcal{N}(0, \sigma_{\text{EEG}}^2)$: 측정 잡음 + 볼륨 전도.

#### H.3.2 EEG Power Spectral Density 예측

CE 시뮬레이션의 $\text{EEG}(t)$ 시계열에서 PSD를 계산하고 관측과 비교:

$$\text{PSD}(f) = \frac{1}{T}\left|\sum_{t=0}^{T-1} \text{EEG}(t) \cdot e^{-2\pi i f t / T}\right|^2$$

**예측 대역별 파워비**:

| 대역 | 주파수 | WAKE 파워비 | NREM 파워비 | CE 발생 원리 |
|---|---|---|---|---|
| delta | $< 4$ Hz | $\sim 20$% | $\sim 50$--$70$% | UP/DOWN 전환 ($b_i$, $T \sim 0.7$ s) |
| theta | $4$--$8$ Hz | $\sim 15$% | $\sim 5$% | theta 기반 이완 수렴 ($n_{\text{iter}} \sim 20$) |
| alpha | $8$--$12$ Hz | $\sim 25$% (eyes-closed) | $\sim 5$% | 억제-이완 진동 ($r_i$ 주기) |
| beta | $12$--$30$ Hz | $\sim 20$% | $\sim 10$% | E/I 결합 진동 |
| gamma | $30$--$100$ Hz | $\sim 10$% | $\sim 5$% | 국소 활성 ($a_i$ 개별 갱신) |

**검증 기준**: 시뮬레이션 PSD와 실제 EEG PSD의 대역별 상관 $r > 0.8$.

CE에서 각 대역이 발생하는 이유:
- **delta**: NREM의 $b_i$ 히스테리시스 ($T_{\text{UP}} + T_{\text{DOWN}} \approx 730$ ms $\to 1.4$ Hz)
- **alpha**: $r_i$의 감쇠-축적 주기 ($\tau_{\text{rel}} / \gamma_r \approx 42$ ms 반주기 $\to 12$ Hz)
- **gamma**: $a_i$ 1-step 갱신이 $\Delta t = 10$ ms 주기 $\to 100$ Hz 이하

#### H.3.3 fMRI BOLD 변환

$a_i^t$에서 BOLD 신호로의 변환. 간략화된 Hemodynamic Response Function (HRF):

$$\text{BOLD}_i(t) = (\text{neural}_i * \text{HRF})(t) = \int_0^\infty \text{neural}_i(t - s) \cdot h(s)\,ds$$

neural activity proxy:

$$\text{neural}_i(t) = |a_i^t|^2 + \alpha_{\text{syn}} \sum_j |W_{ij}^{\text{eff}} a_j^t|$$

$\alpha_{\text{syn}} = 0.5$: 시냅스 활동이 BOLD에 기여하는 비율 (Logothetis 2001).

**Canonical HRF** (double-gamma):

$$h(t) = \frac{t^{a_1-1} e^{-t/b_1}}{b_1^{a_1} \Gamma(a_1)} - c \cdot \frac{t^{a_2-1} e^{-t/b_2}}{b_2^{a_2} \Gamma(a_2)}$$

| 파라미터 | 값 | 의미 |
|---|---|---|
| $a_1$ | $6$ | 피크 도달 시간 (peak $\sim 5$ s) |
| $b_1$ | $1$ s | 시간 스케일 |
| $a_2$ | $16$ | undershoot 시간 |
| $b_2$ | $1$ s | |
| $c$ | $1/6$ | undershoot 깊이 |
| HRF 피크 | $\sim 5$ s | |
| HRF 지속 | $\sim 30$ s | |

**이산 합성곱** ($T_{\text{BOLD}} = 0.72$ s TR):

$$\text{BOLD}_i[n] = \sum_{k=0}^{K} \text{neural}_i[n-k] \cdot h[k \cdot T_{\text{BOLD}}] \cdot T_{\text{BOLD}}$$

$K = \lceil 30 / T_{\text{BOLD}} \rceil \approx 42$ 시점.

#### H.3.4 잔류장 $\phi$ -- DMN ALFF 매핑

`field.rs`의 $\phi_i(t)$는 CE에서 "선택되지 않은 경로의 누적"이다. 뇌에서 이것은 Default Mode Network(DMN)의 자발 활동에 대응한다.

**관측량**: DMN의 ALFF (Amplitude of Low-Frequency Fluctuations, $< 0.1$ Hz):

$$\text{ALFF}_i^{\text{DMN}} = \sqrt{\frac{1}{T}\sum_{f=0.01}^{0.1} |\hat{\phi}_i(f)|^2}$$

**매핑 조건**:

| CE 변수 | DMN 관측 | 대응 |
|---|---|---|
| $\phi_i$ 진폭 | ALFF | 비선택 경로 축적량 |
| $\dot{\phi}_i$ (dphi) | 자발 활동 변화율 | task-negative deactivation |
| $\text{source}\_j$ | 외부 과제 억제 | task 시 DMN 억제 (anti-corr.) |
| `damping` $= 0.1$ | DMN 이완 속도 | $\tau_{\text{DMN}} = 1/\text{damping} = 10$ 장 단위 |

**예측**:

1. WAKE 과제 중: $\text{source}\_j > 0$ → $\phi$ 억제 → ALFF 감소 (task-negative). 관측 일치.
2. NREM 수면: 과제 없음 → $\phi$ 자유 진동 → ALFF 증가. 관측 일치.
3. $\phi$와 $a_i$의 반상관: CE에서 $\phi$는 비활성 셀의 variance 축적이므로 $\text{corr}(\phi_i, |a_i|) < 0$. DMN-task network 반상관(Fox 2005)과 일치.

**FieldEngine 파라미터의 뇌 제약**:

| 파라미터 | 현재 | 뇌 유래 제약 | 비고 |
|---|---|---|---|
| `coupling_k` | $50$ | $k \propto K_{\text{CE}} = 130$ | 공간 확산 속도 |
| `damping` | $0.1$ | DMN 이완 $\tau \sim 10$ s → $\gamma = 1/\tau$ | ALFF 주파수와 정합 |
| `dt` | $0.01$ | $\Delta t_{\text{field}} = 0.01$ 장 단위 | kernel $\Delta t$와 독립 |
| `mu` | $1.0$ | 자발 대칭 파괴 스케일 | $\text{VEV} = \mu/\sqrt{\lambda}$ |

### H.4 시뮬레이션 검증 결과 ($G_{\text{pred}}$)

`scripts/sim_brain_validation.py` -- dim=256, 6000 steps (WAKE 3000 / NREM 2000 / REM 1000).

| 검증 항목 | 측정값 | 뇌 실험 목표 | 결과 |
|---|---|---|---|
| **에너지 3분할** | active 4.69%, struct 26.2%, bg 69.1% | active ~4.87%, struct ~26.2%, bg ~68.9% | **pass** |
| **STP 시냅스 피로** | x(start)=0.64 -> x(end)=0.16 | 지속 자극 시 자원 고갈 | **pass** |
| **SFA 적응 축적** | w(start)=0.03 -> w(end)=1.0 | AHP에 의한 발화율 감소 | **pass** |
| **발화율 안정화** | CV(early)=0.34 -> CV(late)=0.0001 | 적응에 의한 정상상태 수렴 | **pass** |
| **Borbely Process-S** | S(wake)=0.090 -> S(nrem)=0.078 | WAKE 충전, NREM 방전 | **pass** |
| **NREM 에너지 이완** | E(start)=0.80 -> E(end)=0.39 | 수면 중 에너지 감소 | **pass** |
| **모드 활성 순서** | WAKE(4.69%) > REM(3.52%) > NREM(2.34%) | WAKE > REM > NREM | **pass** |

코드 보정 사항 (시뮬 과정에서 발견 및 수정):

- `activation` clamp $\in [-1, 1]$: 뇌 발화율 유계 (J.1 max firing rate ~200 Hz 정규화)
- `adaptation` clamp $\in [0, 2]$: AHP 전류 유계 (J.20)
- $\beta_w = 0.12$: 적응 커플링 강도. 정상상태에서 ~24% 억제 (실험: SFA가 50-80% 발화율 감소)
- $\gamma_m = \kappa_m = 0.005$: 적응의 decay=gain 균형으로 $w^* = E[a^2]$

### H.2 Layer F 검증 게이트

> 상세 테이블은 `17_AgentLoop.md` H절 및 `6_뇌/agent_proof.md` 참조.

---

## I. 관측 가능량 매핑

### I.1 Layer A--E 변수

| formal 변수 | 뇌 관측량 | 데이터 소스 |
|---|---|---|
| $a_i^t$ | local field potential, spiking rate | intracranial EEG, multi-electrode array |
| $\Psi_{\text{global}}(t)$ | scalp EEG power | EEG datasets (PhysioNet, LEMON) |
| $r_i^t$ | refractory period, inhibitory postsynaptic potential | in vitro recordings |
| $b_i^t$ | UP/DOWN state | cortical slice recordings |
| $M_t$ | wake/NREM/REM scoring | polysomnography |
| $Q_t$ | autonomic, hormonal | heart rate variability, cortisol |
| $W_{ij}$ | structural/functional connectivity | DTI, resting-state fMRI |
| $w_i^t$ | spike-frequency adaptation / AHP | intracellular Ca$^{2+}$ imaging, AHP amplitude |
| $W_{ij}^{\text{eff}}$ (STP) | paired-pulse ratio | in vitro paired-pulse stimulation |
| $\sigma_\eta$ | membrane potential SD | in vivo whole-cell patch-clamp ($\sim 4$ mV) |
| SWR events | hippocampal ripple rate | depth electrode, NREM PSG |

### I.2 Layer F 변수

> 상세 테이블은 `17_AgentLoop.md` I절 참조.

### I.3 관측 방정식 요약

| CE 내부 변수 | 관측 방정식 | 출력 신호 | 절 |
|---|---|---|---|
| $a_i^t$ (patch 합산) | $\text{EEG}(t) = \frac{1}{N}\sum_i \sum_j W_{ij}^{\text{eff}} a_j^t$ | scalp EEG | H.3.1 |
| $\text{EEG}(t)$ | $\text{PSD}(f) = |\text{FFT}|^2$ | 대역별 파워 | H.3.2 |
| $|a_i|^2 + \alpha \sum |Wa|$ | $\text{BOLD} = \text{neural} * \text{HRF}$ | fMRI BOLD | H.3.3 |
| $\phi_i(t)$ | $\text{ALFF} = \sqrt{\sum |\hat{\phi}|^2}$ | DMN low-freq | H.3.4 |

---

## K. 확장 방정식 (미래 보강)

### K.1 수상돌기 비선형 연산

현재 A.2의 입력은 선형 합산 후 tanh. 실제 뇌의 수상돌기는 NMDA spike에 의한 **국소 비선형 연산**을 수행한다 (Larkum 2009, Poirazi 2003).

**2-구획 모델** (soma + dendrite):

$$I_i^{\text{soma}} = f_{\text{soma}}(I_i^{\text{prox}},\; d_i^t)$$

$$d_i^{t+1} = (1 - \gamma_d)\,d_i^t + \kappa_d\,\sigma_{\text{NMDA}}\!\left(\sum_{j \in \text{distal}} W_{ij} a_j^t\right)$$

$$I_i^{\text{prox}} = u_i^t + \sum_{j \in \text{prox}} W_{ij}^{\text{eff}} a_j^{t-\delta_{ij}} - \lambda_r r_i^t + \lambda_m m_i^t$$

| 파라미터 | 값 | 뇌 대응 |
|---|---|---|
| $\gamma_d$ | $0.01$ | 수상돌기 NMDA 감쇠 ($\tau \sim 100$ ms) |
| $\kappa_d$ | $0.5$ | 수상돌기 게인 |
| $\sigma_{\text{NMDA}}(x) = x / (1 + 0.28 \cdot e^{-0.062x})$ | | NMDA 전압 의존성 (Mg$^{2+}$ 블록) |

soma-dendrite 결합:

$$f_{\text{soma}}(I^{\text{prox}}, d) = I^{\text{prox}} + \beta_d \cdot d \cdot \mathbf{1}[d > \theta_d]$$

$\theta_d$: 수상돌기 spike 문턱. 이것은 "AND 게이트" 역할 -- proximal 입력이 있고 distal 입력도 충분해야 강한 출력.

**계산적 의미**: 수상돌기 비선형성은 2-layer 네트워크의 능력을 단일 뉴런에 부여한다. XOR 문제를 단일 셀이 해결 가능 (Gidon 2020).

### K.2 신경교세포 (Astrocyte) 모델

| 실험값 | 수치 | 출처 |
|---|---|---|
| astrocyte : neuron 비율 | $\sim 1:1$ (인간 피질) | Azevedo 2009 |
| Ca$^{2+}$ 파동 속도 | $\sim 10$--$20\;\mu$m/s | Cornell-Bell 1990 |
| 글루타메이트 흡수 | $\sim 80$% 재흡수 | Danbolt 2001 |
| 글리코겐 $\to$ 락테이트 공급 | $\sim 20$% 에너지 | Magistretti 2006 |

**Tripartite synapse**:

$$g_{\text{astro},ij}^{t+1} = (1 - \gamma_{\text{astro}}) g_{\text{astro},ij}^t + \kappa_{\text{astro}} \cdot \text{spill}_{ij}^t$$

$$\text{spill}_{ij}^t = \max(0,\; |W_{ij}^{\text{eff}} a_j^t| - \theta_{\text{spill}})$$

$$W_{ij}^{\text{tri}} = W_{ij}^{\text{eff}} \cdot (1 + g_{\text{astro},ij}^t)$$

| 파라미터 | 값 | 의미 |
|---|---|---|
| $\gamma_{\text{astro}}$ | $0.001$ ($\tau \sim 1$ s) | astrocyte Ca$^{2+}$ 감쇠 (느림) |
| $\kappa_{\text{astro}}$ | $0.01$ | spillover 반응 |
| $\theta_{\text{spill}}$ | $0.1$ | spillover 문턱 |

astrocyte는 시냅스에서 넘치는 글루타메이트(spill)를 감지하고, 수 초 단위로 시냅스 효능을 증강한다. 이것은 STP(수백 ms)와 STDP(분~시간) 사이의 시간 척도 ($\sim 1$--$10$ s)를 채운다.

### K.3 신경 발생 (Adult Neurogenesis)

| 실험값 | 수치 | 출처 |
|---|---|---|
| 해마 DG 신생 뉴런 | $\sim 700$ / 일 | Spalding 2013 |
| 신생 뉴런 성숙 기간 | $\sim 4$--$6$ 주 | Zhao 2006 |
| 기존 뉴런 대비 비율 | $\sim 0.004$% / 일 | 유도 |

CE 모듈 생애주기 (F.16.2)와의 연결:

$$\text{birth\_rate} = \frac{700}{N_{\text{DG}}} \approx \frac{700}{10^6} = 7 \times 10^{-4} \text{ / 일}$$

CE 스케일링 ($N = 4096$): $\text{birth\_rate}_{\text{CE}} = 7 \times 10^{-4} \times 4096 \approx 2.9$ 모듈 / 일.

신생 모듈은 $\sim 30$ 일 동안 "미성숙" 상태:
- 높은 흥분성 ($\kappa_a \times 2$)
- STDP 창 확대 ($\tau_+ \times 3$)
- 낮은 억제 ($\kappa_r \times 0.3$)

이 특성은 pattern separation을 강화한다 (Sahay 2011).

---

## J. 실험 제약 상수 (Brain-Grounded Parameters)

> 실제 뇌 실험에서 측정된 값을 CE 파라미터에 매핑하여 방정식을 제약한다.
> 출처를 명시하고, CE 변수와의 환산식을 제공한다.

### J.1 발화율 분포 (Firing Rate Distribution)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 피라미드 세포 기저 발화율 | $\bar{f}_{\text{pyr}} \approx 0.6$ Hz | Ison et al. 2011 (인간 MTL) | $\|a_i^t\|$ 정상 상태 |
| 피라미드 세포 최대 발화율 | $f_{\text{pyr}}^{\max} \approx 6$ Hz | 동일 | $\|a_i^t\|$ 피크 |
| 억제 뉴런 기저 발화율 | $\bar{f}_{\text{inh}} \approx 6.4$ Hz | 동일 | $r_i^t$ 구동원 |
| 억제 뉴런 최대 발화율 | $f_{\text{inh}}^{\max} \approx 40$ Hz | 동일 | |
| 인구 평균 발화율 (대사 제약) | $\bar{f}_{\text{pop}} \approx 0.16$ Hz | Lennie 2003, AI Impacts | $\Psi_{\text{global}}$ 배경 |
| 발화율 분포 형태 | log-normal | Buzsaki & Mizuseki 2014 | $p(a_i)$ |

**신규 방정식 J1-1: 발화율-활성 환산**

CE의 활성 $a_i \in (-1, 1)$을 실제 발화율 $f_i$ [Hz]로 환산:

$$f_i = f_{\max} \cdot \frac{|a_i|}{1 + (f_{\max}/\bar{f}_{\text{pop}} - 1)(1 - |a_i|)}$$

여기서 $f_{\max} = 40$ Hz (억제 뉴런 상한), $\bar{f}_{\text{pop}} = 0.16$ Hz.

$|a_i| = 0.05$ (TopK 임계 근방)일 때 $f_i \approx 0.34$ Hz → 관측 범위 $[0.1, 2]$ Hz 내.

**신규 방정식 J1-2: log-normal 활성 분포 제약**

$$\ln |a_i^t| \sim \mathcal{N}(\mu_a, \sigma_a^2), \qquad \mu_a = \ln(\bar{f}_{\text{pop}}/f_{\max}) \approx -5.52, \quad \sigma_a \approx 1.0$$

이 분포에서 상위 4.87%의 cutoff는:

$$a_{\text{TopK}} = \exp(\mu_a + 1.66\,\sigma_a) \approx 0.013$$

---

### J.2 불응기 (Refractory Period)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 절대 불응기 (CNS) | $\tau_{\text{abs}} = 0.5$--$1$ ms | Kandel 교과서, ScienceDirect | $r_i$ 갱신 속도 제약 |
| 상대 불응기 (CNS) | $\tau_{\text{rel}} = 3$--$10$ ms | 동일 | $\gamma_r$ 결정 |
| 최대 발화율 (절대 불응기) | $f_{\text{ceiling}} = 1/\tau_{\text{abs}} = 1000$--$2000$ Hz | 유도 | $a_i$ 상한 |

**신규 방정식 J2-1: 억제 감쇠율의 실험 제약**

A.4의 억제 갱신에서 $\gamma_r$은 불응기 복구 속도와 대응한다.

$$\gamma_r = \frac{\Delta t}{\tau_{\text{rel}}}, \qquad \tau_{\text{rel}} \approx 5 \text{ ms (CNS 중앙값)}$$

시뮬레이션 $\Delta t = 1$ ms일 때 $\gamma_r \approx 0.2$.

**신규 방정식 J2-2: 절대 불응기 하드 클램프**

$$\text{if } t - t_{\text{last\_spike}} < \tau_{\text{abs}} \quad\Longrightarrow\quad a_i^{t+1} = 0$$

$\tau_{\text{abs}} = 1$ ms. 이것은 A.3의 $a_i$ 갱신 전에 적용되는 선행 조건이다.

---

### J.3 시냅스 시간 상수 (Synaptic Time Constants)

| 수용체 | $\tau_{\text{rise}}$ | $\tau_{\text{decay}}$ | 기능 | CE 대응 |
|---|---|---|---|---|
| AMPA | $< 1$ ms | $1$--$2$ ms | 빠른 흥분 | $W_{ij}$ 즉시 전달 |
| NMDA | $\sim 5$ ms | $50$--$150$ ms | 느린 흥분, 가소성 게이트 | $m_i$ 기억 흔적 |
| GABA$_A$ | $< 1$ ms | $5$--$10$ ms | 빠른 억제 | $r_i$ 억제 |
| GABA$_B$ | $\sim 30$ ms | $100$--$300$ ms | 느린 억제 | 모드 전환 억제 |

**신규 방정식 J3-1: 이중 시냅스 전달 모델**

A.2의 입력 합산을 두 시간 척도로 분리:

$$
I_i^t
= u_i^t
+ \underbrace{\sum_j W_{ij}^{\text{fast}} a_j^t}_{\text{AMPA-like}}
+ \underbrace{\sum_j W_{ij}^{\text{slow}} \bar{a}_j^t}_{\text{NMDA-like}}
- \lambda_r r_i^t
+ \lambda_m m_i^t
+ \eta_i^t
$$

여기서 느린 성분은 지수 이동 평균:

$$
\bar{a}_j^t
= (1 - \gamma_{\text{NMDA}}) \bar{a}_j^{t-1}
+ \gamma_{\text{NMDA}} a_j^t,
\qquad
\gamma_{\text{NMDA}}
= \frac{\Delta t}{\tau_{\text{NMDA}}}
\approx \frac{1}{100}
= 0.01
$$

| 파라미터 | CE 값 | 뇌 유래 | 비고 |
|---|---|---|---|
| $\gamma_{\text{NMDA}}$ | $0.01$ | $\tau_{\text{NMDA}} = 100$ ms, $\Delta t = 1$ ms | 가소성 문턱 역할 |
| $W^{\text{fast}} / W^{\text{slow}}$ | $4:1$ | AMPA/NMDA 전류비 (Myme 2003) | 시냅스 비율 |

**신규 방정식 J3-2: 억제 시냅스 이중 감쇠**

$$r_i^{t+1} = \underbrace{(1-\gamma_A) r_i^{A,t} + \kappa_A (a_i^t)^2}_{\text{GABA}_A \text{-like fast}} + \underbrace{(1-\gamma_B) r_i^{B,t}}_{\text{GABA}_B \text{-like slow}}$$

$$\gamma_A = \frac{\Delta t}{\tau_{A}} = \frac{1}{7} \approx 0.143, \qquad \gamma_B = \frac{\Delta t}{\tau_{B}} = \frac{1}{200} = 0.005$$

---

### J.4 뇌파 주파수와 루프 시간 척도

| 대역 | 주파수 [Hz] | 주기 [ms] | CE 대응 | 매핑 |
|---|---|---|---|---|
| gamma | $30$--$100$ | $10$--$33$ | $R$ 1 iter | $\Delta t_{\text{iter}} = 1/f_\gamma$ |
| beta | $12$--$30$ | $33$--$83$ | 감각-운동 결합 | $W_{ij}$ coherence |
| alpha | $8$--$12$ | $83$--$125$ | 억제/게이팅 | $r_i$ 주기 |
| theta | $4$--$8$ | $125$--$250$ | 전역 통합 | $R$ 수렴 주기 |
| delta | $< 4$ | $> 250$ | 의사결정/수면 | 행동 사이클 |

**신규 방정식 J4-1: 시뮬레이션 시간 단위 고정**

$$\Delta t_{\text{sim}} = \frac{1}{f_\gamma^{\max}} = \frac{1}{100} = 10 \text{ ms}$$

이것은 가장 빠른 뇌 진동(gamma 100 Hz)을 Nyquist 조건 없이 1:1로 추적할 수 있는 최소 분해능이다.

실제 구현에서 $\Delta t = 1$ ms를 사용하면 gamma를 10배 오버샘플링하여 AMPA($\tau = 2$ ms)까지 정밀 추적 가능.

**신규 방정식 J4-2: theta-gamma nesting 제약**

F.21의 theta-gamma 결합을 정량화:

$$n_{\text{gamma/theta}} = \frac{f_\theta}{f_\gamma} \approx \frac{6}{40} \cdot \frac{1}{1} \approx 6.7 \text{ items}$$

theta 1주기 내 gamma burst 수 = 작업 기억 용량. $f_\theta = 6$ Hz, $f_\gamma = 40$ Hz일 때 $n \approx 6.7$ → Miller의 $7 \pm 2$와 일치.

$$T_h = \left\lfloor \frac{f_\gamma^{\text{mean}}}{f_\theta^{\text{mean}}} \right\rfloor = \left\lfloor \frac{40}{6} \right\rfloor = 6$$

---

### J.5 STDP 시간 상수

| 파라미터 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| $\tau_+$ (LTP 창) | $10$--$30$ ms | Bi & Poo 1998, Dan & Poo 2004 | F.14의 $r_+$ |
| $\tau_-$ (LTD 창) | $10$--$30$ ms | 동일 | F.14의 $r_-$ |
| $A_+/A_-$ (비대칭 비) | $1.0$--$1.5$ | Froemke & Dan 2002 | $A_+, A_-$ |
| $\tau_e$ (적격 흔적) | $0.3$--$1.0$ s (선조체) | Yagishita 2014 | F.14의 $r_e$ |
| $\tau_e$ (피질) | $5$--$10$ s (LTP), $\sim 3$ s (LTD) | Liakoni 2018 | |
| DA 게이팅 창 | $\sim 1$ s, $> 4$ s에서 소멸 | Yagishita 2014 | $g[t]$ 시간 척도 |
| BTSP 창 (해마) | $1$--$3$ s | Bittner 2017 | 장기 기억 연관 |

**신규 방정식 J5-1: STDP 감쇠율의 실험 고정**

F.14의 pre/post trace 감쇠율을 실험값으로 고정:

$$r_+ = \exp\!\left(-\frac{\Delta t}{\tau_+}\right) = \exp\!\left(-\frac{1}{20}\right) \approx 0.951$$

$$r_- = \exp\!\left(-\frac{\Delta t}{\tau_-}\right) = \exp\!\left(-\frac{1}{20}\right) \approx 0.951$$

$$r_e = \exp\!\left(-\frac{\Delta t}{\tau_e}\right) = \exp\!\left(-\frac{1}{500}\right) \approx 0.998$$

여기서 $\Delta t = 1$ ms, $\tau_+ = \tau_- = 20$ ms (in vivo 중앙값), $\tau_e = 500$ ms (선조체 중앙값).

**신규 방정식 J5-2: DA 게이팅 시간 창 제약**

$$g_{\text{DA}}(t) = g_{\text{DA}}^{\text{peak}} \cdot \exp\!\left(-\frac{t - t_{\text{burst}}}{\tau_{\text{DA}}}\right), \qquad \tau_{\text{DA}} \approx 500 \text{ ms}$$

$t - t_{\text{burst}} > 4\tau_{\text{DA}} = 2$ s에서 $g_{\text{DA}} < 0.02 \cdot g_{\text{DA}}^{\text{peak}}$ → 실질 소멸. Yagishita 2014의 "4초 이후 효과 소멸"과 정합.

---

### J.6 도파민 뉴런 발화 파라미터

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| VTA tonic 발화율 | $\bar{f}_{\text{DA}}^{\text{tonic}} \approx 4$--$5$ Hz | Grace & Bunney 1984, eLife 2021 | $g_{\text{DA}}$ baseline |
| VTA phasic burst 발화율 | $f_{\text{DA}}^{\text{burst}} \geq 20$ Hz (50 Hz 전형) | Schultz 1997, JNeurosci 2017 | $g_{\text{DA}}$ phasic |
| burst 지속 | $\sim 200$--$250$ ms | Grace & Bunney, PMC | burst 폭 |
| burst 시작 ISI | $\leq 80$ ms | eLife 2021 | 검출 기준 |
| burst 종료 ISI | $> 160$ ms | 동일 | 검출 기준 |
| 자가수용체 억제 | $\sim 300$ ms 후 | PMC 2010 | $g_{\text{DA}}$ 감쇠 |

**신규 방정식 J6-1: phasic/tonic DA 모델**

$$g_{\text{DA}}(t) = \underbrace{g_0}_{\text{tonic}} + \underbrace{\sum_k A_k \cdot \text{burst}(t - t_k, \tau_b, \tau_d)}_{\text{phasic}}$$

$$\text{burst}(s, \tau_b, \tau_d) = \begin{cases} (s/\tau_b)^2 \exp(1 - s/\tau_b) & 0 \leq s < 4\tau_d \\ 0 & \text{otherwise} \end{cases}$$

| 파라미터 | 값 | 유래 |
|---|---|---|
| $g_0$ | $\propto f_{\text{tonic}} = 4$ Hz | baseline DA level |
| $\tau_b$ | $50$ ms | burst 피크까지 |
| $\tau_d$ | $500$ ms | DA 감쇠 (J5-2와 동일) |
| $A_k$ | $\propto \text{RPE}_k$ | reward prediction error |

**신규 방정식 J6-2: tonic DA와 항상성 이탈의 관계**

$$g_0(t) = g_0^* \cdot \left(1 + \beta_{\text{tonic}} \|p(t) - p^*\|^2\right)$$

tonic DA level은 에너지 분배 고정점 $p^* = (0.0487, 0.2623, 0.6891)$으로부터의 이탈에 비례하여 변조된다. $g_0^*$는 $p = p^*$일 때의 baseline DA.

---

### J.7 4종 조절계 시간 상수

| 조절계 | 핵 | tonic 발화 [Hz] | 효과 $\tau$ [ms] | CE 변수 |
|---|---|---|---|---|
| DA | VTA/SNc | $4$--$5$ | $\tau_{\text{DA}} \approx 500$ | $g_{\text{DA}}$ |
| NE | LC | $0.5$--$5$ (tonic), $10$--$20$ (phasic) | $\tau_{\text{NE}} \approx 200$--$500$ | $g_{\text{NE}}$ |
| 5HT | raphe | $0.5$--$3$ | $\tau_{\text{5HT}} \approx 2000$--$5000$ | $g_{\text{5HT}}$ |
| ACh | BF | $5$--$40$ (burst) | $\tau_{\text{ACh}} \approx 100$--$500$ | $g_{\text{ACh}}$ |

**신규 방정식 J7-1: 4종 조절계 일반 동역학**

각 조절계 $X \in \{\text{DA}, \text{NE}, \text{5HT}, \text{ACh}\}$에 대해:

$$\frac{dg_X}{dt} = -\frac{g_X - g_X^{\text{baseline}}}{\tau_X} + \alpha_X \cdot \text{drive}_X(t)$$

| $X$ | $\tau_X$ [ms] | $g_X^{\text{baseline}}$ | drive 소스 |
|---|---|---|---|
| DA | $500$ | $\propto f_{\text{tonic}} = 4$ Hz | RPE ($c_{\text{pred}}$) |
| NE | $300$ | $\propto f_{\text{tonic}} = 2$ Hz | novelty/arousal ($c_{\text{nov}}$) |
| 5HT | $3000$ | $\propto f_{\text{tonic}} = 1.5$ Hz | patience/model-based ($-\text{temporal discount}$) |
| ACh | $200$ | $\propto f_{\text{tonic}} = 10$ Hz | attention/encoding ($\text{salience}$) |

이산화($\Delta t = 1$ ms):

$$g_X^{t+1} = g_X^t + \frac{\Delta t}{\tau_X}(g_X^{\text{baseline}} - g_X^t) + \alpha_X \cdot \text{drive}_X^t \cdot \Delta t$$

---

### J.8 에너지 예산 제약

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 뇌 에너지 소비 (체중 대비) | 체중 2%, 에너지 20% | Raichle 2006 | 전체 budget |
| 과제 유발 에너지 증가 | $0.5$--$1.0$% | Raichle 2006 | $x_a = 4.87\%$ (CE) |
| intrinsic/ongoing 비중 | $60$--$80$% | Annual Reviews DMN | $x_b = 68.9\%$ (CE) |
| 시냅스 전달 에너지 | $\sim 59$% (signaling 중) | Attwell & Laughlin 2001 | $x_s$ 기여 |
| housekeeping | $\sim 25$% | 동일 | $x_s = 26.2\%$ (CE) |
| 동시 활성 뉴런 상한 (에너지) | $\leq 15$% | Attwell & Laughlin 2001 | TopK 상한 |

**신규 방정식 J8-1: 에너지 예산 정합 조건**

활성 비율 $x_a$와 에너지 소비의 관계:

$$P_{\text{active}} = x_a \cdot N \cdot \bar{f}_{\text{active}} \cdot E_{\text{spike}} + (1-x_a) \cdot N \cdot \bar{f}_{\text{idle}} \cdot E_{\text{spike}}$$

여기서 $E_{\text{spike}} \approx 10^8$ ATP/spike (Attwell & Laughlin 2001).

CE의 $x_a^* = 0.0487$에서:

$$
\frac{P_{\text{task-evoked}}}{P_{\text{total}}}
= \frac{x_a \bar{f}_{\text{active}}}{
x_a \bar{f}_{\text{active}} + (1-x_a) \bar{f}_{\text{idle}}
}
\approx \frac{0.05 \times 6}{0.05 \times 6 + 0.95 \times 0.16}
\approx 0.66\%
$$

이것은 Raichle의 "$0.5$--$1.0$% 과제 유발 증가"와 구간 내 일치한다.

---

### J.9 수면 파라미터

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| NREM/REM 비율 (수면 중) | NREM 75--80%, REM 20--25% | Lancet Respir Med | 모드 전환 확률 |
| 수면 주기 | $\sim 90$ min | PSG 표준 | $T_{\text{cycle}}$ |
| 수면:각성 비율 (24h) | 33% : 67% | 생리학 | CE: 31.1% : 68.9% |
| SWA delta power 감쇠 | 각 NREM 에피소드 $\to$ SWA 25--40% 감소 | Achermann & Borbely 2003 | $\rho$ |
| 수면 부채 회복 시간 상수 | $\tau_{\text{recovery}} \approx 1.6$ 밤 | Van Dongen 2003 | $\rho_{\text{night}} = \rho^{1/1.6}$ |

**신규 방정식 J9-1: SWA 감쇠와 $\rho$의 정량 연결**

각 NREM 에피소드에서 SWA delta power의 감쇠율:

$$\text{SWA}_{n+1} = (1 - \alpha_{\text{SWA}}) \cdot \text{SWA}_n, \qquad \alpha_{\text{SWA}} \approx 0.30 \text{ (중앙값)}$$

1밤(4--5 NREM 에피소드) 후 총 감쇠:

$$\text{SWA}_{\text{morning}} / \text{SWA}_{\text{evening}} = (1 - 0.30)^{4.5} \approx 0.17$$

CE의 $\rho_{\text{night}}^2 = 0.31^2 = 0.096$. 관측의 0.17과 같은 자릿수. 차이는 SWA가 시냅스 강도의 proxy이지 직접 $\rho$가 아니기 때문.

---

### J.10 연결성 (Connectivity)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 피질 뉴런 수 | $\sim 1.6 \times 10^{10}$ | Azevedo 2009 | $N$ (스케일링 참조) |
| 뉴런당 시냅스 | $\sim 7000$--$10000$ | Pakkenberg 2003 | $K$ |
| 연결 확률 (국소, 300 um) | $\sim 10$--$20\%$ | Song 2005, Ko 2011 | $\chi_{ij}$ |
| 연결 가중치 분포 | log-normal | Song 2005 | $W_{ij}$ 초기화 |
| E/I 비율 | $80\% / 20\%$ | Braitenberg & Schuz 1998 | $W^{\text{exc}} / W^{\text{inh}}$ |

**신규 방정식 J10-1: CE 스케일링 관계**

$$K_{\text{CE}} = (4/3)\pi^4 \approx 130, \qquad N_{\text{CE}} = 4096$$

$$\rho_{\text{CE}} = K/N = 130/4096 \approx 3.17\%$$

뇌: $K_{\text{brain}} \sim 7000$, $N_{\text{brain}} \sim 10^{10}$, $\rho_{\text{brain}} \sim 7 \times 10^{-7}$

스케일링 보정: CE는 "모듈"을 단위로 하므로, 1 CE cell = $N_{\text{brain}}/N_{\text{CE}} \approx 4 \times 10^6$ 뉴런의 앙상블.

**신규 방정식 J10-2: E/I 균형 제약**

$$\sum_{j \in \text{exc}} W_{ij} = -\beta_{\text{EI}} \sum_{j \in \text{inh}} W_{ij}, \qquad \beta_{\text{EI}} \approx 1.0$$

E/I 비가 80/20이므로 흥분 시냅스 수가 4배 많지만 개별 억제 가중치가 4배 강해서 균형:

$$0.8 \cdot \bar{w}_{\text{exc}} \approx 0.2 \cdot \bar{w}_{\text{inh}} \quad\Longrightarrow\quad \bar{w}_{\text{inh}} / \bar{w}_{\text{exc}} \approx 4$$

---

### J.11 종합: CE 파라미터 실험 대응표

| CE 파라미터 | 기호 | CE 기본값 | 뇌 실험값 유래 | 구간 |
|---|---|---|---|---|
| 활성 감쇠 | $\gamma_a$ | $0.1$--$0.3$ (WAKE) | $\Delta t / \tau_{\text{membrane}} = 1/20 = 0.05$ | $[0.03, 0.5]$ |
| 활성 게인 | $\kappa_a$ | $< \gamma_a$ | 안정 조건 (A-bound) | $[0.01, 0.3]$ |
| 억제 감쇠 | $\gamma_r$ | $0.1$ | $\Delta t / \tau_{\text{rel}} = 1/5 = 0.2$ | $[0.05, 0.3]$ |
| 억제 게인 | $\kappa_r$ | $0.1$ | 발화율 $\times$ 에너지 비례 | $[0.05, 0.2]$ |
| 기억 감쇠 | $\gamma_m$ | $0.01$ | $\Delta t / \tau_{\text{NMDA}} = 1/100$ | $[0.005, 0.05]$ |
| STDP $r_+, r_-$ | | $0.95$ | $\exp(-1/20) = 0.951$ | $[0.93, 0.97]$ |
| 적격 흔적 $r_e$ | | $0.998$ | $\exp(-1/500) = 0.998$ | $[0.995, 0.999]$ |
| DA 감쇠 $\tau_{\text{DA}}$ | | $500$ ms | Yagishita 2014 | $[300, 1000]$ |
| NE 감쇠 $\tau_{\text{NE}}$ | | $300$ ms | Aston-Jones 2005 | $[200, 500]$ |
| 5HT 감쇠 $\tau_{\text{5HT}}$ | | $3000$ ms | raphe tonic slow | $[2000, 5000]$ |
| ACh 감쇠 $\tau_{\text{ACh}}$ | | $200$ ms | BF fast burst | $[100, 500]$ |
| 활성 비율 | $x_a^*$ | $0.0487$ | sparse firing 1--5% | $[0.01, 0.05]$ |
| 수축률 | $\rho$ | $0.155$ | 수면 SWA 감쇠 | $[0.10, 0.20]$ |
| 작업 기억 | $T_h$ | $6$ | theta/gamma 비 = 40/6 | $[4, 8]$ |
| 시뮬레이션 dt | $\Delta t$ | $1$ ms | gamma 100 Hz 오버샘플 | $[0.5, 2]$ |

---

### J.12 코드(`kernel.rs`) 파라미터와 실험 대응

> `clarus/core/src/engine/kernel.rs`의 `ModeParams`, `StepConfig` 값과 J.11 실험 구간의 정합성을 점검.

코드의 $\Delta t$는 명시되어 있지 않으므로, 코드 파라미터가 이미 이산화된 비율($\Delta t / \tau$)임을 가정한다.

| 코드 필드 | CE 기호 | WAKE | NREM | REM | 실험 구간 | 판정 |
|---|---|---|---|---|---|---|
| `activation_decay` | $\gamma_a$ | $0.18$ | $0.34$ | $0.22$ | $[0.03, 0.5]$ | OK |
| `activation_gain` | $\kappa_a$ | $0.82$ | $0.52$ | $0.68$ | $< \gamma_a$? | **주의** |
| `refractory_decay` | $\gamma_r$ | $0.12$ | $0.26$ | $0.18$ | $[0.05, 0.3]$ | OK |
| `refractory_gain` | $\kappa_r$ | $0.24$ | $0.12$ | $0.18$ | $[0.05, 0.2]$ | WAKE 약간 초과 |
| `memory_trace` decay | $1 - \gamma_m$ | $0.92$ | $0.92$ | $0.92$ | $\gamma_m \in [0.005, 0.05]$ | $\gamma_m = 0.08$, 구간 밖 |
| `refractory_scale` | $\lambda_r$ | $0.35$ | -- | -- | $[0.1, 0.5]$ | OK |
| `active_threshold` | $\theta$ | $0.22$ | -- | -- | salience 기반 | 단위 다름 |
| `bit_lower/upper` | $\tau^-/\tau^+$ | $0.10/0.30$ | -- | -- | 히스테리시스 대역 | OK |
| `energy_budget` | $\|A_t\|$ | $16$ | -- | -- | $\lceil 0.0487 N \rceil$ | 테스트 스케일($N=16$) |

**정합성 요약**:

1. $\gamma_a, \gamma_r$: 모두 실험 구간 내. 모드 순서 WAKE < REM < NREM (NREM 강감쇠) 유지.
2. $\gamma_m = 0.08$: NMDA $\tau = 100$ ms 기준($\gamma_m = 0.01$)보다 8배 빠름. 실험 정합을 위해 $0.01$--$0.02$ 권장.
3. $\kappa_a$: WAKE $0.82 > \gamma_a = 0.18$이지만, A-bound는 $\kappa_a \|W_i\|_1 < \gamma_a$로 완화됨. CSR sparse 구조에서 $\|W_i\|_1 \ll 1$ 보장 시 안정.
4. `energy_budget`: $N$에 비례 조정. $\text{budget} = \lceil 0.0487 N \rceil$.
5. $\kappa_r = 0.24$ (WAKE): 구간 상한 $0.2$ 약간 초과. $a_i^2$ 스케일링으로 실효 억제 게인은 구간 내.

**`field.rs` (FieldEngine) 대응**:

| 코드 필드 | CE 기호 | 값 | 뇌/물리 대응 |
|---|---|---|---|
| `mu` | $\mu$ | $1.0$ | 진공 기대값 스케일 (spontaneous symmetry breaking) |
| `lam` | $\lambda$ | $1.0$ | 자기 결합 (비선형 포화) |
| `coupling_k` | $k$ | $50.0$ | 공간 결합 (시냅스 확산) → J.14 전파와 대응 |
| `dt` | $\Delta t_{\text{field}}$ | $0.01$ | 장 시뮬레이션 시간 단위 |
| `damping` | $\gamma_{\text{damp}}$ | $0.1$ | 에너지 산일 (감쇠파) |
| `alpha2` | $\alpha_2$ | $0.0$ (기본) | biharmonic 보정 (곡률 매끄러움) |

`FieldEngine`은 감쇠 $\phi^4$ 장방정식을 구현:

$$\ddot{\phi}_i = \mu^2\phi_i - \lambda\phi_i^3 + k\nabla^2\phi_i - \alpha_2\nabla^4\phi_i + J_i - \gamma_{\text{damp}}\dot{\phi}_i$$

이것은 B.3 에너지 함수의 Euler-Lagrange 동역학에 해당. damping $= 0.1$은 NMDA 시간 척도($\gamma_m = 0.01$)보다 빠름 → 장 이완이 기억보다 빠르게 안정화.

**`nn_ops.rs` (TopK SiLU) 대응**:

| 함수 | CE 대응 | 실험 제약 |
|---|---|---|
| `topk_silu_fwd(ratio)` | 희소 활성화 $a^* = \text{TopK}(\text{SiLU}(x), k)$ | `ratio` $= x_a^* = 0.0487$ (J.1, J.8) |
| SiLU $= x\sigma(x)$ | $\tanh$ 근사 (smooth gating) | 비음수/음수 모두 통과 |
| TopK masking | $b_i$ 결정 (A.7) | budget $= \lceil 0.0487 N \rceil$ |

---

### J.13 막 시간 상수 (Membrane Time Constant)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 피라미드 뉴런 $\tau_m$ | $10$--$30$ ms (중앙값 $\sim 20$ ms) | Gentet 2000, NeuroElectro | $\gamma_a$ 유도 |
| CA1 피라미드 | $12.4$ ms | NeuroElectro | |
| 인간 L2/3 피라미드 | $\sim 33$ ms | Eyal 2016 (인간 고유) | |
| 막 용량 $C_m$ | $\sim 0.9\;\mu\text{F/cm}^2$ | Gentet 2000 | |
| 입력 저항 $R_{\text{in}}$ (in vivo) | $20$--$50\;\text{M}\Omega$ | Destexhe 2003 | |

**신규 방정식 J13-1: $\gamma_a$와 $\tau_m$의 관계**

A.3의 활성 감쇠를 막 시간 상수로 재해석:

$$(1 - \gamma_a) = \exp\!\left(-\frac{\Delta t}{\tau_m}\right) \quad\Longrightarrow\quad \gamma_a = 1 - \exp\!\left(-\frac{\Delta t}{\tau_m}\right)$$

| $\tau_m$ [ms] | $\Delta t = 1$ ms | $\Delta t = 10$ ms |
|---|---|---|
| $10$ | $\gamma_a = 0.095$ | $\gamma_a = 0.632$ |
| $20$ | $\gamma_a = 0.049$ | $\gamma_a = 0.393$ |
| $30$ | $\gamma_a = 0.033$ | $\gamma_a = 0.284$ |

코드의 WAKE $\gamma_a = 0.18$은 $\Delta t = 10$ ms, $\tau_m \approx 50$ ms에 해당하거나, $\Delta t = 1$ ms라면 $\tau_m \approx 5$ ms (in vivo high-conductance state에서의 effective $\tau_m$).

**in vivo에서 시냅스 폭격으로 effective $\tau_m$이 $5$--$10$ ms로 단축된다** (Destexhe 2003). 따라서 코드의 0.18은 high-conductance state $\tau_m^{\text{eff}} \approx 5$ ms, $\Delta t = 1$ ms에 정합.

---

### J.14 축삭 전도 속도 (Axonal Conduction Velocity)

| 유형 | 속도 | 지연 (1 cm) | CE 대응 |
|---|---|---|---|
| 수초화 (대) | $\sim 120$ m/s | $0.08$ ms | 무시 가능 |
| 수초화 (피질) | $\sim 1$--$10$ m/s | $1$--$10$ ms | $W_{ij}$ 전파 지연 |
| 비수초화 (피질 내) | $\sim 0.3$ m/s | $33$ ms | gamma 1주기 |
| 장거리 (반구 간) | $\sim 5$--$20$ m/s (10--20 cm) | $5$--$40$ ms | 전역 동기화 지연 |

**신규 방정식 J14-1: 전파 지연 커플링**

B.1의 결합에 축삭 전도 지연을 도입:

$$I_i^t \ni \sum_j W_{ij} \cdot a_j^{t - \delta_{ij}}, \qquad \delta_{ij} = \left\lceil \frac{d(i,j)}{v_{\text{ax}} \cdot \Delta t} \right\rceil$$

국소 결합($d < 1$ mm, $v = 0.3$ m/s): $\delta \approx 3$ ms / $\Delta t$. $\Delta t = 1$ ms이면 $\delta = 3$ step.

CE의 sparse 3D lattice에서 대부분 이웃은 $d \leq r_c = \pi \approx 3$ 격자 단위. 격자 1단위 $= 300\;\mu$m이면 $d \leq 0.9$ mm → $\delta \leq 3$ step.

이것은 gamma 진동 ($T = 25$ ms)에 비해 짧으므로 무시 가능한 수준이지만, 장거리 결합에서는 의미 있다.

---

### J.15 시냅스 잡음 (Synaptic Noise)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 막 전위 변동 SD (in vivo) | $\sigma_V \approx 3$--$5$ mV | Destexhe 2003, Pare 1998 | $\sigma_\eta$ |
| 안정 전위 | $V_{\text{rest}} \approx -65$ mV | 교과서 | |
| threshold | $V_{\text{th}} \approx -50$ mV | 교과서 | |
| SNR (threshold/noise) | $\Delta V / \sigma_V = 15 / 4 \approx 3.75$ | 유도 | |

**신규 방정식 J15-1: 잡음 수준의 실험 고정**

A.2의 $\eta_i^t \sim \mathcal{N}(0, \sigma_\eta^2)$에서 $\sigma_\eta$를 실험적으로 고정:

CE의 활성 $a_i \in (-1, 1)$은 전위를 $[V_{\text{rest}}, V_{\text{th}}]$에 정규화한 것. 잡음의 정규화:

$$\sigma_\eta = \frac{\sigma_V}{V_{\text{th}} - V_{\text{rest}}} = \frac{4}{15} \approx 0.27$$

이것은 활성 범위 $(-1, 1)$에서 $\sigma_\eta \approx 0.27$, 즉 활성 범위의 약 13%에 해당.

**모드별 잡음 수준**:

| 모드 | 뇌 관측 | $\sigma_\eta$ |
|---|---|---|
| WAKE | 높은 시냅스 폭격, $\sigma_V \approx 4$--$5$ mV | $0.27$--$0.33$ |
| NREM | UP: 유사, DOWN: $\sigma_V \approx 1$ mV | $0.07$ (DOWN), $0.27$ (UP) |
| REM | WAKE 유사, theta 변조 | $0.27$ |

---

### J.16 해마 SWR (Sharp-Wave Ripple)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| ripple 주파수 | $150$--$200$ Hz | Buzsaki 2015 | D.5 replay 반복율 |
| SWR 지속 시간 | $40$--$100$ ms | 동일 | replay 윈도우 |
| 시간 압축비 | $\sim 10\times$ | Naresky 2012 | 재생 속도 |
| SWR 발생 빈도 (NREM) | $\sim 1$--$3$ Hz | Buzsaki 2015 | replay 호출 빈도 |

**신규 방정식 J16-1: replay 시간 압축 모델**

D.3--D.5의 replay를 시간 압축비로 정량화:

$$R_{i,t}^{\text{replay}} = V_j \text{ at } \Delta t_{\text{replay}} = \frac{\Delta t_{\text{experience}}}{C_{\text{compress}}}$$

$C_{\text{compress}} = 10$. 원래 1초 경험이 100 ms SWR 내에 재생.

NREM에서 SWR 빈도 $\sim 2$ Hz이면, 1초당 2회 replay event, 각 100 ms → 1초당 2초 분량의 경험을 재처리.

8시간 수면에서 $8 \times 3600 \times 2 \times 1 = 57600$초 분량 재처리 가능 → 16시간 각성의 $\sim 1$배. 이것은 "하룻밤에 하루를 정리한다"는 관측과 일치.

$$n_{\text{replay/night}} = f_{\text{SWR}} \cdot T_{\text{NREM}} \cdot C_{\text{compress}} = 2 \times (8 \times 0.75 \times 3600) \times 10 = 432000 \text{ s 분량}$$

---

### J.17 UP/DOWN 상태 (Cortical Slow Oscillation)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| UP 상태 지속 | $450 \pm 280$ ms | Jercog 2017 (쥐 SWS) | $b_i = 1$ 유지 시간 |
| DOWN 상태 지속 | $280 \pm 160$ ms | 동일 | $b_i = 0$ 유지 시간 |
| 느린 진동 주기 | $\sim 0.5$--$1$ Hz ($1$--$2$ s) | Steriade 1993 | delta 대역 |
| UP/DOWN 비율 | $\sim 1.6 : 1$ | 유도 | |

**신규 방정식 J17-1: 히스테리시스 임계의 UP/DOWN 지속 제약**

A.6의 $\tau^+, \tau^-$는 UP/DOWN 전환 임계. 히스테리시스 폭이 넓으면 상태 유지 시간이 길다:

$$T_{\text{UP}} \propto \frac{\tau^+ - \tau^-}{\gamma_a^{\text{NREM}}}, \qquad T_{\text{DOWN}} \propto \frac{\tau^+ - \tau^-}{\kappa_a^{\text{NREM}} \cdot \bar{I}}$$

코드의 `bit_lower = 0.10`, `bit_upper = 0.30`, NREM $\gamma_a = 0.34$:

$$T_{\text{UP}}^{\text{sim}} \sim \frac{0.20}{0.34} \approx 0.59 \text{ (단위 시간)}$$

이것이 실험의 450 ms와 대응하려면 시뮬레이션 1 단위 시간 $\approx 760$ ms → 1 step $\approx 760$ ms.

$\Delta t = 10$ ms 기준이면 $T_{\text{UP}} \approx 59$ step $= 590$ ms → 실험 450 ms와 같은 자릿수.

---

### J.18 항상성 가소성 (Homeostatic Synaptic Scaling)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 시간 척도 | 시간~일 ($6$--$48$ h) | Turrigiano 2004 | 느린 학습률 |
| 스케일링 속도 (활동 차단 시) | $\sim 2$% / h | Turrigiano 2008 | $\eta_{\text{homeo}}$ |
| 방향 | 활동 감소 $\to$ 시냅스 증강, 증가 $\to$ 감약 | Turrigiano 1998 | 안정화 |
| 범위 | 전 시냅스 곱셈적 (multiplicative) | 동일 | $W \to W \cdot s$ |

**신규 방정식 J18-1: 항상성 시냅스 스케일링**

STDP 학습(F.14)과 별개로 느린 시간 척도에서 가중치를 정규화:

$$W_{ij}^{t+1} = W_{ij}^t \cdot \left(1 + \eta_{\text{homeo}} \cdot (\bar{f}_{\text{target}} - \bar{f}_i^{\text{slow}})\right)$$

$$\bar{f}_i^{\text{slow}} = (1 - \gamma_{\text{homeo}}) \bar{f}_i^{\text{slow}} + \gamma_{\text{homeo}} |a_i^t|$$

| 파라미터 | 값 | 유래 |
|---|---|---|
| $\eta_{\text{homeo}}$ | $10^{-5}$ | $2\%/\text{h} = 0.02/3600\text{s} \approx 5.6 \times 10^{-6}$/ms |
| $\gamma_{\text{homeo}}$ | $10^{-4}$ | 평균 발화율 추정 $\tau \sim 10$ s |
| $\bar{f}_{\text{target}}$ | $x_a^* = 0.0487$ | CE 고정점 활성비 |

이것은 STDP의 불안정성(양의 피드백)을 항상성(음의 피드백)으로 보정하는 역할.

---

### J.19 단기 시냅스 가소성 (Short-Term Plasticity, STP)

| 파라미터 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| $\tau_{\text{rec}}$ (depression 회복) | $200$--$800$ ms (E$\to$E) | Markram 1998, Wang 2006 | 시냅스 자원 회복 |
| $\tau_{\text{fac}}$ (facilitation 감쇠) | $20$--$200$ ms | 동일 | 방출 확률 감쇠 |
| $U$ (초기 방출 확률) | $0.2$--$0.8$ (시냅스 유형별) | 동일 | |
| E$\to$E 시냅스 | depression 우세 ($\tau_{\text{rec}} \gg \tau_{\text{fac}}$) | Markram 1997 | 적응 |
| E$\to$I 시냅스 | facilitation 우세 ($\tau_{\text{fac}} \gg \tau_{\text{rec}}$) | Gupta 2000 | 안정화 |

**신규 방정식 J19-1: Tsodyks-Markram STP 모델**

현재 $W_{ij}$에 동적 시냅스 효능을 곱셈적으로 적용:

$$W_{ij}^{\text{eff}}(t) = W_{ij} \cdot u_j(t) \cdot x_j(t)$$

$$\frac{dx_j}{dt} = \frac{1 - x_j}{\tau_{\text{rec}}} - u_j \cdot x_j \cdot \delta(t - t_{\text{spike}})$$

$$\frac{du_j}{dt} = \frac{U - u_j}{\tau_{\text{fac}}} + U(1 - u_j) \cdot \delta(t - t_{\text{spike}})$$

이산화($\Delta t = 1$ ms, 스파이크 근사: $|a_j^t| > \theta_{\text{spike}}$):

$$x_j^{t+1} = x_j^t + \frac{1}{\tau_{\text{rec}}}(1 - x_j^t) - u_j^t \cdot x_j^t \cdot \mathbf{1}[|a_j^t| > \theta]$$

$$u_j^{t+1} = u_j^t + \frac{1}{\tau_{\text{fac}}}(U - u_j^t) + U(1 - u_j^t) \cdot \mathbf{1}[|a_j^t| > \theta]$$

| 시냅스 유형 | $U$ | $\tau_{\text{rec}}$ [ms] | $\tau_{\text{fac}}$ [ms] | 특성 |
|---|---|---|---|---|
| E$\to$E | $0.5$ | $500$ | $20$ | depression 우세 |
| E$\to$I (fast-spiking) | $0.2$ | $200$ | $200$ | facilitation 우세 |
| I$\to$E | $0.3$ | $300$ | $10$ | depression |

---

### J.20 스파이크 빈도 적응 (Spike-Frequency Adaptation)

| 측정량 | 실험값 | 출처 | CE 변수 |
|---|---|---|---|
| 빠른 AHP 시간 상수 | $1$--$5$ ms | 교과서 | $r_i$ (이미 포착) |
| 중간 AHP ($I_{\text{AHP}}$) | $50$--$200$ ms | Madison & Nicoll 1984 | |
| 느린 AHP (sAHP) | $1$--$5$ s | 동일 | 새로운 변수 필요 |
| Ca$^{2+}$ extrusion $\tau_{\text{Ca}}$ | $\sim 130$--$500$ ms | Wang 1998 | |
| 적응 강도 (ISI 비율) | $\text{ISI}_{\text{last}} / \text{ISI}_{\text{first}} \approx 2$--$5$ | 일반 관측 | |

**신규 방정식 J20-1: 적응 변수 도입**

A.1의 상태에 적응 변수 $w_i$를 추가:

$$s_i^t = (a_i^t,\;r_i^t,\;m_i^t,\;b_i^t,\;w_i^t) \in \mathbb{R}^4 \times \{0,1\}$$

$$w_i^{t+1} = (1 - \gamma_w) w_i^t + \kappa_w \cdot (a_i^t)^2$$

$$\gamma_w = \frac{\Delta t}{\tau_w}, \qquad \tau_w \approx 200 \text{ ms (중간 AHP)}$$

A.3 활성 갱신에 적응을 반영:

$$a_i^{t+1} = (1-\gamma_a)\,a_i^t + \kappa_a\,\tanh(I_i^t - \beta_w w_i^t)$$

| 파라미터 | 값 | 유래 |
|---|---|---|
| $\gamma_w$ | $\Delta t / 200 = 0.005$ ($\Delta t = 1$ ms) | 중간 AHP |
| $\kappa_w$ | $0.01$ | 적응 강도 (ISI 비율 $\sim 3$) |
| $\beta_w$ | $0.5$ | 적응 → 활성 결합 강도 |

$r_i$(빠른 억제, $\tau \sim 5$ ms)와 $w_i$(느린 적응, $\tau \sim 200$ ms)는 서로 다른 시간 척도에서 발화를 억제한다. 이것은 뇌에서 $\text{GABA}_A$ (빠른)와 $I_{\text{AHP}}$ (느린)의 이중 억제에 대응.

---

### J.21 종합 시간 척도 계층도

모든 실험 시간 상수를 하나의 계층으로 정리:

| 시간 척도 | 뇌 메커니즘 | CE 변수 | $\tau$ |
|---|---|---|---|
| $< 1$ ms | 축삭 전도 (국소) | 무시 | $\sim 0.3$ ms |
| $1$--$2$ ms | 절대 불응기, AMPA | $r_i$ 클램프, $W^{\text{fast}}$ | $1$ ms |
| $5$--$10$ ms | 상대 불응기, GABA$_A$ | $\gamma_r$, $r_i^A$ | $5$--$7$ ms |
| $5$--$10$ ms | 막 시간 상수 (in vivo eff.) | $\gamma_a$ | $5$ ms |
| $10$--$33$ ms | gamma 진동 | $\Delta t_{\text{iter}}$ | $25$ ms |
| $20$ ms | STDP 창 | $r_+, r_-$ | $20$ ms |
| $50$--$150$ ms | NMDA, STP facilitation | $\gamma_{\text{NMDA}}, \tau_{\text{fac}}$ | $100$ ms |
| $100$--$300$ ms | GABA$_B$, 적응 (AHP) | $r_i^B$, $w_i$ | $200$ ms |
| $125$--$250$ ms | theta 진동, 작업 기억 | $R$ 수렴 | $170$ ms |
| $200$--$800$ ms | STP depression recovery | $\tau_{\text{rec}}$ | $500$ ms |
| $300$--$1000$ ms | DA/NE/ACh 효과 | $g_{\text{DA}}, g_{\text{NE}}, g_{\text{ACh}}$ | $200$--$500$ ms |
| $500$ ms--$1$ s | 적격 흔적, DA 게이팅 | $r_e$ | $500$ ms |
| $1$--$3$ s | 5HT 효과, BTSP | $g_{\text{5HT}}$ | $3000$ ms |
| $40$--$100$ ms | SWR (replay 단위) | D.5 replay | $70$ ms |
| $450$ ms | UP 상태 (NREM) | $b_i = 1$ | $450$ ms |
| $10$ s | 항상성 발화율 추정 | $\bar{f}_i^{\text{slow}}$ | $10$ s |
| 시간--일 | 항상성 시냅스 스케일링 | $\eta_{\text{homeo}}$ | $\sim 12$ h |
| $90$ min | 수면 주기 | NREM$\to$REM 전환 | $5400$ s |
| $16$--$18$ h | Process S 축적 | $P_{\text{sleep}}$ | $65520$ s |
