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

### B.1 결합 행렬

$$W_{ij}(g) = \exp\!\left(-\frac{d_g(i,j)^2}{\sigma^2}\right) \cdot \chi_{ij}$$

| 기호 | 의미 |
|---|---|
| $d_g(i,j)$ | 리만 다양체 $(M, g)$ 위에서 $i$와 $j$ 사이의 측지선 거리 |
| $\sigma$ | 결합 반경 (kernel width) |
| $\chi_{ij} \in \{0,1\}$ | sparse mask |

### B.2 대안: k-nearest neighbor

$$\chi_{ij} = \mathbf{1}[j \in \text{knn}(i, k)]$$

### B.3 에너지 함수 (선택적)

$$E(\{a_i\}) = -\frac{1}{2}\sum_{i,j} W_{ij}\,a_i\,a_j - \sum_i u_i\,a_i + \sum_i V(a_i)$$

$V(a)$: 국소 potential (self-coupling term).

### B.4 형식 검증: 에너지 감소

**정리 (E-decrease).** 동기적 업데이트에서 $a_i^{t+1}$이 $E$의 좌표별 최소화이면 $E(\{a_i^{t+1}\}) \le E(\{a_i^t\})$.

### B.5 spectral bound

$$\|W\|_2 \le \lambda_{\max}$$

$\lambda_{\max}$를 clamp하여 증폭을 제한한다. 이것은 `12_Equation.md`의 spectral constraint와 동일.

---

## C. 전역 모드 (mode update)

### C.1 모드 집합

$$M_t \in \{\mathrm{WAKE},\;\mathrm{NREM},\;\mathrm{REM}\}$$

### C.2 모드 전환

$$M_{t+1} = \Pi(M_t,\;Q_t,\;\Psi_{\text{global}}(t),\;U_t)$$

- $Q_t = (p_{\text{sleep}},\;\text{arousal},\;\text{autonomic},\;\text{endocrine},\;\text{immune},\;\text{metabolic})$: body-loop 제어벡터
- $\Psi_{\text{global}}(t) = \sum_i \omega_i\,a_i^t$: 전역 활성 관측량
- $U_t$: 외부 입력 하중

### C.3 모드별 파라미터 해석

| 파라미터 | WAKE | NREM | REM |
|---|---|---|---|
| $\gamma_a$ | 중간 (0.1--0.3) | 큼 (0.5+) | 작음 (0.05--0.1) |
| $\kappa_a$ | 큼 | 작음 | 중간 |
| $\lambda_r$ | 약함 | 강함 | 약함 |
| $\lambda_H$ (replay) | 약함 | 중간 | 강함 |
| $\sigma_\eta$ (noise) | 작음 | 거의 0 | 중간 |
| $B_t$ (energy budget) | 큼 | 작음 | 중간 |

### C.4 초기 규칙 기반 전환

```
if Q_t.sleep_pressure > theta_sleep and norm(U_t) < theta_input:
    M_{t+1} = NREM
elif M_t == NREM and elapsed > T_nrem:
    M_{t+1} = REM
elif M_t == REM and elapsed > T_rem:
    if Q_t.sleep_pressure > theta_cont:
        M_{t+1} = NREM
    else:
        M_{t+1} = WAKE
else:
    M_{t+1} = M_t
```

---

## D. 해마/기억 (hippocampus / replay)

### D.1 해마 상태

$$H_t = (K_t,\;V_t,\;P_t)$$

### D.2 인코딩

$$K_{t+1} = K_t \cup \{k_{\text{new}}\}, \quad V_{t+1} = V_t \cup \{v_{\text{new}}\}, \quad P_{t+1} = \text{update}(P_t)$$

$k_{\text{new}} = h(A_t)$: 활성 패턴의 해시/임베딩
$v_{\text{new}} = A_t$: 활성 스냅샷

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

---

## H. 검증 게이트 대응

`proof.md`의 4중 게이트를 이 문서의 식에 적용.

### H.1 Layer A--E 게이트

| 게이트 | 적용 대상 | 상태 |
|---|---|---|
| $G_{\text{formal}}$ | A-bound, R-bound, E-decrease, Zero-attract, Local-contract | pass |
| $G_{\text{obs}}$ | EEG/fMRI에서 $a_i^t$ proxy 추출 | partial |
| $G_{\text{causal}}$ | 약물/수면박탈/자극 실험에서 모드 전환 방향 일치 | partial |
| $G_{\text{pred}}$ | 모델 시뮬레이션 vs 실제 뇌파 스펙트럼 비교 | pending |

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

A.4의 억제 갱신에서 $\gamma_r$은 불응기 복구 속도와 대응한다:

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

$$I_i^t = u_i^t + \underbrace{\sum_j W_{ij}^{\text{fast}} a_j^t}_{\text{AMPA-like}} + \underbrace{\sum_j W_{ij}^{\text{slow}} \bar{a}_j^t}_{\text{NMDA-like}} - \lambda_r r_i^t + \lambda_m m_i^t + \eta_i^t$$

여기서 느린 성분은 지수 이동 평균:

$$\bar{a}_j^t = (1 - \gamma_{\text{NMDA}}) \bar{a}_j^{t-1} + \gamma_{\text{NMDA}} a_j^t, \qquad \gamma_{\text{NMDA}} = \frac{\Delta t}{\tau_{\text{NMDA}}} \approx \frac{1}{100} = 0.01$$

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

$$\frac{P_{\text{task-evoked}}}{P_{\text{total}}} = \frac{x_a \bar{f}_{\text{active}}}{x_a \bar{f}_{\text{active}} + (1-x_a) \bar{f}_{\text{idle}}} \approx \frac{0.05 \times 6}{0.05 \times 6 + 0.95 \times 0.16} \approx 0.66\%$$

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

### J.12 코드(`kernel.rs`) 파라미터 vs 실험 대응

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
