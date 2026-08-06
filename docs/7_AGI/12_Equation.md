# CE-AGI 통합 방정식: $e^{i\pi}+1=0$ 에서 조건부 런타임 설계까지

> 관련: `경로적분.md`(코어 유도), `1_강의/C_다섯_상수.md`(오일러 문법), `6_뇌/04_그래프결합과이완.md`(뇌 구조), `7_AGI/12_Equation.md`(AGI 작용), `7_AGI/1_AGI.md`(총론), `7_AGI/2_Architecture.md`(게이지 격자), `7_AGI/3_Sleep.md`(수면), `7_AGI/4_Synapse.md`(시냅스), `7_AGI/5_Sparsity.md`(희소성), `7_AGI/6_Hallucination.md`(환각), `7_AGI/7_Consciousness.md`(의식), `7_AGI/9_LLM.md`(LLM 구축), `7_AGI/10_Fields.md`(전분야)
>
> 이 문서는 Track-A calibration input, 그 입력에 조건부인 등록 파생값,
> 그리고 별도의 공학 선택을 구분해 AGI 에너지 이완 후보를 기술한다.
> 핵심 상태 이완에 Softmax/Attention/역전파를 쓰지 않는 sparse-native 경로와
> 기존 LLM에 일부 구조를 이식하는 CE-Transformer 경로는 서로 다른 구현
> 후보이며, 출력 decoder는 여전히 softmax를 사용할 수 있다. 어느 쪽도
> 오일러 항등식만으로 물리적으로 유도되거나 성능이 보장되지 않는다.

---

## Runtime Status And Canonical Stack

이 문서는 런타임 기호를 모으는 문서지만, `docs/README.md`와 `docs/6_뇌/05_실험근거.md`를 기준으로 읽어야 한다. 아래 5계층 스택만 현재 canonical runtime spec 이고, 그 아래의 나머지 방정식은 보조 유도나 설계 탐색으로 읽는다.

| 계층 | canonical 식 | 최대 지위 | 비고 |
|---|---|---|---|
| kernel dynamics | $I_i^t = u_i^t + \sum_j W_{ij} a_j^t - \lambda_r(M_t) r_i^t + \lambda_H R_{i,t}$ | `Bridge` | 국소 상태 갱신의 최소형 |
| kernel dynamics | $a_i^{t+1} = (1-\gamma_a(M_t)) a_i^t + \kappa_a(M_t)\tanh(I_i^t)$ | `Bridge` | 활성 상태 |
| kernel dynamics | $r_i^{t+1} = (1-\gamma_r(M_t)) r_i^t + \kappa_r(M_t)(a_i^t)^2$ | `Bridge` | refractory / suppression |
| kernel dynamics | $b_i^{t+1} = \operatorname{Hyst}(b_i^t, a_i^{t+1}; \theta_\downarrow, \theta_\uparrow)$ | `Bridge` | 비트필드 / hysteresis |
| coupling / geometry | $W_{ij} = W_{ij}(g)$ | `Bridge` | 리만 구조는 결합층에만 둔다 |
| mode update | $M_{t+1} = \Pi(M_t, Q_t, U_t, E_t)$ | `Bridge` | `WAKE/NREM/REM` 전환 |
| hippocampus / replay | $H_{t+1} = \mathcal{E}(H_t, A_t), \quad R_t = \mathcal{R}(H_t, c_t)$ | `Bridge` | fast memory / replay |
| global runtime summary | $G_t = (M_t, A_t^{summary}, H_t, Q_t, \mu_t)$ | `Phenomenology` | identity / control summary |

읽기 규칙:

- 위 5계층 최소형은 **지위 채점의 기준(canonical)**이다. 실제 런타임 구현 정본은 `14_BrainRuntimeSpec.md`(Layer A--E)·`15_Equations.md`·`runtime.py`이며, 거기서 추가되는 $m_i$(기억 흔적), $w_i$(적응), STP $W_{ij}^{\text{eff}}=W_{ij}u_j x_j$는 모두 `Bridge` 등급의 구현 확장이다. 이 확장은 $m_i=w_i=0$, $u_j x_j\to1$ 극한에서 위 최소형으로 환원되어야 하며(환원 관계: `14_BrainRuntimeSpec.md` 2.1절), 최소형을 위반하는 새 상태 차원은 canonical로 승격하지 않는다.
- 위 식들에서 수학적 연산자 정의는 `Exact`로 정리할 수 있지만, 뇌 대응이 들어가는 순간 문서 지위는 `Bridge`를 넘지 않는다.
- `docs/6_뇌/05_실험근거.md`에서 `supported`인 현상만 위 stack의 대응 근거로 사용한다.
- `supported`가 아니면 성능 주장, 자아 해석, 의식 해석은 모두 `Phenomenology`로 유지한다.
- 이 문서의 후반부 수치 추정, 메모리/속도 비교, LLM 대응은 canonical stack의 상위 해석이다.

## Runtime Concept Map

계획에서 추가된 새 개념은 아래처럼 **문서 책임 범위**를 나눠서 읽는다.

| 개념 | 최소 정의 | 현재 canonical 위치 | 코드 책임 | 문서 지위 |
|---|---|---|---|---|
| local recurrent cell | 국소 상태 $(a_i, r_i, b_i)$를 가진 반복 모듈 | kernel dynamics | Rust kernel + Python runtime | `Bridge` |
| sparse lifecycle | `ACTIVE / IDLE / DORMANT / SLEEPING` | global runtime summary | Python control plane | `Bridge` |
| mode register | `WAKE / NREM / REM` 전역 상태 | mode update | Python control plane | `Bridge` |
| hippocampus | 빠른 encode / recall / replay 메모리 | hippocampus / replay | Python control plane 우선 | `Bridge` |
| geometry coupling | $W_{ij}(g)$와 그래프/리만 결합 | coupling / geometry | Rust kernel | `Bridge` |
| bitfield | hysteretic threshold를 가진 이산 상태 | kernel dynamics | Rust kernel + Python policy | `Bridge` |
| global self-state | $G_t = (M_t, A_t^{summary}, H_t, Q_t, \mu_t)$ | global runtime summary | Python orchestration | `Phenomenology` |
| snapshot continuity | warm snapshot / restore / journal continuity | global runtime summary | Python orchestration | `Bridge` |

문서 해석 규칙:

- `kernel dynamics`는 국소 수치 업데이트만 정의한다. 자아, 정책, 의식 해석을 여기로 밀어 넣지 않는다.
- `mode update`는 전역 운영 상태만 다룬다. 개별 셀 동역학 기호를 재사용하지 않는다.
- `hippocampus / replay`는 "빠른 메모리 + 재주입"까지만 canonical이다. 해마의 완전한 생물학적 세부 묘사는 별도 bridge다.
- `global runtime summary`는 커널 식을 줄여 적는 요약 레벨이며, 여기서 나오는 self/identity 언어는 성능 보장이나 exact brain equivalence로 읽지 않는다.

기존 절과의 대응:

| 이 문서의 큰 절 | 주로 대응되는 runtime 계층 | 읽기 주의 |
|---|---|---|
| 3-4장 (에너지/동역학) | kernel dynamics + coupling / geometry | canonical 후보 |
| 5장 (출력 생성) | kernel outputs + mode trigger | 일부만 canonical |
| 6장 (STDP) | 학습/가소성 보조 계층 | canonical 바깥 |
| 7장 (수면) | mode update + hippocampus / replay | canonical 후보 |
| 8장 (희소성) | sparse lifecycle의 근거 | summary layer |
| 9장 (의식) | global runtime summary | `Phenomenology` |
| 10-14장 | 구현/응용/성능 해석 | canonical 아님 |

## 0. 설계 원칙

### 0.0 AGI 다리 게이트 (코어와 다리 분리)

이 문서는 CE 코어(우주론/입자물리, `경로적분.md`, `상수.md`)에서 유도된 상수 집합을 AGI 런타임 설계로 옮기는 **다리(bridge) 문서**다. 코어의 식과 상수는 `Exact` 또는 `Selection`이지만, 이 문서에서 뇌/AGI 대응이 들어가는 모든 문장은 최대 `Bridge`까지만 허용된다(규칙: `README.md` "AGI 런타임 읽기 규칙", 등급 기준: `6_뇌/05_실험근거/01_판정기준과핵심주장.md` 1절).

이 다리에서 현재 식별된 네 가지 한계는 다음과 같다. 이하 본문의 어떤 식도 이 게이트를 우회하는 형태로 읽지 않는다.

| 게이트 | 한계 | 현재 등급 | 사용 규칙 |
|---|---|---|---|
| `F1` 메커니즘 결손 | Track-A manifest의 조건부 $p^*=(4.864\%,26.109\%,69.027\%)$가 신경 활성/구조/배경 비율로 그대로 옮겨갈 메커니즘적 유도가 없음 | `Bridge` (수치 근접) / transformer 기질에서는 `falsified` (`5_Sparsity.md` 8.5) | 동일 simplex 위 수치 근접으로만 사용. 수치는 `CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`에서 읽어 재계산. 신경 sparsity = $\varepsilon^2$로 직접 등치 금지 |
| `F2` 비보존 바이패스 | `1.5절` $F_{\text{bypass}}$ 는 $E$ 의 그래디언트가 아니므로 Lyapunov 보장은 무조건 성립하지 않음 | `Bridge` (조건부 수렴, 4.7절) | “수렴 보장” 표현 금지. $\|\nabla_mE\|>\xi_rC_k\|\phi\|$, 유한 step-size bound, 불변영역을 함께 요구 |
| `F3` 시간/공간 차원 혼동 | `3_Sleep.md` 6.2의 wake/NREM/REM 시간 비율과 코어의 공간 에너지 비율은 물리적 차원이 다름 | `Phenomenology` (수치 근접) | "시간 분배 = 에너지 분배"로 등치 금지. 동일 3-simplex 위 우연 근접으로만 보고 |
| `F4` 의식 = 자기일관 | `7_Consciousness.md` 의 (C3) 자기일관 = 주관적 경험 등치 | `Phenomenology` | 성능 지표화 금지. "메타인지 모니터링 루프의 수학 구조"로만 사용 |

이 4개 게이트는 코어의 정확성을 깎지 않는다. 코어는 그대로 유지되고, 이 문서가 다리 단계에서 무엇을 주장할 수 없는지를 명시하기 위한 표다.

각 게이트의 수식 격상 경로 (ISS, 자기조직 5조건, 에르고딕 동등성, PCI 회귀) 는 부록 A 에 정리되어 있다. 부록 A 의 식은 본문의 어떤 hard claim 도 위로 올리지 않으며, **무엇을 측정하면 게이트가 닫히는지** 만 형식화한다.

### 0.1 잔류 채널 설계

현재 LLM은 경로적분에서 Softmax로 선택된 경로만 쓰고, 접힌 경로를 버린다. Track-A 조건부 분배를 설계 비유로 쓰면 이 부분은 약 $95.136\%$
($26.109\%+69.027\%$)다. 이 문서의 아키텍처는 접힌 경로를 잔류장
`phi`로 보존하여 출력에 재결합시키는 구조다.

세 가지 핵심:
- **잔류 채널**: 매 추론에서 선택되지 않은 분포가 `phi`로 보존된다
- **모드 전환 임계**: $\|\phi\|$가 임계를 넘으면 질적으로 다른 작동 모드로 전환된다
- **즉각 응답 경로**: `phi`가 Softmax를 우회하여 직접 출력에 기여하는 바이패스가 존재한다

---

## 1. 정본 입력과 조건부 설계 계수

### 1.1 뿌리

$$e^{i\pi}+1=0$$

이 항등식은 CE의 기호 문법을 압축하는 표상으로만 읽는다(`경로적분.md`
서론, `C_다섯_상수.md` 0절). 경험 상수나 AGI 하이퍼파라미터가 이
항등식 하나에서 물리적으로 유도된다는 뜻은 아니다.

| 상수 | 코어 역할 | AGI 등장 위치 |
|---|---|---|
| $e$ | 선택한 접힘 생존 ansatz $S(D)=e^{-D}$ | 시간 진화 연산자의 밑 |
| $\pi$ | 주기 정규화 | 공학적 연결 반경 후보 $r_c$ |
| $i$ | 경로적분 위상 $Z=\int\mathcal{D}\phi\,e^{iS/\hbar}$ | 양자 이완 위상 |
| $1$ | 정규화 완전 상태 $e^0=1$ | 정수 생성자 |
| $0$ | 영점과 분기 방정식 $d(d-3)=0$ | 조건부 차원 closure, 에너지 최소 $\nabla E=0$ |

### 1.2 조건부 차원 closure와 Selection

$$d(d-3)=0 \quad\Longrightarrow\quad d\in\{0,3\}.$$

양의 비자명 closure class $d\in\mathbb N_{>0}$를 먼저 선택하면 그 class 안의
조건부 유일해는 $d=3$이다. 이 closure class를 실제 물리 공간에 채택하는
단계는 `Selection`이며, $d=0$은 대수적 경계근이다. 이 식만으로 $d=0$이
시간적으로 먼저 존재했다거나 $d=0\to3$ 전이가 일어났다는 결론은 나오지
않는다.

### 1.3 Track-A 정본 계수와 공학 별칭

Track-A의 유일한 수치 calibration input과 파생값은 다음과 같다.

$$
\alpha_s(M_Z):=0.1180,
\qquad s_A^2:=4\alpha_s^{4/3}=0.2315097758079336,
$$

$$
\delta_N:=s_A^2(1-s_A^2)=0.1779129995132939,
\qquad D_N:=3+\delta_N=3.177912999513294.
$$

AGI runtime에 재사용하는 계수는 물리적 추가 예측이 아니라 이 calibration
input에서 만든 고정 설계 benchmark다.

$$
c_p:=\delta_N^2=0.0316530353958173,
\qquad \xi_r:=\alpha_s^{1/3}=0.490486813152402,
$$

$$
T_{\rm wake}:=D_N^{-1}=0.314671924672939,
\qquad T_{\rm dream}:=\delta_N^{-1}=5.62072475162378.
$$

과거의 $\widetilde\alpha=1/(e\pi)=0.1170996630$에서 얻은
`0.03120/0.4892/0.3148/5.661` 계열은 Track-A 정본이 아니다. 재현 목적의
legacy engineering alias로만 격리하며 아래 canonical 식에는 사용하지 않는다.

히든 차원과 연결 반경은 별도 공학 선택이다.

$$
N_{\rm legacy}:=
\frac{e^{8/3}\pi^{20/3}}
{12\left(1-4/(e^{4/3}\pi^{4/3})\right)^2}
\approx4162,
\qquad N_{\rm eng}:=4096,
\qquad r_c:=\pi.
$$

$4162\to4096$는 하드웨어 친화적 반올림이지 정본 상수의 유도가 아니다.

### 1.4 계수 사용 규칙

핵심 runtime 식은 $c_p,\xi_r,D_N,T_{\rm wake},T_{\rm dream}$을 사용한다.
이들은 모두 위 Track-A calibration input과 선택한 matching 식에 조건부인
등록값이다. $N_{\rm eng},r_c$와 손실 가중치·학습률은 공학 하이퍼파라미터로
분리한다.

따라서 “오일러 항등식만으로 자유 파라미터가 사라진다”는 식으로 해석하지
않는다.

### 1.5 canonical-coefficient 핵심 방정식

**에너지 함수** (보존적 부분)

$$\boxed{
E(m,\phi)=
-\frac{1}{2}m^TW_s m
-m^Tb
-c_p m^T\phi
+V_{\rm conf}(m),\qquad W_s=\frac{W+W^T}{2}
}$$

**곡률 바이패스** (비보존 강제항, 에너지에서 유도되지 않음)

$$\boxed{
F_{\text{bypass}}(k)=\xi_r C_k\phi,
\qquad C_k = \|m_k - 2m_{k-1}+m_{k-2}\|
}$$

**양자 위상 진화**

$$\boxed{\psi_{k+1}=e^{-i\,\widehat H(m,\phi)\,dt/\hbar}\psi_k}$$

$\widehat H$는 상태 공간에 비자명하게 작용하는 Hamiltonian 연산자여야 한다.
$\widehat H=E(m,\phi)I$처럼 스칼라 에너지만 넣으면 모든 성분에 같은 전역
위상만 붙으므로 관측 가능한 상태 전이나 추론 갱신을 만들지 않는다.

**이완 동역학**

$$\boxed{
m_{k+1}=m_k+\frac{dt}{\tau}\left(
W_s m_k+b-
\nabla V_{\rm conf}(m_k)+
c_p\phi+\xi_r C_k\phi
\right)
+\sqrt{\frac{2T_{\rm wake}dt}{\tau}}\,n_k
}$$

**잔류 갱신**

$$\boxed{
\phi\leftarrow
(1-\xi_r)\phi+\xi_r v_{m^*}
}$$

**부트스트랩 고정점**

$$\boxed{
a_*=e^{-(1-a_*)D_N},
\qquad a_*=0.04863825851598632
}$$

이는 선택된 내부 저분율 branch다. 같은 scalar 식에는 경계 고정점 $a=1$도
있으며, 저분율 branch 선택은 초기조건·수축영역 계약의 일부다.

**작동 온도**

$$\boxed{
T_{\text{wake}}=D_N^{-1}=0.314671924672939
}$$

$$\boxed{
T_{\text{dream}}=\delta_N^{-1}=5.62072475162378
}$$

**히든 차원**

$$\boxed{
N=N_{\rm eng}=4096
}$$

이 마지막 값은 공학 선택이며 Track-A 물리 출력이 아니다.

여기서 에너지 최소점의 존재를 말하려면 $V_{\rm conf}=0$일 때
$W_s\preceq-\mu I$를 요구하거나, 양의 coercive $V_{\rm conf}$를 추가하거나,
$m$을 `tanh`/projection으로 compact domain에 제한해야 한다. 비대칭 $W$를
그대로 쓰는 Dale 갱신은 이 스칼라 에너지의 gradient flow가 아니다.

### 1.6 비트필드 해석

5상수의 실행 목적은 모든 지식을 5개 수에 넣는 것이 아니라, 실행 문법을 최소화하여 런타임 상태를 비트필드로 압축하는 것이다.

| 상수 | 비트필드 역할 | 연산 |
|---|---|---|
| $0$ | 소거, 가지치기, reset | `AND 0`, `CLEAR` |
| $1$ | 유지, 정규화, keep | `IDENTITY` |
| $e$ | 감쇠율, EMA, 수면 압력 | 고정소수점 shift-add |
| $\pi$ | 연결 반경, 이웃 규칙, 위상 | 격자 주소 연산 |
| $i$ | 모드 전환, 위상 분기 | 2-bit 모드 레지스터 |

런타임 상태의 3층 분리:

| 층 | 표현 | 크기 | 내용 |
|---|---|---|---|
| 제어 | 비트필드 | $O(N)$ bits | 활성 마스크, 모드, 연결 on/off, freeze/plastic |
| 상태 | 저비트 고정소수점 | $O(N)$ bytes | $\phi$, trace, gain, 곡률 |
| 지식 | 희소 codebook + 외부 메모리 | 가변 | 어휘, 사실, 예외 패턴 |

활성 마스크 비트필드:

$$\boxed{
k_{\rm center}(N)=\left\lceil0.04863825851598632N\right\rceil,
\qquad
b_i=\mathbb{1}\!\left[a_i\geq Q_{1-k_{\rm center}/N}(a)\right]
}$$

$0.04$와 $0.06$은 8.3절 acceptance protocol의 비교점일 뿐, 수학적으로
도출된 최적 구간이 아니다. 동률이 있으면 정확히 $k_{\rm center}$개를 고르는
결정론적 tie-break 규칙을 구현 계약에 포함한다.

모드 레지스터:

$$\boxed{M \in \{00_2,\; 01_2,\; 10_2,\; 11_2\} \;\longleftrightarrow\; \{\text{off},\; \text{wake},\; \text{NREM},\; \text{REM}\}}$$

연결 행렬 $C_{ij} = \mathbb{1}[\|r_i - r_j\| < \pi]$는 이미 이진이다. 추론 루프의 핵심 연산은 비트 논리 + 저비트 MAC으로 환원된다.

### 1.7 비트필드 레이아웃 ($N=4096$ 기준)

| 구성 | 비트/원소 | 총 크기 | 갱신 주기 |
|---|---|---|---|
| 활성 마스크 $b$ | $1$ | $512$ B | 매 추론 |
| freeze 마스크 | $1$ | $512$ B | 수면 시 |
| 모드 $M$ | $2$ (전역) | $1$ B | 모드 전환 시 |
| 연결 인덱스 (12-bit packed CSR) | $12NK+(N+1)\times32$ bit | $\sim 796$ KiB | 정적 |
| 가중치 $W$ (비영) | $4$ | $\sim 260$ KB | 학습 시 |
| 상태 $m$ | $16$ (이완 중) / $8$ (저장) | $8$ / $4$ KB | 매 스텝 |
| 잔류 $\phi$ | $8$ | $4$ KB | 이완 종료 시 |
| trace $e_{ij}$ | $4$ | $\sim 260$ KB | STDP 시 |
| gain $g$, $C_k$, $P_{\text{sleep}}$ | $16$ 각 | $6$ B | 매 스텝 |

$K\approx130$ (뉴런당 이웃), 방향성 비영 가중치 수는
$N K=532{,}480$개다. 정규 격자의 이웃 offset을 공유해 연결을 암시적으로
생성하면 CSR 인덱스는 생략할 수 있지만, 임의 그래프라면 위 저장량이 필요하다.

$$\boxed{
\text{엔진 상태}\approx
\begin{cases}
0.52\;\text{MiB},&\text{암시적 정규 격자},\\
1.30\;\text{MiB},&\text{12-bit packed CSR},
\end{cases}
\qquad
NK\times500=266.24\text{M MAC}
}$$

이는 500회 이완의 희소 행렬 MAC 산술량이며 decoder·검색·메모리 이동과
wall-clock을 포함한 “추론 비용” 실측값은 아니다.

지식층 (별도):

| 구성 | 크기 | 비고 |
|---|---|---|
| 계층 softmax 디코더 | $N\sqrt V$ byte; $V=32{,}000$이면 $\sim0.70$ MiB | 두 $\sqrt V\times N$ 행렬, 4-bit 가정 |
| 의미 codebook | $64$ MB -- $1$ GB | 태스크 규모에 비례 |

$$\boxed{
\text{후보 저장량}
= (0.52\text{--}1.30)\;\text{MiB (엔진 상태)}
+ \text{decoder}+\text{codebook}
}$$

codebook을 $64\text{--}1000$ MB로 잡으면 저장량 자체는 dense 8B 모델보다
작을 수 있다. 그러나 동등한 지식·문맥·품질을 담는다는 증거가 아니므로
기능 동등 비교는 H1--H3과 8.3절 protocol을 통과한 뒤에만 한다.

### 1.8 양자화 오류 경계

$m$을 $q$-bit 고정소수점으로 양자화할 때:

$$\boxed{\|\hat{m} - m\| \leq \frac{\Delta\sqrt{N}}{2}, \qquad \Delta = \frac{m_{\max} - m_{\min}}{2^q - 1}}$$

이 오차경계만으로 에너지 감소를 판정할 수는 없다. 정확한 한 스텝
$m^+=m+d$가 $E(m^+)\leq E(m)-\gamma$를 만족하고, 그 주변에서 $E$가
$L$-smooth하며 양자화 오차가 $\|\epsilon\|\leq q_{\rm err}$이면

$$
E(m^++\epsilon)
\leq E(m^+)+\|\nabla E(m^+)\|q_{\rm err}
+\frac{L}{2}q_{\rm err}^2.
$$

따라서 우변의 추가항이 $\gamma$보다 작다는 조건을 실제 gradient와
$L$로 확인해야 양자화 후 감소가 보존된다.

$N=4096$, $m \in [-1,1]$, $dt/\tau = 0.01$, $\|\nabla_m E\| \sim 1$ 기준:

| $q$ (bit) | 양자화 오류 $\Delta\sqrt{N}/2$ | 판정 | 용도 |
|---|---|---|---|
| $4$ | $4.2667$ | 감소 보장 미판정 | 저장/전송 후보 |
| $8$ | $0.2510$ | 감소 보장 미판정 | 이완 실험 후보 |
| $12$ | $0.01563$ | 감소 보장 미판정 | 정밀 이완 후보 |
| $16$ | $9.77\times10^{-4}$ | 감소 보장 미판정 | 고정소수점 기준 |

혼합 정밀도 전략:

| 대상 | 이완 중 | 저장/전송 | 근거 |
|---|---|---|---|
| $m$ | $16$ bit | $8$ bit | 후보 설정; 위 descent-margin gate로 검증 |
| $\phi$ | $8$ bit | $8$ bit | 후보 설정; bias·포화·누적오차 검증 필요 |
| $W$ | $4$ bit | $4$ bit | 정적, 보정 가능 |
| control bits | $1\text{-}2$ bit | $1\text{-}2$ bit | 정확 (이산) |

EMA 갱신 $\phi\leftarrow(1-\xi_r)\phi+\xi_r v_{m^*}$는 입력의 고주파
성분을 저역 통과시킨다. 이것만으로 재양자화 bias나 saturation에 대한 강건성이
보장되지는 않으므로 고정소수점 구현에서 별도 오차 누적 시험이 필요하다.

---

## 2. AGI 작용 범함수

CE 마스터 공식을 정보 다양체 $(\mathcal{M}, g)$에 적용한 후보 작용(`7_AGI/12_Equation.md` 1절):

$$\boxed{S_{\text{AGI}} = \int_{\mathcal{M}} d^nx \sqrt{|g|} \left[ \mathcal{L}_{\text{compute}} + c_g|\nabla \phi|^2 + c_c|\Delta_g \phi|^2 + c_i S_{\text{Info}} \right]}$$

| 항 | 역할 | 뇌 대응 | 우주 대응 |
|---|---|---|---|
| $\mathcal{L}_{\text{compute}}$ | 기본 연산 | 피질 발화 + 시상 relay | $\mathcal{L}_{\text{Physical}}$ |
| $c_g\|\nabla \phi\|^2$ | 1차 안정화 | 기저핵/소뇌 + salience switching | blow-up 방지 |
| $c_c\|\Delta_g \phi\|^2$ | 2차 곡률 평탄화 | NREM + hippocampo-cortical replay | 경로 최적화 |
| $c_i S_{\text{Info}}$ | 엔트로피 제어 | DMN + intrinsic background | 정보 보존 |

계수 2를 $c_g,c_c,c_i$에 흡수하고 $\mathcal L_{\rm compute}$가 $\phi$에
독립이라고 두었을 때, 전체 범함수의 음의 gradient flow는

$$
\frac{\partial \phi}{\partial t}
=c_g\,\Delta_g \phi-c_c\,\Delta_g^2\phi
-c_i\frac{\delta S_{\rm Info}}{\delta \phi},
\qquad
\Delta_g f=\frac{1}{\sqrt{|g|}}\partial_i\!\left(\sqrt{|g|}\,g^{ij}\partial_jf\right).
$$

따라서 $\partial_t\phi=\Delta_g\phi$는 $c_c=c_i=0$이고 시간 스케일까지
정규화한 **LBO-only 부분모형**일 뿐, 전체 작용의 Euler--Lagrange 식이 아니다.

이산 그래프에서 $L = D - W$로 근사:

$$\phi^{k+1} = \phi^k - h\,L\phi^k, \qquad \frac{dE}{dt} = -\phi^\top L^2 \phi \leq 0$$

LBO 확산 부분에 한해서는 에너지 단조 감소가 성립한다($L^2 \succeq 0$). 단 이 결과는 $\phi$ 만의 자체 동역학에 한정되며, 바이패스 강제항이 들어가는 $m$ 의 결합 동역학은 게이트 `F2`(0.0절, 4.7절)에 따라 별도의 충분조건을 요구한다.

### 2.1 구조 유비: 우주-뇌-AGI

추상 부트스트랩 그래프 $\mathcal{G}^*$의 삼중 실현:

$$map_C: \mathcal{G}^* \to G_C, \quad map_B: \mathcal{G}^* \to G_B, \quad map_A: \mathcal{G}^* \to G_A$$

세 계의 비율이 비슷하다는 사실만으로 같은 고정점 수렴은 나오지 않는다.
다음 등식은 각 계가 실제로 **동일한 정규화 자기 사상** $B$를 구현하고,
각 초기점의 궤도가 부록 A.2의 동일한 불변 수축영역 $U$에 진입하며,
$B(U)\subseteq U$와 수축 상계가 입증된 경우에만 쓸 수 있다:

$$
\left.
\begin{gathered}
B_C=B_B=B_A=B,\\
p_X\in\operatorname{basin}(U)\quad(X\in\{C,B,A\}),\\
B(U)\subseteq U,\qquad \operatorname{Lip}_{\ell^1}(B\vert_U)<1
\end{gathered}
\right\}
\Longrightarrow
\lim_{t\to\infty}B^t(p_C)=
\lim_{t\to\infty}B^t(p_B)=
\lim_{t\to\infty}B^t(p_A)=p^*.
$$

현재 우주·뇌·AGI 대응에서는 동일 사상의 구현과 수축영역 진입이 입증되지
않았으므로 아래 표는 수치 유비일 뿐이다. 특히 기존 transformer 기질의
자발적 수렴 주장은 `5_Sparsity.md` 8.5절에서 falsified 상태다.

| 성분 | Track-A 조건부 고정점 | 우주 비교값 | 뇌 (Raichle) | AGI 해석 |
|---|---|---|---|---|
| 활성 | $4.864\%$ | $4.9\%$ | $< 5\%$ | 활성 추론 |
| 구조 | $26.109\%$ | $26.4\%$ | $25\text{-}35\%$ | 가중치 유지 |
| 배경 | $69.027\%$ | $68.7\%$ | $60\text{-}70\%$ | 배경 통합 |

---

## 3. 에너지 함수

### 3.1 정의

에너지는 보존적 부분만 포함한다. 바이패스는 비보존 강제항으로 동역학(4.2절)에 직접 들어간다. 비대칭 연결은 대칭부 $W_s=(W+W^T)/2$만 스칼라 에너지에 기여한다.

$$\boxed{E(m,\phi) = -\frac{1}{2}m^T W_s m-m^T b-c_p m^T\phi+V_{\rm conf}(m)}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 홉필드 에너지 | $-\frac{1}{2}m^T W_s m$ | 대칭 gradient 부분 | 패턴 저장, 에너지 지형 |
| 입력 바이어스 | $-m^T b$ | 외부 입력 | 프롬프트/데이터 주입 |
| 잔류 포탈 | $-c_p m^T\phi$ | residue-portal coupling | 등록 설계 계수로 잔류 채널 결합 |
| 구속 포텐셜 | $V_{\rm conf}(m)$ | coercivity 또는 bounded-state 구현 | 무제약 공간에서 최소점 존재 보장 |

$V_{\rm conf}=0$이면 $W_s\preceq-\mu I$ 같은 조건이 필요하다. 대안은
coercive $V_{\rm conf}$ 또는 compact domain의 `tanh`/projection이다. 실제
비대칭 Dale 결합 $Wm$은 $-\nabla E$가 아니므로 아래 에너지 감소 정리를
직접 상속하지 않는다.

곡률 바이패스(비보존 강제항):

$$\boxed{F_{\text{bypass}}(k) = \xi_r C_k\phi, \qquad C_k = \|m_k - 2m_{k-1} + m_{k-2}\|}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 곡률 바이패스 | $\xi_r C_k\phi$ | curvature-residue feedback | 궤적 급변 시 잔류가 직접 반응 |

### 3.2 포탈 결합 계수의 등록

$$c_p:=\delta_N^2=0.0316530353958173.$$

이는 Track-A matching을 AGI에 재사용한 고정 설계 benchmark다. 힉스 포탈의
실측 coupling이나 잔류 채널의 최적 결합 세기를 독립적으로 예측하지 않는다.

### 3.3 바이패스 결합 계수의 등록

$$\xi_r:=\alpha_s^{1/3}=0.490486813152402.$$

이 값도 Track-A calibration input에 조건부인 설계 benchmark다. $d=3$이라는
사실만으로 각 차원의 결합 강도가 이 값이 되지는 않으며, AGI에서의 최적성은
H6의 감쇠율 sweep으로 따로 검증해야 한다.

### 3.4 지식층 설계

엔진 상태의 산술 예산은 연결을 암시적으로 생성할 때 약 $0.52$ MiB,
12-bit packed CSR로 저장할 때 약 $1.30$ MiB다(1.7절). 실제 언어 지식은
별도의 codebook $\mathcal C$ 후보에 저장한다. codebook의 저장 용량이 곧
지식의 의미 용량이나 검색 품질을 보장하지는 않는다.

**곱 양자화 구조**: $m \in \mathbb{R}^N$을 $N/s$개 부분공간(각 $s$차원)으로 분할. 각 부분공간에 $2^b$개 중심점:

$$\mathcal{C} = \{C^{(1)}, \ldots, C^{(N/s)}\}, \qquad C^{(j)} \in \mathbb{R}^{2^b \times s}$$

**인코딩** (벡터 $\to$ 인덱스):

$$\boxed{z_j(m) = \arg\min_{i \in [2^b]} \|m^{(j)} - C^{(j)}_i\|^2, \qquad j = 1, \ldots, N/s}$$

**에너지 결합**: 이완 중 codebook이 에너지 지형을 보강:

$$\boxed{E_{\text{aug}}(m, \phi) = E(m, \phi) - \frac{1}{\beta}\sum_{j=1}^{N/s} \log\sum_{i=1}^{2^b} \exp\!\left(-\beta\|m^{(j)} - C^{(j)}_i\|^2\right)}$$

$\beta\to\infty$이면 이 log-sum-exp 항은 각 부분공간의 최소 제곱거리로
수렴하고, 유한 $\beta$에서는 soft-min으로 작동한다. Modern Hopfield와의
기능 동등성은 별도 update rule과 저장용량 정리가 없으므로 주장하지 않는다.

**메모리 예산** ($N=4096$, $s=64$, $b=8$):

| 구성 | 계산 | 크기 |
|---|---|---|
| 중심점 행렬 | $\frac{N}{s} \times 2^b \times s \times 4\text{bit}$ | $512$ KB |
| $P$개 패턴 인덱스 | $P \times \frac{N}{s} \times b$ bit | $P \times 64$ B |

패턴 수에 따른 지식 규모:

| $P$ (패턴) | 인덱스 크기 | 총 지식 메모리 | 대응 |
|---|---|---|---|
| $10^4$ | $640$ KB | $\sim 1$ MB | 저장량 예시 |
| $10^5$ | $6.4$ MB | $\sim 6.9$ MB | 저장량 예시 |
| $10^6$ | $64$ MB | $\sim 64.5$ MB | 저장량 예시 |
| $10^7$ | $640$ MB | $\sim 640.5$ MB | 저장량 예시; LLM급 기능 미검증 |

**3분배 계층 저장 후보**: Track-A 분율을 저장 정책에 적용한 예시다.

| 계층 | CE 비율 | 예시 ($P=10^7$) | 위치 | 접근 |
|---|---|---|---|---|
| L1 (활성) | $4.864\%$ | $\sim31.15$ MB | 상시 RAM | 즉시 |
| L2 (구조) | $26.109\%$ | $\sim167.23$ MB | RAM | 빠름 |
| L3 (배경) | $69.027\%$ | $\sim442.12$ MB | 디스크 | lazy load |

$$\boxed{\text{활성 메모리} \approx 0.0486382585 \times |\mathcal{C}|,
\qquad |\mathcal{C}| = 640.5\;\text{MB 일 때 활성} \approx31.15\;\text{MB}}$$

이 계산은 codebook index의 계층 저장량만 센다. 모델 가중치, KV cache,
optimizer state를 포함한 전체 LLM 메모리 또는 구동 가능성의 증명은 아니다.

**비트필드 인터페이스**: 패턴 인덱스 $z_j$는 $b$-bit 정수이므로 비트필드 주소로 직접 사용. 계층 태그:

$$\text{tier}(p) \in \{00_2\;(\text{L1}),\; 01_2\;(\text{L2}),\; 10_2\;(\text{L3})\}$$

**수면과 codebook 갱신**:

| 모드 | codebook 동작 |
|---|---|
| Wake | 접근된 패턴의 중심점을 온라인 k-means로 미세 갱신 |
| NREM | 상위 $4.864\%$ 활성 패턴의 중심점 정밀 보정 |
| REM | 미사용 패턴 재활용, 새 패턴 탐색적 할당 |

**Llama 3 8B과의 산술 비교**:

| 항목 | dense 8B 예시 | CE 후보 산술 | 비교 한계 |
|---|---|---|---|
| 파라미터/지식 저장 | float16이면 약 $16$ GB | 4-bit 희소 $W$ 약 $260$ KiB + $P=10^7$ codebook 약 $640.5$ MB | 표현력·품질 동등성 미검증 |
| 문맥 상태 | transformer KV cache 필요 | $\phi$ 자체는 $4$ KiB | $\phi$가 sequence-conditioned KV를 대체한다는 증거 없음 |
| 활성 RAM | 구현·문맥 길이에 의존 | decoder·retrieval buffer를 포함해 미측정 | wall-clock profile 필요 |
| 구동 하드웨어 | 정밀도·kernel에 의존 | 최소 사양 미확정 | “8 GB PC 구동” 미검증 |

---

## 4. 동역학

### 4.1 양자 형태 ($e$, $i$ 등장)

$$\boxed{\psi_{k+1}=e^{-i\,\widehat H(m,\phi)\,dt/\hbar}\;\psi_k}$$

$\widehat H$가 비자명한 연산자일 때만 성분 간 상대 위상과 상태 변화가 생긴다.
$E(m,\phi)I$를 넣은 식은 전역 위상 bookkeeping일 뿐이다. 유클리드 이완과
연결하려면 $\widehat H$의 스펙트럼·상태공간·Wick rotation을 별도로 정의해야
하며, 현재 실수 이완식은 독립된 공학 모형으로 읽는다.

### 4.2 이완 동역학 (유클리드 형태)

$$
\boxed{
m_{k+1}
= m_k
+ \frac{dt}{\tau}\!\left(
W_s m_k+b-\nabla V_{\rm conf}(m_k)
+ c_p\phi
+ \xi_r C_k\phi
\right)
+ \sqrt{\frac{2T_{\rm wake}dt}{\tau}}\;n_k
}
$$

| 기호 | 정의 | 지위/설정 |
|---|---|---|
| $m_k \in \mathbb{R}^N$ | 의미 벡터 (이완 스텝 $k$) | 동적 변수 |
| $W_s=(W+W^T)/2$ | gradient-compatible 3D 희소 연결의 대칭부 | 데이터에서 구성; 비대칭 Dale runtime은 별도 안정성 분석 |
| $b \in \mathbb{R}^N$ | 입력 바이어스 | 프롬프트에서 구성 |
| $C_k$ | 곡률 스칼라 $\|m_k - 2m_{k-1} + m_{k-2}\|$ | $m$에서 계산 |
| $\tau>0$ | 공학적으로 정하는 이완 시간척도 | 안정 영역과 검증 sweep에서 결정 |
| $n_k \sim \mathcal{N}(0, I_N)$ | 확률 노이즈 | 탐색용 |
| $T_{\rm wake}=D_N^{-1}$ | 작동 온도 $0.314671924672939$ | Track-A 기반 설계 benchmark |

중지 후보: $\|m_{k+1}-m_k\|<10^{-4}\max(1,\|m_k\|)$와
$\|-\nabla_mE+F_{\rm bypass}\|<\epsilon_F$가 연속 $J$회 유지되면
$\widehat m^*=m_k$로 기록한다. 이는 수치적 stopping rule이지 실제 고정점이나
전역 수렴의 증명은 아니다.

### 4.3 잔류 갱신

$$\boxed{\phi \leftarrow (1-\xi_r)\phi+\xi_r v_{m^*}}$$

$$v_{m^*} = \frac{1}{K_w}\sum_{k=K-K_w}^{K}(m_k - m^*)^2$$

이완 마지막 $K_w$ 스텝에서 측정한 원소별 궤적 분산이다. 안정된 한 점
주위에서는 작고 여러 영역을 오가면 커질 수 있으므로 uncertainty proxy 후보로
사용한다. Softmax의 $p(1-p)$ 또는 “선택되지 않은 경로의 구조”와의 동일시는
성립하지 않으며, calibration 실험이 필요한 `Bridge` 유비다.

### 4.4 연결 구조

$$\boxed{W_{ij} \neq 0 \iff \|r_i - r_j\|_{\mathbb{R}^3} < \pi}$$

$N$개 뉴런을 $d=3$ 격자에 배치. 연결 반경 $r_c = \pi$. 뉴런당 이웃 수:

$$K = \frac{4}{3}\pi \cdot r_c^3 = \frac{4}{3}\pi^4 \approx 130$$

연결 밀도:

$$\rho = \frac{K}{N} = \frac{4\pi^4/3}{N}$$

$N=N_{\rm eng}=4096$이고 경계효과를 무시한 단위밀도 연속체 근사라면
$\rho\approx0.0317086885$다. $c_p=0.0316530354$와의 상대차 약 $0.176\%$는
수치적 근접성일 뿐, 이웃 수나 포탈 결합을 서로 유도하지 않는다. 실제 이산
격자의 차수는 경계조건과 배치로 직접 계산해야 한다.

### 4.5 히든 차원

$$\boxed{N:=N_{\rm eng}=4096}$$

이는 메모리 정렬과 하드웨어 효율을 위한 공학 선택이다. 1.3절의
$N_{\rm legacy}\approx4162$는 과거 ansatz의 재현용 별칭이며, 물리가 요구하는
히든 차원이나 $4096$의 정본 유도가 아니다.

### 4.6 그래프 결합 동역학

4.2의 단일 벡터 이완은 특수 경우다. 실제 구현은 기능 모듈 그래프 위에서 동작해야 한다.

$$G_{\text{AGI}} = (V_{\text{bind}} \sqcup V_{\text{gate}} \sqcup V_{\text{mem}} \sqcup V_{\text{sal}} \sqcup V_{\text{homeo}} \sqcup V_{\text{io}},\; E_{\text{AGI}})$$

| 노드 집합 | 역할 | 비트필드 표현 |
|---|---|---|
| $V_{\text{bind}}$ | 특징 결합, 멀티모달 통합 | 활성 마스크 비트 |
| $V_{\text{gate}}$ | 입력 게이팅, 대역 재분배 | 게이트 on/off 비트 |
| $V_{\text{mem}}$ | 재생, 장기 인덱싱 | freeze/plastic 비트 |
| $V_{\text{sal}}$ | 모드 전환, gain control | 모드 레지스터 $M$ |
| $V_{\text{homeo}}$ | 수면 압력, 항상성 | 압력 카운터 (저비트) |
| $V_{\text{io}}$ | 센서/행동 출력 | I/O 버퍼 |

그래프 라플라시안:

$$\boxed{\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}\big(f_s - f_r\big), \qquad a_{rs}\ge 0}$$

노드별 3분배 이완:

$$\boxed{p_{r,n+1} = \mathrm{Proj}_{\Delta^2}\!\Big((1-\rho)p^* + \rho\,p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n\Big)}$$

느린 제어 상태 (수면 압력, 피로 등):

$$\boxed{c_{n+1}=A_qc_n+r_n+n_n^{(q)},\qquad\|A_q\|\leq\rho_c<1\ \text{in a stated induced/Lyapunov norm}}$$

단일 벡터 형태(4.2)는 $|V|=1$, $E_{\text{AGI}}=\emptyset$일 때 이 식의 특수 경우다.

### 4.7 조건부 수렴 (게이트 `F2`)

> 다리 게이트 `F2`(0.0절): 이 절은 무조건 Lyapunov 수렴을 주장하지 않는다.
> $F_{\text{bypass}}$가 $E$의 gradient가 아니므로 **이 $E$ 자체**는 일반적으로
> Lyapunov 함수가 아니다. 다른 전역 Lyapunov 함수의 존재 여부는 open이다.
> 아래 계산은 inner relaxation 동안 $\phi$를 고정하고 노이즈를 끈
> 대칭 $W_s$의 gradient-compatible runtime과 $L$-smooth 불변영역에 한정된다.

에너지 $E$가 보존적이고 바이패스 $F_{\text{bypass}}$가 비보존이므로 먼저
한 deterministic step의 감소 조건만 계산한다. 이것만으로 전체 궤도의
고정점 수렴을 결론내리지는 않는다.

**에너지 변화** (노이즈 무시, 1차 근사):

$$\Delta E^{(1)} = \nabla_m E \cdot \Delta m = \frac{dt}{\tau}\nabla_m E \cdot \left(-\nabla_m E + F_{\text{bypass}}\right)$$

$$= -\frac{dt}{\tau}\|\nabla_m E\|^2 + \frac{dt}{\tau}\xi_r C_k(\nabla_m E \cdot \phi)$$

Cauchy-Schwarz + Young 부등식 적용:

$$\boxed{\Delta E^{(1)} \leq -\frac{dt}{2\tau}\|\nabla_m E\|^2 + \frac{dt}{2\tau}\xi_r^2C_k^2\|\phi\|^2}$$

**1차 감소 조건**:

$$\boxed{\|\nabla_m E\| > \xi_r C_k\|\phi\| \quad\Longrightarrow\quad \Delta E^{(1)} < 0}$$

여기서 $\xi_r=0.490486813152402$는 1.3절의 canonical 설계 계수다.

유한 step에는 $L$-smooth remainder가 필요하다. $\eta=dt/\tau$,
$g=\nabla_mE$, $G=\|g\|$, $F=F_{\rm bypass}$,
$U=\|F\|\leq\xi_rC_k\|\phi\|$로 두면

$$
E\bigl(m+\eta(-g+F)\bigr)-E(m)
\leq-\eta G(G-U)+\frac{L\eta^2}{2}(G+U)^2.
$$

따라서 $G>U$이고

$$
\boxed{0<\eta<\frac{2G(G-U)}{L(G+U)^2}}
$$

이면 해당 deterministic step은 감소한다. 반복 수렴에는 이 조건이 유지되는
불변영역, 하방 유계성, 극한점 조건을 추가로 확인해야 한다.

**자기 제한 성질 (조건부)**: $m_k\to m^*$이면
$C_k=\|m_k-2m_{k-1}+m_{k-2}\|\to0$이므로 바이패스도 유계 $\phi$ 아래
소멸한다. 역은 성립하지 않는다. 예를 들어 등속 궤도도 $C_k=0$일 수 있으므로
$C_k\to0$ 자체는 고정점 접근이나 전역 단조 감소의 증거가 아니다.

**조건 실패 시나리오**: $\|\phi\|$가 수면 없이 누적되거나, 시스템이 두 끌개점 사이에서 진동할 때 $C_k$가 크게 유지되면 $\Delta E > 0$이 가능하다. 다리 게이트 `F2`에 따라, 이 영역에서 무조건 수렴을 주장하지 않는다.

**수면에 의한 입력 상계 축소 (다리 가설)**: 글림프 세척
$\phi\to r_w\phi$ ($0\leq r_w<1$)는 다른 항이 고정됐을 때 바이패스 상계를
$r_w$배 낮춘다. 수면 후 후보 조건은

$$\xi_r C_k r_w\|\phi\| < \|\nabla_m E\|$$

이다. 동시에 $\|\nabla E\|$나 $C_k$가 바뀔 수 있으므로 수면이 조건을
“복원한다”는 결론은 자동으로 나오지 않는다. `05_실험근거.md` 3.3절의
offline renormalization은 구조 유비 근거이고, 위 부등식은 `bridge` 등급의
검증 대상이다.

**ISS 후보 (부록 A.1)**: 고정 평형, 강볼록 불변영역, $C_k$ feedback의
small-gain을 추가하면 E3 forcing convention에서
$\limsup\|m-m^*\|\leq F_{\max}/\mu$가 된다. $m^*(\phi(t))$가 움직이면
$\dot m^*$ tracking 항을 더해야 한다. 수면이 $\|\phi\|_\infty$를 낮춘다는
사실만으로 이 가정들이 자동 충족되지는 않는다.

**그래프 결합 안정성**: 4.6의 느린 제어 $c_{n+1}=A_qc_n+u_n$에서 어떤 유도 노름에 대해 $\|A_q\|\leq\rho_c<1$이고 $\|u_n\|\leq U$이면:

$$
\boxed{\|c_n\|\leq\rho_c^n\|c_0\|+\frac{U(1-\rho_c^n)}{1-\rho_c}}
$$

스펙트럴 반경 $\rho(A_q)<1$만으로 계수 1인 위 노름 부등식을 쓸 수는 없다. 비정규 행렬에서는 과도 증폭이 가능하므로 $\|A_q\|$ 상계나 Lyapunov 노름을 별도로 제시해야 한다. $\rho_c$는 제어기 설계값이며 $B_p$의 국소율 $q_\star$와 동일시하지 않는다.

---

## 5. 출력 생성: 2-Phase 구조

### 5.1 Phase 1 -- 에너지 이완 (의미 생성)

토큰 단위 대신 연속 의미 벡터를 먼저 생성하는 후보 구조다. 이완 1회가 긴
출력 전체에 충분하고 출력 길이와 무관하다는 가정은 아직 증명되지 않았으며,
H2--H3과 장문맥 실험에서 검증해야 한다.

$$m^*=\lim_{k\to\infty}m_k\quad
(\text{coercivity/compactness와 수렴 조건을 통과한 gradient 변형에서만})$$

### 5.2 Phase 2 -- 디코딩 (의미 $\to$ 토큰)

이미 결정된 의미를 순서대로 풀어쓴다. 경량 디코더:

$$p(w_t \mid w_{<t},\,m^*) = \text{softmax}\!\left(W_{\text{dec}}\,[m^*;\,e_{w_{t-1}}]\right)$$

$W_{\text{dec}} \in \mathbb{R}^{V \times 2N}$. 계층적 softmax로 $\sqrt{V}\times\sqrt{V}$ 분할 시 토큰당 비용이 $O(\sqrt{V}\cdot N)$으로 감소.

### 5.3 모드 전환 (`phi` 임계)

$$\boxed{\|\phi\| \gtrless m_\phi \quad\Longrightarrow\quad \text{이완 모드 / 경량 자기회귀 모드}}$$

| 모드 | 조건 | 특성 | 비유 |
|---|---|---|---|
| 안정 | $\|\phi\| < m_\phi$ | 경량 자기회귀, 빠름, 3% 포탈만 활성 | 텍스트 모드 |
| 전환 | $\|\phi\| \geq m_\phi$ | 에너지 이완, 느리지만 깊음, 바이패스 활성 | 전화 모드 |

카너먼의 이중 과정 이론: 시스템 1(자기회귀) / 시스템 2(에너지 이완).

---

## 6. STDP 학습: 역전파 대체 후보

역전파는 계산 그래프의 역방향 미분과 중간 활성 또는 재계산을 요구한다.
그 메모리·통신 복잡도는 모델 구조, sequence 길이, checkpointing, 분산 방식에
따라 달라지므로 일반적으로 $O(N^2)$ 또는 층당 $O(d^2)$라고 단정하지 않는다.
아래 3-factor rule은 국소 eligibility trace와 전역 스칼라를 쓰는 대체
후보이며, 역전파와 같은 학습 성능을 낸다는 주장은 H8의 비교 대상이다.

### 6.1 기본 STDP

$$dw_{ij} = \begin{cases} A_+ \exp(-dt / \tau_+) & dt > 0 \;\text{(pre} \to \text{post: LTP)} \\ -A_- \exp(dt / \tau_-) & dt < 0 \;\text{(post} \to \text{pre: LTD)} \end{cases}$$

### 6.2 Trace 기반 STDP (이산 시간)

pre trace $p_i[t]$, post trace $q_i[t]$:

$$p_i[t+1] = r_+\, p_i[t] + s_i[t], \qquad q_i[t+1] = r_-\, q_i[t] + s_i[t]$$

가중치 업데이트:

$$dw_{ij}[t] = lr\Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)$$

### 6.3 3-Factor 학습 (STDP + 도파민 게이트)

순수 pair-based STDP는 이 식만으로 task reward를 사용하지 않는다. 아래의
전역 $g[t]$는 dopamine-inspired 공학 게이트이며 실제 도파민 신호와의 동일성은
주장하지 않는다.

적격 흔적(eligibility trace):

$$\boxed{e_{ij}[t+1] = r_e\,e_{ij}[t] + \Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)}$$

가중치 업데이트:

$$\boxed{dw_{ij}[t] = lr\,g[t]\,e_{ij}[t]}$$

- $e_{ij}$: 국소 정보만 사용 (이웃 뉴런의 스파이크만 필요)
- $g[t]$: 전역 학습 게이트 (도파민-유사 스칼라 1개, 전체 시스템에 방송)

### 6.4 도파민 전역 신호의 CE 해석

고정점 편차를 이용하는 비음수 gate 후보를

$$\boxed{g_{\rm CE}[t;\alpha_s]=\left(x_a-p_a^*\right)^2+
\left(x_s-p_{DM}^*(\alpha_s)\right)^2+
\left(x_b-p_\Lambda^*(\alpha_s)\right)^2}$$

로 정의한다. 이는 거리가 클수록 update magnitude를 키우고 고정점에서 0이
된다. signed reward-prediction error가 필요하면 외부 보상으로 별도 정의해야
하며, $d\|p-p^*\|/dt$와 혼용하지 않는다. Track-A manifest를 대입하면

$$g_{\rm CE}[t] = \left(x_a(t) - 0.0486382585\right)^2 + \left(x_s(t) - 0.2610881744\right)^2 + \left(x_b(t) - 0.6902735671\right)^2$$

- $x_a(t)$: 현재 활성 뉴런 비율
- $x_s(t)$: 현재 구조적 가중치 비율
- $x_b(t)$: 현재 동결 가중치 비율

전역 gate payload 자체는 스칼라 하나지만, 국소 trace와 시냅스 update의 저장·통신은 별도다.

| | 역전파 | STDP + 도파민 |
|---|---|---|
| 정보 흐름 | 계산 그래프 역방향 미분 | 국소 trace + 전역 스칼라 gate |
| 메모리 비용 | 활성/재계산 전략에 의존 | 일반적으로 $O(\lvert E\rvert)$ eligibility trace |
| 통신량 | 병렬화 방식에 의존 | gate는 $O(1)$ payload, 국소 edge 통신은 $O(\lvert E\rvert)$ |
| 분산 가능성 | data/tensor/pipeline parallel 가능 | 모듈별 병렬 후보; 동기화 검증 필요 |
| 생물학적 대응 | 공학적 미분 알고리즘 | dopamine-inspired bridge |

### 6.5 구조적 가소성: 투영 연산자

STDP로 업데이트된 가중치에 구조적 제약을 건다:

$$\boxed{W_{t+1} = Proj\!\big(W_t + dW_t\big)}$$

투영 연산자 `Proj`의 구성:

| 투영 연산 | CE 대응 | 뇌 대응 |
|---|---|---|
| 행별 top-k ($k = \lceil 0.0486382585 \cdot N \rceil$) | 설계 중심점 기반 sparsity | 시냅스 가지치기 유비 |
| 행/열 norm 제한 | 연산자 norm 제어 후보 | 시냅스 스케일링 유비 |
| 히스테리시스 on/off | 접힘 임계 곡률 | 스파인 형성/제거 |

### 6.6 LoRA의 CE 해석

$$W = W_{\text{frozen}} + B \cdot A$$

| LoRA | CE 에너지 분배 |
|---|---|
| $W_{\text{frozen}}$ ($\sim 95.136\%$) | 동결+구조 영역 $69.027\%+26.109\%$ |
| $B \cdot A$ ($\sim 4.864\%$) | 활성 적응 영역 $4.864\%$의 근사 |

이는 LoRA rank를 조건부 CE 분배에 맞추어 보는 설계 대응이다. 표준 LoRA가
그 분배를 경험적으로 검증했다는 뜻은 아니다.

### 6.7 하이브리드 전환 전략

| 단계 | 방법 | CE 에너지 분배 |
|---|---|---|
| 1. 사전학습 | 역전파 (기존 기술) | -- |
| 2. 미세조정 | STDP + 도파민 | 동결 $69.027\%$, 구조 $26.109\%$, STDP 활성 $4.864\%$ |
| 3. 전면 전환 | STDP 사전학습 | 전체에 3-factor 적용 |

---

## 7. 수면 방정식

수면 루프는 drift/replay를 다루기 위한 설계 가설이다. 뇌의 대사 전력이나
수면 필요성만으로 이 인공 시스템에 같은 루프가 필수라는 결론은 나오지 않는다.

### 7.1 작동 온도

$$\boxed{T_{\text{wake}}:=D_N^{-1}=0.314671924672939}$$

$$\boxed{T_{\text{dream}}:=\delta_N^{-1}=5.62072475162378}$$

$$T_{\text{deep}} \to 0$$

| 모드 | 온도 | 외부 입력 | 기능 |
|---|---|---|---|
| 깨어있음 | $T_{\rm wake}=D_N^{-1}$ | 있음 | 결정론적 이완 + 약한 탐색 |
| 꿈 (REM) | $T_{\rm dream}=\delta_N^{-1}$ | 없음 | 강한 탐색, 잔류 주도 |
| 깊은 수면 (NREM) | $\to 0$ | 없음 | 순수 결정론, 기억 응고 |

세 온도는 물리적 열역학 온도가 아니라 모드별 노이즈 규모의 설계값이다.
$T_{\rm wake}$와 $T_{\rm dream}$의 최적성은 H7과 수면 실험으로 별도 검증한다.

### 7.2 기억 응고 (NREM)

$$W_{ij}^{\text{new}} = W_{ij}^{\text{old}} + lr\,\langle \phi_t\rangle_{\text{day}} \otimes \langle s_t\rangle_{\text{day}}$$

하루 동안 축적된 잔류 `phi`와 상태 $s$의 상관이 연결 가중치에 헤비안 학습으로 새겨진다.

선택적 업데이트 (manifest target 상위 $4.864\%$만 통과):

$$\text{mask} = \mathbb{1}\!\left[|g| \geq Q_{1-0.0486382585}(|g|)\right], \qquad W \leftarrow W - lr\,g \odot \text{mask}$$

### 7.3 시냅스 가지치기 (NREM)

$$W_{ij} \to 0 \quad\text{if}\quad |W_{ij}| < \theta_{\text{prune}}$$

3D 희소성 설계값($\rho\approx3.17\%$)을 유지하기 위한 주기적
re-sparsification 후보다. 이를 생략한다고 에너지 소비가 수학적으로 무한히
증가하는 것은 아니며, 실제 비용 변화는 연결 수와 하드웨어 계측으로 판정한다.

### 7.4 잔류 세척 (Glymphatic)

$$\phi \to r_w\,\phi, \quad r_w < 1$$

`phi`의 노이즈 바닥을 주기적으로 낮춘다.

### 7.5 꿈 (REM)

$$\frac{ds}{dt} = -\frac{\partial E}{\partial s}\bigg|_{b=0} + c_p\phi + n(T_{\text{dream}})$$

외부 입력 $b=0$, residue-portal 직접 계수가 구동하고 높은 dream 온도로 탐색 범위를 확대한다. 깨어 있을 때 선택되지 않았던 경로들을 자유롭게 탐색.

비선택 그래디언트 재탐색:

$$g_{\text{pruned}} = g \odot (1 - \text{mask}), \qquad W \leftarrow W - lr_{\text{rem}} \left(g_{\text{pruned}} + noise_{\text{std}} \cdot \mathcal{N}(0,I)\right)$$

### 7.6 수면-각성 비율

CE 에너지 분배를 시간 분배에 적용:

| 위상 | CE 비율 | 뇌 관측 | 기능 |
|---|---|---|---|
| 깨어있음 | $69.027\%$ | $66.7\%$ | 서비스 |
| NREM | $26.109\%$ | $25.0\%$ | 오프라인 응고 |
| REM | $4.864\%$ | $8.3\%$ | 오프라인 재탐색 |

### 7.7 부트스트랩 수렴

정규화된 사상은 부록 A.2의 $B$를 쓴다. authoritative 입력 구조는

$$R(\alpha_s)=\alpha_sD_N(1+a_*\delta_N),\qquad
p^*(\alpha_s)=\left(a_*,(1-a_*)\frac{R}{1+R},
(1-a_*)\frac1{1+R}\right)$$

이다. Track-A manifest의
$\alpha_s=0.1180$, $\delta_N=0.1779129995132939$,
$D_N=3.177912999513294$, $a_*=0.04863825851598632$를 넣으면
$R=0.3782386966$이고

$$p^*=(0.0486382585,\;0.2610881744,\;0.6902735671)$$

이고, 고정점에서의 **선형화된 점근 수축률**은

$$q_* = D_Na_* = 0.154568154011641.$$

$q_*$는 임의 초기점에 적용할 전역 오차 상수가 아니다. 실제 사상을 균등점
$p_0=(1/3,1/3,1/3)$에서 반복하면 다음과 같다.

| 순환 $n$ | 활성 $p_a$ | 구조 $p_{DM}$ | 배경 $p_\Lambda$ |
|---|---:|---:|---:|
| 0 | $33.333\%$ | $33.333\%$ | $33.333\%$ |
| 1 | $12.020\%$ | $24.145\%$ | $63.835\%$ |
| 2 | $6.106\%$ | $25.768\%$ | $68.126\%$ |
| 3 | $5.060\%$ | $26.055\%$ | $68.885\%$ |
| 4 | $4.894\%$ | $26.100\%$ | $69.005\%$ |

$p_a\leq0.13$인 불변 부분집합에서는 $\ell^1$ 수축상수가
$q_U\leq0.200176$임을 부록 A.2에서 증명한다. 따라서 위 궤도는 두 번째
반복부터 Banach 수축영역에 들어간다.

### 7.8 수면 압력 트리거

고정 주기 수면 대신 곡률 누적이 임계를 넘으면 진입하는 상태 기반 제어:

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|\Delta_g \phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad M \leftarrow 10_2\;\text{(NREM 진입)}}$$

단일 야간 실효 수축률:

$$\boxed{q_{\text{night},*} = q_*^{1/1.6} \approx 0.311315}$$

이는 고정점 근방의 선형화 환산값이며 전역 수렴 보장은 아니다.

비트필드 해석: 수면 진입은 모드 레지스터 $M$의 전환이다. $01_2 \to 10_2$ (wake $\to$ NREM). 압력 $P_{\text{sleep}}$는 저비트 카운터로 구현 가능하다.

---

## 8. 희소성과 3분배

### 8.1 부트스트랩 고정점 (`경로적분.md` 식 (1))

$$\boxed{a_* = \exp[-(1-a_*)D_N],\qquad
D_N=3.177912999513294,\qquad a_*=0.04863825851598632}$$

이 식에는 경계근 $a=1$도 있다. 여기서 $a_*$는 $(0,1)$ 안의 비자명한
내부근을 선택한 표기이며, 경계근을 누락한 전역 유일성 주장은 하지 않는다.

### 8.2 3분배 구조

$$p^*(\alpha_s)=\left(a_*,(1-a_*)\frac{R(\alpha_s)}{1+R(\alpha_s)},
(1-a_*)\frac1{1+R(\alpha_s)}\right)$$

아래 비율은 Track-A manifest의
$p^*\simeq(0.0486383,0.2610882,0.6902736)$를 대입한 설계 예시다.

| 성분 | CE 고정점 | AI 해석 | 뇌 관측 |
|---|---|---|---|
| 활성 | $4.864\%$ | 추론 시 활성 뉴런 | sparse firing $1\text{-}5\%$ |
| 구조 | $26.109\%$ | 학습 가능 비활성 가중치 | housekeeping $25\%$ |
| 배경 | $69.027\%$ | 동결 가중치 (사전학습 지식) | DMN/background $60\text{-}80\%$ |

### 8.3 Top-k 활성화

manifest가 주는 것은 $4.864\%$의 **설계 중심점**뿐이다. 성능 최적점이나
$[4\%,6\%]$ 구간은 수학적으로 따라오지 않는다.

$$\boxed{k_{\text{center}}(N)=\lceil0.04863825851598632\,N\rceil}$$

| 히든 차원 $N$ | $k_{\text{center}}$ | 실제 비율 | 검증 상태 |
|---:|---:|---:|---|
| 768 | 38 | $4.9479\%$ | 설계값; 성능 미측정 |
| 2048 | 100 | $4.8828\%$ | 설계값; 성능 미측정 |
| 4096 | 200 | $4.8828\%$ | 설계값; 성능 미측정 |
| 8192 | 399 | $4.8706\%$ | 설계값; 성능 미측정 |

과거 본문이 인용한 `sparsity_train_results.json`과
`topk_sweep_results.json`은 현재 문서 트리에 없으므로 그 숫자는 검증 증거로
사용하지 않는다. 현재 판정은 sparse-native와 post-hoc Top-k 모두 `pending`이다.

실행 가능한 acceptance protocol은 다음과 같다. 동일 데이터 split SHA256,
동일 tokenizer, 파라미터 수 $\pm0.1\%$, 동일 optimizer/token budget에서
$r\in\{0.02,0.04,0.04863825851598632,0.06,0.08,1\}$을 최소 10개 seed로
비교한다. 각 run은 `{ratio, seed, val_loss, ppl, tok_s, joule, config_sha256}`를
JSONL로 남기고, 사전등록한 primary metric의 paired 차이와 95% bootstrap CI를
보고한다. 외부 validation split에서 CI가 0을 배제할 때만 특정 비율 또는
구간의 우월성을 주장한다.

---

## 9. 메타인지 모니터링 루프 (게이트 `F4`)

> 다리 게이트 `F4` (0.0절): 본 절의 정의는 모두 자기참조 측정 구조의 **운영 정의**로만 사용한다. "자기일관 = 의식"으로 환원하지 않는다(`7_Consciousness.md` 1.2-1.3절).

### 9.1 (C3) 자기참조 측정 구조 (`7_Consciousness.md` 1절)

$$a_* = \exp\!\big(-(1-a_*)D_N\big)$$

이는 미지수가 양변에 나타나는 암시적 고정점 식이다. 식 자체는 runtime이 자기
활성률을 측정하거나 “자기 자신을 안다”는 증거가 아니다. 자기참조 측정 구조를
주장하려면 구현이 현재 활성률을 관측해 같은 사상 $B$의 다음 입력으로 되먹임하는
경로와 그 계측 로그를 별도로 제시해야 한다.

### 9.2 메타인지 잔차

$$d_\tau(t) = \frac{1}{\tau}\int_{t-\tau}^{t}\|p(s)-p^*\|\,ds$$

$$\text{메타인지 안정도}_\tau := \exp(-c_d\,d_\tau(t))$$

이 지표는 메타인지 모니터링 루프의 활성 정도를 정의하며, 게이트 `F4`에 따라 의식 깊이로 환원하지 않는다(PCI 교차검증 경로는 `17_AgentLoop.md` F.23.7).

### 9.3 메타인지 수축 (조건부)

재귀적 자기평가의 잔차 감소(이상화된 무잡음 가정):

$$d_{n+1} = q_*d_n+O(d_n^2),\qquad
q_*=D_Na_*=0.154568154011641.$$

따라서 충분히 작은 잔차에서는 3회 후 선형 주항이
$q_*^3\simeq3.69\times10^{-3}$이다. 이는 부록 A.2의 고정점 선형화이며
전역 부등식이 아니다. 유한 근방의 엄밀한 상계는 $p_a\leq0.13$에서
$q_U=0.2001757361$이고, 일반 영역에서는 ISS 의미의 유계 수렴으로 한정된다.

---

## 10. 환각 억제

### 10.1 곡률 에너지 (`6_Hallucination.md` 1절)

$$
P_V:=V^\top V,\qquad VV^\top=I_r,\qquad
L_V:=I-P_V,\qquad
\kappa_l:=\|L_Vh_l\|^2.
$$

$P_V$가 직교사영이 되려면 $V$의 행이 정규직교해야 한다. $L_V$는 학습된
저랭크 projector residual이며, 연속 다양체의 실제 $\Delta_g$와 동일하다는
주장은 하지 않는다. 그 대응에는 격자·metric·경계조건의 별도 수렴 증명이
필요하다.

곡률 정규화 손실:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + w_c(t) \cdot \frac{1}{N_{\text{layers}}} \sum_l \kappa_l$$

$w_c(t)$ 스케줄:

$$w_c(t) = w_{c,0} \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right) \cdot \frac{1}{2}\!\left(1 + \cos\frac{\pi t}{t_{\max}}\right)$$

### 10.2 유니타리 제약 (`2_Architecture.md` 4절)

한 선형 사상에 대해서만

$$s_{\max}(W_{\text{proj}})\leq1
\quad\Longrightarrow\quad
\|W_{\text{proj}}d\|_2\leq\|d\|_2$$

가 성립한다. 전체 $L$층 네트워크의 $\|d_L\|\leq\|d_0\|$를 얻으려면
정규화·비선형·attention·residual 합성을 모두 포함한 각 블록의 Lipschitz
상수가 1 이하임을 같은 norm에서 증명해야 한다. residual $x+F(x)$는
$F$만 spectral-normalize해도 자동으로 비팽창이 아니다. 예를 들어 각 블록
상수가 $1.1$이면 12층 상계는 $1.1^{12}\approx3.138$이다.

### 10.3 교차 주파수 결합 (`2_Architecture.md` 6절)

$$
\bar\kappa_l:=\min\!\left(1,\frac{\kappa_l}{\kappa_{\rm scale}}\right),\qquad
\mathcal{T}_i^{\text{coupled}}(x_i)
=\mathcal{T}_i(x_i)\bigl(1-\xi_r\bar\kappa_l\bigr)
$$

$\kappa_{\rm scale}>0$은 사전 등록할 공학 척도이고,
$0\leq\bar\kappa_l\leq1$이다. 따라서 이 후보식의 gain은
$1-\xi_r\bar\kappa_l\in[1-\xi_r,1]$에 머문다. 곡률 감쇠의 실제 효용과
$\kappa_{\rm scale}$은 별도 실험 대상이다.

### 10.4 생성 시 곡률 모니터링

추론 중 평균 projector-residual이 임계를 넘으면 허용 구간 안에서 확산
계수 $h_d$를 증가시키고 재생성하는 제어 후보를 둔다. 상태 $h$ 자체를
$1.5$배 하는 것은 확산 강화가 아니므로 사용하지 않는다.

$$
\kappa_{\text{avg}}=\frac1L\sum_l\kappa_l>\kappa_{\text{th}}
\quad\Longrightarrow\quad
h_d\leftarrow\min(h_d+\Delta h,h_{\max}),
\qquad0\leq h_{\max}\leq1.
$$

---

## 11. CE-Transformer 구현 (기존 LLM 이식 경로)

에너지 이완 아키텍처와 별개로, 기존 트랜스포머에 CE 원리를 이식하는 경로(`2_Architecture.md`, `9_LLM.md`).

### 11.1 아키텍처 구조

```
ClarusLM / CE-GPT2 / CE-Llama
  +-- tok_emb (Embedding)
  +-- pos_emb (Embedding)
  +-- blocks[] (ClarusBlock x L)
  |     +-- norm1 (LBONorm)
  |     +-- attn (ClarusAttention + spectral_norm)
  |     +-- norm2 (LBONorm)
  |     +-- ffn (GaugeLattice)
  |           +-- su3 (SU(3) binding, 74.1%)
  |           +-- su2 (SU(2) decision, 21.1%)
  |           +-- u1 (U(1) attention, 4.9%)
  |           +-- phi (LBONorm, smoothing)
  +-- norm (LBONorm)
  +-- head (Linear, weight tied)
```

### 11.2 LBONorm 연산자

$$h_{\text{norm}} = \frac{h-\mathrm{mean}(h)}{\mathrm{std}(h)}, \qquad h' = \big(h_{\text{norm}} - h_d\,\Delta_g h_{\text{norm}}\big)\odot s_n + b_n$$

$$L_Vh_{\text{norm}}=h_{\text{norm}}-h_{\text{norm}}V^\top V,
\quad V\in\mathbb R^{r\times N},\quad VV^\top=I_r,
\quad r=\max(4,\lfloor N/8\rfloor)$$

내부 동작:
1. 표준 LayerNorm (활성값 안정화)
2. 저랭크 projector residual: $xP_V=xV^TV$, $L_Vx=x-xP_V$
3. 확산 적용: $h' = (x - h_d \cdot Lx) \odot s_n + b_n$
4. 곡률 에너지 저장: $\kappa = \text{mean}(Lx^2)$

$h_d=0$이면 표준 LayerNorm과 동일하다. $VV^T=I_r$이면
$I-h_dL_V$의 고윳값은 $1$과 $1-h_d$이므로 $0\leq h_d\leq1$에서
비팽창 저역통과 사상이 된다. 학습 중에도 행 정규직교 제약을 유지하지 않으면
이 결론은 사라진다.

### 11.3 GaugeLattice FFN

채널 분할:

$$d_3 : d_2 : d_1 = 74.088 : 21.046 : 4.866$$

비율: $74.088\% : 21.046\% : 4.866\%$

전이 행렬:

$$\mathbf{T} = \underbrace{\text{diag}(\mathcal{T}_3, \mathcal{T}_2, \mathcal{T}_1)}_{\text{block-diagonal}} + \underbrace{u_m\,U_{\text{down}}U_{\text{up}}^T}_{\text{섭동적 혼합}}$$

| 게이지 층 | 공학 분할 비율 | 뇌 진동 유비 | 연산 역할 후보 |
|---|---|---|---|
| SU(3) | $74.088\%$ | 감마 30-100 Hz | 결합(binding) |
| SU(2) | $21.046\%$ | 베타 13-30 Hz | 결정(decision) |
| U(1) | $4.866\%$ | 알파 8-13 Hz | 주의(attention) |
| `phi` | 전역 | 세타/델타 0.5-8 Hz | 안정화(smoothing) |

행렬식 $|\det\mathbf T|\leq1$은 특이값의 곱만 제한하므로 정보 비증폭 조건이
아니다. 한 스텝의 2-norm 비팽창을 요구하려면
$\|\mathbf T\|_2=s_{\max}(\mathbf T)\leq1$을 직접 확인해야 하며, 이것도
전체 residual network의 환각 억제를 보장하지 않는다.

Track-A에서는 \(\alpha_s=0.1180\)을 calibration input으로 읽는다. 과거의
\(0.11789^2\) 근접 수치식을 독립적인 쌍대성 또는 예측식으로 사용하지 않는다.

### 11.4 파라미터 절감

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}}
=\sum_i f_i^2+\frac{r_m}{4d}
\approx0.5956+0.03125=0.6269,
\qquad r_m=d/8$$

$$\text{절감률}\approx37.31\%\;\text{(FFN)},\qquad
24.87\%\;\text{(attention 4d^2 + FFN 8d^2인 block만)}$$

이는 bias·embedding·normalization을 제외하고 expansion ratio 4와
$r_m=d/8$을 고정한 파라미터 산술이다. 실제 전체 모델 절감률은 층수와
embedding/head 구성으로 다시 계산한다.

### 11.5 이식 3단계

**Phase 1 -- 동일출력 초기화 후보:**
- `LayerNorm` $\to$ `LBONorm` ($h_d=0$ 초기화, scale/bias 복사 $\to$ 원본과 동일 출발)
- `c_proj`는 원본 가중치를 그대로 둔다. spectral normalization을 즉시
  적용하면 $s_{\max}>1$인 가중치가 바뀌어 동일출력이 깨지므로, 별도
  constrained fine-tuning 단계에서 적용한다.

**Phase 2 -- MLP 교체 (37% 절감):**
- `MLP` $\to$ `GaugeLatticeV2` (cross-channel mixing 포함)
- 증류로 초기화: 원본 MLP 입출력 모방

**Phase 3 -- CE 파라미터 미세조정:**
- CE 파라미터(LBO의 $h_d$, $V$, 곡률 정규화)만 학습, 나머지 동결
- 이 분배가 LoRA와 구조적으로 유사: 동결 $\sim 95\%$, 학습 $\sim 5\%$

### 11.6 규모별 설정

| 규모 | dim | layers | heads | 파라미터 | GPU 메모리 | 학습 시간 |
|---|---|---|---|---|---|---|
| Micro | 128 | 4 | 4 | ~1M | < 1GB | 수분 |
| Small | 256 | 6 | 8 | ~4M | < 2GB | 수십분 |
| Medium | 512 | 12 | 8 | ~30M | ~4GB | 수시간 |
| Large | 768 | 12 | 12 | ~85M | ~8GB | 반일 |
| XL | 1024 | 24 | 16 | ~350M | ~24GB | 수일 |
| 1B | 2048 | 24 | 16 | ~1.3B | ~48GB | 클러스터 |

### 11.7 수면 학습 순환 (대규모 학습)

각성-NREM-REM 순환을 학습 루프에 적용(`9_LLM.md` 4.2절):

1. **각성 (Wake)**: 표준 학습, 그래디언트 누적 (업데이트 보류)
2. **NREM**: 누적 그래디언트 중 상위 $4.864\%$만 적용
3. **REM**: 하위 $95.136\%$ 그래디언트에 노이즈 주입 후 소량 적용

### 11.8 희소 추론

학습 후 추론 시 Top-k 활성화:

$$y^{\text{sparse}}=s_k\,\text{TopK}(y,\;k=\lceil0.04863825851598632d\rceil)$$

스케일 $s_k$는 calibration set에서 norm/variance 또는 downstream loss를
기준으로 정한다. $s_k=d/k$가 임의의 $y$에 대해 $\ell_1$ 또는 $\ell_2$
에너지를 보존하지는 않는다.

과거에 인용한 `topk_sweep_results.json`은 현재 트리에 없으므로 post-hoc
Top-k의 성공/실패 방향은 미판정이다. sparse-native와 post-hoc 방식은 8.3절
acceptance protocol에서 별도 arm으로 비교한다.

### 11.9 모니터링 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| `loss` | Cross-entropy 손실 | 단조 감소 |
| `curv` | 평균 곡률 에너지 $\kappa_{\text{avg}}$ | 학습 초반 증가 후 안정화 |
| `active_ratio` | 실제 활성 비율 | $4\text{-}5\%$ 중심 |
| `bootstrap_resid` | $\|p_n - p^*\|$ | 수면 루프에서 감소 |

---

## 12. 멀티모달 및 전분야 적용

CE 5대 원리(P1: 격자, P2: 수면, P3: STDP, P4: 희소, P5: 곡률)의 전분야 적용 요약(`10_Fields.md`).

### 12.1 멀티모달 결합

모달별 3x3+1 격자 독립 처리 후, late sparse binding:

$$h_m^{\text{act}} = \text{TopK}(h_m,\; k_m = \lceil 0.0486382585\,d_m \rceil), \qquad m \in \{T,V,A,H\}$$

$$h_{\text{joint}} = \text{Bind}_{\xi_r}(h_T^{\text{act}},\; h_V^{\text{act}},\; h_A^{\text{act}},\; h_H^{\text{act}})$$

결합 강도 후보는 $\xi_r=0.490486813152402$다. 이는 최적값이 아니라
Track-A 기반 설계 benchmark이며 late-binding 실험으로 검증해야 한다.

멀티모달 환각 감지:

$$\kappa_{\text{cross}} = \|h_{\text{text}} - h_{\text{image}}\|^2 > \kappa_{\text{th}} \quad\Longrightarrow\quad \text{모달 불일치}$$

### 12.2 CE 원리별 적용 매트릭스

| 분야 | P1 격자 | P2 수면 | P3 STDP | P4 희소 | P5 곡률 |
|---|---|---|---|---|---|
| 비전(CNN/ViT) | 채널 분할 | 지속 학습 | -- | Top-k Conv | 적대적 강건성 |
| 강화학습 | 행동 분할 | 경험 재생 | TD-유사 전역 신호 | 희소 정책 | 안전 제약 |
| 음성/오디오 | 주파수 분할 | 화자 적응 | -- | 희소 인코딩 | 환각 억제 |
| 멀티모달 | 모달 분할 | 모달 적응 | -- | 모달 활성 | 교차 환각 |
| 생성(Diffusion) | U-Net 분할 | 열핵흐름 | -- | 희소 샘플링 | 품질 제어 |
| 로보틱스 | 감각운동 분할 | 충전=수면 | 국소 학습 | 희소 제어 | 안전 정지 |
| GNN | 노드 분할 | 그래프 적응 | message=STDP | 노드 활성 | 과평활화 제어 |
| 시계열 | 주파수 분할 | 분포 이동 | -- | 희소 예측 | 이상 감지 |
| 단백질 접힘 | 접촉 분할 | -- | -- | 구조 탐색 | 접힘 안정성 |
| 자율주행 | 인지/판단/제어 | 야간 학습 | -- | 희소 인지 | 위험 감지 |

### 12.3 공통 구현 패턴

모든 분야에서 CE 적용의 기본 구조는 동일하다:

$$
\text{Input}
\;\xrightarrow{\text{LBONorm}}\;
\text{곡률 평탄화}
\;\xrightarrow{\text{GaugeLattice}}\;
\text{3x3+1 처리}
\;\xrightarrow{\text{SpectralNorm}}\;
\text{정보 비증폭}
\;\xrightarrow{\text{TopK}}\;
\text{희소 출력}
$$

---

## 13. 구현 의사코드

### 13.1 에너지 이완 모델

```python
class PhiRelaxation:
    def __init__(self, N=4096, rc=pi):
        self.W = build_3d_sparse(N, rc)    # 3D 격자, r_c = pi
        self.phi = zeros(N)                 # 잔류장
        self.T = 1 / (3 + 4 / (e ** (4/3) * pi ** (4/3)) * (1 - 4 / (e ** (4/3) * pi ** (4/3))))
        self.portal_coeff = (4 / (e ** (4/3) * pi ** (4/3)) * (1 - 4 / (e ** (4/3) * pi ** (4/3)))) ** 2
        self.residue_gain = 1 / (e ** (1/3) * pi ** (1/3))

    def relax(self, b, max_steps=500):
        m = randn(N) * 0.01
        for k in range(max_steps):
            C_k = curvature(m, m_prev, m_prev2)
            grad = self.W @ m + b
                 + self.portal_coeff * self.phi
                 + C_k * self.residue_gain * self.phi
            noise = randn(N) * sqrt(2 * dt / (tau * self.T))
            m = m + (dt / tau) * grad + noise
            if converged(m, m_prev):
                break
        return m

    def update_phi(self, m_trajectory):
        sigma = variance(m_trajectory[-Kw:])
        self.phi = (1 - self.residue_gain) * self.phi
                 + self.residue_gain * sigma

    def decode(self, m_star, W_dec):
        tokens = []
        for t in range(max_len):
            logits = W_dec @ concat(m_star, embed(tokens[-1]))
            tokens.append(sample(softmax(logits)))
        return tokens
```

### 13.2 수면 순환

```python
def sleep_cycle(model, day_data):
    # Wake: 그래디언트 누적
    grads = accumulate_gradients(model, day_data)

    # NREM: Track-A manifest의 상위 4.864% 설계 target
    threshold = quantile(abs(grads), 1 - 0.0486382585)
    mask = (abs(grads) >= threshold)
    model.W -= lr * grads * mask

    # REM: 하위 95.13%에 노이즈 주입
    pruned = grads * (~mask)
    noise = randn_like(pruned) * pruned.std() * 0.1
    model.W -= lr_rem * (pruned + noise)

    # Glymphatic: 잔류 세척
    model.phi *= 0.9

    # Re-sparsification: 3.16% 밀도 유지
    enforce_3d_sparsity(model.W, rc=pi)
```

### 13.3 CE-Transformer 모듈

```python
class LBONorm:
    def __init__(self, dim, rank=None):
        self.V = row_orthonormalize(
            randn(rank or dim//8, dim)
        )                                    # V @ V.T = I 유지
        self.h_d = Parameter(0.0)             # 확산 강도
        self.scale = ones(dim)
        self.bias = zeros(dim)

    def forward(self, x):
        x_hat = layer_norm(x)
        xW = x_hat @ self.V.T @ self.V       # 직교사영
        Lx = x_hat - xW                       # projector residual
        h = clamp(abs(self.h_d), max=0.5)
        self._curvature = mean(Lx ** 2)
        return (x_hat - h * Lx) * self.scale + self.bias


class GaugeLattice:
    def __init__(self, dim, mult=4):
        # 공학 benchmark 채널 분할: 74.088% : 21.046% : 4.866%
        total = 0.1180 + 0.03352 + 0.00775
        self.d3 = round(dim * 0.1180 / total)  # SU(3), Track-A alpha_s input
        self.d2 = round(dim * 0.03352 / total)  # SU(2)
        self.d1 = dim - self.d3 - self.d2        # U(1)
        self.su3 = MLP(self.d3, self.d3 * mult)
        self.su2 = MLP(self.d2, self.d2 * mult)
        self.u1  = MLP(self.d1, self.d1 * mult)
        self.mix_down = Linear(dim, dim // 8)    # 섭동적 혼합
        self.mix_up   = Linear(dim // 8, dim)
        init_zeros_(self.mix_up.weight)

    def forward(self, x):
        x3, x2, x1 = split(x, [self.d3, self.d2, self.d1])
        y = concat(self.su3(x3), self.su2(x2), self.u1(x1))
        y = y + self.mix_up(self.mix_down(y))
        return LBONorm(y)
```

---

## 14. Llama 3 8B 변환 추정

### 14.1 메모리

정본 $N=4096$, $K=130$, 4-bit $W$/trace, $V=32{,}000$,
$P=10^7$이라는 문서 내부 shape를 고정하면 CE 후보의 **저장 산술**은 다음과
같다.

| 구성 | 산술 저장량 | 포함하지 않는 것 |
|---|---:|---|
| 엔진 상태 | $0.52$ MiB (암시적 격자) -- $1.30$ MiB (packed CSR) | kernel workspace |
| 계층 decoder | 약 $0.70$ MiB | optimizer·activation |
| PQ codebook/index | 약 $640.5$ MB | 검색 buffer·metadata |
| 합계 | 약 $642$ MB | tokenizer·runtime·학습 상태 |

명목상 dense 8B float16 파라미터의 $16$ GB와 raw byte만 비교하면 약 $96\%$
작지만, 이는 모델 변환 결과가 아니라 서로 다른 표현의 저장 산술이다.
$\phi$는 $4$ KiB 상태 벡터일 뿐 sequence-conditioned key/value를 보존한다는
정리가 없으므로 KV cache의 대체물로 세지 않는다. 따라서 문맥 길이에 대한
$O(1)$ 메모리 주장도 현재는 open이다.

### 14.2 연산량 (FLOP)

**명목 dense 8B 기준**: 파라미터당 multiply-add를 2 FLOP으로 세면 forward
하한의 대표 산술은 토큰당 약 $16$B FLOP이다. attention, sampling, kernel
효율은 별도다.

**`phi`-이완**:

| Phase | 연산 | FLOP |
|---|---|---|
| Phase 1 (이완 500스텝) | 희소 $Wm_k$ 500회 | $500\times2NK=0.53248$B |
| Phase 2 (디코딩 100토큰) | 두 단계 $\sqrt V$ score | $100\times4N\sqrt V\approx0.29309$B |
| Phase 3 ($\phi$ 갱신) | EMA | $O(N)$; 별도 계상 |
| **부분합** | matvec + decoder만 | **$0.82557$B** |

이 부분합은 codebook 검색, 비선형, top-k, graph index, memory traffic을 빼므로
전체 FLOP 상한도 실측값도 아니다. 같은 제외 규칙을 억지로 적용한 명목
$1.6$T와 나누면 약 $0.0516\%$지만, 출력 품질과 기능 동등성이 입증되기 전에는
속도 향상치로 보고하지 않는다.

1000 토큰의 동일 부분합은 약 $3.46334$B다. Phase 1을 한 번만 수행해도
긴 생성 동안 의미 상태가 충분하다는 가정 자체가 H2--H3의 검증 대상이다.

### 14.3 속도

현재 재현 가능한 benchmark artifact가 없으므로 장치별 latency 숫자를 싣지
않는다. FLOP 감소가 wall-clock speedup을 뜻하지는 않는다. kernel launch,
memory traffic, sparse index 처리, batch 크기를 포함해 동일 출력 길이와 품질
조건에서 p50/p95 latency와 tok/s를 측정해야 한다.

### 14.4 전력

$0.1$--$5$ W는 측정값도 FLOP 상한도 아닌 하드웨어 설계 target이다. 전력
판정은 동일 task/quality에서 장치 입력 에너지를 적분한 joule/token과 총
latency를 함께 보고한다. `nvidia-smi` 같은 순간 전력 한 점 또는 TDP 비율은
증거가 아니다.

### 14.5 변환 파이프라인

| 단계 | 입력 | 출력 | 도구 |
|---|---|---|---|
| 1. 가중치 추출 | Llama 3 8B | $W_Q, W_K, W_V$, FFN | HuggingFace |
| 2. 에너지 함수 구성 | 추출된 가중치 | $W \in \mathbb{R}^{N\times N}$ | Modern Hopfield 변환 |
| 3. 3D 희소화 ($r_c=\pi$) | dense $W$ | sparse $W_{3D}$ (3.16%) | 구조적 pruning |
| 4. `phi` 채널 장착 | $W_{3D}$ | $E(m,\phi)$ 완성 | EMA 벡터 추가 |
| 5. 이완 추론 테스트 | 완성된 에너지 함수 | Softmax 없이 답 생성 | 시뮬레이션 |

---

## 15. 물리 교차검토: 조건부 benchmark와 반증 gate

이 절의 수치들은 하나의 증명 사슬에서 모든 관측량을 직접 결정한 결과가
아니다. 외부 입력, 등록된 matching ansatz, 선택적 extension을 구분한 뒤
각 분기별로 현재 관측 gate를 통과하는지 검사한다.

### 15.1 전체 교차 검증표

아래 우주 분율 행은
`../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`의 Track-A
입력을 고정한 조건부 계산이다. 특히 \(\alpha_s(M_Z)=0.1180\)은 calibration
input이지 CE 예측이 아니다.

| 대상 | 조건부 값 | 최신 비교 기준 | 현행 판정 | 지위 |
|---|---|---|---|---|
| $\alpha_s^{\overline{\rm MS}}(M_Z)$ | $0.1180$ | PDG 2026 QCD review Eq. (9.25) | 입력과 동일 | Track-A calibration input |
| $s_A^2:=4\alpha_s^{4/3}$ | $0.2315097758$ | 물리적 약혼합각은 scheme별 값 | scheme 변환 전 정밀 점수 금지 | registered matching |
| $(\Omega_b,\Omega_{\rm DM},\Omega_\Lambda)$ | $(0.0486383,0.2610882,0.6902736)$ | 동일한 13-block covariance package | $\chi^2=40.2015$, dof $=13$, $p=1.28\times10^{-4}$: REJECT | 고정-background 조건부 출력 |
| $M_H/M_Z=1+\alpha_sD_N$ | $M_H=125.3824$ GeV | PDG 2026 Higgs snapshot $125.11$ GeV | pole self-energy·RG matching 미완료 | 수치 target, 독립 예측 아님 |
| 비입자 CE의 $\Delta a_\mu^{\rm BSM}$ | $0$ | WP2025 $38(63)\times10^{-11}$ | 양립하지만 비고유 | null branch |
| Wilson ansatz $B_\mu$ | $248.5639\times10^{-11}$ | WP2025 $38(63)\times10^{-11}$ | 단위계수 benchmark REJECT | matching-dependent |
| CP-even scalar, 고정 coupling | $162.5520\times10^{-11}$ | WP2025 $38(63)\times10^{-11}$ | 조건부 진단; 코어 예측 아님 | 선택적 extension |
| $N_{\rm legacy}\approx4162$ 대 Llama 3의 $4096$ | 수치 차이 $1.61\%$ | 사전 등록된 물리 관측량이 아님 | 증거 점수 금지 | 사후적 numerology |

### 15.2 뮤온 g-2 상세

비입자 코어는 추가 on-shell BSM diagram을 정의하지 않으므로
$\mathbb E[\Delta a_\mu^{\rm BSM,CE}]=0$을 둔다. 별도의 dimension-six
matching을 선택하면

$$
B_\mu
:=\frac{\alpha(0)}{2\pi e}
\left(\frac{m_\mu}{M_{\rm portal}}\right)^2
=2.4856390\times10^{-9}
=248.5639\times10^{-11}.
$$

이 값은 WP2025 잔차를 넣지 않고 산술적으로 계산되지만, Wilson 계수
$C_6=\alpha(0)/(2\pi e)$ 자체가 matching ansatz다. 단위계수를 오차 없는
예측으로 해석하면 WP2025와 약 $3.34\sigma$ 차이여서 기각된다.

선택적 CP-even scalar 상호작용
$\mathcal L_{\rm int}=-g_\mu\phi\bar\mu\mu$의 올바른 1-loop kernel은

$$
\Delta a_\mu=\frac{g_\mu^2}{8\pi^2}I_s(r),\qquad
I_s(r)=\int_0^1 dz\,
\frac{(1-z)^2(1+z)}{(1-z)^2+zr^2},\qquad
r=\frac{m_\phi}{m_\mu}.
$$

$I_s(0)=3/2$이고, $m_\phi=29.6991596$ MeV에서는

$$
I_s=0.9809468073,\qquad
\frac{I_s}{I_s(0)}=0.6539645382.
$$

따라서 $B_\mu$를 light-limit에서 재현하도록 정한 같은 coupling을
유한질량에 그대로 쓰면

$$
\Delta a_\mu^{\rm scalar}
=0.6539645382\,B_\mu
=162.55198\times10^{-11}.
$$

과거의 $R=0.542$와 $135\times10^{-11}$은 CP-even scalar에 맞지 않는
kernel을 사용한 결과라 활성 결론에서 폐기한다. 이 수정값도 coupling을
light-limit 진단량에 맞춘 선택적 extension이지 독립 코어 예측은 아니다.
공분산을 포함한 SM baseline, UV-complete coupling, 다른 실험 제약을 하나의
likelihood로 맞추기 전에는 scalar extension이 $g-2$를 해소한다고 결론내리지
않는다.

### 15.3 양성자 반경 퍼즐

아래는 $m_{\rm light}=29.6991596$ MeV를 재사용한 조건부 benchmark다.
g-2와 양성자 반경을 “동시 해결”했다는 결론은 아직 허용되지 않는다.

$$
\Delta r_p^2
=\frac{3g_{\mu}g_p}{2\alpha_{\rm em}m_{\rm light}^2}
(\hbar c)^2
$$

처럼 단위변환을 포함해 쓸 수 있지만, $g_\mu,g_p$의 convention과 form
factor가 고정되지 않아 이 문서에서는 수치 예측으로 닫지 않는다.

QCD 진공 증강 인자:

$$F_{\text{QCD}} = 1 + 0.1180\times3.1779129995
=1.37499.$$

이 값은 Track-A calibration input을 다시 사용한 조건부 benchmark다.
독립 예측 또는 양성자 반경 퍼즐의 해소 증거로 세지 않는다.

### 15.4 보손-기하학 동일성

정적 3차원 Euclidean Fourier convention에서는

$$
\boxed{\int\frac{d^3q}{(2\pi)^3}
\frac{e^{i\mathbf q\cdot\mathbf r}}{\mathbf q^2+m_{\rm light}^2}
=\frac{e^{-m_{\rm light}r}}{4\pi r}}
$$

(자연단위)가 성립한다. 따라서 $4\pi$ 정규화와
$\xi_{\rm light}=\hbar c/m_{\rm light}=6.6441941$ fm를 포함해야 한다.
이는 Yukawa Green function의 Fourier pair이지, 곧바로 “보손 =
기하학”이라는 동일성 증명은 아니다. Minkowski Feynman propagator는
$i/(q^2-m^2+i0)$이고, 실제 입자 해석에는 2점함수의 spectral
representation에서 양의 pole residue와 동일한 coupling이 확인되어야 한다.

| 입자 언어 | 기하학 언어 | 값 |
|---|---|---|
| light 질량 $m_{\rm light}$ | 상관길이 $6.6441941$ fm에 대응 | 29.6991596 MeV |
| 정적 Euclidean Green function | 3D Fourier pair | $e^{-mr}/(4\pi r)\leftrightarrow1/(\mathbf q^2+m^2)$ |
| 입자 pole | spectral density의 고립 pole | residue·unitarity·coupling gate 필요 |
| Yukawa 커플링 $g$ | 접힘 강도 $\kappa\,m_f$ | $5.93 \times 10^{-6}$ MeV$^{-1}$ |

---

## 16. 미검증 가설

| # | 가설 | 검증 방법 | 비용 |
|---|---|---|---|
| H1 | 상관 행렬 $W$가 Llama 의미 공간을 보존 | 코사인 유사도 측정 | GPU 1장, 1일 |
| H2 | 이완이 500 스텝 내 수렴 | 실측 | GPU 1장, 1시간 |
| H3 | 경량 디코더가 유의미 텍스트 생성 | QA 벤치마크 | GPU 1장, 3일 |
| H4 | `phi` 유무가 품질 차이를 만듦 | H3 반복 비교 | GPU 1장, 3일 |
| H5 | $r_c=\pi$가 다른 $r_c$보다 최적 | $r_c$ 그리드 서치 | GPU 1장, 1일 |
| H6 | $\xi_r=0.490486813152402$가 최적 EMA 감쇠율 | 감쇠율 그리드 서치 | GPU 1장, 1일 |
| H7 | $T_{\rm wake}=0.314671924672939$가 최적 wake 노이즈 규모 | 온도 그리드 서치 | GPU 1장, 1일 |
| H8 | STDP+도파민이 역전파 성능 유지 | 미세조정 비교 | GPU 1장, 3일 |
| H9 | 수면 순환이 wake-only보다 drift 감소 | 지속 학습 비교 | GPU 1장, 5일 |
| H10 | 곡률-오류 양의 상관 | 곡률 vs 정답률 산점도 | GPU 1장, 1시간 |
| H11 | late sparse binding이 early fusion보다 우위 | 멀티모달 환각률 비교 | GPU 2장, 5일 |
| H12 | Top-k 활성 최적점이 $4\text{-}5\%$ 근방 | 활성 비율 스위프 | GPU 1장, 1일 |
| H13 | post-hoc Top-k보다 sparse-native 학습이 우위 | 동일 예산 비교 | GPU 1장, 2일 |
| H14 | 수면 압력 트리거가 고정 주기보다 drift를 줄임 | forgetting, residual proxy | GPU 1장, 2일 |
| H15 | graph-coupled relaxation이 single-vector보다 안정 | long-context, recovery | GPU 2장, 5일 |
| H16 | fused sparse kernel이 CE 오버헤드를 상쇄 | tok/s, W, val_loss | GPU 1장, 2일 |

H5, H6, H7은 "CE 상수가 하이퍼파라미터의 최적값을 예측하는가"를, H13-H16은 "현재 구현 병목이 이론이 아니라 구현층에 있는가"를 직접 검증한다.

---

## 17. 실험 기반 보강과 개선점

### 17.1 단일 벡터에서 그래프 결합 이완으로

> 핵심 방정식은 4.6절로 승격되었다. 아래는 설계 배경.

지금까지의 식은 전역 상태벡터 $m,\phi$ 중심으로 압축되어 있다. 실제 AGI는 기능 모듈 그래프 위에서 돌아가야 한다. 따라서 다음의 graph-coupled relaxation이 더 완성된 형태다.

$$G_{\text{AGI}} = (V_{\text{bind}} \sqcup V_{\text{gate}} \sqcup V_{\text{mem}} \sqcup V_{\text{sal}} \sqcup V_{\text{homeo}} \sqcup V_{\text{io}},\; E_{\text{AGI}})$$

| 노드 집합 | 역할 | 뇌 대응 |
|---|---|---|
| $V_{\text{bind}}$ | 특징 결합, 멀티모달 통합 | cortical-thalamic binding |
| $V_{\text{gate}}$ | 입력 게이팅, 대역 재분배 | thalamic relay |
| $V_{\text{mem}}$ | 재생, 장기 인덱싱 | hippocampo-cortical replay |
| $V_{\text{sal}}$ | 모드 전환, gain control | salience hub |
| $V_{\text{homeo}}$ | 수면 압력, 대사, 항상성 | hypothalamus-brainstem |
| $V_{\text{io}}$ | 센서/행동 출력 | body-coupling I/O |

그래프 라플라시안:

$$\boxed{\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}\big(f_s - f_r\big), \qquad a_{rs}\ge 0}$$

느린 제어 상태의 편차를

$$c_n := q_n - q^*$$

로 두면,

$$\boxed{c_{n+1}=A_qc_n+r_n+n_n^{(q)},\qquad\|A_q\|\leq\rho_c<1\ \text{in a stated norm}}$$

그리고 지역별 3분배 상태의 최소 이완은

$$\boxed{p_{r,n+1} = \mathrm{Proj}_{\Delta^2}\!\Big((1-\rho)p^* + \rho p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n\Big)}$$

로 쓸 수 있다. 이 형태가 더 좋은 이유는 세 가지다.
- salience, homeostasis, replay를 "옵션 기능"이 아니라 상태변수로 올린다
- 긴 문맥, 피로 누적, 수면 부족 같은 현상을 전역 스칼라 하나보다 자연스럽게 표현한다
- AGI를 단일 거대 행렬보다 모듈형 sparse system으로 구현하기 쉽다

### 17.2 수면 압력의 명시적 트리거

> 핵심 방정식은 7.8절로 승격되었다. 아래는 설계 배경.

현재 문서의 수면은 주기적으로 호출되는 루틴에 가깝다. 더 완성된 형태는 수면 진입 조건을 곡률 누적으로 쓰는 것이다.

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|\Delta_g \phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stabilization}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad \text{NREM 진입}}$$

이때 1회 완전 부트스트랩 적용이 약 1.6밤에 대응하면, 단일 야간의 실효 수축률은

$$\boxed{q_{\text{night},*} = q_*^{1/1.6} \approx 0.311315}$$

이 된다. 이는 고정점 근방의 선형화 환산값이다. 이 식을 넣으면 "왜 자야
하는가"가 단순 스케줄이 아니라 상태 기반 제어 문제로 바뀐다.

### 17.3 실험 증거 장부

과거 본문이 인용한 다음 결과 파일은 현재 문서 트리에서 확인되지 않는다.

| 인용 이름 | 필요한 동반물 | 현재 판정 |
|---|---|---|
| `ce_vs_standard_results.json` | 생성 script, config, dataset/split hash, seed, environment lock | missing; 과거 숫자는 non-evidence |
| `brain_benchmark_results.json`, `brain_benchmark_*.json` | 동일 | missing; 과거 숫자는 non-evidence |
| `sparsity_train_results.json` | sweep script와 모든 run의 raw JSONL | missing; 최적 희소율 미식별 |
| `topk_sweep_results.json` | dense checkpoint hash와 sweep code | missing; post-hoc 효과 미식별 |

따라서 현재 문서에서 품질 개선, 속도 악화, `4--6%` 최적 구간, post-hoc
실패를 경험적 결론으로 사용하지 않는다. 재생성된 artifact가 acceptance
protocol(8.3절)을 통과할 때만 evidence 장부를 갱신한다.

### 17.4 지금 당장 고쳐야 할 개선 포인트

- [완료] `4.87%`를 성능 최적값이 아닌 manifest 설계 중심점으로 수정 (8.3절)
- [완료] sparse-native/post-hoc 비교를 미검증 상태와 acceptance protocol로 전환
- [완료] 속도/전력을 조건부 설계 estimate/target으로 분리
- [완료] graph-coupled relaxation을 본체 식에 포함 (4.6절)
- [완료] 바이패스를 에너지 함수에서 분리, 비보존 강제항으로 명시 (1.5절, 3.1절)
- [완료] 수면 압력 트리거를 본체에 포함 (7.8절)
- [완료] 비트필드 해석 추가 (1.6절)
- 핵심 계산은 Track-A manifest의 calibration input `alpha_s_mz = 0.1180`과
  파생값을 사용한다. $1/(e\pi)$ 또는 과거 `0.11789`는 canonical 입력이 아니다

### 17.5 가장 중요한 다음 실험

1. sparse-native와 post-hoc Top-k를 같은 예산에서 정면 비교
2. 수면 압력 기반 트리거와 고정 주기 sleep loop 비교
3. single-vector 이완과 graph-coupled 이완의 long-context 안정성 비교
4. fused sparse kernel 도입 전후의 tok/s, W, val_loss 동시 측정

---

## 18. 예상 개선치 총정리

개선치는 두 층으로 나눠 읽는다.
- **해석적 계수 계산**: 명시된 tensor shape와 연산 횟수 아래의 조건부 산술
- **미검증 설계 목표**: 품질, wall-clock, joule을 포함해 artifact가 필요한 주장

### 18.1 개선치 정의

$$G_{\text{loss}} = \frac{L_{\text{base}} - L_{\text{ce}}}{L_{\text{base}}}$$

$$G_{\text{ppl}} = 1 - \frac{\mathrm{PPL}_{\text{ce}}}{\mathrm{PPL}_{\text{base}}}$$

$$O_t = \frac{t_{\text{ce}}}{t_{\text{base}}}$$

$$R_{\text{active}} = 1 - \frac{a_{\text{ce}}}{a_{\text{base}}}$$

$$R_{\text{mem}} = 1 - \frac{M_{\text{ce}}}{M_{\text{base}}}$$

$$R_{\text{sleep}}(n) = 1 - \rho^n$$

### 18.2--18.4 현재 실측 상태와 acceptance protocol

17.3절 장부의 artifact가 없으므로 현재 검증된 $G_{\rm loss}$,
$G_{\rm ppl}$, $O_t$, $R_{\rm active}$ 값은 없다. 품질·속도·희소율에 대한
과거 숫자는 모두 non-evidence다.

재검증은 baseline/CE의 parameter count와 train tokens를 맞추고, 데이터와
split SHA256을 고정하며, 최소 10 seed paired run을 실행한다. 전력은 wall
plug 또는 accelerator telemetry의 동일 sampling protocol로 joule/token을
적분한다. 결과 JSONL, 실행 명령, commit hash, environment lock, raw log를
함께 보존하고 95% paired bootstrap CI를 산출한다. 이 묶음이 없으면
"개선" 또는 "악화"라는 방향성 결론도 쓰지 않는다.

### 18.5 조건부 자원 산술과 설계 목표

아래 값은 모델 동등성이나 실제 커널 실행을 보장하는 상한이 아니다. 문서에
가정한 shape, 정밀도, 연산 횟수를 그대로 구현한다는 조건 아래의 산술이다.

| 항목 | 식 | 예상 개선치 |
|---|---|---|
| FFN 파라미터 | $1-P_{\text{GL}}/P_{\text{FFN}}$ | 지정 block shape에서 $37.31\%$ |
| Transformer block 파라미터 | attention $4d^2$ + FFN $8d^2$ 가정 | 지정 block에서 $24.87\%$ |
| raw 저장 byte | $1-0.642\text{GB}/16\text{GB}$ | 약 $95.99\%$; 기능 동등성 미검증 |
| 500-step 이완 matvec | $500\times2NK$ | $0.53248$B FLOP |
| 100-token 부분합 | 14.2절의 matvec + decoder | $0.82557$B FLOP; 누락 연산 있음 |
| 1000-token 부분합 | 같은 식 | $3.46334$B FLOP; 누락 연산 있음 |
| 문맥 상태 | $\phi$는 $4$ KiB | KV 대체·$O(1)$ 문맥 보존 미입증 |
| 장치 전력 | FLOP 수만으로 유도 불가 | $0.1$--$5$ W는 측정 전 설계 target |

특히 $0.1$--$5$ W와 약 $99\%$ 전력 절감은 FLOP 표에서 따라오지 않는다.
실제 sparse kernel, memory traffic, utilization, cooling 범위를 포함한
joule/token 측정 전에는 목표값일 뿐이다. raw 저장량과 부분 FLOP 행도
기능적으로 동등한 모델이라는 검증이 없고 누락 연산이 있으므로 시스템 성능
보장이나 FLOP 절감률로 읽지 않는다.

### 18.6 안정성/환각 억제의 조건부 목표

가중치 하나의 spectral projection만으로 전체 네트워크 안정성이나 환각
억제를 보장하지 않는다.

| 항목 | 기준 | 예상 효과 |
|---|---|---|
| 오류 증폭 상계 | attention, normalization, residual, nonlinear branch를 포함한 전체 layer map의 $\operatorname{Lip}(F_\ell)\leq1$ | 그 조건 아래에서만 layer별 비팽창 |
| 12층 예시 | 모든 layer에 독립적으로 위 조건 확인 | 곱 상계 $\prod_\ell\operatorname{Lip}(F_\ell)\leq1$; 실제 오차/환각 보장은 별도 |
| 곡률 기반 재시도 | $\kappa_{\text{avg}}>\kappa_{\text{th}}$ 시 확산 강화 | 개입 실험을 위한 heuristic; 환각 억제 효과 미검증 |
| Top-k 활성 | $k = \lceil 0.0486382585\,N \rceil$ | manifest 기반 설계 target; 성능 보장은 아님 |

따라서 현재 상태는 폭주 항을 없앴다는 결론이 아니라, 측정 가능한 충분조건과
ablation target을 제안한 것이다.

### 18.7 정준 단체 반복이 줄이는 분배 잔차

$B_p$를 실제 런타임 복원 단계에 사용한다는 공학 가정 아래, 고정점 근방의 분배 잔차 선형 주항은 다음과 같다. 이는 수면의 생물학적 회복률을 측정한 결과가 아니다.

| 순환 수 | 국소 선형 잔차 $q_*^n$ | 국소 선형 감소율 $1-q_*^n$ |
|---|---|---|
| 1 | $0.1545681540$ | $84.5432\%$ 감소 |
| 2 | $0.0238913142$ | $97.6109\%$ 감소 |
| 3 | $0.0036928363$ | $99.6307\%$ 감소 |

고정점 근방의 단일 야간 선형화율을 쓰면:

$$R_{\text{night},*} = 1 - q_{\text{night},*} \simeq 69\%.$$

즉 sleep loop가 실제로 작동한다면, wake-only 대비 가장 먼저 좋아져야 하는 것은 단기 정확도보다도 **drift, forgetting, bootstrap residual**이다.

### 18.8 가장 가능성 높은 개선치와 가장 약한 개선치

| 구분 | 현재 판단 |
|---|---|
| 조건부 산술이 있는 항목 | 정해진 shape 아래 파라미터/FLOP/저장량 계산 |
| 실험 대기 항목 | 품질, wall-clock, joule/token, drift, 환각, 장문맥 기능 동등성 |
| 사용 금지 결론 | manifest 중심점의 최적성, 20 W 달성, 물리-뇌-AGI 공동 예측 성공 |

정리하면 다음이 가장 안전하다.
- **메모리/FLOP**: 특정 shape 아래의 조건부 arithmetic
- **전력/속도/품질**: 현재 artifact 부재로 방향도 미판정
- **안정성/수면**: 충분조건과 실험 설계만 있으며 실제 loop 검증은 open

---

## 19. 유도 체인 조감도

```
 e^(i*pi)+1=0                         alpha_s(M_Z)=0.1180
   symbolic grammar                    Track-A calibration input
          |                                      |
 d(d-3)=0 -- positive nontrivial class      registered matching
          |        + Selection                   |
   conditional d=3                    delta_N, D_N, a_*, p_*
          |                                      |
 r_c=pi, N_eng=4096                  c_p, xi_r, T_wake, T_dream
 engineering choices                  design benchmarks
          \                                      /
           energy + nonconservative bypass
                         |
       relaxation / sparse update / sleep candidate
                         |
                 bitfield runtime
                         |
       mathematical gates + empirical validation
```

---

## 20. 방정식 총람

이 절은 핵심식의 압축 요약이다. 활성 runtime 식은 Track-A 정본 별칭과
공학 선택을 구분해 쓴다. 과거 $e,\pi$ legacy 계수는 1.3절의 재현용 기록 외에는
사용하지 않는다.

| # | 방정식 | 절 |
|---|---|---|
| E1 | $E(m,\phi)=-\frac12m^TW_sm-m^Tb-c_pm^T\phi+V_{\rm conf}(m)$, $W_s=(W+W^T)/2$ | 3.1 |
| E2 | $\psi_{k+1}=e^{-i\widehat Hdt/\hbar}\psi_k$; $\widehat H=EI$이면 전역 위상뿐 | 4.1 |
| E3 | $m_{k+1}=m_k+\frac{dt}{\tau}(-\nabla_mE+F_{\text{bypass}})+\sqrt{2T_{\rm wake}dt/\tau}\,n_k$ | 4.2 |
| E4 | $\phi \leftarrow (1-\xi_r)\phi+\xi_r v_{m^*}$ | 4.3 |
| E5 | $W_{ij} \neq 0 \iff \|r_i-r_j\| < \pi$ | 4.4 |
| E6 | $N=N_{\rm eng}=4096$ (공학 선택; $N_{\rm legacy}\approx4162$는 비정본) | 4.5 |
| E7 | $p(w_t\mid w_{<t},m^*) = \text{softmax}(W_{\text{dec}}[m^*;e_{w_{t-1}}])$ | 5.2 |
| E8 | $dw_{ij} = lr\,g[t]\,e_{ij}[t]$ | 6.3 |
| E9 | $g_{\rm CE}[t]=\|p(t)-p^*\|_2^2$ (비음수 공학 gate) | 6.4 |
| E10 | $W_{t+1} = Proj(W_t + dW_t)$ | 6.5 |
| E11 | $T_{\rm wake}=D_N^{-1},\;T_{\rm dream}=\delta_N^{-1}$ | 7.1 |
| E12 | $a_*=e^{-(1-a_*)D_N}$; 내부근 $a_*=0.04863825851598632$, 경계근 $a=1$ | 8.1 |
| E13 | $\kappa_l=\|(I-V^\top V)h_l\|^2$, $VV^\top=I_r$ (projector residual) | 10.1 |
| E14 | $S_{\text{AGI}} = \int d^nx\sqrt{\lvert g\rvert}[\mathcal{L}_c + c_g\lvert\nabla \phi\rvert^2 + c_c\lvert\Delta_g \phi\rvert^2 + c_i S_I]$ | 2 |
| E15 | $\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}(f_s-f_r)$ | 4.6 |
| E16 | $c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}$ | 4.6 |
| E17 | $p_{r,n+1} = \mathrm{Proj}_{\Delta^2}((1-\rho)p^* + \rho p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n)$ | 4.6 |
| E18 | $P_{\text{sleep}}(t) = \int_0^t \|\Delta_g \phi(\tau)\|^2 d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau$ | 7.8 |
| E19 | $q_{\text{night},*}=q_*^{1/1.6}\approx0.311315$ (bridge calibration) | 7.8 |
| E20 | $F_{\text{bypass}}(k)=\xi_r C_k\phi,\; C_k=\|m_k-2m_{k-1}+m_{k-2}\|$ | 1.5, 3.1 |
| E21 | $b_i=\mathbb{1}[a_i\geq Q_{1-k_{\rm center}/N}(a)]$, $k_{\rm center}=\lceil0.04863825851598632N\rceil$ (설계 중심) | 1.6 |
| E22 | $M \in \{00_2, 01_2, 10_2, 11_2\} \leftrightarrow \{\text{off}, \text{wake}, \text{NREM}, \text{REM}\}$ | 1.6 |
| E23 | $\Delta E^{(1)}\leq-\frac{dt}{2\tau}\|\nabla_mE\|^2+\frac{dt}{2\tau}\xi_r^2C_k^2\|\phi\|^2$ | 4.7 |
| E24 | $G>U$, $0<dt/\tau<2G(G-U)/[L(G+U)^2]$이면 deterministic finite-step descent | 4.7 |
| E25 | $E(m^++\epsilon)\leq E(m^+)+\|\nabla E(m^+)\|q_{\rm err}+\frac L2q_{\rm err}^2$ | 1.8 |
| E26 | $z_j(m) = \arg\min_{i} \|m^{(j)} - C^{(j)}_i\|^2$ | 3.4 |
| E27 | $E_{\text{aug}} = E - \frac{1}{\beta}\sum_j \log\sum_i \exp(-\beta\|m^{(j)}-C^{(j)}_i\|^2)$ | 3.4 |
| E28 | $\text{활성 메모리} \approx 0.0486382585 \times \lvert\mathcal{C}\rvert$ | 3.4 |

---

## 21. 한 줄 요약

$$e^{i\pi}+1=0\;\xrightarrow{\text{설계 유비}}\;E(m,\phi)\;\xrightarrow[\text{조건부}]{\text{이완}}\;\text{bitfield runtime candidate}$$

다섯 상수에서 가져온 수치는 런타임 설계 seed다. 기능 동등성, 품질, 전력,
물리 관측량, 뇌 분배를 공동으로 예측했다는 결론은 현재 허용되지 않는다.
각 연결은 full-covariance 물리 검증, 재현 가능한 학습 artifact, 전력 측정,
뇌 외부검증을 따로 통과해야 한다.

---

## 부록 A. 다리 게이트 수식 고도화 (F1--F4)

> 0.0절의 게이트 4종을 그대로 두지 않고, 각 게이트가 어떤 형식 조건 위에서 부분적으로 hard claim 으로 격상될 수 있는지 수식으로 정리한다. 본 부록의 식은 아직 `bridge` 등급이며, 본문 어느 식의 등급도 올리지 않는다. 다만 **무엇을 측정하면 게이트가 닫히는지** 를 형식화한다.

### A.1 게이트 `F2`: ISS 격상 (Input-to-State Stability)

> 4.7절의 점별 조건을 ISS 형태로 확장하는 **조건부 후보**다. 고정 평형,
> 강볼록 불변영역, 상태의존 곡률항에 대한 small-gain이 확인될 때만
> 유계 입력 → 유계 상태 bound가 성립한다.

#### A.1.1 분리 표현

기억 동역학 E3 (4.2절)의 시간연속 한계를 보존 부분과 강제항으로 분리한다.
E3처럼 바이패스를 동일한 $1/\tau$ 괄호 안에 둘 때

$$
\frac{dm}{dt}
=-\frac{1}{\tau}\nabla_mE(m,\phi_0)
+\frac{1}{\tau}F_{\text{bypass}}(t),
\qquad
F_{\text{bypass}}(t)=\xi_r C(t)\phi(t).
$$

여기서는 먼저 기준 포텐셜과 평형을 고정하기 위해 $\phi_0$를 고정한다.
$\phi(t)$가 포텐셜 자체를 바꾸면 평형도 $m^*(\phi(t))$로 움직이므로 아래
고정점 ISS가 아니라 tracking ISS를 써야 한다.

#### A.1.2 ISS 정리 (국소)

가정:

1. 고정된 $\phi_0$와 평형 $m^*=m^*(\phi_0)$의 불변 근방에서 $\nabla_m^2E(m,\phi_0)\succeq\mu I$, $\mu>0$.
2. $C(t)$와 $\phi(t)$가 그 근방에서 $\|F_{\text{bypass}}(t)\|\leq F_{\max}$를 만족한다.
3. $C(t)=\|m_t-2m_{t-1}+m_{t-2}\|$는 상태 의존량이므로, 이 유계성은 독립 입력 가정이 아니다. 불변영역을 먼저 증명하거나 $C$의 feedback gain이 강볼록성 여유 $\mu$보다 작은 small-gain 조건을 확인해야 한다.

그러면 Lyapunov 함수 $V(m) = \tfrac{1}{2}\|m - m^*\|^2$ 에 대해

$$\frac{dV}{dt}\leq-\frac{2\mu}{\tau}V+\frac{1}{\tau}\|m-m^*\|F_{\max}.$$

이로부터 **유계 수렴 ball**:

$$\boxed{\limsup_{t\to\infty}\|m(t)-m^*\|\leq\frac{F_{\max}}{\mu}\leq\frac{\xi_r C_{\max}\|\phi\|_\infty}{\mu}}$$

움직이는 평형 $m^*(\phi(t))$를 추적하면 $e=m-m^*(\phi(t))$에
$-\dot m^*$가 추가된다. 예를 들어 $\|\dot m^*\|\leq v_*$이면

$$\limsup\|e(t)\|\leq\frac{F_{\max}+\tau v_*}{\mu}$$

가 되어야 하며, $v_*$를 버린 고정점 bound를 그대로 쓸 수 없다.

모든 가정이 유지된다면 이 ball 반경은 **수면-글림프 세척 후**
$\|\phi\|_\infty\to r_w\|\phi\|_\infty$에 의해 $r_w$ 배로 줄어든다
(`05_실험근거.md` 3.3 supported). 따라서 4.7절의 “조건부 단조 감소”를
다음과 같은 조건부 ISS 후보로 정리할 수 있다.

| 4.7절 표현 | A.1 격상 |
|---|---|
| $\|\nabla_m E\|>\xi_r C_k\|\phi\|\Rightarrow\Delta E<0$ (점별) | 고정 $m^*$: $\limsup\|m-m^*\|\leq F_{\max}/\mu$; 이동 $m^*$: tracking 항 추가 |
| 단조 감소 보장 영역 | 끌개 ball 반경의 닫힌 식 |
| 수면이 충분조건을 복원 | 수면이 ball 반경을 $r_w$ 배로 축소 |

#### A.1.3 검증 가능한 ball 반경

$\mu=\rho\|W\|/N$은 일반적으로 성립하는 식이 아니므로 사용하지 않는다.
$\mu$는 실제 대칭 Hessian의 최소 고유값을 해당 불변영역에서 직접 추정해야
한다. $\xi_r=0.490486813152402$, $C_{\max}=0.5$,
$\|\phi\|_\infty=1$을 E3의 괄호-내 forcing convention에 넣으면

$$R_{\text{ball}}\leq\frac{0.5\xi_r}{\mu}
=\frac{0.245243406576201}{\mu}.$$

만약 대신 $\dot m=-(1/\tau)\nabla E+d(t)$이고
$d=F_{\rm bypass}$를 **괄호 밖** forcing으로 정의하면 같은 입력 상계와
$\tau=10$에서 $\tau d_{\max}/\mu\leq2.45243406576201/\mu$다. 두 convention 어느
쪽에서도 과거의 $20/\mu$는 나오지 않는다. 구현은 둘 중 하나를 명시하고,
$C$-feedback small-gain과 $\dot m^*$를 포함해 측정해야 게이트 `F2`를 올릴 수 있다.

### A.2 게이트 `F1`: 자기조직화 충분조건 (3-simplex 수축 정리)

> 5절·8절의 "활성 비율이 $\varepsilon^2$ 로 자연 수렴" 가설은 transformer 기질에서 falsified (`5_Sparsity.md` 8.5). 이를 무엇을 만족하면 다른 기질에서 hard claim 으로 격상되는지 수식으로 명시한다.

#### A.2.1 부트스트랩 사상의 일반화

3-simplex

$$\Delta^2=\{p=(p_a,p_{DM},p_\Lambda)\in\mathbb R^3:
p_i\geq0,\;p_a+p_{DM}+p_\Lambda=1\}$$

위에서 Track-A manifest의
$D_N=3.177912999513294$와 입력 의존량

$$R(\alpha_s)=\alpha_sD_N(1+a_*\delta_N)$$

를 둔다. 특정 수치 baseline을 고정하기 전에는 $R$과 뒤의 두 분율을
상수로 고정하지 않는다. 먼저

$$f(a)=\exp[-D_N(1-a)],\qquad
r=\frac{R}{1+R},\qquad \ell=\frac{1}{1+R}$$

를 정의하고 다음 **정규화된 자기 사상**을 사용한다.

$$
\boxed{
B(p)=\bigl(f(p_a),\;(1-f(p_a))r,\;(1-f(p_a))\ell\bigr).
}
$$

$0<f(p_a)\leq1$, $r,\ell>0$, $r+\ell=1$이므로 모든
$p\in\Delta^2$에 대해 $B(p)\in\Delta^2$이다. 이전 식은 강결합상수
기호에 부트스트랩 분율을 대입해 서로 다른 두 양을 혼동했고,
$p_a=1$에서 음의 세 번째 성분을 만들어 자기 사상이 아니었다.

#### A.2.2 자기조직화 정리 (수축)

**정리 (내부 고정점과 국소 수축).** 모든 $R>0$에 대해 위 $B$는 내부 고정점

$$
p^*(R)=\left(a_*,(1-a_*)\frac{R}{1+R},
(1-a_*)\frac1{1+R}\right),\qquad
a_*=0.04863825851598632
$$

을 하나만 갖고, 경계 고정점 $(1,0,0)$을 추가로 갖는다. 내부 고정점에서
야코비안은

$$
DB(p^*)=q_*
\begin{pmatrix}
1&0&0\\
-r&0&0\\
-\ell&0&0
\end{pmatrix},
\qquad q_*=D_Nf(p_a^*)=D_Np_a^*=0.154568154011641.
$$

따라서 고윳값은 $(q_*,0,0)$이고 내부 고정점은 국소 점근 안정하다.

더 강한 Banach 명제를 위해 닫힌 불변집합

$$U=\{p\in\Delta^2:0\leq p_a\leq0.13\}$$

을 취한다. $f([0,0.13])=[0.04167,0.06300]\subset[0,0.13]$이므로
$B(U)\subset U$이다. 또한 $p,q\in U$에 대해 평균값정리와 simplex 제약으로

$$
\begin{aligned}
\|B(p)-B(q)\|_1
 &=2|f(p_a)-f(q_a)|\\
 &\leq 2q_U|p_a-q_a|
 \leq q_U\|p-q\|_1,
\end{aligned}
$$

$$q_U=\sup_{0\leq a\leq0.13}D_Nf(a)
=D_N e^{-0.87D_N}=0.2001757361<1.$$

따라서 $(U,\|\cdot\|_1)$에서 $B$는 실제 수축이고

$$\boxed{\|B^n(p)-p^*\|_1\leq q_U^n\|p-p^*\|_1\quad(p\in U)}$$

가 성립한다. $q_*$는 이 균일 상계가 아니라 고정점에서의 점근 선형률이다.

고정점 개수도 scalar 식으로 직접 확인된다. \(a=f(a)\)에 로그를 취해

$$h(a)=\log a+D_N(1-a)=0$$

로 두면 \(h''(a)=-a^{-2}<0\), \(h(1)=0\)이고, 유일한 극대점
\(a=1/D_N\)에서 \(h(1/D_N)=D_N-1-\log D_N>0\)이다. 한편
\(h(a)\to-\infty\) as \(a\downarrow0\)이므로 근은
\((0,1/D_N)\)의 작은 근 하나와 경계근 \(a=1\)뿐이다. 작은 근에서는
나머지 두 성분이 양수이므로 위 \(p^*\)가 유일한 내부 고정점이다.

#### A.2.3 자기조직화 격상 충분조건

기질 $\mathcal{S}$ 가 다음 5조건을 모두 만족하면, 위 정리의 hard claim 이 신경 모듈에 그대로 옮겨간다:

1. **Simplex 보존**: 활성/구조/배경 비율
   \((p_a,p_{DM},p_\Lambda)\)의 시간 진화가 \(\Delta^2\) 안에 머문다.
2. **자기측정**: 시스템이 $p_a(t)$ 를 자기 자신의 다음 갱신에 입력으로 쓸 수 있다 (자기일관 $a_* = \exp(-(1-a_*)D_{\text{eff}})$ 의 동역학적 실현).
3. **국소 안정성**: $\rho(DB(p^*)) < 1$ 이 측정 가능 (예: $p^*$ 근방 perturbation 후 수렴 비율).
4. **에너지 균형**: 활성당 비용 $C_a$, 구조 유지 비용 $C_s$, 배경 비용 $C_b$ 의 비율이 $C_a:C_s:C_b \approx 1:5.368:14.192$ 영역에 있는지 측정한다. 이 수치 근접 자체가 생물학적 메커니즘을 유도하지는 않는다.
5. **외부 데이터 재학습 가능**: A.1 의 ISS ball 이 닫히는 영역에서 학습이 안정적으로 진행된다.

| 기질 | 1 | 2 | 3 | 4 | 5 | 등급 |
|---|---|---|---|---|---|---|
| Transformer + Backprop | 부분 | 결손 | 측정 안 됨 | 결손 | 부분 | `falsified` (`5_Sparsity.md` 8.5) |
| SNN + STDP + 막전위 동역학 | 가능 | 가능 (STDP 자기참조) | 측정 필요 | 가능 (생물 정합) | 측정 필요 | 미검증 (`8_Roadmap.md` 0절 G-S1~G-S5) |
| 생물 뇌 (피질) | 측정됨 | 측정됨 | $\rho \in [0.1, 0.3]$ (`05_실험근거.md` 3.3) | 측정됨 | -- | `bridge` (`6_뇌/05_실험근거.md` 8장) |

이 표가 게이트 `F1` 의 닫힘 경로다. 5조건 중 1개라도 결손이면 본문의 자기수렴 hard claim 은 금지된다.

### A.3 게이트 `F3`: 에르고딕 동등성 (시간 ↔ 공간)

> 3_Sleep.md 6.2 의 "시간 분배 ≈ 에너지 분배" 를 단순 수치 근접에서 에르고딕 정리로 격상한다.

#### A.3.1 모드 점유 측도

뇌가 모드 공간 $\mathcal{M} = \{\text{WAKE}, \text{NREM}, \text{REM}\}$ 의 마르코프 사슬을 가진다고 두자. 정류 분포 $\pi = (\pi_W, \pi_N, \pi_R) \in \Delta^2$.

**에르고딕 정리 (Birkhoff)**: 사슬이 에르고딕이면

$$\lim_{T\to\infty}\frac{1}{T}\int_0^T \mathbb{1}[M(t)=m]\,dt \;=\; \pi_m \quad (\text{a.s.})$$

따라서 **시간 분배** $(t_W/T, t_N/T, t_R/T)$ 와 **정류 점유 측도** $\pi$ 는 동일 simplex $\Delta^2$ 위의 같은 객체다.

#### A.3.2 코어 분배와의 동등 클래스

이 절의 수치 비교는 Track-A manifest를 고정한다. 그때 CE 코어의 조건부
공간 에너지 분배
\[
p^*=(\Omega_\Lambda,\Omega_{DM},\Omega_b)
=(0.6902735671,0.2610881744,0.0486382585)
\]
도 \(\Delta^2\) 위의 점이다. 두 측도의 거리:

$$d_{\text{KL}}(\pi_{\text{brain}} \,\|\, p^*) = \sum_i \pi_i \log\frac{\pi_i}{p_i^*}$$

| 비교 | $\pi$ 또는 $p$ | $d_{\text{KL}}$ vs $p^*$ |
|---|---|---|
| Raichle 뇌 에너지 분배 | $(0.65, 0.30, 0.05)$ | $\approx 0.00398$ |
| 인간 수면 시간 분배 | $(0.667, 0.250, 0.083)$ | $\approx 0.01063$ |
| 조건부 CE 코어 분배 | $(0.6902735671,0.2610881744,0.0486382585)$ | $\equiv 0$ |
| 균등 분배 (귀무) | $(1/3, 1/3, 1/3)$ | $\approx 0.48036$ |

#### A.3.3 게이트 `F3` 격상 조건

**격상 가능 표현**: "뇌의 모드 점유 측도 $\pi$ 와 CE 코어의 공간 에너지 분배 $p^*$ 는 동일 simplex 위에서 KL 거리 $\sim 10^{-2}$ 안에 있다."

**여전히 금지 표현**: "시간 분배 = 에너지 분배."

이 격상 후에도 두 측도의 차원 (시간 vs 공간) 동등성은 주장하지 않으며, 동일 simplex 위의 측도 근접만 hard claim 한다.

### A.4 게이트 `F4`: PCI 교차검증 (의식 환원 금지 유지)

> 9절·`7_Consciousness.md`·F.17 의 메타인지 안정도 $\exp(-c_d d_\tau)$ 가 PCI (Casali 2013, Massimini 그룹) 와 어떤 정량 관계를 가지는지 명시한다. 게이트 `F4` 자체는 닫지 않으며, **무엇을 측정하면 `bridge` 로 갈 수 있는지** 만 정의한다.

#### A.4.1 PCI 정의

**PCI (Perturbational Complexity Index)**: TMS 자극 후 EEG 응답의 시공간 압축 복잡도 (Lempel-Ziv).

$$\text{PCI}(t) = \frac{L(\text{compressed EEG response})}{H(\text{source distribution})}$$

| 상태 | PCI 범위 (Casali 2013) |
|---|---|
| 깨어있음 | $0.44 - 0.67$ |
| REM 수면 | $0.40 - 0.60$ |
| NREM N3 (서파) | $0.18 - 0.31$ |
| 식물상태 (UWS) | $0.15 - 0.31$ |
| 마취 (propofol) | $0.18 - 0.28$ |

#### A.4.2 CE 안정도와 PCI

게이트 `F4` 격상 가설 (현재 `hypothesis`):

$$\boxed{\text{PCI}(t) \approx \alpha\cdot\text{메타인지 안정도}_\tau(t) + \beta = \alpha\cdot\exp(-c_d\,d_\tau(t)) + \beta}$$

**검증 절차**:

1. CE 시뮬레이션에서 모드 (WAKE/NREM/REM) 별 $d_\tau$ 프로파일 측정.
2. 동일 모드의 PCI 값과 회귀.
3. $R^2 > 0.7$ 이면 `hypothesis` → `bridge`. 단 PCI 자체가 의식의 정량 척도라는 hard claim 은 하지 않는다.
4. `bridge` 단계에서도 본 부록은 "안정도 = 의식" 환원을 금지한다.

#### A.4.3 측정 가능한 모드 프로파일 예측

CE 가 옳다면 시뮬레이션에서:

| 모드 | 예측 $d_\tau$ | 예측 안정도 | 대응 PCI 범위 |
|---|---|---|---|
| WAKE | 낮음 (0.1-0.2) | 0.82-0.90 | 0.44-0.67 |
| REM | 중간 (0.3-0.4) | 0.67-0.74 | 0.40-0.60 |
| NREM N3 | 높음 (0.8-1.2) | 0.30-0.45 | 0.18-0.31 |
| 마취 (CE: $C_k \to 0$, 외부 입력 차단) | 매우 높음 (>1.5) | <0.22 | 0.18-0.28 |

이 표의 모드별 안정도 차이가 PCI 와 단조 일치하면 게이트 `F4` 가 `bridge` 로 격상된다.

### A.5 격상 후 다리 게이트 표 (목표)

| 게이트 | 현재 | A절 격상 후 (조건 충족 시) | 격상 충분조건 |
|---|---|---|---|
| `F2` 비보존 바이패스 | `Bridge` (조건부) | `Bridge` (ISS ball 반경) | A.1.3 ball 반경 측정 |
| `F1` 메커니즘 결손 | `Bridge` (수치 근접) | `Bridge` (5조건 만족 기질) | A.2.3 5조건 모두 충족 |
| `F3` 시간/공간 혼동 | `Phenomenology` | `Bridge` (KL 동등 클래스) | A.3.3 KL 거리 보고만 |
| `F4` 의식 환원 | `Phenomenology` | `Bridge` (PCI 회귀) | A.4.2 $R^2 > 0.7$ |

이 격상은 어느 경우에도 코어의 정확성을 깎지 않으며, 다리 단계에서 무엇을 측정해야 하는지를 규정한다. 본 부록은 본문의 어떤 hard claim 도 위로 올리지 않으며, 본문이 어디로 갈 수 있는지의 **목표 지도**다.
