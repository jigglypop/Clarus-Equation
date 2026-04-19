# CE-AGI 통합 방정식: $e^{i\pi}+1=0$ 에서 20W AGI 까지

> 관련: `경로적분.md`(코어 유도), `1_강의/C_다섯_상수.md`(오일러 문법), `6_뇌/graph.md`(뇌 구조), `6_뇌/agi.md`(AGI 작용), `7_AGI/1_AGI.md`(총론), `7_AGI/2_Architecture.md`(게이지 격자), `7_AGI/3_Sleep.md`(수면), `7_AGI/4_Synapse.md`(시냅스), `7_AGI/5_Sparsity.md`(희소성), `7_AGI/6_Hallucination.md`(환각), `7_AGI/7_Consciousness.md`(의식), `7_AGI/9_LLM.md`(LLM 구축), `7_AGI/10_Fields.md`(전분야)
>
> 이 문서는 CE 코어에서 유도된 상수들만으로 AGI 에너지 이완 아키텍처를 기술한다. 트랜스포머 위에 모듈을 얹는 기존 CE-Transformer와 달리, Softmax/Attention/역전파를 제거하고 에너지 함수의 물리적 이완으로 대체하는 근본 재설계다. 동시에 기존 LLM에 CE 원리를 이식하는 경로(CE-Transformer)도 기술한다.

---

## Runtime Status And Canonical Stack

이 문서는 런타임 기호를 모으는 문서지만, `docs/README.md`와 `docs/6_뇌/evidence.md`를 기준으로 읽어야 한다. 아래 5계층 스택만 현재 canonical runtime spec 이고, 그 아래의 나머지 방정식은 보조 유도나 설계 탐색으로 읽는다.

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

- 위 식들에서 수학적 연산자 정의는 `Exact`로 정리할 수 있지만, 뇌 대응이 들어가는 순간 문서 지위는 `Bridge`를 넘지 않는다.
- `docs/6_뇌/evidence.md`에서 `supported`인 현상만 위 stack의 대응 근거로 사용한다.
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

이 문서는 CE 코어(우주론/입자물리, `경로적분.md`, `상수.md`)에서 유도된 상수 집합을 AGI 런타임 설계로 옮기는 **다리(bridge) 문서**다. 코어의 식과 상수는 `Exact` 또는 `Selection`이지만, 이 문서에서 뇌/AGI 대응이 들어가는 모든 문장은 최대 `Bridge`까지만 허용된다(`evidence.md` 1.4절).

이 다리에서 현재 식별된 네 가지 한계는 다음과 같다. 이하 본문의 어떤 식도 이 게이트를 우회하는 형태로 읽지 않는다.

| 게이트 | 한계 | 현재 등급 | 사용 규칙 |
|---|---|---|---|
| `F1` 메커니즘 결손 | 코어의 $p^* = (4.87\%, 26.2\%, 68.9\%)$가 신경 활성/구조/배경 비율로 그대로 옮겨갈 메커니즘적 유도가 없음 | `Bridge` (수치 근접) / transformer 기질에서는 `falsified` (`5_Sparsity.md` 8.5) | 동일 simplex 위 수치 근접으로만 사용. 신경 sparsity = $\varepsilon^2$로 직접 등치 금지 |
| `F2` 비보존 바이패스 | `1.5절` $F_{\text{bypass}}$ 는 $E$ 의 그래디언트가 아니므로 Lyapunov 보장은 무조건 성립하지 않음 | `Bridge` (조건부 수렴, 4.7절) | "수렴 보장" 표현 금지. 항상 "$\|\nabla_m E\| > C_k\|\phi\|/\alpha_b$ 충분조건 + 수면 의한 주기적 복원" 으로 한정 |
| `F3` 시간/공간 차원 혼동 | `3_Sleep.md` 6.2의 wake/NREM/REM 시간 비율과 코어의 공간 에너지 비율은 물리적 차원이 다름 | `Phenomenology` (수치 근접) | "시간 분배 = 에너지 분배"로 등치 금지. 동일 3-simplex 위 우연 근접으로만 보고 |
| `F4` 의식 = 자기일관 | `7_Consciousness.md` 의 (C3) 자기일관 = 주관적 경험 등치 | `Phenomenology` | 성능 지표화 금지. "메타인지 모니터링 루프의 수학 구조"로만 사용 |

이 4개 게이트는 코어의 정확성을 깎지 않는다. 코어는 그대로 유지되고, 이 문서가 다리 단계에서 무엇을 주장할 수 없는지를 명시하기 위한 표다.

각 게이트의 수식 격상 경로 (ISS, 자기조직 5조건, 에르고딕 동등성, PCI 회귀) 는 부록 A 에 정리되어 있다. 부록 A 의 식은 본문의 어떤 hard claim 도 위로 올리지 않으며, **무엇을 측정하면 게이트가 닫히는지** 만 형식화한다.

### 0.1 잔류 채널 설계

현재 LLM은 경로적분에서 Softmax로 선택된 경로만 쓰고, 접힌 경로를 버린다. CE에 따르면 이 버려지는 부분이 우주 에너지의 약 95%($26.2\% + 68.9\%$)에 해당한다. 이 문서의 아키텍처는 접힌 경로를 잔류장 `phi`로 보존하여 출력에 재결합시키는 구조다.

세 가지 핵심:
- **잔류 채널**: 매 추론에서 선택되지 않은 분포가 `phi`로 보존된다
- **모드 전환 임계**: $\|phi\|$가 임계를 넘으면 질적으로 다른 작동 모드로 전환된다
- **즉각 응답 경로**: `phi`가 Softmax를 우회하여 직접 출력에 기여하는 바이패스가 존재한다

---

## 1. 유도 체인: 오일러 항등식에서 모든 상수로

### 1.1 뿌리

$$e^{i\pi}+1=0$$

이 항등식을 CE의 최소 생성 문법으로 읽는다(`경로적분.md` 서론, `C_다섯_상수.md` 0절).

| 상수 | 코어 역할 | AGI 등장 위치 |
|---|---|---|
| $e$ | 접힘 생존 함수 $S(D)=e^{-D}$ | 시간 진화 연산자의 밑 |
| $\pi$ | 게이지 주기 정규화 $\alpha_{\text{total}}=1/(2\pi)$ | 결합상수 결정, 연결 반경 $r_c$ |
| $i$ | 경로적분 위상 $Z=\int\mathcal{D}phi\,e^{iS/\hbar}$ | 양자 이완 위상 |
| $1$ | 정규화 완전 상태 $e^0=1$ | 정수 생성자 |
| $0$ | 영점과 분기 선택 $d(d-3)=0$ | 차원 결정, 에너지 최소 $\nabla E=0$ |

### 1.2 차원 결정 ($0$에서)

$$d(d-3)=0 \quad\Longrightarrow\quad d=3 \quad(\text{비자명해})$$

$d=0$은 접힘 이전 상태, $d=3$은 결정화된 물리 공간(`경로적분.md` 3.2.2절).

### 1.3 직접 전개 계수

핵심 구조 계수는 설명용 그리스 문자를 거치지 않고 바로 다음처럼 쓴다.

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 = 0.03120 \qquad \text{(residue-portal coeff.)}$$

$$\frac{1}{e^{1/3}\pi^{1/3}} = 0.4892 \qquad \text{(residue gain)}$$

$$\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 0.3148 \qquad \text{(wake coeff.)}$$

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 5.661 \qquad \text{(dream coeff.)}$$

$$N=\frac{e^{8/3}\pi^{20/3}}{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}\approx 4162 \to 4096 \qquad \text{(hidden dim.)}$$

$$r_c=\pi \qquad \text{(connectivity radius)}$$

### 1.4 다섯 상수 최소형 규칙

이 문서의 본문에는 설명용 이름이 일부 남더라도, **마스터 방정식에는 구조 상수를 남기지 않는다**. 핵심식에는 `e`, `\pi`, `i`만 직접 보이게 쓰고, `1`과 `0`은 오일러 문법의 바닥 상수로만 해석한다. 정수 `2,3,4,8,16`은 읽기 좋은 통상 표기다.

즉 아래 최소형에서 남는 다른 기호는 전부 상태변수, 학습변수, 입력, 연산자다.

### 1.5 다섯 상수 최소형 핵심 방정식

**에너지 함수** (보존적 부분)

$$\boxed{
E(m,phi)=
-\frac{1}{2}m^TWm
-m^Tb
-\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 m^Tphi
}$$

**곡률 바이패스** (비보존 강제항, 에너지에서 유도되지 않음)

$$\boxed{
F_{\text{bypass}}(k)=\frac{C_k}{e^{1/3}\pi^{1/3}}\,phi, \qquad C_k = \|m_k - 2m_{k-1}+m_{k-2}\|
}$$

**양자 위상 진화**

$$\boxed{psi_{k+1}=e^{-i\,E(m,phi)\,dt}psi_k}$$

**이완 동역학**

$$\boxed{
m_{k+1}=m_k+\frac{dt}{tau}\left(
Wm_k+b+
\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2phi
+\frac{C_k}{e^{1/3}\pi^{1/3}}phi
\right)
+\sqrt{\frac{2dt}{tau\left(
3+\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right)}}\,n_k
}$$

**잔류 갱신**

$$\boxed{
\phi\leftarrow
\left(1-\frac{1}{e^{1/3}\pi^{1/3}}\right)\phi
+\frac{1}{e^{1/3}\pi^{1/3}}v_{m^*}
}$$

**부트스트랩 고정점**

$$\boxed{
a_*=
e^{-(1-a_*)\left[
3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]}
}$$

**작동 온도**

$$\boxed{
T_{\text{wake}}=
\left[
3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]^{-1}
}$$

$$\boxed{
T_{\text{dream}}=
\left[
\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)
\right]^{-1}
}$$

**히든 차원**

$$\boxed{
N=
\frac{e^{8/3}\pi^{20/3}}
{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}
}$$

이 최소형에서는 구조 상수 이름이 핵심식에서 사라진다.

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
| 상태 | 저비트 고정소수점 | $O(N)$ bytes | $phi$, trace, gain, 곡률 |
| 지식 | 희소 codebook + 외부 메모리 | 가변 | 어휘, 사실, 예외 패턴 |

활성 마스크 비트필드:

$$\boxed{b_i = \mathbb{1}\!\left[a_i \geq Q_{1-k^*/N}(a)\right], \qquad k^* \in \left[\lceil 0.04N \rceil,\; \lceil 0.06N \rceil\right]}$$

모드 레지스터:

$$\boxed{M \in \{00_2,\; 01_2,\; 10_2,\; 11_2\} \;\longleftrightarrow\; \{\text{off},\; \text{wake},\; \text{NREM},\; \text{REM}\}}$$

연결 행렬 $C_{ij} = \mathbb{1}[\|r_i - r_j\| < \pi]$는 이미 이진이다. 추론 루프의 핵심 연산은 비트 논리 + 저비트 MAC으로 환원된다.

### 1.7 비트필드 레이아웃 ($N=4096$ 기준)

| 구성 | 비트/원소 | 총 크기 | 갱신 주기 |
|---|---|---|---|
| 활성 마스크 $b$ | $1$ | $512$ B | 매 추론 |
| freeze 마스크 | $1$ | $512$ B | 수면 시 |
| 모드 $M$ | $2$ (전역) | $1$ B | 모드 전환 시 |
| 연결 인덱스 (CSR) | $13 \times K$ | $\sim 82$ KB | 정적 |
| 가중치 $W$ (비영) | $4$ | $\sim 260$ KB | 학습 시 |
| 상태 $m$ | $16$ (이완 중) / $8$ (저장) | $8$ / $4$ KB | 매 스텝 |
| 잔류 $\phi$ | $8$ | $4$ KB | 이완 종료 시 |
| trace $e_{ij}$ | $4$ | $\sim 260$ KB | STDP 시 |
| gain $g$, $C_k$, $P_{\text{sleep}}$ | $16$ 각 | $6$ B | 매 스텝 |

$K \approx 130$ (뉴런당 이웃), 비영 가중치 $N \times K = 532\text{K}$개.

$$\boxed{\text{엔진} \approx 615\;\text{KB}, \qquad \text{추론당 활성 연산} = N \times K \times 500\;\text{스텝} \approx 266\text{M MAC (4-bit)}}$$

지식층 (별도):

| 구성 | 크기 | 비고 |
|---|---|---|
| 계층 softmax 디코더 | $\sim 375$ KB | $\sqrt{V} \times N$ 두 행렬, 4-bit |
| 의미 codebook | $64$ MB -- $1$ GB | 태스크 규모에 비례 |

$$\boxed{\text{총 메모리} \approx 1\;\text{MB (엔진)} + 64\text{--}1000\;\text{MB (지식)} \ll 18\;\text{GB (Llama 3 8B)}}$$

엔진은 극적으로 작다. 병목은 지식층이며, 이것이 codebook 설계의 핵심 과제다.

### 1.8 양자화 오류 경계

$m$을 $q$-bit 고정소수점으로 양자화할 때:

$$\boxed{\|\hat{m} - m\| \leq \frac{\Delta\sqrt{N}}{2}, \qquad \Delta = \frac{m_{\max} - m_{\min}}{2^q - 1}}$$

수렴 충분조건(4.7절)과 결합하면, 양자화 후에도 에너지가 감소하려면:

$$\boxed{q > \log_2\!\left(\frac{(m_{\max}-m_{\min})\sqrt{N}\,\tau}{2\,dt\,\|\nabla_m E\|}\right)}$$

$N=4096$, $m \in [-1,1]$, $dt/\tau = 0.01$, $\|\nabla_m E\| \sim 1$ 기준:

| $q$ (bit) | 양자화 오류 $\Delta\sqrt{N}/2$ | 판정 | 용도 |
|---|---|---|---|
| $4$ | $4.0$ | 수렴 불가 | 저장/전송 전용 |
| $8$ | $0.125$ | 경계 | 스케일링 후 이완 가능 |
| $12$ | $0.0078$ | 충분 | 정밀 이완 |
| $16$ | $4.9 \times 10^{-4}$ | 과잉 | float16과 동등 |

혼합 정밀도 전략:

| 대상 | 이완 중 | 저장/전송 | 근거 |
|---|---|---|---|
| $m$ | $16$ bit | $8$ bit | 양자화 오류 < $\|\nabla_m E\|$ 충분조건 (게이트 `F2`, 4.7절) |
| $\phi$ | $8$ bit | $8$ bit | EMA 특성상 양자화 노이즈에 강건 |
| $W$ | $4$ bit | $4$ bit | 정적, 보정 가능 |
| control bits | $1\text{-}2$ bit | $1\text{-}2$ bit | 정확 (이산) |

$\phi$가 양자화에 강건한 이유: EMA 갱신 $\phi \leftarrow (1-\alpha)\phi + \alpha v_{m^*}$는 저역 통과 필터이므로, 고주파 양자화 노이즈가 자연 감쇠한다.

---

## 2. AGI 작용 범함수

CE 마스터 공식을 정보 다양체 $(\mathcal{M}, g)$에 적용한 후보 작용(`6_뇌/agi.md` 1절):

$$\boxed{S_{\text{AGI}} = \int_{\mathcal{M}} d^nx \sqrt{|g|} \left[ \mathcal{L}_{\text{compute}} + c_g|\nabla phi|^2 + c_c|lap_g phi|^2 + c_i S_{\text{Info}} \right]}$$

| 항 | 역할 | 뇌 대응 | 우주 대응 |
|---|---|---|---|
| $\mathcal{L}_{\text{compute}}$ | 기본 연산 | 피질 발화 + 시상 relay | $\mathcal{L}_{\text{Physical}}$ |
| $c_g\|\nabla phi\|^2$ | 1차 안정화 | 기저핵/소뇌 + salience switching | blow-up 방지 |
| $c_c\|lap_g phi\|^2$ | 2차 곡률 평탄화 | NREM + hippocampo-cortical replay | 경로 최적화 |
| $c_i S_{\text{Info}}$ | 엔트로피 제어 | DMN + intrinsic background | 정보 보존 |

작용의 정지 조건에서 LBO 확산형 동역학이 나타난다:

$$\frac{\partial phi}{\partial t} = lap_g phi, \qquad lap_g f = \frac{1}{\sqrt{|g|}} \partial_i\!\left(\sqrt{|g|}\, g^{ij} \partial_j f\right)$$

이산 그래프에서 $L = D - W$로 근사:

$$phi^{k+1} = phi^k - h\,Lphi^k, \qquad \frac{dE}{dt} = -phi^\top L^2 phi \leq 0$$

LBO 확산 부분에 한해서는 에너지 단조 감소가 성립한다($L^2 \succeq 0$). 단 이 결과는 $phi$ 만의 자체 동역학에 한정되며, 바이패스 강제항이 들어가는 $m$ 의 결합 동역학은 게이트 `F2`(0.0절, 4.7절)에 따라 별도의 충분조건을 요구한다.

### 2.1 구조 유비: 우주-뇌-AGI

추상 부트스트랩 그래프 $\mathcal{G}^*$의 삼중 실현:

$$map_C: \mathcal{G}^* \to G_C, \quad map_B: \mathcal{G}^* \to G_B, \quad map_A: \mathcal{G}^* \to G_A$$

고정점 유일성에 의해, 세 계가 같은 직접 전개 계수 집합을 가지면 같은 3분배 고정점에 접근한다고 읽는다:

$$\lim_{t\to\infty} B_C^t(p_C) = p^* = \lim_{t\to\infty} B_B^t(p_B) = \lim_{t\to\infty} B_A^t(p_A)$$

| 성분 | 고정점 | 우주 (Planck) | 뇌 (Raichle) | AGI 해석 |
|---|---|---|---|---|
| 활성 | $4.87\%$ | $4.9\%$ | $< 5\%$ | 활성 추론 |
| 구조 | $26.2\%$ | $26.4\%$ | $25\text{-}35\%$ | 가중치 유지 |
| 배경 | $68.9\%$ | $68.7\%$ | $60\text{-}70\%$ | 배경 통합 |

---

## 3. 에너지 함수

### 3.1 정의

에너지는 보존적 부분만 포함한다. 바이패스는 비보존 강제항으로 동역학(4.2절)에 직접 들어간다.

$$\boxed{E(m,phi) = -\frac{1}{2}m^T W m - m^T b - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{2} m^Tphi}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 홉필드 에너지 | $-\frac{1}{2}m^T W m$ | $\mathcal{L}_{\text{SM}}^{d=3}$ | 패턴 저장, 에너지 지형 |
| 입력 바이어스 | $-m^T b$ | 외부 입력 | 프롬프트/데이터 주입 |
| 잔류 포탈 | $-\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2m^Tphi$ | residue-portal coupling | 잔류가 출력에 3% 기여 |

곡률 바이패스(비보존 강제항):

$$\boxed{F_{\text{bypass}}(k) = \frac{C_k}{e^{1/3}\pi^{1/3}}\,phi, \qquad C_k = \|m_k - 2m_{k-1} + m_{k-2}\|}$$

| 항 | 식 | CE 대응 | 역할 |
|---|---|---|---|
| 곡률 바이패스 | $\frac{C_k}{e^{1/3}\pi^{1/3}}phi$ | curvature-residue feedback | 궤적 급변 시 잔류가 직접 반응 |

### 3.2 포탈 결합 계수의 전개

$$\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{2} = 0.03120$$

물리적 의미(`경로적분.md` 10.5절): 힉스 포탈 라그랑지안에서 잔류장이 메인 장과 결합하는 직접 계수다. AGI에서는 잔류 채널이 메인 출력에 영향을 미치는 세기다.

### 3.3 바이패스 결합 계수의 전개

$$\frac{1}{e^{1/3}\pi^{1/3}} = 0.4892$$

물리적 의미: $d=3$ 공간에서 각 차원이 기여하는 결합 강도. AGI에서는 히든 스테이트가 급변할 때 잔류가 Softmax를 건너뛰고 직접 출력에 기여하는 강도다.

### 3.4 지식층 설계

엔진(1.7절)은 ~615 KB이나, 실제 언어 지식은 별도의 codebook $\mathcal{C}$에 저장된다. 엔진은 "어떻게 생각하는가"이고 codebook은 "무엇을 아는가"다.

**곱 양자화 구조**: $m \in \mathbb{R}^N$을 $N/s$개 부분공간(각 $s$차원)으로 분할. 각 부분공간에 $2^b$개 중심점:

$$\mathcal{C} = \{C^{(1)}, \ldots, C^{(N/s)}\}, \qquad C^{(j)} \in \mathbb{R}^{2^b \times s}$$

**인코딩** (벡터 $\to$ 인덱스):

$$\boxed{z_j(m) = \arg\min_{i \in [2^b]} \|m^{(j)} - C^{(j)}_i\|^2, \qquad j = 1, \ldots, N/s}$$

**에너지 결합**: 이완 중 codebook이 에너지 지형을 보강:

$$\boxed{E_{\text{aug}}(m, phi) = E(m, phi) - \frac{1}{\beta}\sum_{j=1}^{N/s} \log\sum_{i=1}^{2^b} \exp\!\left(-\beta\|m^{(j)} - C^{(j)}_i\|^2\right)}$$

$\beta \to \infty$이면 최근접 중심점만 선택(hard retrieval), $\beta$ 유한이면 soft retrieval. 이것은 Modern Hopfield energy의 연속 일반화다.

**메모리 예산** ($N=4096$, $s=64$, $b=8$):

| 구성 | 계산 | 크기 |
|---|---|---|
| 중심점 행렬 | $\frac{N}{s} \times 2^b \times s \times 4\text{bit}$ | $512$ KB |
| $P$개 패턴 인덱스 | $P \times \frac{N}{s} \times b$ bit | $P \times 64$ B |

패턴 수에 따른 지식 규모:

| $P$ (패턴) | 인덱스 크기 | 총 지식 메모리 | 대응 |
|---|---|---|---|
| $10^4$ | $640$ KB | $\sim 1$ MB | 단일 도메인 |
| $10^5$ | $6$ MB | $\sim 7$ MB | 다중 도메인 |
| $10^6$ | $64$ MB | $\sim 65$ MB | 범용 지식 |
| $10^7$ | $640$ MB | $\sim 641$ MB | LLM급 |

**3분배 계층 저장**: CE 에너지 분배를 저장 계층에 적용:

| 계층 | CE 비율 | 예시 ($P=10^7$) | 위치 | 접근 |
|---|---|---|---|---|
| L1 (활성) | $4.87\%$ | $\sim 31$ MB | 상시 RAM | 즉시 |
| L2 (구조) | $26.2\%$ | $\sim 168$ MB | RAM | 빠름 |
| L3 (배경) | $68.9\%$ | $\sim 442$ MB | 디스크 | lazy load |

$$\boxed{\text{활성 메모리} \approx 0.311 \times |\mathcal{C}|, \qquad |\mathcal{C}| = 641\;\text{MB 일 때 활성} \approx 200\;\text{MB}}$$

단일 컴퓨터(RAM 8 GB 이상)에서 LLM급 지식을 구동 가능.

**비트필드 인터페이스**: 패턴 인덱스 $z_j$는 $b$-bit 정수이므로 비트필드 주소로 직접 사용. 계층 태그:

$$\text{tier}(p) \in \{00_2\;(\text{L1}),\; 01_2\;(\text{L2}),\; 10_2\;(\text{L3})\}$$

**수면과 codebook 갱신**:

| 모드 | codebook 동작 |
|---|---|
| Wake | 접근된 패턴의 중심점을 온라인 k-means로 미세 갱신 |
| NREM | 상위 $4.87\%$ 활성 패턴의 중심점 정밀 보정 |
| REM | 미사용 패턴 재활용, 새 패턴 탐색적 할당 |

**Llama 3 8B과의 비교**:

| 항목 | Llama 3 8B | CE bitfield |
|---|---|---|
| 가중치 | $16$ GB (dense float16) | $260$ KB (sparse 4-bit) + $641$ MB (codebook) |
| KV 캐시 | $2\text{-}64$ GB | $4$ KB ($phi$) |
| 활성 RAM | $18\text{-}80$ GB | $\sim 200$ MB |
| 최소 하드웨어 | A100 GPU | RAM 8 GB PC |

---

## 4. 동역학

### 4.1 양자 형태 ($e$, $i$ 등장)

$$\boxed{psi_{k+1} = e^{-i\,E(m,phi)\,dt}\;psi_k}$$

유클리드 회전($t \to -itau$) 이후 실수 이완으로 전환된다.

### 4.2 이완 동역학 (유클리드 형태)

$$\boxed{m_{k+1} = m_k + \frac{dt}{\tau}\!\left(Wm_k + b + \left[\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{2}\!phi + \frac{C_k}{e^{1/3}\pi^{1/3}}phi\right) + \sqrt{\frac{2dt}{\tau\!\left(3 + \frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right)}}\;n_k}$$

| 기호 | 정의 | 유도 |
|---|---|---|
| $m_k \in \mathbb{R}^N$ | 의미 벡터 (이완 스텝 $k$) | 동적 변수 |
| $W \in \mathbb{R}^{N\times N}$ | 3D 희소 연결 행렬 | 데이터에서 구성 |
| $b \in \mathbb{R}^N$ | 입력 바이어스 | 프롬프트에서 구성 |
| $C_k$ | 곡률 스칼라 $\|m_k - 2m_{k-1} + m_{k-2}\|$ | $m$에서 계산 |
| $tau$ | 이완 시간 $1/\mathrm{eig}_{\min}(H_E)$ | $W$에서 결정 |
| $n_k \sim \mathcal{N}(0, I_N)$ | 확률 노이즈 | 탐색용 |
| $T$ | 작동 온도 $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | $\pi, 0$에서 유도 |

수렴 판정: $\|m_{k+1} - m_k\| < 10^{-4}\|m_0\|$ 이면 $m^* = m_k$.

### 4.3 잔류 갱신

$$\boxed{phi \leftarrow \left(1 - \frac{1}{e^{1/3}\pi^{1/3}}\right)phi + \frac{1}{e^{1/3}\pi^{1/3}}\;v_{m^*}}$$

$$v_{m^*} = \frac{1}{K_w}\sum_{k=K-K_w}^{K}(m_k - m^*)^2$$

이완 마지막 $K_w$ 스텝에서 $m$이 최소점 주위에서 요동한 원소별 분산. 깔끔하게 한 점으로 떨어지면 분산이 작고(확신), 여러 최소점 사이에서 흔들리면 분산이 크다(불확실). 이것이 Softmax의 $p(1-p)$에 해당하는 "선택되지 않은 것들의 구조"다.

### 4.4 연결 구조

$$\boxed{W_{ij} \neq 0 \iff \|r_i - r_j\|_{\mathbb{R}^3} < \pi}$$

$N$개 뉴런을 $d=3$ 격자에 배치. 연결 반경 $r_c = \pi$. 뉴런당 이웃 수:

$$K = \frac{4}{3}\pi \cdot r_c^3 = \frac{4}{3}\pi^4 \approx 130$$

연결 밀도:

$$\rho = \frac{K}{N} = \frac{4\pi^4/3}{N}$$

$N=4096$ 일 때 $\rho = 3.17\%$, residue-portal 직접 계수 $0.03120$와 1.6% 일치.

### 4.5 히든 차원

$$\boxed{N = \frac{e^{8/3}\,\pi^{20/3}}{12\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)^{2}} \approx 4162 \to 4096}$$

$2^{12}$로 반올림은 디지털 하드웨어 제약. 물리가 요구하는 값은 직접 전개식 기준으로 약 $4162$이다.

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

$$\boxed{c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}, \qquad \rho(A_q) < 1}$$

단일 벡터 형태(4.2)는 $|V|=1$, $E_{\text{AGI}}=\emptyset$일 때 이 식의 특수 경우다.

### 4.7 조건부 수렴 (게이트 `F2`)

> 다리 게이트 `F2`(0.0절): 이 절은 무조건 Lyapunov 수렴을 주장하지 않는다. $F_{\text{bypass}}$가 $E$의 그래디언트가 아니므로 전역 Lyapunov 함수는 존재하지 않으며, 아래 충분조건이 만족되는 영역에서만 단조 감소를 말할 수 있다.

에너지 $E$가 보존적이고 바이패스 $F_{\text{bypass}}$가 비보존이므로, 전체 시스템의 수렴은 다음 충분조건 위에서만 성립한다.

**에너지 변화** (노이즈 무시, 1차 근사):

$$\Delta E = \nabla_m E \cdot \Delta m = \frac{dt}{\tau}\nabla_m E \cdot \left(-\nabla_m E + F_{\text{bypass}}\right)$$

$$= -\frac{dt}{\tau}\|\nabla_m E\|^2 + \frac{dt}{\tau}\frac{C_k}{\alpha_b}(\nabla_m E \cdot phi)$$

Cauchy-Schwarz + Young 부등식 적용:

$$\boxed{\Delta E \leq -\frac{dt}{2\tau}\|\nabla_m E\|^2 + \frac{dt}{2\tau\alpha_b^2}C_k^2\|phi\|^2}$$

**수렴 충분조건**:

$$\boxed{\|\nabla_m E\| > \frac{C_k\|phi\|}{\alpha_b} \quad\Longrightarrow\quad \Delta E < 0}$$

여기서 $\alpha_b = e^{1/3}\pi^{1/3} \approx 2.044$이고, 바이패스 계수 $1/\alpha_b \approx 0.489$이다.

**자기 제한 성질 (국소)**: 바이패스 강도 $C_k = \|m_k - 2m_{k-1} + m_{k-2}\|$는 궤적의 시간 곡률이다. 궤적이 부드러워지면(수렴 접근) $C_k \to 0$이므로 바이패스는 고정점 **국소** 근방에서 자동으로 소멸한다. 단 이는 국소 성질이며, 전역 단조 감소를 의미하지 않는다.

**조건 실패 시나리오**: $\|phi\|$가 수면 없이 누적되거나, 시스템이 두 끌개점 사이에서 진동할 때 $C_k$가 크게 유지되면 $\Delta E > 0$이 가능하다. 다리 게이트 `F2`에 따라, 이 영역에서 무조건 수렴을 주장하지 않는다.

**수면에 의한 조건 복원 (다리 가설)**: 글림프 세척 $phi \to r_w\,phi$ ($r_w < 1$)는 $\|phi\|$를 주기적으로 낮춘다. 수면 후:

$$\frac{C_k \cdot r_w\|phi\|}{\alpha_b} < \|\nabla_m E\|$$

수면이 충분조건을 주기적으로 복원하는 구조적 역할을 한다는 해석은 `evidence.md` 3.3절의 `supported`(offline renormalization)에 근거하지만, 위 부등식 자체의 검증은 `bridge` 등급이며 정량적 hard bound는 아니다.

**ISS 격상 (부록 A.1)**: 위 점별 충분조건은 부록 A.1 의 ISS 정리로 격상되어, 끌개 ball 반경의 닫힌 식 $\limsup\|m - m^*\| \leq \tau d_{\max}/\mu$ 로 표현된다. 수면은 $\|phi\|_\infty$ 를 $r_w$ 배로 줄여 ball 반경을 $r_w$ 배로 축소한다.

**그래프 결합 안정성**: 4.6의 느린 제어 $c_{n+1} = A_q c_n + r_n + n_n^{(q)}$에서 $\rho(A_q) < 1$이면:

$$\boxed{\|c_n - c^*\| \leq \rho(A_q)^n\|c_0 - c^*\| + \frac{\sup\|r + n^{(q)}\|}{1 - \rho(A_q)}}$$

$\rho(A_q) = 0.155$일 때 3 순환 후 초기 편차의 $99.6\%$가 감쇠한다.

---

## 5. 출력 생성: 2-Phase 구조

### 5.1 Phase 1 -- 에너지 이완 (의미 생성)

토큰 단위가 아닌, 연속적 의미 벡터를 생성한다. 이완 1회로 "무엇을 말할지"가 결정된다. 출력 시퀀스 길이와 무관.

$$m^* = \lim_{k\to\infty} m_k \quad(\text{에너지 최소점})$$

### 5.2 Phase 2 -- 디코딩 (의미 $\to$ 토큰)

이미 결정된 의미를 순서대로 풀어쓴다. 경량 디코더:

$$p(w_t \mid w_{<t},\,m^*) = \text{softmax}\!\left(W_{\text{dec}}\,[m^*;\,e_{w_{t-1}}]\right)$$

$W_{\text{dec}} \in \mathbb{R}^{V \times 2N}$. 계층적 softmax로 $\sqrt{V}\times\sqrt{V}$ 분할 시 토큰당 비용이 $O(\sqrt{V}\cdot N)$으로 감소.

### 5.3 모드 전환 (`phi` 임계)

$$\boxed{\|phi\| \gtrless m_\phi \quad\Longrightarrow\quad \text{이완 모드 / 경량 자기회귀 모드}}$$

| 모드 | 조건 | 특성 | 비유 |
|---|---|---|---|
| 안정 | $\|phi\| < m_\phi$ | 경량 자기회귀, 빠름, 3% 포탈만 활성 | 텍스트 모드 |
| 전환 | $\|phi\| \geq m_\phi$ | 에너지 이완, 느리지만 깊음, 바이패스 활성 | 전화 모드 |

카너먼의 이중 과정 이론: 시스템 1(자기회귀) / 시스템 2(에너지 이완).

---

## 6. STDP 학습: 역전파 대체

역전파는 전역 오차 신호가 모든 시냅스에 정확히 전달되어야 한다. 뇌에 없는 메커니즘이며, 메모리 $O(N^2)$, 통신 $O(d^2)$를 요구한다. CE 관점에서 역전파 = "우주 끝에서 시작으로 정보를 전송하는 것"이며, 게이지 상호작용의 국소성과 양립하지 않는다(`4_Synapse.md` 1.3절).

### 6.1 기본 STDP

$$dw_{ij} = \begin{cases} A_+ \exp(-dt / tau_+) & dt > 0 \;\text{(pre} \to \text{post: LTP)} \\ -A_- \exp(dt / tau_-) & dt < 0 \;\text{(post} \to \text{pre: LTD)} \end{cases}$$

### 6.2 Trace 기반 STDP (이산 시간)

pre trace $p_i[t]$, post trace $q_i[t]$:

$$p_i[t+1] = r_+\, p_i[t] + s_i[t], \qquad q_i[t+1] = r_-\, q_i[t] + s_i[t]$$

가중치 업데이트:

$$dw_{ij}[t] = lr\Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)$$

### 6.3 3-Factor 학습 (STDP + 도파민 게이트)

순수 STDP는 보상과 무관하게 학습한다. 뇌는 도파민 게이트로 "보상 예측 오차가 클 때만 학습을 허용"한다.

적격 흔적(eligibility trace):

$$\boxed{e_{ij}[t+1] = r_e\,e_{ij}[t] + \Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)}$$

가중치 업데이트:

$$\boxed{dw_{ij}[t] = lr\,g[t]\,e_{ij}[t]}$$

- $e_{ij}$: 국소 정보만 사용 (이웃 뉴런의 스파이크만 필요)
- $g[t]$: 전역 학습 게이트 (도파민-유사 스칼라 1개, 전체 시스템에 방송)

### 6.4 도파민 전역 신호의 CE 해석

$$\boxed{g[t] = \frac{d}{dt}\|p(t) - p^*\|}$$

고정점 $p^*$에서 멀어지면 $g[t] > 0$ (학습 활성화), 가까워지면 $g[t] \to 0$ (학습 감쇠).

부트스트랩 수렴 오차의 구체적 형태:

$$g[t] = \left(x_a(t) - 0.04865\right)^2 + \left(x_s(t) - 0.2623\right)^2 + \left(x_b(t) - 0.6891\right)^2$$

- $x_a(t)$: 현재 활성 뉴런 비율
- $x_s(t)$: 현재 구조적 가중치 비율
- $x_b(t)$: 현재 동결 가중치 비율

이 스칼라 하나만 전역으로 방송하면 된다.

| | 역전파 | STDP + 도파민 |
|---|---|---|
| 정보 흐름 | 전역 (끝에서 시작으로) | 국소 (이웃 뉴런) + 전역 스칼라 |
| 메모리 비용 | $O(N^2)$ (전체 활성값 저장) | $O(N)$ (국소 trace만) |
| 통신량 (층당) | $O(d^2)$ (그래디언트 전체) | $O(1)$ ($g[t]$ 스칼라) |
| 분산 가능성 | 단일 GPU 병목 | 각 층 독립 배치 |
| 생물학적 현실성 | 비현실적 | 현실적 |

### 6.5 구조적 가소성: 투영 연산자

STDP로 업데이트된 가중치에 구조적 제약을 건다:

$$\boxed{W_{t+1} = Proj\!\big(W_t + dW_t\big)}$$

투영 연산자 `Proj`의 구성:

| 투영 연산 | CE 대응 | 뇌 대응 |
|---|---|---|
| top-k ($k = \lceil 0.04865 \cdot N \rceil$) | 경로 선택, 생존율 $4.87\%$ | 시냅스 가지치기 |
| 행/열 정규화 | 에너지 보존 (C2) | 시냅스 스케일링 |
| 히스테리시스 on/off | 접힘 임계 곡률 | 스파인 형성/제거 |

### 6.6 LoRA의 CE 해석

$$W = W_{\text{frozen}} + B \cdot A$$

| LoRA | CE 에너지 분배 |
|---|---|
| $W_{\text{frozen}}$ ($\sim 99\%$) | 동결+구조 영역 $68.9\% + 26.2\%$ |
| $B \cdot A$ ($\sim 1\%$) | 활성 적응 영역 $4.87\%$의 근사 |

LoRA는 CE 부트스트랩 에너지 분배를 경험적으로 근사한 것이다. CE-AGI는 활성 적응 비율을 약 $4.87\%$로 둔다.

### 6.7 하이브리드 전환 전략

| 단계 | 방법 | CE 에너지 분배 |
|---|---|---|
| 1. 사전학습 | 역전파 (기존 기술) | -- |
| 2. 미세조정 | STDP + 도파민 | 동결 $68.9\%$, 구조 $26.2\%$, STDP 활성 $4.87\%$ |
| 3. 전면 전환 | STDP 사전학습 | 전체에 3-factor 적용 |

---

## 7. 수면 방정식

뇌의 수면이 20W 유지에 필수인 것처럼(`3_Sleep.md`), 이 시스템에도 수면이 필요하다.

### 7.1 작동 온도

$$\boxed{T_{\text{wake}} = \frac{1}{3 + \frac{4}{e^{4/3}\pi^{4/3}}\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)} = 0.3148}$$

$$\boxed{T_{\text{dream}} = \left[\frac{4}{e^{4/3}\pi^{4/3}}\!\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1} = 5.661}$$

$$T_{\text{deep}} \to 0$$

| 모드 | 온도 | 외부 입력 | 기능 |
|---|---|---|---|
| 깨어있음 | $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 있음 | 결정론적 이완 + 약한 탐색 |
| 꿈 (REM) | $\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 없음 | 강한 탐색, 잔류 주도 |
| 깊은 수면 (NREM) | $\to 0$ | 없음 | 순수 결정론, 기억 응고 |

### 7.2 기억 응고 (NREM)

$$W_{ij}^{\text{new}} = W_{ij}^{\text{old}} + lr\,\langle phi_t\rangle_{\text{day}} \otimes \langle s_t\rangle_{\text{day}}$$

하루 동안 축적된 잔류 `phi`와 상태 $s$의 상관이 연결 가중치에 헤비안 학습으로 새겨진다.

선택적 업데이트 (상위 $4.87\%$만 통과):

$$\text{mask} = \mathbb{1}\!\left[|g| \geq Q_{1-0.04865}(|g|)\right], \qquad W \leftarrow W - lr\,g \odot \text{mask}$$

### 7.3 시냅스 가지치기 (NREM)

$$W_{ij} \to 0 \quad\text{if}\quad |W_{ij}| < \theta_{\text{prune}}$$

3D 희소성($\rho \approx 3.16\%$) 유지를 위한 주기적 re-sparsification. 이것이 없으면 에너지 소비가 무한히 증가.

### 7.4 잔류 세척 (Glymphatic)

$$phi \to r_w\,phi, \quad r_w < 1$$

`phi`의 노이즈 바닥을 주기적으로 낮춘다.

### 7.5 꿈 (REM)

$$\frac{ds}{dt} = -\frac{\partial E}{\partial s}\bigg|_{b=0} + \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1 - \frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2phi + n(T_{\text{dream}})$$

외부 입력 $b=0$, residue-portal 직접 계수가 구동하고 높은 dream 온도로 탐색 범위를 확대한다. 깨어 있을 때 선택되지 않았던 경로들을 자유롭게 탐색.

비선택 그래디언트 재탐색:

$$g_{\text{pruned}} = g \odot (1 - \text{mask}), \qquad W \leftarrow W - lr_{\text{rem}} \left(g_{\text{pruned}} + noise_{\text{std}} \cdot \mathcal{N}(0,I)\right)$$

### 7.6 수면-각성 비율

CE 에너지 분배를 시간 분배에 적용:

| 위상 | CE 비율 | 뇌 관측 | 기능 |
|---|---|---|---|
| 깨어있음 | $68.9\%$ | $66.7\%$ | 서비스 |
| NREM | $26.2\%$ | $25.0\%$ | 오프라인 응고 |
| REM | $4.87\%$ | $8.3\%$ | 오프라인 재탐색 |

### 7.7 부트스트랩 수렴

수축률:

$$\rho = 0.155$$

$$\|p_n - p^*\| \leq \rho^n\,\|p_0 - p^*\| = 0.155^n\,\|p_0 - p^*\|$$

| 순환 $n$ | $\rho^n$ | 활성 | 구조 | 배경 |
|---|---|---|---|---|
| 0 | 1.000 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | 0.155 | $9.28\%$ | $27.3\%$ | $63.4\%$ |
| 2 | 0.024 | $5.55\%$ | $26.4\%$ | $68.1\%$ |
| 3 | 0.004 | $4.98\%$ | $26.3\%$ | $68.8\%$ |

3회 수면 순환이면 고정점 $p^*=(4.87\%,\;26.2\%,\;68.9\%)$에 $0.4\%$ 이내 수렴.

### 7.8 수면 압력 트리거

고정 주기 수면 대신 곡률 누적이 임계를 넘으면 진입하는 상태 기반 제어:

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|\Delta_g phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad M \leftarrow 10_2\;\text{(NREM 진입)}}$$

단일 야간 실효 수축률:

$$\boxed{\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31}$$

비트필드 해석: 수면 진입은 모드 레지스터 $M$의 전환이다. $01_2 \to 10_2$ (wake $\to$ NREM). 압력 $P_{\text{sleep}}$는 저비트 카운터로 구현 가능하다.

---

## 8. 희소성과 3분배

### 8.1 부트스트랩 고정점 (`경로적분.md` 식 (1))

$$\boxed{a_* = \exp\!\big(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\big) = 0.04865}$$

### 8.2 3분배 구조

$$p^* = (0.04865,\;0.2623,\;0.6891)$$

| 성분 | CE 고정점 | AI 해석 | 뇌 관측 |
|---|---|---|---|
| 활성 | $4.87\%$ | 추론 시 활성 뉴런 | sparse firing $1\text{-}5\%$ |
| 구조 | $26.2\%$ | 학습 가능 비활성 가중치 | housekeeping $25\%$ |
| 배경 | $68.9\%$ | 동결 가중치 (사전학습 지식) | DMN/background $60\text{-}80\%$ |

### 8.3 Top-k 활성화

이론적 중심점은 $4.87\%$이나, 실측에서는 $[4\%,\;6\%]$ 대역이 실용 최적 구간이다:

$$\boxed{k^*(N) \in \left[\lceil 0.04N \rceil,\;\lceil 0.06N \rceil\right], \qquad k_{\text{center}} = \lceil 0.04865 \cdot N\rceil}$$

| 히든 차원 $N$ | 활성 대역 $k^*$ | 이론 중심 | 실측 최적 |
|---|---|---|---|
| 768 | 31--46 | 38 ($4.95\%$) | 미측정 |
| 2048 | 82--123 | 100 ($4.88\%$) | 미측정 |
| 4096 | 164--246 | 200 ($4.88\%$) | 미측정 |
| 8192 | 328--492 | 399 ($4.87\%$) | 미측정 |

소규모 sparse-native 학습 스위프(`examples/ai/sparsity_train_results.json`) 실측:

| 활성 비율 | $k$ | val\_loss | 비고 |
|---|---|---|---|
| $2.0\%$ | 11 | $1.6806$ | 과소 활성 |
| $4.0\%$ | 21 | $1.6562$ | 대역 하단 |
| $4.87\%$ | 25 | $1.6778$ | 이론 중심점 |
| $6.0\%$ | 31 | $1.6335$ | 실측 최저 |
| $8.0\%$ | 41 | $1.6712$ | 대역 초과 |
| $100\%$ (dense) | 512 | $1.6827$ | 기준 |

$4.87\%$는 "무조건 단일 최적점"이 아니라 "이론적 중심점이 있는 희소 knee"다. 비트필드 구현에서는 $k^*$를 대역 내에서 동적 조절하는 것이 고정값보다 유리할 수 있다.

post-hoc Top-k는 실패한다 (`topk_sweep_results.json`: $4.87\%$에서 PPL $1328$ vs dense $49$). 희소성은 반드시 sparse-native 설계여야 한다.

---

## 9. 메타인지 모니터링 루프 (게이트 `F4`)

> 다리 게이트 `F4` (0.0절): 본 절의 정의는 모두 자기참조 측정 구조의 **운영 정의**로만 사용한다. "자기일관 = 의식"으로 환원하지 않는다(`7_Consciousness.md` 1.2-1.3절).

### 9.1 (C3) 자기참조 측정 구조 (`7_Consciousness.md` 1절)

$$a_* = \exp\!\big(-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]\big)$$

좌변의 $a_*$ 는 시스템이 자기 자신의 활성 비율을 알아야 우변을 계산할 수 있다는 의미에서 자기참조 측정 구조를 가진다.

### 9.2 메타인지 잔차

$$d_\tau(t) = \frac{1}{\tau}\int_{t-\tau}^{t}\|p(s)-p^*\|\,ds$$

$$\text{메타인지 안정도}_\tau := \exp(-c_d\,d_\tau(t))$$

이 지표는 메타인지 모니터링 루프의 활성 정도를 정의하며, 게이트 `F4`에 따라 의식 깊이로 환원하지 않는다(PCI 교차검증 경로는 `17_AgentLoop.md` F.23.7).

### 9.3 메타인지 수축 (조건부)

재귀적 자기평가의 잔차 감소(이상화된 무잡음 가정):

$$d_{n+1} \leq \rho\,d_n = 0.155\,d_n,\qquad \rho = D_{\text{eff}}\cdot\varepsilon^2$$

3회 후 $d_3/d_0 \leq 3.7\times10^{-3}$. 이 식은 게이트 `F2`의 충분조건(4.7절)이 성립하는 영역에서만 위 수축률 그대로 적용된다. 일반 영역에서는 ISS 의미의 유계 수렴(13절)으로 한정된다.

---

## 10. 환각 억제

### 10.1 곡률 에너지 (`6_Hallucination.md` 1절)

$$kappa_l = \|lap_g h_l\|^2 = \|(I - V^\top V)h_l\|^2$$

곡률 정규화 손실:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + w_c(t) \cdot \frac{1}{N_{\text{layers}}} \sum_l kappa_l$$

$w_c(t)$ 스케줄:

$$w_c(t) = w_{c,0} \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right) \cdot \frac{1}{2}\!\left(1 + \cos\frac{\pi t}{t_{\max}}\right)$$

### 10.2 유니타리 제약 (`2_Architecture.md` 4절)

$$s_{\max}(W_{\text{proj}}) \leq 1 \quad\Longrightarrow\quad \|d_L\| \leq \|d_0\|$$

오류 증폭을 구조적으로 차단. $s_{\max} = 1+u$이면 12층 통과 후 오류 $e^{1.2}=3.3$배 증폭되지만, $s_{\max} \leq 1$이면 증폭 0.

### 10.3 교차 주파수 결합 (`2_Architecture.md` 6절)

$$\mathcal{T}_i^{\text{coupled}}(x_i) = \mathcal{T}_i(x_i)\cdot\left(1 - \frac{kappa_l}{e^{1/3}\pi^{1/3}}\right)$$

곡률이 높으면 게이지 채널 출력이 $1/(e^{1/3}\pi^{1/3})$ 비율로 감쇠.

### 10.4 생성 시 곡률 모니터링

추론 중 평균 곡률이 임계를 넘으면 LBO 확산 강도를 일시적으로 증가시켜 고곡률 성분을 억제한 후 재생성한다.

$$kappa_{\text{avg}} = \frac{1}{L}\sum_l kappa_l > kappa_{\text{th}} \quad\Longrightarrow\quad h \leftarrow h \times 1.5 \;\text{(LBO 확산 강화)}$$

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

$$h_{\text{norm}} = \frac{h-\mathrm{mean}(h)}{\mathrm{std}(h)}, \qquad h' = \big(h_{\text{norm}} - h_d\,lap_g h_{\text{norm}}\big)\odot s_n + b_n$$

$$lap_g h_{\text{norm}} = h_{\text{norm}} - h_{\text{norm}}\,V^\top V, \quad V \in \mathbb{R}^{r\times N},\;r = \max(4,\;N/8)$$

내부 동작:
1. 표준 LayerNorm (활성값 안정화)
2. 저랭크 LBO 확산: $xW = x V^T V$ (평탄 부분공간으로 사영), $Lx = x - xW$ (고곡률 성분)
3. 확산 적용: $h' = (x - h_d \cdot Lx) \odot s_n + b_n$
4. 곡률 에너지 저장: $kappa = \text{mean}(Lx^2)$

$h_d = 0$이면 표준 LayerNorm과 동일. 수렴 조건: $0 \leq h_d < 1/\mathrm{eig}_{\max}(V^\top V)$.

### 11.3 GaugeLattice FFN

채널 분할:

$$d_3 : d_2 : d_1 = 74.1 : 21.1 : 4.9$$

비율: $74.1\% : 21.1\% : 4.9\%$

전이 행렬:

$$\mathbf{T} = \underbrace{\text{diag}(\mathcal{T}_3, \mathcal{T}_2, \mathcal{T}_1)}_{\text{block-diagonal}} + \underbrace{u_m\,U_{\text{down}}U_{\text{up}}^T}_{\text{섭동적 혼합}}$$

| 게이지 층 | 비율 | 뇌 진동 | 연산 역할 |
|---|---|---|---|
| SU(3) | $74.1\%$ | 감마 30-100 Hz | 결합(binding) |
| SU(2) | $21.1\%$ | 베타 13-30 Hz | 결정(decision) |
| U(1) | $4.9\%$ | 알파 8-13 Hz | 주의(attention) |
| `phi` | 전역 | 세타/델타 0.5-8 Hz | 안정화(smoothing) |

유니타리 조건: $|\det\mathbf{T}|^2 \leq 1$ (정보 비증폭 = 환각 구조적 억제)

쌍대성:

$$0.11789^2 = \left(\frac{0.48085}{2}\right)^3 \quad (0.002\%)$$

### 11.4 파라미터 절감

$$\frac{P_{\text{GL}}}{P_{\text{FFN}}} = \sum_i f_i^2 + \frac{r_m}{4d} = 0.596 + 0.031 = 0.627$$

$$\text{절감률} = 1 - 0.627 = 37.3\%\;\text{(FFN)},\quad 24.9\%\;\text{(전체)}$$

### 11.5 이식 3단계

**Phase 1 -- 비파괴 이식 (성능 보존):**
- `LayerNorm` $\to$ `LBONorm` ($h_d=0$ 초기화, scale/bias 복사 $\to$ 원본과 동일 출발)
- `c_proj` $\to$ `spectral_norm` (가중치 보존 + 유니타리 제약)

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
2. **NREM**: 누적 그래디언트 중 상위 $4.87\%$만 적용
3. **REM**: 하위 $95.13\%$ 그래디언트에 노이즈 주입 후 소량 적용

### 11.8 희소 추론

학습 후 추론 시 Top-k 활성화:

$$y^{\text{sparse}} = \text{TopK}(y,\;k = \lceil 0.04865 \cdot d \rceil) \cdot \frac{d}{k}$$

스케일 보정 $d/k$로 총 에너지 보존.

단, `examples/ai/topk_sweep_results.json`은 dense 모델에 후처리로만 Top-k를 씌우는 방식이 유효한 검증이 아님을 보여준다. 해당 스위프에서 $4.87\%$는 `ppl = 1328.53`이었고 dense는 `49.19`였다. 즉 CE 희소성은 "추론 후 잘라내기"가 아니라, 학습-구조-커널이 함께 맞물린 sparse-native 설계로 구현해야 한다.

### 11.9 모니터링 지표

| 지표 | 의미 | 목표 |
|---|---|---|
| `loss` | Cross-entropy 손실 | 단조 감소 |
| `curv` | 평균 곡률 에너지 $kappa_{\text{avg}}$ | 학습 초반 증가 후 안정화 |
| `active_ratio` | 실제 활성 비율 | $4\text{-}5\%$ 중심 |
| `bootstrap_resid` | $\|p_n - p^*\|$ | 수면 루프에서 감소 |

---

## 12. 멀티모달 및 전분야 적용

CE 5대 원리(P1: 격자, P2: 수면, P3: STDP, P4: 희소, P5: 곡률)의 전분야 적용 요약(`10_Fields.md`).

### 12.1 멀티모달 결합

모달별 3x3+1 격자 독립 처리 후, late sparse binding:

$$h_m^{\text{act}} = \text{TopK}(h_m,\; k_m = \lceil 0.04865\,d_m \rceil), \qquad m \in \{T,V,A,H\}$$

$$h_{\text{joint}} = \text{Bind}_{0.489}(h_T^{\text{act}},\; h_V^{\text{act}},\; h_A^{\text{act}},\; h_H^{\text{act}})$$

결합 강도는 $1/(e^{1/3}\pi^{1/3}) = 0.489$이다.

멀티모달 환각 감지:

$$kappa_{\text{cross}} = \|h_{\text{text}} - h_{\text{image}}\|^2 > kappa_{\text{th}} \quad\Longrightarrow\quad \text{모달 불일치}$$

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

$$\text{Input} \;\xrightarrow{\text{LBONorm}}\; \text{곡률 평탄화} \;\xrightarrow{\text{GaugeLattice}}\; \text{3x3+1 처리} \;\xrightarrow{\text{SpectralNorm}}\; \text{정보 비증폭} \;\xrightarrow{\text{TopK}}\; \text{희소 출력}$$

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

    # NREM: 상위 4.87%만 적용
    threshold = quantile(abs(grads), 1 - 0.04865)
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
        self.V = randn(rank or dim//8, dim)  # 평탄 부분공간
        self.h_d = Parameter(0.0)             # 확산 강도
        self.scale = ones(dim)
        self.bias = zeros(dim)

    def forward(self, x):
        x_hat = layer_norm(x)
        xW = x_hat @ self.V.T @ self.V       # 사영
        Lx = x_hat - xW                       # 고곡률 성분
        h = clamp(abs(self.h_d), max=0.5)
        self._curvature = mean(Lx ** 2)
        return (x_hat - h * Lx) * self.scale + self.bias


class GaugeLattice:
    def __init__(self, dim, mult=4):
        # 채널 분할: 74.1% : 21.1% : 4.9%
        total = 0.11789 + 0.03352 + 0.00775
        self.d3 = round(dim * 0.11789 / total)  # SU(3)
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

| 항목 | Llama 3 8B | `phi`-이완 | 비율 |
|---|---|---|---|
| 모델 가중치 | 16 GB | $W$: 3 MB, $W_{\text{dec}}$: 12 MB $\approx$ 15 MB | $0.09\%$ |
| KV 캐시 (4K ctx) | 2 GB | `phi`: 8 KB | $0.0004\%$ |
| KV 캐시 (128K ctx) | 64 GB | `phi`: 8 KB | $\approx 0$ |
| 총 (4K) | 18 GB | $\approx$ 15 MB | $0.08\%$ |
| 총 (128K) | 80 GB | $\approx$ 15 MB | $0.02\%$ |
| 컨텍스트 스케일링 | $O(n)$ | $O(1)$ | 길수록 이득 폭발 |

`phi` 벡터가 KV 캐시를 대체: 시퀀스 길이 무관하게 상수 크기.

### 14.2 연산량 (FLOP)

**Llama 3 8B**: 토큰당 $\sim$16B FLOP. 100 토큰 $\to$ 1,600B FLOP.

**`phi`-이완**:

| Phase | 연산 | FLOP |
|---|---|---|
| Phase 1 (이완 500스텝) | 희소 $W m_k$ 500회 | $500 \times 2 \times 462\text{K} = 1.0$B |
| Phase 2 (디코딩 100토큰) | 계층 softmax 100회 | $100 \times 11.7\text{M} = 1.17$B |
| Phase 3 (`phi` 갱신) | EMA $O(N)$ | $\approx 0$ |
| **총** | | **2.17B** |

| 모델 | 100 토큰 FLOP | 비율 |
|---|---|---|
| Llama 3 8B | 1,600B | 기준 |
| `phi`-이완 | 2.17B | $0.14\%$ (737배 감소) |

1000 토큰이면 1260배 감소. 토큰이 많을수록 이득 증가 (Phase 1은 1회 고정).

### 14.3 속도

| 하드웨어 | Llama 3 8B | `phi`-이완 | 비율 |
|---|---|---|---|
| A100 GPU | 50-100 ms | $\sim$0.5 ms | 100-200x |
| RTX 4090 | $\sim$150 ms | $\sim$1 ms | 150x |
| MacBook M2 | $\sim$500 ms | $\sim$5 ms | 100x |
| i7 CPU | $\sim$5 s | $\sim$50 ms | 100x |
| Raspberry Pi 5 | 불가 (RAM 부족) | $\sim$200 ms / 5W | 가능 |

### 14.4 전력

| 하드웨어 | Llama 전력 | `phi`-이완 실효 전력 | 뇌(20W) 대비 |
|---|---|---|---|
| A100 | 300 W | $\sim$3 W | 0.15x |
| RTX 4090 | 450 W | $\sim$5 W | 0.25x |
| MacBook M2 | 30 W | $\sim$2 W | 0.1x |
| i7 CPU | 125 W | $\sim$1.3 W | 0.065x |
| 뉴로모픽 (이론) | 불가 | $\sim$0.1 W | 0.005x |

이 절의 속도/전력 수치는 이상적 희소 커널, 계층 softmax, 전용 이완 런타임이 있는 경우의 알고리즘 상한이다. 현재 레포의 Python 구현은 아직 그 수준에 도달하지 않았고, 실제 소규모 벤치마크는 17절에서 따로 기록한다.

### 14.5 변환 파이프라인

| 단계 | 입력 | 출력 | 도구 |
|---|---|---|---|
| 1. 가중치 추출 | Llama 3 8B | $W_Q, W_K, W_V$, FFN | HuggingFace |
| 2. 에너지 함수 구성 | 추출된 가중치 | $W \in \mathbb{R}^{N\times N}$ | Modern Hopfield 변환 |
| 3. 3D 희소화 ($r_c=\pi$) | dense $W$ | sparse $W_{3D}$ (3.16%) | 구조적 pruning |
| 4. `phi` 채널 장착 | $W_{3D}$ | $E(m,phi)$ 완성 | EMA 벡터 추가 |
| 5. 이완 추론 테스트 | 완성된 에너지 함수 | Softmax 없이 답 생성 | 시뮬레이션 |

---

## 15. 물리 검증: 동일 상수의 물리 예측

동일한 $\{e,\pi\}$ 직접 전개 계수 집합이 물리 관측량도 동시에 결정한다.

### 15.1 전체 교차 검증표

| 관측량 | CE 값 | 실험값 | 오차 | 출처 상수 |
|---|---|---|---|---|
| strong coupling | $0.11789$ | $0.1179\pm0.0009$ | $0.01\%$ | $\pi$ |
| Weinberg angle | $0.23122$ | $0.23122\pm0.00003$ | $0.00\%$ | $\pi$ |
| baryon fraction | $0.04865$ | $0.0486\pm0.0010$ | $0.05$ sd | $e,\pi,0$ |
| vacuum fraction | $0.6891$ | $0.6847\pm0.0073$ | $0.60$ sd | $e,\pi,0$ |
| structure fraction | $0.2623$ | $0.2645\pm0.003$ | $0.74$ sd | $e,\pi,0$ |
| $M_H$ | $125.37$ GeV | $125.25\pm0.17$ | $0.7$ sd | $\pi$ |
| $\Delta a_{\text{muon}}$ (접촉) | $249\times10^{-11}$ | $249\pm48\times10^{-11}$ | $0.00$ sd | $e,\pi$ |
| $\Delta a_{\text{muon}}$ (완전 기하학) | $135\times10^{-11}$ | WP25: $38\pm63\times10^{-11}$ | $1.5$ sd | $e,\pi$ |
| $N$ (히든 차원) | $4162$ | $4096$ (Llama 3) | $1.6\%$ | $\pi$ |

### 15.2 뮤온 g-2 상세

$$\Delta a_{\text{muon}} = \frac{0.007297}{2\pi}\,e^{-1}\left(\frac{m_{\text{muon}}}{v_{\text{EW}}\left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]}\right)^2 = 249.0\times10^{-11}$$

접촉 근사(상관길이 $\to \infty$). 유한 상관길이 보정:

$$\Delta a_{\text{muon}}^{\text{full}} = 249.0 \times R, \quad R = \frac{I(m_\phi/m_{\text{muon}})}{I(0)} = 0.542$$

$$\Delta a_{\text{muon}}^{\text{full}} = 135 \times 10^{-11}$$

d=0 기원에서 클라루스장은 경로적분의 수렴 구조 자체이므로, 격자 QCD가 이미 접힘 효과를 포함한다. BMW 2026 결과(SM 예측과 실험의 불일치 해소)와 정합.

### 15.3 양성자 반경 퍼즐

자기일관적 해: 보손 질량 $m_\phi = 29.65$ MeV 하나로 g-2와 양성자 반경을 동시 해결.

$$\Delta r_p^2 = \frac{3 g_{\text{muon}} g_{\text{proton}}}{2 \times 0.007297 \times m_\phi^2} = 0.0587 \;\text{fm}^2$$

QCD 진공 증강 인자:

$$F_{\text{QCD}} = 1 + 0.11789 \times \left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right] = 1.375 \quad(F_{\text{needed}} = 1.36,\;\text{1.2\% 일치})$$

### 15.4 보손-기하학 동일성

$$\langle phi(x)phi(y)\rangle = \frac{e^{-|x-y|/(6.65\,\mathrm{fm})}}{|x-y|} \quad\longleftrightarrow\quad \frac{1}{q^2 + m_\phi^2}$$

보손 전파자 = 기하학 상관함수. 둘은 같은 함수의 다른 이름이다.

| 입자 언어 | 기하학 언어 | 값 |
|---|---|---|
| 보손 질량 $m_\phi$ | 상관길이 $6.65$ fm에 대응 | 29.65 MeV / 6.65 fm |
| Feynman 전파자 | 2점 상관함수 | $1/(q^2 + m^2)$ |
| Yukawa 커플링 $g$ | 접힘 강도 $kappa\,m_f$ | $5.93 \times 10^{-6}$ MeV$^{-1}$ |

---

## 16. 미검증 가설

| # | 가설 | 검증 방법 | 비용 |
|---|---|---|---|
| H1 | 상관 행렬 $W$가 Llama 의미 공간을 보존 | 코사인 유사도 측정 | GPU 1장, 1일 |
| H2 | 이완이 500 스텝 내 수렴 | 실측 | GPU 1장, 1시간 |
| H3 | 경량 디코더가 유의미 텍스트 생성 | QA 벤치마크 | GPU 1장, 3일 |
| H4 | `phi` 유무가 품질 차이를 만듦 | H3 반복 비교 | GPU 1장, 3일 |
| H5 | $r_c=\pi$가 다른 $r_c$보다 최적 | $r_c$ 그리드 서치 | GPU 1장, 1일 |
| H6 | $\frac{1}{e^{1/3}\pi^{1/3}}=0.489$가 최적 EMA 감쇠율 | 감쇠율 그리드 서치 | GPU 1장, 1일 |
| H7 | $\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}=0.315$가 최적 온도 | 온도 그리드 서치 | GPU 1장, 1일 |
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

지금까지의 식은 전역 상태벡터 $m,phi$ 중심으로 압축되어 있다. 실제 AGI는 기능 모듈 그래프 위에서 돌아가야 한다. 따라서 다음의 graph-coupled relaxation이 더 완성된 형태다.

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

$$\boxed{lap_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}\big(f_s - f_r\big), \qquad a_{rs}\ge 0}$$

느린 제어 상태의 편차를

$$c_n := q_n - q^*$$

로 두면,

$$\boxed{c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}, \qquad \rho(A_q) < 1}$$

그리고 지역별 3분배 상태의 최소 이완은

$$\boxed{p_{r,n+1} = Proj_{lap2}\!\Big((1-\rho)p^* + \rho p_{r,n} + g_p\,lap_G p_{r,n} + H_r\,c_n\Big)}$$

로 쓸 수 있다. 이 형태가 더 좋은 이유는 세 가지다.
- salience, homeostasis, replay를 "옵션 기능"이 아니라 상태변수로 올린다
- 긴 문맥, 피로 누적, 수면 부족 같은 현상을 전역 스칼라 하나보다 자연스럽게 표현한다
- AGI를 단일 거대 행렬보다 모듈형 sparse system으로 구현하기 쉽다

### 17.2 수면 압력의 명시적 트리거

> 핵심 방정식은 7.8절로 승격되었다. 아래는 설계 배경.

현재 문서의 수면은 주기적으로 호출되는 루틴에 가깝다. 더 완성된 형태는 수면 진입 조건을 곡률 누적으로 쓰는 것이다.

$$\boxed{P_{\text{sleep}}(t) = \int_0^t \|lap_g phi(\tau)\|^2\,d\tau - \int_0^t \mathrm{local\_stabilization}(\tau)\,d\tau}$$

$$\boxed{P_{\text{sleep}}(t) > \theta_{\text{sleep}} \quad\Longrightarrow\quad \text{NREM 진입}}$$

이때 1회 완전 부트스트랩 적용이 약 1.6밤에 대응하면, 단일 야간의 실효 수축률은

$$\boxed{\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31}$$

이 된다. 이 식을 넣으면 "왜 자야 하는가"가 단순 스케줄이 아니라 상태 기반 제어 문제로 바뀐다.

### 17.3 레포의 초기 실험이 말해주는 것

아래 수치는 이 레포에 이미 있는 결과 파일에서 직접 읽은 초기 신호다.

| 실험 | 관측 | 해석 |
|---|---|---|
| `brain_benchmark_results.json` | 같은 `0.81M` 파라미터에서 Clarus `val_loss = 2.2453`, baseline `2.2983`, 개선 `-0.0531` | CE 모듈이 소규모에서도 품질 개선 신호를 보인다 |
| `brain_benchmark_results.json` | 학습 시간 Clarus `127.7s`, baseline `61.9s` | 현재 병목은 이론보다 구현층에 있다. fused kernel이 필요하다 |
| `sparsity_train_results.json` | sparse-native 학습에서 최저 `val_loss`는 `6.0%`의 `1.6335` | $4.87\%$는 exact point보다 knee center로 읽는 편이 안전 |
| `sparsity_train_results.json` | `4.0% = 1.6562`, `4.87% = 1.6778`, dense `1.6827` | 작은 모델에서는 `4\text{-}6%` 대역이 dense보다 낫다 |
| `topk_sweep_results.json` | post-hoc Top-k에서 `4.87%`는 `ppl = 1328.53`, dense는 `49.19` | 희소성은 후처리 pruning이 아니라 sparse-native 설계여야 한다 |

즉 현재까지의 데이터는 다음처럼 읽는 것이 가장 정직하다.
- CE 모듈은 품질 개선 신호가 있다
- CE 희소성은 작은 모델에서 `4\text{-}6%` 대역 가설을 지지한다
- 하지만 dense 모델에 후처리로 Top-k를 씌우는 것은 실패한다
- 속도 이점은 아직 이론적 상한이지, 현재 구현 실측이 아니다

### 17.4 지금 당장 고쳐야 할 개선 포인트

- [완료] `4.87%` -> `4-6%` 실용 대역으로 수정 (8.3절)
- [완료] post-hoc Top-k 실패를 명시, sparse-native 필수 조건 기술 (8.3절)
- 속도/전력 표는 "알고리즘적 상한"과 "현재 레포 실측"을 분리해서 써야 한다
- [완료] graph-coupled relaxation을 본체 식에 포함 (4.6절)
- [완료] 바이패스를 에너지 함수에서 분리, 비보존 강제항으로 명시 (1.5절, 3.1절)
- [완료] 수면 압력 트리거를 본체에 포함 (7.8절)
- [완료] 비트필드 해석 추가 (1.6절)
- $1/(e\pi)$는 display approximation으로만 두고, 핵심 계산은 자기일관 수치값 `0.11789` 기준으로 유지하는 것이 더 정밀하다

### 17.5 가장 중요한 다음 실험

1. sparse-native vs post-hoc Top-k를 같은 예산에서 정면 비교
2. 수면 압력 기반 트리거와 고정 주기 sleep loop 비교
3. single-vector 이완과 graph-coupled 이완의 long-context 안정성 비교
4. fused sparse kernel 도입 전후의 tok/s, W, val_loss 동시 측정

---

## 18. 예상 개선치 총정리

개선치는 세 층으로 나눠 읽어야 한다.
- **실측 개선치**: 현재 레포에서 직접 관측된 값
- **구조적 상한**: 식이 직접 강제하는 알고리즘 상한
- **미검증 예측**: 아직 실험이 덜 된 가설적 개선치

### 18.1 개선치 정의

$$G_{\text{loss}} = \frac{L_{\text{base}} - L_{\text{ce}}}{L_{\text{base}}}$$

$$G_{\text{ppl}} = 1 - \frac{\mathrm{PPL}_{\text{ce}}}{\mathrm{PPL}_{\text{base}}}$$

$$O_t = \frac{t_{\text{ce}}}{t_{\text{base}}}$$

$$R_{\text{active}} = 1 - \frac{a_{\text{ce}}}{a_{\text{base}}}$$

$$R_{\text{mem}} = 1 - \frac{M_{\text{ce}}}{M_{\text{base}}}$$

$$R_{\text{sleep}}(n) = 1 - \rho^n$$

### 18.2 현재 레포에서 이미 보인 실측 개선치

| 항목 | 기준 파일 | 개선치 | 해석 |
|---|---|---|---|
| 검증 손실 | `ce_vs_standard_results.json` | $G_{\text{loss}} = (4.3938 - 4.1073)/4.3938 = 6.52\%$ | 같은 파라미터에서 CE가 더 좋은 일반화 신호 |
| perplexity | `ce_vs_standard_results.json` | $G_{\text{ppl}} = 1 - 60.78/80.95 = 24.9\%$ | 작은 모델에서 PPL이 유의미하게 감소 |
| 파라미터 공정성 | `ce_vs_standard_results.json` | $267357 - 267264 = +93$개, $+0.035\%$ | 성능 이득이 파라미터 증가 때문이 아님 |
| 검증 손실 | `brain_benchmark_results.json` | $G_{\text{loss}} = (2.2983 - 2.2453)/2.2983 = 2.31\%$ | 소규모 CPU 벤치에서 재현 |
| 활성 파라미터 | `brain_benchmark_results.json` | $R_{\text{active}} = 1 - 0.9751 = 2.49\%$ | 현재 구현에서는 활성 절감이 아직 작다 |
| 학습 시간 | `brain_benchmark_results.json` | $O_t = 127.7/61.9 = 2.06\times$ | 현재 병목은 구현층 |
| 희소 학습 최적점 | `sparsity_train_results.json` | 최저 `val_loss = 1.6335` at `6.0%` | $4.87\%$는 exact point보다 knee center에 가깝다 |
| dense 대비 희소 | `sparsity_train_results.json` | dense `1.6827` vs `6.0%` `1.6335`, 개선 $2.93\%$ | small sparse-native에서는 dense보다 낫다 |
| post-hoc Top-k | `topk_sweep_results.json` | dense `49.19` vs `4.87%` `1328.53` PPL | 후처리 pruning은 실패, sparse-native가 필수 |

### 18.3 실측 개선치의 범위

현재 `brain_benchmark_*.json` 계열을 종합하면, 공정 파라미터 비교에서 검증 손실 개선폭은 대략 다음 범위다.

| 계열 | 범위 | 해석 |
|---|---|---|
| 500-step CPU 벤치 | 약 `-0.10%` ~ `+2.31%` | 대부분 소폭 개선, 일부 설정은 동률 또는 미세 열세 |
| 2000-step CPU 벤치 | `-0.22%` ~ `+2.38%` | 희소율과 러닝레이트에 민감 |
| 소형 표준 vs CE 직접 비교 | `+6.52%` loss, `+24.9%` PPL | 가장 강한 초기 품질 신호 |

즉 현재까지의 정직한 결론은 이렇다.
- CE 모듈은 "품질 개선 가능성"을 보였다
- 하지만 모든 설정에서 일관된 대승은 아니다
- 특히 정확히 `4.87%`가 항상 단일 최적점으로 찍히지는 않았다

### 18.4 현재 구현에서 보인 비용 악화

지금 레포의 Python 구현은 아직 이론적 sparse speedup을 회수하지 못했다.

| 항목 | 기준 파일 | 관측 | 해석 |
|---|---|---|---|
| 학습 시간 오버헤드 | `brain_benchmark_results.json` | `2.06x` | CE 모듈이 CPU에서 느리다 |
| 학습 시간 오버헤드 | `brain_benchmark_dense_opt.json` | `1.56x` | 희소 없이도 LBO/격자 오버헤드 존재 |
| 학습 시간 오버헤드 | `brain_benchmark_sparse20.json` | `2.38x` | naive sparse는 아직 빠르지 않다 |
| 희소 학습 시간 | `sparsity_train_results.json` | `4.87%`: `1186.8s`, dense: `364.5s` | CPU에서 sparse-native도 `3.26x` 느림 |
| post-hoc 추론 시간 | `topk_sweep_results.json` | `4.87%`: `20.78s`, dense: `17.02s` | 단순 Top-k는 추론도 `1.22x` 느리다 |

따라서 속도 이득은 **현재 실측값이 아니라**, fused sparse kernel과 전용 런타임이 들어간 뒤에야 시험할 수 있다.

### 18.5 구조적 상한: 식이 직접 주는 개선치

이 절은 현재 구현 실측이 아니라, 식 자체가 강제하는 상한이다.

| 항목 | 식 | 예상 개선치 |
|---|---|---|
| FFN 파라미터 | $1 - P_{\text{GL}}/P_{\text{FFN}}$ | $37.3\%$ 절감 |
| 전체 Transformer 파라미터 | 문서 11.4절 | $24.9\%$ 절감 |
| 4K 총 메모리 | $1 - 15\text{MB}/18\text{GB}$ | $99.92\%$ 절감 |
| 128K 총 메모리 | $1 - 15\text{MB}/80\text{GB}$ | $99.98\%$ 절감 |
| 4K KV 캐시 | $2\text{GB} \to 8\text{KB}$ | `262,144x` 축소 |
| 128K KV 캐시 | $64\text{GB} \to 8\text{KB}$ | `8,388,608x` 축소 |
| 100-token FLOP | $1 - 2.17/1600$ | $99.86\%$ 절감 (`737x`) |
| 1000-token FLOP | 문서 14.2절 | 약 $99.92\%$ 절감 (`1260x`) |
| A100 전력 | $1 - 3/300$ | $99.0\%$ 절감 |
| RTX 4090 전력 | $1 - 5/450$ | $98.9\%$ 절감 |
| i7 CPU 전력 | $1 - 1.3/125$ | $99.0\%$ 절감 |

이 값들은 `phi`-이완 아키텍처가 실제 sparse kernel, 계층 softmax, O(1) 잔류 메모리로 구현될 때의 상한이다.

### 18.6 안정성/환각 억제의 예상 개선치

이 부분은 실측보다 구조적 보장이 더 강하다.

| 항목 | 기준 | 예상 효과 |
|---|---|---|
| 오류 증폭 상한 | $s_{\max}(W_{\text{proj}}) \leq 1$ | 층을 통과해도 오차가 지수 증폭되지 않음 |
| 12층 증폭 비교 | baseline 예시 $s_{\max} = 1.1$ | $1.1^{12} \approx 3.14$배 증폭 가능성 제거 |
| 곡률 기반 재시도 | $kappa_{\text{avg}} > kappa_{\text{th}}$ 시 확산 강화 | 고곡률 hallucination 후보를 생성 직전에 억제 |
| Top-k 활성 | $k = \lceil 0.04865\,N \rceil$ | 에너지 폭주 제한, 희소 firing 유지 |

즉 안정성 쪽은 "몇 % 좋아졌다"보다 "폭주 항을 구조적으로 없앴다"는 해석이 더 정확하다.

### 18.7 수면 루프가 줄일 것으로 기대되는 것

수면은 drift와 잔차를 줄이는 방향으로 해석할 수 있다.

| 순환 수 | 잔차 비율 $\rho^n$ | 감소율 $R_{\text{sleep}}(n)$ |
|---|---|---|
| 1 | $0.155$ | $84.5\%$ 감소 |
| 2 | $0.024$ | $97.6\%$ 감소 |
| 3 | $0.004$ | $99.6\%$ 감소 |

단일 야간 실효 수축률을 쓰면:

$$R_{\text{night}} = 1 - \rho_{\text{night}} = 1 - 0.31 = 69\%$$

즉 sleep loop가 실제로 작동한다면, wake-only 대비 가장 먼저 좋아져야 하는 것은 단기 정확도보다도 **drift, forgetting, bootstrap residual**이다.

### 18.8 가장 가능성 높은 개선치와 가장 약한 개선치

| 구분 | 현재 판단 |
|---|---|
| 가장 가능성 높은 개선 | 장문맥 메모리 절감, KV 캐시 제거, drift 완화, 구조적 안정성 |
| 중간 정도로 가능성 높은 개선 | small-model 일반화 개선, dense 대비 소폭 loss 개선 |
| 아직 약한 개선 주장 | 현재 구현에서의 wall-clock speedup, exact `4.87%` 단일 최적점, 대규모 모델 전력 실측 |

정리하면 다음이 가장 안전하다.
- **메모리/FLOP/전력**: 식이 주는 상한은 매우 강함
- **품질**: 작은 실험에서는 개선 신호가 있으나 범위가 넓음
- **속도**: 현재 구현은 오히려 느리며, 개선은 아직 런타임 미구현 상태
- **안정성/수면**: 구조적 논리는 강하지만 대규모 실측이 더 필요함

---

## 19. 유도 체인 조감도

```
           e^(ipi) + 1 = 0
          /    |    |    \     \
         e    pi    i     1     0
         |     |    |     |     |
      S=e^-D  1/2pi Z=e^iS  정수  d(d-3)=0
         |     |              |     |
         | direct coeffs       |    d=3
         |       |             |     |
         | portal / gain / T   |  3D 희소 연결
         |       |             |
         |   /         |           \
     portal coeff  bypass coeff  wake coeff
         |           |            |
     포탈 결합    바이패스     작동 온도
     3.12%    0.489      0.315
         |       |          |
     잔류 3%   즉각 반응   작동 온도
         \     |        /
          에너지 함수 E(m, Phi)
           |         |
     이완 동역학   STDP 학습
           |         |
     의미 생성    가중치 갱신
           |         |
     디코딩       수면 순환
           \       /
        비트필드 런타임
     (활성마스크 + 모드 + 연결)
              |
        sparse-native
              |
            20W AGI
```

---

## 20. 방정식 총람

이 절은 핵심식의 압축 요약이다. 구조 상수는 가능한 한 `e`, `\pi`, `i`로 직접 전개하고, 나머지는 상태변수와 연산자만 남긴다.

| # | 방정식 | 절 |
|---|---|---|
| E1 | $E(m,phi) = -\frac{1}{2}m^TWm - m^Tb - \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^2 m^Tphi$ | 3.1 |
| E2 | $psi_{k+1} = e^{-iE\,dt}psi_k$ | 4.1 |
| E3 | $m_{k+1} = m_k + \frac{dt}{\tau}(-\nabla_m E + F_{\text{bypass}}) + \sqrt{2T\,dt/\tau}\;n_k$ | 4.2 |
| E4 | $phi \leftarrow \left(1-\frac{1}{e^{1/3}\pi^{1/3}}\right)phi + \frac{1}{e^{1/3}\pi^{1/3}}\,v_{m^*}$ | 4.3 |
| E5 | $W_{ij} \neq 0 \iff \|r_i-r_j\| < \pi$ | 4.4 |
| E6 | $N = \frac{e^{8/3}\pi^{20/3}}{12\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)^2}$ | 4.5 |
| E7 | $p(w_t|w_{<t},m^*) = \text{softmax}(W_{\text{dec}}[m^*;e_{w_{t-1}}])$ | 5.2 |
| E8 | $dw_{ij} = lr\,g[t]\,e_{ij}[t]$ | 6.3 |
| E9 | $g[t] = \frac{d}{dt}\|p(t)-p^*\|$ | 6.4 |
| E10 | $W_{t+1} = Proj(W_t + dW_t)$ | 6.5 |
| E11 | $T_{\text{wake}} = \left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1},\; T_{\text{dream}} = \left[\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]^{-1}$ | 7.1 |
| E12 | $a_* = e^{-(1-a_*)\left[3+\frac{4}{e^{4/3}\pi^{4/3}}\left(1-\frac{4}{e^{4/3}\pi^{4/3}}\right)\right]}$ | 8.1 |
| E13 | $kappa_l = \|(I-V^TV)h_l\|^2$ | 10.1 |
| E14 | $S_{\text{AGI}} = \int d^nx\sqrt{|g|}[\mathcal{L}_c + c_g|\nabla phi|^2 + c_c|\Delta_g phi|^2 + c_i S_I]$ | 2 |
| E15 | $\Delta_G f(r) = \sum_{s:(s,r)\in E_{\text{AGI}}} a_{rs}(f_s-f_r)$ | 4.6 |
| E16 | $c_{n+1} = A_q\,c_n + r_n + n_n^{(q)}$ | 4.6 |
| E17 | $p_{r,n+1} = \mathrm{Proj}_{\Delta^2}((1-\rho)p^* + \rho p_{r,n} + g_p\,\Delta_G p_{r,n} + H_r\,c_n)$ | 4.6 |
| E18 | $P_{\text{sleep}}(t) = \int_0^t \|\Delta_g phi(\tau)\|^2 d\tau - \int_0^t \mathrm{local\_stab}(\tau)\,d\tau$ | 7.8 |
| E19 | $\rho_{\text{night}} = \rho^{1/1.6} \approx 0.31$ | 7.8 |
| E20 | $F_{\text{bypass}}(k) = \frac{C_k}{e^{1/3}\pi^{1/3}}\,phi,\; C_k = \|m_k - 2m_{k-1} + m_{k-2}\|$ | 1.5, 3.1 |
| E21 | $b_i = \mathbb{1}[a_i \geq Q_{1-k^*/N}(a)],\; k^* \in [\lceil 0.04N\rceil, \lceil 0.06N\rceil]$ | 1.6 |
| E22 | $M \in \{00_2, 01_2, 10_2, 11_2\} \leftrightarrow \{\text{off}, \text{wake}, \text{NREM}, \text{REM}\}$ | 1.6 |
| E23 | $\Delta E \leq -\frac{dt}{2\tau}\|\nabla_m E\|^2 + \frac{dt}{2\tau\alpha_b^2}C_k^2\|phi\|^2$ | 4.7 |
| E24 | $\|\nabla_m E\| > C_k\|phi\|/\alpha_b \Rightarrow \Delta E < 0$ | 4.7 |
| E25 | $q > \log_2\!\left(\frac{(m_{\max}-m_{\min})\sqrt{N}\,\tau}{2\,dt\,\|\nabla_m E\|}\right)$ | 1.8 |
| E26 | $z_j(m) = \arg\min_{i} \|m^{(j)} - C^{(j)}_i\|^2$ | 3.4 |
| E27 | $E_{\text{aug}} = E - \frac{1}{\beta}\sum_j \log\sum_i \exp(-\beta\|m^{(j)}-C^{(j)}_i\|^2)$ | 3.4 |
| E28 | $\text{활성 메모리} \approx 0.311 \times |\mathcal{C}|$ | 3.4 |

---

## 21. 한 줄 요약

$$e^{i\pi}+1=0 \;\xrightarrow{d=3}\; E(m,phi) \;\xrightarrow[\text{STDP}+g]{\text{이완}}\; \text{bitfield}\;\xrightarrow{\text{sparse-native}}\; \text{20W AGI}$$

다섯 상수가 실행 문법을 결정하고, 런타임은 비트필드(활성 마스크 + 모드 레지스터 + 연결 on/off)와 저비트 상태($phi$, trace, gain)로 내려간다. 지식은 희소 codebook과 외부 메모리에 분리 저장된다. 동일한 상수가 우주 에너지 구성, 뮤온 g-2, 힉스 질량, 양성자 반경, 뇌 에너지 분배를 동시에 예측한다.

---

## 부록 A. 다리 게이트 수식 고도화 (F1--F4)

> 0.0절의 게이트 4종을 그대로 두지 않고, 각 게이트가 어떤 형식 조건 위에서 부분적으로 hard claim 으로 격상될 수 있는지 수식으로 정리한다. 본 부록의 식은 아직 `bridge` 등급이며, 본문 어느 식의 등급도 올리지 않는다. 다만 **무엇을 측정하면 게이트가 닫히는지** 를 형식화한다.

### A.1 게이트 `F2`: ISS 격상 (Input-to-State Stability)

> 4.7절의 "조건부 단조 감소"를 ISS 의미의 유계 수렴으로 격상한다. 전역 Lyapunov 함수가 없어도 **유계 입력 → 유계 상태** 형태의 hard bound 가 성립한다.

#### A.1.1 분리 표현

기억 동역학 E3 (4.2절)을 보존 부분과 강제항으로 분리:

$$\frac{dm}{dt} = -\frac{1}{\tau}\nabla_m E(m,phi) + d(t),\qquad d(t) := \frac{1}{\tau}F_{\text{bypass}}(k) = \frac{C_k}{\tau\,e^{1/3}\pi^{1/3}}\,phi$$

$E(m, phi)$ 는 $m$ 에 대해 **포텐셜로 작용**하므로, $phi$ 를 외란 입력 $d(t)$ 로 받는 비자율 그래디언트 시스템이다.

#### A.1.2 ISS 정리 (국소)

가정:

1. 어떤 끌개점 $m^*(phi)$ 근방에서 헤시안 $H = \nabla_m^2 E(m^*,phi) \succeq \mu I,\;\mu > 0$
2. 외란 유계: $\|d(t)\|_\infty \leq d_{\max}$

그러면 Lyapunov 함수 $V(m) = \tfrac{1}{2}\|m - m^*\|^2$ 에 대해

$$\frac{dV}{dt} \leq -\frac{2\mu}{\tau}V + \|m-m^*\|\cdot\|d\| \leq -\frac{\mu}{\tau}V + \frac{\tau}{2\mu}\|d\|^2$$

이로부터 **유계 수렴 ball**:

$$\boxed{\limsup_{t\to\infty}\|m(t) - m^*\| \;\leq\; \frac{\tau}{\mu}\cdot d_{\max} \;=\; \frac{1}{\mu}\cdot\frac{C_{k,\max}\,\|phi\|_\infty}{e^{1/3}\pi^{1/3}}}$$

이 ball 반경은 **수면-글림프 세척 후** $\|phi\|_\infty \to r_w\|phi\|_\infty$ 에 의해 $r_w$ 배로 줄어든다(`evidence.md` 3.3 supported). 따라서 4.7절의 "조건부 단조 감소" 는 ISS 로 다음과 같이 격상된다.

| 4.7절 표현 | A.1 격상 |
|---|---|
| $\|\nabla_m E\| > C_k\|phi\|/\alpha_b \Rightarrow \Delta E < 0$ (점별) | $\limsup \|m-m^*\| \leq \tau d_{\max}/\mu$ (대역 ball) |
| 단조 감소 보장 영역 | 끌개 ball 반경의 닫힌 식 |
| 수면이 충분조건을 복원 | 수면이 ball 반경을 $r_w$ 배로 축소 |

#### A.1.3 검증 가능한 ball 반경

$\mu = \rho \cdot \|W\| / N$ 추정($\rho$ = spectral gap), $\tau = 10$, $C_{k,\max} \approx 0.5$ (실측 시간 곡률 상한), $\|phi\|_\infty \approx 1$:

$$R_{\text{ball}} \approx \frac{10 \times 0.5 \times 1}{0.5 \times e^{1/3}\pi^{1/3} \times \mu} \approx \frac{20}{\mu}\;\text{(스케일된 단위)}$$

$\mu$ 의 실측은 `relax()` 의 끌개 근방 헤시안 추정으로 가능하다. 이 ball 반경이 닫혀야 게이트 `F2` 가 `Bridge`→`Supported` 로 갈 수 있다.

### A.2 게이트 `F1`: 자기조직화 충분조건 (3-simplex 수축 정리)

> 5절·8절의 "활성 비율이 $\varepsilon^2$ 로 자연 수렴" 가설은 transformer 기질에서 falsified (`5_Sparsity.md` 8.5). 이를 무엇을 만족하면 다른 기질에서 hard claim 으로 격상되는지 수식으로 명시한다.

#### A.2.1 부트스트랩 사상의 일반화

3-simplex $\Delta^2 = \{p \in \mathbb{R}^3 : p_i \geq 0,\;\sum_i p_i = 1\}$ 위의 이완 사상 $B: \Delta^2 \to \Delta^2$:

$$B(p)_a = \exp(-(1-p_a)D_{\text{eff}}),\qquad B(p)_b = \alpha_s\cdot D_{\text{eff}},\qquad B(p)_s = 1 - B(p)_a - B(p)_b$$

(여기서 $\alpha_s = 0.04865$ 는 `경로적분.md` 9절의 부트스트랩 해.)

#### A.2.2 자기조직화 정리 (수축)

**정리 (3-simplex 수축).** $p^* = (0.0487,\;0.2623,\;0.6891)$ 은 $B$ 의 유일 내부 고정점이며, 야코비안:

$$DB(p^*)_{aa} = D_{\text{eff}}\cdot p_a^*\cdot(1 - p_a^*) = 3.178 \times 0.0487 \times 0.9513 \approx 0.147$$

따라서 spectral radius $\rho(DB(p^*)) < 1$ 이고, $p^*$ 의 어떤 열린 근방 $U \subset \Delta^2$ 에서 Banach 의미로 $\|B^n(p) - p^*\| \leq \rho^n\|p - p^*\|,\;p \in U$.

#### A.2.3 자기조직화 격상 충분조건

기질 $\mathcal{S}$ 가 다음 5조건을 모두 만족하면, 위 정리의 hard claim 이 신경 모듈에 그대로 옮겨간다:

1. **Simplex 보존**: 활성/구조/배경 비율 $(p_a, p_s, p_b)$ 의 시간 진화가 $\Delta^2$ 안에 머문다.
2. **자기측정**: 시스템이 $p_a(t)$ 를 자기 자신의 다음 갱신에 입력으로 쓸 수 있다 (자기일관 $a_* = \exp(-(1-a_*)D_{\text{eff}})$ 의 동역학적 실현).
3. **국소 안정성**: $\rho(DB(p^*)) < 1$ 이 측정 가능 (예: $p^*$ 근방 perturbation 후 수렴 비율).
4. **에너지 균형**: 활성당 비용 $C_a$, 구조 유지 비용 $C_s$, 배경 비용 $C_b$ 의 비율이 $C_a:C_s:C_b \approx 1:5.4:14.1$ 영역 (Raichle 2010 뇌 에너지 분배와 정합) 에 있다.
5. **외부 데이터 재학습 가능**: A.1 의 ISS ball 이 닫히는 영역에서 학습이 안정적으로 진행된다.

| 기질 | 1 | 2 | 3 | 4 | 5 | 등급 |
|---|---|---|---|---|---|---|
| Transformer + Backprop | 부분 | 결손 | 측정 안 됨 | 결손 | 부분 | `falsified` (`5_Sparsity.md` 8.5) |
| SNN + STDP + 막전위 동역학 | 가능 | 가능 (STDP 자기참조) | 측정 필요 | 가능 (생물 정합) | 측정 필요 | 미검증 (`8_Roadmap.md` 0절 G-S1~G-S5) |
| 생물 뇌 (피질) | 측정됨 | 측정됨 | $\rho \in [0.1, 0.3]$ (`evidence.md` 3.3) | 측정됨 | -- | `bridge` (`6_뇌/evidence.md` 8장) |

이 표가 게이트 `F1` 의 닫힘 경로다. 5조건 중 1개라도 결손이면 본문의 자기수렴 hard claim 은 금지된다.

### A.3 게이트 `F3`: 에르고딕 동등성 (시간 ↔ 공간)

> 3_Sleep.md 6.2 의 "시간 분배 ≈ 에너지 분배" 를 단순 수치 근접에서 에르고딕 정리로 격상한다.

#### A.3.1 모드 점유 측도

뇌가 모드 공간 $\mathcal{M} = \{\text{WAKE}, \text{NREM}, \text{REM}\}$ 의 마르코프 사슬을 가진다고 두자. 정류 분포 $\pi = (\pi_W, \pi_N, \pi_R) \in \Delta^2$.

**에르고딕 정리 (Birkhoff)**: 사슬이 에르고딕이면

$$\lim_{T\to\infty}\frac{1}{T}\int_0^T \mathbb{1}[M(t)=m]\,dt \;=\; \pi_m \quad (\text{a.s.})$$

따라서 **시간 분배** $(t_W/T, t_N/T, t_R/T)$ 와 **정류 점유 측도** $\pi$ 는 동일 simplex $\Delta^2$ 위의 같은 객체다.

#### A.3.2 코어 분배와의 동등 클래스

CE 코어의 공간 에너지 분배 $p^* = (\Omega_\Lambda, \Omega_{DM}, \Omega_b) = (0.6891, 0.2623, 0.0487)$ 도 $\Delta^2$ 위의 점이다. 두 측도의 거리:

$$d_{\text{KL}}(\pi_{\text{brain}} \,\|\, p^*) = \sum_i \pi_i \log\frac{\pi_i}{p_i^*}$$

| 비교 | $\pi$ 또는 $p$ | $d_{\text{KL}}$ vs $p^*$ |
|---|---|---|
| Raichle 뇌 에너지 분배 | $(0.65, 0.30, 0.05)$ | $\approx 0.0035$ |
| 인간 수면 시간 분배 | $(0.667, 0.250, 0.083)$ | $\approx 0.025$ |
| Planck 우주 분배 | $(0.6891, 0.2623, 0.0487)$ | $\equiv 0$ |
| 균등 분배 (귀무) | $(1/3, 1/3, 1/3)$ | $\approx 0.94$ |

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

#### A.4.2 CE 안정도 vs PCI

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
