# 자기비동일성 운동학적 암흑부문 — 최종 연구 보고

Status: INTERIM — R2 IN PROGRESS

## 1. 최종 판정

이 run은 기존 exponential SMQ 식을 억지로 유지하지 않았다. 완전 반례가
지목한 두 결함을 식 수준에서 교체했다.

첫째, 기존 식

$$
V(\Theta)=\rho_*e^{-\Theta}
$$

에는

$$
(\Theta,\rho_*)\longmapsto(\Theta+\Delta,\rho_*e^\Delta)
$$

라는 정확한 redundancy가 있다. 자유 amplitude shooting이 operational clock의
절대 원점을 흡수하므로, 이 식의 background 적합은 0D 또는 자기측정 origin을
식별하지 못한다.

둘째, 증가하는 누적 readout과 양의 dust inventory를 zero current에서 동시에
만들 수 없다. 새 anchored action에서도 initial current가 zero이면

$$
P_T=-\rho_\infty\Gamma e^{-\Gamma T}<0
$$

때문에 바로 $J<0$, $\delta<0$, $c_s^2<0$으로 간다. 이것은 해석 문제가
아니라 완전 반례다.

수정된 후속식은 이 두 결함을 다음과 같이 직접 다룬다.

$$
P(T,X)=\rho_\infty\left[
\frac\kappa2\left(\frac{X}{X_*}-1\right)^2
-\left(1-e^{-\Gamma T}\right)
\right],
$$

$$
\left(T,a^3P_X\dot T\right)\big|_{\Sigma_*}
=\left(0,\Pi_{\rm fold}\right),
\qquad \Pi_{\rm fold}>0.
$$

따라서 R0 배경판정은 **조건부 IR EFT PASS**다. 이것은 run 전체나 R1의
최종 판정이 아니다. positive fold-matched initial current를 채택하면 한
clock field가 DE-like readout과 small-$\delta$ DM-like kinetic inventory를
조건부로 함께 구현한다. 그러나
$\Pi_{\rm fold}=\mathcal R_\Pi[\mu_F,C_{\rm self};\text{scales}]$의 미시적
quantum derivation은 아직 공리이므로 현재 run 전체 상태는 `IN PROGRESS`다.

## 2. 아이디어 세 층의 조립

### 흐름

이산적으로 $T_{n+1}\ne T_n$가 반복된다는 것은 continuum에서

$$
X=-\frac12g^{\mu\nu}\nabla_\mu T\nabla_\nu T>0
$$

인 변화 readout으로 보낸다. 하지만 $X>0$은 미래 방향을 고르지 않는다.
$\dot T>0$은 물리적 matching surface에서 주는 별도 future-branch Cauchy
자료다.

### 0차원 측정과 이웃 부트스트랩

operational outcome의 0차원성과 spacetime의 새 차원을 동일시하지 않는다.
최소 type chain은

$$
Z_{\rm phys}\longrightarrow R\longrightarrow\mu_F
\xrightarrow{\mathcal R_T}(T,X,\Pi_{\rm fold})
$$

이다. strict 0D object, record, retained carrier measure와 bulk field는 서로
다른 타입이다. 이웃 양자가 이웃 전이를 gate하고 하나의 환경장이 carrier
활성을 매개한다는 선행 모형은 $\mathcal R_T$의 microscopic candidate다. 그것이
곧 energy source라는 결론은 따르지 않는다.

### 기회비용

고정 dephasing의 무차원 누적량은

$$
\theta=\Gamma T,
\qquad c(\theta)=1-e^{-\theta}.
$$

$c$ 자체는 energy가 아니다. 이 run은 독립 density scale $\rho_\infty$와
공변 action을 추가해 $\rho_\infty c$를 DE-like readout으로 정의한다. 그
증가분은 무에서 생기지 않고 positive initial kinetic inventory에서 이동한다.

## 3. 재유도된 정리 사슬

채택 action의 변분은

$$
\nabla_\mu(P_X\nabla^\mu T)+P_T=0,
\qquad
T_{\mu\nu}=P_X\nabla_\mu T\nabla_\nu T+Pg_{\mu\nu}
$$

를 준다. FLRW에서

$$
\rho=2XP_X-P,
\qquad p=P,
\qquad
\frac{d}{dt}(a^3P_X\dot T)=a^3P_T.
$$

energy split은

$$
\rho_V=\rho_\infty(1-e^{-\Gamma T}),
\qquad p_V=-\rho_V,
$$

$$
\rho_K=\rho_\infty\left(2\kappa\delta+\frac32\kappa\delta^2\right),
\qquad
p_K=\frac12\kappa\rho_\infty\delta^2.
$$

따라서

$$
w_K=\frac{\delta}{4+3\delta},
\qquad
c_s^2=\frac{\delta}{2+3\delta},
$$

$$
P_X+2XP_{XX}=
\frac{\kappa\rho_\infty}{X_*}(2+3\delta)>0
$$

이다. $0<\delta\ll1$에서만 positive, cold, DM-like이고 큰 $\delta$에서는
radiation-like가 된다.

정확한 current와 energy ledger는

$$
a^3J(t)=\Pi_{\rm fold}-
\int_{t_i}^{t}a^3\rho_\infty\Gamma e^{-\Gamma T}\,dt,
$$

$$
\dot\rho_K+3H(\rho_K+p_K)=-\dot\rho_V,
\qquad \dot\rho_V>0.
$$

그러므로 $\Pi_{\rm fold}$를 별도 energy로 다시 더하면 이중 계상이다. 그것은
한 field의 initial canonical current이며 bulk에서는 이미 $\rho_K$로 읽힌다.

finite solve 뒤에는 first-zero contradiction을 사용했다. $N_f$ 이후 첫
$q=0$이 있다고 가정하면 그 전까지 $u>0$, $E\le1$,
$E\ge\sqrt{V_f}$, $\tau'\ge1$이다. $\gamma>3$에서 남은 손실은

$$
\Delta q_{\rm tail}\le
\frac{\gamma e^{3N_f-\gamma\tau_f}}
{2\sqrt{V_f}(\gamma-3)}.
$$

$q_f>\Delta q_{\rm tail}$이면 첫 zero가 모순이므로 모든 미래에서 current가
양수다. $\gamma>3$은 이 보수적 bound의 충분조건이지 물리적 필요조건이 아니다.

## 4. 구현과 관측 판정

normalized background는 별도 standard-CDM term 없이 kinetic inventory 자체를
DM-like sector로 사용한다. 초기 구현의 DM 중복 계상, 음속식 오류, finite-future
검증, standalone import 의존, large-$\gamma$ shooting precision loss는 각각
재현 후 수정했다. stable snapshot의 focused test는 8개 모두 통과했다.

same-fraction flat $\Lambda$CDM의 pinned DESI DR2 background control은

$$
\chi^2_{\Lambda{\rm CDM}}=13.442354.
$$

대표 finite-$\gamma$ 결과는

| $\gamma$ | $\chi^2$ | $\Delta\chi^2$ | $\Delta$AIC |
|---:|---:|---:|---:|
| 5 | 45.102774 | +31.660420 | +33.660420 |
| 10 | 16.187284 | +2.744930 | +4.744930 |
| 20 | 13.552734 | +0.110380 | +2.110380 |
| 30 | 13.451082 | +0.008728 | +2.008728 |

이다. finite $\gamma$는 baseline보다 나아지지 않으며 large $\gamma$에서
$\Lambda$CDM으로 접근한다. AIC는 scan slope 한 모수만 추가 반영하며 grid
look-elsewhere를 완전히 보정하지 않는다. 따라서 이 결과는 finite-$\gamma$
signal, CE prediction 또는 dark-sector discovery의 증거가 아니다.

## 5. 형식 지위와 다음 falsifier

| 항목 | 지위 |
|---|---|
| old exponential origin 식별 | 완전 반례로 기각 |
| zero current에서 DE와 positive DM 동시 생성 | 완전 반례로 기각 |
| anchored action의 stress와 안정 domain | 조건부 정리 |
| positive $\Pi_{\rm fold}$의 DE/DM-like 동시 구현 | 조건부 정리 |
| 무한 미래 current positivity | 명시한 tail inequality 아래 조건부 정리 |
| $0$D record에서 $\Pi_{\rm fold}$의 값 도출 | 미완성 matching 공리 |
| intrinsic one-way irreversibility | 미완성 open-system bridge |
| finite-$\gamma$ 관측 선호 | 현재 background 자료에서 없음 |

다음 단계의 falsifier는 $\mathcal R_\Pi$의 공변 quantum construction,
$\rho_\infty$, $\Gamma$, $X_*$, $\kappa$의 scale derivation, EFT cutoff와
caustic control, 그리고 full CMB/LSS/lensing/halo likelihood다. 이 다리를
닫기 전 claim ceiling은 **fold-matched homogeneous conditional EFT**다.

## 6. 재현 경로

- source SHA-256:
  `2B3AC2F652F826F3EC94572F940EDAAD964387B485258C1334AB11BDE695FDA2`
- focused test SHA-256:
  `4E9811F7647BE6C527CC6AAE9471BEB982FE057389DB3BC4483DDE1FFD31F407`
- artifact SHA-256:
  `84BD629738B6AFE7D38752CA9D01D01E0B00D196AB696032AEE4614943BD2281`
- numerical JSON SHA-256:
  `92AD1B126310F7AAF89BCAC097545B894CE2C13AFAF4C416659DC142D0A8BD52`

세부 수학, 구현 및 검증은 각각 `11-math.md`, `30-implementation.md`,
`31-validation.md`에 동결했다.

## 7. R1 중간 판정 — 유계 연속 Gaussian 저장소

R1은 기록·저장소 자유도를 명시적 공변 작용에 넣어 R0보다 한 단계 전진했다.
비유계 선형 $T\phi$ 결합은 Hamiltonian 하한을 잃어 제거했고, 유계 source
$s_A(T)=\mu_A^3F_A(\Gamma T)$와 연속 Gaussian bath를 최소 생존 경로로
채택했다. 이 경로에서는 bath를 적분해 retarded kernel과 양의 noise kernel을
얻고, interaction stress까지 포함해 clock+bath 총응력을 보존할 수 있다.

그러나 $\Pi_F$는 여전히 양의 초기 Gaussian 상태에 주는 평균 운동량
displacement다. 위치형 bath 결합은 그 값을 산출하지 않는다. 또한 strict
0D 점원을 균일 FLRW 자료로 보내는 사상, 근본적인 시간 비가역성, full
metric-mixed perturbation 안정성, CMB/LSS/렌즈/halo likelihood가 열려 있다.
현재 R1의 주장 상한은 다음 한 문장이다.

> 유계 source, 연속 Gaussian 저장소, 양의 초기 Gaussian 상태와 균일
> coarse-graining을 채택하면 인과적 축약동역학과 총응력 보존을 갖는
> 조건부 열린계 초기화가 존재한다.

이는 “비선택 경로가 암흑물질·암흑에너지임이 완전히 유도됐다”는 결론이
아니다. 해당 동일성과 두 존재량의 수치는 계속 활성 미완성 주장이다.

## 8. R2-A 진전 — 음의 질량 부호에서 실제 성장량으로

포화하고 단조 증가하는 비상수 $V(T)$는 전 구간에서 $V''\ge0$일 수 없다.
따라서 현재 분리형 $K(X)-V(T)$를 유지하면서 전역 무타키온을 요구하는
경로는 완전 반례로 닫힌다. 대신 동결 배경에서 실제 Hubble-normalized
성장량을 계산했다.

$a=10^{-4}$부터 오늘까지 다섯 $\gamma$ 배경을 독립 재적분한 결과,
$\max |m_{\rm eff}^2|/H^2=3.05\times10^{-18}$이고 가장 보수적인
fixed-metric log 성장상계도 $2.49\times10^{-17}$이었다. 따라서 기존
$\kappa=10^{17}$ 조건 아래 이 특정 장파장 타키온은 우주론적 시간에
증폭되지 않는다.

그러나 $\pi=\delta T$는 gauge-dependent이다. Einstein+clock 단일-clock
제약을 제거하면 물리적 곡률섭동은

$$
S_\zeta^{(2)}=\int a^3Q_s\left[
\dot\zeta^2-c_s^2(\nabla\zeta)^2/a^2\right]
$$

을 따르고 독립 질량항이 없다. 동결 관측구간에서
$Q_s>0$, $c_s^2>0$, 두 번째 장파장 $\zeta$ mode의 $\dot\zeta$와
적분함수 감소를 확인했다. 이는 그 mode의 무한 미래 수렴 주장과 다르다. 따라서
음의 fixed-metric 질량을 곧바로 물리적 우주 불안정성으로 읽던 해석을
수정했다.

작은 sound speed의 strong-coupling 계산에서도 에너지 cutoff
$\Lambda_E=\Lambda_3c_s^{7/4}$와 물리 파수 cutoff
$q_{\rm sc}=\Lambda_3c_s^{3/4}$를 분리했다. 관측구간 최솟값은
$\Lambda_E/H=9.28\times10^{18}$,
$q_{\rm sc}/[(1\,{\rm Mpc}^{-1})/a]=2.17\times10^{24}$이므로 현재 선형
우주론 scale은 cutoff 아래에 충분히 놓인다.

더 중요한 돌파구는 $k^4$ crossover다. 올바른 조건

$$
q_\times=\frac{c_s\sqrt A}{\bar M}\le q_{\rm sc}
$$

은 오늘 $\bar M\gtrsim0.225\,{\rm eV}$, 관측구간 전체 최악에도
$\bar M\gtrsim7.31\,{\rm eV}$만 요구한다. bare derivative scale
$\Lambda_3\simeq80\,{\rm eV}$ 정도의 후보는 이 조건을 만족할 수 있다.
따라서 $c_s\to0$을 이유로 모형을 축소하는 대신, 배경을 보존하는
higher-spatial-derivative completion을 다음 활성 경로로 확보했다.

이것은 여전히 최종 안정성 증명이 아니다. 다음 필수 계산은
baryon·radiation·reservoir perturbation과 lapse·shift를 함께 제거한
metric--clock--bath reduced quadratic matrix, $k^4$ 연산자의 부호와
degeneracy 및 그 영역의 새 cutoff, 그리고 CMB/LSS의 gauge-invariant mode다.
