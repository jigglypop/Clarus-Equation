# Probe-selective dressing과 public scaffold 루프

## 0. 판정

> **합성 데이터에서 A/B 선택성, held-out-designated measured-\(R\) 조건부 예측,
> pump-off 3-probe
> common-kernel factorization과 공분산 에너지 장부의 스키마 control은 닫혔다.
> 실제 측정 데이터, 물리적 pump-off, apparatus-memory 제거와 sample transfer는
> 없으므로 public scaffold와 new matter는 모두 `False`다.**

이 루프는 경전 서사를 물리 증거로 사용하지 않는다. “그에게만 물이 땅”은
probe-dependent dressing을 생각하게 하는 비유이고, “다리를 놓음”은 관측자와
무관하게 전달 가능한 구조를 요구하는 falsification rule로만 사용한다.

## 1. 두 branch를 직선으로 합치지 않는다

### 1.1 private branch

\[
H(t)=H_{\rm env}+\sum_iH_i+u(t)V_{\rm pump}
     +\sum_i g_i(z_i)V_i,
\]

\[
\left(G_i^R\right)^{-1}
=\left(G_{i,0}^R\right)^{-1}-\Sigma_i^R[u,z_i].
\]

pump/controller가 켜진 동안 probe \(A\)만 다른 response를 보이면
`private dressing` 후보다. 이 효과는 pump-off 환경 구조를 뜻하지 않는다.

### 1.2 public branch

pump-off 뒤 probe \(p\)의 baseline-subtracted response를

\[
d_p=c_p K_{\rm post}+\epsilon_p
\]

로 쓴다. \(c_p\)는 pump 전에 고정한 probe calibration이고
\(K_{\rm post}\)는 probe와 무관한 scalar environmental kernel control이다.
두 probe로 \(K_{\rm post}\)를 맞추고 미리 held-out으로 지정했다고 선언한 세 번째
probe를 예측한다. 현재 코드는 그 선언을 받지만 외부 타임스탬프·manifest·hash를
검증하지 않으므로 이를 사전등록 완료라고 부르지 않는다.

공적이라는 말은 모든 probe가 같은 숫자를 본다는 뜻이 아니다. 알려진 서로 다른
\(c_p\)를 통해 **같은 kernel이 서로 다른 response를 예측한다**는 뜻이다.

두 branch는 독립이다.

```text
PRIVATE
pump×controller factorial selectivity
→ phase/noise training
→ designated held-out measured-R response prediction
→ CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING

PUBLIC
declared pump-off time ordering (physical data open)
→ residual-drive/apparatus-memory veto
→ 2-probe common-kernel fit
→ held-out third-probe prediction
→ informative signed energy ledger
→ CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE
```

따라서 private branch가 실패해도 public branch는 원리상 통과할 수 있고, private
branch가 통과해도 public branch의 증거가 되지 않는다.

## 2. A/B 선택성은 2×2 factorial contrast다

단순 controller on/off는 controller EMI나 열을 제거하지 못한다. 각 probe \(i\)에
대해

\[
I_i=
[\mu_{i,\mathrm{pump\,on,matched}}
 -\mu_{i,\mathrm{pump\,on,sham}}]
-
[\mu_{i,\mathrm{pump\,off,matched}}
 -\mu_{i,\mathrm{pump\,off,sham}}],
\]

\[
S_{AB}=I_A-I_B
\]

를 paired trial에서 계산한다.

통과에는 다음 세 조건을 동시에 요구한다.

1. \(I_A\)의 신뢰구간 하한이 분석 전에 입력한 최소효과보다 큼
2. \(I_B\)의 전체 신뢰구간이 \([-\delta_B,+\delta_B]\) 안에 들어감
3. \(S_{AB}\)의 신뢰구간 하한이 최소 선택성보다 큼

“B가 유의하지 않다”는 equivalence가 아니다. 그래서 두 번째 항은
zero-equivalence gate다. \(A\)와 \(B\)가 함께 변하는 global heating 반례와
pump on/off 양쪽에 동일한 controller-only 효과가 있는 반례는 실패한다.
reference equivalence bound는 probe-A·selective 최소효과 중 작은 값의 절반을
넘지 못한다. 각 raw mean은 최소 4개 관측을 요구하고, 사용자가 작은
`confidence_multiplier`로 오차막대를 줄일 수 없도록 양측 95% Student-\(t\)
\(df=3\) 임계값 \(3.1824463\)을 보수적 하한으로 둔다.

## 3. phase/noise sweep

측정 phase offset \(\theta_j\)의 raw resultant는

\[
R=\left|\frac{\sum_jw_je^{i\theta_j}}{\sum_jw_j}\right|
\]

이고 현재 finite-sample correction은

\[
R_{\rm bc}^2=
\max\left(0,\frac{N_{\rm eff}R^2-1}{N_{\rm eff}-1}\right),
\qquad
N_{\rm eff}=\frac{(\sum_jw_j)^2}{\sum_jw_j^2}
\]

다.

최소 5개 noise level 중 하나를 held-out으로 지정하고, 최소 4개 training
level에서

\[
S_{AB}=\alpha+\beta R_{\rm bc}
\]

를 맞춘다. held-out level에서도 \(R_{\rm bc}\) 자체는 측정하고, 그 측정값에
조건부인 \(S_{AB}\)만 예측한다. 즉 현재 gate는 noise에서 \(R(D)\)를 예측하지
않는다. held-out residual equivalence bound는 최소 선택 효과의 절반을 넘지
못하고, held-out \(R_{\rm bc}\)는 training 범위 안의 interpolation이어야 한다.
회귀 prediction interval에는 residual-model error, training-mean uncertainty,
held-out response error를 모두 최악 상관 triangle bound로 합치고
\(df=2\) Student-\(t\) 하한 \(4.3026527\)을 적용한다. 단일 \(R>R_*\)
threshold나 가장 좋아 보이는 주파수의 사후 선택은 통과 조건이 아니다.
audit은 각 level의 raw phase tuple과 A/reference의 8개 factorial response
stream을 보존한다. expected sign·세 effect/equivalence threshold·요청 confidence는
sweep 전체의 단일 config로 저장하고, validator가 raw stream에서 point summary를
다시 계산한다. level마다 sign이나 threshold를 바꾸는 summary 재작성은 실패한다.

현재 \(N_{\rm eff}\)는 weight-based Kish 값일 뿐 시계열 autocorrelation을
교정하지 않는다. 따라서 실제 phase trace에서는 block bootstrap/autocorrelation
effective sample size와 noisy Adler 모델

\[
d\theta=(\Delta\omega-K\sin\theta)dt+\sqrt{2D}\,dW_t
\]

의 held-out \(R(D)\), phase-slip 예측이 추가되어야 한다. held-out 지정 역시
코드 내부 선언일 뿐 외부 사전등록 증거는 아니다. 현재 pass는 합성
측정스키마 control이지 phase causation 증명이 아니다. 주기구동이 만드는
effective Hamiltonian/dressing 자체는 표준적 도구지만 pump가 꺼진 뒤의 새 상을
자동으로 뜻하지 않는다
([Oka–Kitamura](https://arxiv.org/abs/1804.03212)).

## 4. pump-off common-kernel gate

최소 세 개의 unique calibrated probe와 정확히 하나의 held-out probe를 요구한다.
training probe들의 \(d_p/c_p\)가 equivalence bound 안에서 일치해야 하고, 그
kernel이 held-out \(d_h/c_h\)를 예측해야 한다. raw response가 같은지만 비교하면
서로 다른 coupling을 무시하므로 실패다.

동시에

\[
K_{\rm post}^{\rm LCB}
-g_{u\to K}^{\rm UCB}|u_{\rm residual}|^{\rm UCB}
-g_{n\to K}^{\rm UCB}|n_{\rm nuisance}|^{\rm UCB}
-K_{\rm apparatus}^{\rm UCB}
\ge \Delta K_{\min}
\]

을 요구한다. 즉, 잔류 pump와 nuisance monitor가 각각 알려진 최대 gain을 거친
효과 및 detector/cavity memory 상한을 빼고도 kernel 하한이 남아야 한다.
pre-pump null, time ordering, minimum dwell, blind analysis와 별도 held-out
readout chain도 필수다. probe의 raw pre/post/sham tuple과 두 monitor의 raw
post/sham tuple을 audit에 보존하고 validator에서 모든 interval을 다시 계산한다.

training probe들이 common clock·pump drift를 공유할 수 있으므로
`UNMEASURED_WORST_CASE_CORRELATION`을 적용한다. pooled kernel의 표준오차는
어떤 training probe의 표준오차보다도 작아지지 않으며, probe 차이의 표준오차는
독립 `hypot` 대신 두 표준오차의 합으로 둔다. 이는 가짜 정밀도를 막는 보수적
처리이지 probe 간 covariance를 실제 측정했다는 뜻은 아니다. held-out
지정, pump 전 calibration 고정, blinding, 별도 readout chain은 현재 모두 선언
metadata이며 외부 기록으로 검증되지 않았다.

그러나 현재 kernel은 scalar 한 점이다. 실제 causal boundary에는
\(\chi^R(t<0)=0\), stable pole, finite-band tail bound, Kramers–Kronig와
frequency-dependent held-out response가 필요하다. sample을 이동시켜 효과가
source에 남지 않고 destination을 따라가는 blinded swap도 아직 없다. 그러므로
`TRANSFERABLE_PUBLIC_SCAFFOLD`로 승격하지 않는다.

## 5. fixed signed energy ledger

catch-all `other` 항은 금지하고 다음 열을 고정한다.

| 입력 \(+\) | 출력·저장 \(-\) |
|---|---|
| pump work | decoupled candidate energy |
| controller work | radiation |
| probe work | thermal/mechanical energy |
| transfer work | reservoir storage |
| pre-existing reservoir release | recovered work |

평균 energy vector \(e\), sign vector \(s\), 전체 calibration covariance \(C\)에
대해

\[
\epsilon_E=s^\top e,\qquad
\sigma_{E,\rm cal}^2=s^\top C s
\]

를 계산하고 trial scatter와 합친다. sampling error와 calibration error의
독립성이 schema에 없으므로 `hypot`이 아니라 두 SE의 합을 사용한다. 통과에는

\[
|\epsilon_E|\le\epsilon_{\rm abs}+z_\alpha\sigma_E,
\]

\[
\frac{|\epsilon_E|}{E_{\rm candidate}}\le\eta_{\rm residual},
\qquad
\frac{\sigma_E}{E_{\rm candidate}}\le\eta_{\rm uncertainty}
\]

가 모두 필요하다. 두 번째 uncertainty 조건이 없으면 오차막대가 클수록 쉽게
통과하는 반례가 생긴다. absolute tolerance 자체도
\(\eta_{\rm uncertainty}E_{\rm candidate}\)보다 클 수 없어서 큰 tolerance로
closure를 만드는 우회도 막는다.

covariance는 대칭·positive semidefinite여야 하며 모든 원시 diagonal variance는
동적 범위와 무관하게 음수가 될 수 없다. 대칭성과 Cholesky pivot의 반올림
tolerance는 각각 해당 pair/pivot의 local scale에서 64 ULP로만 허용한다. 상대
residual 상한은 최대 10%, 상대 uncertainty 상한은 최대 25%로 제한해 사용자가
gate 자체를 공허하게 늘리는 것도 막는다.
검증 report에는 원시 10채널 trial tuple과 10×10 covariance를 모두 보존한다.
평균·채널 최솟값·trial별 잔차·sampling SE와
\(\sigma_{E,\rm cal}=\sqrt{s^\top Cs}\)를 다시 계산한다. 따라서 음수 trial을
minimum summary에서 숨기거나 `total_sigma`만 작게 변조해 raw scatter와
declared covariance를 우회할 수 없다.

product energy는 pump와 controller가 decouple된 endpoint에서만 분리했다고
선언해야 한다. 이 선언과 에너지 보존 통과도 microscopic creation mechanism을
유도하지 않는다.

## 6. 합성 control 결과

```text
noise-phase correlation       -0.963103391
phase-response correlation    +1.000000000
heldout response residual     -6.66e-16
private stage                 CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING

fitted common kernel           1.200000000
heldout kernel residual        0
energy residual                0 J
energy sigma                   3.162e-2 J
public stage                  CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE

public scaffold candidate      False
physical scaffold              False
new matter                     False
```

통과용 합성 데이터를 만들 수 있다는 것은 gate가 계산 가능하다는 뜻일 뿐 자연에
그 현상이 존재한다는 뜻이 아니다.

## 7. 잠긴 반례

- controller-only 효과가 pump on/off에서 동일함
- \(A,B\)가 함께 변하는 global heating
- reference·monitor equivalence bound를 효과 scale보다 크게 잡는 우회
- 95% 미만 confidence 요청 또는 소표본 normal-\(z\) 오차막대 축소
- noise level의 held-out 미지정·중복
- 최소 선택 효과보다 큰 held-out residual equivalence bound
- phase와 무관한 일정 response
- 두 probe만으로 common kernel을 자가적합
- calibrated gain을 무시하고 raw response 크기만 비교
- pre-pump signal, residual pump, apparatus memory 설명
- 잘못된 pump/readout 시간 순서
- blinding·사전 calibration·독립 readout 누락
- 에너지 한 열 누락
- giant covariance 또는 giant absolute tolerance로 만든 vacuous closure
- 거대한 다른 variance에 가려진 음수 diagonal covariance
- 상관된 training probe를 독립 표본처럼 pooling해 만든 가짜 정밀도
- anti-correlated probe 차이에 독립 `hypot`을 써서 만든 가짜 equivalence
- nuisance readout은 작지만 nuisance→kernel gain이 큰 설명
- raw phase/probe/monitor tuple은 둔 채 summary만 다시 쓰는 변조
- raw covariance는 크게 둔 채 summary sigma만 줄이는 report 변조
- raw energy trial은 유지한 채 sampling scatter·channel minimum만 줄이는 변조
- 문자열 `"false"`를 truthy declaration으로 쓰는 타입 변조
- pump와 결합된 endpoint에서 bare와 interaction energy 중복계수
- physical/new-matter/stress/wormhole claim-lock 변조

## 8. 현재 남은 물리 루프

1. config hash와 raw-data hash를 고정한 preregistration manifest
2. common-clock leakage, equal in-band power, phase-scrambled drive control
3. time-series \(N_{\rm eff}\), block bootstrap와 Adler held-out prediction
4. cavity ringdown, detector IIR, thermal/charge/mechanical memory의 독립 상한
5. blinded prepared/sham sample swap와 독립 전원·clock·detector
6. scalar kernel을 \(D^R(\omega,k)\)로 확장하고 causality/passivity 검사
7. pump/controller/probe/sample을 함께 적분한 동역학적 energy ledger
8. \(c_p\) calibration uncertainty·\(\operatorname{Cov}(d_p,c_p)\)의 전파
9. common-kernel run과 energy run을 묶는 sample/config/raw-data hash 및
   candidate-energy 독립 측정 provenance
10. 그 뒤에만 pole·질량·양자수·입자수 inventory로 new matter를 검사

## 9. 재현

```powershell
uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_probe_scaffold_pilot.py -q

uv --cache-dir .uv-cache run python `
  examples/physics/probe_scaffold_pilot_gate.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_casimir_carrier_target.py `
  tests/test_clarus_resonant_matter.py `
  tests/test_global_throat_exact_certificate.py `
  tests/test_physical_multimode_realization.py `
  tests/test_casimir_global_resonance.py `
  tests/test_resonance_stress_identifiability.py `
  tests/test_probe_scaffold_pilot.py -q
```

현재 suite는 `21 passed`다. 기존 resonance/global-throat 6-file 회귀까지 묶은
7-file focused suite는 `112 passed`다.

## 참고

- Oka and Kitamura, [Floquet Engineering of Quantum Materials](https://arxiv.org/abs/1804.03212)
- Nakao, [Phase reduction approach to synchronization of nonlinear oscillators](https://arxiv.org/abs/1704.03293)
- Schröder, Timme and Witthaut, [A universal order parameter for synchrony](https://arxiv.org/abs/1704.04130)
