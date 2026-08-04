# 박막/결함 QFT 현실성 루프

날짜: 2026-08-04

## 질문

두 외부 시공간을 잘라 붙이는 thin-shell wormhole은 정확한 공간 지름길
기하를 준다. 남은 질문은 접합면에 필요한 표면 응력에너지를 동역학적
경계 QFT가 실제로 만들 수 있는가이다.

이번 루프는 가장 단순한 대조군인 대칭 Schwarzschild cut-and-paste 목과
등방·무질량 scale-free 2+1차원 경계 QFT를 검사한다. 모든 결함 QFT에 대한
no-go를 주장하지 않는다.

## Israel 접합조건

목의 반지름을 `a`, Schwarzschild lapse를

\[
f(a)=1-\frac{2GM}{c^2a},\qquad 0<f\le1
\]

이라 하면 정적 대칭 껍질의 표면 에너지와 횡압은

\[
\sigma=-\frac{c^4}{2\pi Ga}\sqrt f,
\qquad
p=\frac{c^4}{8\pi Ga}\frac{1+f}{\sqrt f}
\]

이다. 따라서 모든 허용 영역에서 `sigma < 0`, `p > 0`이다. 이는 기하가
요구하는 응력이지 미시적 물질의 유도가 아니다.

## 1 m 수치

| lapse `f` | 총 음의 에너지 질량환산 | 횡압 `p` | `p/|sigma|` |
|---:|---:|---:|---:|
| 1 | 450.95 지구질량 | `9.63e42 N/m` | 0.5 |
| `1e-6` | 0.45095 지구질량 | `4.82e45 N/m` | `2.50e5` |
| `1e-12` | `4.51e-4` 지구질량 | `4.82e48 N/m` | `2.50e11` |

`f -> 0`이면 `|sigma|`는 `sqrt(f)`로 감소하지만 `p`는 `1/sqrt(f)`로
발산한다. 그러므로 지평선 근접 배치는 음의 에너지 비용을 제거하지 않고
압력, 적색편이 및 미세조정 비용으로 이동시킨다.

## scale-free edge QFT 방정식상태 gate

등방 2+1차원 conformal surface stress가 traceless라면

\[
-\sigma+2p=0,\qquad p=\sigma/2
\]

이다. 음의 Casimir surface energy에서는 이 압력도 음수다. 반면 Israel
껍질은 모든 `f`에서 양의 `p`를 요구한다. 따라서 크기, 모드 수, 공명 Q를
바꾸어 진폭을 증폭해도 tensor의 부호 불일치는 사라지지 않는다.

1 m, `f=1`, 단위 Casimir 계수라는 낙관적 정규화에서도 필요한 유효
자유도는

\[
N_{\rm eff}=\frac{|\sigma|a^3}{\hbar c}
=6.09\times10^{68}
\]

이다. species cutoff `sqrt(N_eff) l_P`는 `0.399 a`가 되어 국소 경계 EFT의
큰 scale separation도 없다. 이 cutoff 평가는 양자중력 species 추정에
의존하는 보조 경고이며, 독립적인 엄밀 no-go로 사용하지 않는다.

## 판정

| 명제 | 판정 |
|---|---|
| cut-and-paste 기하와 요구 표면응력 | `Exact control` |
| 지평선 근접화로 모든 비용 감소 | `Refuted`: 압력 발산 |
| 등방 scale-free/CFT Casimir edge 단독 구현 | `Refuted`: EoS 부호 불일치 |
| 일반 massive·비등방·상호작용 결함 QFT | `Open` |
| CE에 그런 명시적 defect action과 안정성 증명 존재 | `No` |

따라서 thin-shell 경로 전체가 수학적으로 불가능하다고 증명된 것은 아니다.
정확히 제거된 후보는 **scale-free isotropic edge QFT**이다. 생존하려면
명시적 비등각 defect action으로 위의 `sigma(a), p(a)`를 함께 유도하고,
방사형·비구면·양자 섭동 안정성 및 중력 backreaction을 닫아야 한다.

## 일반 barotropic 유체의 방사형 안정성 루프

껍질 보존식 `sigma'=-2(sigma+p)/a`와 `eta=dp/dsigma`를 사용해 운동
퍼텐셜을 두 번 미분하면 정적점에서

\[
a^2V''=2\eta(1-3f)-\frac{1+3f^2}{2f}
\]

를 얻는다. 방사형 안정성은 `V''>0`이다.

- `0<f<1/3`에서는 안정성에
  `eta > (1+3f^2)/(4f(1-3f)) > 1`이 필요하다.
- `f=1/3`에서는 eta 항이 사라지고 `a^2 V''=-2`이다.
- `1/3<f<=1`에서는 안정성에 음의 `eta`가 필요하다.

따라서 통상적인 국소 barotropic 물질의 인과적·gradient-stable 범위
`0<=c_s^2=eta<=1`과 방사형 안정 영역은 전 구간에서 교차하지 않는다.
이는 특정 constant-w 선택보다 강한 결과다.

추가 판정:

| 명제 | 판정 |
|---|---|
| 인과적 barotropic defect fluid가 정적 Schwarzschild 박막을 안정화 | `Refuted` |
| eta>1 또는 eta<0인 유효 EoS를 형식적으로 선택 | `Radial pass / microphysical fail` |
| 비barotropic·비등방·고차미분 elastic defect | `Open`, 다음 gate |

## 최소 탄성막과 negative-tension brane 루프

구면 `l=0` 변형은 순수 trace strain이므로 traceless shear strain은 정확히
0이다. 따라서 전단계수를 추가해 비구면 모드를 바꿀 수는 있어도 위의
방사형 `V''`를 바꿀 수 없다. 인과적인 bulk 및 shear sound speed를 가진
최소 국소 등방 탄성막은 barotropic radial no-overlap을 피하지 못한다.

배경 방정식상태만 보면 negative-tension Nambu--Goto 막은
`sigma=T<0`, `p=-T>0`라서 `p=-sigma`를 준다. Israel 응력과 이는 정확히
`f=1/3`에서 일치한다. 그러나 그 점은

\[
a^2V''=-2
\]

로 방사형 불안정하고, `T<0`는 transverse bending fluctuation의 시간
kinetic coefficient도 음수로 만들어 ghost를 낳는다.

추가 판정:

| 후보 | 판정 |
|---|---|
| causal 최소 등방 elastic membrane | `Refuted by l=0 radial mode` |
| negative-tension Nambu--Goto brane | `Refuted: radial instability + bending ghost` |
| 내부 자유도와 mode mixing을 가진 비최소/nonlocal defect | `Open` |

## smooth quantum layer와 QEI 규모 루프

4차원 massless field의 flat-space quantum inequality를

\[
|\rho|\lesssim K N\frac{\hbar}{c^3\tau^4},
\qquad K=\frac{3}{32\pi^2}
\]

라는 표준 Lorentzian-sampling control로 두고, 두께 `d`를 통과하는 시간
`tau=d/c`를 대입해 적분하면

\[
|\sigma|\lesssim K N\frac{\hbar c}{d^3}
\]

이다. 1 m, `f=1`, 한 유효 species에는

- `d <= 2.50e-24 m`,
- `tau <= 8.35e-33 s`,
- `hbar c/d >= 7.89e16 eV`

가 필요하다. species 수를 `10^12`로 늘려도 두께는 세제곱근인 `10^4`배만
완화된다.

이 QEI는 smooth free-space layer에 대한 control이며 이상적인 정적 경계
Casimir 상태에 그대로 적용하는 엄밀 no-go는 아니다. 그러나 그 경계
loophole을 사용하려면 반사체/결함의 미시 작용과 그 양의 에너지, 압력,
backreaction까지 함께 계산해야 한다. 현재 CE에는 그런 completion이 없다.

판정은 `one-species smooth layer: physical-scale fail`, `material boundary:
deferred, full stress required`이다.

## 내부 공명 mode-mixing 루프

방사형 변위 `x`와 안정한 내부 모드 `y`의 가장 일반적인 국소 2차 potential을

\[
V_2=\frac12(K_{rr}+D)x^2+Bxy+\frac12Cy^2,\qquad C>0
\]

로 둔다. 내부 모드가 수동적으로 완화되면 `y=-B x/C`이고

\[
K_{\rm eff}=K_{rr}+D-\frac{B^2}{C}
\]

이다. 따라서 안정한 mode를 결합하는 공명 mixing 자체는 음의 방사형
고유값을 올리지 못하고 항상 그대로 두거나 더 낮춘다. 안정화를 수행하는 것은
mixing이 아니라 별도의 직접 강성 `D > -K_rr+B^2/C`이다.

| 후보 | 판정 |
|---|---|
| passive stable internal resonance mixing | `Refuted as stabilizer` |
| explicit positive direct radial stiffness | `Mathematically possible / source open` |
| driven Floquet 또는 feedback stabilization | `Open, non-static and noise/backreaction required` |

그러므로 “다중공명”만 추가하는 것은 현실성 gate를 통과시키지 않는다. 능동
제어를 쓰면 정적 wormhole 문제가 아니라 시간의존 시스템의 Floquet 안정성,
구동 에너지와 고장 시 붕괴 문제로 바뀐다.

## driven Floquet 안정화 루프

능동 매개구동은 차원이 없는 식

\[
x''+[-r^2+\epsilon\cos\tau]x=0,
\qquad r=\Gamma/\Omega
\]

으로 검사했다. 고주파 평균 곡률은 `-r^2+epsilon^2/2`이지만, 평균식만
믿지 않고 RK4로 두 기본해를 한 주기 적분해 monodromy의 `|tr M|<2`를
직접 판정했다.

`r=0.05`, `epsilon=0.1`에서 평균 곡률은 양수이고 exact Floquet gate도
통과하며 symplectic check `det M=1`을 수치오차 안에서 만족한다. 반면
`epsilon=0.05`는 threshold 아래라 실패한다.

negative-tension match 점 `a^2 V''=-2`에서는 proper radial growth rate가
대략 `Gamma=c/a`이다. `a=1 m`, `r=0.05` control은

- drive angular frequency `Omega ~= 6.00e9 rad/s`, 약 `0.95 GHz`,
- parametric coefficient `epsilon*Omega^2 ~= 3.60e18 s^-2`

를 뜻한다. 이것은 기하 모드의 **능동 안정화 control**이지 Israel 접합조건의
음의 `sigma`를 만드는 source가 아니다. drive loss 후 안정성은 0이고 구동
stress, 발열, noise, payload coupling도 아직 없다.

판정: `Floquet stabilization = demonstrated control`, `wormhole realization =
not advanced unless a physical stress actuator is derived`.

## 재현

```powershell
$env:PYTHONPATH='.;reality_stone/python'
uv run python examples/physics/thin_shell_defect_reality_gate.py
uv run pytest tests/test_thin_shell_defect_reality.py -q
```

주요 문헌:

- E. Poisson and M. Visser, *Thin-shell wormholes: Linearization stability*,
  arXiv:gr-qc/9506083.
- F. S. N. Lobo, *Thin shells around traversable wormholes*,
  arXiv:gr-qc/0401083.
