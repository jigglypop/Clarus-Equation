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

| lapse `f` | 총 음의 에너지 질량환산 | 횡압 `p` | \(p/\lvert\sigma\rvert\) |
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

방사형 변위 `x`와 유한개의 안정한 내부 모드 벡터 `y`의 가장 일반적인 국소 2차 potential을

\[
V_2=\frac12(K_{rr}+D)x^2+xB^{\mathsf T}y
    +\frac12y^{\mathsf T}Cy,\qquad C=C^{\mathsf T}\succ0
\]

로 둔다. 내부 모드가 수동적으로 완화되면

\[
y_*=-C^{-1}Bx,\qquad
K_{\rm eff}=K_{rr}+D-B^{\mathsf T}C^{-1}B
\]

이다. Cholesky 분해 `C=LL^T`를 쓰면

\[
B^{\mathsf T}C^{-1}B=\|L^{-1}B\|^2\ge0
\]

이므로 안정한 mode를 결합하는 공명 mixing 자체는 음의 방사형 고유값을
올리지 못하고 항상 그대로 두거나 더 낮춘다. 더 강하게 전체 Hessian `H`는

\[
P^{\mathsf T}HP=\operatorname{diag}(K_{\rm eff},C),\qquad
P=\begin{pmatrix}1&0\\-C^{-1}B&I\end{pmatrix}
\]

로 합동변환된다. 따라서 `C>0`일 때 `K_eff<0`이면 음의 고유방향은 정확히
하나이고, `K_eff=0`은 안정이 아니라 marginal zero mode다. strict 안정화 조건은

\[
D>-K_{rr}+B^{\mathsf T}C^{-1}B
\]

이며 우변은 minimum이 아니라 **strict lower bound**다.

또한 양의 질량행렬 `M`과 상수 감쇠·gyroscopic 행렬을 갖는
`M q¨+(R+G)q˙+Hq=0`, `G^T=-G`에서도 `K_eff<0`이면
`det H<0`이다. 실수 특성식은 `p(0)<0`, `p(λ)→+∞`이므로 양의 실수 성장률을
반드시 갖는다. 상수 감쇠나 gyroscopic coupling은 이 tachyon을 제거하지 못한다.
시간주기 drive, feedback, clamping, singular 내부 블록, 연속 스펙트럼은 이
유한차원 정적 정리의 범위 밖이다.

| 후보 | 판정 |
|---|---|
| finite passive stable internal resonance mixing | `Exact static quadratic no-go` |
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

## Floquet-to-Israel actuator 역산

동적 대칭 shell의 Israel pressure 식을 선형화하면 추가 표면압력은

\[
\delta\ddot a=\frac{4\pi G\sqrt f}{c^2}\delta p
\]

의 정상 가속도를 만든다. Floquet 항 `-h cos(Omega t) delta a`와 맞추면

\[
\frac{\partial p}{\partial a}
=\frac{c^2h}{4\pi G\sqrt f}
\]

가 필요하다. `a=1 m`, `f=1/3`, `a^2V''=-2`, `Gamma/Omega=0.05`,
`epsilon=h/Omega^2=0.1`, 허용 변위 `delta a/a=1e-6`의 결과는 다음과 같다.

| 양 | 결과 |
|---|---:|
| instability growth rate | `2.998e8 s^-1` |
| drive frequency | `9.543e8 Hz` |
| parametric coefficient `h` | `3.595e18 s^-2` |
| pressure stiffness | `6.68e44 N/m^2` |
| pressure modulation | `6.68e38 N/m` |
| background pressure 대비 | `6.00e-5` |
| peak reactive mechanical power bound | `5.03e43 W` |
| drive-loss e-fold | `3.34 ns` |

피크 반응성 출력은 `area * |delta p| * peak velocity`이며 평균 소비전력과
동일하다고 가정하지 않는다. 손실과 actuator action이 없으므로 평균 실전력은
아직 계산할 수 없다. 그러나 actuator가 취급해야 할 stress와 bandwidth가
중력 규모라는 사실은 변하지 않는다.

판정: `mathematical feedback control PASS`, `physical stress actuator FAIL/ABSENT`,
`negative-energy source still absent`.

## rigid negative-tension brane 루프

음의 장력 막의 bending ghost를 국소 extrinsic-curvature rigidity
`alpha K^2`로 고치는 최소 시도를 검사했다. 굽힘 모드의 2차 inverse
propagator를

\[
P(z)=Tz+\alpha z^2,\qquad T<0
\]

로 쓰면

\[
\frac1{P(z)}=\frac1T\frac1z-\frac1T\frac1{z+T/\alpha}.
\]

원래 massless pole의 residue `1/T`는 계속 음수이고 새 pole은 반대 residue를
갖는다. `alpha`의 크기를 바꾸면 새 pole의 위치만 움직일 뿐 두 residue를 모두
양수로 만들 수 없다.

판정: `local K^2 rigidity cure = Refuted`. 유도중력이나 추가 healthy field가
전체 2차 kinetic coefficient를 양수로 뒤집어야 하며, 이는 별도의 수정중력
작용과 backreaction 문제다.

## induced-gravity/nonlocal defect 분기

DGP형 localized Einstein--Hilbert 항은 계수 선택에 따라 ghost-free 영역이
가능하므로 이론 클래스 전체를 반증할 수는 없다. 다만 현재 문제의 shell
worldvolume은 2+1차원이고, 순수 2+1 Einstein gravity 자체에는 국소 graviton
자유도가 없다. 새로운 bending kinetic은 ambient bulk와의 혼합에서 와야 한다.

따라서 다음 자료가 모두 새로 필요하다.

1. bulk action과 localized EH coefficient
2. 기존 Israel 식을 대체하는 modified junction equation
3. 음의 장력 background를 실제로 만족하는 global solution
4. brane-bending, bulk KK, scalar mode의 전체 pole/residue
5. crossover/strong-coupling scale와 1 m throat의 hierarchy

현재 CE에는 1, 2가 없으며 3--5도 계산할 입력이 없다. 비국소 kernel 역시
spectral density와 retarded boundary condition이 없으면 안정성·인과성·손실을
판정할 수 없다.

판정: `induced gravity = external open frontier`, `CE completion = absent`,
`reality pass = no`. 이 항목은 thin-shell active branch 내부에 유지하며 독립
현실화 성공으로 중복 집계하지 않는다.

관련 근거:

- [Brane induced gravity: Ghosts and naturalness](https://arxiv.org/abs/1506.02666)
- [Ghost problem and constraints on brane-localized gravity](https://arxiv.org/abs/2310.16297)

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
