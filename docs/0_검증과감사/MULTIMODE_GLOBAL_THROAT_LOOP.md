# 다중 모드 전역 throat 루프

## 목적

앞선 루프는 이상적 Casimir 상태방정식

\[
p_r=3\rho,\qquad p_t=-\rho
\]

을 전역에서 고정하면 유한 redshift와 유한 ADM 질량을 동시에 얻을 수 없음을 보였다.
이번 루프는 그 반증을 우회하는 데 필요한 **가변 비등방 목표 응력**이 실제로 존재하는지
역설계하고, 그 목표가 유한 개 공간 모드로 수렴하는지 검사한다.

## 전역 기하 ansatz

\(x=r/r_0\)에 대해

\[
\frac{b(x)}{r_0}=\frac23+\frac13e^{-(x-1)},
\qquad
\Phi(x)=\frac12e^{-(x-1)}
\]

를 택했다. 목 \(x=1\)에서

\[
b=r_0,\qquad b'=-\frac13,\qquad r_0\Phi'=-\frac12.
\]

Einstein tensor를 역으로 읽은 무차원 응력은

\[
(\rho,p_r,p_t)_{r_0}=\left(-\frac13,-1,\frac13\right)
\]

이므로 목에서는 정확히 이상적 Casimir 비율을 만족한다. 하지만 바깥에서는 압력비를
고정하지 않는다. 이것이 앞선 고정-EoS no-go와 모순되지 않는 핵심이다.

## 표본 없이 닫히는 전구간 기하

\(s=e^{1-x}\), \(y=b/r_0=(2+s)/3\)라 쓰면 shape gap은

\[
g=x-y,qquad g(1)=0,qquad g'=1+\frac{s}{3}\ge1.
\]

따라서 \(x>1\)에서 \(b/r<1\)은 유한 격자 검사가 아니라 해석적으로 성립한다.
또한 \(1-b/r=4(x-1)/3+O((x-1)^2)\)라 proper radial distance의 throat
특이점은 적분 가능하고 양쪽 end로 연장할 수 있다. 기존 lapse는
\(e^{2\Phi}=e^s\ge1\)이고

\[
y/x\to0,qquad \Phi\to0,qquad
\frac{M_{\rm ADM}}{r_0}=\frac{y(\infty)}2=\frac13
\]

이므로 각 asymptotic end의 점근평탄성과 유한 ADM 질량도 cutoff와 무관하게
exact다. 과거 API가 `radial_cutoff=2`에서 점근평탄을 `False`로 바꾸던 것은
유한-cutoff 진단값을 전역 성질로 오인한 버그였으며 교정했다.

일반 \(f=1-y/x\)에 대해 Einstein tensor를 역산한 응력은

\[
\rho=\frac{1-f-xf'}{x^2},\quad
p_r=\frac{f-1}{x^2}+\frac{2f\Phi'}x,
\]

\[
p_t=f\left(\Phi''+\Phi'^2+\frac{\Phi'}x\right)
 +\frac{f'\Phi'}2+\frac{f'}{2x}.
\]

이를 대입하면

\[
\boxed{p_r'+(\rho+p_r)\Phi'-\frac2x(p_t-p_r)\equiv0}
\]

에서 모든 항이 정확히 소거된다. 실행값 `1.33e-15`는 증명이 아니라 이 항등식의
부동소수점 회귀 residual이다. 더구나 이것은 `T=G/kappa`로 역정의한 source의
Bianchi identity이지, 독립 CE 물질 작용의 EOM/Noether 보존 증명이 아니다.

## 기존 exponential redshift에서 새로 확인한 꼬리 결함

기존 target은 전구간에서

\[
\rho=-\frac{s}{3x^2},\qquad
\rho+p_r=-\frac{2+s(3x^2-x+1)-xs^2}{3x^3}<0.
\]

즉 radial NEC 위반은 목 부근에 국소화되지 않고 모든 \(x\ge1\)에서 지속된다.
무한대 꼬리는

\[
x^3p_r\to-\frac23,qquad x^3p_t\to\frac13.
\]

완전한 양쪽 radial null geodesic의 affine ANEC는 throat에서
`(x-1)^(-1/2)`, 무한대에서 `x^(-3)`이므로 유한하고 엄밀히 음수다. asymptotic
Killing energy를 1로 둔 무차원 수치 control은 `-2.49755541`이다. 반면

\[
\int_1^X x^2(\rho+p_r)\,dx
=-\frac23\ln X+O(1)\to-\infty
\]

이고 proper-volume weight도 같은 로그 발산을 갖는다. 따라서 `finite positive
ADM`과 `finite negative affine ANEC`는 맞지만, 기존 target은 **공간적으로
국소화된 finite exotic source가 아니다**.

## 1순위 꼬리 보강: ADM-matched redshift

shape는 유지하고 redshift만

\[
\boxed{
\Phi_{\rm match}
=\frac12\ln\left(1-\frac{2}{3x}\right)+\frac32e^{1-x}}
\]

로 바꾼다. 그러면 \(\Phi'_{\rm match}(1)=-1/2\)라 throat Casimir tensor를
그대로 보존하고

\[
e^{2\Phi_{\rm match}}
=\left(1-\frac{2}{3x}\right)e^{3s}\ge\frac13
\]

이므로 지평선도 없다. 무한대에서는 ADM 질량 \(M/r_0=1/3\)에 맞는
Schwarzschild lapse와 지수감쇠 보정만 남는다. 직접 계산하면

\[
\boxed{
\rho+p_r=-\frac{s}{x^2}
\left[\frac13+\frac1{3x-2}+(3x-2-s)\right]<0}
\]

이고 `x^2(rho+p_r) ~ -3x exp(1-x)`라 affine ANEC, coordinate-volume NEC,
proper-volume NEC가 모두 유한·음수다. 즉 NEC 자체는 제거하지 않지만 기존의
로그 무한 source burden은 제거한다.

이 보강 profile은 throat에서 비최소장 국소 계수 `K/F=7/12>0`도 만들지만,
전역 reconstruction에서는 `min K/F≈-1.83055` at `x≈1.15638`로 다시 실패한다.
따라서 `tail repair PASS`를 `healthy global scalar PASS`로 승격하지 않는다.

| gate | 기존 exponential | ADM-matched 보강 |
|---|---:|---:|
| throat Casimir tensor | exact | exact |
| shape gap·flare-out·양쪽 연장 | exact | exact |
| lapse lower bound | 1 | 1/3 |
| 각 end ADM 질량 \(M/r_0\) | 1/3 | 1/3 |
| Bianchi identity | exact | exact |
| radial affine ANEC | finite, negative | finite, negative |
| coordinate/proper volume NEC burden | log divergent | finite |
| 독립 CE matter EOM·stability | not derived | not derived |

## 유한 다중 모드 분해

구간 \(1\le x\le10\)에서 공통 Chebyshev 공간 기저를 사용하고 각 모드에
\((\rho,p_r,p_t)\) 계수 벡터를 주었다. 독립 검증 격자의 최대 정규화 오차는 다음과 같다.

| 모드 수 | 최대 정규화 오차 |
|---:|---:|
| 4 | \(3.1243\times10^{-1}\) |
| 8 | \(4.5984\times10^{-2}\) |
| 16 | \(1.4304\times10^{-3}\) |
| 32 | \(1.6589\times10^{-7}\) |

따라서 이 목표 텐서는 compact radial 구간에서 유한 모드 합으로 빠르게 수렴한다.
이는 **목표의 수학적 합성 가능성**만 증명한다.

## 아직 닫히지 않은 물리 bridge

다음 명제들은 이번 결과로 증명되지 않았다.

1. Chebyshev 기저가 실제 경계장치의 고유 공명 spectrum이라는 명제
2. 약 169 GeV carrier와 거시적 radial envelope 사이의 결합
3. 양자화·재규격화된 CE 모드가 목표 부호와 크기의 음의 \(T_{\mu\nu}\)를 낸다는 명제
4. 양자부등식, backreaction, 선형 섭동 안정성의 동시 통과

따라서 현재 판정은 다음과 같다.

| 층 | 판정 |
|---|---|
| 고정 Casimir EoS 전역해 | `REFUTED` |
| 기존 exponential 전역 기하 target | `EXACT GEOMETRY / SOURCE-TAIL FAIL` |
| ADM-matched 전역 기하 target | `EXACT GEOMETRY + FINITE-TAIL CONTROL` |
| 유한 모드 target 근사 | `NUMERICAL CONTROL PASS` |
| CE 물리 공명 spectrum과의 동일시 | `OPEN` |
| renormalized negative stress | `OPEN` |
| 안정한 통과가능 wormhole | `NOT PROVED` |

## 재현

```powershell
uv run pytest tests/test_multimode_global_throat.py -q
uv run python examples/physics/multimode_global_throat_gate.py
```
