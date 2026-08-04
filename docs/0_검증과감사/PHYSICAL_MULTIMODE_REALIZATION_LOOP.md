# 다중 공명 현실화 gate

## 판정 기준

이번 루프부터 기하학적 역설계와 모드 근사만으로는 후보를 통과시키지 않는다. 다음 연결이
모두 있어야 현실화 후보로 남긴다.

\[
\text{물리적 장과 경계}
\rightarrow \text{고유모드 spectrum}
\rightarrow \langle T_{\mu\nu}\rangle_{\rm ren}
\rightarrow \text{backreaction}
\rightarrow \text{수명·안정성}.
\]

## 1 m 전역 target의 정확한 물리량

새 전역 target의 목에서는

\[
C=\frac{c^4}{8\pi G r_0^2},\qquad
\rho=-\frac C3,\qquad p_r=-C,
\]

이므로 \(|\rho+p_r|=4C/3\)이다. 이는 앞선 \(b'=-1\) scale control과 다른
기하이므로 이상적 Casimir 간격을 다시 계산해야 한다.

\[
u_{\rm Casimir}=\frac{\pi^2\hbar c}{720a^4}=\frac C3
\]

에서 \(r_0=1\) m이면

\[
a=4.0536\times10^{-18}\ {\rm m},\qquad
\lambda=2a,\qquad h f=152.93\ {\rm GeV}.
\]

따라서 이전의 3.66 am/169 GeV 값은 \(b'=-1\) 음의-null scale control에는 맞지만,
현재의 정확한 \(b'=-1/3\) Casimir target에는 적용하지 않는다.

## 물질 경계와 크기 trade-off

양성자 전하반지름 0.84 fm을 매우 관대한 경계 해상도 기준으로 사용해도 1 m target의
간격은 그 약 0.00483배다. 이 에너지에서 완전도체처럼 동작하는 구면 경계의 미시 모델은
제시되지 않았다.

Casimir 간격은 \(a\propto\sqrt{r_0}\)이므로 간격을 0.84 fm까지 키우려면

\[
r_0\simeq4.294\times10^4\ {\rm m}
\]

가 필요하다. 에너지 비용에는 서로 다른 적분을 혼동하지 않는 것이 중요하다.
한쪽 asymptotic end에서 Misner--Sharp 질량은

\[
m_{\rm MS}(r)=\frac{c^2b(r)}{2G},\qquad
M_{\rm ADM}=m_{\rm MS}(\infty)=\frac{c^2r_0}{3G}
\]

이고, areal-coordinate measure로 적분한 한쪽 density는

\[
\int_{r_0}^{\infty}\rho\,4\pi r^2dr
=\left[m_{\rm MS}(\infty)-m_{\rm MS}(r_0)\right]c^2
=-\frac{c^4r_0}{6G}.
\]

따라서 양쪽 변화량의 절댓값은

\[
|\Delta E_{\rm MS}|_{\rm two-sided}=\frac{c^4r_0}{3G}
\]

이다. 이 양은 양쪽 ADM energy도, proper-volume matter energy도 아니다. 정적 공간절편의
proper volume은

\[
dV=\frac{4\pi r^2dr}{\sqrt{1-b(r)/r}}
\]

이므로, \(x=r/r_0\), \(y=b/r_0=2/3+e^{-(x-1)}/3\)에 대해

\[
I=\int_1^\infty\frac{-y'(x)}{\sqrt{1-y(x)/x}}dx
=0.631466179317807\ldots,
\]

\[
|E_{\rm proper}|_{\rm two-sided}
=I\frac{c^4r_0}{G}.
\]

목의 적분가능 특이점은 \(x=1+s^2\)로 제거했다. 이때 integrand의 \(s=0\) 극한은
\(1/\sqrt3\)이고, \(s=8\) 이후 analytic tail bound는 \(5.4\times10^{-29}\)보다
작다. Simpson 4,000/2,000 분할의 차이도 코드에서 별도로 기록한다. 다만 interval
arithmetic 인증은 아니므로 소수값의 상태는 `Numerical control`이다.

1 m에서 coordinate-density 적분은 \(4.03\times10^{43}\) J, 질량환산 지구 약
75.2개지만 proper matter energy는 \(7.642\times10^{43}\) J, 지구 약 142.38개다.
42.9 km에서는 각각 태양 약 9.69개와 18.36개 질량환산이다. 두 비용 모두 반지름에
선형 증가하므로 경계 해상도를 개선해도 현재 모형 안에는 공학적으로 유리한 크기
구간이 없다.

## 다중 모드가 해결하지 못하는 것

32개 모드는 목표 tensor의 공간 모양을 잘 근사하지만, 같은 Einstein tensor를 만드는
한 총 적분 에너지는 고정된다. 선형 모드 수 \(N\)을 늘리면 각 모드가 분담할 수는 있어도
합계 \(E_-\)를 줄이지 못한다.

실험적으로 확인된 동적 Casimir 효과는 SQUID 경계를 약 11 GHz에서 변조하여 마이크로파
광자와 two-mode squeezing을 만든 경우다. 이는 다중모드 양자제어가 실재함을 보여주지만,
153 GeV의 정적·중력적 음의 응력원을 보여주지는 않는다.

펄스/squeezed 경로의 규모 제어로 평탄공간 질량 없는 scalar의 Lorentzian sampling QI를
사용하면 1 m target 밀도에서 \(\tau\sim1.23\times10^{-26}\) s다. 1 m 광통과시간과의
비는 약 \(2.70\times10^{17}\)이다. 이 계산은 정적 Casimir 경계진공에 직접 적용하는
no-go가 아니지만, 동적 공명 펄스가 장시간 정적 source를 대신한다는 주장을 기각한다.

## 현실성에 따른 후보 재정렬

| 경로 | 판정 | 이유 |
|---|---|---|
| 물질판 Casimir + 다중 공명 | `DEFERRED/PHYSICAL FAIL` | 4.05 am 반사경과 재규격화 응력 모델 없음 |
| 동적 Casimir/squeezed pulse | `REFUTED AS STATIC SOURCE` | 실험은 광자 생성이며 지속 음의 중력 source가 아님 |
| 4D magnetic charged-fermion long wormhole | `THEORETICALLY PHYSICAL, NOT A SHORTCUT` | Einstein-Maxwell+질량 없는 fermion의 명시 모델이나 외부보다 느린 long wormhole |
| CE 비물질 경계/위상 sector | `OPEN, DECISIVE` | 물질판 없이 음의 재규격화 응력을 유도해야 함 |
| 수정중력으로 요구 \(T_{\mu\nu}\) 축소 | `EXTERNAL EXTENSION` | 현재 CE 작용에는 필요한 항이 없음 |

알려진 QFT 안에서 상대적으로 명시적인 4차원 후보는 Maldacena--Milekhin--Popov의
Einstein--Maxwell + charged massless fermion long wormhole이다. 그러나 이는 ambient
space의 인과율을 깨지 않고, 외부 경로보다 빠른 순간이동 shortcut을 제공하지 않는 쪽이다.
즉 현실성을 유지하면 shortcut을 잃고, shortcut을 유지하려면 CE 고유의 새로운
재규격화 응력 또는 중력 작용을 실제로 유도해야 한다.

## 근거 문헌

- [Maldacena, Milekhin, Popov: Traversable wormholes in four dimensions](https://arxiv.org/abs/1807.04726)
- [Kontou: Wormhole restrictions from quantum energy inequalities](https://arxiv.org/abs/2405.05963)
- [Fewster, Eveson: Bounds on negative energy densities in flat spacetime](https://arxiv.org/abs/gr-qc/9805024)
- [Wilson et al.: Observation of the Dynamical Casimir Effect](https://arxiv.org/abs/1105.4714)

## 재현

```powershell
uv run pytest tests/test_physical_multimode_realization.py -q
uv run python examples/physics/physical_multimode_realization_gate.py
```
