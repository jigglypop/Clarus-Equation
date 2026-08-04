# Warp-drive 현실성 루프

## 범위와 판정

이 루프는 물질이나 정보를 순간 복제하는 양자 텔레포트가 아니라, 주어진 warp
metric이 외부 좌표에서 공간지름길을 만들 수 있는지와 그 metric에 필요한 중력원 비용을
감사한다. 현재 결과는 다음 단계에 머문다.

\[
\boxed{\text{normalized geometry + numerical stress control / physical source absent}}
\]

## 정규화한 Alcubierre profile

단위 lapse, 평탄한 공간절편, shift \(\beta_x=-vf(r)\)를 사용하고

\[
f(r)=
\frac{\tanh((r+R)/\Delta)-\tanh((r-R)/\Delta)}
{2\tanh(R/\Delta)}
\]

로 두었다. 이 profile은

\[
f(0)=1,\qquad f'(0)=0,\qquad \lim_{r\to\infty}f(r)=0
\]

를 만족한다. 이전 half-tanh는 \(r=0\)에서 미분이 정확히 0이 아니어서 3차원 radial
scalar의 원점 매끄러움을 엄밀히 만족하지 않았다.

수치 구현은 `cosh((r-R)/Delta)`를 직접 계산하지 않는다. 따라서
\(R/\Delta=1000\)인 10 m/1 cm wall에서도 overflow가 없고, 적분변수
\(u=(r-R)/\Delta\)를 사용해 벽 두께와 무관하게 wall을 해상한다. 4,000/2,000점
Simpson 결과 차이를 수치 오차 지표로 남긴다.

## Eulerian energy와 WEC

Eulerian observer가 측정하는 에너지밀도는

\[
\rho_E(r,\theta)=
-\frac{c^4\beta^2}{32\pi G}\,[f'(r)]^2\sin^2\theta
\]

이고, 각도적분한 총량은

\[
E_E=-\frac{c^4\beta^2}{12G}
\int_0^\infty r^2[f'(r)]^2dr.
\]

따라서 \(v\ne0\)인 비자명한 wall의 적도에서는 \(\rho_E<0\)이며 WEC 위반이 직접
계산된다. 10 m 반지름, 1 m wall에서는 다음과 같다.

| 속도 | Eulerian wall energy | 질량환산 | 최소 Eulerian density |
|---:|---:|---:|---:|
| `0.5c` | `-8.4317e43 J` | 음의 지구질량 `157.09`개 | `-7.5241e40 J/m^3` |
| `1c` | `-3.3727e44 J` | 음의 지구질량 `628.34`개 | `-3.0097e41 J/m^3` |
| `2c` | `-1.3491e45 J` | 음의 지구질량 `2513.37`개 | `-1.2039e42 J/m^3` |
| `10c` | `-3.3727e46 J` | 음의 지구질량 `62834.34`개 | `-3.0097e43 J/m^3` |

이 계산은 WEC 위반을 증명하지만, 코드가 전체 \(T_{\mu\nu}k^\mu k^\nu\)를 직접
계산했다고 과장하지 않는다. `explicit_null_projection_computed=False`로 분리했다.
일반 Natario형 warp metric의 NEC no-go 조건이 적용된다는 판정은
[Santiago--Schuster--Visser](https://arxiv.org/abs/2105.03079)의 외부 정리를
사용한 것이다.

## 초광속 축 지평선

버블 중심과 함께 움직이는 축방향 null characteristic은 \(\beta=v/c>1\)일 때

\[
f(r_h)=1-\frac1\beta
\]

를 만족하는 front/back root를 갖는다. 코드는 이 조건을 boolean으로 기록하지 않고
정규화 profile에서 직접 이분법으로 푼다.

| 속도 | target `f(r_h)` | 축 지평선 반지름 |
|---:|---:|---:|
| `0.5c`, `1c` | 없음 | 없음 |
| `2c` | `0.5` | `10.000000004 m` |
| `10c` | `0.9` | `8.901387732 m` |

이는 고정속도 metric의 kinematic horizon 계산이며 형성·가속·안정성 해법은 아니다.

## 양의 에너지 shell 주장 재감사

[2024 constant-velocity solution](https://arxiv.org/abs/2405.02709)은 아광속
구면 shell과 양의 ADM mass의 내부 control을 제시했지만 초광속 지름길은 아니다.
더 최근의 source-consistent 재감사에서는 smoothing tail에서 Type-IV 에너지조건
위반이 검출됐고, 조사한 source-first 구성도 완전한 허용해를 만들지 못했다
([Le 2026](https://arxiv.org/abs/2605.25417)). 따라서 portfolio의
`all_observer_nec_gate`는 `False`로 내렸다. 이는 2024 내부 계산 전체가 무효라는 뜻이
아니라, 전역 all-observer 통과 주장이 현재 닫히지 않았다는 뜻이다.

## 남은 결정적 실패

- CE 또는 표준모형의 명시적 공변 material action이 이 stress를 만들지 못했다.
- renormalized source, backreaction, 형성 과정, 전체 선형·비선형 안정성이 없다.
- 초광속일 때 축 지평선이 생기며 제어·정지 문제를 별도로 풀어야 한다.
- 양의 Eulerian density 한 관찰자만 확인해서는 all-observer WEC/NEC가 되지 않는다.

따라서 현재 warp 결과는 공간지름길의 `W1/Kinematic + stress negative control`이며,
현실 텔레포트 장치나 CE 물질원 증명이 아니다.

## 재현

```powershell
uv run python examples/physics/warp_drive_reality_gate.py
uv run python -m pytest tests/test_warp_drive_reality.py -q
```
