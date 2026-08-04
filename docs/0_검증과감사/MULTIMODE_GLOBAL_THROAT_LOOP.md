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

## 전역 gate 결과

| gate | 결과 |
|---|---:|
| flare-out \(b'(r_0)<1\) | PASS |
| \(x>1\)에서 \(b/r<1\) | PASS |
| 유한 lapse, 지평선 없음 | PASS |
| 보존식 최대 잔차 | \(1.33\times10^{-15}\) |
| \(b/r\to0,\ \Phi\to0\) | PASS |
| \(b(\infty)=2r_0/3\) | PASS |
| ADM 질량 길이 \(M=b(\infty)/2\) | \(r_0/3\), finite |
| 양쪽 throat로 기하학적 연장 | AVAILABLE |

따라서 **전역 기하/제어 target의 존재**는 통과했다. 이것은 물질장을 먼저 가정한
해가 아니라, 필요한 \(T_{\mu\nu}(r)\)를 Einstein 방정식으로 역산한 결과다.

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
| 가변 비등방 전역 기하 target | `EXACT/FINITE CONTROL PASS` |
| 유한 모드 target 근사 | `NUMERICAL CONTROL PASS` |
| CE 물리 공명 spectrum과의 동일시 | `OPEN` |
| renormalized negative stress | `OPEN` |
| 안정한 통과가능 wormhole | `NOT PROVED` |

## 재현

```powershell
uv run pytest tests/test_multimode_global_throat.py -q
uv run python examples/physics/multimode_global_throat_gate.py
```
