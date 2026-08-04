# 0D 목표에서 국소 응력원으로의 작동 루프

## 1. 검증할 브리지

앞 단계는 전체 이력의 비용을 0차원 경계에서 집계하여 목표 label
$x_*$를 선택했다. 이번 단계는 다음 사상이 실제로 닫히는지 검사한다.

```text
x* -> 목적지별 장 응답 -> NEC 위반 응력 -> 통과 가능 목
```

목표 label은 정보이고 $T_{\mu\nu}(x)$는 물리장이다. 둘 사이에는
위치 의존 actuator, 명령 전달 경로, 충분한 응력 크기와 공간적 coherence가
모두 필요하다.

## 2. 위치 국소화 관문

$A_{ia}$를 목표 명령 $a$를 주었을 때 위치 $i$에서 생기는 음의 null-projected
응력 크기라 하자. 명령 $a$가 위치 $a$를 고르려면 최소한

\[
A_{aa}>A_{ia}\quad(i\ne a)
\]

이어야 한다. 모든 열이 같은 broadcast 행렬은 rank 1이고 어느 위치도
고유하게 고르지 못한다. 대각 행렬은 목표를 고르지만, 그 행렬을 입력한
것 자체가 위치별 actuator를 가정한 것이다. 현재 CE 작용에서 이
$A_{ia}$는 유도되지 않았다.

## 3. 인과적 명령 전달 관문

$N$개 후보 중 새 목표를 적응적으로 정하면 최소 $\log_2N$ bit를 목적지
측 receiver에 전달해야 한다. 신호속도가 $v\le c$이면 거리 $L$에 대해

\[
t_{\min}=\frac{L}{v}\ge\frac{L}{c}.
\]

1광년 거리의 8개 후보는 3 bit가 필요하고, 가장 빠른 전달도 31,557,600초,
즉 1년이다. 사전 설치 receiver는 장을 받을 장치를 제공할 뿐, 지금 새로
고른 정보를 광원뿔보다 먼저 전달하지 않는다. 미리 합의한 스케줄은
적응적 목표 선택이 아니다.

따라서 비국소 CE 항을 별도로 도입하지 않는 한 즉시 원격 작동은 반증된다.
비국소 항을 도입하면 no-signalling과 에너지-운동량 보존을 새로 증명해야
하므로 현재 이론의 결론으로 셀 수 없다.

## 4. 밀도–coherence 반지름 창

zero-redshift Morris--Thorne 제어목에서

\[
|\rho+p_r|=\frac{c^4(1-b')}{8\pi G r_0^2}.
\]

후보 음의 밀도 크기를 $\epsilon$이라 하면 밀도 조건은

\[
r_0\ge r_{\min}
=\sqrt{\frac{c^4(1-b')}{8\pi G\epsilon}}
\]

을 요구한다. 한편 현재 코드의 보수적 단일-domain coherence 관문은 CE
상관길이 $\xi$가 목을 덮도록 $r_0\le r_{\max}=\xi$를 요구한다.

문서의 명시값

- $\epsilon=3.3787398404\times10^{26}\,\mathrm{J/m^3}$,
- $\xi=6.65\times10^{-15}\,\mathrm m$,
- $b'=-1$

을 넣으면

\[
r_{\min}=1.6883\times10^8\,\mathrm m,
\qquad
r_{\max}=6.65\times10^{-15}\,\mathrm m.
\]

$r_{\min}/r_{\max}\simeq2.54\times10^{22}$이므로 이 제어모형에서는 가능한
반지름 창이 없다. 다중 셀의 장거리 coherent state가 이 상관길이 제한을
바꿀 수는 있지만, 그러려면 그 상태의 전체 renormalized $T_{\mu\nu}$,
보존, backreaction과 안정성을 별도로 제시해야 한다.

## 5. 판정

| 명제 | 판정 |
|---|---|
| broadcast 응력장이 목표 위치를 고유 지정 | `REFUTED` |
| 외부에서 주어진 full-rank actuator가 목표를 국소화 | `PROVED / FINITE` |
| 그 actuator가 CE 작용에서 유도됨 | `OPEN` |
| 1광년의 새 적응적 목표를 즉시 전달 | `REFUTED / LOCAL CAUSALITY` |
| 현재 CE 셀 밀도와 단일-domain coherence가 동시에 허용하는 목 반지름 | `EMPTY / CONTROL MODEL` |
| 안정한 통과 가능 웜홀 | `NOT ESTABLISHED` |

후속 `PREINSTALLED_MOUTH_NETWORK_LOOP.md`에서는 목적지 입구가 이미 있다는
가정 아래 유한 graph 라우팅을 검사했다. 이 경우 원격 응력 생성은 피할 수
있지만 도착점은 설치된 입구 집합으로 제한되고, 유한 통과시간과 chronology
cycle 검사가 남는다.

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_targeted_spatial_actuation.py -q
uv run python examples/physics/targeted_spatial_actuation_gate.py
```
