# 사전 설치 공간접힘 입구망 루프

## 1. 바뀐 문제

목표 지점에 응력원을 즉시 생성하는 브리지는 앞 루프에서 닫히지 않았다.
여기서는 원격 입구와 응력원이 이미 설치되어 있다고 가정하고, 0차원
경계 선택기가 유한 입구 집합 위의 경로만 선택하게 한다.

```text
전체 이력 readout -> 0D 목적지 선택 -> 설치된 mouth graph의 최단시간 경로
```

이 가정은 원격 물질 생성을 제거하지만 입구 제작의 어려움을 해결하지는
않는다.

## 2. 유한 입구망 라우팅

입구를 graph node, 목을 directed edge로 두고 edge $i\to j$의 비용을

\[
\tau_{ij}=\frac{\ell_{ij}}{\beta c}+\tau_{\rm switch},
\qquad 0<\beta<1
\]

로 둔다. 모든 비용이 음이 아니므로 Dijkstra 경로가 최소 통과시간을
준다. 1광년 떨어진 입구 두 개를 길이 10m의 목으로 연결하고
$\beta=0.1$이면

\[
\tau=3.33564095\times10^{-7}\ {\rm s}.
\]

외부 빛보다 매우 빠르지만 유한 목과 아광속 이동이므로 정확히 0초는
아니다. 이 계산은 기하학적 control gate이며 실제 입구의 존재를 유도하지
않는다.

## 3. 임의 위치 문제

사전 설치 방식의 도착점은 mouth 좌표의 유한 집합이다. 요청 위치 $x$와
가장 가까운 입구의 거리를

\[
d_{\min}(x)=\min_i\|x-x_i\|
\]

라 하면 허용오차 $\epsilon$ 안의 도착은 $d_{\min}\le\epsilon$일 때만
가능하다. 실행 gate에서 두 입구의 정확한 중간점은 1m 허용오차로 덮이지
않았다. 따라서 입구망은 임의의 연속 좌표로 순간이동시키지 않는다.

## 4. 시간 오프셋과 chronology gate

각 directed edge의 좌표시간 경과 $\Delta t_{ij}$에는 통과시간과 두 입구의
시계 오프셋을 함께 넣는다. 어떤 directed cycle $C$가

\[
\sum_{(i,j)\in C}\Delta t_{ij}<0
\]

이면 출발보다 이른 좌표시간으로 돌아오는 이산 chronology loop다.
Bellman--Ford negative-cycle gate가 예제의 $-2+1=-1$초 회로를 검출했다.

negative cycle이 없는 것은 해당 유한 graph의 통과일정만 검사한 결과다.
연속 시공간에 전역 시간함수가 존재한다는 증명은 아니며, 그 조건은 기존
시간여행 루프의 A1으로 남는다.

## 5. 판정

| 명제 | 판정 |
|---|---|
| 설치된 유한 입구망의 최소시간 경로 선택 | `PROVED / FINITE` |
| 1광년 외부거리·10m 목의 효과적 shortcut | `PROVED / KINEMATIC` |
| 정확히 0초 이동 | `REFUTED` |
| 유한 입구망으로 임의 연속 위치 도착 | `REFUTED` |
| 음의 좌표시간 cycle 검출 | `PROVED / FINITE` |
| graph 통과가 연속 시공간 chronology protection을 증명 | `REFUTED` |
| 입구망의 물리적 제작·유지 | `OPEN` |

후속 `MOUTH_NETWORK_CHRONOLOGY_PROTECTION_LOOP.md`에서 입구별 clock offset을
difference constraints로 풀었다. 모든 cycle의 총시간은 offset 변경에
불변이므로 음의 cycle은 재동기화로 제거할 수 없고 edge 차단이 필요하다.

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_preinstalled_mouth_network.py -q
uv run python examples/physics/preinstalled_mouth_network_gate.py
```
