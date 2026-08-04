# 실시간 chronology interlock 루프

## 1. 동적 문제

정적 mouth graph가 안전해도 통과시간, 입구 시계, switching latency는 시간에
따라 변할 수 있다. 프레임 $k$에서 측정된 edge 시간을 $\hat w_{ij}^{(k)}$,
측정 불확실성을 $u$, 최대 edge drift rate를 $r$, 샘플 간격을 $\Delta t$라
하면 다음 샘플 전까지 사용할 보수적 하한은

\[
\underline w_{ij}^{(k)}
=\hat w_{ij}^{(k)}-u-r\Delta t
\]

로 둔다.

## 2. Fail-closed 활성화 규칙

각 프레임에서 edge를 결정적 순서로 하나씩 추가한다. 추가한 graph가

\[
\underline w_{ij}+s_j-s_i\ge\epsilon
\]

을 만족하는 clock-offset 해를 잃으면 그 edge를 enable하지 않는다.
따라서 최종 enabled graph는 입력한 uncertainty와 drift bound 안에서 항상
정적 difference-constraint gate를 통과한다.

센서 프레임에 NaN이 하나라도 있으면 해당 프레임의 모든 edge를 닫는다.
알 수 없는 값을 안전하다고 추정하지 않는 fail-closed 정책이다.

## 3. 실행 결과

정상 양방향 회로는 두 edge를 모두 유지했다. 같은 측정에 uncertainty와
다음 샘플까지의 drift를 적용하여 robust cycle이 위험해진 프레임에서는
두 edge 중 하나를 차단했다. $-2+1=-1$초의 정적 음의 회로에서도 두 번째
edge가 회로를 닫는 순간 차단되어 enabled count 1, disabled count 1이 됐다.

센서 fault 프레임은 enabled count 0이 되어 fail-closed를 확인했다.

## 4. 증명 범위

이 greedy interlock은 안전한 부분 graph를 만들지만 다음을 증명하지 않는다.

- 차단 edge 수가 최소라는 것
- 네트워크 처리량이 최대라는 것
- 주어진 uncertainty와 drift bound가 현실에서 지켜진다는 것
- 샘플링·연산·스위치 차단 자체의 latency가 bound 안이라는 것
- graph 밖 연속 시공간의 모든 causal curve가 안전하다는 것

특히 실제 장치에서는 $u$, $r$, $\Delta t$, 계산시간과 차단시간을 모두
$\epsilon$ budget에 포함해야 한다. bound 위반 감지 자체가 늦을 수 있으므로
독립 하드웨어 차단 경로도 필요하다.

## 5. 판정

| 명제 | 판정 |
|---|---|
| bounded uncertainty/drift 아래 enabled graph의 동기화 가능성 | `PROVED / FINITE` |
| 위험 cycle을 닫는 edge의 자동 차단 | `PROVED / FINITE` |
| 센서 fault fail-closed | `PROVED / CODE` |
| greedy 차단이 최소 차단 또는 최대 처리량 | `NOT CLAIMED` |
| 실제 uncertainty/drift bound의 물리적 보장 | `OPEN` |
| 연속 시공간 chronology protection | `OPEN` |

## 6. 실행

```powershell
uv run --extra dev python -m pytest tests/test_preinstalled_mouth_network.py -q
uv run python examples/physics/preinstalled_mouth_network_gate.py
```
