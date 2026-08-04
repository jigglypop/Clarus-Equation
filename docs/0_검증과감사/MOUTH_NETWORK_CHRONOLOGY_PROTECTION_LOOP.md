# 입구망 시간 동기화와 chronology protection 루프

## 1. 목표

사전 설치 입구망의 각 directed 통과시간을 $w_{ij}$라 하자. 입구마다
시계 offset $s_i$를 주어 모든 허용 통과가 최소 $\epsilon>0$만큼 미래로
향하게 만들 수 있는지 검사한다.

\[
w'_{ij}=w_{ij}+s_j-s_i\ge\epsilon.
\]

$\epsilon$은 단순 양수 조건뿐 아니라 clock drift, switching jitter와
측정오차를 흡수할 안전여유로 읽을 수 있다.

## 2. 차분 제약식

조건을 정리하면

\[
s_i\le s_j+w_{ij}-\epsilon
\]

인 difference constraints가 된다. Bellman--Ford feasibility gate로 offset
해를 만들거나 모순 cycle을 검출할 수 있다.

모든 directed cycle $C$에 대해 offset 항은 telescoping되어

\[
\sum_C w'_{ij}=\sum_Cw_{ij}
\]

이다. 따라서 cycle의 총시간은 시계 좌표 선택에 불변이다. 각 edge에
$\epsilon$의 미래 여유를 주려면 필요조건이자 유한 graph에서의 충분조건은

\[
\sum_{(i,j)\in C}w_{ij}\ge |C|\epsilon
\quad\text{for every directed cycle }C
\]

이다.

## 3. 실행 반례와 양성 대조

두 입구 회로 $w_{01}=-2$초, $w_{10}=3$초는 총 $+1$초다.

- $\epsilon=0.4$초이면 cycle 요구량은 $0.8$초이므로 offset 해가 존재한다.
- $\epsilon=0.6$초이면 요구량은 $1.2$초이므로 해가 없다.

$w_{01}=-2$초, $w_{10}=1$초인 총 $-1$초 회로는 $\epsilon=0$에서도
해가 없다. 좌표 재설정으로 과거행 회로를 제거할 수 없다는 직접 반례다.

총시간이 정확히 0인 회로는 $\epsilon=0$의 비감소 labeling은 가능하지만
어떤 $\epsilon>0$도 허용하지 않는다. chronology protection에는 엄격한
양의 margin이 필요하다.

반대로 요청 margin이 0일 때 Bellman--Ford가 반환한 임의의 offset에 0인 edge가
남는다는 이유만으로 엄격 시간함수의 **존재**를 기각하면 안 된다. 예를 들어

\[
w_{01}=0,\qquad w_{10}=1
\]

에서 offset $(0,0)$은 adjusted edge $(0,1)$을 주지만, $(s_0,s_1)=(0,0.5)$는
$(0.5,0.5)$를 주므로 엄격 해가 존재한다. 이전 코드는 첫 witness만 보고 이를
거짓 음성으로 판정했다.

유한 graph에서 가능한 최대 균일 margin은 cycle mean으로 특징지어진다.

\[
\epsilon_*=\min_C\frac{\sum_C w_{ij}}{|C|}.
\]

따라서 directed cycle이 있으면 모든 cycle total이 양수일 때, DAG이면 제한 없이
엄격 시간함수가 존재한다. 구현은 all-pairs shortest paths로 최소 directed-cycle
total $\delta$를 구한다. $\delta>0$이면 node 수를 $n$이라 할 때
\(\epsilon=\delta/(2n)>0\)을 안전한 witness margin으로 선택해 difference
constraints를 다시 푼다. 위 2-node 반례에서는 최소 cycle total 1초와 양의 strict
witness가 실제로 반환된다.

## 4. 공학적 보호 규칙

유한 제어망에서는 다음 순서가 필요하다.

1. 모든 mouth edge의 통과시간과 clock offset을 합쳐 $w_{ij}$를 측정한다.
2. 예상 drift와 jitter보다 큰 $\epsilon$을 사전 지정한다.
3. difference-constraint gate가 통과할 때만 edge를 enable한다.
4. 실패하면 해당 cycle의 edge 하나 이상을 차단해야 한다. clock relabel만
   바꾸는 것은 해결책이 아니다.
5. 운용 중 $w_{ij}$가 변할 때마다 gate를 다시 계산한다.

이는 graph 수준의 interlock이다. 연속 시공간의 모든 timelike/null curve를
포괄하지 않으므로 전역 시간함수와 반작용 안정성은 별도의 물리 증명으로
남는다.

## 5. 판정

| 명제 | 판정 |
|---|---|
| 유한 mouth graph의 clock-offset feasibility | `PROVED / FINITE` |
| cycle 총시간의 clock-offset 불변성 | `PROVED` |
| 음의 시간 cycle을 시계 재설정으로 제거 | `REFUTED` |
| zero-time cycle이 엄격한 미래 margin 허용 | `REFUTED` |
| margin 0의 임의 witness만으로 strict 존재를 판정 | `REFUTED / CODE CORRECTED` |
| 모든 cycle total 양수 또는 DAG일 때 strict witness 구성 | `PROVED / FINITE` |
| 양의 cycle budget 안에서 미래 margin 배분 | `PROVED / FINITE` |
| graph interlock만으로 연속 시공간 chronology 보호 | `OPEN` |

후속 `REALTIME_CHRONOLOGY_INTERLOCK_LOOP.md`에서 측정 불확실성과 샘플 사이
drift를 뺀 robust lower bound로 매 프레임 edge를 재검사하고, 위험 edge와
센서 fault를 fail-closed로 차단하는 동적 gate를 구현했다.
