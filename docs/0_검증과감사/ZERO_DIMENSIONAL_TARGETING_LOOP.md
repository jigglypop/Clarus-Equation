# 0차원 관조와 3차원 위치 선택 루프

## 1. 요구사항의 최소 정식화

(d=0)을 공간좌표가 없는 singleton (X_0=\{*\})으로 읽는다. 목표는
3차원 후보위치 집합 (X=\{x_1,\ldots,x_N\}) 중 하나를 선택하고, 그
위치에 공간접힘 입구를 국소화하는 것이다.

## 2. 무입력 0D 위치식별 불가능 정리

singleton에서 위치집합으로 가는 함수 (f:X_0\to X)는 하나의 상수값만
낸다. 가능한 상수함수는 (N)개지만 어떤 함수를 택할지는 (X_0) 내부
정보에서 나오지 않는다. 위치 label 하나에는 최소

\[
I_{\rm target}=\log_2N
\]

bits가 필요하지만 singleton의 무입력 구별정보는 0 bits다. 공간 대칭을
깨는 외부 readout이나 미리 인코딩된 상수가 없으면 모든 위치는 동일하며
분포는 (1/N)이다.

따라서 “0차원 자체가 3차원 전체를 보고 위치를 정한다”는 명제는
반증된다. 0D 선택 노드는 위치를 **내재적으로 보지 않고**, 3D 또는
경로공간에서 계산된 정보를 입력받아야 한다.

연속 공간에서도 유한 정밀도 (delta x)로 길이 (L)인 3차원 영역의
위치를 고르려면 최소

\[
I\ge3\log_2(L/\delta x)
\]

bits가 필요하다. 1광년 입방영역에서 1 m 분해능이면 약 159.2 bits다.

## 3. 가능한 형태: 0D 경계 선택기

전체 역사 (h), 전체 시간 (t), 후보위치 (x)에서 계산된 비용을
(C(h,t,x))라 하자. 0D 경계 노드는 좌표를 갖는 대신 다음 전역
함수형을 입력받는다.

\[
J(x)=\sum_h p(h)\sum_tw(t)C(h,t,x).
\]

목표분포와 선택은

\[
p_\beta(x)=\frac{e^{-\beta J(x)}}{\sum_y e^{-\beta J(y)}},
\qquad
x_*\in\operatorname*{argmin}_xJ(x)
\]

로 정의할 수 있다. 이 유한 선택 연산은 실행 gate에서 닫힌다. 하지만
위치정보는 (C(h,t,x))가 공급하므로 실제 정보원은 3D 역사/readout이다.
“0D”는 집계 결과의 좌표 없는 경계 노드라는 뜻이다.

## 4. 모든 시간을 볼 때 생기는 자기일관성 조건

선택 (a)가 완전역사 (gamma_a)를 바꾸면 단순 최적화가 아니라

\[
a\in\operatorname*{argmin}_xJ(x;\gamma_a)
\]

를 만족해야 한다. 유한 cost matrix (C_{ax}=J(x;\gamma_a))의 각 행에서
선택된 최소점이 행 label (a) 자신을 포함할 때만 고정점이다.

실행 반례는 고정점 0개, 2개, 유일한 1개가 모두 가능함을 보인다. 따라서
“전체 미래를 본 뒤 선택”은 항상 유일하게 정의되지 않는다. 존재·유일성
조건 또는 추가 선택법칙이 필요하다.

## 5. 공간접힘과의 연결

현재 닫힌 파이프라인은 다음까지다.

```text
3D 완전역사/공간 readout
        -> 전역 비용 J(x)
        -> 좌표 없는 0D 경계 집계
        -> 자기일관 target x*
```

아직 닫히지 않은 단계는

```text
x*
  -> 목적지의 국소 CE stress 생성
  -> 두 번째 입구 생성/지정
  -> 안정한 통과 가능 웜홀
```

이다. 목표 label을 계산하는 것과 목적지에 물리적 장을 생성하는 것은
별개다. 후자는 목적지와의 기존 인과 채널, 사전 설치된 입구 또는 비국소
작용을 요구한다.

## 6. 판정

| 명제 | 판정 |
|---|---|
| 무입력 0D가 여러 3D 위치를 구별 | `REFUTED` |
| 3D 전역 readout을 받은 0D 경계 노드가 위치 label 선택 | `PROVED / FINITE` |
| 전체시간 readout 선택의 고정점이 항상 존재·유일 | `REFUTED` |
| 선택된 위치에 CE 장을 비국소 생성 | `OPEN` |
| 선택만으로 공간접힘 웜홀 생성 | `REFUTED` |

후속 `TARGET_TO_STRESS_ACTUATION_LOOP.md`에서 이 열린 단계를 세분했다.
위치별 actuator는 CE에서 유도되지 않았고, 적응적 원격 명령은 광원뿔
지연을 받으며, 현재 CE 셀 밀도와 단일-domain coherence 제어조건의 목
반지름 구간은 서로 겹치지 않는다.

## 실행

```powershell
uv run --extra dev python -m pytest tests/test_zero_dimensional_targeting.py -q
uv run python examples/physics/zero_dimensional_targeting_gate.py
```
