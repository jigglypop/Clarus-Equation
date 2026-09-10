# 47. 세 좌표 연결의 Maxwell 반례 — CE-GF1

## 47.1 계산 전 등록

**[검사할 판본]** 2026-09-10. CE-GN3의 세 매끄러운 상태 좌표 $q^a(x)$,
$a=1,2,3$와 고정 위상에서 $A_\mu=\mathcal A_a(q)\partial_\mu q^a$만을
물리적 gauge 연결로 사용하는 해석을 검사한다. 독립적인 시간 연결이나
추가 상태 좌표·독립 gauge장은 가정하지 않는다.

- 모든 3차원 표적공간 2-form의 4차원 pullback이 $F\wedge F=0$을 만족하는지 증명한다.
- 고정 Maxwell 대조는 $F_*=dt\wedge dz+dx\wedge dy$다.
  국소 Minkowski 영역에서 Bianchi 식과 원천 없는 Maxwell 식을 확인한다.
- Jacobian 네 개와 표적 반대칭 행렬 세 개를 정수로 고정하고 12개 pullback의
  Pfaffian·rank를 정확 연산으로 검산한다. 특이 Jacobian도 포함한다.
- 독립 경로로 Levi-Civita 수축을 계산한다. 작은 진폭의 pullback에 단순히
  $F^2$ 작용을 붙였을 때 quadratic 광자항이 생기는지도 검사한다.
- 위 반례를 표시하는 최소 확장 대조는 네 좌표의 두 2-form 쌍이다.
  이는 새 통일장 후보의 완성이 아니라 차원 제한의 진단이다.
- 피팅 없음. 정확 항등식의 실패 또는 대조 Maxwell 해를 표현하지 못하면
  해당 세 좌표 연결을 일반적인 Maxwell 극한으로 해석하는 판본을 기각한다.

위 조건은 계산 전에 등록했다. 다음은 그 유도·검산·판정이다.

## 47.2 세 좌표 pullback의 정확한 제한

4차원 시공간에서 $q:M^4\to\mathcal T^3$를 매끄러운 사상이라 하자.
표적공간 연결을 $\mathcal A=\mathcal A_a(q)dq^a$라 하면 시공간 연결과 곡률은

$$
A=q^*\mathcal A,\qquad F=dA-iA\wedge A=q^*\mathcal F,
\qquad F_{\mu\nu}=\mathcal F_{ab}(q)\partial_\mu q^a\partial_\nu q^b.
$$

pullback은 외미분·쐐기곱·행렬 곱과 양립한다.
3차원에는 4-form이 없으므로 행렬값 연결에도 정확히

$$
F\wedge F=q^*(\mathcal F\wedge\mathcal F)=0,
\qquad \operatorname{tr}(F\wedge F)=0
$$

이다. 같은 세 좌표를 쓰는 서로 다른 Abelian 성분 사이에서도
$F^A\wedge F^B=0$이다. 비가환 대수의 차원이나 물질 표현 차원을 늘리는 것만으로
쐐기곱에 사용할 독립 좌표 1-form의 수가 늘어나지는 않는다.

Abelian의 점별 행렬로 보면 $F=J\mathcal FJ^{\mathsf T}$,
$J_\mu{}^a=\partial_\mu q^a$다. 3×3 반대칭 행렬의 rank는 최대 2이므로
$\operatorname{rank}F\leq2$다. 4×4 반대칭 행렬의 Pfaffian은

$$
\operatorname{Pf}(F)=F_{01}F_{23}-F_{02}F_{13}+F_{03}F_{12}=0.
$$

$E_i=F_{0i}$, $B=(F_{23},F_{31},F_{12})$로 부호를 정하면 이는
$\mathbf E\cdot\mathbf B=0$이다. 다른 전기장 부호 관례에서도 영이라는 결론은 같다.
계량이나 좌표계의 변경으로 이 4-form의 소멸을 바꿀 수 없다.

CE-GN3의 Abelian 곡률은 더 직접적으로 $\mathcal F=-dq^1\wedge dq^2$다.
이는 Euler potential로 쓰이는 퇴화 2-form과 같은 형태다.
[Gralla·Jacobson의 연구](https://arxiv.org/abs/1401.6159)는 이런 퇴화장과 Euler
potential을 force-free 전자기학에서 다룬다. 퇴화 조건만 만족한다고 force-free
운동방정식까지 유도된 것은 아니며, 일반 Maxwell 장과도 구분해야 한다.

## 47.3 표현할 수 없는 국소 Maxwell 해

평탄한 Minkowski 영역, 좌표 순서 $(t,x,y,z)$에서 공통 단위의 장

$$
F_*=dt\wedge dz+dx\wedge dy
$$

를 택한다. 계수는 모두 상수이므로
$\partial_{[\lambda}F_{\mu\nu]}=0$과 $\partial_\mu F^{\mu\nu}=0$이다.
예를 들어 $A_*=t\,dz+x\,dy$의 외미분으로 얻는다.
이 장은 $E_z=B_z=1$인 표준 원천 없는 국소 해이며

$$
F_*\wedge F_*=2\,dt\wedge dx\wedge dy\wedge dz\ne0,
\quad \operatorname{Pf}(F_*)=1,\quad\operatorname{rank}F_*=4.
$$

따라서 세 좌표 pullback으로 이 장을 같은 열린 영역에서 표현할 수 없다.
일정한 장을 무한 공간 전체의 유한 에너지 상태라고 가정한 것이 아니다.
표준 국소 Maxwell 극한을 검사하는 데 필요한 것은 위 영역에서의 매끄러운 해다.

이 제한은 차원 있는 계수나 $\ell$을 재피팅해도 바뀌지 않는다.
또한 매끄러운 pullback 장들이 점별로 $F_*$에 수렴한다는 주장도 Pfaffian의
연속성과 모순된다. 특이 사상으로 열린 영역 전체의 매끄러운 비퇴화장을 대신하는
것은 별도 정의·극한·보존 검사를 요구한다.

## 47.4 물질 표현이나 $F^2$ 작용만 추가하면 해결되는가

[CE-MR1](46_공통_수송의_물질_표현과_이상_상쇄.md)의 텐서 표현은
같은 연결의 표현 행렬을 바꾼다. 새 독립 좌표를 추가하지 않으므로
그것만으로 위의 rank 제한을 제거하지 못한다.
전자기 연결을 약한 부문과 혼합하더라도, Higgs 방향을 포함해 모든 성분이
같은 세 $q$에만 의존하는 pullback이면 같은 제한이 적용된다.
독립 Higgs 방향이나 별도 gauge장을 동적으로 공급한다면 다른 판본이므로
그 자유도와 운동방정식을 다시 검사해야 한다.

외부에서 Yang–Mills 또는 Maxwell 형태의 작용 $\int\operatorname{tr}F_{\mu\nu}F^{\mu\nu}$를
붙이거나, 물질 loop에서 그런 항을 얻었다고 해도 장의 정의역 제한은 그대로다.
특히 상수 배경 $q=q_0+\varepsilon\varphi(x)$에서는

$$
F_{\mu\nu}=\varepsilon^2\mathcal F_{ab}(q_0)
\partial_\mu\varphi^a\partial_\nu\varphi^b+O(\varepsilon^3),
\qquad \operatorname{tr}F^2=O(\varepsilon^4).
$$

이 항은 그 배경에서 독립 광자장의 quadratic 운동항을 제공하지 않는다.
CE-GN4의 scalar 프레임 운동항과도 다른 차수다.
변하는 배경에서는 전개의 차수가 달라질 수 있지만, 그 사실만으로 독립 벡터의
자유도·편광·Maxwell 해 공간을 복원했다고 할 수 없다.

이 판정은 현재 pullback을 물리적 고전 장 자체로 읽는 해석에 대한 것이다.
양자 평균이나 거친 평균으로 정의한 다른 집단 장에서는 비선형 항등식과 평균의
순서가 달라질 수 있다. 그런 평균 장의 운동학·요동·Maxwell 극한은 별도로
유도해야 하며 이번 후보에 이미 포함됐다고 가정하지 않는다.

## 47.5 다음 구성에서 필요한 최소 변경

반례를 제거하는 가장 작은 운동학적 대조는 독립 1-form 네 개를 허용하는 것이다.
네 상태 좌표 $r^0,r^1,r^2,r^3$에서

$$
\mathcal F_4=dr^0\wedge dr^3+dr^1\wedge dr^2,
\qquad r^\mu=x^\mu
$$

로 두면 위 $F_*$가 된다. 두 독립 Euler-potential 쌍을 허용한 것이다.
이는 **독립 상태 좌표 네 개가 적어도 이 rank-4 반례를 표현할 수 있다는 대조**이며,
그것으로 완전한 Maxwell·Yang–Mills·Einstein 이론을 얻었다는 뜻은 아니다.
기존 세 부문의 기하·질량·정규화·보완 동역학을 보존하는 확장이 필요하다.

사용자의 ‘세 각도에서 본 세 힘’은 세 가지 힘의 투영을 뜻할 수 있고,
전체 물리 상태를 실수 scalar 좌표 세 개로 제한해야 한다는 요구가 아니다.
이번 반례는 그 두 개념을 동일시한 특정 판본에 적용한다.
무한 상태공간이나 네 힘의 공통 기원이라는 최상위 목표를 기각하지 않는다.

## 47.6 검산과 판정

[검산 코드](../../verify/ce_three_coordinate_maxwell.py)와
[고정 산출](../../verify/ce_three_coordinate_maxwell.json)을 사용한다.

```powershell
python -B verify/ce_three_coordinate_maxwell.py
```

**[산출]** 고정한 네 Jacobian·세 표적 반대칭 행렬의 12개 조합에서,
유리수 소거법의 rank는 모두 2 이하였고 Pfaffian과 별도 Levi-Civita 수축은 정확히 0이었다.
$F_*$는 rank 4, Pfaffian 1, $\epsilon^{\mu\nu\rho\sigma}F_{\mu\nu}F_{\rho\sigma}=8$이다.
상수 계수 Maxwell·Bianchi 잔차는 0이다. 마지막 잔차 검사는 이미 상수인 계수의
미분 항등식을 옮긴 것이며 수치 PDE 솔버 검증으로 세지 않는다.

진폭을 1·2·3·4배로 바꾼 고정 pullback의 Euclidean $F$ 노름제곱은
12·192·972·3072로 정확히 진폭의 네제곱에 비례했다.
이는 Lorentzian 에너지나 관측 오차의 산출이 아니라 전개 차수의 검산이다.
무한 해 공간의 반례는 유한 사례 수가 아닌 §47.2–47.3의 외대수 증명에 근거한다.

**판정:** 고정 위상의 세 매끄러운 좌표 pullback만을 일반 전자기장으로 읽는 판본은
Maxwell 극한에서 기각한다. 기존 수송 대수·물질 표현·비관측 상태의 조건부 수학은
보존하되, 이 제한된 가족의 통일장 이론 해석으로 승격하지 않는다.
다음 후보는 필요한 독립 장 성분을 실제 작용에 포함하고 기존 성공 결과와 함께
검증해야 한다. 전체 목표와 공동 관측 검증은 계속 미완성이다.
