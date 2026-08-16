# CE 우주 밀도 사상 1단계 유도 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/cosmology-theory-repository-audit-20260815

## 1. 연구 질문

선행 감사에서 `[공리: 물리 사상]`으로 남은
$q_{\rm ext}\mapsto\Omega_b(t_0)$를 가장 작은 단위로 분해한다. 먼저
Poisson 고정점이 목표 관측값을 넣지 않은 국소 공변 작용의 안정한 균일
해로 구현될 수 있는지 증명한다. 다음으로 그 무차원 장을 실제 바리온
에너지분율과 동일시하려면 어떤 conserved current, stress tensor,
cosmological boundary condition이 추가로 필요한지 판정한다.

이번 run은 DM/DE 분할, primordial spectrum과 $H_0$를 유도하지 않는다.
이 첫 다리가 닫히지 않으면 후속 층으로 넘어가지 않는다.

## 2. 고정할 명제

- **B1 (변분 임베딩):** $D>1$, $x\in(0,1]$에서 고정점 방정식을
  Euler--Lagrange의 균일 정지조건으로 갖는 무차원 potential을 관측
  $\Omega_b$를 입력하지 않고 구성할 수 있다.
- **B2 (가지와 안정성):** 작은 근 $q_*$는 허용 영역의 안정한 국소
  최소점이고 $x=1$은 별도 불안정 정지점임을 Hessian과 경계로 증명한다.
- **B3 (공변 동역학):** metric 부호 $(-+++)$에서 dimension-four
  Lorentzian action, stress tensor와 평탄 FLRW 균일 방정식을 유도하고
  ghost·gradient·tachyon 조건과 attraction timescale을 적는다.
- **B4 (물리 사상):** $x=q_*$만으로
  $x=\rho_b/\rho_{\rm crit}$가 정리로 따라오는지 판정한다. 불가능하면
  최소 추가 공리와 covariant two-fluid/current 후보를 명시하며,
  존재구성과 자연 유도를 구분한다.
- **B5 (보존법칙):** 바리온 current 보존, 총 stress 보존과 표준
  $\rho_b\propto a^{-3}$를 동시에 요구할 때 상수 attractor가 현재
  $\Omega_b$를 고정할 수 있는지 검사한다.
- **B6 (승격 조건):** 자유 매개변수, 초기조건, freeze-out hypersurface,
  후보 선택과 관측 독립성을 회계해 `[정리]`, `[공리]`, `[산출]`,
  `[경험식]`, `[미완성]`, `[예측]`을 확정한다.

## 3. 출발 후보와 금지 사항

검산할 최소 scalar embedding은

$$
S_x=\int d^4x\sqrt{-g}\left[
-\frac{F^2}{2}g^{\mu\nu}\partial_\mu x\partial_\nu x
-M^4v_D(x)\right],
$$

$$
v_D(x)=x\log x-x+D\left(x-\frac{x^2}{2}\right)+C,
\qquad x\in(0,1].
$$

이는 조사할 **후보**이지 정본 공리가 아니다. $F,M$은 양의 scale이고
$C$는 운동방정식에 영향을 주지 않는 상수다.

다음 방식은 유도로 인정하지 않는다.

1. $x:=\Omega_b$를 정의한 뒤 고정점 수치 일치를 증명이라고 부르기.
2. potential 또는 coupling에 관측된 $\Omega_b$, $H_0$나 현재 cosmic
   time을 숨겨 넣기.
3. 여러 potential·readout을 본 뒤 가장 가까운 후보만 보고하기.
4. 총 stress 보존 또는 바리온 current를 쓰지 않고 density fraction이라
   부르기.
5. 고정점 위치만 맞고 ghost·gradient·경계·다른 가지를 검사하지 않기.

## 4. 정의역·차원·허용 오차

- 자연단위 $c=\hbar=1$, $[d^4x]=-4$, $[\partial]=1$,
  $[x]=[D]=0$, $[F]=[M]=1$을 쓴다.
- $\log x$와 고정점 지수의 인자는 무차원이어야 한다.
- $D>1$, $x\in(0,1]$이며 $x\to0^+$ 경계와 $x=1$ 가지를 별도 검사한다.
- 대수·정지점 잔차 허용오차는 $10^{-12}$, ODE/적분은 독립 수렴 비교를
  기록한다.
- 관측량을 사용한다면 release·오차·공분산·접근일을 고정하지만, 이번
  단계에서는 관측 중심값을 action 구성에 사용하지 않는다.

## 5. 통과 기준

B1--B3은 완전 유도, 차원 검사와 독립 수치 검산이 모두 닫혀야 한다.
B4--B5는 정리로 닫히거나, 닫히지 않는 정확한 no-go와 최소 추가 공리를
제시해야 한다. 단순 action engineering은 “가능성의 존재 증명”으로만
인정하며 $\Omega_b$ 자연 유도로 승격하지 않는다. `[예측]` 승격은 물리
사상, 초기조건·freeze-out과 blind 비교 절차가 추가 선택 없이 고정될 때만
허용한다.

## 6. 산출물

- `10-sources.md`: 공변 relativistic fluid/current, reacting mixture와
  cosmological density fraction에 관한 1차 이론 출처
- `11-math.md`: B1--B6 유도·no-go·무차원·안정성 검산
- `12-routes.md`: action/current 연결 후보, 자유도와 kill test
- `20-audit.md`: 형식 지위와 다음 단계 진입 여부
- `30-implementation.md`, `31-validation.md`: gate 승인 범위의 최소 코드와
  회귀 결과 또는 SKIPPED 사유
- `40-final-report.md`: 첫 다리의 자기완결적 결론과 후속 계약
