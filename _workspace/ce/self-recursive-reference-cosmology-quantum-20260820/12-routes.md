# 자기재귀 식 정비 경로

Status: COMPLETE  
Date: 2026-08-20

## 1. 채택할 공통 코어

공통 코어는 특정 지수함수가 아니라 typed iteration이다.

\[
(X,T,x_0),\qquad
\operatorname{Fix}(T),\qquad
\operatorname{Orb}_T(x_0)=\{T^{\circ n}(x_0)\}_{n\ge0}.
\]

점별 극한이 존재할 때만

\[
\mathfrak R_T(x_0)=\lim_{n\to\infty}T^{\circ n}(x_0)
\]

를 `무한 재귀 readout`으로 정의한다. 이 정의는 세 물리계를 같은 함수로
만들지 않고, 각 계에서 무엇을 추가해야 하는지 드러낸다.

## 2. Route P: Poisson 분지확률

\[
X=[0,1],\quad T=F_D,\quad x_0=0
\]

를 사용한다. 이 route에서만

\[
\mathfrak R_{F_D}(0)=q_{\rm ext}=\min\operatorname{Fix}_{[0,1]}F_D
\]

가 성립한다. `무한 self-reference`보다 `세대별 소멸확률의 생성함수
반복`이 정확한 이름이다.

필수 표기:

- (D)의 Poisson 평균 의미와 무차원성
- (q_0=0)
- (D>1)의 두 고정점
- (x_0=1) 예외 basin
- multitype이면 (A\), 최소벡터와 Jacobian

## 3. Route Q: 양자 CPTP 반복

\[
X=\mathsf D(\mathcal H),\quad T=\mathcal E
\]

를 사용한다. 고정점은 \(\rho=\mathcal E(\rho)\)이고, 수렴은 주변
스펙트럼과 fixed-space dimension으로 판정한다. Poisson (F_D)를 density
matrix에 직접 적용하지 않는다.

### 3.1 허용되는 classical bridge

다음 사슬을 실제로 구성했을 때만 Route P로 내려갈 수 있다.

\[
\text{CPTP reduced dynamics}
\to \text{instrument records}
\to \text{Markov counting process}
\to \text{offspring genealogy}
\to A_{ij}\ge0.
\]

이때 (F_A)는 양자 진폭 자체가 아니라 **기록된 classical genealogy의
확률생성함수**다.

### 3.2 제외 경로

상태 의존 정규화로

\[
\rho\mapsto
\frac{e^{-C(\rho)}\rho}{\operatorname{Tr}[e^{-C(\rho)}\rho]}
\]

같은 비선형 map을 만든 뒤 quantum channel이라고 부르는 경로는 제외한다.
일반적으로 선형 CPTP가 아니며 Born/instrument 구조를 보존하지 않는다.

## 4. Route C: 우주론 flow

우주론은 discrete (F_D)가 아니라

\[
\dot{\boldsymbol y}=\boldsymbol G(\boldsymbol y;	heta),
\quad \boldsymbol C(\boldsymbol y)=0
\]

를 출발점으로 둔다. 여기서 \(\boldsymbol C=0\)은 Friedmann 등의
제약면이다. fixed point와 attractor는

\[
\boldsymbol G(\boldsymbol y_*)=0,qquad
\max\operatorname{Re}\operatorname{spec}
D\boldsymbol G(\boldsymbol y_*)<0
\]

로 판정한다. 이 식은 지정한 forward time convention에서만 유효하다.

Poisson (q_{\rm ext})를 우주론에 쓰려면 다음 둘 중 하나가 필요하다.

1. action/current/counting process가 Route P를 실제 물리 subsystem으로
   만들고, freeze surface에서 그 조성을 읽는 경로
2. (q_{\rm ext})를 독립 boundary/model axiom으로 넣고 관측 예측이 아닌
   조건부 산출로 유지하는 경로

## 5. 정본 최소 수정안

1. `14_자기재귀성_대칭.md` 앞부분에 typed iteration, orbit, fixed point,
   conditional limit를 추가한다.
2. 같은 문서에 quantum CPTP 반복과 unitary/dephasing 반례를 짧게
   추가한다.
3. `9_우주론_수식_의미와_후보.md`의 `자기재귀 고정점`을
   `Poisson 세대 재귀와 우주론 readout 경계`로 좁히고 (q_0=0),
   (q=1), (q_{\rm ext}<1/D)를 명시한다.
4. `00_선택과_접힘.md`의 양자-to-branching 미완성 목록에
   instrument/unravelling과 genealogy probability space를 추가한다.
5. 기존 (q_{\rm ext}\mapsto\Omega_b)는 legacy axiom 표지를 보존하며
   정리로 승격하지 않는다.

## 6. Kill tests

- 초기값을 지우면 Route P 설명은 실패한다.
- `CPTP이므로 수렴`이라고 쓰면 unitary 2-cycle에 의해 실패한다.
- `CPTP 고정점은 유일`이라고 쓰면 dephasing fixed simplex에 의해 실패한다.
- 계산 iteration을 cosmic time으로 쓰면 timebase/stress 부재로 실패한다.
- (q)와 \(\Omega\)가 모두 무차원이라는 이유만으로 동일시하면
  \(\Omega=cq\) family 반례로 실패한다.
- multitype (A)를 scalar (D)로 축약하면서 공통 행합·균일 부분공간을
  쓰지 않으면 실패한다.

## 7. 구현 순서

형식 감사가 `PASS/REVISE-CLOSED`를 주면 위 5개 중 1--4만 최소 변경한다.
식별 공리나 수치 상수를 바꾸지 않는다. 이후 scalar basin, quantum
counterexample, dimensionless arguments를 하나의 focused verification으로
검사한다.
