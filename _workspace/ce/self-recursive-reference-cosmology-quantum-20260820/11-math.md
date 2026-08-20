# 무한 자기재귀 참조함수 수학 감사

Status: COMPLETE  
Date: 2026-08-20

## 1. 공통 타입

`자기참조`라는 말 하나로 방정식과 동역학을 합치지 않는다. 집합 \(X\),
사상 \(T:X\to X\), 초기상태 \(x_0\in X\)를 명시하고

\[
\operatorname{Fix}(T):=\{x\in X:T(x)=x\},\qquad
x_n:=T^{\circ n}(x_0)
\]

를 서로 다른 대상으로 정의한다. 고정점은 반복 없이도 정의할 수 있다.
반면 무한 반복 readout

\[
\mathfrak R_T(x_0):=\lim_{n\to\infty}T^{\circ n}(x_0)
\]

은 극한이 존재할 때에만 정의한다. 이는 **[정의 규약]**이지 모든
자기참조 구조에 관한 보편 정리가 아니다. 극한이 없으면 고정점, 주기
궤도, \(\omega\)-limit set 또는 Cesàro 평균을 구분한다.

## 2. 스칼라 Poisson 반복

### 2.1 정의역과 무차원성

\[
F_D(x)=\exp[-D(1-x)]
\]

에서 (D\)는 무차원 Poisson 평균이고 (x,q\)는 무차원 확률이다. 따라서
지수 인자 (D(1-x))는 무차원이다. (x\in[0,1])이면

\[
F_D([0,1])=[e^{-D},1]\subset[0,1].
\]

다만 ([0,1])은 함수가 존재하는 유일한 영역이 아니라 **소멸확률 해석의
영역**이다. 확률생성함수는 더 넓은 실수/복소 영역에서 논할 수 있다.

### 2.2 해, 선택 규칙과 basin

단일 초기 개체의 세대 (n) 이내 소멸확률은

\[
q_0=0,\qquad q_{n+1}=F_D(q_n)
\]

이다. 단조극한은 (F_D)의 최소 고정점을 고른다.

- (0\le D\le1): ([0,1])의 고정점은 (1)뿐이다.
- (D>1): (q_{\rm ext}\in(0,1/D))와 (1)이 고정점이다.
- 고정점 multiplier는 (F_D'(q)=Dq)다.
- (D>1)에서 (q_{\rm ext})는 안정, (1)은 불안정이다.
- 더 강하게 모든 (x_0\in[0,1))은 (q_{\rm ext})로 수렴하지만,
  (x_0=1)은 영원히 (1)에 머문다.

따라서 `유일한 자기일관해`라는 무표시 표현은 거짓이다. 허용 표현은
`[0,1/D]의 유일한 고정점`, 또는 `q_0=0에서 선택되는 최소 소멸
고정점`이다.

benchmark (D=3.1777584234)에서는

\[
q_{\rm ext}=0.0486467196445741,qquad
F_D'(q_{\rm ext})=Dq_{\rm ext}=0.1545875231
\]

이고 residual은 약 (-1.04\times10^{-16})이다. 초기 반복은

\[
0, 0.041678977, 0.047581431, 0.048482318,
0.048621312, 0.048642792,\ldots
\]

다. 이 수치는 표시된 rounded (D)를 사용한 산출이며 full-precision CE
registry 판본과 혼합하지 않는다.

### 2.3 P0 초기값 반례

(D>1, x_0=1)이면 모든 (n)에 대해

\[
F_D^{\circ n}(1)=1.
\]

그러므로 `모든 초기값에서 최소근으로 수렴`이라는 보편 부모 주장은
완전 반례로 제거된다.

## 3. 다형 Poisson 반복

유한차원 비음수 평균행렬 \(A=(A_{ij})\)에 대해

\[
G_i(\boldsymbol q)
=\exp\!\left[-\sum_jA_{ij}(1-q_j)\right],
\qquad
J_G(\boldsymbol q)=\operatorname{diag}(\boldsymbol q)A
\]

다. \(\boldsymbol q^{(0)}=\boldsymbol0\) 반복이 최소 소멸벡터를 고른다.
\(A\)가 irreducible이면
\(\rho(A)\le1\)에서 \(\boldsymbol q=\boldsymbol1\)이고,
\(\rho(A)>1\)에서 모든 성분이 1보다 작은 최소 소멸벡터가 존재한다.
reducible인 경우에는 성분별 임계값과 basin을 별도로 분석해야 한다.
한 고정점의 국소 안정은

\[
\rho\!\left(\operatorname{diag}(\boldsymbol q)A\right)<1
\]

로 검사한다. 공통 행합 (D)와 균일 초기값이 있을 때에만 scalar
부분공간으로 정확히 축약된다.

반례 (A=\operatorname{diag}(2,0.5))에서는

\[
\boldsymbol q=(0.20318786998,1),\qquad
J_G(\boldsymbol q)=\operatorname{diag}(0.40637573996,0.5).
\]

따라서 임의의 reducible (A)에 scalar 유일성·단일 basin을 이식할 수 없다.

## 4. 양자채널 반복

유한차원 Hilbert 공간의 density operator 집합을
\(\mathsf D(\mathcal H)\)라 하자. 양자 반복은 선형 CPTP 사상

\[
\rho_{n+1}=\mathcal E(\rho_n),\qquad
\rho_n=\mathcal E^n(\rho_0)
\]

이다. 각 \(\mathcal E^n\)도 CPTP다. 이는 비선형 scalar 사상 (F_D)와
타입부터 다르다.

모든 초기상태에서 \(\mathcal E^n(\rho)\)가 수렴하려면 1 이외의 주변
고유값이 없어야 하며, \(\lambda=1\)의 Jordan block은 semisimple이어야
한다. 극한이 존재해도 고정상태가 여러 개면 극한은 초기상태에 의존할 수
있다. 모든 초기 density state가 같은 상태로 수렴한다고 주장하려면 고정
density state 집합이 singleton이어야 한다. primitive channel은
충분조건이지 모든 CPTP channel의 성질이 아니다.

### 4.1 P0 주기 반례

qubit의 Pauli (X)에 대한 unitary channel

\[
\mathcal U_X(\rho)=X\rho X
\]

는 CPTP지만

\[
|0\rangle\!\langle0|
\longleftrightarrow
|1\rangle\!\langle1|
\]

의 2-cycle을 만든다. 주변 고유값 (-1) 때문에 점별 극한이 없다.

### 4.2 P0 비유일 반례

완전 dephasing channel

\[
\Delta(\rho)=\sum_{i=0}^1|i\rangle\!\langle i|\rho
|i\rangle\!\langle i|
\]

에서는 모든 diagonal density matrix가 고정점이다. CPTP와 즉시 수렴이
성립해도 고정점은 유일하지 않다.

## 5. quantum-to-branching 게이트

CP reduced dynamics와 population closure만으로 offspring genealogy가
생기지 않는다. diagonal invariant algebra를 얻어도 보통의 stochastic
transition matrix를 얻었을 뿐이다. 다음이 추가로 필요하다.

1. 실제 기록 outcome을 갖는 quantum instrument 또는 지정 unraveling
2. Markov jump rate와 coarse-graining time
3. reproduction count를 정의하는 확률공간
4. 세대·계통의 조건부 독립성
5. 그 자료에서 비음수 평균행렬 (A_{ij})의 식별

unraveling은 일반적으로 유일하지 않으므로 같은 master equation이 하나의
genealogy를 자동 선택하지 않는다. 이 bridge는 **[미완성]**이다.

## 6. 우주론 fixed point

우주론 상태를 (\boldsymbol y), 물리 생성자를 (\boldsymbol G)라 하면

\[
\frac{d\boldsymbol y}{dt}=\boldsymbol G(\boldsymbol y,t),
\qquad
\boldsymbol y(t_2)=\Phi_{t_2,t_1}(\boldsymbol y(t_1))
\]

처럼 시간과 flow를 먼저 정의해야 한다. autonomous 경우 fixed point는
\(\boldsymbol G(\boldsymbol y_*)=0\)이고, forward-time attraction은
\(D\boldsymbol G(\boldsymbol y_*)\) 고유값의 실수부로 판정한다.
\(N=\log a\)를 시간으로 쓰면

\[
N=\log a,\qquad
\frac{d\boldsymbol y}{dN}
=\frac{\boldsymbol G(\boldsymbol y)}{H(\boldsymbol y)},\qquad H>0
\]

를 명시한다. 두 flow의 fixed-point 조건은 같지만 안정성 고유값에는
\(H\) 재척도와 시간 방향이 반영된다.

차원은

\[
[\boldsymbol G]=[\boldsymbol y]/T,qquad [t\boldsymbol G]=[\boldsymbol y]
\]

이어야 한다. 양자 semigroup \(e^{t\mathcal L}\)에서도
\([\mathcal L]=T^{-1}\)다. FLRW의 (H)는 (T^{-1}),
\(\rho/(3M_{\rm Pl}^2H^2)\)는 무차원이다.

(F_D)의 iteration index에는 시간 단위, stress tensor, Friedmann 제약이
없다. 그러므로 그 반복 횟수는 우주 시간·상전이·인플레이션이 아니다.

### 6.1 P0 동역학 비유일성 반례

\[
\dot x=-x,qquad \dot x=-x^3
\]

는 고정점 집합 \(\{0\}\)을 공유하지만 전자는 지수적으로, 후자는
대수적으로 접근한다. 대응 potential (x^2/2)와 (x^4/4)도 다르다.
따라서 stationary set만으로 물리 동역학이나 작용을 고를 수 없다.

## 7. 확률-to-density P0 반례

(q=0.0486467196445741)에 대해 dimensionless 사상

\[
\Omega_b=cq
\]

를 생각하면 (c=1)과 (c=2) 모두 ([0,1]) 값을 만들지만 각각

\[
\Omega_b=0.0486467\ldots,qquad
\Omega_b=0.0972934\ldots
\]

를 준다. 양쪽 변수가 무차원이라는 사실은 사상을 유도하지 않는다.
current, energy weight, total yield, critical-density normalization 없이
(q\mapsto\Omega_b)는 고정점 정리가 될 수 없다.

## 8. 계약 판정

| 항목 | 판정 | 보정 |
|---|---|---|
| SR-1 | PASS-WITH-NARROWING | 보편 정리가 아니라 CE 정의 규약 |
| SR-2 | PASS-WITH-NARROWING | ([0,1])은 확률 해석 영역이지 함수의 유일 정의역이 아님 |
| SR-3 | PASS | 초기값 1 반례와 최소근 selection 명시 |
| SR-4 | PASS | CPTP 선형 사상과 Poisson 비선형 사상 타입 분리 |
| SR-5 | PASS | unitary/dephasing 반례로 수렴·유일성 분리 |
| SR-6 | REVISE | instrument/unraveling과 genealogy 확률공간을 의무에 추가 |
| SR-7 | PASS | physical flow, timebase, constraint 필요 |
| SR-8 | PASS | energy-weight 반례로 readout 독립성 확인 |

열린 수학 P0는 없다. 그러나 SR-6의 물리 bridge와 우주론 readout은 계속
`[미완성]`이다. 이는 식을 약화시키는 것이 아니라 수학 코어와 물리 사상을
정확히 분리한 결과다.
