# 행동결합형 국소 인과 세계 시뮬레이터

> 상태: `Mathematical toy model + synthetic engineering gate`
> 구현: `reality_stone/python/reality_stone/clarus/causal_world_simulator.py`
> 실행: `examples/agi/causal_world_simulator_gate.py`
> 테스트: `tests/test_causal_world_simulator.py`

---

## 0. 주장 경계

이 문서는 인간 뇌를 다음 계산 원리의 후보로 본다.

\[
\boxed{
\text{부분관측으로 세계상태를 추론하고,}
\quad
\text{행동별 미래를 반사실적으로 실행하며,}
\quad
\text{오차로 내부 법칙을 수정하는 시스템}
}
\]

이를 `행동결합형 국소 인과 세계 시뮬레이터`라 부른다. 여기서 세계는
우주의 모든 미시상태가 아니라 몸과 행동에 필요한 압축된 인과상태다.

이 문서에서 증명하는 것은 선언한 유한·선형 모형 안의 정리다. 합성 데이터
통과는 다음을 증명하지 않는다.

1. 인간 뇌가 실제로 같은 행렬을 사용한다.
2. 예측오차 최소화 하나로 의식이나 지능이 모두 설명된다.
3. 뇌 주름이 현재 기능공간의 직접적인 3차원 그림이다.
4. 합성 개입 예측이 생물학적 인과성의 증거다.
5. 이 모듈이 AGI이거나 P8 mammalian stage를 통과했다.

---

## 1. 최소 구조

잠재 세계상태, 행동, 감각을 각각

\[
z_t\in\mathbb R^d,
\qquad
a_t\in\mathbb R^p,
\qquad
y_t\in\mathbb R^m
\]

로 둔다. 최소 controlled structural equation은

\[
z_{t+1}=Az_t+Ba_t+w_t,
\]

\[
y_t=Cz_t+v_t
\]

이다. (w_t,v_t)는 각각 process noise와 observation noise다.

뇌의 감각계에 대응시키면 (C)는 하나의 균질 센서가 아니라 여러 국소
chart를 쌓은 행렬이다.

\[
C=
\begin{bmatrix}
C_{\rm visual}\\
C_{\rm auditory}\\
C_{\rm body}\\
C_{\rm action-copy}
\end{bmatrix}.
\]

각 chart는 세계 전체를 보지 못해도, 합친 atlas는 행동에 필요한 상태를
식별할 수 있다. 이것이 공통 감각중추를 anchor로 쓰는 가장 작은 수학적
의미다.

---

## 2. 부분감각에서 전역상태를 복원하는 조건

### 정리 1. 감각 atlas의 정확 복원 필요충분조건

무잡음 관측 (y=Cz)에서 모든 (z\in\mathbb R^d)를 유일하게 복원할 수
있을 필요충분조건은

\[
\boxed{\operatorname{rank}C=d}
\]

이다. 이때 최소제곱 복원은

\[
\hat z=C^\dagger y=z
\]

이다.

**증명.** (operatorname{rank}C=d)이면 (C)는 full column rank이고
(C^\dagger C=I_d)다. 따라서 (C^\dagger y=C^\dagger Cz=z)다. 반대로
(operatorname{rank}C<d)이면 (0\ne h\in\ker C)가 존재한다. (z)와
(z+h)는 (Cz=C(z+h))라는 같은 관측을 만들므로 유일 복원은 불가능하다.
(\square)

이 정리는 한 감각이 불완전하더라도 여러 감각의 null space가 서로 다르면
전체 상태가 식별될 수 있음을 보여준다.

\[
\bigcap_k\ker C_k=\{0\}
\quad\Longleftrightarrow\quad
\operatorname{rank}
\begin{bmatrix}C_1\\\vdots\\C_K\end{bmatrix}
=d.
\]

### 따름정리 1.1. 관측잡음 안정성

(y=Cz+v), (operatorname{rank}C=d)이면

\[
\hat z-z=C^\dagger v
\]

이므로

\[
\boxed{
\lVert\hat z-z\rVert
\le
\lVert C^\dagger\rVert\,\lVert v\rVert
}
\]

이다. 최소 singular value가 작으면 atlas는 이론적으로 식별 가능해도 잡음을
크게 증폭한다. 따라서 rank만이 아니라 condition number를 보고해야 한다.

### 반례 1.2. 단일 chart의 비식별성

\(C_{\rm visual}\)의 rank가 2이고 \(d=4\)이면 두 개의 잠재방향은 보이지
않는다. 이 방향으로 다른 두 세계상태는 모든 시각관측이 같으므로 시각만으로
구별할 수 없다. 구현은 이 반례를 ablation으로 등록한다.

---

## 3. 공통 감각 anchor가 내부 좌표계를 만드는 법

피질 또는 계산 graph의 Laplacian을 (L\succeq0)라 하고, 공통 감각 anchor
집합을 (A), 나머지 node를 (U)라 하자. anchor 값 (f_A)를 고정하고

\[
(Lf)_U=0,
\qquad
f_A=\text{fixed}
\]

를 푼다.

### 정리 2. anchor harmonic extension의 존재와 유일성

graph의 모든 연결성분이 적어도 하나의 anchor를 포함하면 (L_{UU})는
positive definite이고

\[
\boxed{
f_U=-L_{UU}^{-1}L_{UA}f_A
}
\]

가 유일한 harmonic extension이다.

**증명.** graph Laplacian의 이차형식은

\[
x^{\mathsf T}Lx
=\frac12\sum_{i,j}w_{ij}(x_i-x_j)^2.
\]

(x_A=0)인 벡터에서 이 값이 0이면 연결된 모든 node가 같은 값을 가져야
한다. 각 연결성분에 값 0인 anchor가 있으므로 (x_U=0)이다. 따라서 제한
이차형식 (x_U^{\mathsf T}L_{UU}x_U)는 (x_U\ne0)에서 양수이고
(L_{UU}\succ0)다. block equation을 풀면 표시한 유일해를 얻는다.
(\square)

### 정리 3. harmonic extension의 최소 에너지 성질

정리 2의 (f^*)는 같은 anchor 경계값을 가진 모든 (f) 중 Dirichlet
energy

\[
\mathcal D(f)=\frac12f^{\mathsf T}Lf
\]

를 유일하게 최소화한다.

**증명.** 임의의 허용 perturbation (h_A=0)에 대해

\[
\mathcal D(f^*+h)
=\mathcal D(f^*)
+h^{\mathsf T}Lf^*
+\frac12h^{\mathsf T}Lh.
\]

((Lf^*)_U=0), (h_A=0)이므로 교차항은 0이다. 정리 2에 의해 마지막 항은
(h\ne0)에서 양수다. (\square)

이 결과는 시각·청각·몸·운동 anchor 사이를 임의로 채우지 않고, 등록된
연결기하에서 가장 매끄러운 공통 좌표를 정의한다.

---

## 4. 세계의 전이법칙을 언제 정확히 배울 수 있는가

자료행렬을

\[
D=
\begin{bmatrix}
z_0^{\mathsf T}&a_0^{\mathsf T}\\
\vdots&\vdots\\
z_{T-1}^{\mathsf T}&a_{T-1}^{\mathsf T}
\end{bmatrix},
\qquad
Y=
\begin{bmatrix}
z_1^{\mathsf T}\\
\vdots\\
z_T^{\mathsf T}
\end{bmatrix}
\]

로 둔다. (Theta=[A\;B])이면 무잡음 자료는 (Y=D\Theta^{\mathsf T})다.

### 정리 4. persistent excitation 아래 정확 식별

무잡음이고

\[
\operatorname{rank}D=d+p
\]

이면 최소제곱 추정량

\[
\hat\Theta^{\mathsf T}=D^\dagger Y
\]

은

\[
\boxed{\hat A=A,\qquad\hat B=B}
\]

를 만족한다.

**증명.** full column rank이므로 (D^\dagger D=I_{d+p})다. 따라서

\[
D^\dagger Y
=D^\dagger D\Theta^{\mathsf T}
=\Theta^{\mathsf T}.
\]

(\square)

### 정리 5. 개입 다양성이 없을 때의 인과 비식별성

(operatorname{rank}D<d+p)이면 일반적으로 ((A,B))는 유일하게 식별되지
않는다.

**증명.** (0\ne h\in\ker D)를 택하고 임의의 출력방향 (q\in\mathbb R^d)에
대해 (Delta\Theta^{\mathsf T}=h q^{\mathsf T})로 둔다. 그러면
(D\Delta\Theta^{\mathsf T}=0)이므로 (Theta)와
(Theta+\Delta\Theta)는 같은 학습자료를 만든다. (\square)

따라서 수동 관측만으로 action effect를 인과적으로 해석할 수 없다. 행동이
충분히 다양한 방향을 자극하거나 별도 개입자료가 있어야 한다.

---

## 5. 여러 국소 시뮬레이터가 하나의 세계가 되는 조건

국소 chart (i)의 좌표를 (x_i=Q_i z)라 하자. (Q_i)가 invertible이면
chart 전이는

\[
T_{ij}=Q_jQ_i^{-1}
\]

다.

### 정리 6. cocycle과 무frustration holonomy

전역상태 (z)에서 유도된 chart 전이는

\[
T_{jk}T_{ij}=T_{ik}
\]

를 만족하고 모든 닫힌 cycle에서

\[
\boxed{
\mathcal H_\gamma
=T_{i_0i_n}\cdots T_{i_1i_2}T_{i_0i_1}
=I
}
\]

이다.

**증명.** 중간항이 소거되어

\[
T_{jk}T_{ij}
=Q_kQ_j^{-1}Q_jQ_i^{-1}
=Q_kQ_i^{-1}
=T_{ik}
\]

이다. 닫힌 cycle에서는 시작 chart와 마지막 chart가 같아 전체 곱이
(Q_{i_0}Q_{i_0}^{-1}=I)다. (\square)

### 정리 7. cycle 일관성에서 전역 atlas 구성

연결된 chart graph의 모든 (T_{ij})가 invertible이고 모든 cycle
holonomy가 (I)라 하자. 그러면 기준 chart (r)을 택해 각 (Q_i)를
(r\to i) 경로의 전이곱으로 정의할 수 있으며, 경로와 무관하고

\[
T_{ij}=Q_jQ_i^{-1}
\]

가 성립한다.

**증명.** 두 (r\to i) 경로의 전이곱이 다르면 한 경로와 다른 경로의
역경로를 이어 닫힌 cycle을 만들 수 있다. 그 cycle holonomy가 (I)이므로
두 곱은 같다. 따라서 (Q_i)가 well-defined다. edge ((i,j))에 대해
(r\to i\to j) 경로와 (r\to j) 경로가 같은 (Q_j)를 주므로
(Q_j=T_{ij}Q_i), 즉 결론이 따른다. (\square)

관측 전이의 불일치를

\[
F_\gamma
=\frac12
\lVert\mathcal H_\gamma-I\rVert_F^2
\]

로 둔다. (F_\gamma>0)이면 그 cycle의 국소 시뮬레이터들을 하나의 정확한
전역좌표로 동시에 붙일 수 없다. 이 값은 현재 gear frustration의 matrix
transition 버전이다.

---

## 6. 지각은 감각으로 제약된 상태 시뮬레이션

한 시점의 observation energy를

\[
E_y(z)
=\frac12
\lVert Cz-y\rVert_{R^{-1}}^2,
\qquad R\succ0
\]

로 둔다. 상태추론을

\[
\dot z=-G^{-1}\nabla E_y(z),
\qquad G\succ0
\]

로 수행한다.

### 정리 8. 감각제약 추론의 에너지 감소

모든 고전해에서

\[
\boxed{
\frac{dE_y}{dt}
=-
\nabla E_y^{\mathsf T}G^{-1}\nabla E_y
\le0
}
\]

이다.

**증명.** 연쇄법칙에 추론식을 대입한다. (G^{-1}\succ0)이므로 우변은
비양수다. (\square)

(C)가 full column rank이면 (E_y)는 strictly convex이고 유일 최소해는
가중 최소제곱 해다. rank가 부족하면 감각으로 보이지 않는 null direction은
prediction prior나 다른 감각 chart가 채워야 한다.

### 정리 9. 수축 observer의 ISS bound

선형 observer 오차가

\[
e_{t+1}=(A-LC)e_t+\xi_t
\]

이고 어떤 norm에서

\[
\lVert A-LC\rVert\le q<1,
\qquad
\lVert\xi_t\rVert\le\bar\xi
\]

이면

\[
\boxed{
\lVert e_t\rVert
\le
q^t\lVert e_0\rVert
+\frac{1-q^t}{1-q}\bar\xi
}
\]

이다.

**증명.** variation-of-constants 전개에 operator norm 상계와 등비급수를
적용한다. (\square)

---

## 7. 반사실과 행동선택

### 정리 10. 선형 structural model의 한 단계 개입효과

같은 현재상태 (z_t)에서 행동 (a)와 (a')를 각각 강제로 넣은 두
반사실 세계의 차이는

\[
\boxed{
z_{t+1}^{\operatorname{do}(a)}
-z_{t+1}^{\operatorname{do}(a')}
=B(a-a')
}
\]

이다.

**증명.** 두 structural equation
(Az_t+Ba+w_t), (Az_t+Ba'+w_t)에서 같은 exogenous noise를 고정하고
빼면 된다. (\square)

이 식은 (B)가 관측상관이 아니라 개입에 불변인 structural coefficient일
때만 인과 의미를 갖는다.

### 정리 11. 한 단계 quadratic counterfactual planner

(Q\succeq0), (R\succ0)이고

\[
J(a)
=(Az+Ba-z_*)^{\mathsf T}Q(Az+Ba-z_*)
+a^{\mathsf T}Ra
\]

라 하자. 그러면 유일한 최적행동은

\[
\boxed{
a^*
=-
(B^{\mathsf T}QB+R)^{-1}
B^{\mathsf T}Q(Az-z_*)
}
\]

이고 모든 (a)에 대해 (J(a^*)\le J(a))다. 특히 무행동 (a=0)보다
나쁠 수 없다.

**증명.** Hessian은

\[
\nabla_a^2J=2(B^{\mathsf T}QB+R)\succ0
\]

이므로 (J)는 strictly convex다. gradient를 0으로 놓으면 표시한 해를
얻고, strictly convex 함수의 stationary point는 유일 전역최소다.
(\square)

### 정리 12. rollout error의 기하급수 상계

참 전이 (F)가 state에 대해 Lipschitz constant (ho\ge0)를 갖고 모델의
한 단계 defect가 모든 관심영역에서

\[
\lVert\hat F(z,a)-F(z,a)\rVert\le\delta
\]

라 하자. 같은 행동열로 rollout할 때 초기오차가 (e_0)이면

\[
\boxed{
e_t
\le
\rho^t e_0
+\delta\sum_{k=0}^{t-1}\rho^k
}
\]

이다. (ho<1)이면 장기오차는 (delta/(1-ho)) 이하로 제한된다.

**증명.** 삼각부등식과 Lipschitz 조건으로

\[
e_{t+1}
\le
\rho e_t+\delta
\]

를 얻고 귀납적으로 전개한다. (\square)

따라서 one-step score가 높다는 사실만으로 장기 시뮬레이터를 주장할 수
없다. (ho\ge1)이면 작은 model defect도 horizon과 함께 증폭될 수 있다.

---

## 8. 전체 알고리즘

```text
입력: 부분감각 y_t, 이전 상태 z_{t-1}, 목표 z_*, 후보 행동 집합

1. 여러 감각 chart를 결합해 현재 상태 z_t를 추론한다.
2. chart cycle holonomy를 계산해 내부 세계의 모순을 찾는다.
3. 각 행동 후보를 transition model 안에서 rollout한다.
4. 목표비용, 위험, 불확실성으로 반사실 미래를 평가한다.
5. 최적 행동을 실제 환경에 적용한다.
6. 새 감각과 예측의 residual을 계산한다.
7. 상태를 빠르게, transition과 chart를 느리게 갱신한다.
8. holdout 성능과 개입효과가 악화되면 구조 갱신을 되돌린다.
```

두 시간척도는

\[
\eta_z\gg\eta_{A,B,C}
\]

로 둔다. 빠른 변수 (z)는 현재 세계를 추론하고, 느린 변수 (A,B,C)는
세계의 법칙과 감각좌표 자체를 학습한다.

---

## 9. 등록된 합성 데이터 게이트

### 9.1 자료 생성

현재 구현은 (d=4), (p=2)인 안정 controlled linear world를 사용한다.

\[
z_{t+1}=Az_t+Ba_t+w_t,
\qquad
y_t=Cz_t+v_t.
\]

- 총 action step: 1,600
- train: 처음 1,000 step
- test: 이후 600 step
- action: seed가 고정된 독립 uniform excitation
- process/observation noise 표준편차: 0.002
- visual chart: rank 2
- visual+body stacked atlas: rank 4
- test 자료는 fit과 hyperparameter 선택에 사용하지 않음

이 데이터는 프로그램이 생성해
`artifacts/agi/causal_world_data.csv`에 기록한다.

### 9.2 사전등록 판정

1. stacked atlas rank (=4)
2. 전체 atlas reconstruction RMSE가 visual-only의 10% 미만
3. held-out next-state (R^2>0.99)
4. persistence 대비 (R^2) 증가 (>0.1)
5. 등록 action contrast의 counterfactual effect error (<0.01)
6. learned planner의 실제 one-step cost가 zero/random action보다 낮음
7. harmonic residual (<10^{-12})
8. exact chart cycle frustration (<10^{-24})
9. 한 transition을 오염하면 frustration (>10^{-5})

### 9.3 실행 결과

seed `20260810`의 실제 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| stacked observation rank | 4 |
| visual-only rank | 2 |
| 전체 atlas reconstruction RMSE | 0.00201624 |
| visual-only reconstruction RMSE | 0.13639672 |
| (\lVert\hat A-A\rVert_F) | 0.00296120 |
| (\lVert\hat B-B\rVert_F) | 0.00041833 |
| test model (R^2) | 0.99985778 |
| persistence (R^2) | 0.64422653 |
| counterfactual effect error | 0.00034964 |
| learned planner cost | 0.07712060 |
| zero-action cost | 0.14115489 |
| random-action cost | 0.24148138 |
| harmonic residual | (1.11\times10^{-16}) |
| exact holonomy frustration | (1.54\times10^{-33}) |
| corrupted holonomy frustration | 0.007488 |
| 종합 판정 | `PASS` |

### 9.4 이 수치가 닫는 범위

이 결과는 다음만 확인한다.

- 정리 1의 full-rank sensor fusion이 수치적으로 작동한다.
- 단일 rank-deficient chart가 정보를 잃는 ablation이 재현된다.
- persistently excited train 자료에서 (A,B)가 작은 오차로 복원된다.
- 학습하지 않은 시간 block에서 one-step 예측이 persistence를 이긴다.
- 학습된 (B)가 등록한 action intervention contrast를 복원한다.
- 정확한 quadratic planner가 실제 생성계에서도 zero/random control을 이긴다.
- exact atlas와 오염 atlas의 cycle frustration이 분리된다.

뇌, 현실 세계, 장기 계획, nonlinear object permanence는 아직 검증하지 않았다.

---

## 10. 뇌와 주름 가설에 연결되는 최소 범위

공통 감각중추는 발생기 coordinate anchor 후보로 쓸 수 있다.

\[
A_{\rm visual},
A_{\rm auditory},
A_{\rm somatic},
A_{\rm motor}.
\]

정리 2--3은 이 anchor들 사이를 연결 graph에서 harmonic coordinate로 채울
수 있음을 보인다. 그러나 이 coordinate가 실제 피질 성장계량을 만든다는
결론은 나오지 않는다. 필요한 bridge는

\[
g_{\rm connect}
\xrightarrow{\text{unknown developmental map}}
\bar g_{\rm growth}
\xrightarrow{\text{morphoelasticity}}
X(\mathcal S)\subset\mathbb R^3
\]

이다.

현재 허용되는 작은 가설은 다음이다.

> 공통 감각 anchor와 초기 구조연결이 만든 anisotropic coordinate field가
> 피질의 차등성장에 작은 방향성 perturbation을 주어, 주름의 전체 생성력이
> 아니라 위치와 방향을 편향시킬 수 있다.

일차감각 영역은 주름과 기능의 대응이 비교적 강하지만, 전체 피질 경계를
고랑만으로 정할 수는 없다. 인간 피질의 개별 parcellation은 구조, myelin,
task activation, resting connectivity를 함께 써야 한다
([Glasser et al., 2016](https://www.nature.com/articles/nature18933)).

주름 발생의 기본 동력은 여전히 차등성장과 역학적 불안정이다
([Tallinen et al., 2016](https://www.nature.com/articles/nphys3632)). 감각
anchor가 성장장에 영향을 주는지는 태아 종단 MRI/dMRI 자료에서 별도로
검증해야 한다.

---

## 11. 성숙 뒤 새 고랑 없이도 세계모델이 변하는 이유

발생기에는 물리형상 (X)와 연결 (W)가 모두 변한다.

\[
\dot X\ne0,
\qquad
\dot W\ne0.
\]

성숙기에는 거시형상 가동성이 작아지고

\[
\dot X\approx0,
\qquad
\dot W\ne0,
\qquad
\dot A\ne0
\]

인 regime으로 넘어간다고 가정할 수 있다. 따라서 주름은 현재 세계모델의
실시간 그림이라기보다 발생기에 컴파일된 공통 좌표골격 후보이며, 성인
학습은 주로 transition과 coupling parameter에서 일어난다.

이 역시 이 문서의 선형정리에서 도출된 생물학적 사실이 아니라, 주름과
지속학습을 양립시키기 위한 bridge 가설이다.

---

## 12. 반증 조건

다음 결과가 나오면 현재 가설을 축소하거나 폐기한다.

1. 여러 감각 chart를 결합해도 행동 관련 latent state의 holdout 예측이
   단일 chart보다 좋아지지 않는다.
2. observational fit은 좋지만 등록한 intervention effect의 부호나 크기를
   반복적으로 틀린다.
3. counterfactual rollout이 model-free 또는 persistence baseline보다 실제
   행동비용을 낮추지 못한다.
4. chart cycle frustration이 오류·갈등·추가 감각 필요성을 예측하지 못한다.
5. nonlinear baseline이 항상 이기고 국소 chart/gluing 구조의 추가 이득이
   없다.
6. pre-fold 구조연결 coordinate가 두께·성장률·초기형상을 통제한 뒤 미래
   sulcus 위치에 아무 추가 예측력을 주지 않는다.
7. 공통 감각 anchor가 개인별 기능좌표의 안정된 경계조건으로 작동하지 않는다.

---

## 13. 재현 명령

```powershell
python examples/agi/causal_world_simulator_gate.py

.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_causal_world_simulator.py
```

출력:

- `artifacts/agi/causal_world_report.json`
- `artifacts/agi/causal_world_data.csv`

---

## 14. 최종 지위표

| 명제 | 지위 |
|---|---|
| full-rank sensor atlas의 exact reconstruction | `Exact` |
| rank-deficient chart의 비식별성 | `Exact` |
| anchor harmonic extension의 유일성과 최소에너지 | `Exact conditional` |
| persistent excitation 아래 선형 (A,B) 식별 | `Exact conditional` |
| chart cocycle와 identity holonomy | `Exact` |
| cycle consistency에서 전역 linear atlas 구성 | `Exact conditional` |
| 감각제약 gradient inference의 에너지 감소 | `Exact conditional` |
| 수축 observer의 ISS bound | `Exact conditional` |
| 선형 SCM의 one-step intervention effect | `Exact conditional` |
| quadratic one-step planner의 최적성 | `Exact conditional` |
| Lipschitz rollout error bound | `Exact conditional` |
| 등록 합성 데이터 gate | `Engineering PASS` |
| 인간 뇌가 이 알고리즘을 사용함 | `Open` |
| 감각 anchor가 주름 위치를 인과적으로 결정함 | `Open` |
| 장기 nonlinear 세계모델·object permanence | `Not tested` |
| 의식이 전역 compatible section임 | `Speculative` |

현재 가장 강한 결론은 다음이다.

\[
\boxed{
\text{부분감각 결합}
+\text{식별 가능한 행동전이}
+\text{일관된 chart gluing}
+\text{반사실 제어}
}
\]

는 작은 세계 시뮬레이터를 구성하기 위한 수학적으로 충분한 한 세트이며,
등록된 선형 합성계에서 실제로 함께 작동한다. 이것이 인간 뇌의 생성
알고리즘인지는 이후 nonlinear·발달·생물 자료 게이트의 문제다.
