# A1/Q0 공변 작용 강화 루프

> **현재 총괄 상태: `OPEN`**
>
> 이 문서는 A1의 보통 functional Hessian을 물리적 텐서 또는
> 에너지-운동량 텐서로 곧바로 읽을 수 없다는 점을 먼저 고정하고,
> 그 간극을 닫기 위한 최소 조건과 Q0 통과 기준을 정의한다.
> 아래의 수학적 보정들은 모두 명시한 작용·배경·장공간 기하·게이지
> 고정·정규화가 주어졌을 때만 `CONDITIONAL`이다. 이 문서 자체는
> CE+SM의 완결된 양자 작용, 유한한 물리 스펙트럼, 양의 jump rate 또는
> Poisson offspring 행렬을 유도하지 않는다.

## 1. 이 루프가 닫아야 하는 간극

A1을 개략적으로

$$
\Phi\sim\frac{\delta^2 S}{\delta\gamma\,\delta\gamma}
$$

라고 쓸 때 서로 다른 네 객체가 한 문장 안에서 섞이기 쉽다.

1. **보통 functional Hessian**:
   선택한 장 좌표에서 두 번 미분한 kernel
2. **공변 field-space Hessian**:
   장공간 connection으로 좌표변환의 비텐서 항을 제거한 bilinear form
3. **양자 fluctuation operator**:
   배경, gauge fixing, ghost와 경계조건을 고정한 뒤 propagator와
   determinant를 정의하는 연산자
4. **에너지-운동량 텐서**:
   renormalized action 또는 effective action을 시공간 metric으로 한 번
   변분해 얻는 응답

이 네 객체는 일반적으로 같지 않다. 특히 field-space index와 spacetime
index는 이름이 비슷해도 서로 다른 공간의 index다. 이들을 연결하려면
명시적인 map과 그 map의 공변성·차원·Ward identity를 따로 증명해야 한다.

따라서 이 루프가 답해야 할 질문은 다음과 같다.

> 어떤 완결된 CE+SM 작용과 양자화 규약 아래에서 A1의 이차 변분이
> parametrization과 gauge에 안전한 물리적 fluctuation operator가 되며,
> 같은 작용의 metric variation으로 정의한 stress tensor가 diffeomorphism
> Noether identity를 만족하는가?

현재 이 질문의 답은 `OPEN`이다.

## 2. 보통 Hessian이 텐서가 아닌 최소 반례

### 2.1 유한차원에서 이미 발생하는 실패

장공간의 유한차원 모형으로 한 변수 \(q\)와 스칼라 함수

$$
S(q)=q
$$

를 생각하자. \(q\) 좌표에서 보통 Hessian은

$$
\frac{d^2S}{dq^2}=0
$$

이다. 이제 \(y>0\)인 국소 영역에서 가역적인 비선형 좌표변환

$$
q=y^2
$$

를 사용하면

$$
S(y)=y^2,
\qquad
\frac{d^2S}{dy^2}=2
$$

가 된다. 0인 rank-2 tensor를 좌표변환했다면 여전히 0이어야 하므로,
보통 Hessian은 일반 좌표변환에서 tensor가 아니다.

### 2.2 일반 변환식

장 좌표를 \(\phi^a\), 새 좌표를 \(\psi^\alpha\)라 쓰면

$$
\frac{\partial^2S}{\partial\psi^\alpha\partial\psi^\beta}
=
\frac{\partial\phi^a}{\partial\psi^\alpha}
\frac{\partial\phi^b}{\partial\psi^\beta}
\frac{\partial^2S}{\partial\phi^a\partial\phi^b}
+
\frac{\partial^2\phi^a}
{\partial\psi^\alpha\partial\psi^\beta}
\frac{\partial S}{\partial\phi^a}.
$$

마지막 항이 tensor 변환법칙을 깨뜨린다. 이 항은 다음과 같은 제한된
경우에만 사라진다.

- 좌표변환이 affine인 경우
- 평가 배경이 정확한 stationary point여서
  \(\partial_aS=0\)인 경우
- connection 항을 포함한 공변 Hessian을 사용한 경우

따라서 “\(S\)가 스칼라이므로 \(\delta^2S\)는 자동으로 tensor”라는
주장은 일반적으로 성립하지 않는다. 안장점에서 우연히 일치하는 사실도
off-shell 유효작용이나 background scan 전체에 대한 공변성을 보장하지
않는다.

### 2.3 functional 경우

DeWitt condensed index

$$
A=(I,x)
$$

를 사용하면 \(I\)는 장 종류·내부·spacetime tensor index를, \(x\)는
시공간 점을 나타내며 반복 index에는 적분이 포함된다. 비선형
field redefinition \(\phi^A=\phi^A[\psi]\) 아래에서도

$$
S_{,\alpha\beta}
=
\phi^A{}_{,\alpha}\phi^B{}_{,\beta}S_{,AB}
+
\phi^A{}_{,\alpha\beta}S_{,A}
$$

라는 같은 비텐서 항이 나타난다. coincident functional derivative에는
추가로 분포의 곱과 UV 발산이 생기므로, 좌표변환 문제를 해결한 뒤에도
regulator와 renormalization이 별도로 필요하다.

**현재 판정:** A1의 보통 functional Hessian을 그 자체로 공변 물리
텐서라고 읽는 단계는 `OPEN`이다.

## 3. 조건부 보정: 공변 field-space Hessian

### 3.1 최소 구성

장공간 \(\mathcal F\)에 metric \(G_{AB}[\phi]\)와 이에 호환되는
connection \(\Gamma^C{}_{AB}\)를 지정하면

$$
\nabla_A\nabla_BS
\equiv
S_{,AB}-\Gamma^C{}_{AB}S_{,C}
$$

를 정의할 수 있다. connection의 비텐서 변환항이
\(\phi^A{}_{,\alpha\beta}S_{,A}\)를 상쇄하므로 이 객체는 장공간의
공변 rank-2 tensor로 변환한다.

2절의 반례에서도 이를 직접 볼 수 있다. \(q\) 좌표의
\(G_{qq}=1\)을 \(y\) 좌표로 옮기면 \(G_{yy}=4y^2\)이고
\(\Gamma^y{}_{yy}=1/y\)이다. 따라서

$$
\nabla_y\nabla_yS
=
\frac{d^2S}{dy^2}
-\Gamma^y{}_{yy}\frac{dS}{dy}
=
2-\frac{1}{y}(2y)
=0,
$$

가 되어 \(q\) 좌표의 0과 일치한다. 보통 Hessian의 2를 물리량으로
읽은 것이 좌표 인공물이었다는 뜻이다.

그러나 이 식만 적는 것으로 물리적 구성이 끝나지는 않는다.

- \(G_{AB}\)의 선택과 measure가 작용·대칭·차원과 양립해야 한다.
- connection이 \(G_{AB}\)의 Levi-Civita connection인지, torsion이나
  다른 구조를 허용하는지 고정해야 한다.
- metric, gauge field, fermion과 CE sector를 포함하는 전체 장공간에서
  index와 reality condition을 명시해야 한다.
- Lorentzian과 Euclidean signature, contour와 경계조건을 구분해야 한다.
- 배경이 off-shell이면 connection 항과 tadpole을 버리면 안 된다.

정확한 stationary background에서는 \(S_{,A}=0\)이므로 보통 Hessian과
공변 Hessian이 그 점에서 일치한다. 이는 특정 점에서의 일치일 뿐,
장공간 전체의 좌표불변성을 보통 Hessian에 부여하지 않는다.

### 3.2 gauge theory에서 한 단계 더 필요한 것

게이지 변환은 물리적으로 같은 장 구성을 잇기 때문에 gauge orbit
방향의 bare Hessian은 일반적으로 zero mode를 가진다. 따라서

$$
\det S_{,AB}
$$

를 gauge fixing 없이 물리 determinant로 읽을 수 없다. 최소한 다음 중
어떤 구성을 쓰는지 고정해야 한다.

- background-field gauge와 Faddeev–Popov ghost를 포함한 gauge-fixed
  fluctuation operator
- field-space를 gauge orbit으로 나눈 quotient의 horizontal projection
- parametrization·gauge-condition 의존성을 제어하는
  Vilkovisky–DeWitt형 effective action

Vilkovisky–DeWitt라는 이름만 도입해도 자동으로 통과하지 않는다.
field-space metric, orbit projection, connection, measure와 regularization을
실제로 명시하고 해당 Ward/BRST identity를 검사해야 한다.

**현재 판정:** 공변 Hessian은 수학적 보정 후보로 `CONDITIONAL`이다.
CE+SM 전체 장공간 metric과 gauge quotient는 아직 `OPEN`이다.

## 4. stress tensor는 metric variation으로 정의한다

### 4.1 Hessian과 stress tensor의 역할 분리

renormalized quantum effective action을

$$
\Gamma_{\mathrm{ren}}[g,\phi;\mu,\mathcal S]
$$

라고 하자. 여기서 \(\mu\)는 renormalization scale,
\(\mathcal S\)는 scheme을 나타낸다. 부호 convention을 고정하면
stress tensor는

$$
T_{\mu\nu}(x)
=
-\frac{2}{\sqrt{-g(x)}}
\frac{\delta\Gamma_{\mathrm{ren}}}{\delta g^{\mu\nu}(x)}
$$

로 정의한다. Euclidean signature에서는 선택한 convention에 따라
전체 부호가 달라질 수 있으므로 문서와 코드에서 한 convention을
일관되게 사용해야 한다.

이 정의는 field-space Hessian과 다르다.

- \(\nabla_A\nabla_BS\)는 장 fluctuation 두 개에 대한 이차 응답이다.
- \(T_{\mu\nu}\)는 spacetime metric에 대한 일차 응답이다.
- metric 자체를 field-space 좌표에 포함해도 metric–metric Hessian은
  stress tensor가 아니라 stress tensor의 추가 선형응답이다.

따라서 “propagator를 정하는 Hessian이므로 \(T_{\mu\nu}\)에 대응한다”는
문장만으로 두 객체의 동일성이 나오지 않는다. 대응을 주장하려면
명시적 작용으로 양쪽을 독립 계산하고, index map·차원·정규화·보존식을
검사해야 한다.

### 4.2 diffeomorphism Noether identity

\(\Gamma_{\mathrm{ren}}\)이 diffeomorphism invariant이고 anomaly가
없다면, 임의의 vector field \(\xi^\mu\)에 대해

$$
0
=
\delta_\xi\Gamma_{\mathrm{ren}}
=
\int d^4x
\left[
\frac{\delta\Gamma_{\mathrm{ren}}}{\delta g_{\mu\nu}}
\mathcal L_\xi g_{\mu\nu}
+
\frac{\delta\Gamma_{\mathrm{ren}}}{\delta\phi^I}
\mathcal L_\xi\phi^I
\right]
$$

가 성립해야 한다. 이것이 일반적인 off-shell Noether identity다.
예를 들어 scalar sector에서는

$$
\nabla^\mu T_{\mu\nu}
=
\sum_I E_I\,\nabla_\nu\phi^I,
\qquad
E_I
\equiv
\frac{1}{\sqrt{-g}}
\frac{\delta\Gamma_{\mathrm{ren}}}{\delta\phi^I}
$$

형태가 된다. 따라서

$$
E_I=0
\quad\Longrightarrow\quad
\nabla^\mu T_{\mu\nu}=0
$$

이다. 보존은 단순히 \(T_{\mu\nu}\)라는 이름을 붙인 결과가 아니라,

1. 전체 action/effective action의 diffeomorphism invariance,
2. quantum measure와 regulator의 대칭 보존 또는 anomaly 상쇄,
3. 관련 장의 운동방정식,
4. 적절한 경계조건

에서 나오는 **on-shell 결론**이다. gauge field와 tensor field가 있으면
각 장의 Lie derivative에서 추가 항이 생기며, gauge-fixed 표현에서는
ghost와 gauge-fixing sector를 포함한 background Ward identity 또는
BRST/Slavnov–Taylor identity로 검사해야 한다.

Bianchi identity는 Einstein 방정식의 좌변 발산이 0임을 말한다. 임의로
정의한 \(\mathcal K_{\mu\nu}\)의 보존이나 그것이 metric variation으로
얻은 stress tensor라는 사실을 대신 증명하지 않는다.

**현재 판정:** 명시된 \(\Gamma_{\mathrm{ren}}\)의 metric variation이라는
정의와 Noether 논리는 `CONDITIONAL`이다. A1의 Hessian 기대값을 이
stress tensor와 동일시하는 CE bridge는 `OPEN`이다.

## 5. gauge fixing, ghost, measure와 renormalization

Q0가 물리적 quadratic operator를 내놓으려면 적어도 다음 자료가 한
묶음으로 고정되어야 한다.

### 5.1 작용과 배경

- CE, metric, SM gauge field, Higgs, fermion 및 필요한 보조장의 전체
  bare action 또는 renormalized action
- 각 장의 normalization, 질량차원, reality condition과 coupling
- background split과 배경 운동방정식
- Lorentzian/Euclidean signature, integration contour와 경계조건
- higher-derivative 항이 있다면 추가 pole과 ghost의 물리적 처분

### 5.2 gauge와 물리 상태

- gauge-fixing functional, gauge parameter와 Faddeev–Popov operator
- ghost action 및 필요한 Jacobian
- BRST transformation과 nilpotency
- Ward 또는 Slavnov–Taylor residual
- 물리 pole·cut·on-shell observable의 gauge-parameter 안정성

off-shell Hessian 원소와 개별 effective potential 계수는 일반적으로
gauge-dependent일 수 있다. 따라서 특정 gauge의 비대각 원소를 곧바로
관측 가능한 혼합률이나 decay rate로 읽을 수 없다.

### 5.3 measure와 field redefinition

형식 기호 \(\mathcal D\phi\)는 자동으로 좌표불변 measure가 아니다.
다음 항목을 고정해야 한다.

- field-space metric에서 유도한 measure인지 여부
- nonlinear field redefinition의 functional Jacobian
- gauge orbit volume의 제거 방식
- zero mode와 collective coordinate의 처리
- regulator가 diffeomorphism, gauge/BRST, \(Z_2\)를 보존하는지 여부

measure 또는 regulator가 대칭을 깨면 anomaly나 유한 counterterm이
Noether identity에 들어갈 수 있다.

### 5.4 renormalization

coincident Hessian, determinant와 stress tensor expectation은 일반적으로
UV divergent다. 최소한 다음을 명시해야 한다.

- regulator와 subtraction scheme
- 허용되는 모든 local counterterm
- curved background에서 필요한 curvature counterterm
- renormalization condition, scale \(\mu\)와 RG running
- operator mixing과 composite-operator renormalization
- scheme/scale 변화 아래 최종 물리 pole·cut의 안정성

“시공간 체적으로 나누면 에너지 밀도가 된다”는 차원 논리는 이
renormalization 문제를 해결하지 않는다.

**현재 판정:** CE+SM에 대한 이 양자화 자료의 완결은 `OPEN`이다.

## 6. \(Z_2\), \(v_\Phi=0\) portal의 Q0 기준점

### 6.1 진공 전개

다음 portal convention을 택하자.

$$
\mathcal L_{\mathrm{portal}}
=
-\lambda_{\mathrm{HP}}|H|^2\Phi^2,
\qquad
H=
\frac{1}{\sqrt2}
\begin{pmatrix}
0\\
v+h
\end{pmatrix}.
$$

그러면

$$
\mathcal L_{\mathrm{portal}}
=
-\frac{\lambda_{\mathrm{HP}}}{2}
(v+h)^2\Phi^2
=
-\frac{\lambda_{\mathrm{HP}}v^2}{2}\Phi^2
-\lambda_{\mathrm{HP}}vh\Phi^2
-\frac{\lambda_{\mathrm{HP}}}{2}h^2\Phi^2.
$$

\(Z_2:\Phi\mapsto-\Phi\)가 보존되고 \(v_\Phi=0\)인 배경
\((h,\Phi)=(0,0)\)에서

$$
\left.
\frac{\partial^2\mathcal L_{\mathrm{portal}}}
{\partial h\,\partial\Phi}
\right|_{0}
=0.
$$

즉 portal은 \(\Phi\)의 quadratic mass block을 이동시키지만,
\(h\)-\(\Phi\) bilinear mixing block은 만들지 않는다. 반면

$$
\left.
\frac{\partial^3\mathcal L_{\mathrm{portal}}}
{\partial h\,\partial\Phi^2}
\right|_{0}
=-2\lambda_{\mathrm{HP}}v,
\qquad
\left.
\frac{\partial^4\mathcal L_{\mathrm{portal}}}
{\partial h^2\,\partial\Phi^2}
\right|_{0}
=-2\lambda_{\mathrm{HP}}
$$

이므로 \(\lambda_{\mathrm{HP}}v\neq0\)이면 \(h\Phi^2\) cubic vertex가,
\(\lambda_{\mathrm{HP}}\neq0\)이면 \(h^2\Phi^2\) quartic vertex가 남는다.
정확한 Feynman-rule 부호와 조합계수는 전체 action convention에 맞춰
고정해야 한다.

### 6.2 이 계산이 말하지 않는 것

이 전개는 지정된 portal Lagrangian과 진공 아래의 대수적 결과로만
`CONDITIONAL`이다. 다음 결론은 자동으로 따라오지 않는다.

- \(\Phi\)가 asymptotic physical particle이라는 결론
- \(h\to\Phi\Phi\)가 kinematically 열려 있다는 결론
- 해당 vertex가 양의 Markov jump rate를 정의한다는 결론
- 한 번의 pair-production을 단위 offspring Poisson 사건으로
  바꿀 수 있다는 결론
- portal coupling의 값이나 CE의 \(D_{\mathrm{eff}}\) 계수

실제 rate에는 propagating spectrum, LSZ 또는 적절한 open-system
observable, loop self-energy, physical cut와 phase space가 필요하다.
한 사건이 두 \(\Phi\)를 만든다면 단순 단위-batch Poisson이 아니라
compound-Poisson 또는 상관된 offspring 후보도 함께 비교해야 한다.

### 6.3 canonical 포탈 benchmark의 비가시 폭 gate

위 convention에서 \(h\Phi^2\) vertex의 크기는
\(2\lambda_{\mathrm{HP}}v\)다. 채널이 열려 있으면 tree level에서

$$
\Gamma(h\to\Phi\Phi)
=
\frac{\lambda_{\mathrm{HP}}^2v^2}{8\pi m_h}
\sqrt{1-\frac{4m_\Phi^2}{m_h^2}}
$$

다. 2026-08-06 canonical manifest의
\(\lambda_{\mathrm{HP}}=\delta_N^2=0.0316530354\),
\(v=246.21965\,\mathrm{GeV}\), \(\mu_\Phi=0\),
\(m_\Phi=v\sqrt{\lambda_{\mathrm{HP}}}=43.8056765\,\mathrm{GeV}\),
PDG 2026 Higgs snapshot \(m_h=125.11\,\mathrm{GeV}\)와
\(\Gamma_h^{\mathrm{SM}}=4.10\,\mathrm{MeV}\)를 넣으면

$$
\Gamma_{\mathrm{inv}}=13.790042\,\mathrm{MeV},
\qquad
\mathrm{BR}_{\mathrm{inv}}=0.77082222.
$$

[PDG 2026 Higgs review](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-higgs-boson.pdf)가
열거한 ATLAS direct Run-2 observed 95% CL 상한
\(\mathrm{BR}_{\rm inv}<0.107\)을 적용하면 이 benchmark는 통과하지
못한다.
과거 \(13.75\,\mathrm{MeV}\), \(0.772\)는 구 입력 snapshot으로만 남기고
현행 판정에는 사용하지 않는다.

이 계산은 선택한 portal EFT의 조건부 tree-level 반증 gate다. 포탈을
CE에서 유도하지 않으며, 코어의 “독립 on-shell scalar 없음” 분기에는
적용되지 않는다. 기존 `a1_q0_action_bridge.py`가 다른 입력 snapshot을
고정했다면 이 절의 canonical manifest와 일치하도록 별도 갱신해야 하며,
그 실행값을 현행 결과로 재사용하지 않는다.

## 7. Q0 acceptance gates

Q0의 출력은 아직 spectral positivity나 branching을 증명하는 것이
아니다. Q0는 다음 단계가 계산 가능한 **공변·게이지 일관 작용과
vertex/pole 자료**를 제공하는 데까지만 책임진다.

| Gate | 제출물 | 통과 조건 | 현재 상태 |
|---|---|---|---|
| `Q0.0-scope` | 전체 field content, action 종류, signature, background, boundary, 단위와 convention manifest | 누락된 sector가 없고 bare/renormalized/effective action을 혼용하지 않음 | `OPEN` |
| `Q0.1-field-space` | \(G_{AB}\), connection, measure와 nonlinear reparametrization test | 보통 Hessian의 비텐서 항을 재현하고 공변 Hessian의 두 좌표계 residual이 허용치 이하 | `OPEN` |
| `Q0.2-background` | background EOM과 tadpole residual | on-shell이면 EOM residual 통과, off-shell이면 connection·tadpole 항을 보존 | `OPEN` |
| `Q0.3-gauge` | gauge fixing, ghost action, BRST/Slavnov–Taylor 또는 background Ward 검사 | zero mode 처리가 명시되고 identity residual과 gauge-parameter holdout이 사전 기준 통과 | `OPEN` |
| `Q0.4-operator` | 전체 quadratic fluctuation operator와 propagator pole 표 | 물리 pole, constraint, negative/zero mode와 higher-derivative ghost의 처분이 일관됨 | singlet bare tree block `CONTROL PASS`; full `OPEN` |
| `Q0.5-vertices` | cubic/quartic vertex 목록과 symmetry selection rule | \(Z_2,v_\Phi=0\)에서 \(h\)-\(\Phi\) cross-Hessian 0, \(h\Phi^2,h^2\Phi^2\) vertex 및 누락 sector 재현 | singlet portal local block `CONTROL PASS`; full `OPEN` |
| `Q0.6-stress` | \(\Gamma_{\rm ren}\)의 metric variation과 off-shell Noether identity | Hessian과 stress tensor를 분리하고, on-shell conservation 및 경계항을 명시적으로 검증 | `OPEN` |
| `Q0.7-quantum` | measure, regulator, counterterm, anomaly와 RG manifest | 필요한 발산이 흡수되고 물리 pole/cut이 scheme·scale·gauge holdout에서 안정 | two-real-scalar finite 1-loop `DIAGNOSTIC`; full `OPEN` |
| `Q0.8-reproduction` | symbolic derivation, 수치 spectrum, test와 고정 입력 manifest | 깨끗한 환경에서 동일 결과 재현, 실패 반례와 허용치를 함께 보존 | `OPEN` |

Q0.4–Q0.5의 `CONTROL PASS`는 선택적 \(Z_2\) singlet block에만 적용된다.
canonical bare tree kernel

\[
K_F=p^2-(m_0^2+\lambda_{HP}v^2)+i0
\]

의 residue와 dispersion, 그리고
\(h\Phi^2,h^2\Phi^2,\chi^2\Phi^2\) local derivative를 재현했다. 그러나
29.6991596 MeV는 bare mass를 목표에서 역산해야만 이 tree pole이 되고,
renormalized CE pole·LSZ·full vertex 목록은 여전히 `OPEN`이다. 상세 반례와
수치는 [CE_TWO_POINT_AND_VERTEX_LOOP.md](CE_TWO_POINT_AND_VERTEX_LOOP.md)에
둔다.

후속 loop에서는 Q0 action definition을 canonical SHA-256에 묶고, 수치
\(\Gamma_R^{(2)}\) replica가 들어올 경우 simple root, residue, first cut,
gauge/scale drift와 dispersion을 caller bool 없이 재계산하는 gate를 추가했다.
현재 CE에는 complete renormalized action/counterterm manifest와 kernel data가
없어 최고 단계는 `REGISTERED_SCALE`이다. 선택적 portal의 two-real-scalar
finite one-loop 합은 light target 질량제곱의 약 5301.42배지만 subtraction scale에
따라 부호가 바뀌므로 물리 pole correction으로 승격하지 않는다. 상세 식은
[CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md](CE_RENORMALIZED_POLE_AND_ONE_LOOP_LOOP.md)에
둔다.

그 다음 Q1 입력을 저장소 전체에서 찾았지만 CE operator에 묶인 raw paired
Euclidean ensemble과 covariance는 없었다. connected subtraction·delete-one
jackknife와 유한 two-point positivity 필요조건을 계산하는 scaffold를 추가했고,
fixed exponential kernel의 augmented nullspace에서 같은 유한 correlator와
총 weight를 만드는 서로 다른 두 비음수 spectrum을 구성했다. 따라서 합성 대조군이 screening control을
통과해도 unique spectrum, full reflection positivity, Minkowski pole과 LSZ로
승격하지 않는다. 상세 범위는
[CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md](CE_EUCLIDEAN_CORRELATOR_AND_SPECTRAL_LOOP.md)에
둔다.

### 7.1 Q0 통과 판정 규칙

Q0 전체는 위 hard gate가 모두 통과하기 전까지 `OPEN`으로 유지한다.
일부 계산이 성공한 경우 허용되는 가장 강한 문장은 다음과 같다.

> 지정한 작용, 배경, field-space geometry, gauge, regulator와
> renormalization scheme 아래에서 공변 quadratic operator와
> cubic/quartic vertex를 조건부로 구성했다.

다음 문장은 Q0만으로 허용하지 않는다.

- “A1로부터 spacetime stress tensor가 직접 유도되었다.”
- “CE+SM 완전 작용이 유일하게 결정되었다.”
- “Hessian의 비대각 원소 제곱이 물리 decay rate다.”
- “\(Z_2\) portal이 Poisson branching과 offspring 행렬 \(A\)를
  유도한다.”
- “Ward identity 하나로 renormalization과 gauge independence가 모두
  해결되었다.”

## 8. 반례 우선 실행 순서

다음 루프는 성공 예제를 맞추기 전에 실패해야 하는 입력을 먼저 고정한다.

1. **비선형 좌표 반례:** \(S(q)=q,\ q=y^2\)에서 보통 Hessian 불일치
2. **공변 보정 회귀:** 같은 반례에서 connection을 포함한 Hessian 일치
3. **gauge zero-mode 반례:** gauge fixing 전 determinant의 singularity
4. **off-shell 반례:** \(S_{,A}\neq0\)에서 connection 항 삭제 시
   parametrization dependence 재발
5. **portal 기준점:** \(Z_2,v_\Phi=0\)에서 cross-Hessian 0과
   cubic/quartic vertex 비영을 동시에 재현
6. **보존 기준점:** off-shell에서는 EOM source가 남고 on-shell에서만
   stress tensor divergence가 사라짐을 재현
7. **scheme/gauge holdout:** 중간 Hessian 원소는 변할 수 있지만 사전
   지정한 물리 pole·cut은 허용치 안에서 안정한지 검사

이 순서가 통과한 뒤에만 Q1의 spectral density와 흡수 self-energy
계산으로 이동한다.

## 9. 증명 의존성 및 현재 진행상황

```text
ordinary Hessian
  └─[nonlinear reparametrization 반례]→ 일반 tensor 해석 불가

명시적 CE+SM action + field-space metric/connection
  └─[CONDITIONAL]→ covariant field-space Hessian
       + gauge fixing/ghost/measure
       + background EOM 또는 off-shell connection/tadpole
       + regulator/counterterm/BRST
       └─[OPEN: Q0]→ physical quadratic operator + vertices + poles

renormalized effective action
  └─[metric variation, CONDITIONAL]→ stress tensor
       └─[diffeomorphism identity + EOM + anomaly 부재]
          → on-shell conservation

Q0 산출물
  └─[수치 pole control gate 구현; 실제 CE kernel 없음]→ Q1 spectral positivity
       └─[Euclidean scaffold 구현; 실제 paired ensemble 없음;
          finite-grid spectrum 비유일성 구성]→ REGISTERED_SCALE 유지
       → Q2–Q5 CP/Markov/classical closure
       → Q6 offspring genealogy
       → 실제 next-generation matrix A
```

| 명제 | 현재 판정 | 부족한 증거 |
|---|---|---|
| 보통 functional Hessian은 일반 field redefinition에서 tensor다 | `OPEN`으로 사용할 수 없음 | 명시적 반례가 있으므로 공변 보정 없이 채택 불가 |
| 공변 field-space Hessian을 구성할 수 있다 | `CONDITIONAL` | 전체 \(G_{AB}\), connection, gauge quotient와 measure |
| A1의 Hessian 기대값이 spacetime \(\mathcal K_{\mu\nu}\)다 | `OPEN` | index map, 복합연산자 정규화, 공변성과 차원 |
| \(\mathcal K_{\mu\nu}=T_{\mu\nu}\)다 | `OPEN` | 동일 action에서의 독립 metric variation과 matching |
| \(T_{\mu\nu}\)가 보존된다 | `CONDITIONAL` | diffeomorphism/BRST identity, EOM, 경계와 anomaly 감사 |
| \(Z_2,v_\Phi=0\)에서 cross-Hessian은 0이고 portal vertex는 남는다 | `CONDITIONAL` | 지정 portal action·배경 아래 대수는 성립, 전체 CE 물리 해석은 미완 |
| CE+SM Q0 action이 완결되었다 | `OPEN` | `Q0.0`–`Q0.8` 전체 |
| Q0가 양의 jump rate와 Poisson branching을 준다 | `OPEN` | Q1–Q6은 Q0와 별도 gate |

## 10. 이 루프의 정지 조건

다음 중 하나가 발생하면 성공으로 포장하지 않고 해당 지점에서 멈춘다.

- 전체 장공간 metric 또는 gauge quotient를 일관되게 정의할 수 없음
- BRST/Slavnov–Taylor identity가 regulator와 counterterm 뒤에도 복구되지
  않음
- 물리 spectrum에 처분되지 않은 음의 norm 또는 불안정 pole이 남음
- stress tensor matching이 parametrization, gauge 또는 scheme 변화에
  따라 물리적으로 달라짐
- \(Z_2\) 진공에서 비영 \(h\)-\(\Phi\) bilinear가 나오지만 그 원인이
  대칭 깨짐·잘못된 background·미포함 tadpole로 설명되지 않음
- Q0 결과가 재현되지 않거나 입력 convention 변경에 취약함

실패 결과는 CE 전체의 즉시 폐기를 뜻하지 않는다. 실패한 action,
background, connection, gauge 또는 matching 규칙을 특정하여 다음
모형 선택 루프의 입력으로 돌린다. 반대로 Q0가 통과하더라도 결론은
공변 fluctuation 문제의 조건부 완결까지이며, CE+SM에서 양의 Poisson
재귀가 유도되었다는 결론은 Q1–Q6 이전에는 계속 `OPEN`이다.
