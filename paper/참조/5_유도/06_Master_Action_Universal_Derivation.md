# 6. Master action: 변분 원리, EFT branch, 일관된 축약

이 장의 master action은 여러 분야의 방정식을 하나의 scalar 식에서 자동으로
산출한다는 명제가 아니다. field content, symmetry와 boundary condition을
명시한 작용에서 Euler--Lagrange 방정식이 어떻게 나오고, 어떤 축약이
일관적인지를 정리한다.

독자는 변분법, 미분기하의 공변미분, 그리고 유효장론의 기본 용어를 안다고
가정한다. 6.1절은 경계조건을 포함한 일반 변분 구조를 고정하고, 6.2–6.4절은
그 구조를 scalar--tensor branch와 축약 조건에 적용한다. 6.5절은 소산계를
같은 단일 작용으로 오인하지 않도록 경계를 세우며, 6.6–6.7절은 유한
Euclidean cutoff와 무차원 수치 목적함수를 별도 도구로 분리한다.

## 6.1 일반 변분 구조

개별 모형을 고르기 전에, 작용에서 방정식을 얻을 때 경계항이 어디에
들어가는지부터 고정해야 한다. 이 절은 그 공통 변분 구조를 제시하고,
다음 절에서 특정 EFT family를 선택할 수 있는 출발점을 만든다.

**[정의]** 시공간 $M$ 위 field를 $\Phi^A$, coupling을
$\lambda_I$라 하고
$$
S[\Phi;\lambda]
=\int_M d^4x\,\sqrt{-g}\,
L(\Phi^A,\nabla_\mu\Phi^A;\lambda_I)
$$
를 정의한다. 자연단위에서 $S$는 무차원이고 SI에서는 $S/\hbar$가
무차원이다.

**[정리]** 일차미분 Lagrangian의 variation은
$$
\delta S
=\int_M d^4x\,\sqrt{-g}\,
\mathcal E_A\,\delta\Phi^A
+\int_{\partial M}\Theta(\Phi,\delta\Phi),
$$
$$
\mathcal E_A
=\frac{\partial L}{\partial\Phi^A}
-\nabla_\mu
\frac{\partial L}{\partial(\nabla_\mu\Phi^A)}
$$
다. boundary term을 없애는 경계조건 또는 적절한 boundary action을
선택하면
$$
\delta S=0\quad\Longleftrightarrow\quad \mathcal E_A=0
$$
가 모든 독립 variation에 대해 성립한다.

**[정리]** 미분동형사상 불변인 matter+gravity action에는 metric
Euler--Lagrange tensor의 divergence와 다른 field equation을 잇는
Noether identity가 있다. matter와 scalar equation이 성립하면 metric
equation의 Bianchi identity로부터 그 우변의 covariant conservation이
따른다. 이는 서로 독립적으로 고른 background와 source를 자동으로
정합시켜 주지는 않는다.

## 6.2 scalar--tensor EFT branch

일반 변분식은 field content를 정하지 않는다. 다음 작용은 여러 가능한
EFT 가운데 이 문서가 조건부 계산을 위해 택하는 family이며, 계수의
자연적 값을 유도하지 않는다.

**[공리] 모델 선택:**
$$
S_{\rm ST}
=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}F(\phi)R
-\frac12K(\phi)(\nabla\phi)^2
-V(\phi)+\mathcal L_m(g,\Psi)
\right]
$$
를 하나의 재사용 가능한 EFT family로 택한다. $[\phi]=1$,
$F,K$는 무차원이고 $V$의 질량차원은 4다. timelike 또는 spacelike
경계가 있으면 fixed induced metric과 fixed $\phi$를 쓰고
$M_{\rm Pl}^2\int_{\partial M}\sqrt{|h|}\,F K_{\rm ext}$ 같은
generalized Gibbons--Hawking--York 항을 포함하거나, variation을 compact
support로 제한한다.

**[산출]** metric과 scalar variation은
$$
M_{\rm Pl}^2F G_{\mu\nu}
=T_{\mu\nu}^{m}
+K\left(\nabla_\mu\phi\nabla_\nu\phi
-\frac12g_{\mu\nu}(\nabla\phi)^2\right)
-g_{\mu\nu}V
+M_{\rm Pl}^2
(\nabla_\mu\nabla_\nu F-g_{\mu\nu}\Box F),
$$
$$
K\Box\phi+\frac12K_{,\phi}(\nabla\phi)^2
+\frac{M_{\rm Pl}^2}{2}F_{,\phi}R
-V_{,\phi}=0.
$$

**[공리] 안정 영역:**
$$
F(\phi)>0,\qquad
\frac{K}{F}
+\frac{3M_{\rm Pl}^2}{2}
\left(\frac{F_{,\phi}}F\right)^2>0
$$
인 field domain만 사용한다. 첫 조건은 effective graviton kinetic 부호,
둘째는 Einstein-frame scalar kinetic 부호의 충분한 국소 조건이다.

**[미완성]** $F,K,V$의 구체적 함수, quantum correction,
cutoff와 matter coupling은 CE의 공통 수학에서 고정되지 않는다.

## 6.3 상수-field branch

일반 scalar--tensor family 안에서도 상수장을 택하면 일부 해는 Einstein
방정식으로 정확히 축약된다. 그러나 이 축약은 scalar 방정식의 pointwise
조건을 함께 만족하는 branch에만 허용된다.

**[공리] 분기 선택:** $\phi=\phi_0$가 상수이고
$$
\frac{M_{\rm Pl}^2}{2}F_{,\phi}(\phi_0)R
-V_{,\phi}(\phi_0)=0
$$
인 배경을 택한다.

**[산출]** $F_0=F(\phi_0)>0$, $V_0=V(\phi_0)$이면
$$
M_{\rm Pl}^2F_0G_{\mu\nu}
+V_0g_{\mu\nu}=T_{\mu\nu}^{m}.
$$
따라서
$$
M_{\rm Pl,eff}^2=M_{\rm Pl}^2F_0,\qquad
\Lambda_{\rm eff}=\frac{V_0}{M_{\rm Pl}^2F_0}.
$$
이 branch는 Einstein 방정식의 표준 해를 조건부로 포함한다. $F_0,V_0$의
값을 무입력으로 정하지는 않는다.

## 6.4 일관된 축약

상수장 같은 ansatz를 작용에 먼저 대입하는 계산은 편리하지만, 버린
방정식이 자동으로 사라지는지는 별도 문제다. 이를 판별하는 정확한 조건을
다음에 둔다.

**[정의]** full field를 ansatz
$\Phi^A=\iota^A(\psi^a)$로 제한해
$$
S_{\rm red}[\psi]:=S[\iota(\psi)]
$$
를 정의한다.

**[정리]** reduced equation
$\delta S_{\rm red}/\delta\psi^a=0$은 full Euler--Lagrange equation의
ansatz tangent 방향 투영을 준다. 이 해가 full solution이 되려면 ansatz에
수직인 discarded equation도
$$
\mathcal E_A[\iota(\psi)]=0
$$
을 만족해야 한다. 이 추가 조건을 만족하는 축약만 consistent truncation이다.

**[산출]** homogeneous scalar--FLRW, constant-field black-hole branch처럼
symmetry가 discarded mode를 source하지 않는 경우에는 일관된 축약을
구성할 수 있다. 반면 유체, open quantum system과 dissipative material에는
각각 fluid variables, bath와 constitutive law를 추가해야 한다.

## 6.5 dissipative equation의 경계

일관된 축약을 확인해도 모든 유효 방정식이 단일 복사본 Lorentzian 작용에서
나오는 것은 아니다. 다음 정리는 보존적 변분 구조와 소산적 coarse-grained
동역학 사이에 추가 구조가 필요한 이유를 분명히 한다.

**[정리]** 일반적인 single-copy local Lorentzian action의 보존적
Euler--Lagrange flow와 점성 Navier--Stokes 또는 GKSL semigroup는 같은
수학 구조가 아니다. dissipative effective equation을 action으로 기술하려면
environment를 유지한 뒤 적분소거하거나 Schwinger--Keldysh doubled field,
influence functional과 positivity/noise 조건을 사용해야 한다.

**[미완성]** 하나의 실수 scalar potential만으로 viscosity, Lindblad rate,
superconducting collision integral과 nuclear transport coefficient를 모두
산출하는 matching은 없다.

## 6.6 Euclidean cutoff와 saddle

Lorentzian 변분 원리와 수치적 확률 적분을 구분한 뒤, 이제 유한 cutoff에서
정말 존재하는 측도와 saddle 근사를 확인한다. 이 절의 지수는 처음부터
$S_E/\hbar$라는 무차원 양으로 쓴다.

**[공리] Euclidean branch:** 유한 cutoff 뒤의 모든 자유도를 기준척도로
나눈 무차원 좌표 $z\in\mathbb R^N$로 쓴다.
$s_N(z)=S_{E,N}(z)/\hbar$가 연속인 실함수이고 어떤
$c,p>0$, $C\in\mathbb R$에 대해
$$
s_N(z)\ge c\|z\|^p-C
\tag{6.6}
$$
라고 가정한다.

**[정리]** 이 전제에서
$$
0<Z_N=\int_{\mathbb R^N}d^Nz\,e^{-s_N(z)}<\infty .
\tag{6.7}
$$
실제로 integrand는
$e^{C}e^{-c\|z\|^p}$로 지배된다. 지수 $s_N=S_E/\hbar$는
무차원이다.

**[정리] Laplace 선도항:** $z_j$가 고립된 비퇴화 최소점이고
$\mathcal H_j=\nabla^2s_N(z_j)>0$라 하자. 그 최소점의 고정된 작은
근방에 대한
$$
Z_j(\lambda)=\int_{\mathcal U_j}d^Nz\,
e^{-\lambda s_N(z)}
$$
는 $\lambda\to\infty$에서
$$
Z_j(\lambda)\sim
e^{-\lambda s_N(z_j)}
\left(\frac{2\pi}{\lambda}\right)^{N/2}
(\det\mathcal H_j)^{-1/2}.
\tag{6.8}
$$
zero mode, negative mode 또는 gauge orbit가 있으면 이 식의 전제가
깨지며 collective coordinate와 적분 contour를 별도로 정해야 한다.

**[미완성]** continuum limit, gauge fixing, reflection positivity,
renormalization과 Lorentzian reconstruction은 유한 cutoff 정리와 별개다.

## 6.7 dimensionless residual objective

유한 cutoff 적분의 존재와 별개로, 수치 해법은 방정식 잔차를 서로 비교할
척도가 필요하다. 다음 정의는 물리 작용을 대체하지 않는 무차원 최적화
목적함수를 구성한다.

**[정의]** 각 equation residual $\mathcal E_A$의 질량차원을 $d_A$,
질량차원 1의 reference scale을 $\Lambda_A>0$라 하고
$$
\widehat{\mathcal E}_A
:=\frac{\mathcal E_A}{\Lambda_A^{d_A}}
$$
로 무차원화한다. $x^\mu=L_{\rm ref}\hat x^\mu$도 함께 사용한다.

**[공리] 수치모형:** Euclidean positive norm과 무차원 weight
$w_A(\hat x)>0$를 택해
$$
\mathcal J[\Phi]
=\frac12\sum_A\int d^4\hat x\,
w_A(\hat x)|\widehat{\mathcal E}_A|^2
$$
를 최적화 objective로 둔다.

**[정리]**
$$
\mathcal J\ge0,\qquad
\mathcal J=0
\Longleftrightarrow
\mathcal E_A=0\ \text{a.e. for every }A.
$$
이는 positive residual norm의 직접 귀결이다.

이 $\mathcal J$는 physical Lorentzian action $S$와 다른 수치 목적함수다.
$\mathcal J$의 stationary point가 $\mathcal J=0$이거나 full equation의
해라는 보장은 없다.

## 6.8 분야별 필요한 추가 구조

앞의 일반 원리를 각 분야에 적용할 때에는 버린 자유도와 측정 규칙을
명시해야 한다. 다음 표는 master EFT만으로는 부족한 구조와 그때에만
보존되는 조건부 결과를 나란히 둔다.

| 축약 대상 | master EFT 외에 필요한 것 | 보존되는 조건부 결과 |
|---|---|---|
| FLRW·암흑에너지 | $F,K,V$, matter와 초기조건 | Friedmann·scalar equation |
| black hole | constant-field 조건, horizon 경계 | Einstein branch·Wald entropy |
| Navier--Stokes | 유체장, equation of state, viscosity | energy identity·weak form |
| 열린 양자계 | bath, coupling, coarse graining | GKSL·KMS 조건 |
| 초전도체 | fermion band, pairing kernel, bath | gap equation·Floquet expansion |
| 핵융합 | nuclear EFT, plasma self-energy, transport | Yukawa potential·WKB exponent |

**[공리] 외부 scale:** $29.65\,{\rm MeV}$ 같은 benchmark는
$\Lambda_A$ 또는 scan coordinate로 선택할 수 있지만, 그 선택만으로
어느 sector의 pole·coupling·해도 산출되지 않는다.

## 6.9 요약

이 장의 역할은 하나의 보편 방정식을 선언하는 데 있지 않고, 변분·축약·
Euclidean 수치 도구의 정확한 적용 범위를 갈라 놓는 데 있다. 다음 표는
그 구분을 마지막으로 요약한다.

| 항목 | 지위 |
|---|---|
| Euler--Lagrange variation과 Noether identity | [정리] |
| scalar--tensor EFT | [공리]과 [산출] |
| constant-field Einstein branch | [산출] |
| consistent truncation 조건 | [정리] |
| Euclidean finite-cutoff measure와 saddle | [정리] |
| dimensionless residual objective | [공리]과 [정리] |
| 모든 분야의 coupling을 정하는 단일 CE 작용 | [미완성] |
| UV completion과 quantum measure | [미완성] |
