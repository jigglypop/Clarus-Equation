# CE 공리계와 이론물리 모형 사전

이 문서는 CE에서 수학적으로 닫힌 구조, 선택한 물리 모형과 아직 필요한
미시적 사상을 분리한다. 관측 적합성은 정리의 진위를 결정하지 않는다.
명시한 작용 안에서 일관된 toy/EFT는 관측 자료가 없어도 보존한다.

우주론의 정밀값·과거 재현값·런타임 호환값과 Claim ID는
[우주론 판본·주장 원장](3_상수/00_우주론_원장.md)에서 관리한다.

## 1. 출처와 기본 규약

| 표지 | 뜻 |
|---|---|
| **[정의]** | 대상·기호·정의역 |
| **[정리]** | 적힌 전제에서 증명된 명제 |
| **[공리]** | 모형·가지·경계조건·물리 사상의 선택 |
| **[산출]** | 정리와 공리의 직접 계산 |
| **[경험식]** | 자료·보정·유효계수를 사용하는 관계 |
| **[미완성]** | 작용·사상·증명 또는 자료가 더 필요한 항목 |
| **[예측]** | 입력과 판정 기준을 미리 고정한 독립 관측량 |

**[정의]** 장론 절에서는 \(c=\hbar=1\), metric 부호
\((-+++)\)를 쓰고 \(M_{\rm Pl}^{-2}=8\pi G\)인 reduced Planck mass를
사용한다. 4차원에서

\[
[d^4x]=-4,\quad [\phi]=[H]=1,\quad [R]=2,\quad [M_{\rm Pl}]=1.
\]

따라서 지수·로그·확률의 인자는 무차원이어야 하고, 작용은 무차원이다.

## 2. 서로 다른 세 대상

**[정의]**

1. \(\phi(x)\): 시공간 위의 독립 실수 스칼라장.
2. \(\mathcal H_\gamma=\delta^2S[\gamma]\): 경로 또는 장배치에서의
   이차 변분 연산자.
3. \(R(x)\): 시공간 metric의 Ricci scalar.

probe \(u\)와 경계조건을 고정했을 때만

\[
\Phi_H[\gamma;u]
=\frac{\langle u,\mathcal H_\gamma u\rangle}{\langle u,u\rangle}
\]

를 스칼라 Hessian readout으로 정의한다. \(\phi\), \(\Phi_H\), \(R\)는
자동으로 같지 않다.

**[정리]** 일반 좌표에서 보통 Hessian은 off shell 장공간 텐서가 아니다.
공변 Hessian을 쓰려면

\[
\nabla_A\nabla_BS
=\partial_A\partial_BS-\Gamma^C{}_{AB}\partial_CS
\]

처럼 field-space connection을 지정해야 한다. gauge theory에서는
gauge fixing, ghost, measure와 renormalization도 별도로 필요하다.

**[공리]** CE의 활성 장론 branch는 Hessian을 물질장으로 동일시하지 않고
독립장 \(\phi\)를 채택한다. Hessian은 그 작용의 saddle과 작은 요동을
분석하는 연산자로만 쓴다.

## 3. 공변 \(Z_2\) singlet-portal EFT

### 3.1 작용

**[공리: EFT branch]** 4차원 globally hyperbolic 시공간과
\(\phi\mapsto-\phi\) 대칭을 택하고

\[
S_{\rm EFT}=\int d^4x\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\Lambda_0
-\frac12(\nabla\phi)^2
+\mathcal L_{\rm SM}^{\rm kin+gauge+Yuk}
-V(H,\phi)
\right]
\]

를 정의한다. potential은

\[
V(H,\phi)
=V_H(H)+\frac12m_\phi^2\phi^2
+\frac{\lambda_\phi}{4}\phi^4
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H .
\]

모든 항의 질량차원은 4다. \(\xi,\lambda_\phi,\lambda_{H\phi}\)는
무차원이고 \(\Lambda_0\)의 차원은 4다.

이것은 CE를 표현할 수 있는 일관된 저에너지 모형의 정의다. 계수의 수치나
자연의 branch 선택을 코어 정리가 고정한다는 뜻은 아니다.

### 3.2 안정성 정의역

**[공리]** EFT를 다음 영역으로 제한한다.

\[
M_{\rm Pl}^2-\xi\phi^2>0,\qquad
\lambda_H>0,\qquad
\lambda_\phi>0,\qquad
\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}.
\]

첫 조건은 Jordan-frame 중력 kinetic 부호가 바뀌지 않게 한다.
\(\lambda_H,\lambda_\phi>0\) 아래 마지막 엄격한 부등식은 큰 장값에서
portal quartic이 coercive이기 위한 필요충분조건이다.

**[정리]** 위 quartic 조건에서 potential은 큰
\((H^\dagger H,\phi^2)\)에 대해 아래로 유계다.
[증명](참조/핵심_정리_증명.md#portal-boundedness)

### 3.3 운동방정식과 보존법칙

**[산출]** \(\phi\) 운동방정식은

\[
\Box\phi-\left(m_\phi^2+\xi R
+\lambda_{H\phi}H^\dagger H\right)\phi
-\lambda_\phi\phi^3=0.
\]

nonminimal 항을 물질 쪽에 둔 stress tensor는

\[
T_{\mu\nu}^{\phi}
=\nabla_\mu\phi\nabla_\nu\phi
-g_{\mu\nu}\!\left[\frac12(\nabla\phi)^2+V_\phi
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H\right]
+\xi\left(G_{\mu\nu}+g_{\mu\nu}\Box-\nabla_\mu\nabla_\nu\right)\phi^2 .
\]

portal이 있으면 \(T^\phi_{\mu\nu}\)와 Higgs stress는 따로 보존되지 않을
수 있지만 총 stress는 완결된 운동방정식 위에서 보존된다.

**[정리]**

\[
\nabla^\mu T_{\mu\nu}^{\rm total}=0
\]

은 미분동형사상 불변 작용의 Noether 항등식과 모든 장의 on-shell
방정식에서 따른다.
[증명](참조/핵심_정리_증명.md#noether-stress)

### 3.4 \(Z_2\) 진공과 질량

**[산출]** \(H^\dagger H=v^2/2\), 배경곡률 \(R=\bar R\)에서
\(\langle\phi\rangle=0\)인 \(Z_2\)-보존 branch의 tree-level 곡률질량은

\[
m_{\phi,\rm eff}^2
=m_\phi^2+\xi\bar R+\frac{\lambda_{H\phi}}2v^2.
\]

\(m_{\phi,\rm eff}^2>0\)이면 \(\phi=0\)은 국소 안정하다. 정확한
\(Z_2\) 때문에 이 branch에는 \(h\)-\(\phi\) 선형 질량혼합이 없고 가장
가벼운 \(\phi\) 양자는 안정하다. 질량과 portal 계수는 자유 EFT
매개변수이며 특정 GeV 값은 산출되지 않는다.

## 4. Euclidean scalar--Higgs 경로적분의 닫힌 cutoff 모형

**[공리]** gauge·ghost·fermion을 제외한 scalar--Higgs bosonic
truncation에서 \(\xi=0\)과 고정 Euclidean 배경, 유한 격자와 양의
Euclidean kinetic operator를 택해

\[
S_{E,N}(\varphi)
=\frac12\varphi^TK_N\varphi+\sum_x a^4V(H_x,\phi_x)
\]

를 정의한다. 3.2절의 엄격한 quartic 조건과 유한한 격자 부피를 둔다.
이 다항식 작용은 \(\mathbb R^N\)에서 연속이고 유한하므로 measurable이다.

**[정리]**

\[
Z_N=\int_{\mathbb R^N}e^{-S_{E,N}(z)}\,d^Nz
\]

는 유한하고 모든 다항식 모멘트가 존재한다.
[증명](참조/핵심_정리_증명.md#finite-lattice-measure)

이 결과는 유한 cutoff에서의 확률측도를 완결한다. continuum limit,
reflection positivity, renormalized parameter의 cutoff 독립성과
Lorentzian 재구성은 **[미완성]**이다.

### 4.1 Laplace 근사

**[정리: 반고전 근사]** 유한 격자에서 고립된 비퇴화 국소 최소점
\(\varphi_j\)의 양의 Hessian을 \(\mathcal H_j\)라 하자. 작은
무차원 매개변수 \(\varepsilon=\hbar/S_*\)에 대한 국소 Laplace 기여는

\[
Z_j\simeq
a_j(\varphi_j)
e^{-S_E[\varphi_j]/\hbar}
\left(\det\frac{\mathcal H_j}{2\pi\hbar}\right)^{-1/2}.
\]

따라서 두 최소점의 비는 작용차뿐 아니라 determinant와 국소 measure
prefactor의 비를 포함한다.
[증명과 정확한 정의역](참조/핵심_정리_증명.md#laplace-saddle)
\(e^{-1}\)은 \(\Delta S_E/\hbar=1\)과 전체 prefactor 비 1을 함께
선택한 특수 산출이다. 영모드·gauge orbit·음의 모드가 있는 saddle에는
집단좌표, gauge fixing 또는 지정한 thimble이 더 필요하다.

## 5. Hodge 구조와 공간 branch

**[정리]** 방향과 양의 내적을 가진 \(d\)차원 공간에서

\[
*:\Lambda^2V^*\to\Lambda^{d-2}V^*
\]

이고, 이 Hodge 사상이 2-form과 1-form을 추가 지표 없이 동형으로
연결하는 차원은 \(d=3\)이다.
[증명](참조/핵심_정리_증명.md#hodge-closure)

**[공리]** 이 양의 비자명 branch를 실제 공간 차원으로 읽는다.

다음은 서로 독립인 모형 선택이다.

\[
N_c=d,\qquad N_w=d-1,\qquad N_{\rm gen}=d.
\]

이 선택을 \(d=3\)에 적용하면 표준모형의 수와 일치하지만 Hodge 정리가
내부 게이지군, 표현, hypercharge와 세대 수를 유도하지는 않는다.

## 6. 전자약 혼합 통계량

**[공리]** one-Higgs-doublet 표준모형을 외부 장론 sector로 채택한다.
중성 보손 질량행렬은

\[
\mathcal M_0^2=\frac{v^2}{4}
\begin{pmatrix}
g^2&-gg'\\
-gg'&g'^2
\end{pmatrix},
\qquad
M_Z^2=\frac{v^2}{4}(g^2+g'^2).
\]

**[산출]** \(\tan\theta_W=g'/g\)이면 정규화된 비대각 진폭과 그 제곱은

\[
\frac{|\mathcal M^2_{W^3B}|}{M_Z^2}
=\sin\theta_W\cos\theta_W,\qquad
\delta=\sin^2\theta_W\cos^2\theta_W.
\]

**[정리]** \(0\leq\delta\leq1/4\)다.
[증명](참조/핵심_정리_증명.md#delta-bound)

**[공리: CE readout]** 이 기저가 물리적으로 지정된 중성 sector라는
조건 아래 \(\delta\)를 하나의 무차원 channel weight로 사용한다.
진폭의 제곱만으로 양의 고전 jump rate가 되지는 않는다.

**[경험식]**

\[
\sin^2\theta_W=4\alpha_s^{4/3}
\]

은 같은 scheme과 scale에서 시험할 수 있는 매개변수 관계다. 게이지
작용과 RG 흐름에서 산출되지 않았으므로 공리계의 정리가 아니다.

## 7. 다형 Poisson 재귀 모형

### 7.1 확률모형

**[공리: stochastic toy branch]** \(A_{ij}\geq0\)를 type \(i\)
개체 하나가 다음 세대에 만드는 type \(j\) 자손 수의 평균으로 정의하고,
각 자손 수를 독립 Poisson 변수로 둔다.

**[정리]** 소멸확률 벡터의 최소 고정점은

\[
q_i=\exp\!\left[-\sum_jA_{ij}(1-q_j)\right]
\]

의 \([0,1]^n\) 최소해다. 모든 행합이 \(D\)이면
\(\boldsymbol q=q_{\rm ext}\boldsymbol1\)로 정확히 축약되고

\[
q_{\rm ext}=e^{-D(1-q_{\rm ext})}.
\]

\(D>1\)에서 최소해는

\[
q_{\rm ext}=-\frac1D W_0(-De^{-D}),\qquad
Dq_{\rm ext}<1
\]

이며 균일 반복에 국소 안정하다.
[단일형 증명](참조/핵심_정리_증명.md#poisson-fixed-point),
[다형 축약 증명](참조/핵심_정리_증명.md#multitype-poisson)

### 7.2 CE 균일 sector

**[공리]** \(B\geq0\), \(B\boldsymbol1=\boldsymbol1\)인 row-stochastic
전달 연산자를 택하고

\[
A=dI+\delta B
\]

로 둔다.

**[산출]**

\[
A\boldsymbol1=(d+\delta)\boldsymbol1,\qquad
D_{\rm eff}=d+\delta.
\]

따라서 이 toy family의 균일 부분공간에서는 기존 스칼라 고정점이 정확히
닫힌다. 이는 임의의 CE+SM Hessian이 \(A\)와 같다는 주장이 아니다.

### 7.3 양자장론에서 확률모형으로 가는 문

양자 진폭은 복소수이고 간섭한다. \(A\geq0\)를 얻으려면 최소한 다음
사슬이 필요하다.

1. 완결된 CE+SM 작용과 진공
2. 환경을 포함한 reduced dynamics
3. 양의 Kossakowski 행렬과 완전양성 사상
4. population sector의 폐쇄
5. Markov·독립 증가량 근사
6. offspring genealogy와 \(A_{ij}\)의 식별

이 사슬은 **[미완성]**이다. 따라서 \(\mathcal H\), 질량행렬 또는 부호
있는 진폭을 성분별로 제곱해 \(A\)라고 부르지 않는다.

## 8. 확률에서 물리량으로 가는 readout

**[정리]** 연속 곱적 readout \(I:(0,1]\to(0,1]\)가
\(I(xy)=I(x)I(y)\), \(I(1)=1\)을 만족하면

\[
I(x)=x^c,\qquad c\geq0.
\]

[증명](참조/핵심_정리_증명.md#multiplicative-readout)

**[공리]** 선형 확률 readout branch에서는 \(c=1\)을 택한다.

**[공리: 과거 우주론 경계모형 `C-B-LEGACY-01`]**

\[
\Omega_b:=\frac{\rho_b(t_0)}{\rho_{\rm crit}(t_0)}
\]

를 현재 바리온 밀도분율로 정의하고

\[
q_{\rm ext}\mapsto\Omega_b
\]

를 택할 수 있다. 이는 확률과 에너지 밀도 사이의 새로운 사상이며
분지과정 정리의 결론이 아니다. 새 경로에서는 소멸 조건부 조성
\(Dq_{\rm ext}\)와 공변 전이 면의 \(1/D\)를 서로 독립적으로 얻어야 한다.
species current, total yield, stress tensor와 섭동 방정식의 공동 유도는
**[미완성]**이다.

## 9. Euclidean 보조축

**[정의]** 양의 자기수반 연산자 \(A\)에 대해

\[
\Phi(\chi)=e^{-\chi A}\Phi(0),\qquad \chi\geq0
\]

를 Euclidean 또는 Schwinger 보조 흐름으로 정의한다.
\(\chi A\)가 무차원이도록 \([\chi]=[A]^{-1}\)로 둔다.

**[정리]** 이 흐름은 수축 semigroup이고 \(A\)-고유값 \(\omega\)인
모드를 \(e^{-\chi\omega}\)로 억제한다.
[증명](참조/핵심_정리_증명.md#euclidean-semigroup)

이 정식화는 기존의 “허수 연산” 직관에서 수학적으로 보존되는 부분이다.
\(\chi\)를 물리적 두 번째 시간 또는 독립 장으로 승격하는 것은 별도
**[공리]**이며, 인과성·unitarity·초기값 문제를 따로 제시해야 한다.

## 10. 우주론 branch

### 10.1 canonical scalar

**[공리]** 3절에서 \(\xi=0\)인 최소 결합 branch를 택한다. 또한
\(\lambda_{H\phi}=0\)으로 분리하거나 Higgs를 일관되게 적분해 국소
\(V_{\rm eff}(\phi)\)만 남겨 scalar와 다른 sector 사이 에너지 교환이
없는 경우에 \(\phi\)를 평탄 FLRW의 균일 canonical scalar로 제한한다.

**[정리]**

\[
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,\qquad
\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0.
\]

\(\rho_\phi>0\)이면 \(w_\phi\geq-1\)이다.
[증명](참조/핵심_정리_증명.md#canonical-scalar-flrw)

따라서 이 최소 branch에서 phantom crossing을 주장하려면 kinetic 구조,
추가 자유도 또는 modified gravity를 바꿔야 한다.

### 10.2 상수 진공항

**[정리]** \(-\int\sqrt{-g}V_0\)는

\[
T_{\mu\nu}^{(0)}=-V_0g_{\mu\nu},\qquad p_0=-\rho_0
\]

를 준다.
[증명](참조/핵심_정리_증명.md#vacuum-stress)

\(V_0\)의 관측 절대값과 radiative stability는 **[미완성]**이다.

### 10.3 inflation 선택지

**[공리: 별도 branch]** Einstein frame의 최소 결합 정준 단일장,
\(V_0>0\), scalar-dominated 배경, adiabatic Bunch--Davies 초기상태와
tree-level leading potential slow-roll을 택하고

\[
V(\phi)=V_0
\left(1-e^{-\sqrt{2/3}\phi/M_{\rm Pl}}\right)^2
\]

를 택하면 slow-roll에서

\[
n_s=1-\frac2N+O(\log N/N^2),\qquad
r=\frac{12}{N^2}+O(\log N/N^3)
\]

를 얻는다.
[계산](참조/핵심_정리_증명.md#starobinsky-slow-roll)

이 branch는 quartic \(Z_2\) portal 진공과 자동으로 같은 모형이 아니다.
두 potential 사이의 matching을 제시하지 않는 한 별개 모형으로 관리한다.

## 11. flavor와 질량

**[정리]** 임의의 양의 질량 고유값과 unitary CKM 행렬은 Dirac Yukawa
행렬로, 임의의 양의 중성미자 질량과 unitary PMNS 행렬은 복소 대칭
Majorana 행렬로 일관되게 실현할 수 있다.
[구성 증명](참조/핵심_정리_증명.md#flavor-realization)

이로써 CKM·PMNS 경험식을 장론에 넣는 존재 문제는 닫힌다. 그러나 왜
그 특정 각과 질량이 선택되는지, RG 아래 유지되는지와 매개변수 감소는
**[미완성]**이다.

**[정리]** Koide \(Q=2/3\)은 제곱근 질량벡터와 민주축의 각이
\(45^\circ\)라는 조건과 동치다.
[증명](참조/핵심_정리_증명.md#koide-angle)

이 대수적 부분은 보존한다. 실제 질량을 선택하는 flavor symmetry와
potential은 별도 모형이 필요하다.

## 12. 보존된 원리와 남은 다리

| 항목 | 현재 지위 |
|---|---|
| 지수형 생존 character | **[정리]** |
| Hodge \(2\leftrightarrow1\) 폐쇄의 \(d=3\) | **[정리]** |
| 공변 \(Z_2\) portal EFT | **[공리]**, 내부적으로 일관된 모형 |
| scalar--Higgs bosonic 유한 격자 Euclidean 측도 | **[정리]** |
| on-shell 총 stress 보존 | **[정리]** |
| SM 중성 혼합 \(\delta\) | **[산출]** |
| 다형 Poisson 최소 소멸확률 | **[정리]** |
| \(A=dI+\delta B\) 균일 toy sector | **[공리]**과 **[산출]** |
| quantum-to-branching | **[미완성]** |
| 과거 \(q_{\rm ext}\mapsto\Omega_b\) 경계모형 | **[공리]** |
| conditioned \(Dq\)와 전이 면 \(1/D\)의 합성 | **[산출]**, 공동 물리 실현은 **[미완성]** |
| CKM·PMNS·질량 수치 관계 | **[경험식]** 또는 **[미완성]** |
| 진공에너지 절대값과 재규격화 | **[미완성]** |

CE의 이론물리 핵심은 서로 다른 모형 층을 등식으로 뭉개는 데 있지 않다.
각 층을 일관된 작용·확률공간으로 닫고, 층 사이 사상을 별도 공리와
검증 대상으로 드러내는 데 있다.
