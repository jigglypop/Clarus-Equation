# 4. 암흑에너지 모형: FLRW, canonical scalar, scalar--tensor branch

이 장은 암흑에너지의 절대크기를 무입력으로 유도하지 않는다. 일반상대론의
FLRW 방정식, canonical scalar와 명시적인 scalar--tensor 작용에서 닫히는
조건부 결과를 정리한다. reduced Planck mass는
$M_{\rm Pl}^{-2}=8\pi G$로 정의한다.

독자는 일반상대론의 Einstein 방정식과 균일·등방 우주론의 기초를 안다고
가정한다. 4.1절은 모든 뒤 절이 공유하는 배경 방정식을 만들고, 4.2절은
최소 결합 canonical scalar의 강한 경계를 보인다. 그 경계를 넘는 모형은
4.3–4.5절의 별도 scalar--tensor branch에서만 다루며, 4.6절은 관측
입력과 아직 비어 있는 CE 사상을 분리한다. 공통 작용·기호는
[공리계](../axium.md)와 [master action](06_Master_Action_Universal_Derivation.md)을
따른다.

## 4.1 FLRW 배경

암흑에너지의 후보를 비교하려면 먼저 모든 후보가 공유하는 균일·등방
배경과 가속 조건을 고정해야 한다. 이 절의 방정식은 source의 절대 scale을
정하지 않으며, 뒤 branch의 공통 운동학만 제공한다.

**[공리] 우주론 branch:** 동질·등방 metric
$$
ds^2=-dt^2+a(t)^2\gamma_{ij}dx^idx^j,
\qquad {}^{(3)}R=6k
$$
와 perfect-fluid total stress tensor
$T^\mu{}_\nu=\operatorname{diag}(-\rho,p,p,p)$를 택한다.

**[정의]** $H:=\dot a/a$다.

**[정리]** Einstein 방정식은
$$
H^2+\frac{k}{a^2}=\frac{\rho}{3M_{\rm Pl}^2},
$$
$$
\dot H-\frac{k}{a^2}
=-\frac{\rho+p}{2M_{\rm Pl}^2},
\qquad
\frac{\ddot a}{a}
=-\frac{\rho+3p}{6M_{\rm Pl}^2}
$$
를 준다. Bianchi identity와 matter 방정식에서
$$
\dot\rho+3H(\rho+p)=0
$$
가 따른다.

**[산출]** 팽창 중인 branch에서 $\ddot a>0$의 필요충분조건은
$\rho+3p<0$이다. 이는 source의 equation of state에 관한 조건이며
$\rho$의 절대값을 정하지 않는다.

## 4.2 canonical scalar

배경 가속 조건만으로 source의 미시 구조는 정해지지 않는다. 가장 작은
동적 후보인 canonical scalar를 먼저 택하면, 어떤 한계가 작용 자체에서
따르는지 명확해진다.

**[공리] 모델 선택:**
$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\nabla\phi)^2-V(\phi)
\right]+S_m
$$
를 택하고 $\phi=\phi(t)$로 제한한다.

**[정리]**
$$
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,
$$
$$
\ddot\phi+3H\dot\phi+V_{,\phi}=0,
\qquad
\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0
$$
이다.

**[정리]** $\rho_\phi>0$이면
$$
w_\phi+1
=\frac{\rho_\phi+p_\phi}{\rho_\phi}
=\frac{\dot\phi^2}{\rho_\phi}\ge0.
$$
따라서 positive-kinetic canonical single scalar는 $w=-1$ 아래로
넘지 않는다. phantom crossing에는 추가 자유도, noncanonical kinetic
구조 또는 modified gravity가 필요하다.

**[정리]** $V_{,\phi}(\phi_0)=0$, $\dot\phi=0$인 상수 branch는
$$
T_{\mu\nu}^{(\phi_0)}=-V(\phi_0)g_{\mu\nu},
\qquad p_\phi=-\rho_\phi
$$
를 준다. $V(\phi_0)>0$이면 cosmological constant와 같은 배경 source다.

**[미완성]** $V(\phi_0)$의 관측 절대값, radiative stability와
초기조건을 CE가 고정하는 메커니즘은 없다.

## 4.3 scalar--tensor 작용

canonical branch의 $w\geq-1$ 경계는 positive kinetic 단일장이라는
전제의 결과다. 따라서 중력 결합까지 바꾸려면 새 작용과 안정 정의역을
명시해야 한다.

**[공리] 별도 branch:**
$$
S_{\rm ST}=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}F(\phi)R
-\frac12K(\phi)(\nabla\phi)^2
-V(\phi)
\right]+S_m[g,\Psi]
$$
를 택한다. $F,K$는 무차원이고 $[\phi]=1$이다.

**[정리]** 변분하면
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

**[공리] 안정 영역:** $F>0$이고 Einstein-frame scalar kinetic
coefficient
$$
K_E(\phi)
=\frac{K}{F}
+\frac{3M_{\rm Pl}^2}{2}
\left(\frac{F_{,\phi}}{F}\right)^2
$$
가 양수인 field domain만 사용한다.

**[정리]** $g^E_{\mu\nu}=F g_{\mu\nu}$의 Weyl 변환은 이 영역에서
중력항을 Einstein--Hilbert 꼴로 만들고 scalar kinetic coefficient를
$K_E$로 바꾼다. matter는 일반적으로 Einstein frame에서 $\phi$와
결합하므로 서로 다른 frame의 중간 변수를 직접 관측량처럼 비교하지 않는다.

## 4.4 scalar--tensor FLRW 방정식

작용을 택했으므로 이제 그 추가 결합이 균일 우주에서 Friedmann 항을
어떻게 바꾸는지 계산할 수 있다. 다음 식은 같은 작용과 homogeneous scalar
전제에서만 쓰이며, 상수장 해에는 scalar 방정식도 별도로 필요하다.

**[산출]** 평탄하지 않을 수 있는 FLRW와 homogeneous $\phi$에서
$$
3M_{\rm Pl}^2F
\left(H^2+\frac{k}{a^2}\right)
=\rho_m+\frac12K\dot\phi^2+V
-3M_{\rm Pl}^2H\dot F,
$$
$$
-2M_{\rm Pl}^2F
\left(\dot H-\frac{k}{a^2}\right)
=\rho_m+p_m+K\dot\phi^2
+M_{\rm Pl}^2(\ddot F-H\dot F).
$$
상수 $\phi=\phi_0$, $F(\phi_0)=F_0>0$이면 derivative 항이 사라지고
유효 Planck mass는 $M_{\rm Pl}^2F_0$가 된다. 다만 이것이 full
상수장 해가 되려면 scalar equation도 pointwise로
$$
\frac{M_{\rm Pl}^2}{2}F_{,\phi}(\phi_0)R
-V_{,\phi}(\phi_0)=0
$$
을 만족해야 한다.

**[미완성]** 원하는 $H(a)$를 먼저 고른 뒤 독립적인 $\mu(a,k)$를
추가하는 것은 일반적으로 하나의 공변 작용과 자동으로 정합하지 않는다.
$F,K,V$, matter coupling과 초기조건을 함께 고정해야 한다.

## 4.5 선형 성장의 조건부 식

배경 방정식이 주어져도 구조 성장률은 자동으로 정해지지 않는다. 다음 식은
명시한 섭동 근사 안에서 background와 growth를 연결하며, 그 근사 밖에서는
사용하지 않는다.

**[공리] 섭동 branch:** Jordan frame에서 보존되는 pressureless matter,
sub-horizon quasi-static regime와 negligible matter anisotropic stress를
가정한다. 고정 reference wave number $k_{\rm ref}>0$와
$q:=k/k_{\rm ref}$, 고정 reference Planck mass $M_{\rm ref}>0$를
도입한다.

**[정의]** scale factor $a$는 무차원이고 prime은 $d/d\ln a$,
$D(a,q)$는 linear matter growth factor다. 또한
$$
\Omega_m(a):=\frac{\rho_m}{3M_{\rm ref}^2H^2},
\qquad
\mu(a,q):=\frac{G_{\rm growth}(a,q)}
{(8\pi M_{\rm ref}^2)^{-1}}
$$
로 정의한다. 여기서 $G_{\rm growth}$는 quasi-static Poisson
constraint의
$-k^2\Psi/a^2=4\pi G_{\rm growth}\rho_m\delta_m$로 정한 계수다.

**[정리]** 위 근사에서
$$
D''+\left(2+\frac{H'}H\right)D'
-\frac32\Omega_m(a)\mu(a,q)D=0.
$$
$M_{\rm ref}=M_{\rm Pl}$, $\mu=1$이면 minimally coupled GR의
표준 식이다. $\mu$는 무차원이고
scalar--tensor branch에서는 action의 constraint equation에서 계산해야 한다.

**[공리] toy parameterization:**
$$
\mu(a,q)=1+\epsilon\,S(a,q),\qquad
0\le S(a,q)\le1
$$
를 자료 비교용 family로 둘 수 있다. $\epsilon$과 $S$는 무차원이다.

**[경험식]** $\epsilon$, $S$ 또는 초기 normalization을 growth data로
고르면 그 결과는 fit이다. 같은 data point를 다시 독립 예측으로 세지 않는다.

## 4.6 절대 scale과 관측

조건부 배경·성장식을 얻어도 실제 우주의 수치는 작용계수와 관측 입력 없이는
결정되지 않는다. 이 절은 외부 입력으로 계산할 수 있는 것과 CE가 아직
제공하지 못하는 matching을 분리한다.

**[공리] 외부 입력:** 실제 우주 계산에는 $H_0$, matter·radiation
density, neutrino sector, recombination과 survey likelihood를 데이터
snapshot과 함께 공급한다.

**[산출]** 지정한 $F,K,V$, 초기조건과 외부 입력을 넣어 얻은
$H(a),D(a,k)$는 그 모형의 조건부 수치해다.

**[미완성]** 다음은 현재 닫히지 않는다.

1. vacuum contribution과 관측 dark-energy scale 사이의 radiatively stable
   matching,
2. CE 확률변수 또는 Hessian readout에서 $F,K,V$로 가는 action-level map,
3. background·CMB·BAO·lensing·growth를 한 parameter set으로 잇는 likelihood,
4. 자료를 보기 전에 고정한 구별 가능한 관측 예측.

## 4.7 자기비동일성 시계장의 조건부 암흑부문

앞 절들은 일반적인 암흑에너지 장을 다뤘다. 이제 “자신이 직전의 자신과 다름이
반복된다”는 생각을 하나의 시계장으로 옮기고, 같은 장의 두 읽기값이 어떤
조건에서 암흑에너지형과 암흑물질형으로 갈라지는지 계산한다. 이 절은 비선택
양자경로가 실제 암흑부문이라는 결론을 전제로 하지 않는다. 먼저 닫힌 유효작용을
세우고, 그 작용에서 성립하는 정리와 아직 외부 입력인 미시 사상을 분리한다.

**[정의]** 시간 차원의 시계장 $T$에 대해

$$
X:=-\frac12g^{\mu\nu}\nabla_\mu T\nabla_\nu T,\qquad
\delta:=\frac{X}{X_*}-1
\tag{19}
$$

로 둔다. 자연단위에서 $[T]=-1$이므로 $X$, $X_*$와 $\delta$는 무차원이다.
균일 FLRW에서는 $X=\dot T^2/2$다. $X>0$은 변화가 계속됨을 나타내지만
$\dot T>0$과 $\dot T<0$을 가르지 못하므로 시간 화살은 별도 초기자료다.

**[공리: 모델 선택]** 다음 한 장의 비정준 작용을 채택한다.

$$
S_T=\int d^4x\sqrt{-g}\,P(T,X),
\tag{20}
$$

$$
P(T,X)=\rho_\infty\left[
\frac{\kappa}{2}\delta^2-\left(1-e^{-\Gamma T}\right)
\right].
\tag{21}
$$

$\rho_\infty$는 에너지밀도, $\Gamma$는 역시간이고 $\kappa$와 $X_*$는
무차원이다. $1-e^{-\Gamma T}$는 무차원 기회 누적량이며 그 자체가 에너지가
아니다. 식 (21)이 에너지 차원을 갖는 것은 독립 척도 $\rho_\infty$를
공리로 넣었기 때문이다.

작용을 $T$에 대해 변분하면

$$
\nabla_\mu(P_X\nabla^\mu T)+P_T=0
\tag{22}
$$

을 얻는다. 계량 변분으로 얻는 응력텐서는

$$
T_{\mu\nu}^{(T)}
=P_X\nabla_\mu T\nabla_\nu T+Pg_{\mu\nu},
\qquad
\rho_T=2XP_X-P,\qquad p_T=P.
\tag{23}
$$

따라서 $\delta$로 정리한 정확한 분해는

$$
\rho_V=\rho_\infty(1-e^{-\Gamma T}),\qquad p_V=-\rho_V,
\tag{24}
$$

$$
\rho_K=\rho_\infty\left(2\kappa\delta+\frac32\kappa\delta^2\right),
\qquad
p_K=\frac12\kappa\rho_\infty\delta^2
\tag{25}
$$

다. 운동학적 성분의 상태방정식과 음속은

$$
w_K=\frac{\delta}{4+3\delta},\qquad
c_s^2=\frac{\delta}{2+3\delta},
\tag{26}
$$

이고 유령이 없기 위한 계수는

$$
P_X+2XP_{XX}
=\frac{\kappa\rho_\infty}{X_*}(2+3\delta)
\tag{27}
$$

다. 따라서 $\kappa>0$, $X_*>0$, $\delta\ge0$에서 두 미분 수준의 유령·기울기
조건을 통과하며, $0<\delta\ll1$이면 $\rho_K$가 압력 없는 물질에 가깝다.
큰 $\delta$에서는 복사형으로 이동하므로 전 적색편이에서 곧바로 차가운
암흑물질이라고 부를 수 없다.

## 4.8 영 전류 반례와 보존 장부

앞 절의 분해가 양의 암흑물질형 성분을 보장하려면 초기 전류의 부호를 따로
검사해야 한다. 균일장 전류를 $J=P_X\dot T$라 하면 식 (22)는

$$
\frac{d}{dt}(a^3J)
=-a^3\rho_\infty\Gamma e^{-\Gamma T}
\tag{28}
$$

을 준다. 따라서 $J_i=0$이면 충분히 가까운 미래에

$$
J<0,\qquad \delta<0,\qquad c_s^2<0
\tag{29}
$$

가 된다. 닫힌 한 장 이론이 영 전류에서 증가하는 암흑에너지형 성분과 양의
암흑물질형 재고를 함께 저절로 만든다는 부모 주장은 식 (29)의 완전 반례로
제거된다.

조건부로 살아남는 해는 초기 초곡면 $\Sigma_i$에서

$$
\left(T,a^3P_X\dot T\right)\big|_{\Sigma_i}
=\left(0,\Pi_{\rm F}\right),\qquad \Pi_{\rm F}>0
\tag{30}
$$

을 별도 맞춤자료로 채택한 해다. 적분형은

$$
a^3(t)J(t)=\Pi_{\rm F}
-\int_{t_i}^{t}a^3(s)\rho_\infty\Gamma
e^{-\Gamma T(s)}\,ds
\tag{31}
$$

이므로 오른쪽이 양수인 구간에서만 안정 가지가 유지된다. 에너지 교환은

$$
\dot\rho_K+3H(\rho_K+p_K)=-\dot\rho_V,\qquad
\dot\rho_V>0
\tag{32}
$$

로 닫힌다. 증가한 $\rho_V$는 초기 $\rho_K$ 재고에서 지불된다.
$\Pi_{\rm F}$를 별도 에너지밀도로 다시 더하면 같은 재고를 두 번 세는 오류다.

## 4.9 고전 경계항 단독 경로의 완전 반례

식 (30)의 $\Pi_{\rm F}$를 0차원 기록에서 유도하려면 경계 변분을 확인해야 한다.
차원부터 고정하면

$$
[J]=[\Pi_{\rm F}]=4,\qquad [B]=3,\qquad
\Pi_{\rm F}=\Lambda_\Pi^4f(\mu_F,C_{\rm self},\ldots)
\tag{33}
$$

이다. $\Lambda_\Pi$는 질량차원 1의 독립 맞춤 척도다. 무차원 0차원 정보만으로
이 척도가 생기지는 않는다.

초기 경계의 벌크 변분을 부호 규약에 맞춰 쓰면

$$
\delta S_{\rm bulk}\big|_{\Sigma_i}
=\int_{\Sigma_i}d^3x\sqrt h\,J_i\,\delta T_i
\tag{34}
$$

다. 국소 경계밀도 $B=-\Pi_{\rm F}T$를 더하고 $T_i$를 자유롭게 변분하면

$$
\delta(S_{\rm bulk}+S_\Sigma)\big|_{\Sigma_i}
=\int_{\Sigma_i}d^3x\sqrt h\,
(J_i-\Pi_{\rm F})\delta T_i=0
\tag{35}
$$

에서 $J_i=\Pi_{\rm F}$를 얻는다. 그러나 정확한 디리클레 조건 $T_i=0$을
동시에 고정하면 허용 변분은 $\delta T_i=0$이므로 식 (35)는 운동량 조건을
전혀 주지 않는다. 경계 승수 $\lambda T_i$를 넣어도

$$
J_i=\Pi_{\rm F}-\lambda
\tag{36}
$$

가 되어 자유 승수가 전류를 흡수한다. 그러므로 고전 국소 경계항 하나가
정확한 $T_i=0$, 양의 전류, 그 수치와 시간 화살을 모두 유도한다는 주장은
성립하지 않는다.

공간 타입에도 별도 문제가 있다. 유한한 엄격한 0차원 점원은
$\delta^{(3)}(\mathbf x-\mathbf X_j)$ 지지집합을 가지므로 공간 상수인
FLRW 초기자료와 같지 않다. 균일한 초기면, 공간 분포와 거친갈기 척도는
추가 공리다.

## 4.10 폐쇄시간경로 초기상태 후보

고전 경계항 반례 뒤에도 부모 목표는 남는다. 즉, 기록과 저장소를 포함한
양의 초기 양자상태가 평균 전류를 준비하고 총 에너지·운동량을 보존할 수
있는지를 시험할 수 있다. 폐쇄시간경로 변수를

$$
T_r=\frac{T_++T_-}{2},\qquad T_a=T_+-T_-
\tag{37}
$$

로 두고, 초기 경계 영향작용의 후보를

$$
S_{{\rm IF},\Sigma}
=\int_{\Sigma_i}d^3x\sqrt h
\left[-\Pi_{\rm F}T_a-K_\Sigma T_aT_r\right]
+\frac{i}{2}\int_{\Sigma_i\times\Sigma_i}
d^3x\,d^3y\sqrt{h_xh_y}\,
T_a(x)N_\Sigma(x,y)T_a(y)
\tag{38}
$$

로 둔다. 물리 극한에서 $T_a$를 변분하면

$$
\langle J_i\rangle
=\Pi_{\rm F}+K_\Sigma\langle T_i\rangle
\tag{39}
$$

을 얻는다. 평균 앵커 $\langle T_i\rangle=0$을 준비하면
$\langle J_i\rangle=\Pi_{\rm F}>0$이다. 이는 두 양의 분산까지 0으로
고정한다는 뜻이 아니다. 초기 밀도행렬은 양의 정규화 상태여야 하고
$N_\Sigma$는 양의 준정부호여야 한다.

실제 일방향 응답을 주장하려면 초기상태만으로 부족하다. 벌크 영향작용에는
지연 응답핵 $D_R$와 잡음핵 $N$을 넣고

$$
D_R(x,y)=0\quad\bigl(x\notin J^+(y)\bigr),
\qquad N\succeq0
\tag{40}
$$

을 요구해야 한다. 축약된 시계장만 보면 저장소와 에너지를 교환하므로
일반적으로 $\nabla_\mu T_T^{\mu\nu}=Q^\nu$다. 물리적으로 닫힌 장부는

$$
\nabla_\mu\left\langle
T_{\rm clock}^{\mu\nu}
+T_{\rm record}^{\mu\nu}
+T_{\rm reservoir}^{\mu\nu}
\right\rangle=0
\tag{41}
$$

이어야 한다.

이 경로의 현재 주장 상한은 명확하다. 양의 초기 밀도행렬, 균일한 초기면,
차원 있는 $\Lambda_\Pi$, 지연핵·잡음핵과 저장소 응력을 외부 입력으로
명시하면 양의 평균 초기 전류를 준비하는 조건부 열린계 모형을 쓸 수 있다.
그러나 $\Pi_{\rm F}$의 값, 비선택 양자경로와의 동일성, 내재적 시간 화살,
암흑물질 존재량과 암흑에너지 절대크기는 아직 산출되지 않았다.

### 4.10.1 유계 Gaussian 저장소의 작용과 변분

초기 상태를 양자역학적으로 준비할 수 있다는 사실만으로는 시간이 지난 뒤의 에너지 장부가 닫히지 않는다. 이를 확인하려면 저장소를 이름만 붙인 환경으로 남기지 않고 같은 작용 안의 장으로 써야 한다. **[공리: R1 열린계 branch]** 부호 $(-+++)$에서 시계장과 저장소 스칼라 $\phi_A$의 총작용을

$$
S_{\rm tot}=\int d^4x\sqrt{-g}\left\{P(T,X)-\sum_A\left[
\frac12\nabla_\mu\phi_A\nabla^\mu\phi_A
+\frac12m_A^2\phi_A^2+s_A(T)\phi_A\right]\right\}
\tag{42}
$$

로 둔다. 여기서 $s_A(T)=\mu_A^3F_A(\Gamma T)$이고, $F_A(0)=0$ 및 $\|F_A\|_\infty<\infty$를 요구한다. $[T]=-1$, $[\phi_A]=[m_A]=[\Gamma]=[\mu_A]=1$이므로 $[s_A]=3$이고 식 (42)의 모든 항은 질량차원 4다. 유계 함수 조건은 큰 $T$에서 결합 자체가 무한히 깊은 우물을 만들지 않게 하는 최소 조건이다. 연속 스펙트럼을 쓴다면 $\sum_A\mu_A^6\|F_A\|_\infty^2/m_A^2<\infty$에 해당하는 UV 수렴 조건도 추가로 고정해야 한다.

이 선택은 단순 선형 결합 $gT\phi$의 완전 반례를 피한다. 실제로

$$
\frac12m^2\phi^2+gT\phi
=\frac12m^2\left(\phi+\frac{gT}{m^2}\right)^2
-\frac{g^2T^2}{2m^2}
\tag{43}
$$

이므로, 식 (21)의 포화 potential은 $T\to\infty$에서 마지막 음의 항을 상쇄하지 못한다. $g\ne0$인 선형 경로는 물질 Hamiltonian의 하한을 잃으므로 활성 경로에서 제거한다. 이는 고정배경 계산의 불편한 선택지가 아니라, 명시된 정의역 안에서 완전한 반례다.

식 (42)를 각각 $T$와 $\phi_A$에 대해 변분하면

$$
\nabla_\mu(P_X\nabla^\mu T)+P_T-\sum_A s'_A(T)\phi_A=0,
\qquad (\Box-m_A^2)\phi_A=s_A(T)
\tag{44}
$$

를 얻는다. 첫 식의 마지막 항은 축약된 시계장 재고가 저장소로 옮긴 만큼을 나타낸다. 따라서 여기서의 자기측정과 이웃 부트스트랩은 에너지를 스스로 만드는 장치가 아니라, 기록과 저장소를 후보 자유도로 추가한 모형 선택이다.

### 4.10.2 영향함수와 인과성

저장소를 적분해 시계장만 남기면, 국소 작용 대신 과거에 의존하는 영향함수가 나온다. 폐쇄시간경로에서 $s_r=(s_++s_-)/2$, $s_a=s_+-s_-$로 놓으면 Gaussian 저장소의 영향작용은

$$
S_{\rm IF}=-\int_{xy}s_a(x)D_R(x,y)s_r(y)
+\frac{i}{2}\int_{xy}s_a(x)N(x,y)s_a(y)
\tag{45}
$$

의 꼴이다. 여기서 $D_R$은 retarded Green 함수이고 $N$은 대칭 잡음핵이다. 원인보다 앞선 응답을 금지하려면

$$
D_R(x,y)=0\qquad\text{for}\qquad x\notin J^+(y),
\qquad \int_{xy}f(x)N(x,y)f(y)\geq0
\tag{46}
$$

가 필요하다. 첫 조건은 인과성, 둘째 조건은 Gaussian 잡음의 양성을 뜻한다. 유한 개의 완전한 저장소는 재귀할 수 있으므로, 비가역적인 거친갈기 화살을 말하려면 연속 스펙트럼과 시간척도 분리도 더 필요하다. 전체 계의 유니터리 진화가 근본적으로 일방향이 되었다는 결론은 식 (45)에서 나오지 않는다.

이 단계의 표준 도구는 Hu--Paz--Zhang의 Gaussian 환경 영향함수, JanaㆍLoganayagamㆍRangamani의 실시간 Schwinger--Keldysh 열린계 정식화, 그리고 CrossleyㆍGloriosoㆍLiu 및 GloriosoㆍCrossleyㆍLiu의 소산 유효장이론이다. 이 출처들은 지연 응답, 잡음, 두 CTP 사본 및 상태 조건이 함께 필요함을 뒷받침한다. 그러나 어느 출처도 CE의 비선택 경로가 이 저장소라는 동일성이나 $\Pi_{\rm F}$의 값을 증명하지 않는다.

### 4.10.3 Gaussian 준비와 고전적 읽기값

이제 평균 경계조건이 실제 양자상태와 양립하는지 확인한다. 부피 $V_c$인 거친갈기 cell에서 균일 mode의 정준쌍을 $(q,p)$로 쓰면

$$
\bar q=0,\qquad \bar p=V_ca_i^3\Pi_{\rm F}>0,\qquad
\mathbf V+\frac{i\hbar}{2}\mathbf\Omega\succeq0
\tag{47}
$$

를 만족하는 변위 Gaussian 상태를 택할 수 있다. 마지막 부등식은 Robertson 불확정성 조건이며, $\mathbf V$는 공분산, $\mathbf\Omega$는 정준 symplectic 형식이다. 따라서 $\langle T_i\rangle=0$과 $\langle J_i\rangle=\Pi_{\rm F}>0$는 양립하지만, 연산자 $T_i$와 전류의 분산까지 동시에 0이라는 뜻은 아니다. 국소 비도말 핵은 $\delta(0)$를 만들므로 $L_c$ 또는 UV cutoff가 필요하다.

또한 관측 가능한 고전 재고로 읽으려면 좁은 packet 조건

$$
\left|\langle J\rangle-J_{\rm cl}\right|\leq\epsilon_J
\tag{48}
$$

을 독립적으로 통과해야 한다. 이 준비는 $\Pi_{\rm F}$가 가능함을 보일 뿐 그 크기를 계산하지 않는다. $\Pi_{\rm F}$는 여전히 초기 Gaussian canonical momentum의 평균 변위이며, 0차원 측정 사건에서 그 값ㆍ균일 분포ㆍ초기면을 보내는 사상은 **[미완성]**이다. 여기서 0차원은 추가 시공간 축이 아니라 측정 사건과 기록의 타입이라는 뜻이다.

### 4.10.4 총 Ward 장부와 수치 게이트

축약된 시계장만 보존된다고 쓰면 저장소로 간 에너지와 상호작용 항을 잃는다. 식 (42)의 계량 변분으로 상호작용 응력

$$
T_{\rm int}^{\mu\nu}=-g^{\mu\nu}\sum_A s_A(T)\phi_A
\tag{49}
$$

까지 포함하면, 모든 장의 식 (44) 위에서

$$
\nabla_\mu\left(T_T^{\mu\nu}+T_\phi^{\mu\nu}+T_{\rm int}^{\mu\nu}\right)=0
\tag{50}
$$

이 따른다. 이것은 명시한 총작용의 조건부 Ward 결과다. bath를 Einstein source에서 빼거나 $\Pi_{\rm F}$를 별도 밀도로 더하는 순간 이 장부는 다시 깨진다.

제한된 Gaussian 표본에 대한 재현 artifact는 식 (46)의 비인과 성분 최대값 $0$, 잡음 공분산의 최소 고윳값 $-8.03\times10^{-16}$ (부동소수점 오차 안의 0), Robertson 여유 $0.22$, 총 에너지 상대 드리프트 $4.73\times10^{-14}$, 교환식 잔차 $1.19\times10^{-17}$을 보고했다. 이는 유한 표본에서 차원ㆍ인과성ㆍ공분산 양성ㆍ총 장부가 함께 맞는지 확인한 조건부 구현 결과다. 우주론의 수치 예측이나 0차원 기원의 증거는 아니다. 같은 작업트리에서 사전 `doctor` 진단이 삭제된 `reality_stone` 모듈 때문에 실패한 일은 환경 결함이며, 이 과학 결과와 분리한다.

### 4.10.5 R2: 좌표 섭동과 물리 곡률섭동의 분리

열린계 작용을 얻은 뒤 다음으로 확인할 것은 음의 부호가 실제 물리 mode의
불안정성인지다. 고정된 FLRW 계량에서 $T\mapsto T+\pi$만 변분하면 장파장
근사는

$$
m_{\rm eff}^2\simeq-\frac{P_{TT}}{P_X+2XP_{XX}}<0
\tag{51}
$$

를 준다. 이 식은 유용한 음성대조군이지만 $\pi=\delta T$는 gauge-dependent
변수다. 시간 재매개변환으로 $\pi$를 0으로 두는 unitary gauge가 가능하므로,
식 (51)을 곧바로 물리적 스칼라의 질량이라고 부를 수 없다.

R2는 먼저 이 제한을 수치로 드러냈다. 동결한 다섯 배경과 $a=10^{-4}$부터
오늘까지의 관측창에서 $|m_{\rm eff}^2|/H^2$의 최댓값은
$3.0455\times10^{-18}$이고, 상수계수 비교에서 얻은 보수적 로그 성장 상계는
$2.4838\times10^{-17}$이다. 이는 선택 초기값 $\pi_i=1$, $\pi_i'=0$ 및
고정 계량에서만 성립하는 진단이다. 임의 초기 섭동의 정리도, metricㆍbath를
포함한 우주론의 안정성 증명도 아니다.

그 다음 Einstein 중력과 이 시계장만 남긴 single-clock 부분계를 unitary gauge로
정리한다. ADM 계량에서 lapse를 $N=1+\alpha$, shift를
$N_i=\partial_i\beta$, 공간계량을 $h_{ij}=a^2e^{2\zeta}\delta_{ij}$로 둔다.
여기서 $\zeta$는 공간 곡률섭동이다. $\alpha$와 $\beta$에는 시간 이차 미분이
없으므로 독립 전파장이 아니라 제약 변수다. 선형 Hamiltonianㆍmomentum 제약을
차례로 풀면

$$
\alpha=\frac{\dot\zeta}{H},\qquad
\beta=-\frac\zeta H+\chi,\qquad
\frac{\partial^2\chi}{a^2}=\frac{\Sigma}{M_{\rm Pl}^2H^2}\dot\zeta,
\qquad \Sigma=XA.
\tag{52}
$$

를 얻는다. 이 식을 원래 작용에 대입하면 제약 변수는 사라지고

$$
S^{(2)}_\zeta=\int dt\,d^3x\,a^3Q_s\left[
\dot\zeta^2-c_s^2\frac{(\nabla\zeta)^2}{a^2}\right],
\qquad
Q_s=\frac{XA}{H^2},\qquad c_s^2=\frac BA,
\tag{53}
$$

를 얻는다. 여기서 $\zeta$는 공간 곡률섭동이고,
$A=P_X+2XP_{XX}$, $B=P_X$다. 이 작용에는 독립적인 $\zeta^2$ 질량항이 없다.
따라서 장파장 해는

$$
\zeta(t)=C_1+C_2\int^t\frac{dt'}{a^3(t')Q_s(t')}
\tag{54}
$$

로 쓴다. 식 (53)은 식 (51)의 음의 부호를 물리적 tachyon 판정으로 옮길 수
없다는 정확한 이유다.

현재 작용에서는 $\delta>0$일 때 $A>0$, $B>0$이다. R2의 관측창 수치에서
$\min c_s^2=9.2138\times10^{-19}$,
$\min(Q_s/M_{\rm Pl}^2)=3.3167\times10^5$였고,
$d\ln(a^3Q_s)/dN$ 및 $d\ln(Ha^3Q_s)/dN$의 최솟값은 각각 $3.93909$와
$3.469545$였다. 그러므로 이 구간에서는 두 번째 해의 $\dot\zeta$와 식 (54)의
적분함수가 감소한다. 이 결과는 유한 관측창의 단일-clock 부분계에 한정된다.
무한 미래 수렴, baryonㆍradiationㆍreservoir entropy mode, 또는 전체 ADM
행렬의 양성을 주장하지 않는다.

### 4.10.6 작은 음속과 cutoff의 두 척도

작은 $c_s$가 양수라는 사실만으로 EFT가 닫히지는 않는다. $\delta\to0$의
decoupling 전개에서는 $X_*=1/2$와 $A_0=4\kappa\rho_\infty$를 쓰므로,
정준장은 $\varphi=\sqrt{A_0}\,\pi=2\sqrt{\kappa\rho_\infty}\,\pi$다. 이
배경 주위 전개에서 leading 상호작용은

$$
\mathcal L_3=\frac{\dot\varphi^3-\dot\varphi(\nabla\varphi)^2}
{4\sqrt{\kappa\rho_\infty}},\qquad
\mathcal L_4=\frac{\dot\varphi^4-2\dot\varphi^2(\nabla\varphi)^2+(\nabla\varphi)^4}
{32\kappa\rho_\infty}.
\tag{55}
$$

따라서 bare 척도는 $\Lambda_3=2(\kappa\rho_\infty)^{1/4}=79.7$--$80.6\,{\rm eV}$,
$\Lambda_4=(32\kappa\rho_\infty)^{1/4}=94.8$--$95.8\,{\rm eV}$다. 작은 음속의
재스케일링에서는 $x=c_s\widetilde x$ 및
$\widetilde\varphi=c_s^{3/2}\varphi$로 둔다. 그러면 공간 미분을 포함한
cubic 항은

$$
\int dt\,d^3x\,\frac{\dot\varphi(\nabla\varphi)^2}{\Lambda_3^2}
=\int dt\,d^3\widetilde x\,
\frac{\dot{\widetilde\varphi}(\widetilde\nabla\widetilde\varphi)^2}
{\Lambda_3^2c_s^{7/2}}
\tag{56}
$$

로 바뀐다. 따라서 이 항이 강해지는 에너지와 그에 대응하는 물리 파수는 서로 다르며,

$$
\Lambda_E=\Lambda_3c_s^{7/4},\qquad
q_{\rm sc}=\Lambda_3c_s^{3/4}=\frac{\Lambda_E}{c_s}.
\tag{57}
$$

관측창 전체의 최솟값은 각각 $1.3336\times10^{-14}\,{\rm eV}$와
$1.3893\times10^{-5}\,{\rm eV}$이며, $\Lambda_E/H>9.2757\times10^{18}$,
$q_{\rm sc}/[(1\,{\rm Mpc}^{-1})/a]>2.1725\times10^{24}$다. 즉 이 특정
선형 우주론 scale은 두 cutoff 아래에 있다. 이 수치는 UV completion의 존재나
비선형 안정성을 증명하지 않는다.

ghost-condensate형 $k^4$ 경로를 보려면 crossover 파수

$$
q_\times=\frac{c_s\sqrt A}{\bar M}
\tag{58}
$$

가 strong-coupling 파수보다 먼저 와야 한다. $q_\times\le q_{\rm sc}$는
$\bar M\gtrsim(\kappa\rho_\infty)^{1/4}c_s^{1/4}$를 준다. 오늘의 하한은
$0.225\,{\rm eV}$, 관측창 최악 하한은 $7.31\,{\rm eV}$이므로
$\bar M\sim80\,{\rm eV}$ 후보는 crossover의 필요조건을 충족할 수 있다.
그러나 이는 실제 $k^4$ 연산자의 부호ㆍdegeneracyㆍ새 cutoff를 정하지 않는다.
또한 미래에 $u\propto a^{-3}$이면 $c_s\propto a^{-3/2}$,
$\Lambda_E\propto a^{-21/8}$, $q_{\rm sc}\propto a^{-9/8}$,
$q_\times/q_{\rm sc}\propto a^{-3/8}$다. 따라서 two-derivative 단독 이론의
무한 미래는 미완성이고, 유한 $\bar M$의 $k^4$ 완성은 다음 계산 경로다.

## 4.11 요약

이 장의 계산은 FLRW 방정식과 명시한 branch의 조건부 결론을 보존하지만,
암흑에너지 scale의 기원까지 닫지는 않는다. 다음 표는 이 구분을 지위와
전제로 압축한다.

| 항목 | 지위 | 범위 |
|---|---|---|
| Friedmann·continuity 방정식 | [정리] | FLRW와 Einstein 방정식 |
| canonical scalar의 $w\ge-1$ | [정리] | positive kinetic, $\rho_\phi>0$ |
| 상수 scalar의 vacuum stress | [정리] | stationary constant field |
| scalar--tensor field equation | [정리] | 명시한 $F,K,V$ 작용 |
| growth equation | [정리] | sub-horizon quasi-static branch |
| $\mu=1+\epsilon S$ | [공리] | dimensionless toy family |
| 암흑에너지 절대값과 CE 기원 | [미완성] | UV·재규격화·관측 사상 부재 |
| 자기비동일성 시계장의 DE형·DM형 분해 | [조건부 정리] | 식 (21), $\delta\ge0$, 양의 초기 전류 |
| 영 전류에서 양의 DM형 재고 생성 | [반례로 제거] | 식 (28)–(29) |
| 경계항 하나의 앵커·전류 동시 유도 | [반례로 제거] | 디리클레 변분과 식 (36) |
| 선형 $gT\phi$ 저장소 | [반례로 제거] | 식 (43)의 하방 비유계성 |
| 유계 Gaussian 저장소의 인과ㆍ양성ㆍWard 장부 | [조건부 정리] | 식 (42), Gaussian 상태, smearing |
| 폐쇄시간경로 초기상태 준비 | [공리·미완성] | $\Pi_{\rm F}$ㆍ0차원 사상ㆍ균일 초기면 미산출 |
| R2 단일-clock 관측창 $Q_s,c_s^2$와 $\dot\zeta$/적분함수 감소 | [조건부 산출] | Einstein+clock 부분계, 유한 관측창 |
| R2 cutoff 분리와 $k^4$ crossover 필요조건 | [조건부 산출] | 고정 $\kappa$, canonical 전개, completion 자체는 미완성 |
| 결합 우주론 안정성ㆍ수치 예측 | [미완성] | metric/bath/$k^4$/CMB-LSS gate |
