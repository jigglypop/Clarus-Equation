# 4. 암흑에너지 모형: FLRW, canonical scalar, scalar--tensor branch

이 장은 암흑에너지의 절대크기를 무입력으로 유도하지 않는다. 일반상대론의
FLRW 방정식, canonical scalar와 명시적인 scalar--tensor 작용에서 닫히는
조건부 결과를 정리한다. reduced Planck mass는
\(M_{\rm Pl}^{-2}=8\pi G\)로 정의한다.

## 4.1 FLRW 배경

**[공리] 우주론 branch:** 동질·등방 metric
\[
ds^2=-dt^2+a(t)^2\gamma_{ij}dx^idx^j,
\qquad {}^{(3)}R=6k
\]
와 perfect-fluid total stress tensor
\(T^\mu{}_\nu=\operatorname{diag}(-\rho,p,p,p)\)를 택한다.

**[정의]** \(H:=\dot a/a\)다.

**[정리]** Einstein 방정식은
\[
H^2+\frac{k}{a^2}=\frac{\rho}{3M_{\rm Pl}^2},
\]
\[
\dot H-\frac{k}{a^2}
=-\frac{\rho+p}{2M_{\rm Pl}^2},
\qquad
\frac{\ddot a}{a}
=-\frac{\rho+3p}{6M_{\rm Pl}^2}
\]
를 준다. Bianchi identity와 matter 방정식에서
\[
\dot\rho+3H(\rho+p)=0
\]
가 따른다.

**[산출]** 팽창 중인 branch에서 \(\ddot a>0\)의 필요충분조건은
\(\rho+3p<0\)이다. 이는 source의 equation of state에 관한 조건이며
\(\rho\)의 절대값을 정하지 않는다.

## 4.2 canonical scalar

**[공리] 모델 선택:**
\[
S=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac12(\nabla\phi)^2-V(\phi)
\right]+S_m
\]
를 택하고 \(\phi=\phi(t)\)로 제한한다.

**[정리]**
\[
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,
\]
\[
\ddot\phi+3H\dot\phi+V_{,\phi}=0,
\qquad
\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0
\]
이다.

**[정리]** \(\rho_\phi>0\)이면
\[
w_\phi+1
=\frac{\rho_\phi+p_\phi}{\rho_\phi}
=\frac{\dot\phi^2}{\rho_\phi}\ge0.
\]
따라서 positive-kinetic canonical single scalar는 \(w=-1\) 아래로
넘지 않는다. phantom crossing에는 추가 자유도, noncanonical kinetic
구조 또는 modified gravity가 필요하다.

**[정리]** \(V_{,\phi}(\phi_0)=0\), \(\dot\phi=0\)인 상수 branch는
\[
T_{\mu\nu}^{(\phi_0)}=-V(\phi_0)g_{\mu\nu},
\qquad p_\phi=-\rho_\phi
\]
를 준다. \(V(\phi_0)>0\)이면 cosmological constant와 같은 배경 source다.

**[미완성]** \(V(\phi_0)\)의 관측 절대값, radiative stability와
초기조건을 CE가 고정하는 메커니즘은 없다.

## 4.3 scalar--tensor 작용

**[공리] 별도 branch:**
\[
S_{\rm ST}=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}F(\phi)R
-\frac12K(\phi)(\nabla\phi)^2
-V(\phi)
\right]+S_m[g,\Psi]
\]
를 택한다. \(F,K\)는 무차원이고 \([\phi]=1\)이다.

**[정리]** 변분하면
\[
M_{\rm Pl}^2F G_{\mu\nu}
=T_{\mu\nu}^{m}
+K\left(\nabla_\mu\phi\nabla_\nu\phi
-\frac12g_{\mu\nu}(\nabla\phi)^2\right)
-g_{\mu\nu}V
+M_{\rm Pl}^2
(\nabla_\mu\nabla_\nu F-g_{\mu\nu}\Box F),
\]
\[
K\Box\phi+\frac12K_{,\phi}(\nabla\phi)^2
+\frac{M_{\rm Pl}^2}{2}F_{,\phi}R
-V_{,\phi}=0.
\]

**[공리] 안정 영역:** \(F>0\)이고 Einstein-frame scalar kinetic
coefficient
\[
K_E(\phi)
=\frac{K}{F}
+\frac{3M_{\rm Pl}^2}{2}
\left(\frac{F_{,\phi}}{F}\right)^2
\]
가 양수인 field domain만 사용한다.

**[정리]** \(g^E_{\mu\nu}=F g_{\mu\nu}\)의 Weyl 변환은 이 영역에서
중력항을 Einstein--Hilbert 꼴로 만들고 scalar kinetic coefficient를
\(K_E\)로 바꾼다. matter는 일반적으로 Einstein frame에서 \(\phi\)와
결합하므로 서로 다른 frame의 중간 변수를 직접 관측량처럼 비교하지 않는다.

## 4.4 scalar--tensor FLRW 방정식

**[산출]** 평탄하지 않을 수 있는 FLRW와 homogeneous \(\phi\)에서
\[
3M_{\rm Pl}^2F
\left(H^2+\frac{k}{a^2}\right)
=\rho_m+\frac12K\dot\phi^2+V
-3M_{\rm Pl}^2H\dot F,
\]
\[
-2M_{\rm Pl}^2F
\left(\dot H-\frac{k}{a^2}\right)
=\rho_m+p_m+K\dot\phi^2
+M_{\rm Pl}^2(\ddot F-H\dot F).
\]
상수 \(\phi=\phi_0\), \(F(\phi_0)=F_0>0\)이면 derivative 항이 사라지고
유효 Planck mass는 \(M_{\rm Pl}^2F_0\)가 된다. 다만 이것이 full
상수장 해가 되려면 scalar equation도 pointwise로
\[
\frac{M_{\rm Pl}^2}{2}F_{,\phi}(\phi_0)R
-V_{,\phi}(\phi_0)=0
\]
을 만족해야 한다.

**[미완성]** 원하는 \(H(a)\)를 먼저 고른 뒤 독립적인 \(\mu(a,k)\)를
추가하는 것은 일반적으로 하나의 공변 작용과 자동으로 정합하지 않는다.
\(F,K,V\), matter coupling과 초기조건을 함께 고정해야 한다.

## 4.5 선형 성장의 조건부 식

**[공리] 섭동 branch:** Jordan frame에서 보존되는 pressureless matter,
sub-horizon quasi-static regime와 negligible matter anisotropic stress를
가정한다. 고정 reference wave number \(k_{\rm ref}>0\)와
\(q:=k/k_{\rm ref}\), 고정 reference Planck mass \(M_{\rm ref}>0\)를
도입한다.

**[정의]** scale factor \(a\)는 무차원이고 prime은 \(d/d\ln a\),
\(D(a,q)\)는 linear matter growth factor다. 또한
\[
\Omega_m(a):=\frac{\rho_m}{3M_{\rm ref}^2H^2},
\qquad
\mu(a,q):=\frac{G_{\rm growth}(a,q)}
{(8\pi M_{\rm ref}^2)^{-1}}
\]
로 정의한다. 여기서 \(G_{\rm growth}\)는 quasi-static Poisson
constraint의
\(-k^2\Psi/a^2=4\pi G_{\rm growth}\rho_m\delta_m\)로 정한 계수다.

**[정리]** 위 근사에서
\[
D''+\left(2+\frac{H'}H\right)D'
-\frac32\Omega_m(a)\mu(a,q)D=0.
\]
\(M_{\rm ref}=M_{\rm Pl}\), \(\mu=1\)이면 minimally coupled GR의
표준 식이다. \(\mu\)는 무차원이고
scalar--tensor branch에서는 action의 constraint equation에서 계산해야 한다.

**[공리] toy parameterization:**
\[
\mu(a,q)=1+\epsilon\,S(a,q),\qquad
0\le S(a,q)\le1
\]
를 자료 비교용 family로 둘 수 있다. \(\epsilon\)과 \(S\)는 무차원이다.

**[경험식]** \(\epsilon\), \(S\) 또는 초기 normalization을 growth data로
고르면 그 결과는 fit이다. 같은 data point를 다시 독립 예측으로 세지 않는다.

## 4.6 절대 scale과 관측

**[공리] 외부 입력:** 실제 우주 계산에는 \(H_0\), matter·radiation
density, neutrino sector, recombination과 survey likelihood를 데이터
snapshot과 함께 공급한다.

**[산출]** 지정한 \(F,K,V\), 초기조건과 외부 입력을 넣어 얻은
\(H(a),D(a,k)\)는 그 모형의 조건부 수치해다.

**[미완성]** 다음은 현재 닫히지 않는다.

1. vacuum contribution과 관측 dark-energy scale 사이의 radiatively stable
   matching,
2. CE 확률변수 또는 Hessian readout에서 \(F,K,V\)로 가는 action-level map,
3. background·CMB·BAO·lensing·growth를 한 parameter set으로 잇는 likelihood,
4. 자료를 보기 전에 고정한 구별 가능한 관측 예측.

## 4.7 요약

| 항목 | 지위 | 범위 |
|---|---|---|
| Friedmann·continuity 방정식 | [정리] | FLRW와 Einstein 방정식 |
| canonical scalar의 \(w\ge-1\) | [정리] | positive kinetic, \(\rho_\phi>0\) |
| 상수 scalar의 vacuum stress | [정리] | stationary constant field |
| scalar--tensor field equation | [정리] | 명시한 \(F,K,V\) 작용 |
| growth equation | [정리] | sub-horizon quasi-static branch |
| \(\mu=1+\epsilon S\) | [공리] | dimensionless toy family |
| 암흑에너지 절대값과 CE 기원 | [미완성] | UV·재규격화·관측 사상 부재 |
