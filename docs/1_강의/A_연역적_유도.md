# CE 연역 구조: 증명과 물리 사상의 분리

이 강의는 적은 수의 수학 명제에서 무엇이 실제로 따라오는지 정리한다.
물리적 동일시나 수치 관계는 증명과 분리한다.

## 0. 형식 출처

| 표지 | 의미 |
|---|---|
| **[정의]** | 기호·정의역 선언 |
| **[정리]** | 명시한 가정에서 증명된 명제 |
| **[공리]** | 선택·정규화·물리적 식별 입력 |
| **[산출]** | 앞선 입력을 대입한 계산 |
| **[경험식]** | 자료를 참고해 고른 관계 |
| **[미완성]** | 필요한 증명·동역학·자료가 남은 관계 |
| **[예측]** | 모델과 비교 규약을 사전에 고정한 관측 명제 |

## 1. 경로공간 기호

**[정의]** 작용 \(S[\gamma]\)가 정의된 경로 \(\gamma\)에서 두 번째
변분 연산자를

\[
\mathcal H_\gamma:=\delta^2S[\gamma]
\]

로 쓴다. probe \(u\)를 정하면

\[
\Phi_H[\gamma;u]:=
\frac{\langle u,\mathcal H_\gamma u\rangle}{\langle u,u\rangle},
\qquad u\ne0
\]

를 Hessian readout이라 한다. \(\mathcal H_\gamma\), \(\Phi_H\), 독립
스칼라장 \(\phi\), Ricci scalar \(R\)는 서로 다른 대상이다.

이들 사이의 등식이나 결합은 독립 작용에서 제시해야 하며 현재
**[미완성]**이다.

## 2. 곱법칙에서 지수함수

**[정리]** 연속이고 양수인 \(S:[0,\infty)\to(0,\infty)\)가

\[
S(x+y)=S(x)S(y),\qquad S(0)=1
\]

을 만족하면 \(S(D)=e^{aD}\)다. 비증가 조건을 더하면

\[
S(D)=e^{-\lambda D},\qquad \lambda\geq0.
\]

[자족 증명](../참조/핵심_정리_증명.md#exp-theorem)

비자명 감쇠 \(S\not\equiv1\)을 더 요구하면 \(\lambda>0\)이다.

**[공리]** 이 비자명 가지에서 깊이 좌표를 재척도해 \(\lambda=1\)로 둔다. 따라서
\(S(D)=e^{-D}\)를 쓴다. 이 정리는 물리적 \(D\)의 정의나 특정 과정의
감쇠율을 제공하지 않는다.

## 3. 형식 차원

**[정리]** \(d\)차원 벡터공간에서

\[
\dim\Lambda^1=d,\qquad
\dim\Lambda^2=\binom d2=\frac{d(d-1)}2.
\]

두 성분 수가 같다는 방정식은

\[
d=\frac{d(d-1)}2
\quad\Longleftrightarrow\quad
d(d-3)=0
\]

이고 정수해는 \(d=0,3\)이다. 양의 정수로 제한하면 \(d=3\)이다.

**[공리]** 이 비자명 분기를 실제 공간 차원으로 읽는다. 이 대수 조건은
다른 차원의 모든 물리 모형이나 생물 가능성을 배제하는 보편 정리가 아니다.

[자족 증명](../참조/핵심_정리_증명.md#exterior-dimension)

다음 동일시들도 각각 별도 **[공리]**다.

\[
N_c=d,\qquad N_w=d-1,\qquad N_{\rm gen}=d.
\]

게이지군, 힘의 수와 물질 표현을 위 정수만으로 결정하는 동역학은
**[미완성]**이다.

## 4. 전자약 혼합과 유효 차원

**[공리]** 외부 장론으로 one-Higgs-doublet 표준모형을 사용하고

\[
\tan\theta_W=\frac{g'}g,\qquad
M_Z^2=\frac{(g^2+g'^2)v^2}{4}
\]

를 둔다.

**[산출]** 표준 \((W^3,B)\) 기저의 중성 보손 질량행렬에서

\[
\frac{|\mathcal M^2_{W^3B}|}{M_Z^2}
=\sin\theta_W\cos\theta_W.
\]

**[공리]** 이 기저 의존 혼합 진폭의 제곱을 CE 차원 증분으로 읽는다.

\[
\delta:=\sin^2\theta_W\cos^2\theta_W,\qquad
D_{\rm eff}:=d+\delta.
\]

두 번째 등식도 선형 결합을 택한 모델 규칙이다.

**[정리]** \(0\leq\theta_W\leq\pi/2\)이면

\[
0\leq\delta\leq\frac14,
\]

이고 최대는 \(\theta_W=\pi/4\)에서 얻는다.

[자족 증명](../참조/핵심_정리_증명.md#delta-bound)

**[경험식]** 수치 계산에서는 외부 \(\alpha_s(M_Z)\)에 대해

\[
\sin^2\theta_W=4\alpha_s^{4/3}
\]

를 사용한다. 이 관계와 \(\alpha_s\)는 차원 방정식에서 나오지 않는다.

## 5. 최소 소멸 고정점

**[정의]**

\[
F_D(q):=e^{-D(1-q)}.
\]

**[정리]** \(D>1\)이면 \(I_D=[0,1/D]\)에서 \(F_D\)는 자신으로 가는
축소사상이다. 따라서 \(I_D\) 안에 유일한 고정점 \(q_{\rm ext}\)가 있다.

\[
q_{\rm ext}=-\frac1D W_0(-De^{-D}).
\]

[자족 증명과 Poisson 최소해](../참조/핵심_정리_증명.md#poisson-fixed-point)

전체 \([0,1]\)에는 \(q=1\)도 고정점이므로 전 구간 유일성이나 임의
초기값에서의 수렴은 이 정리의 결론이 아니다.

단일 초기 개체 \(Z_0=1\)과 \(\operatorname{Poisson}(D)\) offspring를 둔
분기과정에서 \(q_{\rm ext}\)는 소멸확률이고
\(s_{\rm branch}=1-q_{\rm ext}\)가 생존확률이다.

## 6. 확률 readout

**[정리]** 연속함수 \(I:(0,1]\to(0,1]\)가

\[
I(PQ)=I(P)I(Q),\qquad I(1)=1
\]

을 만족하면 \(I(P)=P^c\)인 \(c\geq0\)가 존재한다. \(I\)가 상수가
아니거나 \(I(0^+)=0\)이면 \(c>0\)이다.

[자족 증명](../참조/핵심_정리_증명.md#multiplicative-readout)

**[공리]** CE readout에서는 \(c=1\)을 선택한다. \(q_{\rm ext}\) 또는
\(s_{\rm branch}\)를 바리온 분율 같은 관측량에 대응시키는 일은 또 다른
물리 사상이며 현재 **[미완성]**이다.

## 7. 공변 장론으로 살리는 최소 branch

**[공리]** \(Z_2:\phi\mapsto-\phi\)를 가진 singlet-portal EFT를

\[
S=\int\sqrt{-g}\left[
\frac12(M_{\rm Pl}^2-\xi\phi^2)R-\frac12(\nabla\phi)^2
+\mathcal L_{\rm SM}-V(H,\phi)\right]d^4x
\]

로 택한다.

\[
V_4=\lambda_H(H^\dagger H)^2
+\frac{\lambda_{H\phi}}2\phi^2H^\dagger H
+\frac{\lambda_\phi}{4}\phi^4.
\]

**[정리]** \(\lambda_H,\lambda_\phi>0\)에서 이 사차항이 아래로
유계이기 위한 조건은

\[
\lambda_{H\phi}\geq-2\sqrt{\lambda_H\lambda_\phi}
\]

이다.
[증명](../참조/핵심_정리_증명.md#portal-boundedness)

**[정리]** 완결된 공변 작용의 총 stress tensor는 모든 장의
운동방정식 위에서 보존된다.
[증명](../참조/핵심_정리_증명.md#noether-stress)

이 branch는 질량과 결합을 자유 매개변수로 가진다. 따라서 특정 수치를
유도하지는 않지만, 대칭·안정성·보존법칙을 갖춘 이론물리 모형으로는
완결되어 있다.

## 8. Euclidean cutoff와 측도

**[공리]** scalar--Higgs bosonic truncation의 연속·유한 다항식 작용,
유한 격자와 양의 kinetic operator를 택한다.

**[정리]** 사차 potential이 coercive이면

\[
Z_N=\int_{\mathbb R^N}e^{-S_{E,N}(z)}d^Nz
\]

와 모든 다항식 모멘트가 유한하다.
[증명](../참조/핵심_정리_증명.md#finite-lattice-measure)

이는 경로적분을 유한 cutoff에서 실제 확률측도로 만든다. continuum
limit와 Lorentzian 재구성은 별도 문제다.

## 9. 우주론의 조건부 정리

**[공리]** Einstein frame에서 최소 결합되고 양의 kinetic term을 가진
균일 canonical scalar를 평탄 FLRW에 둔다.

**[정리]**

\[
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V,\qquad
w_\phi+1=\frac{\dot\phi^2}{\rho_\phi}\geq0
\]

이다. 따라서 \(\rho_\phi>0\)인 최소 단일장 모형은 phantom crossing을
허용하지 않는다.
[증명](../참조/핵심_정리_증명.md#canonical-scalar-flrw)

**[정리]** 상수 \(V_0\)는
\(T_{\mu\nu}=-V_0g_{\mu\nu}\), \(w=-1\)을 준다.
[증명](../참조/핵심_정리_증명.md#vacuum-stress)

방정식상태는 닫히지만 \(V_0\)의 절대값은 여전히 자유 매개변수다.

## 10. 물리량의 출처 지도

| 항목 | 형식 출처 |
|---|---|
| 지수형 함수의 형태 | **[정리]** |
| 감쇠 단위 \(\lambda=1\) | **[공리]** |
| 양의 정수 조건에서 \(d=3\) | **[정리]** |
| \(d\)를 공간·색·세대로 읽기 | **[공리]** |
| \(\delta\), \(D_{\rm eff}\) 물리 사상 | **[공리]** |
| 주어진 \(D\)의 최소 고정점 | **[정리]**, **[산출]** |
| \(\sin^2\theta_W\)와 결합 관계 | **[경험식]** |
| 암흑 성분·질량·혼합·측정 동역학 | **[미완성]** |

관측값은 이 표의 정리나 공리를 만드는 입력으로 섞지 않고, 모델을 모두
고정한 뒤 별도 비교 자료로만 사용한다.

## 11. 남은 과제 **[미완성]**

- 선택한 EFT의 양자 재규격화와 continuum limit
- 내부 게이지군·물질 표현의 선택 원리
- Yukawa·중성미자·CP 섹터의 질량행렬
- 고정점 확률과 우주론 자료를 연결하는 likelihood
- Born 측도와 단일 시행 측정을 산출하는 instrument
- 공학 벤치마크의 외부 재현과 사전 고정 평가 규약

이 자료가 마련되기 전에는 물리적 대응을 수학적 정리로 올리지 않는다.
