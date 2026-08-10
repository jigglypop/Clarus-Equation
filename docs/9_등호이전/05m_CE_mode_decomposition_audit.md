# 05m. Mode 분해와 mean-field 오차

## 0. 범위

평균의 가산 분해
\[
\mathbb E\Phi=\sum_j\mathbb EY_j
\]
는 확률변수의 독립성, 분산 또는 분포를 정하지 않는다. 이 문서는
intensive mode 모형에서 정확히 계산되는 오차와, 물리적 mode 사상 사이의
경계를 기록한다.

## 1. 평균 분해의 식별 한계

**[정리: 식별 불가]** 무차원 확률변수 \(\Phi\)와 \(m>0\)에 대해
\(\mathbb E\Phi=m\)만 알아서는
\(\mathbb Ee^{-\Phi}\)를 결정할 수 없다.

**증명.** \(\Phi\equiv m\)이면
\(\mathbb Ee^{-\Phi}=e^{-m}\)이다. 반면 \(0<a<m<b\)를 택하고 평균이
\(m\)이 되도록 \(a,b\)에 양의 확률을 주는 두 점 분포는 strict
convexity에 의해
\[
\mathbb Ee^{-\Phi}>e^{-m}
\]
를 준다. 같은 평균에 대해 분산과 Laplace transform이 다르다.
\(\square\)

따라서 “채널 수”가 평균 \(m\)을 분해한다는 문장만으로 microscopic
mode 수 \(N_{\rm eff}\), 독립성 또는 mean-field 오차가 나오지 않는다.

## 2. 독립 Gamma-mode benchmark

**[공리: 모델 선택]** \(m>0\), 정수 \(N\geq1\)을 고정하고
\[
\Phi_N\sim
\operatorname{Gamma}\!\left(
k=\frac N2,\ \theta=\frac m k
\right)
\]
로 둔다. 그러면 \(\mathbb E\Phi_N=m\)이고
\(\operatorname{Var}(\Phi_N)=m^2/k=2m^2/N\)이다.

**[정리]** Mean-field 비는 정확히
\[
R_N(m)
:=
\frac{\mathbb Ee^{-\Phi_N}}{e^{-\mathbb E\Phi_N}}
=
\exp\!\left[
m-k\log\!\left(1+\frac mk\right)
\right]>1.
\]
또한 \(N\to\infty\)에서
\[
\log R_N(m)
=
\frac{m^2}{N}
-\frac{4m^3}{3N^2}
+O(N^{-3}).
\]

**증명.** Gamma Laplace transform
\(\mathbb Ee^{-s\Phi_N}=(1+\theta s)^{-k}\)에 \(s=1\),
\(\theta=m/k\)를 대입한다. 전개는
\(\log(1+x)=x-x^2/2+x^3/3+O(x^4)\)에서 따른다.
\(\square\)

**[산출]** 허용 상대오차 \(\delta>0\)를 외부에서 정하면 이 benchmark의
정확한 조건은
\[
m-\frac N2\log\!\left(1+\frac{2m}{N}\right)
\leq\log(1+\delta)
\]
다. 작은 \(\delta\), 큰 \(N\)에서는
\[
N\gtrsim\frac{m^2}{\log(1+\delta)}
\]
가 선도 필요조건이다. \(m,\delta\)에 관측값을 대입한 수치는
**[경험식]** 입력에 의존하며 보편 정리가 아니다.

## 3. 상관된 동일모드의 분산

**[공리: 모델 선택]** \(Y_1,\dots,Y_N\)이
\[
\mathbb EY_j=\frac mN,\qquad
\operatorname{Var}(Y_j)=r\frac{m^2}{N^2}
\]
를 갖고, 서로 다른 쌍의 상관계수가 모두 \(\rho\)라고 하자.
Equicorrelation covariance가 존재하려면
\[
-\frac1{N-1}\leq\rho\leq1
\]
이어야 한다.

**[정리]** \(\Phi_N=\sum_jY_j\)에 대해
\[
\operatorname{Var}(\Phi_N)
=
\frac{rm^2}{N}\bigl(1+(N-1)\rho\bigr).
\]

**증명.** \(N\)개 분산과 \(N(N-1)\)개의 ordered covariance를
합한다. \(\square\)

\(\rho>0\)가 \(N\)과 함께 0으로 가지 않으면 분산은
\(rm^2\rho\)에 접근하므로 independence에서 얻는 \(N^{-1}\) 억제가
사라진다. 다만 분산만으로 \(\mathbb Ee^{-\Phi_N}\)의 정확한 값은
결정되지 않는다.

추가로 \(0\leq\Phi_N\leq C\) almost surely이고 상대오차를 \(\delta\)
이하로 만들 충분조건을 원하면
[05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md) 4절에 의해
\[
\frac{rm^2}{N}\bigl(1+(N-1)\rho\bigr)
\leq
2e^{-C}\log(1+\delta)
\]
를 사용할 수 있다. 이는 bounded-mode 모형의 충분조건이지 일반적인
필요조건이 아니다.

## 4. Intensive와 extensive scaling

**[정리]** \(\Phi_N\equiv m\)인 결정론 모형에서는 mean-field identity
\[
\mathbb Ee^{-\Phi_N}=e^{-\mathbb E\Phi_N}
\]
가 정확하다. 결정론 모형은 확률론만으로 배제되지 않는다.

**[정리]** [05l_CE_uncertainty_floor.md](05l_CE_uncertainty_floor.md)
4절의 독립·유계 intensive 모드 조건에서는
\[
\frac{\mathbb Ee^{-\Phi_N}}
{e^{-\mathbb E\Phi_N}}
=1+O(N^{-1}).
\]

반면 [05k_CE_hard_constraint.md](05k_CE_hard_constraint.md)의 extensive
Gaussian action \(U_N\sim\operatorname{Gamma}(N/2,1)\)은
\[
\frac{\mathbb Ee^{-U_N}}{e^{-\mathbb EU_N}}
=
\frac{2^{-N/2}}{e^{-N/2}}
=
\left(\frac e2\right)^{N/2},
\]
이므로 mean-field 비가 발산한다.

같은 “mode 합”이라도 총량이 \(O(1)\)인 intensive normalization과 평균이
\(O(N)\)인 extensive action을 섞을 수 없다.

## 5. Smooth tilt와 단일 threshold

Layer-cake identity
\[
\mathbb Ee^{-\Phi}
=
\int_0^\infty e^{-t}\,\mathbb P(\Phi\leq t)\,dt
\]
는 smooth tilt가 모든 threshold 분율의 가중 평균임을 말한다. 이 항등식은
어떤 하나의 threshold \(t_*\)나 quantile \(q\)를 선택하지 않는다.

따라서
\[
\mathbb Ee^{-\Phi}
\quad\text{와}\quad
\mathbb P(U_N<u_N)
\]
를 같은 수로 맞추려면 \(u_N\) 또는 \(q\)를 별도
**[공리: 모델 선택]**으로 주어야 한다.

## 6. 남은 물리 사상

다음 항목은 **[미완성]**이다.

- CE의 \(\Phi\)가 결정론인지 확률변수인지
- microscopic \(Y_j\)의 작용, 분포, cutoff와 mode 수
- mode 사이 covariance의 부호와 크기
- \(\Phi\)가 intensive한 이유와 총 Euclidean action과의 관계
- 허용오차 \(\delta\)를 정하는 독립 likelihood

Gamma와 equicorrelation 계산은 이 자료를 가정한 benchmark로 보존하며,
관측 중심값과 우연히 가까운 수치만으로 mode 구조의 예측으로 승격하지
않는다.
