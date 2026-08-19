# 계량–라우팅 식 수학 감사

Status: COMPLETE

## 1. 먼저 분리해야 하는 네 공간

물리적 피질 표면 $M$, source neural observation space $X_A$, source latent chart $Z$, 미래 output space $O$는 서로 다른 공간이다.

$$
M\xrightarrow{\text{measurement}}X_A
\xrightarrow{\psi_j}Z_j,
\qquad
Z_j\xrightarrow{p(o\mid z,h,c)}\mathcal P(O).
$$

물리적 embedding $r:M\to\mathbb R^3$가 만드는

$$
h_{ab}=\partial_a r\cdot\partial_b r
$$

와 $Z_j$ 위에서 아래에 정의할 $G$는 정의역과 단위가 다르다. 식별 map 없이 $G=h$, $G-h$, 또는 “기능 계량이 물리 곡률을 바꾼다”는 식은 정의되지 않는다.

## 2. 조건부 Fisher pullback

### 2.1 정의

세션, 문맥, nuisance history를 고정한 output likelihood를

$$
p_j(o\mid z,h,c)
$$

라 한다. 적용조건은 $p_j$의 $z$-미분 가능성, finite score second moment, 미분과 기대값의 교환 가능성이다. Gaussian 특수형에서는 $\mu,\Sigma\in C^1$이고 $\Sigma=\Sigma^T\succ0$라고 가정한다. score covector는

$$
s_a(o;z,h,c)=\partial_a\log p_j(o\mid z,h,c)
$$

이고, 조건부 Fisher tensor는

$$
G_{ab}(z,c)
=\mathbb E_{h\mid z,c}\mathbb E_{o\mid z,h,c}[s_as_b].
\tag{1}
$$

임의의 tangent vector $v$에 대해

$$
v^TGv=\mathbb E[(v^as_a)^2]\ge0
$$

이므로 $G\succeq0$다. $v^as_a=0$ almost surely인 output-null direction이 있으면 $v^TGv=0$이므로 이것은 Riemannian metric이 아니라 pseudometric이다.

### 2.2 공변 reference regularization

SPD가 필요한 계산에서는

$$
G_\lambda=G+\lambda G_{\rm ref},
\qquad G_{\rm ref}\succ0,
\quad\lambda>0
\tag{2}
$$

를 쓴다. $G_{\rm ref}$는 independent calibration에서 고정한 동일 종류의 $(0,2)$ tensor여야 한다. 좌표를 바꿀 때 $G_{\rm ref}$도 함께 변환한다. 모든 chart에 숫자 행렬 $I$를 그대로 더하면 일반 $GL(d)$ 공변성이 깨진다.

$G$와 $G_{\rm ref}$의 단위가 같으므로 $\lambda$는 무차원이다. 작은 generalized eigenvalue가 reference 항에 의해 지배되는 비율을 보고하지 않으면 “데이터가 만든 기하”와 “regularizer가 만든 기하”를 구별할 수 없다.

### 2.3 Gaussian 전개

$$
o\mid z,h,c\sim\mathcal N(\mu(z,h,c),\Sigma(z,h,c))
$$

라 하자. Gaussian score의 평균 관련 항과 covariance 관련 항을 전개하고 odd centered moment가 0임을 쓰면

$$
\begin{aligned}
I_{ab}(z,h,c)
&=(\partial_a\mu)^T\Sigma^{-1}(\partial_b\mu)\\
&\quad+\frac12\operatorname{tr}
\left[
\Sigma^{-1}(\partial_a\Sigma)
\Sigma^{-1}(\partial_b\Sigma)
\right],\\
G_{ab}(z,c)&=\mathbb E_{h\mid z,c}[I_{ab}(z,h,c)].
\end{aligned}
\tag{3}
$$

따라서

$$
J_\mu^T\Sigma^{-1}J_\mu
\tag{4}
$$

만 사용하는 mean-response pullback은 $\partial_a\Sigma=0$이라는 homoscedastic 가정 아래의 특수형이다. 이 가정은 결과를 보고 선택할 수 없다.

## 3. 좌표변환 정리

### 3.1 국소 chart 변환

$z'=\phi(z)$가 미분동형이고 $J=\partial z'/\partial z$라 하자. chain rule에서

$$
\frac{\partial}{\partial z'^\alpha}
=(J^{-1})^a_{\ \alpha}\frac{\partial}{\partial z^a}.
$$

따라서 score는 covector로 변하고,

$$
G'(z',c)=J^{-T}G(z,c)J^{-1}.
\tag{5}
$$

$dz'=Jdz$이므로

$$
dz'^TG'dz'=dz^TGdz.
\tag{6}
$$

식 (5)--(6)은 정의 (1)의 tensor law다. 선형 $z'=Az$는 특수 경우다.

### 3.2 수치 손검산

$$
G=\begin{pmatrix}2&0\\0&3\end{pmatrix},\quad
A=\begin{pmatrix}2&1\\0&1\end{pmatrix},\quad
dz=\begin{pmatrix}1\\-2\end{pmatrix}
$$

이면 $dz^TGdz=14$다. 식 (5)로

$$
G'=\begin{pmatrix}0.5&-0.5\\-0.5&3.5\end{pmatrix},
\qquad dz'=\begin{pmatrix}0\\-2\end{pmatrix}
$$

이고 $dz'^TG'dz'=14$다.

### 3.3 AIRM의 허용 범위

동일한 tangent space에 놓인 두 SPD tensor의 generalized eigenvalue를 $\rho_i>0$라 하면

$$
d_{\rm AI}(G_0,G_1)
=\left(\sum_i\log^2\rho_i\right)^{1/2}
=\left\|\log(G_0^{-1/2}G_1G_0^{-1/2})\right\|_F.
\tag{7}
$$

공통 congruence 아래 generalized spectrum이 보존되므로 (7)은 불변이다. 단 다음 조건이 필요하다.

1. 같은 상태점의 두 context tensor이거나, 두 점 사이 transport가 사전 지정되어야 한다.
2. 전역 Euclidean latent chart를 공통 trivialization으로 쓸 경우 그 선택을 명시해야 한다.
3. 서로 다른 session/animal의 latent axes에는 alignment map과 불확실성 없이 raw SPD 거리·평균을 쓰지 않는다.
4. Karcher mean도 동일한 식별 조건을 만족한 tensor에만 쓴다.

실제 field summary에는 모든 context에서 공통인 calibration measure $q_j(z)$를 사전 고정하고

$$
\bar G_{j,c}
=\arg\min_{Q\succ0}
\mathbb E_{z\sim q_j}
d_{\rm AI}^2\!\left(Q,G_{\lambda,j,c}(z)\right)
\tag{7a}
$$

를 쓴다. 이 식도 공통 chart/trivialization 또는 사전 지정 transport가 있을 때만 허용된다. 동물간 통합은 우선 session 내부의 $d_{\rm AI}$, scale, shape 같은 불변 scalar를 만든 뒤 animal-level hierarchy로 수행한다.

## 4. 무차원 감사

확률밀도는 기준 measure에 의존하지만, score의 $z$ 미분과 동일 target measure에서의 log-density 차이는 기준 measure 상수를 제거한다.

$$
[\partial_a\log p]=[z_a]^{-1},
\qquad
[G_{ab}]=[z_a]^{-1}[z_b]^{-1}.
$$

따라서

$$
[ds^2]=[dz^aG_{ab}dz^b]=1.
\tag{8}
$$

Gaussian 평균항에서도

$$
[\partial_a\mu]=[o][z_a]^{-1},
\qquad [\Sigma^{-1}]=[o]^{-2}
$$

이므로 식 (3)은 같은 단위를 가진다. $G_0^{-1}G_1$의 generalized eigenvalue와 그 log는 무차원이며 AIRM도 무차원이다.

held-out routing score는 같은 target measure에 대한 log-density 차이의 평균이므로 nat/sample이다. nat은 무차원 정보 단위다. $\ell,\delta,H$는 bin 또는 초의 차원을 가지며, sampling rate가 다른 기록을 합칠 때 초로 환산한다.

## 5. 기존 $C^{-1}$ 후보의 정확한 지위

고정된 affine chart에서

$$
C=\operatorname{Cov}(z),\qquad g_C=C^{-1}
\tag{9}
$$

는 Mahalanobis precision을 정의하고 linear reparameterization에서는 congruence law를 따른다. 그러나 다음 이유로 일반 local Riemann tensor가 아니다.

### 5.1 비선형 chart 완전 반례

$Z\sim\mathcal N(0,1)$, $z'=z+az^3$ ($a>0$)라 하자. 원점에서 $dz'/dz=1$이므로 local tensor law라면 원점 metric 성분은 변하지 않아야 한다. 그러나

$$
\operatorname{Var}(z')=1+6a+15a^2
$$

이므로 $\operatorname{Var}(z')^{-1}\ne\operatorname{Var}(z)^{-1}$. Global covariance inverse는 일반 nonlinear chart의 local tensor가 아니다.

### 5.2 output 구별 실패 반례

두 시스템 모두 $C=I_2$라 하자. 첫 시스템은 $o=z_1+\epsilon$, 둘째는 $o=z_2+\epsilon$이다. 두 경우 $C^{-1}=I_2$는 같지만 output Fisher는 각각

$$
\operatorname{diag}(\sigma^{-2},0),
\qquad
\operatorname{diag}(0,\sigma^{-2})
$$

로 다르다. 따라서 $C^{-1}$은 어떤 미래 output이 접근 가능한지 정하지 않는다. 또한 $C\leftrightarrow C^{-1}$은 일대일이므로 precision은 covariance에 없던 정보를 추가하지 않는다.

## 6. 라우팅 식

target history와 nuisance를

$$
\mathcal H_t^B=\{x^B_{\le t},h_{\le t}\}
$$

라 한다. nested held-out model을

$$
\begin{aligned}
p_0&=p(x^B_{t+\delta}\mid\mathcal H_t^B,c),\\
p_1&=p(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c)
\end{aligned}
$$

로 고정하면

$$
R_{j,c}^{A\to B}(\ell,\delta)
=\frac1{N_{\rm test}}
\sum_{t\in\rm test}\left(\log p_1-\log p_0\right)
\tag{10}
$$

이다. finite model, penalty, tuning budget는 두 nested model 사이에서 사전 고정해야 한다. route의 존재와 context modulation을 분리한다. 두 model은 context baseline $b_c$를 모두 포함하고, source-history effect만 공유 $f$와 context-specific $f_c$로 다르게 둔다.

$$
\begin{aligned}
p_{\rm static}&=p(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c;\ b_c+f(z^A_{t-\ell:t})),\\
p_{\rm interaction}&=p(x^B_{t+\delta}\mid\mathcal H_t^B,z^A_{t-\ell:t},c;\ b_c+f_c(z^A_{t-\ell:t})).
\end{aligned}
$$

그러면

$$
\Delta R_{\rm ctx}
=\frac1{N_{\rm test}}\left[
\operatorname{ELPD}(p_{\rm interaction})
-\operatorname{ELPD}(p_{\rm static})
\right]
\tag{11}
$$

을 별도 primary estimand로 둔다. 따라서 식 (10)과 (11)은 모두 nat/sample이다.

가역 source reparameterization 뒤 model family를 정확히 pushforward/pullback하면 조건부 분포가 같으므로 (10)은 불변이다. 유한 ridge, spline, neural net, truncated latent는 동일 family가 아닐 수 있으므로 실제 분석에서는 affine transform audit와 refit parity가 필요하다.

## 7. 비식별 정리와 완전 반례

### 7.1 같은 $G$, 다른 $R$

$$
o=z+\epsilon,\quad \epsilon\sim\mathcal N(0,1)
$$

이면 $G=1$이다. target을

$$
x^B_{t+1}=rz_t+\eta_t
$$

로 두면 $r=0$에서 $R=0$, $r\ne0$에서 $R>0$가 가능하다. 같은 output geometry가 route를 정하지 않는다.

### 7.2 같은 $R$, 다른 $G$

target 식 $x^B_{t+1}=az_t+\eta_t$를 고정하고 output을 $o=\beta z+\epsilon$으로 둔다. route model은 그대로지만

$$
G=\beta^2/\sigma_\epsilon^2
$$

이므로 $\beta$에 따라 metric이 달라진다. route가 output geometry를 정하지 않는다.

### 7.3 common-input 반례

$$
U_{t+1}=\rho U_t+\xi_t,
\qquad x_t^A=U_t+\epsilon_t^A,
\qquad x_{t+1}^B=U_{t+1}+\epsilon_{t+1}^B.
\tag{12}
$$

$A\to B$ edge는 없지만 $A_t$가 hidden $U_t$의 추가 측정치라서 $R^{A\to B}>0$일 수 있다. 그러므로 (10)은 causal routing이 아니다.

### 7.4 mediation 비식별 반례

1차원 metric 성분의 양성을 보존한 완전 매개 구조

$$
G=\exp(S+U),\qquad X=\log G+E
$$

와 비매개 구조

$$
G=\exp(S+U),\qquad X=S+U+E
$$

는 모두 $G>0$이고 관측상 $X=S+U+E$를 만들어 $(S,G,X)$의 같은 joint distribution을 낼 수 있다. 독립적인 $do(G)$, exclusion restriction 또는 충분한 causal state model 없이는 $S\to G\to X$를 식별할 수 없다.

### 7.5 곡률 비식별

Riemann curvature는 metric field의 공간 미분을 요구한다. 하나의 SPD 행렬이나 context별 상수 행렬에는 필요한 local derivative 정보가 없다. 좌표 pullback 때문에 Christoffel symbol이 0이 아니어도 flat space의 Riemann tensor는 0일 수 있다. 따라서 “비대각 SPD”, “AIRM 변화”, “비영 Christoffel”은 곡률 증거가 아니다.

## 8. 결합의 유일한 현재 허용형

$$
\mathcal B_{j,c}^{A\to B}(z)
=\left(G_{j,c}^{o\leftarrow A}(z),R_{j,c}^{A\to B}\right)
\tag{13}
$$

는 두 추정량을 같은 관측 프로토콜 아래 나란히 둔 ordered pair다. context contrast도

$$
\Xi_j
=\left(
d_{\rm AI}(\bar G_{j,c_0},\bar G_{j,c_1}),
\Delta R_{j,\rm ctx}^{A\to B}
\right)
\tag{14}
$$

로 보고한다. 두 성분이 모두 무차원이라는 사실은 서로 더할 근거가 아니다. $\alpha d_{\rm AI}+\beta\Delta R$는 metric 1단위와 routing 1 nat의 교환비 $\alpha/\beta$라는 새 공리를 요구한다.

## 9. 형식 지위

| 항목 | 지위 | 허용되는 문장 |
|---|---|---|
| 식 (1), (5), (8) | **[정의]+[정리]** | output-relative Fisher tensor와 그 좌표·단위 법칙 |
| 식 (2) | **[공리: regularization 선택]** | calibration-only SPD reference를 택한 조건부 metric |
| 식 (3) | **[정리]** | Gaussian conditional family의 정확한 Fisher 전개 |
| 식 (9) | **[경험식 후보]** | 고정 affine selectivity chart의 상대 precision |
| 식 (10)--(11) | **[정의: 경험 추정량]** | held-out lagged conditional predictive transfer |
| 식 (13)--(14) | **[정의]** | metric과 route의 공동 보고 형식 |
| $G\to R$, $W\to G\to x$ | **[미완성]** | 독립 개입 전에는 검정되지 않은 가설 |
| curvature, geodesic dynamics | **[미완성]** | smooth field, transport, held-out dynamics가 필요 |
| SCC=기억/의식 | **[삭제]** | 현재 정본에 생물학적 동일시를 두지 않음 |

## 10. 수학 판정

계량과 라우팅을 분리한 식 (13)은 좌표·단위·식별 경계를 만족한다. 단, 이것은 아직 실제 뇌 결과가 아니라 실제자료 검정을 가능하게 만드는 정의 체계다.

$$
\boxed{
\text{현재 닫힌 것은 식의 일관성이고, 열려 있는 것은 실제 뇌에서의 경험적 성립과 인과 생산자다.}
}
$$
