# 뇌·양자·분지 재귀의 다중시간척도 방정식 초안

Status: DRAFT-FOR-MATH-AUDIT  
Date: 2026-08-20

## 1. 목적과 금지 동일시

이 문서는 BrainRuntime의 빠른 상태 재귀, 관측된 신경사건의 계보, 느린
가소성, 양자 instrument의 고전 기록 경계를 하나의 typed system으로
정의한다. 다음 두 행렬은 동일하지 않다.

\[
W^{(r)}\in\mathbb R^{N\times N}
\quad\text{signed recurrent coupling},
\]

\[
A^{(r)}\in\mathbb R_+^{K\times K}
\quad\text{nonnegative event-offspring expectation}.
\]

\(W=A\), \(A=|W|\), \(A=W^{\odot2}\)는 모두 금지한다. 우주론의
\(D_{\rm eff}\)와 \(q_{\rm ext}\)도 뇌 parameter나 활성 목표로 대입하지
않는다.

## 2. 인덱스, 상태와 세 시간척도

- \(i,j=1,\ldots,N\): runtime cell index
- \(u\in M\), \(a,b=1,2\): 피질 또는 기능적 표면의 물질좌표와 접공간 지표
- \(n\): 빠른 runtime tick, 물리 간격 \(\Delta_B>0\)
- \(\ell\): 사건 계보 추정 block
- \(r\): 느린 weight-update epoch
- \(k=1,\ldots,K\): 사전 고정한 event type

연속 상태는

\[
z_i^n=(a_i^n,r_i^n,m_i^n,w_i^n)
\in[-1,1]\times\mathbb R_+ \times\mathbb R\times\mathbb R_+
\]

이고 hysteresis state는 \(b_i^n\in\{0,1\}\)다. STP state는
\(U_j^n,X_j^n\in[0,1]\)다. 한 계보 추정 block 안에서는
\(W^{(r)}\), mode \(M\), threshold, delay rule과 input normalization을
동결한다.

## 3. 연결 방향과 Dale 조건

행렬 방향을

\[
W_{ij}:\quad j\text{ sender/pre}\longrightarrow i\text{ receiver/post}
\]

로 고정한다. sender type \(\epsilon_j\in\{+1,-1\}\)에 대해

\[
W_{ij}^{(r)}=\epsilon_j\,\lvert V_{ij}^{(r)}\rvert,\qquad
V_{ij}^{(r)}\ge0
\]

가 Dale 조건이다. 즉 부호는 column \(j\)에 붙는다. receiver row에
부호를 붙이는 현재 Python 구현은 이 정의와 모순이며 parity gate가
닫히기 전까지 BR-1은 구현 차단 상태다.

## 4. 선택적 양자 입력 경계

양자 상태는 뇌 상태에 직접 더하지 않는다. outcome \(y\)를 가진 instrument
\(\{\mathcal I_y\}\)에 대해

\[
\mathcal I_y:\mathsf D(\mathcal H)\to
\mathsf L_+(\mathcal H),\qquad
\sum_y\mathcal I_y\ \text{is CPTP},
\]

\[
p(y_n\mid\rho_n)=\operatorname{Tr}\mathcal I_{y_n}(\rho_n),
\qquad
\rho_{n+1\mid y_n}
=\frac{\mathcal I_{y_n}(\rho_n)}
{\operatorname{Tr}\mathcal I_{y_n}(\rho_n)}
\]

로 기록을 만든다. runtime input은

\[
u^n=u_{\rm ext}^n+L\,\psi(y_n)
\]

이다. \(\psi(y_n)\), \(L\psi(y_n)\)와 \(u^n\)은 runtime normalization
뒤 무차원이어야 한다. outcome을 버리고 nonselective CPTP map만 남기면
고전 event genealogy는 정의되지 않는다.

## 5. 빠른 BrainRuntime 상태식

### 5.1 drive

\[
I_i^n
=u_i^n
+\sum_jW_{ij}^{(r)}U_j^nX_j^n
a_j^{\,n-d_{ij}}\,\chi_j^n
-\lambda_r r_i^n+\lambda_m m_i^n-\beta_w w_i^n+\eta_i^n .
\tag{B1}
\]

\(d_{ij}\in\mathbb N_0\)은 edge별 delay이고
\(\chi_j^n\in\{0,1\}\)은 하나로 통일된 source-admission mask다.
\(\eta_i^n\)도 runtime scale로 정규화한다.

### 5.2 bounded activation

현재 문서의 \(\kappa_a>\gamma_a\) runtime도 상태범위를 지키게 하려면
projection을 식에 명시한다.

\[
a_i^{n+1}
=\Pi_{[-1,1]}\!\left[
(1-\gamma_a)a_i^n+\kappa_a\tanh(I_i^n)
\right].
\tag{B2}
\]

projection을 쓰지 않는 route는
\(0<\kappa_a\le\gamma_a\), \(|a_i^0|\le1\) 같은 더 강한 조건을
필요로 한다.

### 5.3 sequential auxiliary updates

현재 hot path와 맞추기 위해 새 activation을 사용한다.

\[
r_i^{n+1}
=(1-\gamma_r)r_i^n+\kappa_r(a_i^{n+1})^2,
\tag{B3}
\]

\[
m_i^{n+1}
=(1-\gamma_m)m_i^n+\gamma_m a_i^{n+1},
\tag{B4}
\]

\[
w_i^{n+1}
=\Pi_{[0,w_{\max}]}\!\left[
(1-\gamma_w)w_i^n+\kappa_w(a_i^{n+1})^2
\right].
\tag{B5}
\]

### 5.4 hysteresis

\[
b_i^{n+1}
=
\begin{cases}
1,&a_i^{n+1}>\theta_i^+,\\
0,&a_i^{n+1}<\theta_i^-,\\
b_i^n,&\theta_i^-\le a_i^{n+1}\le\theta_i^+,
\end{cases}
\qquad \theta_i^-<\theta_i^+ .
\tag{B6}
\]

이 이산 state 때문에 전체 hybrid system은 continuous contraction만으로
유일 attractor를 보장하지 않는다.

### 5.5 STP

\[
X_j^{n+1}
=\Pi_{[0,1]}\!\left[
X_j^n+\frac{\Delta_B}{\tau_{\rm rec}}(1-X_j^n)
-U_j^nX_j^n e_j^n
\right],
\tag{B7}
\]

\[
U_j^{n+1}
=\Pi_{[0,1]}\!\left[
U_j^n+\frac{\Delta_B}{\tau_{\rm fac}}(U_0-U_j^n)
+U_0(1-U_j^n)e_j^n
\right].
\tag{B8}
\]

\(e_j^n\in\{0,1\}\)이고 update order를 고정한다. projection이
\(U,X\in[0,1]\)을 보존하며, projection ablation route는
\(\Delta_B/\tau\)와 event jump의 별도 invariant-set 증명을 요구한다.

## 6. 빠른 상태의 충분 수축조건

mode, \(W\), event mask를 동결하고

\[
\eta^n=0,\qquad u^n=u_\star
\]

를 둔다. delay buffer는 고정점 history와 일치시키거나 augmented state에
포함한다. 다음 상계를 둔다.

\[
\overline W
:=\sup_n\left\|
W^{(r)}\operatorname{diag}(U^n\odot X^n\odot\chi^n)
\right\|_\infty .
\]

activation row의 gain vector는

\[
g=
\left(
\lvert1-\gamma_a\rvert+\kappa_a\overline W,\,
\kappa_a\lambda_r,\,
\kappa_a\lambda_m,\,
\kappa_a\beta_w
\right).
\]

sequential update (B3)--(B5)를 반영한 비음수 gain matrix를

\[
\mathcal G=
\begin{pmatrix}
g\\
2\kappa_r g+(0,\lvert1-\gamma_r\rvert,0,0)\\
\gamma_m g+(0,0,\lvert1-\gamma_m\rvert,0)\\
2\kappa_w g+(0,0,0,\lvert1-\gamma_w\rvert)
\end{pmatrix}
\tag{B9}
\]

로 정의한다. projection, tanh와 \(a^2\)가 지정 영역에서 각각
1, 1, 2-Lipschitz이므로

\[
\rho(\mathcal G)<1
\tag{B10}
\]

은 위 autonomous, zero-noise, delay-consistent 조건에서 continuous state
\((a,r,m,w)\)의 유일 고정점과 수렴을 주는 충분조건 후보다. bounded
time-varying input과 noise가 있으면 결론은 고정점 수렴이 아니라
incremental contraction 또는 ISS bound로 낮춘다. threshold crossing,
mode switch와 plasticity를 포함한 전체 runtime의 전역 수렴 정리는 아니다.
B10은 \(U,X\)와 delay history를 외생적으로 고정한
\((a,r,m,w)\) 부분계에만 적용한다. 이들을 동적으로 갱신하는 전체계에는
STP와 delay shift block을 포함한 augmented gain matrix가 필요하다.

## 7. 하나로 통일할 사건 정의

현재 lifecycle mask, post-state TopK와 STDP threshold는 서로 다른 event다.
계보식에는 하나의 사전 고정 predicate만 허용한다. 뉴런마다 baseline,
변동 폭과 역치가 다르므로 공통 raw threshold를 모든 세포에 적용하지
않는다. calibration-only block에서 얻고 평가 block 동안 동결한
\(\mu_i^{(r)}\), \(s_i^{(r)}>0\), \(\vartheta_i^{(r)}\)를 사용해

$$
\zeta_i^n
=\frac{\lvert a_i^n\rvert-\mu_i^{(r)}}
{\max(s_i^{(r)},s_{\min})},
\qquad
e_i^n
=\mathbf1\!\left[\zeta_i^n\ge\vartheta_i^{(r)}\right]
\tag{E1}
$$

로 정의한다. 여기서 \(s_{\min}>0\)은 사전 고정한 무차원 floor다. 역치
초과 강도도 버리지 않고

$$
q_i^n
=e_i^n\min\!\left\{1,
\frac{[\zeta_i^n-\vartheta_i^{(r)}]_+}{q_{\rm sat}}
\right\}
\in[0,1]
\tag{E1a}
$$

로 별도 기록한다. \(q_{\rm sat}>0\)도 무차원 calibration 상수다.
\(e_i^n\)는 사건 발생 여부이고 \(q_i^n\)는 발생한 사건의 상대 강도이므로
서로 대체하지 않는다. calibration block에 held-out target이나 decoder
결과를 넣지 않는다.

activation 부호가 firing magnitude가 아니라 별도 의미를 가진다면 양·음
event type을 분리한 polarity-aware predicate로 E1을 교체해야 한다. 교체
후에는 STDP, genealogy와 source admission이 같은 \((e,q)\) receipt를
소비해야 한다. 뉴런별 역치를 평가 자료의 발화율에 맞춰 사후 조정하면
회로 강도와 비교 결과가 함께 바뀌므로 apparatus leakage다.

## 8. 인과 계보 receipt와 offspring estimator

각 child event \(e_i^{n+1}\)에 대한 parent responsibility를

\[
C_{ij}^n\in[0,1],\qquad
C_{ij}^n=0\ \text{if }e_j^{n-d_{ij}}e_i^{n+1}=0,
\qquad
\sum_jC_{ij}^n\le e_i^{n+1}
\tag{E2}
\]

로 정의한다. 마지막 부등식은 한 child를 여러 parent에게 완전 중복
계상하는 것을 막는다. \(C\)는 randomized intervention, predeclared
single-parent tie rule 또는 검증된 responsibility model에서 얻어야 한다.
CSR support나 signed weight만으로 만들지 않는다.

edge별 관측가능 index를

\[
\mathcal O_{ij,r}
:=
\left\{
n:\ n-d_{ij}\in\mathcal B_r,\ n+1\in\mathcal B_r,\ 
\text{source, arrival, child receipt가 모두 완전함}
\right\}
\tag{E3a}
\]

로 둔다. 이는 left/right boundary censoring과 incomplete receipt를
분모에서 제외한다. 표준 branching 방향을 parent row, child column으로
두면 descriptive estimator는

\[
\widehat A_{ji}^{(r)}
=
\frac{\sum_{n\in\mathcal O_{ij,r}}C_{ij}^n}
{\sum_{n\in\mathcal O_{ij,r}}e_j^{n-d_{ij}}}
\ge0
\tag{E3}
\]

이다. 분모 0 parent type은 missing으로 남기며 0으로 채워 PASS시키지
않는다. block bootstrap 또는 사전 고정한 count-process interval을 함께
보고한다. receipt completeness, censoring, conditional independence와
Poisson goodness-of-fit gate 전에는 E3을 E4의 generative parameter로
승격하지 않는다. \(W_{ij}\)와 \(A_{ji}\)는 방향도 transpose 관계이고
의미도 다르다.

receipt가 완전한 edge의 비음수 전달 강도는

$$
\omega_{ij}^n
=C_{ij}^n q_j^{\,n-d_{ij}}q_i^{\,n+1}
U_j^{\,n-d_{ij}}X_j^{\,n-d_{ij}}
\rho_{ij}^{(r)}\ge0,
\qquad
\rho_{ij}^{(r)}
=\min\!\left\{\rho_{\max},
\frac{|W_{ij}^{\rm syn,(r)}|}{W_{\rm ref}}
\right\}
\tag{E3b}
$$

로 정의할 수 있다. \(W_{\rm ref}>0\)는 weight와 같은 단위의 frozen
reference이고 \(\rho_{\max}\)는 무차원 cap이다. 식 E3b는
\(C_{ij}^n>0\)인 인과 receipt가 먼저 존재할 때만 사용한다. signed weight
support만으로 \(C\)나 \(\omega\)를 만들지 않는다. 여기의
\(W^{\rm syn}\)은 G7의 양의 routing gain을 곱하기 전 structural weight다.
geometry 효과는 \((e,q,C)\)에 이미 반영되므로 magnitude 항에서 다시
곱하지 않는다. block별 edge 강도는

$$
\alpha_{ij}^{(r)}
=\frac1{|\mathcal O_{ij,r}|}
\sum_{n\in\mathcal O_{ij,r}}\omega_{ij}^n,
\qquad
\alpha_{ij}^{(r)}\in[0,\rho_{\max}]
\tag{E3c}
$$

로 정의하고 block bootstrap 불확도를 함께 보고한다. 회로 안의 edge
차이를 평균 하나로 지우지 않는다.

## 9. 정확한 Poisson 계보 route

parent type \(j\)가 child type \(i\)를 독립적으로
\(\operatorname{Poisson}(A_{ji})\)개 만든다는 추가 공리 아래

\[
G_j(\boldsymbol q)
=
\exp\!\left[
-\sum_iA_{ji}(1-q_i)
\right],
\tag{E4}
\]

\[
\boldsymbol q^{(0)}=\boldsymbol0,\qquad
\boldsymbol q^{(h+1)}=G(\boldsymbol q^{(h)})
\tag{E5}
\]

이다. \(\boldsymbol q^{(h)}\)는 \(h\)세대 안의 소멸확률이고
\[
\boldsymbol q_\infty
=\lim_{h\to\infty}\boldsymbol q^{(h)}
=\min\operatorname{Fix}_{[0,1]^K}G .
\tag{E6}
\]

Jacobian은

\[
J_G(\boldsymbol q)=\operatorname{diag}(\boldsymbol q)A .
\tag{E7}
\]

irreducible \(A\)에서 \(\rho(A)\le1\)이면
\(\boldsymbol q_\infty=\boldsymbol1\)이고,
\(\rho(A)>1\)이면 비자명 생존확률이 생긴다.

## 10. 뇌에 적합한 유한-horizon readout

정상적인 subcritical event process는 \(\rho(A)<1\)을 요구한다. 이때
무한시간 소멸확률은 자명하게

\[
\boldsymbol q_\infty=\boldsymbol1 .
\]

따라서 뇌의 cascade persistence readout은 우주론의 비자명 최소근이
아니라

\[
\boldsymbol s_H
:=\boldsymbol1-\boldsymbol q^{(H)}
\tag{E8}
\]

로 정의한다. \(H\)와 bin width는 사전 고정한다.
\(\rho(A)\uparrow1\)이면 장기 finite-horizon persistence가 커질 수 있지만,
이를 의식, 기억 또는 임계 최적성으로 읽는 것은 별도 실험가설이다.

## 11. E/I Hawkes 대안 route

signed recurrence를 point-process로 표현하려면 excitation과 inhibition을
분리한다.

\[
\lambda_i(t)
=\lambda_{0i}\,
\varphi_i\!\left(
\beta_i+
\sum_j\int_0^\infty k^+_{ji}(\tau)\,dN_j(t-\tau)
-\sum_j\int_0^\infty k^-_{ji}(\tau)\,dN_j(t-\tau)
\right),
\tag{H1}
\]

여기서 \(\lambda_{0i}\)는 \(T^{-1}\), \(\beta_i\)와 적분 안의 합은
무차원, \(k^\pm\ge0\)다. \(\varphi_i\)의 Lipschitz 상수를 \(L_i\)라 하면

\[
K_{ji}
:=
L_i\lambda_{0i}
\int_0^\infty\left(k^+_{ji}(\tau)+k^-_{ji}(\tau)\right)d\tau
\tag{H2}
\]

는 무차원 majorant다. 조건

\[
\rho(K)<1
\tag{H3}
\]

은 nonlinear Hawkes 안정성의 충분조건 route다.

Poisson cluster extinction 식 E4는 선형 비음수 excitation kernel의
cluster representation 또는 독립 genealogy가 있을 때만 정확하다.
일반 E/I nonlinear Hawkes에 E4를 그대로 적용하지 않는다.

## 12. 적격 흔적과 느린 가소성

이 절은 STDP orientation이 causal일 때만 사용하며 row post \(i\),
column pre \(j\)를 보존한다. 현재 legacy orientation은 별도 호환 route이고
P2와 동일한 식이 아니다.

\[
P_j^{n+1}=\alpha_+P_j^n+e_j^n,\qquad
Q_i^{n+1}=\alpha_-Q_i^n+e_i^n,
\tag{P1}
\]

\[
E_{ij}^{n+1}
=\alpha_eE_{ij}^n
+A_+e_i^nP_j^n
-A_-Q_i^n e_j^n .
\tag{P2}
\]

\[
\alpha_\bullet
=\exp(-\Delta_B/\tau_\bullet)
\tag{P3}
\]

이므로 모든 지수 인자는 무차원이다. 한 block 동안 \(W\)를 동결하고
block 끝에서만

\[
W^{(r+1)}
=
\Pi_{\mathcal D}\!\left[
W^{(r)}+\eta_W g_r E_r
\right]
\tag{P4}
\]

를 적용한다. \(\Pi_{\mathcal D}\)는 weight bound, sparsity와 sender-column
Dale sign을 보존한다. \(g_r\)는 사전 정의한 dimensionless scalar이고,
zero-gate와 sign-flip control이 필요하다.

항상성 scaling을 쓰면 우주론 \(q_{\rm ext}\)가 아니라 독립적으로 측정하거나
개발 split에서 고정한 target \(f_i^\star\)를 사용한다.

\[
\bar f_i^{r+1}
=(1-\alpha_h)\bar f_i^r
+\alpha_h\frac1{|\mathcal B_r|}
\sum_{n\in\mathcal B_r}e_i^n,
\tag{P5}
\]

\[
\widetilde W_{ij}^{(r+1)}
=W_{ij}^{(r+1)}
\exp\!\left[\eta_h(f_i^\star-\bar f_i^{r+1})\right].
\tag{P6}
\]

P6의 양의 row factor는 sender-column Dale sign을 바꾸지 않는다.

## 13. 시간척도 분리

block-end update를 기본 route로 둔다.

\[
\tau_{\rm fast},\tau_{\rm Hawkes}
\ll T_{\rm block}
\ll T_W ,
\tag{T1}
\]

\[
\sup_r\rho(\mathcal G_r)<1,\qquad
\sup_r\rho(K_r)<1,\qquad
\eta_W\lVert g_rE_r\rVert\ll1 .
\tag{T2}
\]

동시 stochastic approximation을 쓰는 대안 route는 step sizes
\(\alpha_r\)와 \(\eta_r\)에 대해

\[
\sum_r\alpha_r=\sum_r\eta_r=\infty,\qquad
\sum_r(\alpha_r^2+\eta_r^2)<\infty,\qquad
\frac{\eta_r}{\alpha_r}\to0
\tag{T3}
\]

같은 두 시간척도 조건과 uniform ergodicity를 별도로 증명해야 한다.
작은 learning rate만으로 averaging 정리가 생기지 않는다.

## 14. 전체 자기참조 폐곡선

한 slow epoch의 typed composition은

\[
\rho_n
\xrightarrow{\mathcal I_{y_n}}
y_n
\xrightarrow{\psi}
u^n
\xrightarrow{\mathcal B_{W^{(r)}}}
z^{n+1}
\xrightarrow{\text{E1--E3}}
\widehat A^{(r)}
\xrightarrow{\text{E4--E8}}
\boldsymbol s_H^{(r)}
\]

\[
z^{0:T},e^{0:T}
\xrightarrow{\text{P1--P3}}
E_r
\xrightarrow{\text{P4}}
W^{(r+1)}
\xrightarrow{\mathcal B_{W^{(r+1)}}}
\text{next trajectory}
\tag{C1}
\]

다. 이것이 뇌와 분지 재귀가 실제로 만나는 자기참조 구조다. 현재
\(\boldsymbol s_H\)는 audit readout이며 P4의 gate에 넣지 않는다.
readout을 controller로 되먹이는 것은 독립 preregistration 뒤의 다음
route다.

## 15. 이질적 회로의 기능적 계량장

앞 절의 재귀는 사건과 weight를 연결하지만 사건이 기능적 거리를 어떻게
바꾸는지는 정의하지 않았다. 이 절은 그 빈자리를 채우는 별도
**[공리: 모델 선택]**이다. 기하식의 수학적 정합성은 아래 정의에서
검사할 수 있지만, 실제 뇌와 Loop 8/9의 개선은 아직 **[미완성]**이다.

### 15.1 발달 기준기하와 실제 주름

피질의 물질표면을 \(M\), 발달시각 \(\tau\)의 3차원 embedding을
\(X_{\rm dev}(u,\tau):M\to\mathbb R^3\)라 둔다. 해부학적 기준계량과
제2기본형은

$$
g^{\rm dev}_{ab}
=\partial_aX_{\rm dev}\cdot\partial_bX_{\rm dev},
\qquad
b^{\rm dev}_{ab}
=n\cdot\partial_a\partial_bX_{\rm dev}
\tag{G1}
$$

이다. \(g^{\rm dev}\)는 표면 내부의 거리이고 \(b^{\rm dev}\)는 표면이
3차원에서 접힌 정도다. 아래의 회로 계량은 고정된 물질좌표 위의 기능적
거리이므로, 식 G1의 실제 피질 주름과 동일시하지 않는다.

### 15.2 edge 강도와 회로별 강도

block \(r\)에서 \(\alpha_{ij}^{(r)}>\alpha_{\min}\)인 receipt-complete
directed edge만 사용해 cycle 집합 \(\mathscr C_r\)를 만든다. 연결성과
차수 조건이 닫히지 않으면 하나의 Euler circuit라고 부르지 않고 cycle
decomposition으로 남긴다. 회로 \(\Gamma\in\mathscr C_r\)의 길이와
요약 강도는

$$
L_\Gamma=\sum_{(j,i)\in\Gamma}\ell_{ij},
\qquad
\mathcal A_\Gamma^{(r)}
=\frac1{L_\Gamma}
\sum_{(j,i)\in\Gamma}\ell_{ij}\alpha_{ij}^{(r)},
\qquad
\mathcal B_\Gamma^{(r)}
=\min_{(j,i)\in\Gamma}\alpha_{ij}^{(r)}
\tag{G2}
$$

로 정의한다. \(\mathcal A_\Gamma\)는 길이 가중 평균이고
\(\mathcal B_\Gamma\)는 병목 강도다. 실제 계량 선원은 평균만 쓰지 않고
edge별 \(\alpha_{ij}\)를 보존한다.

회로를 \(\gamma_\Gamma(s)\)로 매개화하고 단위 접벡터를 \(T^a\)라 하면

$$
S_{ab}^{\Gamma}(u,r)
=\sum_{(j,i)\in\Gamma}\alpha_{ij}^{(r)}
\int_{e_{ij}}T_aT_b\,
\delta_{g^{\rm dev}}(u,\gamma(s))\,ds
\tag{G3}
$$

는 \(L^{-1}\) 차원의 비음수 방향성 선원이다. 뉴런별 역치, 사건 초과
강도, STP, receipt responsibility와 weight magnitude 차이는 E1--E3c를
거쳐 G3에 들어간다. 흥분·억제의 Dale 부호를 기하 수축·팽창의 부호로
자동 변환하지 않는다.

### 15.3 회로별 영향 범위와 SPD 계량

각 회로에 무차원 결합 \(\kappa_\Gamma\ge0\)와 영향 길이
\(\xi_\Gamma>0\)를 둔다. 회로별 대칭 변형장은

$$
\left(I-\xi_\Gamma^2\Delta_L\right)
k_{ab}^{\Gamma}
=-2\kappa_\Gamma\xi_\Gamma S_{ab}^{\Gamma}
\tag{G4}
$$

를 만족하며 경계가 있으면 Dirichlet 또는 Neumann 조건을 사전에
고정한다. \(\Delta_L\)은 기준계량의 Lichnerowicz Laplacian이며 여기서는
\(-\Delta_L\succeq0\)인 부호 규약을 쓴다. 따라서
\(I-\xi_\Gamma^2\Delta_L\)은 frozen source에서 coercive다.
\(\xi_\Gamma S^\Gamma\), \(k^\Gamma\)와 \(\kappa_\Gamma\)는 모두
무차원이다. 여러 회로가 동시에 있으면

$$
k=\sum_{\Gamma\in\mathscr C_r}k^\Gamma
\tag{G5}
$$

로 합친다. additive \(g^{\rm dev}+k\)는 작은 변형의 1차 근사일 뿐이다.
큰 변형에서도 양의 정부호를 보존하기 위해 기능적 계량을

$$
g^{\rm eff}
=\left(g^{\rm dev}\right)^{1/2}
\exp\!\left[
\left(g^{\rm dev}\right)^{-1/2}
k
\left(g^{\rm dev}\right)^{-1/2}
\right]
\left(g^{\rm dev}\right)^{1/2}
\succ0
\tag{G6}
$$

로 정의한다. 등방성 검산 route에서는
\(g^{\rm eff}=e^{2\varphi}g^{\rm dev}\)로 축약할 수 있다.

### 15.4 synaptic weight와 routing gain의 분리

기하가 synaptic weight를 직접 덮어쓰지 않도록 기준거리
\(d_{\rm dev}(i,j)\)와 현재 geodesic distance \(d_g(i,j)\)에서 양의
routing gain을

$$
\mathcal R_{ij}(g)
=\Pi_{[R_{\min},R_{\max}]}
\exp\!\left[
-\frac{d_g(i,j)^2-d_{\rm dev}(i,j)^2}{\sigma_g^2}
\right],
\qquad
W_{ij}^{\rm eff}
=W_{ij}^{\rm syn}\mathcal R_{ij}(g)
\tag{G7}
$$

로 정의한다. 지수 인자는 무차원이고 \(\mathcal R_{ij}>0\)이므로
sender-column Dale 부호는 보존된다. \(W^{\rm syn}\)은 P4가 갱신하고
\(g\)는 receipt-complete activity만으로 갱신한다. 따라서 target,
decoder 또는 held-out endpoint가 G3--G7에 들어가면 apparatus invalid다.

전체 기능적 자기참조 route는

$$
z^{0:T}
\xrightarrow{\mathrm{E1--E3c}}
\alpha^{(r)}
\xrightarrow{\mathrm{cycle\ decomposition}}
\mathscr C_r
\xrightarrow{\mathrm{G3--G6}}
g^{\rm eff}
\xrightarrow{\mathrm{G7}}
W^{\rm eff}
\xrightarrow{\mathcal B}
z^{T+1:}
\tag{G8}
$$

이다. G4가 frozen source에서 유일한 elliptic 해를 갖는 것과 G8 전체가
수렴하는 것은 다른 명제다. 전체 결합계에는 runtime, event extraction과
geometry response를 함께 포함한 block Jacobian의 spectral radius 조건이
추가로 필요하다.

## 16. 완전 반례와 식별성 경계

1. projection 없는 기존 activation 식에서
   \(\gamma_a=0.18,\kappa_a=0.82\), 큰 양의 drive를 두 tick 유지하면
   \(a_1\approx0.82\), \(a_2\approx1.4924>1\)이다. bounded-state 부모
   주장의 반례다.
2. \(h_{n+1}=\tanh(-2h_n)\)은 bounded지만 2-cycle을 가질 수 있다.
   boundedness는 fixed-point convergence가 아니다.
3. 같은 \(W\)에 서로 다른 observation threshold/attribution을 쓰면
   서로 다른 \(A\)가 나온다.
4. 같은 \(A\)에 서로 다른 signed \(W\)를 붙이면 같은 offspring 통계와
   다른 state trajectory를 만들 수 있다.
5. \(\rho(A)\ge1\)이면 stationary linear Hawkes mean
   \((I-A)^{-1}\mu\) route가 닫히지 않는다.
6. 같은 nonselective CPTP channel도 서로 다른 instrument/unravelling
   기록을 가질 수 있다. CPTP만으로 genealogy는 식별되지 않는다.
7. \(q_{\rm ext}\to\Omega\), event branching에서 cosmological density,
   finite cascade persistence에서 memory/AGI로 가는 사상은 각각 별도
   물리·경험 bridge다.

## 17. 필수 검증

- sender-column Dale orientation과 Python/Rust parity
- STDP causal orientation과 legacy negative control
- B2의 state-bound 및 projection ablation
- B3--B5의 old/new activation update-order parity
- edge delay receipt와 zero-delay reduction
- E1 predicate를 STDP, source admission, genealogy가 동일하게 소비
- 뉴런별 \(\mu_i,s_i,\vartheta_i\)의 calibration-only freeze와
  common-threshold negative control
- E2 child responsibility 총합과 no-double-count
- E3 boundary censoring, completeness, zero-denominator와 interval policy
- E3b--E3c의 edge 강도, circuit permutation과 equal-mean/different-bottleneck control
- E4--E7 residual, branch, Jacobian과 scalar/uniform reduction
- E8 horizon/bin sensitivity와 parent/time shuffle controls
- H3 spectral-radius gate
- P2 row-post/column-pre orientation, zero-gate, sign-flip, trace reset
- P4 block-frozen W와 snapshot parity
- G4의 boundary-condition, Green-solver residual과 \(\kappa_\Gamma=0\) control
- G6의 symmetry/SPD 및 좌표변환 parity
- G7의 Dale sign, gain cap, frozen-geometry와 circuit-shuffle control
- cosmology constants 및 held-out outcome의 training path 접근 0건

## 18. 현재 지위

- B1--B10: 수학 후보. 기존 코드와 orientation, memory feedback, delay,
  update order, projection에서 불일치가 있어 REVISE. B10은 autonomous
  zero-noise 조건의 sufficient gate이며 일반 input에서는 ISS로만 읽는다.
- E1--E3c: 이질적 역치·강도를 보존하는 새 apparatus 정의. causal receipt와
  calibration freeze 구현이 없으므로 미완성.
- E4--E8: 명시한 Poisson genealogy 안의 조건부 정리.
- H1--H3: E/I point-process 대안. E4와 자동 동일시 금지.
- P1--P4: 현재 M1/local eligibility와 가장 가까운 구현 route.
- P5--P6: 독립 target이 필요한 model choice.
- C1: 전체 typed research architecture. 뇌의 자연법칙 또는 AGI 증명은 아님.
- G1: 해부학적 표면의 표준 정의. G2--G8: 회로별 이질성을 보존하는
  기능적 계량 model choice. 실제 피질 변형, 기억 또는 Loop 8/9 개선은 미완성.
