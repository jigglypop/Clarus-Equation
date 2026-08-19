# 신경 기하 후보식 원장

Status: FROZEN V2 VALIDITY AMENDMENT AFTER INVALIDATED V1

Freeze disclosure: the earlier single-candidate E17 feasibility calculation for
`S4-H` at $H=5$ was inspected before this v2 universe was closed.  Therefore every
E17 comparison in this run is retrospective discovery, never a locked confirmation.
V1 was later invalidated because `S7-H,H=1` was a tautology, one zero-ridge rank-one
field passed through floating-point noise, and input/tuple provenance was incomplete.
The V2 amendment changes only those structural eligibility and provenance rules;
candidate families, finite grids and score directions were not selected from V1
outcomes. Every E17 comparison remains retrospective discovery, never confirmation.

Pre-outcome audit amendment 1 fixes optimizer, low-dimensional, graph-degeneracy
and endpoint details below.  No multi-candidate tournament output had been opened
when this amendment was made.

## 1. 닫힘 규칙

여기서 "후보식 전부"는 계수, horizon과 ridge를 연속적으로 바꾼 무한한 식을 뜻하지 않는다. 서로 다른 입력 객체와 수학적 타입을 대표하는 유한 후보 계열과 완전히 열거한 초매개변수 tuple을 뜻한다. 새 식은 아래 후보와 좌표변환·스케일·극한에서 동치가 아니며, 새로운 관측 입력 또는 새로운 기하 타입을 요구할 때만 다음 preregistration 세대에 추가한다. 결과를 본 뒤 후보, horizon, chart, regularization, rank, edge cost 또는 optimizer를 늘리지 않는다.

후보는 다섯 타입으로 분리한다.

1. 상태 접공간의 대칭 양의 정부호(SPD) 계량
2. 그래프 node 사이의 대칭 metric 또는 유향 quasi-metric
3. drift를 포함한 유향 path action/Finsler 비용
4. 시행 분포 사이의 확률분포 metric
5. 이미 정한 계량에서 계산하는 geodesic·curvature·flow readout

타입이 다른 식은 하나의 승자표로 합치지 않는다. 각 타입은 그 식이 실제로 예측하는 관측량으로 따로 평가한다.

## 2. 공통 동역학과 좌표 법칙

고정한 상태 chart에서 신경 동역학과 관측을 다음처럼 쓴다.

$$
z_{t+1}=F_c(z_t;W,A)+B_cu_t+\varepsilon_t,
\qquad
\operatorname{Cov}(\varepsilon_t)=Q_t,
\qquad
y_t=h_c(z_t)+\nu_t.
\tag{R1}
$$

$J_t=\partial F_c/\partial z|_{z_t}$, $C_t=\partial h_c/\partial z|_{z_t}$이고, 시간가변 상태전이 곱은

$$
\Phi_{a,a}=I,
\qquad
\Phi_{a,b}=J_{b-1}\cdots J_a\quad (b>a)
\tag{R2}
$$

로 둔다. 식 (R1)의 $\varepsilon_{t+k}$가 들어간 직후 상태 $t+k+1$에서 공통 종점 $t+H$까지 보내는 곱은

$$
\Psi_{t,H,k}=\Phi_{t+k+1,t+H}
=J_{t+H-1}\cdots J_{t+k+1},
\qquad k=0,\ldots,H-1,
\tag{R2a}
$$

이며 $k=H-1$에서는 빈 곱 $I$다. 선형 chart 변화 $z'=Pz$에서 공분산형 객체는 $M'=PMP^\top$, 계량형 객체는 $g'=P^{-\top}gP^{-1}$로 변해야 한다. 따라서 covariance ridge $R_C$는 $R_C'=PR_CP^\top$, reference metric $G_0$는 $G_0'=P^{-\top}G_0P^{-1}$로 변환한다. 두 객체를 같은 기호로 쓰지 않는다. 고정 $R_C=G_0=I$를 쓰는 후보는 chart를 등록된 표준화 chart와 그 직교변환으로 제한한다. 비선형 chart에서는 stochastic calculus convention과 위치별 Jacobian을 별도로 고정해야 한다.

모든 SPD 후보는 다음 gate를 통과해야 한다. 공분산의 역을 쓰는 식은 $M+\lambda_C R_C\succ0$이어야 한다. 직접 계량을 만드는 식은 $G+\lambda_G G_0\succ0$이어야 한다. $\lambda=0$은 원행렬이 full rank일 때만 허용한다. gate 실패는 pseudoinverse로 몰래 바꾸지 않고 `INELIGIBLE_SINGULAR`로 기록한다.

## 3. SPD 상태 계량 후보

### `S0` Euclidean null

$$
g_E=I.
\tag{S0}
$$

이 식은 등록된 표준화 chart에서만 기준모형으로 쓴다. 생물학적 주장이 없는 null이다.

### `S1` 상태 공분산 정밀도

$$
\Sigma_z=\operatorname{Cov}(z_t),
\qquad
g_{\Sigma}=(\Sigma_z+\lambda_C R_C)^{-1}.
\tag{S1}
$$

활동 분포의 Mahalanobis 기하다. gain, sampling과 상태 점유율이 바뀌어도 변하므로 연결 기하와 동일하지 않다.

### `S2` 증분 공분산 정밀도

$$
D_{\Delta}=\operatorname{Cov}(z_{t+1}-z_t),
\qquad
g_{\Delta}=(D_{\Delta}+\lambda_C R_C)^{-1}.
\tag{S2}
$$

한 frame에서 관측한 확산 방향의 경험적 기하다. drift와 measurement noise가 섞인다.

### `S3` 과정 잡음 정밀도

$$
g_Q=(Q+\lambda_C R_C)^{-1}.
\tag{S3}
$$

동역학 평균을 제거한 국소 diffusion metric 후보다. $Q$가 생물학적 process noise인지 calcium observation noise인지 식별해야 한다.

### `S4-H` 유한 horizon 과정 잡음 도달가능성

$$
C_{Q,H}(t)=\sum_{k=0}^{H-1}\Psi_{t,H,k}Q_{t+k}\Psi_{t,H,k}^{\top},
\qquad
g_{Q,H}=(C_{Q,H}+\lambda_C R_C)^{-1}.
\tag{S4}
$$

이는 process-noise reachability covariance의 역이지 controllability Gramian이 아니다. 후보 horizon은 30 Hz 자료에서 $H\in\{1,5,15,30\}$으로 닫는다.

### `S5` 정상상태 Lyapunov 정밀도

$$
C_\infty=JC_\infty J^\top+Q,
\qquad
g_\infty=(C_\infty+\lambda_C R_C)^{-1},
\qquad
\rho(J)<1.
\tag{S5}
$$

정상 선형계에서만 정의한다. 안정성 또는 정상성이 깨지면 후보 자격을 실패 처리한다.

### `S6-H` Cauchy--Green flow pullback

$$
G_{\mathrm{CG},H}(t)=
\Phi_{t,t+H}^{\top}G_{0,t+H}\Phi_{t,t+H}
+\lambda_G G_{0,t}.
\tag{S6}
$$

미래 flow가 현재 접벡터를 얼마나 늘리는지 재는 deformation metric이다. noise reachability와 다른 대상을 잰다. 후보 horizon은 `S4-H`와 같다.

### `S7-H` 관측가능성 계량

$$
G_{O,H}=\sum_{k=0}^{H-1}
\Phi_{t,t+k}^{\top}C_{t+k}^{\top}R_{y,t+k}^{-1}C_{t+k}\Phi_{t,t+k}
+\lambda_G G_{0,t}.
\tag{S7}
$$

서로 가까운 상태를 미래 관측으로 구별할 수 있는 정도를 잰다. $C$와 관측 잡음 $R_y$가 측정돼야 한다. $C=I$, $R_y=I$는 생물학적 관측가능성 주장이 아니라 identity-observation 기준모형이다.

E17의 identity-observation 구현에서 $H=1$이면 $G_{O,1}=I$이고 등록된 integrated target도 같은 초기 차이의 제곱노름이다. 따라서 입력과 정답이 동일한 완전한 항등식이므로 모든 `S7-H,H=1` tuple은 `INELIGIBLE_TAUTOLOGY`이며 순위와 외부 평가에서 제외한다. $H\ge2$도 실제 관측모형의 증거가 아니라 같은 fitted dynamics를 사용한 기술적 기준모형으로만 해석한다.

### `S8` Fisher 정보 계량

$$
[I_F(z)]_{ij}=\mathbb E_{y\sim p(y\mid z)}
\left[\partial_i\log p(y\mid z)\,\partial_j\log p(y\mid z)\right].
\tag{S8}
$$

실행 후보는 $g_F=I_F+\lambda_G G_0$이며 SPD gate를 통과해야 한다. 이 식은 조건부 반응분포의 국소 식별가능성을 재는 통계 계량이다. likelihood와 nuisance를 train 자료에서 고정해야 하며 structural $W$를 직접 뜻하지 않는다.

### `S9` decoder pullback / Gauss--Newton 계량

$$
g_{\mathrm{PB}}(z)=J_h(z)^\top G_y(h(z))J_h(z)+\lambda_G G_0.
\tag{S9}
$$

출력 공간의 계량 $G_y$를 신경 상태로 pullback한다. Gaussian 출력이면 $G_y=R_y^{-1}$, 분류 출력이면 Fisher/Gauss--Newton을 쓴다. decoder와 task label을 outer-train에서만 맞춘다.

### `S10-H` 제어가능성 에너지 계량

$$
W_{C,H}(t)=\sum_{k=0}^{H-1}
\Psi_{t,H,k}B_{t+k}R_{u,t+k}^{-1}B_{t+k}^{\top}\Psi_{t,H,k}^{\top},
\qquad
g_C=(W_{C,H}+\lambda_C R_C)^{-1}.
\tag{S10}
$$

입력 channel $B$와 control cost $R_u$가 직접 고정될 때만 제어가능성 후보다. $Q$를 $BR_u^{-1}B^\top$ 대신 넣지 않는다.

### `S11` LQR/Riccati value metric

$$
P=Q_x+J^\top PJ-J^\top PB(R_u+B^\top PB)^{-1}B^\top PJ,
\qquad
g_{\mathrm{LQR}}=P+\lambda_G G_0.
\tag{S11}
$$

이 식은 명시한 state/control cost의 value matrix다. stabilizability/detectability 조건과 SPD gate를 통과하기 전에는 계량이 아니다. 뇌의 내재적 거리라고 부르려면 $Q_x$, $R_u$, $B$를 독립적으로 측정해야 한다.

### `S12` 유연한 diagonal SPD 대조군

$$
g_{\mathrm{diag}}=\operatorname{diag}(\operatorname{softplus}(\theta)+\epsilon).
\tag{S12}
$$

### `S13` 유연한 low-rank SPD 대조군

$$
g_{\mathrm{LR}}=\operatorname{diag}(\operatorname{softplus}(d)+\epsilon)+UU^\top.
\tag{S13}
$$

E17의 표준화 chart에서는 $\epsilon=\epsilon_g=10^{-6}$으로 고정한다. `S12`는 fit $H$-step residual covariance $S_r=n^{-1}\sum_nr_nr_n^\top$에서

$$
[g_{\mathrm{diag}}]_{ii}
=\max\left((S_{r,ii}+\epsilon_g)^{-1},\epsilon_g\right)
\tag{S12a}
$$

로 정하는 diagonal Gaussian MLE floor다. `S13`은 unconstrained $d\in\mathbb R^r$, $U\in\mathbb R^{r\times k}$에 대해

$$
\mathcal L(d,U;S_r,\eta)
=\frac12\left[\operatorname{tr}(g_{\mathrm{LR}}S_r)
-\log\det g_{\mathrm{LR}}\right]
+\frac{\eta}{2}\lVert U\rVert_F^2
\tag{S13a}
$$

를 최소화한다. 초기값은

$$
d_i^{(0)}=\operatorname{softplus}^{-1}\!\left(
\max((S_{r,ii}+\epsilon_g)^{-1}-\epsilon_g,\epsilon_g)
\right),
\qquad
U_{ij}^{(0)}=10^{-3}\sin((i+1)(j+1))
\tag{S13b}
$$

이며 $\operatorname{softplus}^{-1}(x)=x+\log(-\operatorname{expm1}(-x))$의 stable form을 쓴다. L-BFGS-B variables에는 bound를 두지 않는다. optimizer가 success를 반환하지 않거나 목적함수, eigenvalue 또는 parameter가 finite가 아니면 그 tuple은 `OPTIMIZER_FAILED`이고 다른 초기값으로 재시도하지 않는다. $r=1$에서는 positive rank grid가 없으므로 `S13` 전체를 `INELIGIBLE_DIMENSION`; `S12`가 rank-zero flexible baseline이다.

`S12`와 `S13`은 자유도를 명시해 학습하는 비기전적 상한 대조군이다. 후보별 유효 parameter 수를 보고하고 가장 가까운 자유도의 대조군과 비교한다. 기전 후보가 이를 못 이기면 특정 식의 이점이 아니다.

### `S14` raw operator의 self-adjoint SPD projection

등록된 고정 whitening chart에서 $W$의 symmetric part를

$$
S_W=\frac12(W+W^\top)=U\operatorname{diag}(s_i)U^\top
$$

로 두고

$$
g_{\mathrm{sym}W}
=U\operatorname{diag}\!\left(
\tau\log(1+e^{s_i/\tau})+\lambda_G
\right)U^\top
\tag{S14}
$$

를 계산한다. 이는 raw-$W$를 곧바로 계량으로 만드는 가장 직접적인 대조군이지만 일반 affine chart에 공변하지 않는다. E17에서는 $W^s$가 아니라 fitted effective operator $J$를 넣으므로 structural-connectivity 후보로 해석하지 않는다.

수치 구현은 $\tau\log(1+e^{s_i/\tau})=\tau\operatorname{logaddexp}(0,s_i/\tau)$로만 평가한다. overflow가 난 direct `exp` 구현은 부적격이다.

### `S15` prediction-residual pullback

$$
g_{\mathrm{res}}=(I-J)^\top G_0(I-J)+\lambda_G G_0.
\tag{S15}
$$

이는 한 단계 고정점 residual map $z\mapsto z-F(z)$의 pullback이다. `S6-H`의 flow expansion과 다른 산출이지만, 독립적인 연결 측정 없이 fitted $J$를 쓰면 direct-dynamics 재표현일 뿐이다.

### `S16` energy/potential Hessian

$$
g_V(z)=\nabla^2V(z)+\lambda_G G_0,
\qquad V(z)=-\log p(z)\ \text{or a preregistered physical potential}.
\tag{S16}
$$

Hessian은 일반적으로 PSD가 아니며 비선형 chart에서 그 자체가 tensor도 아니다. affine chart, 독립적으로 고정한 $V$, SPD gate가 모두 필요하다. Gaussian $p$에서는 `S1`과 동치이므로 새 후보가 아니고, E17에는 결과 전에 고정된 비이차 $V$가 없어 `UNTESTABLE`이다.

## 4. 그래프 기하 후보

### `G1` effective resistance

연결된 무향 그래프의 conductance $K_{ij}=K_{ji}\ge0$, $K_{ii}=0$와 $D=\operatorname{diag}(K\mathbf1)$, $L=D-K$에 대해

$$
d_R^2(i,j)=(e_i-e_j)^\top L^+(e_i-e_j).
\tag{G1}
$$

이는 node metric이지만 상태 접공간의 SPD tensor가 아니다. 유향 $W$를 $K$로 대칭화하는 규칙을 결과 전에 고정해야 한다.

E17에서는 support $\{(i,j):K_{ij}>0\}$가 connected인지 breadth-first search로 판정한다. disconnected이거나 positive degree가 없는 node가 있으면 tuple은 `INELIGIBLE_GRAPH_DISCONNECTED`이며 component별 점수를 사후 평균하지 않는다.

### `G2` diffusion distance

stationary distribution $\pi$에 대해 reversible인 Markov kernel $P$의 $L^2(\pi)$-orthonormal 고유쌍 $(\lambda_\ell,\psi_\ell)$에 대해

$$
d_{\mathrm{diff},s}^2(i,j)=
\sum_{\ell\ge1}\lambda_\ell^{2s}
[\psi_\ell(i)-\psi_\ell(j)]^2.
\tag{G2}
$$

detailed balance가 없는 유향 연결에서 reversibilization 선택은 추가 공리다.

E17의 등록된 symmetric conductance에서

$$
d_i=\sum_jK_{ij},\qquad
P=D^{-1}K,\qquad
\pi_i=\frac{d_i}{\sum_jd_j},\qquad
S=D^{-1/2}KD^{-1/2}
\tag{G2a}
$$

로 유일하게 고정한다. $S u_\ell=\lambda_\ell u_\ell$, $u_\ell^\top u_m=\delta_{\ell m}$에서 $\psi_\ell(i)=\sqrt{\sum_jd_j}\,u_\ell(i)/\sqrt{d_i}$로 두면 $\psi_0=1$이고 $L^2(\pi)$-orthonormal이다. eigenvalue 1에 해당하는 stationary mode 하나를 제외한 $r-1$개를 모두 쓴다. zero degree, disconnected support, stationary multiplicity가 1이 아님, detailed-balance residual $\max_{ij}|\pi_iP_{ij}-\pi_jP_{ji}|>10^{-10}$ 중 하나면 `INELIGIBLE_GRAPH_KERNEL`이다. negative nonstationary eigenvalue는 식 (G2)의 even power $\lambda_\ell^{2s}$ 그대로 보존한다.

### `G3a/G3b` 유향 최단경로 quasi-metric

행렬 convention을 $W_{ij}$가 source $j$에서 target $i$로 가는 작용이라고 고정한다. self-edge를 제거하고

$$
p_{j\to i}=\begin{cases}
\dfrac{|W_{ij}|}{\sum_{\ell\ne j}|W_{\ell j}|},
&\sum_{\ell\ne j}|W_{\ell j}|>0,\\
0,&\text{otherwise},
\end{cases}
\qquad
c^{\log}_{j\to i}=-\log\max(p_{j\to i},\epsilon_p),
\tag{G3a}
$$

또는

$$
c^{\mathrm{rec}}_{j\to i}
=\frac{s_W}{|W_{ij}|+\epsilon_ws_W},
\qquad
s_W=\operatorname{median}_{a\ne b:|W_{ab}|>0}|W_{ab}|,
\tag{G3b}
$$

$|W_{ij}|>0$인 edge에만 위 cost를 부여하고

$$
d_q(i,j)=\min_{\pi:i\to j}\sum_{e\in\pi}c_e
\tag{G3c}
$$

로 둔다. 이번 격자는 $\epsilon_p=10^{-6}$, $\epsilon_w=10^{-3}$ 두 식뿐이다. positive off-diagonal weight가 하나도 없어 $s_W$가 정의되지 않으면 `G3b`는 자격 실패다. zero-outdegree 또는 disconnected pair는 거리 $+\infty$로 기록한다.

$d_q(i,j)\ne d_q(j,i)$일 수 있으므로 리만 거리로 부르지 않는다. parallel edge와 tie rule을 보존한다. edge cost를 결과 뒤 다시 정의하는 것은 새 후보 세대다.

## 5. 유향 path 후보

### `D1` Gaussian transition action

$$
S_Q[x]=\frac12\sum_t
[x_{t+1}-F(x_t)]^\top Q_t^{-1}[x_{t+1}-F(x_t)].
\tag{D1}
$$

서로 다른 $Q$ 후보를 점수화할 때는 위 quadratic term만 비교하지 않고 Gaussian negative log predictive density

$$
\operatorname{NLPD}_Q[x]=\frac12\sum_t\left[
r_t^\top Q_t^{-1}r_t+\log\det Q_t+d\log(2\pi)
\right],
\qquad r_t=x_{t+1}-F(x_t),
\tag{D1a}
$$

를 쓴다. 그렇지 않으면 $Q$를 크게 부풀린 후보가 부당하게 유리하다.

### `D2` Freidlin--Wentzell action

연속시간 SDE $dx=f(x)dt+\sqrt{\epsilon}\,\sigma(x)dB_t$와 $a=\sigma\sigma^\top$에 대해

$$
S_{\mathrm{FW}}[x]=\frac12\int
[\dot x-f(x)]^\top a(x)^{-1}[\dot x-f(x)]\,dt.
\tag{D2}
$$

`D1`과 `D2`는 path probability의 방향성 있는 비용이며 대칭 거리나 리만 geodesic이 아니다. `D1`은 $Q_t\succ0$이어야 한다. `D2`는 small-noise rate functional이고 $a(x)\succ0$인 calibrated continuous-time SDE에서만 쓴다. E17은 이산 `D1` proxy만 허용하며 `D2`를 검정하지 않는다.

### `D3` Randers/Finsler 후보

$$
F(z,v)=\sqrt{v^\top g(z)v}+\beta_z(v),
\qquad
\lVert\beta\rVert_g<1.
\tag{D3}
$$

drift 또는 목표 편향을 one-form $\beta$로 넣을 수 있다. 속도 제한과 Zermelo navigation 사상이 없으면 $\beta$는 비식별이므로 자료만 맞추는 자유항으로 쓰지 않는다.

## 6. 분포 기하 후보

### `P1` Wasserstein-2

$$
W_2^2(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}
\int\lVert x-y\rVert_{G_0}^2\,d\pi(x,y).
\tag{P1}
$$

### `P2` Benamou--Brenier dynamic representation

$$
W_2^2(\mu_0,\mu_1)=
\inf_{\rho,v}\int_0^1\!\int \rho_t(x)\lVert v_t(x)\rVert_{G_0}^2dxdt,
\quad
\partial_t\rho+\nabla\cdot(\rho v)=0.
\tag{P2}
$$

적절한 정칙성 아래 `P1`과 `P2`는 같은 $W_2$의 static/dynamic 표현이므로 서로 다른 승자 후보로 세지 않는다. 두 식은 시행 또는 population 분포 사이의 거리다. 개별 신경상태의 local metric이나 structural $W$의 함수는 아니다. ground metric $G_0$ 선택이 결과를 지배할 수 있다.

## 7. 계량 producer가 아닌 readout과 update 가설

계량 $g$를 먼저 고른 뒤에만 다음을 계산한다.

$$
L_g[\gamma]=\int\sqrt{\dot\gamma^\top g(\gamma)\dot\gamma}\,ds,
\qquad
\ddot x^k+\Gamma^k_{ij}\dot x^i\dot x^j=0,
\qquad
R_{ij}[g],\;R[g].
\tag{R3}
$$

geodesic, Christoffel symbol과 Ricci curvature는 $g$의 정의가 아니라 산출이다. curvature는 smooth state-dependent field $g(z)$가 있어야 하며, E17의 sessionwise constant matrix는 고정 chart에서 flat하므로 비자명한 curvature 검정을 만들지 못한다. 그래프 Ollivier--Ricci curvature도 Markov kernel과 ground metric이 먼저 필요하다.

수면 재정규화의 가장 강한 후보 update는

$$
\partial_s g=-2\alpha\operatorname{Ric}(g)+\mathcal F_{\rm replay}
-\gamma(g-g_{\rm ref})
\tag{R4}
$$

로 쓸 수 있으나 현재는 `[미완성: 모델 선택]`이다. $g$, replay forcing, gauge, boundary와 시간척도를 먼저 측정하지 않으면 Ricci flow라는 이름만으로 검정할 수 없다.

## 8. E17 자격 판정

| 후보 | E17 상태 | 허용되는 해석 |
|---|---|---|
| `S0`--`S4-H`, `S6-H`, `S14`, `S15` | 계산 가능 | 고정 chart, 선형-Gaussian held-out feasibility |
| `S5` | 조건부 | 각 calibration $J$에서 $\rho(J)<1$일 때만 |
| `S7-H` | identity-output 기준만 가능 | 실제 관측가능성 주장 금지 |
| `S8`, `S9` | Figure 2 condition-label decoder로 부분 가능 | condition information geometry 탐색; task/behavior 주장 금지 |
| `S10-H`, `S11` | `UNTESTABLE` | $B,R_u,Q_x$ 없음 |
| `S12`, `S13` | discovery 대조군으로 가능 | 생물 기전 아님 |
| `S16` | `UNTESTABLE` | 독립적으로 고정된 nonquadratic $V$ 없음 |
| `G1`, `G2` | 등록된 두 대칭화로 계산 가능 | held-out pair correlation 기술량; structural graph 주장 금지 |
| `G3a`, `G3b` | 등록된 비용으로 계산 가능 | held-out directed lag association 기술량; quasi-metric만 |
| `D1` | 계산 가능 | 이산 유향 action/NLPD feasibility |
| `D2` | `UNTESTABLE` | calibrated continuous-time SDE와 small-noise scale 없음 |
| `D3` | `UNTESTABLE` | 식별 가능한 $\beta$와 속도 제약 없음 |
| `P1/P2` | 하나의 $W_2$로 기술적으로 가능 | distribution shift, 구조적 metric 아님 |
| `R3`, `R4` | producer 검증 전 `UNTESTABLE` | curvature/flow 승격 금지 |

E17의 Figure 3 synapse summary, Figure 4 longitudinal dendrite summary와 Figure 2 DCZ/saline trial은 같은 unit chain이 아니다. 모든 후보를 계산해도 `NRM-H1A`는 계속 `UNTESTABLE`이다. Figure 2의 released array order에는 검증된 timestamp provenance가 없으므로 아래 test는 held-out trial이지 입증된 future trial이 아니다.

## 9. E17 유한 실행 격자

이번 run의 candidate는 식 ID와 아래 tuple을 합친 것이다. 이 절에 없는 연속 tuning은 금지한다.

1. 자료는 Figure 2의 11 sessions, 3 animals, saline/DCZ branch trial만 쓴다. 조건별 trial 수가 $n$일 때 released array order의 $n_{\rm fit}=\lfloor0.5n\rfloor$, $n_{\rm inner}=\lfloor0.25n\rfloor$, 나머지를 test로 둔다.
2. chart는 각 session의 두 조건 fit trial만 합쳐 ROI별 mean과 standard deviation으로 고정한다. 분산 $10^{-8}$ 이하 ROI는 fit 단계에서 제거한다. $R_C=G_0=I$이고 일반 affine 불변성을 주장하지 않는다.
3. 평균 동역학은 조건별 affine VAR(1) $z_{t+1}=Jz_t+b+\varepsilon_t$ 하나다. minimum fit transitions는 parameter 하나당 10개다. 평균모형 family를 후보별로 바꾸지 않는다.
4. horizon은 $H\in\{1,5,15,30\}$이며 서로 다른 target table이다. covariance와 metric ridge는 각각 $\lambda_C,\lambda_G\in\{0,10^{-6},10^{-3},10^{-1}\}$다. $\lambda=0$은 SPD gate를 통과할 때만 남긴다.
5. uncertainty 후보의 inverse metric $M$은 covariance shape로 쓴다. fit residual $r_n$에서 positive scalar의 Gaussian MLE $a=\max(10^{-12},(nr)^{-1}\sum_nr_n^\top M^{-1}r_n)$를 맞춘다. 같은 candidate/horizon의 tuple은 saline/DCZ inner NLPD를 먼저 session 안에서 평균하고, session을 animal 안에서 평균한 뒤 outer-train animal 평균으로 고른다. held-out animal의 inner block은 어느 parameter에도 쓰지 않는다. tie는 더 작은 자유도, 더 작은 ridge, lexicographic ID 순으로 푼다.
6. `S12`는 식 (S12a), `S13`은 식 (S13a)--(S13b)를 쓴다. `S13` rank는 $r\ge2$에서 $k\in\{1,\ldots,\min(3,r-1)\}$, penalty는 $\eta\in\{0,10^{-4},10^{-2},1\}$다. SciPy L-BFGS-B는 unbounded variables, `jac='2-point'`, `finite_diff_rel_step=10^{-6}`, `maxiter=500`, `ftol=10^{-10}`으로 고정한다. unrestricted full covariance와 isotropic covariance를 별도 baseline으로 두고 둘 다 $10^{-6}I$ floor를 쓴다.
7. `S14`의 $\tau\in\{0.1,1,10\}$다. `S8/S9` condition-label logistic decoder의 L2 penalty는 $\eta_D\in\{0,10^{-4},10^{-2},1\}$, optimizer와 seed는 6번과 같다.
8. `G1/G2`의 fixed symmetrization은 $K^{\rm abs}_{ij}=(|J_{ij}|+|J_{ji}|)/2$와 $K^{+}_{ij}=(\max(J_{ij},0)+\max(J_{ji},0))/2$ 두 개이고 diagonal은 0이다. `G2` diffusion time은 $s\in\{1,5,15,30\}$다. `G3a/G3b`의 cost와 epsilon은 식에 적은 두 tuple뿐이다.
9. `P1/P2`는 하나의 empirical $W_2$ family다. ground metric은 $I$와 $(\Sigma_{\rm fit}+10^{-6}I)^{-1}$ 두 개다. 각 condition에서 released order에 균등 간격으로 고른 같은 수의 최대 256 test states를 쓰고 exact linear assignment를 계산한다. trial label 전체를 trial block 단위로 바꾸는 permutation 256회, PCG64 seed 1729를 고정한다.
10. energy score는 test residual 중 released order의 균등 간격 최대 256개와 candidate Gaussian에서 뽑은 256개 PCG64(seed 1729) draw로 계산한다. 두 번째 expectation에는 같은 generator에서 이어서 뽑은 독립 256개 draw를 쓴다. raw tuple score, inner selection, eligibility failure와 test score를 모두 machine-readable하게 남긴다.

outer loop는 animal leave-one-out 세 fold다. held-out animal에서도 앞 50% trial로 session-local $J,Q,g$와 scalar calibration을 맞추지만 그 animal의 inner/test 결과를 쓰지 않는다. 마지막 25% test는 outer-train inner score로 선택된 tuple을 한 번 점수화하고 animal 안에서 session 평균을 낸다. 모든 raw tuple의 fit/inner score는 보존하지만 held-out test raw sweep는 하지 않는다. 이미 열어 본 E17 자료이므로 이 절차가 누수를 줄여도 confirmatory lock으로 승격되지는 않는다.

## 10. 타입별 토너먼트

### A. 불확도 SPD 토너먼트

`S0`--`S5`, `S12/S13`이 같은 fit-only 평균동역학 아래 held-out $H$-step residual의 Gaussian NLPD와 energy score를 예측한다. direct full state-space, diagonal과 isotropic covariance를 함께 비교한다. 이 표의 승자는 uncertainty shape의 승자일 뿐, 독립적인 spatial mechanism의 승자가 아니다.

### B. deformation/관측/task 토너먼트

각 condition/session/block에서 trial을 released order로 `(0,1), (2,3), ...` 짝짓는다. 첫 trial의 각 admissible time $t\le T-H-1$에 대해 두 번째 trial에서 initial Euclidean distance를 최소화하는 $u\le T-H-1$를 고른다. tie는 더 작은 $u$다. endpoint는 pair 선택에 쓰지 않는다. 생성된 pair가 512개를 넘으면 전체 released pair order에서 균등 간격 512개를 고른다.

초기 차이 $\delta_0=z_t^{(a)}-z_u^{(b)}$와

$$
y_H=\lVert z_{t+H}^{(a)}-z_{u+H}^{(b)}\rVert_2^2,
\qquad
y_{O,H}=\sum_{k=0}^{H-1}
\lVert z_{t+k}^{(a)}-z_{u+k}^{(b)}\rVert_2^2
\tag{E1}
$$

를 만든다. `S6/S14/S15`는 $q=\delta_0^\top g\delta_0$로 $y_H$를, `S7`은 같은 $q$로 $y_{O,H}$를 예측한다. fit pair에서 nonnegative scale

$$
a=\max\left(0,\frac{\sum_nq_ny_n}{\sum_nq_n^2}\right)
\tag{E2}
$$

를 맞추고, denominator가 0이면 `INELIGIBLE_ZERO_FEATURE`다. inner/test primary score는

$$
\operatorname{nRMSE}=
\frac{\sqrt{N^{-1}\sum_n(aq_n-y_n)^2}}
{\sqrt{N^{-1}\sum_ny_n^2}+10^{-12}}
\tag{E3}
$$

이며 Spearman correlation은 secondary다. condition을 session 안에서, session을 animal 안에서 평균하는 선택 규칙은 uncertainty tournament와 같다. `S6`의 $\lambda_G=0$ prediction은 direct linear-flow prediction과 algebraically 동일하므로 이를 독립 기전 우위로 세지 않는다.

`S8/S9`는 fit states의 balanced saline/DCZ logistic decoder를 공유한다. label $c\in\{0,1\}$, $p(z)=\operatorname{sigmoid}(w^\top z+b)$에 대해 class당 total weight $1/2$인 weighted log loss와 $\eta_D\lVert w\rVert_2^2/2$를 최소화한다. $(w,b)=(0,0)$에서 시작하고 `S13`과 같은 L-BFGS-B 규칙을 쓴다. $\eta_D$는 outer-train inner balanced log loss로 고른다. 실행 field는

$$
g_F(z)=p(z)(1-p(z))ww^\top+\lambda_GI,
\qquad
g_{\rm PB}=ww^\top+\lambda_GI.
\tag{E4}
$$

$r>1$에서 $\lambda_G=0$인 두 field는 rank-one 이하이므로 부동소수 eigenvalue와 무관하게 `INELIGIBLE_SINGULAR`다. held-out session의 decoder와 field는 그 session의 fit block에서 만들고, field의 minimum eigenvalue와 log determinant도 fit state에서만 계산한다. outer-train inner balanced log loss는 decoder penalty 하나를 선택하며, held-out test에서는 선택된 decoder의 balanced log loss 하나만 평가한다. $\lambda_G$ field들은 독립 predictive tournament 후보가 아닌 fit-only algebraic gate다. 두 식은 decoder를 공유하므로 같은 label log loss를 독립 복제 두 건으로 세지 않는다. 이 release에는 task/behavior label이 없으므로 label metric을 task geometry나 trajectory likelihood 승자로 재해석하지 않는다.

### C. graph 토너먼트

test block의 모든 finite state row를 합쳐 `G1/G2`는 $i<j$에 대한 $|\operatorname{corr}(z_i,z_j)|$, `G3a/G3b`는 $i\ne j$에 대한 $|\operatorname{corr}(z_{t,j},z_{t+1,i})|$를 만든다. primary 기술량은 graph distance와 각 association의 Spearman $\rho$이며 가까운 node가 더 연관된다는 사전 방향은 $\rho<0$이다. finite distance와 nonconstant association을 가진 유효 pair가 3개 미만이면 `INSUFFICIENT_PAIRS`다. node-pair rows를 동물 수로 세지 않는다.

### D. directional 토너먼트

`D1`은 condition별 fit VAR(1)과 $Q+\lambda_CI$로 forward held-out one-step NLPD를 계산한다. reverse control은 각 test trial의 time axis를 뒤집고, shuffle control은 PCG64 seed 1729에서 trial별 독립 permutation을 뽑아 time axis에 적용한다. 같은 frozen mean/covariance에 세 path를 점수화하며 primary contrast는 `reverse minus forward`와 `shuffle minus forward` NLPD다. $\lambda_C$는 outer-train inner forward NLPD로만 고른다. `D2/D3`는 E17에서 eligibility failure만 기록한다. 대칭 SPD 길이와 별도 점수표를 쓴다.

### E. distribution 토너먼트

`P1/P2`는 saline/DCZ held-out state distribution의 empirical $W_2$와 trial-block label-permutation 위치를 기록한다. behavior association은 해당 field가 없어서 `UNTESTABLE`이다.

## 11. discovery와 봉인 검증

1. 후보 universe, chart, horizon, ridge와 자유도 예산을 outcome 전에 고정한다.
2. discovery animal 안에서만 nested group CV로 후보와 초매개변수를 고른다.
3. session, trial, window와 cell은 animal fold를 넘지 않는다.
4. locked animal에는 선택한 후보 하나를 한 번 적용한다. lock에서 여러 후보를 동시에 열면 전체 family에 max-$T$ 또는 Holm 보정을 적용한다.
5. E17은 동물 3마리뿐이므로 식 구현과 교란 민감도 탐색까지만 한다. winner와 population success를 선언하지 않는다.

## 12. 성공 조건

한 후보가 다음 조건을 모두 통과해야 "해당 뇌 영역·과제에서 성공"이라고 부른다.

1. SPD, 좌표 법칙, 차원과 타입 gate를 통과한다.
2. 필요한 $W,B,Q,C,R$가 같은 unit과 시간축에서 실제 측정된다.
3. 독립 동물 outer-test에서 primary proper score의 사전등록 효과가 0을 넘는다.
4. direct dynamics와 parameter-matched flexible metric을 이긴다.
5. gain-only, noise-only, chart, shuffle, time reversal과 target permutation kill test를 통과한다.
6. 별도 코호트에서 같은 방향으로 복제된다.
7. `Delta W^s -> Delta g -> Delta x`를 주장할 때는 randomized/validly instrumented $W^s$ 개입과 같은 시냅스 pre/post 측정이 있다.

test를 다시 열어 다른 후보를 고르는 것은 성공이 아니다. 실패한 lock 뒤에는 새 자료와 새 preregistration으로 다음 세대를 시작한다.

## 13. 대표 일차 출처

- Mahalanobis distance: https://cir.nii.ac.jp/crid/1370017279880961561
- Linear dynamics, controllability and observability: https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD724842.xhtml
- Cauchy--Green strain and finite-time deformation: https://doi.org/10.1146/annurev-fluid-010313-141322
- Fisher information in correlated neural populations: https://pmc.ncbi.nlm.nih.gov/articles/PMC4451760/
- Neural response information geometry: https://doi.org/10.1364/JOSAA.21.000307
- Structural brain-network controllability: https://doi.org/10.1038/ncomms9414
- Pullback geometry in neural networks: https://papers.neurips.cc/paper/6873-principles-of-riemannian-geometry-in-neural-networks.pdf
- High-dimensional neural population geometry: https://doi.org/10.1038/s41586-019-1346-5
- Graph resistance distance: https://doi.org/10.1007/BF01164627
- Diffusion maps: https://doi.org/10.1016/j.acha.2006.04.006
- Freidlin--Wentzell action: https://doi.org/10.1007/978-3-642-25847-3
- Brain-state transition action model: https://pmc.ncbi.nlm.nih.gov/articles/PMC10503743/
- Randers geometry: https://doi.org/10.1103/PhysRev.59.195
- Dynamic optimal transport: https://doi.org/10.1007/s002110050002

어느 출처도 전체 경험 사슬 $\Delta W\to\Delta g\to\Delta x$를 검증하지 않는다. 위 문헌은 각 수학 객체의 출처 또는 제한된 신경 응용 근거일 뿐이다. forced sleep update 식 (R4)의 결합형 전체를 지지하는 일차 출처는 없으며 이 식은 현상론적 미완성 가설로 남긴다.
