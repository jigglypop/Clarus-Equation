# Research contract

Status: COMPLETE

Revision: 1 — 수학 레인의 P0 반례에 따라 수동적 flow-pullback과 능동적 reachability-energy를 분리했다.

PREDECESSOR: `_workspace/ce/dandi-neuropal-circuit-connection-20260820`

## 질문과 범위

고정된 뉴런 위치와 발달 해부학을 유지하면서, 뉴런별 역치·회로 강도·전달 지연·입력 위치가 신경 상태공간의 변형과 도달 난이도를 얼마나 바꾸는지 처음부터 다시 정의한다. 다음 세 대상을 섞지 않는다.

1. **해부학적 피질 표면**의 실제 굽힘과 주름,
2. **자유 동역학의 상태공간 pullback**,
3. **외부 제어로 종단 상태에 도달하는 최소에너지**.

이번 run은 식과 반례를 닫는 수학 단계다. 새 response endpoint나 봉인된 confirmation 자료는 열지 않는다.

## PREDECESSOR_EVIDENCE

| 선행 결과 | 증거와 SHA-256 | 상태 | 보존하는 좁은 주장 | 재시도 금지 |
|---|---|---|---|---|
| A3 hard-threshold static operator | `artifacts/a3-development-result.json`, `87b97ebea002d772c1ff4a50cc4523810f541214e0b457b25bfab23ccd86fa36` | STOP | 뉴런별 calibration threshold와 edge statistic은 계산 가능 | threshold·lag·decoder retune으로 $L_c,\Omega$ 재시도 금지 |
| A4 instantaneous edge contraction | `artifacts/a4-development-result.json`, `79542de8e7bf08c62b50bb2b0cc932e62e2ed303c6f06da6e6d329907591661c` | STOP | positive conductance에서 graph Laplacian은 PSD | correlation $\alpha$를 $e^h$ spatial contraction으로 다시 매핑 금지 |
| A5 $L_0+\Delta L_t$ prediction | `artifacts/a5-development-result.json`, `2e00e5cf12512958f1f1219a9f85daf51ed951b304f70888943926dd4c071ba5` | STOP | fixed anatomy와 functional change는 분리해야 함 | RMS·clip·ridge·horizon retune 금지 |
| 종합 | `40-final-report.md`, `f3b4bf364402c8edd4900494eec63aa11623a48bdba9e05fe8298b8f820ad5ba` | COMPLETE_NEGATIVE_RESULT | confirmation은 unopened; 새 mechanism에는 사용 가능 | 실패를 성공으로 재명명 금지 |

## 후보 집합과 선택

1. **A6-P 선택 — passive flow-pullback.** 회로가 만든 동역학의 flow map을 미분하고, 그 Jacobian으로 종단 기준계량을 초기 상태공간에 pullback한다. full-rank 영역에서만 진짜 Riemann 계량이다.
2. **A6-C 선택 — finite-horizon reachability-energy.** 같은 선형화에서 입력 map과 비용을 함께 지정하여 종단 변위의 최소 제어에너지를 계산한다. 이는 고정 horizon의 endpoint value function이며 일반적인 local Riemann/sub-Riemann 계량이 아니다.
3. **관측 Fisher/observability metric — 보류.** 미래 출력의 구별성은 유효하지만 A6-P와 A6-C 어느 쪽과도 동일하지 않다.
4. **directed Finsler travel-time metric — open alternative.** 비대칭 전달시간을 표현할 수 있지만 추가 공리가 필요하다.
5. **elastic physical-fold PDE — BLOCKED_INPUT.** longitudinal cortical embedding, thickness, material law와 boundary condition이 없으므로 실제 주름 변형식으로 승격하지 않는다.
6. **A3--A5 threshold/scale retune — RETIRED.** 새 mechanism이 아니므로 후보에서 제외한다.

선택 근거는 A6-P/C가 이전 STOP의 핵심인 임의의 $\alpha\mapsto g$ 사상을 제거하고, 회로가 먼저 동역학의 미분을 바꾼 뒤 기하량이 계산되게 하기 때문이다.

## 타입·정규화·가정

- $q$: 뉴런 수, $T$: finite horizon tick 수, $D$: 최대 정수 지연. 세 기호를 겹쳐 쓰지 않는다.
- $a_i^n\in[-1,1]$: 뉴런 $i$의 무차원 활성도.
- $W_{ij}$: sender $j$에서 receiver $i$로 가는 signed 무차원 회로 강도.
- $p_{ij}^n\in[0,1]$: 무차원 edge efficacy/occupancy.
- $d_{ij}\in\{0,\ldots,D\}$: tick 단위 지연.
- $\theta_i,b_i$: 활성도와 같은 기준으로 무차원화한 뉴런별 역치와 bias.
- $0<\lambda_i\le1$: 뉴런별 update fraction.
- $u_n\in\mathbb R^m$: 무차원 intervention/control input, $B_n\in\mathbb R^{q\times m}$: 입력 위치와 gain.
- $\xi_n=(a_n,a_{n-1},\ldots,a_{n-D})\in\mathbb R^{q(D+1)}$: delay-augmented state.
- $R_n\in\mathbb R^{m\times m}$, $R_n\succ0$: 무차원 control-cost matrix.

선형화 정리에서는 기준 trajectory, 초기 history, $p_{ij}^n$, $B_n$을 고정한 외생 시계열로 둔다. 각 $\phi_i$는 기준 drive에서 $C^1$이어야 한다. 회로 매개변수에 대한 식 (A6.7b), (A6.10a)의 response derivative를 사용할 때는 $\phi_i\in C^2$이고 reference trajectory가 $\varepsilon$에 대해 미분가능해야 한다. $p$나 $B$가 활성도·가소성 상태에 따라 변하면 그 변수를 augmented state에 넣고 모든 도함수 항을 다시 계산해야 한다.

## A6.1 — 역치·강도·지연을 포함한 기본 동역학

receiver drive와 bounded update를

$$
h_i^n=\sum_{j=1}^{q}W_{ij}p_{ij}^n a_j^{\,n-d_{ij}}
+(B_nu_n)_i+b_i-\theta_i,
$$

$$
a_i^{n+1}=(1-\lambda_i)a_i^n+\lambda_i\phi_i(h_i^n),
\qquad |\phi_i(h)|\le1
\tag{A6.1}
$$

로 둔다. $W$의 부호는 동역학에 보존하며 nonnegative offspring matrix나 공간 conductance로 바꾸지 않는다. 역치 $\theta_i$는 drive와 local gain $\phi_i'(h_i^n)$을, 강도는 $W_{ij}p_{ij}^n$을, 지연은 참조하는 history slot을 바꾼다.

단, 식 (A6.1)에서는 $b_i$와 $\theta_i$가 $c_i=b_i-\theta_i$라는 net offset으로만 나타난다. 별도의 calibration 또는 intervention receipt가 없으면 두 값을 독립 식별할 수 없다. 따라서 “뉴런별 역치를 반영했다”는 말은 독립적으로 측정한 $\theta_i$가 있을 때만 가능하며, 그렇지 않으면 neuron-specific net drive만 반영한 것이다.

## A6.2 — 지연이 포함된 접선 동역학

기준 trajectory 주위에서

$$
\delta\xi_{n+1}=A_n\delta\xi_n+\mathcal B_n\delta u_n
\tag{A6.2}
$$

로 선형화한다. current-state top block은

$$
(A_n)_{i,(j,d)}
=(1-\lambda_i)\delta_{ij}\mathbf1[d=0]
+\lambda_i\phi_i'(h_i^n)W_{ij}p_{ij}^n\mathbf1[d=d_{ij}],
\tag{A6.3a}
$$

입력 top block은

$$
(\mathcal B_n)_{i\mu}
=\lambda_i\phi_i'(h_i^n)(B_n)_{i\mu}.
\tag{A6.3b}
$$

history를 한 칸씩 내리는 shift block은

$$
(A_n)_{(i,d+1),(j,d)}=\delta_{ij},
\qquad d=0,\ldots,D-1,
\tag{A6.3c}
$$

이고 나머지 history entry와 history input row는 0이다.

$p=p(a,e)$라면 식 (A6.3a)에는 일반적으로

$$
\lambda_i\phi_i'(h_i^n)
\sum_j W_{ij}a_j^{\,n-d_{ij}}
\frac{\partial p_{ij}^n}{\partial a_\ell^{\,n-r}}
$$

같은 항이 추가된다. 이를 생략한 frozen-$p$ 식을 activity-dependent plasticity의 Jacobian이라고 부르지 않는다.

## A6-P — 수동적 flow-pullback 계량

초기 history와 입력 경로를 고정한다. $\iota:\mathbb R^q\to\mathbb R^{q(D+1)}$가 초기 current activation perturbation을 current slot에 넣고, $P:\mathbb R^{q(D+1)}\to\mathbb R^q$가 종단 current activation을 읽는다고 하자. 상태전이행렬을

$$
\Phi(T,k)=A_{T-1}A_{T-2}\cdots A_k,
\qquad \Phi(T,T)=I
$$

로 두면 flow map의 접선은

$$
J_T=P\Phi(T,0)\iota
=\frac{\partial a_T}{\partial a_0}.
\tag{A6.4}
$$

endpoint에서 outcome과 독립적으로 고정한 기준계량 $G_T\succ0$가 있을 때 초기 상태공간의 pullback은

$$
g_{\rm pass}(T)=J_T^\top G_TJ_T.
\tag{A6.5}
$$

$\operatorname{rank}J_T=q$인 영역에서만 식 (A6.5)는 양의 정부호 Riemann 계량이다. rank가 떨어지면 positive-semidefinite degenerate pullback이며 ridge나 pseudoinverse로 그 사실을 숨기지 않는다.

초기 기준계량 $G_0\succ0$와 방향 $v\ne0$에 대한 수동적 길이 변화는

$$
s_{\rm pass}(v;T)
=\sqrt{\frac{v^\top g_{\rm pass}(T)v}{v^\top G_0v}},
\qquad
\ell_{\rm pass}(v;T)=\log s_{\rm pass}(v;T).
\tag{A6.6}
$$

$s>1$이면 해당 infinitesimal direction이 endpoint 표현에서 늘어나고, $0<s<1$이면 줄어든다. $s=0$은 무한 수축이 아니라 tangent information의 rank loss다. 회로 전후 두 계량이 모두 SPD일 때 principal stretch ratio는 generalized eigenproblem

$$
g_{\rm pass,post}v_m
=\Lambda_m g_{\rm pass,pre}v_m,
\qquad r_{{\rm pass},m}=\sqrt{\Lambda_m}
\tag{A6.7}
$$

으로 정의한다.

### 회로 하나가 생길 때의 metric response

새 directed circuit $\Gamma$의 signed edge mask를 $C_\Gamma$라 두고

$$
W(\varepsilon)=W_0+\varepsilon C_\Gamma,
\qquad 0\le\varepsilon\le1
\tag{A6.7a}
$$

로 회로를 0에서 전체 강도까지 연속적으로 켠다. 점은 고정되고 edge만 추가된다는 뜻이다. 이 response branch에서는 $p,B,b,\theta,\lambda,d,G_T$와 initial history를 $\varepsilon$에 대해 고정하고 $W'=C_\Gamma$만 허용한다. $\dot a=\partial_\varepsilon a$는

$$
\dot h_i^n
=\sum_j\left[
(C_\Gamma)_{ij}p_{ij}^na_j^{\,n-d_{ij}}
+W_{ij}(\varepsilon)p_{ij}^n\dot a_j^{\,n-d_{ij}}
\right],
$$

$$
\dot a_i^{n+1}
=(1-\lambda_i)\dot a_i^n
+\lambda_i\phi_i'(h_i^n)\dot h_i^n,
\qquad \dot a_i^{n}=0\;(n\le0)
\tag{A6.7a-1}
$$

로 함께 전진시킨다. 이때 current top block의 total derivative는

$$
(\dot A_n)_{i,(j,d)}
=\lambda_i\left[
\phi_i''(h_i^n)\dot h_i^nW_{ij}(\varepsilon)p_{ij}^n
+\phi_i'(h_i^n)(C_\Gamma)_{ij}p_{ij}^n
\right]\mathbf1[d=d_{ij}],
\tag{A6.7a-2}
$$

이고 shift block derivative는 0이다. 입력 tangent map의 total derivative는

$$
(\dot{\mathcal B}_n)_{i\mu}
=\lambda_i\phi_i''(h_i^n)\dot h_i^n(B_n)_{i\mu},
\tag{A6.7a-3}
$$

이며 history row는 0이다. 따라서 $\dot A_n=\partial_\varepsilon A_n$과 $\dot{\mathcal B}_n$은 기준 trajectory 변화까지 포함한 재현 가능한 total derivative다. 이를 사용하면

$$
\dot\Phi(T,0)
=\sum_{r=0}^{T-1}
\Phi(T,r+1)\dot A_r\Phi(r,0),
$$

$$
\dot J_T=P\dot\Phi(T,0)\iota,
\qquad
\dot g_\Gamma
=\dot J_T^\top G_TJ_T+J_T^\top G_T\dot J_T
\tag{A6.7b}
$$

가 그 회로의 1차 metric-response tensor다. $G_T$ 자체를 바꾸는 모델이면 $J_T^\top\dot G_TJ_T$가 더해지지만, 이번 계약에서는 $G_T$를 고정한다. 식 (A6.7a-2)에서 $\phi_i''\dot h_i$ 항을 빼고 직접 $C_\Gamma$ 항만 남기면 그것은 total response가 아니라 frozen-trajectory partial response다.

회로를 완전히 켠 finite response는 식 (A6.7)의 $\Lambda_m$들로 읽는다. 두 계량이 SPD일 때 총 metric-volume 변화는

$$
\Delta_\Gamma\log V_g
=\frac12\log\frac{\det g_{\rm pass}(1)}{\det g_{\rm pass}(0)}
=\frac12\sum_{m=1}^{q}\log\Lambda_m.
\tag{A6.7c}
$$

rank가 달라지면 이 determinant ratio를 계산하지 않고 `PASSIVE_RANK_CHANGED`로 보고한다. 따라서 “얼마나 끌렸는가”는 방향별 $\sqrt{\Lambda_m}$, 국소 tensor $\dot g_\Gamma$, 전체 volume 변화 $\Delta_\Gamma\log V_g$의 세 수준으로 분해된다.

## A6-C — 능동적 finite-horizon reachability-energy

zero initial perturbation에서 종단 perturbation은

$$
\delta\xi_T=H_TU,
$$

$$
H_T=
\begin{bmatrix}
\Phi(T,1)\mathcal B_0&
\Phi(T,2)\mathcal B_1&\cdots&
\mathcal B_{T-1}
\end{bmatrix},
\quad
\bar R_T=\operatorname{diag}(R_0,\ldots,R_{T-1}).
$$

finite-horizon controllability Gramian은

$$
\mathcal W_c(T)
=H_T\bar R_T^{-1}H_T^\top
=\sum_{k=0}^{T-1}
\Phi(T,k+1)\mathcal B_kR_k^{-1}\mathcal B_k^\top
\Phi(T,k+1)^\top\succeq0.
\tag{A6.8}
$$

$\mathcal R_T=\operatorname{Im}\mathcal W_c(T)=\operatorname{Im}H_T$라 하면 종단 변위 $v$의 최소 quadratic control energy는

$$
E_T^*(v)=
\min_{H_TU=v}U^\top\bar R_TU
=
\begin{cases}
v^\top\mathcal W_c(T)^\dagger v,&v\in\mathcal R_T,\\
+\infty,&v\notin\mathcal R_T.
\end{cases}
\tag{A6.9}
$$

식 (A6.9)는 고정된 trajectory·actuator $B$·비용 $R$·horizon 아래의 endpoint value function이다. 일반적인 local Riemann metric도 아니고, time-local control distribution과 path energy를 별도로 세우지 않은 상태에서 sub-Riemannian metric이라고 부르지도 않는다.

같은 predeclared terminal displacement $v_m$에 대해

$$
\rho_{{\rm reach},m}
=\log\frac{E^*_{\rm post}(v_m)}{E^*_{\rm pre}(v_m)},
\qquad
\frac{\sqrt{E^*_{\rm post}(v_m)}}
{\sqrt{E^*_{\rm pre}(v_m)}}=e^{\rho_{{\rm reach},m}/2}
\tag{A6.10}
$$

은 두 에너지가 모두 유한하고 양수일 때만 정의한다. $\rho<0$이면 그 **actuator와 비용 아래에서** 목표 변위가 더 쉬워졌다는 뜻이다. reachable subspace가 전후에 다르면 비와 별개로 `SUBSPACE_CHANGED`, 에너지가 0이면 `ZERO_ENERGY_DIRECTION`을 기록한다.

full-rank $\mathcal W_c\succ0$이고 $B,R$을 고정한 회로 매개변수 $\varepsilon$에 대해서는

$$
\dot H_T=
\begin{bmatrix}
\dot\Phi(T,1)\mathcal B_0+\Phi(T,1)\dot{\mathcal B}_0&
\cdots&
\dot{\mathcal B}_{T-1}
\end{bmatrix},
$$

$$
\dot{\mathcal W}_c
=\dot H_T\bar R_T^{-1}H_T^\top
+H_T\bar R_T^{-1}\dot H_T^\top,
$$

$$
\frac{dE_T^*(v)}{d\varepsilon}
=-v^\top\mathcal W_c^{-1}\dot{\mathcal W}_c
\mathcal W_c^{-1}v.
\tag{A6.10a}
$$

따라서 식 (A6.10a)가 음수이면 회로 추가가 그 목표 변위의 최소에너지를 1차적으로 낮춘다. Gramian rank가 바뀌는 지점에서는 이 미분식을 사용하지 않고 식 (A6.9)의 finite comparison과 subspace status를 사용한다.

## 가소성은 별도 augmented-state 모델이다

예를 들어

$$
e_{ij}^{n+1}=\gamma_e e_{ij}^n
+s_i^{n+1}s_j^{\,n-d_{ij}},
$$

$$
p_{ij}^{n+1}=\Pi_{[0,1]}
\left(p_{ij}^n+\eta_p\,m_n e_{ij}^n\right)
\tag{A6.11}
$$

같은 eligibility-gated update는 **[공리: 모델 선택]**이다. 이를 켜면 $z_n=(\xi_n,e_n,p_n)$를 새 상태로 삼고 식 전체를 다시 미분한다. projection 경계와 hard event threshold는 $C^1$이 아니므로 smooth surrogate, hybrid saltation derivative 또는 직접 nonlinear reachability가 필요하다. 식 (A6.11)은 생물학 정리나 answer-blind 학습의 증명이 아니다.

## 실제 뇌 주름과의 타입 경계

피질 표면 embedding $X(\sigma^1,\sigma^2,t)\in\mathbb R^3$가 측정되어야 실제 해부학적 계량과 곡률을

$$
g^{\rm anat}_{ab}=\partial_aX\cdot\partial_bX,
\qquad
b_{ab}=n\cdot\partial_a\partial_bX
\tag{A6.12}
$$

로 정의할 수 있다. A6.5의 좌표는 activation이고 A6.12의 좌표는 cortical material coordinate이므로 두 tensor는 타입이 다르다. longitudinal $X$, thickness, constitutive law, boundary condition, 그리고 anatomy-to-state observation map이 없으면

$$
g_{\rm pass}=g^{\rm anat}
$$

라고 둘 수 없다. 유전·발달기에 형성된 주름은 A6에서는 고정된 anatomical substrate 또는 입력·연결 제약일 뿐, 매 순간 회로가 피질 표면을 물리적으로 끌어당긴다는 증거가 아니다.

## 닫을 주장과 필수 falsifier

### 닫을 수학 주장

1. 식 (A6.5)는 $J_T$ full-rank일 때만 genuine pullback Riemann metric이다.
2. 식 (A6.8)은 symmetric PSD다.
3. 식 (A6.9)는 주어진 finite-horizon linearized system의 minimum quadratic endpoint control energy다.
4. $W,p,d,\theta$가 같아도 $B,R,T$가 다르면 A6-C가 달라지므로 회로만의 고유 기하가 아니다.
5. A6-P와 A6-C는 같은 현상의 두 표기가 아니며 서로를 증명하지 않는다.

### 필수 반례와 중단 조건

- activity-dependent $p$의 도함수를 누락하면 식 (A6.3a)은 틀린 Jacobian이다.
- $B=0$이면 $\mathcal W_c=0$이어서 큰 $W$와 무관하게 제어 도달성이 없다.
- saturated $\phi_i'(h_i)=0$에서는 $J_T$ rank와 입력 controllability가 사라질 수 있다.
- nonnormal $A_n$에서는 eigenvalue 안정성만으로 transient stretch나 reachability를 결정할 수 없다.
- 같은 $W$에 다른 $B$ 또는 $R$을 주면 서로 다른 A6-C가 나온다.
- hard threshold·projection 경계에서 ordinary Jacobian을 쓰지 않는다.
- Eulerian circuit의 존재는 A6-P stretch나 A6-C energy 개선을 함의하지 않는다.

## 성공 판정

math lane은 식 (A6.1)--(A6.12)의 indexing·dimensions·pullback 조건·minimum-energy 정리와 반례를 닫아야 한다. source lane은 pullback metric, LTV Gramian, delay augmentation, neural threshold와 cortical folding의 타입 경계를 권위 자료로 확인한다. audit gate 전에는 구현·데이터 실행·정본 승격을 하지 않는다.
