# 수학 레인

Status: PASS_TO_DEVELOPMENT

A3의 핵심 변화는 $W\odot R$을 폐기하고 geometry를 저차원 operator basis로 쓰는 것이다. $c_{ij}\ge0$이고 대칭이면 $L_c\succeq0$이다. $\Omega^\top=-\Omega$는 순간적인 Euclidean norm을 직접 증가시키는 symmetric quadratic form을 갖지 않지만, $\beta_0I-\beta_dL_c+\beta_\circlearrowright\Omega$의 이산시간 spectral radius는 별도로 확인해야 한다.

좌표, $z,q,c,\Omega,L_c,\beta$는 고정 reference scale 아래 무차원이다. exp/log를 쓰지 않으므로 추가 차원 있는 지수 인자는 없다.

이 식은 improvement theorem이 아니다. 특히 identity가 불완전하면 cross-worm comparison이 정의되지 않는다. common input과 calcium convolution 때문에 forward/reverse lag 차이만으로 causal direction을 증명할 수 없다. 따라서 $\Omega$는 observational circulation이고 schema receipt와 matched controls가 필수다.

## 역치·회로강도·지연의 구현 연결

calibration-only 구간에서 뉴런별 위치와 잡음척도를 $\mu_i,s_i$로 고정하고

$$
z_i(t)=\frac{x_i(t)-\mu_i}{\max(s_i,10^{-6})},\qquad
q_i(t)=\mathbf 1[z_i(t)\ge2.5]
\min\!\left(1,\frac{[z_i(t)-2.5]_+}{2.5}\right)
$$

로 둔다. calibration-only top-3 공통성분을 제거한 잔차가 $r_i(t)$일 때 회로 구성에 실제로 들어가는 값은

$$
u_i(t)=q_i(t)r_i(t)
$$

이다. 즉 역치 식은 장식이 아니라 회로장 계산의 gate다. $C^+_{ij}=\mathbb E_B[u_i(t+1)u_j(t)]$와 $C^-_{ij}=\mathbb E_B[u_i(t-1)u_j(t)]$에서 만든 $c_{ij}$와 $\Omega_{ij}$는 edge마다 달라지는 관측 강도다. $c_{ij}\ge0$는 대칭 conductance, $\Omega_{ij}=-\Omega_{ji}$는 방향 민감 circulation을 나타낸다.

이 자료에는 edge별 물리적 축삭 지연 $d_{ij}$의 독립 receipt가 없다. 따라서 $+1$ sample은 4 Hz에서 0.25초인 관측 lag이며 물리적 지연으로 승격하지 않는다. $17,31,47$ sample도 shift-null이지 delay 추정치가 아니다. 독립 receipt가 있는 미래 자료에서만 $u_j(t-d_{ij})$로 확장할 수 있다.

phase-randomized 대조는 construction 구간의 각 연속 block과 각 뉴런에서 Fourier 진폭을 보존하고 위상만 고정 seed로 바꾼다. identity-shuffle은 좌표-뉴런 결합을 바꾸어 operator를 다시 만든다. 두 대조 모두 endpoint를 보기 전에 동결한다.
