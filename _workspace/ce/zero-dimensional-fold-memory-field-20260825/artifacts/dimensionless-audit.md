# 0D fold memory-field 무차원 감사

Status: COMPLETE

## 코어 식

지속 carrier 활성 중심식:

비선택 history의 weighted pushforward

$$
\mu_{F,t}(B)=\int w(\gamma)\mathbf1_B(F_t(\gamma))
\nu_{\rm ns}(d\gamma)
$$

에서 $w$, indicator, subprobability는 모두 무차원이고 $\mu_F(B)$는 weighted
count다. 이 무차원성은 $\mu_F$를 에너지나 $\Omega$로 바꾸지 않는다.

$$
\chi=\frac{\psi}{\psi_s},\qquad
\chi=b+A\int ds\int K^F_{\ell,R}\,
\sigma(\chi)\mu_F(d^3y),qquad
\mathcal B=A\rho(W).
$$

$\mu_F(d^3y)$는 무차원 count이고 $[K^F_{\ell,R}]=T^{-1}$이므로
$ds\,K^F_{\ell,R}\,\mu_F(d^3y)$는 무차원이다. $A$, $b$, $\chi$,
$\sigma(\chi)$, $W$, $\mathcal B$도 모두 무차원이다. 차원 있는 $\psi$로
쓸 때는 적분항에 $\psi_s$를 곱하고 saturation 인자를 $\sigma(\psi/\psi_s)$로
써야 한다. finite-carrier 식

$$
\tau\dot\chi_i=-(\chi_i-b_i)+A\sum_jW_{ij}\sigma(\chi_j)
$$

의 모든 항은 무차원이고, 양변 전체를 $\tau$로 나눈 미분식의 차원은
$T^{-1}$이다. 지연 $d_{ij}$와 $\tau$는 모두 $T$이고
$z d_{ij}$, $\tau z$는 무차원이다.

새 접힘 생성이라는 별도 event 경로의 코어 식:

식:

$$
\mathcal R=A\beta\int K_{\ell,R}(x,y)dV_x=A\beta\tau
$$

| 코어 인자 | 차원 벡터 $(L,T)$ | 무차원? | 정규화 |
|---|---:|---:|---|
| $\mathcal R$ | $(0,0)$ | yes | $A$는 무차원, $[\beta]=T^{-1}$, $[\tau]=T$ |
| $\psi/\psi_s$ | $(0,0)$ | yes | 기억장 scale $\psi_s$ |
| $x=\psi/\psi_s$ | $(0,0)$ | yes | 기억장 scale $\psi_s$ |
| $u=t/\tau$ | $(0,0)$ | yes | 기억시간 $\tau$ |
| $\eta=A\lambda_0\tau/\psi_s$ | $(0,0)$ | yes | $\psi_s$와 $\tau$ |
| $st$ | $(0,0)$ | yes | $[s]=T^{-1}$ |
| $\mathbf k\cdot\mathbf r$ | $(0,0)$ | yes | $[\mathbf k]=L^{-1}$ |
| $\mathcal R(q-1)$ | $(0,0)$ | yes | $q$와 $\mathcal R$ 모두 무차원 |
| $\log(1+M/M_s)$ | $(0,0)$ | yes | finite-cell total memory scale $M_s$ |
| $(t-\ell/c_\psi)/\tau$ | $(0,0)$ | yes | $[c_\psi]=LT^{-1}$ |
| $r/(c_\psi t)$ | $(0,0)$ | yes | causal-cone ratio |

강도식

$$
\lambda=\lambda_0+
\frac{\beta\psi}{1+\psi/\psi_s}
$$

에서 $\lambda_0$와 $\beta\psi$는 모두 $L^{-3}T^{-1}$이고 분모는
무차원이다. homogeneous closure

$$
\dot\psi=-\frac{\psi}{\tau}
+A\lambda_0+
\frac{A\beta\psi}{1+\psi/\psi_s}
$$

의 모든 항은 $L^{-3}T^{-1}$이다.

Poisson thinning

$$
N_\psi(dV_y)=
\int_0^\infty
\mathbf1_{\{z\le\lambda(y\mid\psi_{y^-})\}}
\Pi(dV_y,dz)
$$

에서는 mark $z$를 intensity와 같은 단위로 잡아 indicator 비교를
동차적으로 만든다. $dV_y\,dz$는 무차원 count intensity다.

명시적 witness

$$
K_{\ell,R}^{\rm w}(t,r)=
\Theta(t-\ell/c_\psi)e^{-(t-\ell/c_\psi)/\tau}
\frac{3\Theta(c_\psi t-r)}{4\pi(c_\psi t)^3}
$$

에서 exponential과 두 step-function의 인자는 모두 무차원이고,
$(c_\psi t)^{-3}$가 $L^{-3}$를 공급한다.

차원 상태: 무차원 게이트 통과. 이 판정은 self-excitation의 물리적 기원,
에너지 보존 또는 양자중력 동일성을 증명하지 않는다.

코드 검증 명령:

    .codex\hooks\python.cmd python _workspace\ce\zero-dimensional-fold-memory-field-20260825\artifacts\verify_fold_memory_field.py

결과:

    status PASS
    reproduction_number_is_dimensionless true
    feedback_matches_intensity_dimension true
