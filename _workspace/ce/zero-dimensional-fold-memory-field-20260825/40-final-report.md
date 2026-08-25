# 지속 0차원 접힘과 단일 환경장: 조건부 수학 모형

Status: COMPLETE

## 초록

이 연구는 “0차원 공간 접힘들이 환경에 남고 하나의 장을 통해 서로를 실행한다”는 문장을 persistent carrier와 dynamic field로 분리하여 수학화했다. 각 carrier는 공간 절편에서 0차원이지만 지속하는 시공간 support는 worldline이며, carrier measure는 고정 배경이고 유일한 동적 상태는 무차원 환경장 $\chi$다. finite-width retarded Volterra 식은 exponential memory와 finite carrier에서 정확한 network ODE로 환원된다. nonnegative constant-row-sum network에서는 $\mathcal B=A\rho(W)>1$일 때 $\chi_*=\mathcal B-1$인 양의 uniform branch와 국소 안정성을 얻고, 같은 gain 조건은 임의 비음 propagation delay에도 안정성 충분조건을 준다. 27개 집중 검사는 고정점, Jacobian, delay bound, 인과 kernel, UV scaling과 반례를 재현했다. 비선택 history가 carrier로 남는 물리 사상, covariant stress와 암흑물질·암흑에너지 동일성은 도출되지 않았다.

## 1. 정의: 무엇이 0차원이고 무엇이 남는가

공간 절편 $\Sigma_t$에서 한 carrier의 support가 singleton $\{\mathbf X_j\}$이면 공간적으로 0차원이다. 같은 carrier가 시간 동안 지속하면 시공간에서는

$$
\mathcal W_j=\{(t,\mathbf X_j):t\ge t_i\}
$$

인 worldline이다. 따라서 “공간적으로 0차원”과 “시공간적으로 순간 사건”은 다르다. 이 연구의 중심은 매 순간 새 점 사건을 생성하는 모형이 아니라, 공간 절편마다 점으로 남아 있는 carrier가 환경장을 통해 다시 활성화되는 모형이다.

비선택 quantum history 공간을 $\Gamma_{\rm ns}$, subprobability measure를 $\nu_{\rm ns}$라 하고 CE의 별도 map $F_t:\Gamma_{\rm ns}\to\Sigma_t$와 weight $w(\gamma)\ge0$를 채택하면

$$
\mu_{F,t}(B)=\int_{\Gamma_{\rm ns}}w(\gamma)
\mathbf1_B(F_t(\gamma))\nu_{\rm ns}(d\gamma)
$$

라는 weighted pushforward를 정의할 수 있다. 이것은 수학적으로 measure를 만드는 법이지만, 표준 양자역학이 실제 carrier를 생성한다는 정리는 아니다. $F_t$, $w$, instrument dependence와 no-double-counting은 물리 사상 공리에 속한다.

finite atomic, fixed-comoving 특수화는

$$
\mu_F(d^3y)=\sum_{j=1}^n w_j\delta_{\mathbf X_j}(d^3y),
\qquad w_j\ge0
$$

이다. 이 최소모형에서는 $\mu_F$의 위치와 weight가 독립적으로 진화하지 않는다. 따라서 carrier 배경과 primitive reservoir가 존재하더라도 동적 state field는 하나로 유지된다.

## 2. 단일 환경장 공리

환경 변수 $\psi$를 saturation scale $\psi_s$로 나눈

$$
\chi=\frac{\psi}{\psi_s},
\qquad
\sigma(\chi)=\frac{\chi}{1+\chi}
$$

를 유일한 동적 상태로 둔다. finite-width, nonnegative, retarded kernel $K^F_{\ell,R}$와 seed $b$를 채택하면

$$
\boxed{
\chi(t,\mathbf x)=b(t,\mathbf x)+A\int_{t_i}^{t}ds
\int_{\Sigma_s}K^F_{\ell,R}(t,\mathbf x;s,\mathbf y)
\sigma(\chi(s,\mathbf y))\mu_F(d^3y)
}
$$

이다. $A,b,\chi,\sigma$와 carrier count를 무차원으로 두고 $[K^F]=T^{-1}$로 잡으면 적분 전체가 무차원이다. kernel은 미래 support를 가져서는 안 되고, 무한 carrier 집합에는 local summability가 필요하다.

delta는 support 표기일 뿐 strict point coupling의 허가가 아니다. width $\ell>0$의 form factor $S_{\ell,j}$를 사용하여

$$
\mu_{F,\ell}(\mathbf y)=\sum_jw_jS_{\ell,j}(\mathbf y),
\qquad
\int S_{\ell,j}(\mathbf y)d^3y=1
$$

로 coarse-grain해야 한다. Gaussian witness에서 $\int g_\ell^2d^3x\propto\ell^{-3}$, $\int|\nabla g_\ell|^2d^3x\propto\ell^{-5}$이므로 $\ell\to0$ local stress limit는 발산한다.

## 3. Volterra 기억에서 정확한 finite-network 식으로

finite carrier와

$$
h_\tau(u)=\tau^{-1}e^{-u/\tau}\Theta(u)
$$

를 택하면

$$
\chi_i(t)=b_i+A\sum_jW_{ij}\int_0^\infty
h_\tau(u)\sigma(\chi_j(t-u))du.
$$

$q_j=h_\tau*\sigma(\chi_j)$라 두면 $\tau\dot q_j=-q_j+\sigma(\chi_j)$이고 $\boldsymbol\chi-\mathbf b=AW\mathbf q$다. 상수 $\mathbf b$에서 두 식을 결합하면 근사가 아니라 정확히

$$
\boxed{
\tau\dot{\boldsymbol\chi}
=-(\boldsymbol\chi-\mathbf b)+AW\sigma(\boldsymbol\chi)
}
$$

를 얻는다. propagation delay를 보존하면

$$
\tau\dot\chi_i(t)=-(\chi_i(t)-b_i)+A\sum_jW_{ij}
\sigma(\chi_j(t-d_{ij})),
\qquad d_{ij}\ge d(\mathbf X_i,\mathbf X_j)/c.
$$

이웃 carrier는 독립 에너지원이 아니라 $W$와 공통 환경장 $\chi$를 통해 다음 활성화를 바꾸는 gate다.

## 4. 부트스트랩 threshold, 양의 branch와 안정성

$W\ge0$에서 무차원 bootstrap gain을

$$
\mathcal B=A\rho(W)
$$

로 둔다. $\mathbf b=0$, $\sigma'(0)=1$이면 $\mathcal B=1$이 zero branch의 Perron-mode threshold다. 그러나 zero initial history와 zero seed이면 $\boldsymbol\chi\equiv0$이 항상 해이므로 threshold가 spontaneous seed 법칙은 아니다.

$W\mathbf1=w\mathbf1$인 constant-row-sum network에서 $\mathcal B=Aw>1$이면 uniform fixed point는

$$
\chi_*=\mathcal B-1.
$$

전체 Jacobian spectrum은

$$
\operatorname{spec}J_*
=\left\{\frac{-1+A\sigma'(\chi_*)\lambda_k(W)}{\tau}\right\}_k,
$$

이며

$$
A\sigma'(\chi_*)\rho(W)=\frac1{\mathcal B}<1.
$$

$|\lambda_k(W)|\le\rho(W)$이므로 모든 mode의 실수부가 음수여서 양의 branch는 국소 안정이다. delay characteristic equation

$$
(1+\tau z)v_i=A\sigma'(\bar\chi)
\sum_jW_{ij}e^{-zd_{ij}}v_j
$$

에서도 $A\sigma'(\bar\chi)\rho(W)<1$이면 modulus bound가 $\operatorname{Re}z\ge0$ root를 배제한다. 이는 안정성 충분조건이며 임의 network의 positive branch 존재나 전역 안정성 정리는 아니다.

## 5. 수치 witness

8-node ring, row sum $1$, $A=1.2$인 경우

$$
\mathcal B=1.2,
\qquad \chi_*=0.2,
\qquad \max\operatorname{Re}\operatorname{spec}J_*=-0.06666666666666665.
$$

effective delayed gain은 $1/\mathcal B=0.8333333333333334<1$이다. fixed-point residual과 zero-state RHS는 machine precision에서 0이었다. 이 witness는 formula의 일관성을 확인하지만 실제 우주의 parameter 측정은 아니다.

## 6. 보조 event 경로와 no-go

“실행”을 기존 carrier의 활성화가 아니라 새 fold 사건의 생성으로 해석할 때만 counting process $N_\psi=N[\psi,\Pi]$와 Hawkes형 intensity를 사용한다. 이 경로의 reproduction number $\mathcal R=A\beta\tau$는 중심 carrier gain $\mathcal B$와 다른 양이다. 선형 자기흥분은 subcritical에서 소멸하고 supercritical에서 finite stationary mean을 잃으며, 포화 deterministic closure가 안정한 fixed point를 가져도 stochastic almost-sure survival을 자동 보장하지 않는다.

또한 감쇠 없이 additive positive activation을 영원히 누적하면 finite stationary field가 없다. 현재 중심 모형은 carrier $\mu_F$는 지속하지만 activation $\chi$는 relax할 수 있으므로 이 no-go와 충돌하지 않는다. canonical real scalar 하나의 정적 localized lump도 Derrick scaling에서 $E''(1)=-2T<0$이므로 중심 후보가 아니다.

## 7. 에너지와 암흑부문 경계

$\mu_F$는 carrier count/weight이고 $\chi$는 환경 활성 변수다. 어느 것도 현재 정의만으로 energy density가 아니다. auxiliary scalar $\phi$에 source $J$를 붙인다고 가정하면

$$
\nabla_\mu T^{\mu}{}_{\nu}{}^{(\phi)}=-J\nabla_\nu\phi
$$

이므로 field-only stress는 보존되지 않고 source/reservoir를 포함한 총 장부가 필요하다. 따라서 $\chi$의 action, carrier stress, total conservation, abundance와 perturbation을 주지 않은 상태에서 이를 암흑물질·암흑에너지로 동일시할 수 없다.

현재 살아 있는 결론은 좁다. persistent spatial-0D carrier가 고정되어 있고, 하나의 causal saturating environment field가 carrier activation을 매개한다는 유효모형은 수학적으로 일관되며 안정한 positive branch를 가질 수 있다. 비선택 quantum path가 실제 carrier로 남는다는 것과 그 carrier가 중력 source라는 것은 독립적인 미완성 물리 사상이다.

## 8. 관측 비교와 한계

이 run은 우주론 자료를 fit하지 않았다. Amari형 one-field dynamics와 Volterra/Hawkes 문헌은 비국소 자기활성 및 안정성 분석의 수학 도구를 제공하지만 spatial fold 또는 dark-sector 존재론을 확립하지 않는다. strict point interaction, stochastic infinite-volume limit, quantum CP dynamics, Lorentz-covariant microscopic completion과 Einstein--Boltzmann observables는 후속 연구가 필요하다.

## 9. 재현성

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\zero-dimensional-fold-memory-field-20260825\artifacts\verify_fold_memory_field.py'
```

## 참고문헌

- S.-i. Amari, “Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields,” *Biological Cybernetics* 27 (1977), [doi:10.1007/BF00337259](https://doi.org/10.1007/BF00337259), accessed 2026-08-25.
- T. Naito et al., “Characterizations of Linear Volterra Integral Equations with Nonnegative Kernels,” *J. Math. Anal. Appl.* 335 (2007), [doi:10.1016/j.jmaa.2007.01.070](https://doi.org/10.1016/j.jmaa.2007.01.070), accessed 2026-08-25.
- A. G. Hawkes, “Spectra of Some Self-Exciting and Mutually Exciting Point Processes,” *Biometrika* 58 (1971), [doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83), accessed 2026-08-25.
- P. Brémaud and L. Massoulié, “Stability of Nonlinear Hawkes Processes,” *Ann. Probab.* 24 (1996), [doi:10.1214/aop/1065725193](https://doi.org/10.1214/aop/1065725193), accessed 2026-08-25.
- G. H. Derrick, “Comments on Nonlinear Wave Equations as Models for Elementary Particles,” *J. Math. Phys.* 5 (1964), [doi:10.1063/1.1704233](https://doi.org/10.1063/1.1704233), accessed 2026-08-25.
